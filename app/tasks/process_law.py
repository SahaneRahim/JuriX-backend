"""
Tâche de traitement des documents juridiques pour JuriX.

Pipeline:
1. Extraire le texte (OCR si nécessaire)
2. Détecter la langue
3. Classifier la catégorie
4. Découper les articles
5. Générer les embeddings
6. Mettre à jour les tsvectors PostgreSQL (remplace la recherche plein texte)
7. Mettre à jour la base de données

Ce module expose deux fonctions:
- process_law_async(): traitement asynchrone via BackgroundTasks FastAPI
- process_law_sync(): version synchrone (conservée pour compatibilité)

Author: JuriX Development Team
Version: 3.0.0 (PostgreSQL natif)
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.exc import DataError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, SyncSessionLocal
from app.models.law import Article, Law
from app.services.search_vectors import REINDEX_ARTICLES_SQL, REINDEX_LAWS_SQL
from app.utils.chunk_refiner import DocumentContext, normalize_for_chunking, refine
from app.utils.text_chunker import ArticleExtractionError, extract_articles

logger = logging.getLogger(__name__)


# ==================== POINT D'ENTREE ASYNCHRONE ====================


async def process_law_async(law_id: int, file_id: str = None) -> Dict[str, Any]:
    """
    Traite un document juridique en arrière-plan (BackgroundTasks FastAPI).

    Appelé via:
        background_tasks.add_task(process_law_async, law_id, file_id)

    Pipeline: load → extract → analyse → articles → embeddings → index PG FTS → update

    Args:
        law_id: ID de la loi en base de données
        file_id: UUID du fichier uploadé (optionnel)

    Returns:
        Dict avec status, language, category, articles_count, etc.
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
    start_time = time.time()
    errors = []

    logger.info(f"🚀 Starting async pipeline for law ID: {law_id} (file_id={file_id})")

    try:
        # Utilise une session synchrone pour le pipeline (Gemini API est sync)
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_sync_pipeline, law_id, file_id
        )
        result["duration"] = round(time.time() - start_time, 2)

        # Après le pipeline sync, mettre à jour les tsvectors via async session
        async with AsyncSessionLocal() as db:
            await _update_fts_vectors_async(db, law_id)
            await db.commit()

        logger.info(f"✅ Async pipeline completed for law {law_id} in {result['duration']}s")
        return result

    except Exception as e:
        logger.error(f"❌ Async pipeline failed for law {law_id}: {e}", exc_info=True)
        errors.append(str(e))

        # Mark law as failed in DB
        async with AsyncSessionLocal() as db:
            await db.execute(
                # 'refused' et non 'draft' : draft signifie "en cours de
                # redaction", pas "le traitement a echoue". L'admin doit pouvoir
                # filtrer les echecs et les relancer.
                text(
                    "UPDATE laws SET status='refused', processing_error=:err, "
                    "processing_progress=0 WHERE id=:id"
                ),
                {"err": str(e), "id": law_id},
            )
            await db.commit()

        return {
            "law_id": law_id,
            "status": "failed",
            "errors": errors,
            "duration": round(time.time() - start_time, 2),
        }


async def _update_fts_vectors_async(db: AsyncSession, law_id: int) -> None:
    """
    Met à jour les tsvectors PostgreSQL pour une loi et ses articles.
    Met a jour les tsvector PostgreSQL.

    Args:
        db: Session async SQLAlchemy
        law_id: ID de la loi à réindexer
    """
    await db.execute(
        text(f"{REINDEX_LAWS_SQL} WHERE id = :law_id"),
        {"law_id": law_id},
    )
    await db.execute(
        text(f"{REINDEX_ARTICLES_SQL} AND a.law_id = :law_id"),
        {"law_id": law_id},
    )
    logger.info(f"✅ FTS tsvectors updated for law {law_id}")


# ==================== SYNC PIPELINE (core logic) ====================


def _run_sync_pipeline(
    law_id: int, file_id: str = None, extraction_cache_seulement: bool = False
) -> Dict[str, Any]:
    """
    Exécute le pipeline de traitement de façon synchrone.

    Args:
        law_id: ID de la loi
        file_id: UUID du fichier uploadé (optionnel)
        extraction_cache_seulement: ne relire que l'extraction deja faite (voir
            _extract_pdf_text), sans jamais convertir

    Returns:
        Dict de résultats
    """
    assert isinstance(law_id, int) and law_id > 0

    errors = []

    # 1. Charger la loi
    law = _load_law(law_id)
    if not law:
        raise ValueError(f"Law {law_id} not found")
    logger.info(f"📄 Loaded law: {law.title}")

    # 2. Extraire le texte du fichier si fourni
    if file_id:
        law, file_errors = _ingest_file_content(
            law_id, law, file_id, extraction_cache_seulement
        )
        errors.extend(file_errors)

    # 3. Valider le contenu
    text = law.content
    if not text or len(text) < 50:
        raise ValueError("Insufficient content for processing")

    # 4. Extraire le titre si pending
    extracted_title = None
    if law.title and law.title.startswith("PENDING-"):
        extracted_title = _extract_title_from_text(text)
        if extracted_title:
            logger.info(f"📝 Extracted title: {extracted_title}")

    # 5. Pipeline d'analyse
    result = _run_analysis_pipeline(law_id, law, text, extracted_title, errors)
    result["errors"] = errors
    return result


def _run_analysis_pipeline(law_id: int, law, text: str, extracted_title=None, file_errors=None):
    """
    Exécute le pipeline complet: langue → catégorie → articles → embeddings → metadata.
    """
    assert text and len(text) >= 50
    assert isinstance(law_id, int) and law_id > 0

    # Langue
    language_result = _detect_language(text)
    logger.info(f"🌍 Language: {language_result['language']} ({language_result['confidence']:.2%})")

    # Catégorie
    category_result = _classify_category(
        extracted_title or getattr(law, "title", "") or "",
        text,
        getattr(law, "type", None),
    )
    logger.info(
        f"📂 Category: {category_result['category']} "
        f"({category_result['confidence']:.2%}, regle {category_result['rule']})"
    )

    # Articles. La langue et le domaine calcules ci-dessus partent avec le
    # decoupage : ils etaient relus en base, ou ils ne sont ecrits qu'a la fin
    # (_update_law_metadata). A la premiere ingestion, embed_text n'avait donc
    # pas de categorie, et la langue etait la valeur par defaut de la ligne.
    articles_count = _split_and_save_articles(
        law_id,
        text,
        language=language_result["language"],
        category=category_result["category"],
    )
    logger.info(f"📑 Articles extracted: {articles_count}")

    # Embeddings
    embeddings_count = _generate_article_embeddings(law_id)
    logger.info(f"🔢 Embeddings generated: {embeddings_count}")

    # Une loi decoupee en articles mais sans le moindre vecteur reste
    # introuvable par la recherche semantique. Le pipeline n'echoue pas pour
    # autant — la recherche plein texte fonctionne — mais l'anomalie est
    # tracee sur la ligne au lieu de disparaitre dans les journaux.
    embeddings_error = category_result.get("error")
    # articles_count compte TOUS les chunks, embeddings_count seulement les
    # vectorisables : un document entierement fait de visas et de formules
    # d'execution n'a legitimement aucun vecteur.
    if articles_count > 0 and embeddings_count == 0:
        embeddings_error = (
            f"Aucun embedding genere pour {articles_count} articles : "
            f"recherche semantique indisponible sur ce document"
        )
        logger.error(f"❌ {embeddings_error}")

    # Les pages que l'extraction n'a pas pu lire restaient dans les journaux :
    # le document passait pour complet. Elles rejoignent processing_error,
    # visible et interrogeable par l'administrateur.
    anomalies = [e for e in [*(file_errors or []), embeddings_error] if e]

    # Metadata
    _update_law_metadata(
        law_id,
        language=language_result["language"],
        language_confidence=language_result["confidence"],
        category=category_result["category"],
        category_id=category_result.get("category_id"),
        category_confidence=category_result["confidence"],
        suggested_categories=category_result.get("suggested"),
        title=extracted_title,
        processing_error=" | ".join(anomalies) or None,
    )

    return {
        "law_id": law_id,
        "status": "completed",
        "language": language_result["language"],
        "language_confidence": language_result["confidence"],
        "category": category_result["category"],
        "category_confidence": category_result["confidence"],
        "articles_count": articles_count,
        "embeddings_generated": embeddings_count,
    }


# ==================== HELPER FUNCTIONS ====================


def _ingest_file_content(law_id: int, law, file_id: str, extraction_cache_seulement: bool = False):
    """
    Extrait le texte depuis le fichier uploadé (PDF ou DOCX).
    Essaie LlamaParse d'abord, puis OCR, puis pypdf comme fallback.
    """
    assert file_id, "file_id must not be empty"
    assert law is not None

    errors = []
    from app.services.file_upload_service import get_upload_service
    upload_service = get_upload_service()

    # Localiser le fichier
    file_path = None
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{file_id}{ext}"
        if p.exists():
            file_path = p
            break

    if not file_path:
        logger.error(f"❌ File {file_id} not found in storage")
        errors.append(f"File {file_id} not found")
        return law, errors

    logger.info(f"📂 Found file: {file_path}")
    extracted_text = ""

    if file_path.suffix == ".pdf":
        extracted_text, pdf_errors = _extract_pdf_text(file_path, extraction_cache_seulement)
        errors.extend(pdf_errors)
    elif file_path.suffix == ".docx":
        extracted_text, docx_errors = _extract_docx_text(file_path)
        errors.extend(docx_errors)

    if not extracted_text or len(extracted_text) <= 50:
        # Auparavant : un simple warning, puis le pipeline continuait et le
        # document finissait publie avec son contenu de remplacement. Les
        # erreurs d'extraction partent avec le message : « 0 caracteres » seul
        # laissait croire que l'OCR n'avait rien trouve, quand le processus
        # avait ete tue.
        details = f" — {' | '.join(errors)}" if errors else ""
        raise ValueError(
            f"Extraction insuffisante pour {file_path.name} : "
            f"{len(extracted_text or '')} caracteres (minimum 50). "
            f"Document non publie.{details}"
        )

    _update_law_content(law_id, extracted_text)
    law.content = extracted_text
    logger.info(f"✅ {len(extracted_text)} caracteres extraits")
    return law, errors


def _extract_pdf_text(file_path, cache_seulement: bool = False) -> tuple:
    """
    Extrait le texte d'un PDF par le moteur configure (PDF_EXTRACTION_ENGINE).

    Docling par defaut, en local : OCR pleine page sur toutes les pages. La
    couche texte des PDF prc.cm n'est jamais lue telle quelle : c'est l'OCR du
    scanner, mesure a ~20 % de rappel (filigrane injecte au milieu des phrases,
    cachets lus comme du charabia, 21 % des documents sans aucun texte
    exploitable). Aucun repli degrade non plus : un document publie avec ce
    contenu passait pour complet, sans aucun signal.

    Args:
        cache_seulement: ne rien convertir, relire l'extraction deja faite. La
            passe d'indexation (scripts/ingest_corpus.py) l'impose : charger
            Docling a cote d'EmbeddingGemma doublerait la memoire, et un
            document que la passe d'extraction n'a pas fini doit attendre.

    Returns:
        (texte, erreurs) — `erreurs` nomme les pages que le moteur n'a pas pu
        lire. Le document reste publie sans elles : perdre une page vaut mieux
        que perdre les autres. La liste part dans `laws.processing_error`.

    Raises:
        PdfExtractionError: extraction impossible
    """
    from app.services.pdf_extraction_service import (
        PdfExtractionError,
        _extracteur_docling,
        get_pdf_extractor,
    )

    # Le cache relu est celui de la passe d'extraction, donc celui de Docling,
    # QUEL QUE SOIT PDF_EXTRACTION_ENGINE : un .env partage avec l'API et regle
    # sur gemini faisait sinon refuser tout le corpus a l'indexation.
    service = _extracteur_docling() if cache_seulement else get_pdf_extractor()
    if not service.is_available():
        raise PdfExtractionError(service.raison_indisponible())

    logger.info(f"📄 Extraction {service.nom} : {file_path.name}")
    resultat = service.extraire(file_path, cache_seulement=cache_seulement)
    logger.info(f"✅ {service.nom} : {len(resultat.texte)} caracteres, {resultat.nb_pages} page(s)")

    erreurs = []
    if resultat.pages_en_echec:
        cause = f" ({resultat.erreurs[-1]})" if resultat.erreurs else ""
        erreurs.append(
            "Pages non extraites : "
            + ", ".join(str(n) for n in resultat.pages_en_echec) + cause
        )
    if resultat.pages_illisibles:
        erreurs.append(
            "Pages sans texte lisible (rendu rate ou cachet seul) : "
            + ", ".join(str(n) for n in resultat.pages_illisibles)
        )
    for erreur in erreurs:
        logger.error(f"❌ {file_path.name} : {erreur}")
    return resultat.texte, erreurs


def _extract_docx_text(file_path) -> tuple:
    """Extrait le texte d'un fichier DOCX."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return _clean_extracted_text(text), []
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return "", [f"DOCX extraction failed: {str(e)}"]


def _load_law(law_id: int) -> Law:
    """Charge une loi depuis la base de données (synchrone)."""
    with SyncSessionLocal() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            _ = law.articles  # Eager load
        return law


@lru_cache(maxsize=1)
def _language_detector():
    """
    Un seul detecteur par processus.

    Il etait recree a chaque document, et rechargeait donc les 131 Mo de
    fastText (lid.176.bin) a chaque fois : 0,2 a 0,4 s par document, soit 7 a
    15 minutes sur le corpus, pour un modele qui ne change jamais.
    """
    from app.services.language_detector import LanguageDetector

    return LanguageDetector()


def _detect_language(text: str) -> Dict[str, Any]:
    """Détecte la langue du texte."""
    try:
        result = _language_detector().detect(text)
        return {"language": result["language"], "confidence": result["confidence"]}
    except Exception as e:
        logger.warning(f"⚠️ Language detection failed, using fallback: {e}")
        if any(word in text.lower() for word in ["le", "la", "les", "de", "et", "article"]):
            return {"language": "fr", "confidence": 0.75}
        return {"language": "en", "confidence": 0.75}


def _classify_category(title: str, text: str, doc_type: str = None) -> Dict[str, Any]:
    """
    Determine le domaine juridique du document et resout son identifiant.

    Le TITRE est passe en premier parce qu'il est le signal decisif : mesure sur
    les 2238 titres du corpus prc.cm, il tranche seul pour quatre documents sur
    cinq. L'ancienne version ne recevait que `content`, et une Loi de finances
    bourree de « president » et de « Vu la Constitution » finissait en Droit
    Constitutionnel.

    L'identifiant est resolu PAR LE NOM contre la table `categories`. Il n'est
    plus, comme avant, une position dans un dictionnaire Python ecrite telle
    quelle dans une cle etrangere.
    """
    from app.services.category_resolver import load_domain_map
    from app.services.legal_domain_classifier import get_legal_domain_classifier

    result = get_legal_domain_classifier().classify(title or "", text or "", doc_type)

    with SyncSessionLocal() as session:
        domain_map = load_domain_map(session)

    category_id = domain_map.get(result.domain.lower())
    # suggested_categories est declaree ARRAY(Integer) : ce sont des
    # identifiants, pas des noms. Les domaines absents de la table sont omis
    # plutot que remplaces par un identifiant invente.
    suggested = [
        domain_map[name.lower()]
        for name in [result.domain, *(d for d, _ in result.runners_up)]
        if name.lower() in domain_map
    ]

    error = None
    if category_id is None:
        # Pas de creation automatique de ligne : l'anomalie est tracee sur la
        # loi plutot que masquee par une categorie inventee.
        error = f"Domaine '{result.domain}' absent de la table categories"
        logger.error("❌ %s", error)

    return {
        "category": result.domain,
        "category_id": category_id,
        "confidence": result.confidence,
        "rule": result.rule,
        "suggested": suggested,
        "error": error,
    }


def _split_and_save_articles(
    law_id: int,
    text: str,
    language: Optional[str] = None,
    category: Optional[str] = None,
) -> int:
    """
    Extrait les articles du texte de loi et les sauvegarde en base de données.

    Args:
        language: langue detectee ; sinon celle de la ligne. Elle decide du
            role de « Section N » (article en anglais, subdivision en francais).
        category: domaine juridique calcule ; sinon celui de la ligne.

    Returns:
        Nombre d'articles extraits et sauvegardés
    """
    logger.info(f"📑 Extracting articles from law {law_id}")

    try:
        # normalize_for_chunking AVANT le decoupage : LlamaParse rend du
        # markdown, et "**ARTICLE 1ER**:" n'est pas reconnu par les motifs
        # d'article, qui attendent "Article" en debut de ligne. Sans cette
        # passe, des documents entiers ressortaient sans un seul article.
        with SyncSessionLocal() as session:
            law = session.query(Law).filter(Law.id == law_id).first()
            langue = language or (law.language if law else None) or "fr"

            normalized = normalize_for_chunking(text)
            extracted = extract_articles(
                normalized, strict=False, min_article_length=1, language=langue
            )
            if not extracted:
                logger.warning(f"⚠️ No articles extracted from law {law_id}")
                return 0

            logger.info(f"📋 Found {len(extracted)} articles")

            context = DocumentContext(
                reference=(law.reference if law else "") or "",
                title=(law.title if law else "") or "",
                doc_type=(law.type if law else None),
                date=law.publication_date.isoformat() if law and law.publication_date else None,
                category=category or (law.category.name if law and law.category else None),
                language=langue,
            )

            # Raffinage : classe chaque chunk, decide ce qui merite un vecteur,
            # et prepare embed_text (contenu prefixe de l'en-tete du document).
            # Rien n'est supprime — un visa ou une formule d'execution reste
            # consultable et cherchable en plein texte — mais il ne consomme
            # plus d'appel d'embedding et ne pollue plus les resultats
            # semantiques.
            refined = refine(extracted, context)
            logger.info(
                f"🧹 Raffinage : {refined.stats.get('chunks_out')} chunks, "
                f"{refined.stats.get('embeddable')} a vectoriser, "
                f"natures={refined.stats.get('kinds')}"
            )

            session.query(Article).filter(Article.law_id == law_id).delete()

            for article_data in refined.chunks:
                article = Article(
                    law_id=law_id,
                    number=str(article_data.get("number", "")),
                    title=article_data.get("title"),
                    section=article_data.get("section"),
                    content=article_data.get("content", ""),
                    order=article_data.get("position", 0),
                    page_number=article_data.get("page_number"),
                    kind=article_data.get("kind"),
                    embed=bool(article_data.get("embed", True)),
                    embed_text=article_data.get("embed_text"),
                )
                session.add(article)
                logger.debug(f"📄 Created Article {article.number} ({article.kind})")

            session.commit()
            logger.info(f"✅ Saved {len(refined.chunks)} chunks for law {law_id}")

        return len(refined.chunks)

    except ArticleExtractionError as e:
        logger.warning(f"⚠️ Article extraction failed: {e}")
        return 0
    except DataError as e:
        # Depassement de largeur de colonne. On RELEVE au lieu de rendre 0 :
        # les articles existants ont deja ete supprimes plus haut, donc rendre 0
        # laisserait la loi publiee avec zero article et sans autre trace qu'une
        # ligne de log. Le handler de process_law_async la passe en 'refused' et
        # ecrit la cause dans laws.processing_error, ou elle est interrogeable.
        logger.error(f"❌ Article rejete par la base (largeur de colonne) : {e}")
        raise
    except Exception as e:
        # On RELEVE au lieu de rendre 0 : les articles existants ont deja ete
        # supprimes plus haut, et rendre 0 publiait la loi sans un seul article,
        # sans autre trace qu'une ligne de journal. Le pipeline la passe en
        # 'refused' avec la cause dans processing_error, et une relance la
        # reprendra.
        logger.error(f"❌ Error saving articles: {e}", exc_info=True)
        raise


def _generate_article_embeddings(law_id: int) -> int:
    """
    Génère les embeddings pour tous les articles d'une loi via Gemini API.
    Cache les embeddings dans la table embedding_cache PostgreSQL.

    Returns:
        Nombre d'articles avec embeddings générés
    """
    assert isinstance(law_id, int) and law_id > 0

    from app.services.embedding_service import (
        EmbeddingService,
        EmbeddingServiceError,
        QuotaExhaustedError,
    )

    logger.info(f"🔢 Generating embeddings for law {law_id} chunks...")

    try:
        # use_cache=False : le cache d'embeddings sert les QUESTIONS, qui se
        # repetent. Un article ne se re-encode jamais — embed_text porte la
        # reference du document, unique — et son vecteur est deja garde dans
        # articles.embedding. Mesure : le cache aurait ajoute 0,2 a 0,4 Go a
        # la base, plus que les vecteurs eux-memes, sans un seul succes.
        embedding_service = EmbeddingService(use_cache=False)
        max_len = EmbeddingService.MAX_TEXT_LENGTH

        with SyncSessionLocal() as session:
            # Seuls les chunks marques embed sont vectorises. Les visas, les
            # formules d'execution et les fragments restent en base et
            # cherchables en plein texte, mais ils ne consomment plus d'appel
            # d'embedding et ne polluent plus les resultats semantiques.
            articles = (
                session.query(Article)
                .filter_by(law_id=law_id)
                .filter(Article.embed.is_(True))
                .all()
            )
            if not articles:
                logger.warning(f"⚠️ No embeddable chunk for law {law_id}")
                return 0

            logger.info(f"📄 Found {len(articles)} chunks to process")

            # Troncature AVANT l'appel. generate_batch_embeddings valide tous
            # les textes d'abord et leve des qu'UN seul depasse la limite :
            # un article trop long annulait donc les embeddings de la loi
            # ENTIERE, l'exception etait avalee plus bas et la loi publiee
            # sans un seul vecteur. Mieux vaut un article tronque que zero
            # article indexe.
            texts = []
            for article in articles:
                # embed_text si le raffinage en a produit un : c'est le contenu
                # prefixe de l'en-tete du document (reference, titre, date,
                # categorie, section, page). Sans ce contexte, "Article 3.- La
                # depense sera imputee sur le budget de l'Etat" est
                # indistinguable des milliers d'articles identiques du corpus.
                content = article.embed_text or article.content or ""
                if len(content) > max_len:
                    logger.warning(
                        f"⚠️ Article {article.number} tronqué pour l'embedding "
                        f"({len(content)} > {max_len} caractères)"
                    )
                    content = content[:max_len]
                texts.append(content)

            provider = embedding_service.provider
            logger.info(f"🚀 Generating {len(texts)} embeddings via {provider.label}...")
            embeddings = embedding_service.generate_batch_embeddings(texts=texts, normalize=True)

            success_count = 0
            for article, embedding in zip(articles, embeddings):
                try:
                    article.embedding = embedding.tolist()
                    # La provenance part AVEC le vecteur, dans la meme
                    # ecriture : un vecteur dont on ignore le modele ne peut
                    # pas etre compare sans risque.
                    article.embedding_model = provider.fingerprint
                    success_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to save embedding for article {article.number}: {e}")

            session.commit()
            logger.info(f"✅ Generated {success_count}/{len(articles)} embeddings for law {law_id}")
            return success_count

    except QuotaExhaustedError as e:
        # Distinguee d'une panne : le document est correctement decoupe et
        # indexe en plein texte, seuls les vecteurs manquent. Ils se rattrapent
        # avec scripts/regenerate_embeddings.py une fois le quota reinitialise.
        logger.error(
            f"❌ Quota journalier epuise : {e}. "
            f"Relancer scripts/regenerate_embeddings.py --all apres reinitialisation."
        )
        return 0
    except EmbeddingServiceError as e:
        logger.error(f"❌ Embedding service error: {e}")
        return 0
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}", exc_info=True)
        return 0


def _update_law_metadata(
    law_id: int,
    language: str,
    language_confidence: float,
    category: str,
    category_confidence: float,
    category_id: int = None,
    suggested_categories=None,
    title: str = None,
    processing_error: str = None,
) -> None:
    """Met à jour les métadonnées de la loi en base de données (synchrone)."""
    with SyncSessionLocal() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            law.language = language
            law.detected_language = language
            law.language_confidence = language_confidence
            law.category_confidence = category_confidence
            # La proposition de la machine est TOUJOURS enregistree, meme
            # quand elle ne s'applique pas : la colonne etait declaree partout
            # et ecrite nulle part depuis le debut du projet.
            if suggested_categories:
                law.suggested_categories = list(suggested_categories)
            # Le pipeline ne remplit qu'une categorie NULLE. L'administrateur
            # choisit une categorie a l'upload (routes/laws.py) et le pipeline
            # l'ecrasait quelques secondes plus tard.
            if category_id is not None and law.category_id is None:
                law.category_id = category_id
            law.status = "published"
            law.processing_error = processing_error
            if title:
                law.title = title
                logger.info(f"📝 Updated title to: {title}")
            session.commit()
            logger.info(f"✅ Updated metadata for law {law_id}")


def _extract_title_from_text(text: str) -> str:
    """Extrait le titre du document depuis les premières lignes du texte."""
    if not text:
        return None

    import re
    header = text[:500].strip()
    lines = [line.strip() for line in header.split("\n") if line.strip()]
    if not lines:
        return None

    title_patterns = [
        r"^(LAW\s+N[OoØ°]\.?\s*\d+[/-]\d+.*?)(?:\n|$)",
        r"^(LOI\s+N[OoØ°]\.?\s*\d+[/-]\d+.*?)(?:\n|$)",
        r"^(THE\s+CONSTITUTION.*?)(?:\n|$)",
        r"^(LA\s+CONSTITUTION.*?)(?:\n|$)",
        r"^(CONSTITUTION.*?)(?:\n|$)",
        r"^(DECREE\s+N[OoØ°]\.?\s*\d+.*?)(?:\n|$)",
        r"^(DÉCRET\s+N[OoØ°]\.?\s*\d+.*?)(?:\n|$)",
    ]

    for line in lines[:3]:
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = re.sub(r"\s+", " ", match.group(1).strip())
                return title[:200]

    first_line = lines[0]
    if len(first_line) > 10 and len(first_line) < 200:
        if first_line.isupper() or first_line[0].isupper():
            return first_line[:200]

    return None


def _update_law_content(law_id: int, content: str) -> None:
    """Met à jour le contenu de la loi en base de données (synchrone)."""
    with SyncSessionLocal() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            law.content = content
            session.commit()
            logger.info(f"✅ Updated content for law {law_id}")


def _clean_extracted_text(text: str) -> str:
    """Nettoie le texte extrait avec modifications minimales."""
    if not text:
        return ""

    import re

    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n([a-zàâçéèêëïîôùûüœæ])", r"\1", text)
    text = re.sub(r"([ldnsjcmtLDNSJCMT])'\n([A-Za-zÀ-ÿ])", r"\1'\2", text)
    text = re.sub(r"(qu|Qu)'\n([A-Za-zÀ-ÿ])", r"\1'\2", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<<PAGE:\s*\d+\s*>>\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]{5,}", "    ", text, flags=re.MULTILINE)

    return text.strip()


def delete_from_search_index(law_id: int) -> bool:
    """
    Désindexe une loi des tsvectors PostgreSQL lors d'une suppression.
    Retire la loi de l'index plein texte.

    Args:
        law_id: ID of the law to deindex

    Returns:
        True if successful, False otherwise
    """
    try:
        with SyncSessionLocal() as session:
            session.execute(
                text("UPDATE laws SET search_vector = NULL WHERE id = :law_id"),
                {"law_id": law_id},
            )
            session.commit()
        logger.info(f"🗑️ Deindexed law {law_id} from PG FTS")
        return True
    except Exception as e:
        logger.warning(f"⚠️ PG FTS deindex failed for law {law_id}: {e}")
        return False


# Legacy alias for any code that references delete_from_meilisearch
delete_from_meilisearch = delete_from_search_index


# ==================== LEGACY SYNC ENTRY POINT ====================

def process_law_sync(
    law_id: int, file_id: str = None, extraction_cache_seulement: bool = False
) -> Dict[str, Any]:
    """
    Version synchrone du pipeline, celle de l'ingestion par script.
    Depuis l'API, préférer process_law_async() via BackgroundTasks.

    Args:
        law_id: ID de la loi
        file_id: UUID du fichier uploadé
        extraction_cache_seulement: voir _extract_pdf_text

    Returns:
        Dict avec status et résultats
    """
    assert isinstance(law_id, int) and law_id > 0

    start_time = time.time()
    logger.info(f"🔄 Starting synchronous processing for law {law_id}, file {file_id}")

    try:
        result = _run_sync_pipeline(law_id, file_id, extraction_cache_seulement)
    except Exception as e:
        # Sans ce marquage, une loi en echec restait 'processing' : ni publiee,
        # ni signalee, et une relance de l'ingestion la sautait comme deja
        # traitee. 'refused' + processing_error la rend visible et rejouable.
        logger.error(f"❌ Synchronous pipeline failed for law {law_id}: {e}")
        with SyncSessionLocal() as session:
            session.execute(
                text(
                    "UPDATE laws SET status='refused', processing_error=:err, "
                    "processing_progress=0 WHERE id=:id"
                ),
                {"err": str(e)[:2000] or type(e).__name__, "id": law_id},
            )
            session.commit()
        raise

    # Mettre à jour les tsvectors synchroniquement
    try:
        with SyncSessionLocal() as session:
            session.execute(
                text(f"{REINDEX_LAWS_SQL} WHERE id = :law_id"),
                {"law_id": law_id},
            )
            session.execute(
                text(f"{REINDEX_ARTICLES_SQL} AND a.law_id = :law_id"),
                {"law_id": law_id},
            )
            session.commit()
        logger.info(f"✅ FTS tsvectors updated for law {law_id}")
    except Exception as e:
        logger.warning(f"⚠️ FTS update failed (non-fatal): {e}")

    result["duration"] = round(time.time() - start_time, 2)
    logger.info(f"✅ Synchronous processing completed for law {law_id} in {result['duration']}s")
    return result
