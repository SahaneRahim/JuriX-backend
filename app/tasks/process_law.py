"""
Tâche de traitement des documents juridiques pour JuriX.

Pipeline:
1. Extraire le texte (OCR si nécessaire)
2. Détecter la langue
3. Classifier la catégorie
4. Découper les articles
5. Générer les embeddings
6. Mettre à jour les tsvectors PostgreSQL (remplace Meilisearch)
7. Mettre à jour la base de données

Ce module expose deux fonctions:
- process_law_async(): version async pour FastAPI BackgroundTasks (remplace Celery)
- process_law_sync(): version synchrone (conservée pour compatibilité)

Author: JuriX Development Team
Version: 3.0.0 (no Celery, no Meilisearch)
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.law import Article, Law
from app.utils.text_chunker import extract_articles, ArticleExtractionError

logger = logging.getLogger(__name__)


# ==================== ASYNC ENTRY POINT (remplace Celery) ====================


async def process_law_async(law_id: int, file_id: str = None) -> Dict[str, Any]:
    """
    Traite un document juridique en arrière-plan (BackgroundTasks FastAPI).

    Remplace la tâche Celery. Appelé via:
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
                text("UPDATE laws SET status='draft', processing_error=:err WHERE id=:id"),
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
    Remplace l'indexation Meilisearch.

    Args:
        db: Session async SQLAlchemy
        law_id: ID de la loi à réindexer
    """
    await db.execute(
        text("""
            UPDATE laws
            SET search_vector =
                to_tsvector('french', coalesce(title, '') || ' ' || coalesce(content, ''))
                || to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
            WHERE id = :law_id
        """),
        {"law_id": law_id},
    )
    await db.execute(
        text("""
            UPDATE articles
            SET search_vector =
                to_tsvector('french', coalesce(content, ''))
                || to_tsvector('english', coalesce(content, ''))
                || to_tsvector('simple', coalesce(number, ''))
            WHERE law_id = :law_id
        """),
        {"law_id": law_id},
    )
    logger.info(f"✅ FTS tsvectors updated for law {law_id}")


# ==================== SYNC PIPELINE (core logic) ====================


def _run_sync_pipeline(law_id: int, file_id: str = None) -> Dict[str, Any]:
    """
    Exécute le pipeline de traitement de façon synchrone.

    Args:
        law_id: ID de la loi
        file_id: UUID du fichier uploadé (optionnel)

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
        law, file_errors = _ingest_file_content(law_id, law, file_id)
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
    result = _run_analysis_pipeline(law_id, law, text, extracted_title)
    result["errors"] = errors
    return result


def _run_analysis_pipeline(law_id: int, law, text: str, extracted_title=None):
    """
    Exécute le pipeline complet: langue → catégorie → articles → embeddings → metadata.
    """
    assert text and len(text) >= 50
    assert isinstance(law_id, int) and law_id > 0

    # Langue
    language_result = _detect_language(text)
    logger.info(f"🌍 Language: {language_result['language']} ({language_result['confidence']:.2%})")

    # Catégorie
    category_result = _classify_category(text)
    logger.info(f"📂 Category: {category_result['category']} ({category_result['confidence']:.2%})")

    # Articles
    articles_count = _split_and_save_articles(law_id, text)
    logger.info(f"📑 Articles extracted: {articles_count}")

    # Embeddings
    embeddings_count = _generate_article_embeddings(law_id)
    logger.info(f"🔢 Embeddings generated: {embeddings_count}")

    # Metadata
    _update_law_metadata(
        law_id,
        language=language_result["language"],
        language_confidence=language_result["confidence"],
        category=category_result["category"],
        category_confidence=category_result["confidence"],
        title=extracted_title,
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


def _ingest_file_content(law_id: int, law, file_id: str):
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
        extracted_text, pdf_errors = _extract_pdf_text(file_path)
        errors.extend(pdf_errors)
    elif file_path.suffix == ".docx":
        extracted_text, docx_errors = _extract_docx_text(file_path)
        errors.extend(docx_errors)

    if extracted_text and len(extracted_text) > 50:
        _update_law_content(law_id, extracted_text)
        law.content = extracted_text
        logger.info(f"✅ Extracted {len(extracted_text)} chars from file")
    else:
        logger.warning("⚠️ No text extracted or text too short")

    return law, errors


def _extract_pdf_text(file_path) -> tuple:
    """Extrait le texte d'un PDF via LlamaParse → OCR → pypdf."""
    import asyncio
    from app.core.config import settings

    try:
        from app.services.llama_parse_service import LlamaParseService, LlamaParseError
        logger.info(f"📄 Processing PDF with LlamaParse: {file_path.name}")
        llama_service = LlamaParseService()
        if llama_service.is_available():
            text = asyncio.run(llama_service.extract_text(file_path))
            logger.info(f"✅ LlamaParse Complete: {len(text)} chars")
            return text, []
        raise LlamaParseError("LlamaParse not configured")
    except Exception as e:
        logger.warning(f"LlamaParse failed ({e}), trying OCR fallback...")
        return _extract_pdf_ocr_fallback(file_path, settings)


def _extract_pdf_ocr_fallback(file_path, settings) -> tuple:
    """OCR puis pypdf fallback pour extraction PDF."""
    import asyncio
    try:
        from app.services.ocr_service import OCRService
        ocr_service = OCRService(tesseract_path=settings.TESSERACT_PATH)
        ocr_result = asyncio.run(ocr_service.process_pdf(file_path))
        logger.info(f"✅ OCR Fallback: {len(ocr_result.text)} chars")
        return ocr_result.text, []
    except Exception as ocr_e:
        logger.error(f"OCR failed ({ocr_e}), reverting to pypdf fallback...")
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            pages_text.append(f"<<PAGE:{page_num}>>\n{page.extract_text()}")
        text = _clean_extracted_text("\n".join(pages_text))
        logger.info(f"✅ pypdf Fallback: {len(text)} chars from {len(reader.pages)} pages")
        return text, []


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
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings

    sync_engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
    Session = sessionmaker(bind=sync_engine)

    with Session() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            _ = law.articles  # Eager load
        return law


def _detect_language(text: str) -> Dict[str, Any]:
    """Détecte la langue du texte."""
    try:
        from app.services.language_detector import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect(text)
        return {"language": result["language"], "confidence": result["confidence"]}
    except Exception as e:
        logger.warning(f"⚠️ Language detection failed, using fallback: {e}")
        if any(word in text.lower() for word in ["le", "la", "les", "de", "et", "article"]):
            return {"language": "fr", "confidence": 0.75}
        return {"language": "en", "confidence": 0.75}


def _classify_category(text: str) -> Dict[str, Any]:
    """Classifie la catégorie du document."""
    try:
        from app.services.document_classifier import DocumentClassifier
        classifier = DocumentClassifier()
        results = classifier.classify(text)
        if results:
            cat_id, confidence, _ = results[0]
            category_name = classifier.get_category_name(cat_id)
            return {"category": category_name, "confidence": confidence}
        return {"category": "Autre", "confidence": 0.0}
    except Exception as e:
        logger.warning(f"⚠️ Category classification failed, using fallback: {e}")
        if "fiscal" in text.lower() or "impôt" in text.lower():
            return {"category": "Droit Fiscal", "confidence": 0.70}
        elif "pénal" in text.lower() or "crime" in text.lower():
            return {"category": "Droit Pénal", "confidence": 0.70}
        return {"category": "Droit Civil", "confidence": 0.70}


def _split_and_save_articles(law_id: int, text: str) -> int:
    """
    Extrait les articles du texte de loi et les sauvegarde en base de données.

    Returns:
        Nombre d'articles extraits et sauvegardés
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings

    logger.info(f"📑 Extracting articles from law {law_id}")

    try:
        extracted = extract_articles(text, strict=False, min_article_length=1)
        if not extracted:
            logger.warning(f"⚠️ No articles extracted from law {law_id}")
            return 0

        logger.info(f"📋 Found {len(extracted)} articles")

        sync_engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
        Session = sessionmaker(bind=sync_engine)

        with Session() as session:
            session.query(Article).filter(Article.law_id == law_id).delete()

            for article_data in extracted:
                article = Article(
                    law_id=law_id,
                    number=str(article_data.get("number", "")),
                    title=article_data.get("title"),
                    section=article_data.get("section"),
                    content=article_data.get("content", ""),
                    order=article_data.get("position", 0),
                    page_number=article_data.get("page_number"),
                )
                session.add(article)
                logger.debug(f"📄 Created Article {article.number}")

            session.commit()
            logger.info(f"✅ Saved {len(extracted)} articles for law {law_id}")

        return len(extracted)

    except ArticleExtractionError as e:
        logger.warning(f"⚠️ Article extraction failed: {e}")
        return 0
    except Exception as e:
        logger.error(f"❌ Error saving articles: {e}", exc_info=True)
        return 0


def _generate_article_embeddings(law_id: int) -> int:
    """
    Génère les embeddings pour tous les articles d'une loi via Gemini API.
    Cache les embeddings dans la table embedding_cache PostgreSQL.

    Returns:
        Nombre d'articles avec embeddings générés
    """
    assert isinstance(law_id, int) and law_id > 0

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.services.embedding_service import EmbeddingService, EmbeddingServiceError

    logger.info(f"🔢 Generating embeddings for law {law_id} chunks...")

    try:
        embedding_service = EmbeddingService(use_cache=True)

        engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""), pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            articles = session.query(Article).filter_by(law_id=law_id).all()
            if not articles:
                logger.warning(f"⚠️ No articles found for law {law_id}")
                return 0

            logger.info(f"📄 Found {len(articles)} chunks to process")

            texts = [article.content for article in articles]
            article_ids = [article.id for article in articles]

            logger.info(f"🚀 Generating {len(texts)} embeddings via Gemini API...")
            embeddings = embedding_service.generate_batch_embeddings(texts=texts, normalize=True)

            success_count = 0
            for article, embedding in zip(articles, embeddings):
                try:
                    article.embedding = embedding.tolist()
                    success_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to save embedding for article {article.number}: {e}")

            session.commit()
            logger.info(f"✅ Generated {success_count}/{len(articles)} embeddings for law {law_id}")
            return success_count

        finally:
            session.close()

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
    title: str = None,
) -> None:
    """Met à jour les métadonnées de la loi en base de données (synchrone)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings

    sync_engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
    Session = sessionmaker(bind=sync_engine)

    with Session() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            law.language = language
            law.detected_language = language
            law.language_confidence = language_confidence
            law.status = "published"
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
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings

    sync_engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
    Session = sessionmaker(bind=sync_engine)

    with Session() as session:
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
    Remplace delete_from_meilisearch().

    Args:
        law_id: ID of the law to deindex

    Returns:
        True if successful, False otherwise
    """
    try:
        from sqlalchemy import create_engine
        from app.core.config import settings

        engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE laws SET search_vector = NULL WHERE id = :law_id"),
                {"law_id": law_id},
            )
            conn.commit()
        logger.info(f"🗑️ Deindexed law {law_id} from PG FTS")
        return True
    except Exception as e:
        logger.warning(f"⚠️ PG FTS deindex failed for law {law_id}: {e}")
        return False


# Legacy alias for any code that references delete_from_meilisearch
delete_from_meilisearch = delete_from_search_index


# ==================== LEGACY SYNC ENTRY POINT ====================

def process_law_sync(law_id: int, file_id: str = None) -> Dict[str, Any]:
    """
    Version synchrone du pipeline (conservée pour compatibilité).
    Préférer process_law_async() via BackgroundTasks.

    Args:
        law_id: ID de la loi
        file_id: UUID du fichier uploadé

    Returns:
        Dict avec status et résultats
    """
    assert isinstance(law_id, int) and law_id > 0

    start_time = time.time()
    logger.info(f"🔄 Starting synchronous processing for law {law_id}, file {file_id}")

    result = _run_sync_pipeline(law_id, file_id)

    # Mettre à jour les tsvectors synchroniquement
    try:
        from sqlalchemy import create_engine
        from app.core.config import settings

        engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE laws SET search_vector =
                        to_tsvector('french', coalesce(title,'') || ' ' || coalesce(content,''))
                        || to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
                    WHERE id = :law_id
                """),
                {"law_id": law_id},
            )
            conn.execute(
                text("""
                    UPDATE articles SET search_vector =
                        to_tsvector('french', coalesce(content,''))
                        || to_tsvector('english', coalesce(content,''))
                        || to_tsvector('simple', coalesce(number,''))
                    WHERE law_id = :law_id
                """),
                {"law_id": law_id},
            )
            conn.commit()
        logger.info(f"✅ FTS tsvectors updated for law {law_id}")
    except Exception as e:
        logger.warning(f"⚠️ FTS update failed (non-fatal): {e}")

    result["duration"] = round(time.time() - start_time, 2)
    logger.info(f"✅ Synchronous processing completed for law {law_id} in {result['duration']}s")
    return result
