"""
Explication d'un article isole, rendue sur la page du document.

Le bouton « Expliquer l'article » de la page de lecture appelle ce service. Il
ne passe PAS par le RAG : il n'y a rien a rechercher, l'article est deja connu.
Le pipeline de `RAGService.ask` — recuperation, reclassement, historique,
citations, persistance des `Message` — ne s'appliquerait a rien ici.

DEUX VOIES DE RESOLUTION, et l'ordre compte.

La page ne connait pas l'identifiant de l'article : elle reconstruit sa liste
par expression reguliere sur `law.content`, et trouve environ 193 articles la
ou la base en compte 230 pour le meme document. Le numero seul suffit donc dans
le cas nominal, mais pas toujours.

  1. Par la BASE, avec `article_reference.normalize_number` des deux cotes.
     Autoritaire : le texte envoye a Gemini vient de la base, jamais du client.
  2. A defaut, par l'EXTRAIT fourni par la page — mais seulement apres avoir
     verifie qu'il provient bien du document demande.

Cette verification est tout l'argument de surete. La route est publique, le
depot n'a aucun limiteur de debit, et du texte client accepte tel quel ferait
de l'endpoint un proxy de prompt gratuit : n'importe qui posterait 12 ko de
texte arbitraire et lirait la reponse du modele. Le controle ramene l'extrait
au rang de POINTEUR dans un document qu'on sert deja publiquement par
`GET /laws/{id}`.

Author: JuriX Team
"""

import logging
import re
import time
from typing import Any, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.law import Law
from app.schemas.law import ArticleExplanationResponse
from app.schemas.search import ChunkResult
from app.services.article_reference import normalize_number
from app.services.gemini_service import (
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiServiceError,
    get_gemini_service,
)
from app.services.prompts import (
    CONTEXT_TEMPLATE,
    EXPLAIN_TASK_TEMPLATES,
    build_context_string,
    get_system_prompt,
)

logger = logging.getLogger(__name__)


# Ton du produit, fixe cote serveur. Le client ne le choisit pas : ce serait lui
# laisser changer la voix du site depuis son navigateur.
EXPLANATION_PERSONA = "citoyen"

# Pas 0.7 comme le RAG conversationnel. Le texte a expliquer est fixe et la
# seule chose que l'alea peut produire ici est une obligation inventee qui ne
# figure pas dans l'article.
EXPLANATION_TEMPERATURE = 0.3

# Meme raison que RAGService.ANSWER_MAX_TOKENS : 1 000 jetons ne couvrent pas la
# reflexion d'un modele a raisonnement, et la reponse remonte alors tronquee ou
# vide. Redeclare ici plutot qu'importe de rag_service, dont la chaine d'imports
# tire SearchService et le reclasseur pour rien.
EXPLANATION_MAX_TOKENS = 4096

# Marqueurs de page poses par l'extraction. La page les retire avant d'afficher
# (parseContent), la base garde l'original : sans les neutraliser des deux
# cotes, aucune sonde ne correspondrait sur un document multipage.
_PAGE_MARK = re.compile(r"<<PAGE:?\s*\d+>>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

# Longueur de la sonde de provenance. 200 caracteres suffisent a prouver qu'un
# texte vient de ce document ; en exiger 40 au minimum empeche trois mots
# communs de correspondre a n'importe quoi.
SONDE_MIN = 40
SONDE_MAX = 200


class ExplanationError(Exception):
    """Echec d'explication. Traduit en 500 par la route."""


class ArticleNotFoundError(ExplanationError):
    """Article irresolu par les deux voies. Traduit en 404."""


class ExplanationQuotaError(Exception):
    """
    Quota de generation epuise. Traduit en 429.

    Volontairement PAS une sous-classe d'ExplanationError : la route attrape
    ExplanationError en dernier ressort pour rendre un 500, et cette erreur-ci
    doit passer avant sans dependre de l'ordre des blocs except.
    """


class ExplanationOverloadedError(Exception):
    """Service de generation sature. Traduit en 503. Voir ExplanationQuotaError."""


def _empreinte(texte: str) -> str:
    """
    Forme comparable d'un texte de document.

    La page retire les marqueurs de page, retaille les lignes et laisse tomber
    l'emphase markdown ; la base garde le texte d'origine. Comparer les deux
    formes brutes echouerait sur des espaces invisibles.
    """
    sans_marqueurs = _PAGE_MARK.sub(" ", texte or "").replace("*", "")
    return _WHITESPACE.sub(" ", sans_marqueurs).strip().lower()


class ExplanationService:
    """
    Explique un article a partir de son texte et de son document parent.

    Le LLM est un PARAMETRE et non un appel a get_gemini_service() : un test
    injecte une doublure. C'est la regle deja posee par reranker.py — la suivre
    ici evite d'avoir a patcher le SDK pour tester quoi que ce soit.
    """

    def __init__(self, db: AsyncSession, *, llm: Optional[Any] = None):
        self.db = db
        self.llm = llm if llm is not None else get_gemini_service()

    async def explain(
        self,
        law_id: int,
        number: str,
        language: str = "fr",
        excerpt: Optional[str] = None,
    ) -> ArticleExplanationResponse:
        """
        Produit l'explication d'un article.

        Raises:
            ArticleNotFoundError: loi absente, ou article irresolu
            ExplanationQuotaError: quota Gemini epuise
            ExplanationOverloadedError: service de generation sature
            ExplanationError: tout autre echec de generation
        """
        debut = time.perf_counter()
        langue = language if language in EXPLAIN_TASK_TEMPLATES else "fr"

        law = await self._charger_loi(law_id)
        chunks, article_id, numero_resolu, origine = self._resoudre(law, number, excerpt)

        prompt = self._construire_prompt(law, chunks, numero_resolu, langue)
        systeme = get_system_prompt(EXPLANATION_PERSONA, langue)

        texte = await self._generer(prompt, systeme)

        duree_ms = int((time.perf_counter() - debut) * 1000)
        logger.info(
            f"✅ Article {numero_resolu} de la loi {law_id} explique "
            f"({origine}, {duree_ms} ms)"
        )

        return ArticleExplanationResponse(
            law_id=law_id,
            article_id=article_id,
            number=numero_resolu,
            explanation=texte,
            language=langue,
            persona=EXPLANATION_PERSONA,
            resolved_from=origine,
            generation_time_ms=duree_ms,
        )

    async def _charger_loi(self, law_id: int) -> Law:
        """Charge la loi avec ses articles et sa categorie, en une requete."""
        resultat = await self.db.execute(
            select(Law)
            .options(selectinload(Law.articles), selectinload(Law.category))
            .where(Law.id == law_id)
        )
        law = resultat.scalar_one_or_none()
        if law is None:
            raise ArticleNotFoundError(f"Document {law_id} introuvable.")
        return law

    def _resoudre(
        self,
        law: Law,
        number: str,
        excerpt: Optional[str],
    ) -> Tuple[List[ChunkResult], Optional[int], str, str]:
        """
        Trouve l'article et construit les blocs de contexte.

        Returns:
            (chunks, article_id, numero_resolu, origine)
        """
        vise = normalize_number(number)

        # sorted() et non .sort() : muter la collection de l'identity map
        # reordonnerait aussi les articles vus par les autres consommateurs de
        # la meme session. Meme cle de tri que GET /laws/{id}.
        ordonnes = sorted(law.articles, key=lambda a: (a.order or 0, a.id))

        for position, article in enumerate(ordonnes):
            if normalize_number(article.number) == vise:
                chunks = [self._chunk(law, article, "explain")]
                # Les textes juridiques renvoient sans cesse a « l'article
                # precedent » : sans les voisins le modele explique une phrase
                # dont il ne connait pas le renvoi. Deux suffisent, et bornent
                # la depense.
                for voisin in self._voisins(ordonnes, position):
                    chunks.append(self._chunk(law, voisin, "explain_context"))
                return chunks, article.id, str(article.number), "database"

        return self._resoudre_par_extrait(law, vise, excerpt)

    def _resoudre_par_extrait(
        self,
        law: Law,
        vise: str,
        excerpt: Optional[str],
    ) -> Tuple[List[ChunkResult], Optional[int], str, str]:
        """
        Repli quand la base n'a pas de ligne pour ce numero.

        L'extrait n'est retenu que s'il se retrouve dans `law.content`. Sans ce
        controle, la route accepterait n'importe quel texte et servirait de
        proxy de prompt gratuit sur un quota partage avec le chat.
        """
        if not excerpt:
            raise ArticleNotFoundError(
                f"Article {vise} introuvable dans le document {law.id}."
            )

        sonde = _empreinte(excerpt)[:SONDE_MAX]
        if len(sonde) < SONDE_MIN or sonde not in _empreinte(law.content):
            raise ArticleNotFoundError(
                f"Article {vise} introuvable dans le document {law.id}."
            )

        chunk = ChunkResult(
            article_id=None,
            law_id=law.id,
            number=vise,
            article_title=None,
            section=None,
            page_number=None,
            content=excerpt,
            excerpt=excerpt[:400],
            reference=law.reference or "",
            law_title=law.title,
            type=law.type or "loi",
            language=law.language,
            status=law.status or "published",
            category_id=law.category_id,
            category_name=law.category.name if law.category else None,
            publication_date=law.publication_date,
            relevance_score=1.0,
            source="explain_excerpt",
        )
        return [chunk], None, vise, "excerpt"

    @staticmethod
    def _voisins(ordonnes: List[Any], position: int) -> List[Any]:
        """L'article precedent et le suivant, quand ils portent du texte."""
        candidats = []
        if position > 0:
            candidats.append(ordonnes[position - 1])
        if position + 1 < len(ordonnes):
            candidats.append(ordonnes[position + 1])
        return [a for a in candidats if (a.content or "").strip()]

    @staticmethod
    def _chunk(law: Law, article: Any, source: str) -> ChunkResult:
        """Construit un ChunkResult a partir des lignes ORM Law + Article."""
        content = article.content or ""
        return ChunkResult(
            article_id=article.id,
            law_id=law.id,
            number=str(article.number) if article.number else None,
            article_title=article.title,
            section=article.section,
            page_number=article.page_number,
            content=content,
            excerpt=content[:400],
            reference=law.reference or "",
            law_title=law.title,
            type=law.type or "loi",
            language=law.language,
            status=law.status or "published",
            category_id=law.category_id,
            category_name=law.category.name if law.category else None,
            publication_date=law.publication_date,
            relevance_score=1.0,
            source=source,
        )

    @staticmethod
    def _construire_prompt(
        law: Law,
        chunks: List[ChunkResult],
        numero: str,
        langue: str,
    ) -> str:
        """
        Assemble l'en-tete du document parent, le contexte et la tache.

        L'assemblage est ici et non dans prompts.py : ce module-la ne connait
        aucun modele ORM et doit le rester.

        Le bornage est celui de build_context_string — 6 000 caracteres par bloc,
        24 000 au total, tronques sur une frontiere de paragraphe, le PREMIER
        bloc etant toujours conserve. Mettre la cible en tete est donc ce qui
        garantit qu'elle n'est jamais la sacrifiee. Ne rien ajouter par-dessus.
        `law.content` n'entre pas dans le prompt : des centaines de kilo-octets
        sur ce corpus, tronques en bruit.
        """
        entete = [
            "[Document parent]",
            f"Référence : {law.reference or '—'} — {law.title}",
            f"Type : {law.type or 'loi'} | Langue : {law.language or '—'}",
        ]
        if law.publication_date:
            entete.append(f"Publié le : {law.publication_date.isoformat()}")
        if law.category:
            entete.append(f"Catégorie : {law.category.name}")
        entete.append(f"Le document compte {len(law.articles)} articles.")

        contexte = CONTEXT_TEMPLATE.format(context_docs=build_context_string(chunks))
        tache = EXPLAIN_TASK_TEMPLATES[langue].format(number=numero)

        return "\n".join(entete) + "\n\n" + contexte + "\n" + tache

    async def _generer(self, prompt: str, systeme: str) -> str:
        """
        Appelle le modele et traduit ses echecs.

        L'ordre des blocs except n'est pas arbitraire : GeminiOverloadedError
        est une sous-classe de GeminiServiceError, donc le cas general vient en
        dernier.
        """
        try:
            reponse = await self.llm.generate(
                prompt=prompt,
                system=systeme,
                temperature=EXPLANATION_TEMPERATURE,
                max_tokens=EXPLANATION_MAX_TOKENS,
            )
        except GeminiQuotaError as e:
            raise ExplanationQuotaError(str(e)) from e
        except GeminiOverloadedError as e:
            raise ExplanationOverloadedError(str(e)) from e
        except GeminiServiceError as e:
            raise ExplanationError(str(e)) from e

        texte = (reponse or {}).get("response", "")
        if not texte.strip():
            raise ExplanationError("Le modèle n'a produit aucune explication.")
        return texte
