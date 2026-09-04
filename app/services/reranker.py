"""
Re-ranking des chunks remontes par la recherche.

Le classement issu de la fusion RRF melange deux signaux (plein texte et
vecteurs) mais n'en connait aucun autre. Or la forme dominante des questions de
ce produit est "que dit l'article 161 du Code OHADA" — un cas ou la bonne
reponse est connaissable sans aucune semantique. Et la composition du corpus
joue contre le classement brut : 52,6 % des documents sont des listes
nominatives dont les articles d'execution ("sera enregistre, publie au Journal
Officiel") sont quasi identiques d'un decret a l'autre, s'encodent au meme
endroit et saturent le haut du classement.

Deux etages :

- Etage 1 (`rerank_chunks`), toujours actif : traits lexicaux, aucune
  dependance, aucun reseau, quelques millisecondes. Fonction PURE — pas de
  session, pas d'E/S — donc testable et utilisable depuis le harnais
  d'evaluation sans base de donnees.
- Etage 2 (`rerank_with_llm`), optionnel : notation des meilleurs chunks par
  Gemini. Ajoute un appel facture et plusieurs centaines de millisecondes sur le
  chemin critique, d'ou son reglage a False par defaut. Il DEGRADE toujours vers
  l'ordre de l'etage 1 et ne leve jamais : son point d'appel est dans le chemin
  RAG, dont les exceptions deviennent des HTTP 500.

Les poids ci-dessous ne sont PAS calibres sur ce corpus. Ils le seront par
scripts/eval/run_eval.py, et la valeur retenue devra citer le fichier de run.

Author: JuriX Team
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.schemas.search import ChunkResult
from app.services.text_features import content_words, fold_accents, jaccard

logger = logging.getLogger(__name__)


# ==================== POIDS (non calibres) ====================

W_FUSION = 1.00       # score de fusion entrant
# Le poids le plus fort, volontairement : "article 161 du Code OHADA" est la
# forme dominante des questions de ce produit, et le seul cas ou la bonne
# reponse est connaissable sans aucune semantique.
W_ARTICLE_NUM = 0.80  # le numero demande correspond
W_DENSITY = 0.40      # densite des termes de la question dans le contenu
W_PHRASE = 0.35       # expression de la question presente telle quelle
W_LAW_TITLE = 0.25    # recoupement avec le titre / la reference de la loi
W_SECTION = 0.15      # recoupement avec l'en-tete de section
PENALTY_BOILERPLATE = 0.25  # formule d'execution reconnue
PENALTY_SHORT = 0.10        # chunk trop court pour repondre a quoi que ce soit
SHORT_CHUNK_CHARS = 80

# Formules d'execution, quasi identiques dans des milliers de decrets.
_BOILERPLATE = re.compile(
    r"(sera\s+enregistr|publi\w*\s+au\s+journal\s+officiel"
    r"|en\s+francais\s+et\s+en\s+anglais)",
    re.IGNORECASE,
)


def _article_number_in(question: str) -> Optional[str]:
    """Numero d'article mentionne dans la question, forme normalisee."""
    from app.services.rag_service import _normalize_article_number

    match = re.search(
        r"(?:article|art\.?|section)\s+([LRD]\s*)?(\d+(?:[-.]\d+)*)",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_article_number(match.group(2))


def _phrase_score(question_folded: str, content_folded: str) -> float:
    """
    1.0 si la question apparait telle quelle dans le contenu, sinon la part de
    ses bigrammes qui s'y trouvent.

    Les bigrammes plutot que les mots isoles : "responsabilite des dirigeants"
    et "dirigeants" ne designent pas la meme chose, et c'est exactement ce que
    la recherche vectorielle confond.
    """
    if not question_folded or not content_folded:
        return 0.0
    if question_folded in content_folded:
        return 1.0

    words = [w for w in question_folded.split() if len(w) > 2]
    if len(words) < 2:
        return 0.0

    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    hits = sum(1 for bigram in bigrams if bigram in content_folded)
    return hits / len(bigrams)


def score_chunk(
    chunk: ChunkResult,
    question: str,
    *,
    question_words: Optional[set] = None,
    question_folded: Optional[str] = None,
    requested_article: Optional[str] = None,
) -> float:
    """Score brut d'un chunk, avant normalisation sur le lot."""
    from app.services.rag_service import _normalize_article_number

    q_words = question_words if question_words is not None else content_words(question)
    q_folded = question_folded if question_folded is not None else fold_accents(question)

    content = chunk.content or chunk.excerpt or ""
    content_folded = fold_accents(content)
    chunk_words = content_words(content)

    score = W_FUSION * float(chunk.relevance_score or 0.0)

    if requested_article and chunk.number:
        if _normalize_article_number(chunk.number) == requested_article:
            score += W_ARTICLE_NUM

    if q_words:
        density = len(q_words & chunk_words) / len(q_words)
        score += W_DENSITY * min(density, 1.0)

    score += W_PHRASE * _phrase_score(q_folded, content_folded)

    law_words = content_words(f"{chunk.law_title or ''} {chunk.reference or ''}")
    score += W_LAW_TITLE * jaccard(q_words, law_words)

    if chunk.section:
        score += W_SECTION * jaccard(q_words, content_words(chunk.section))

    # Les articles d'execution sont repetes a l'identique dans des milliers de
    # decrets : ils se concentrent au meme endroit de l'espace vectoriel et
    # saturent le haut du classement. La formule reconnue est un signal bien
    # plus sur que la seule brievete, d'ou deux penalites distinctes — un
    # article court peut etre la bonne reponse, une formule d'enregistrement
    # ne l'est jamais.
    if _BOILERPLATE.search(content_folded):
        score -= PENALTY_BOILERPLATE
    elif len(content) < SHORT_CHUNK_CHARS:
        score -= PENALTY_SHORT

    return score


def rerank_chunks(
    question: str,
    chunks: List[ChunkResult],
    *,
    requested_article: Optional[str] = None,
) -> List[ChunkResult]:
    """
    Reclasse les chunks par traits lexicaux. Fonction pure.

    Ne touche PAS a relevance_score — contrairement a _rrf_fusion, qui l'ecrase
    en place. Le score de recuperation reste inspectable : c'est ce qui permet
    d'attribuer un ecart au re-ranking plutot qu'a la recherche. Le classement
    se lit dans `rerank_score`.

    Args:
        question: la question BRUTE, pas la version reduite aux mots-cles :
            retirer les mots vides detruit precisement la structure d'expression
            dont depend le trait de phrase.
        chunks: resultats de la recherche, dans leur ordre d'origine
        requested_article: numero d'article deja extrait, sinon deduit

    Returns:
        Une nouvelle liste, permutation de l'entree, triee par score decroissant.
    """
    if not chunks:
        return []

    wanted = requested_article or _article_number_in(question)
    q_words = content_words(question)
    q_folded = fold_accents(question)

    scored = [
        (
            score_chunk(
                chunk, question,
                question_words=q_words,
                question_folded=q_folded,
                requested_article=wanted,
            ),
            index,
            chunk,
        )
        for index, chunk in enumerate(chunks)
    ]

    raw = [item[0] for item in scored]
    lowest, highest = min(raw), max(raw)
    span = highest - lowest

    ranked: List[ChunkResult] = []
    for raw_score, _index, chunk in sorted(scored, key=lambda x: (-x[0], x[1])):
        # Normalisation min-max sur le lot : rerank_score est declare le=1.0.
        # Lot homogene, tout le monde a 1.0 — l'ordre d'origine departage.
        normalised = 1.0 if span < 1e-9 else (raw_score - lowest) / span
        ranked.append(chunk.model_copy(update={"rerank_score": normalised}))

    return ranked


# ==================== ETAGE 2 : notation par le modele ====================

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["id", "score"],
            },
        }
    },
    "required": ["scores"],
}

_LLM_PROMPT = """Tu classes des extraits de textes juridiques camerounais selon leur utilité pour répondre à une question.

Question de l'utilisateur :
{question}

Extraits numérotés :
{blocks}

Attribue à chaque extrait une note de 0 à 10 : 10 si l'extrait répond directement à la question, 0 s'il est hors sujet. Réponds uniquement par le JSON demandé, une entrée par extrait, en reprenant les numéros donnés."""


def _format_blocks(chunks: List[ChunkResult], excerpt_chars: int = 600) -> str:
    parts = []
    for index, chunk in enumerate(chunks):
        header = f"[{index}] {chunk.reference}"
        if chunk.number:
            header += f" — article {chunk.number}"
        if chunk.section:
            header += f" ({chunk.section})"
        body = (chunk.content or chunk.excerpt or "")[:excerpt_chars]
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _parse_llm_scores(raw: str, expected: int) -> Dict[int, float]:
    """
    Extrait les notes. Tout ce qui n'est pas exploitable est ignore, jamais
    propage : identifiants inconnus ecartes, notes bornees a [0, 10].
    """
    payload = json.loads(raw)
    scores: Dict[int, float] = {}
    for entry in payload.get("scores", []):
        try:
            index = int(entry["id"])
            value = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < expected:
            scores[index] = max(0.0, min(10.0, value))
    return scores


async def rerank_with_llm(
    question: str,
    chunks: List[ChunkResult],
    *,
    llm: Any,
    top_n: int = 20,
    timeout: float = 4.0,
    blend: float = 0.5,
) -> List[ChunkResult]:
    """
    Notation des `top_n` premiers chunks par le modele, melangee a l'etage 1.

    Le LLM est un PARAMETRE et non un appel a get_gemini_service() : un test
    injecte une doublure, et le harnais d'evaluation peut changer de modele.

    Toute defaillance — expiration, exception, reponse vide, JSON invalide,
    identifiants inconnus — renvoie l'ordre recu, inchange. Aucune exception ne
    sort : l'appelant est dans le chemin RAG, dont les exceptions deviennent des
    HTTP 500.

    Le melange (moitie etage 1, moitie note du modele) evite qu'une mauvaise
    reponse du modele enterre un chunk sur lequel tous les signaux lexicaux
    s'accordent.
    """
    import asyncio

    if not chunks or llm is None:
        return chunks

    head, tail = chunks[:top_n], chunks[top_n:]

    try:
        response = await asyncio.wait_for(
            llm.generate(
                prompt=_LLM_PROMPT.format(
                    question=question, blocks=_format_blocks(head)
                ),
                temperature=0.0,
                max_tokens=1024,
                response_mime_type="application/json",
                response_schema=_LLM_SCHEMA,
            ),
            timeout=timeout,
        )
        raw = (response or {}).get("response", "") if isinstance(response, dict) else ""
        if not raw:
            logger.warning("⚠️ Re-ranking LLM : réponse vide, ordre conservé")
            return chunks

        scores = _parse_llm_scores(raw, len(head))
        if not scores:
            logger.warning("⚠️ Re-ranking LLM : aucune note exploitable, ordre conservé")
            return chunks

    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Re-ranking LLM : expiration après {timeout}s, ordre conservé")
        return chunks
    except Exception as exc:
        logger.warning(f"⚠️ Re-ranking LLM indisponible ({exc}), ordre conservé")
        return chunks

    blended: List[ChunkResult] = []
    for index, chunk in enumerate(head):
        stage1 = chunk.rerank_score if chunk.rerank_score is not None else chunk.relevance_score
        if index in scores:
            value = (1 - blend) * float(stage1) + blend * (scores[index] / 10.0)
        else:
            # Chunk envoye mais non note : il garde son score d'etage 1.
            value = float(stage1)
        blended.append(chunk.model_copy(update={"rerank_score": value}))

    blended.sort(key=lambda c: c.rerank_score, reverse=True)
    return blended + tail
