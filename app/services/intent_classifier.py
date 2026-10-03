"""
Aiguillage d'un message utilisateur : question de droit, ou conversation ?

POURQUOI CE MODULE EXISTE. `RAGService.ask()` envoyait TOUTE question a la
recherche hybride, sans exception. Une salutation remonte quand meme des
chunks — le plein texte et le vectoriel trouvent toujours quelque chose — si
bien que le garde-fou « aucun resultat » ne se declenchait jamais. Mesure sur
une vraie session : « comment vas tu ? » a produit un pave a rubriques
(« **Source legale** L'information demandee n'est pas presente dans les
documents fournis, notamment la Loi N°2023/014 portant Code Minier ») et un
bloc « Sources » citant un article du Code Minier. Le cout n'etait pas
qu'esthetique : chaque salutation payait un appel d'embedding Gemini, occupait
`embedding_cache` sept jours et laissait une ligne dans `search_events`.

CE MODULE NE CLASSE PAS DES DOCUMENTS. `legal_domain_classifier.py` range un
texte du corpus dans un domaine juridique a l'ingestion ; ici on lit la
question de l'utilisateur, ce qui n'a ni les memes entrees ni le meme verdict.

LA MECANIQUE EST CELLE DE `reranker.rerank_with_llm` : le modele est un
parametre, l'appel est borne par `asyncio.wait_for`, et AUCUNE exception ne
sort. Le repli d'un composant optionnel, c'est l'absence du composant.

Author: JuriX Team
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.core.config import settings
from app.services.intent_local import ClasseurLocal, question_precedente
from app.services.prompts import (
    CLASSIFICATION_SYSTEM,
    construire_prompt_de_classification,
    format_conversation_history,
)
from app.services.reranker import _article_number_in

logger = logging.getLogger(__name__)


# Ordre explicite, jamais l'ordre d'insertion d'un dictionnaire.
INTENTS: Tuple[str, ...] = ("juridique", "smalltalk", "meta", "hors_sujet")

# Le seul qui emprunte le RAG. Les trois autres partagent le chemin
# conversationnel, avec un prompt different chacun.
INTENT_JURIDIQUE = INTENTS[0]

# Messages dont la TOTALITE est une formule de politesse. Le rapprochement se
# fait sur le message ENTIER normalise, jamais par sous-chaine : « bonjour »
# court-circuite, « bonjour, puis-je divorcer sans avocat ? » non — c'est
# precisement le piege qu'un filtre lexical naif tend, et il classerait une
# vraie question de droit en salutation.
#
# Le gain n'est pas theorique : sur le palier gratuit, chaque message coute
# desormais deux appels au modele, et « bonjour » ouvre une conversation sur
# deux. Ceux-la n'en coutent plus aucun.
_SALUTATIONS_EXACTES = frozenset({
    "bonjour", "bonsoir", "salut", "coucou", "hello", "hi", "hey",
    "merci", "merci beaucoup", "merci bien", "thanks", "thank you",
    "au revoir", "bye", "a bientot", "bonne journee", "bonne soiree",
    "ok", "d'accord", "tres bien", "parfait",
})

# Ponctuation et accents retires avant comparaison : « Bonjour ! », « bonjour »
# et « BONJOUR. » sont le meme message.
_PONCTUATION = str.maketrans("", "", "!?.,;:…\"'«»()")


def _est_une_salutation_pure(question: str) -> bool:
    """Le message ENTIER est-il une formule de politesse, et rien d'autre ?"""
    from app.services.text_features import fold_accents

    nettoye = fold_accents((question or "").strip().lower()).translate(_PONCTUATION)
    return " ".join(nettoye.split()) in _SALUTATIONS_EXACTES


@dataclass(frozen=True)
class IntentResult:
    """
    Verdict de routage.

    `intent` vaut TOUJOURS l'un de INTENTS, jamais None : l'appelant n'a pas de
    branche « je ne sais pas » a ecrire. `rule` nomme ce qui a decide, ce qui
    rend chaque erreur diagnosticable dans les journaux sans rejouer la
    requete — un `defaut-timeout` massif et un `defaut-json` massif appellent
    des corrections opposees.
    """

    intent: str
    confidence: float
    rule: str


# Schema impose a la sortie du modele. `gemini_service.generate` expose
# `response_schema` explicitement, precisement pour ce genre d'appel.
_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intention": {"type": "string", "enum": list(INTENTS)},
        "confiance": {"type": "number"},
    },
    "required": ["intention"],
}


def _parse_intent(raw: Any) -> Optional[Tuple[str, float]]:
    """
    Lit le verdict du modele. Rend None sur tout ce qui n'est pas exploitable.

    Fonction PURE : elle ne journalise pas et ne leve pas, ce qui la rend
    testable sans doublure.
    """
    if not raw or not isinstance(raw, str):
        return None

    try:
        charge = json.loads(raw)
    except (ValueError, TypeError):
        return None

    if not isinstance(charge, dict):
        return None

    intention = charge.get("intention")
    if intention not in INTENTS:
        return None

    try:
        confiance = float(charge.get("confiance", 0.5))
    except (TypeError, ValueError):
        confiance = 0.5

    return intention, max(0.0, min(1.0, confiance))


# ==================== CLASSEMENT LOCAL ====================

_classeur_local: Optional[ClasseurLocal] = None


def _similarite_corpus(vecteur) -> Optional[float]:
    """
    Similarite du message avec l'article vectorise le plus proche (index HNSW).
    None si la base ne repond pas : le garde-fou s'efface, il ne bloque rien.
    """
    from sqlalchemy import text

    from app.core.database import SyncSessionLocal

    litteral = "[" + ",".join(f"{float(x):.6f}" for x in vecteur) + "]"
    try:
        with SyncSessionLocal() as session:
            return session.execute(text(
                "SELECT 1 - (embedding <=> CAST(:v AS vector)) FROM articles "
                "WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:v AS vector) LIMIT 1"
            ), {"v": litteral}).scalar()
    except Exception as exc:
        logger.warning("🧭 Garde-fou du corpus indisponible : %s", exc)
        return None


def classeur_local() -> Optional[ClasseurLocal]:
    """Le classeur local du processus, sur le service d'embeddings partage."""
    global _classeur_local
    if _classeur_local is None:
        from app.services import search_service

        search_service._init_global_singletons()
        service = search_service._embedding_service_instance
        if service is None:
            return None
        _classeur_local = ClasseurLocal(service, similarite_corpus=_similarite_corpus)
    return _classeur_local


def prechauffer_classement_local() -> None:
    """
    Vectorise la banque d'exemples au demarrage : quelques secondes de calcul
    qui, sinon, retomberaient sur la premiere question posee. Synchrone, a
    lancer hors de la boucle d'evenements. Ne leve jamais.
    """
    if not settings.INTENT_ROUTING_ENABLED or settings.INTENT_CLASSIFIER != "local":
        return
    try:
        classeur = classeur_local()
        if classeur is not None:
            classeur.banque()
    except Exception as exc:
        logger.warning("🧭 Banque d'intentions non prechauffee : %s", exc)


async def _classer_localement(question: str, history: Optional[List[Any]]) -> IntentResult:
    """Classement local ; sur toute panne, « juridique » — jamais d'exception."""
    try:
        classeur = classeur_local()
        if classeur is None:
            return IntentResult(INTENT_JURIDIQUE, 0.0, "defaut-local-indisponible")
        intention, confiance, regle = await asyncio.wait_for(
            asyncio.to_thread(classeur.classer, question, question_precedente(history)),
            timeout=settings.INTENT_TIMEOUT_S,
        )
        return IntentResult(intention, confiance, regle)
    except asyncio.TimeoutError:
        logger.warning("🧭 Classement local expire : repli sur juridique")
        return IntentResult(INTENT_JURIDIQUE, 0.0, "defaut-local-timeout")
    except Exception as exc:
        logger.warning("🧭 Classement local en echec (%s) : repli sur juridique", exc)
        return IntentResult(INTENT_JURIDIQUE, 0.0, "defaut-local-erreur")


async def classify_intent(
    question: str,
    *,
    llm: Any,
    history: Optional[List[Any]] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> IntentResult:
    """
    Classe un message. Ne leve jamais.

    Args:
        question: le message de l'utilisateur, tel qu'il l'a ecrit
        llm: service de generation. PARAMETRE et non `get_gemini_service()` en
            dur, meme raison que `rerank_with_llm` : un test injecte une
            doublure, le harnais d'evaluation peut changer de modele.
        history: les derniers messages, pour les questions de suivi. « et
            l'article 12 ? » n'est juridique que par ce qui precede.

    Returns:
        IntentResult, toujours. Sur echec : `juridique`, ce qui redonne le
        comportement anterieur.

    POURQUOI LE DEFAUT EST « juridique ». L'erreur n'est pas symetrique.
    Classer une salutation en juridique coute un pave disgracieux — le bug
    d'hier, donc une degradation vers l'etat anterieur. Classer « quelles sont
    les conditions du permis de recherche ? » en conversation coute une reponse
    INVENTEE et SANS SOURCE sur une question de droit, et l'utilisateur n'a
    aucun moyen de s'en apercevoir : on lui a justement retire le bloc Sources
    qui l'aurait alerte. S'y ajoute que les pannes sont correlees a la charge,
    pas au contenu : un defaut conversationnel ferait basculer TOUTE la
    plateforme en mode bavardage pendant une saturation du fournisseur.
    """
    if not settings.INTENT_ROUTING_ENABLED:
        return IntentResult(INTENT_JURIDIQUE, 1.0, "desactive")

    # Court-circuit deterministe. « que dit l'article 33 du Code Minier ? » n'a
    # pas besoin d'un modele pour etre reconnue comme juridique, et le
    # re-ranking documente cette forme comme dominante sur ce produit. La regex
    # est celle du re-ranking, pas une seconde : deux auraient diverge.
    #
    # Pas de court-circuit lexical symetrique cote salutation (une liste
    # « bonjour / merci / salut ») : il classerait « Bonjour, puis-je
    # divorcer ? » en smalltalk. L'asymetrie est voulue — on ne court-circuite
    # que dans le sens qui degrade vers le comportement actuel.
    if _article_number_in(question or ""):
        return IntentResult(INTENT_JURIDIQUE, 1.0, "court-circuit-article")

    # Symetrique, et sur une egalite STRICTE du message entier : c'est ce qui
    # le rend sur. Voir _SALUTATIONS_EXACTES.
    if _est_une_salutation_pure(question):
        return IntentResult("smalltalk", 1.0, "court-circuit-salutation")

    # Par defaut, sans Gemini : voir intent_local.py. Une panne retombe sur
    # « juridique », jamais sur le modele — le choix du local est precisement
    # de ne plus depenser le quota pour classer.
    if settings.INTENT_CLASSIFIER == "local":
        return await _classer_localement(question, history)

    prompt = construire_prompt_de_classification(
        question, format_conversation_history(history or [])
    )

    try:
        reponse = await asyncio.wait_for(
            llm.generate(
                prompt=prompt,
                # Systeme court et neutre, OBLIGATOIRE : `generate` retombe sur
                # `GeminiService.SYSTEM_INSTRUCTION` quand `system` est absent,
                # et cette instruction-la impose de citer des articles et de
                # terminer par « Sources: ». Sur une tache de classification,
                # elle produirait de la prose au lieu du JSON demande.
                system=CLASSIFICATION_SYSTEM,
                temperature=0.0,
                max_tokens=max_tokens or settings.INTENT_MAX_TOKENS,
                reflexion=settings.GEMINI_REFLEXION_CLASSIFICATION,
                response_mime_type="application/json",
                response_schema=_INTENT_SCHEMA,
            ),
            timeout=timeout if timeout is not None else settings.INTENT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("🧭 Classification expiree : repli sur juridique")
        return IntentResult(INTENT_JURIDIQUE, 0.0, "defaut-timeout")
    except Exception as exc:
        # Quota, saturation, panne reseau : tout est avale. Une defaillance du
        # routeur ne doit jamais faire echouer la question de l'utilisateur.
        logger.warning("🧭 Classification en echec (%s) : repli sur juridique", exc)
        return IntentResult(INTENT_JURIDIQUE, 0.0, "defaut-erreur")

    # Extraction TOTALE, jamais `reponse["response"]`. `generate` rend un dict,
    # mais ce module promet de ne jamais lever : une doublure ou une future
    # variante rendant une chaine nue ne doit pas transformer le routeur en
    # panne. Tout ce qui n'est pas exploitable retombe sur le defaut.
    brut = reponse.get("response") if isinstance(reponse, dict) else reponse
    verdict = _parse_intent(brut)
    if verdict is None:
        logger.warning("🧭 Verdict illisible : repli sur juridique")
        return IntentResult(INTENT_JURIDIQUE, 0.0, "defaut-json")

    intention, confiance = verdict
    return IntentResult(intention, confiance, "llm")
