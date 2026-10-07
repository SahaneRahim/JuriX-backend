"""
Choix du service de generation, et familles d'erreurs communes aux fournisseurs.

Le chat, la comparaison et l'explication d'article choisissaient chacun leur
fournisseur (Mistral ou Gemini) par un `if settings.LLM_PROVIDER` recopie trois
fois — et deux des trois copies utilisaient `settings` sans l'importer : chaque
comparaison et chaque explication repondait 500. Le choix vit ici, une fois.

Les erreurs sont regroupees par NATURE, quel que soit le fournisseur : un appelant
attrape `ERREURS_QUOTA` sans savoir qui a repondu. L'ordre des `except` reste a la
charge de l'appelant : quota et saturation AVANT l'erreur generique, dont elles
derivent.

Author: JuriX Team
"""

from typing import Any, Dict, Optional

from app.core.config import settings
from app.services.gemini_service import (
    GeminiBudgetEpuiseError,
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiServiceError,
    get_gemini_service,
)
from app.services.mistral_service import (
    FIN_INTERROMPUE,
    FINS_INCOMPLETES,
    MistralOverloadedError,
    MistralQuotaError,
    MistralServiceError,
    get_mistral_service,
)

# Quota epuise : la reponse porte un delai, l'utilisateur doit attendre.
ERREURS_QUOTA = (GeminiQuotaError, MistralQuotaError)
# Saturation passagere du fournisseur : un nouvel essai aboutira.
ERREURS_SATURATION = (GeminiOverloadedError, MistralOverloadedError)
# Toute erreur de generation, y compris les deux familles ci-dessus.
ERREURS_LLM = (GeminiServiceError, MistralServiceError)
# Budget epuise avant toute sortie : un budget plus grand peut suffire.
ERREURS_BUDGET = (GeminiBudgetEpuiseError,)

# Ajoutees a une reponse que le modele n'a pas pu finir : sans elles, une
# phrase coupee ressemble a une reponse complete. Deux causes, deux remedes :
# la LONGUEUR (poser une question plus precise) et l'INTERRUPTION — flux
# rompu, ou fin « error » du fournisseur — (reposer la question). Dans la
# langue de la reponse ; sans conseil quand il n'y a pas de question a
# reformuler (l'explication d'un article).
_MENTIONS = {
    "fr": {
        "longueur": "Réponse interrompue : longueur maximale atteinte.",
        "conseil_longueur": "Posez une question plus précise pour obtenir la suite.",
        "interruption": "Réponse interrompue : la connexion avec le modèle a été coupée.",
        "conseil_interruption": "Reposez la question pour obtenir une réponse complète.",
    },
    "en": {
        "longueur": "Answer cut short: maximum length reached.",
        "conseil_longueur": "Ask a more specific question to get the rest.",
        "interruption": "Answer interrupted: the connection with the model was lost.",
        "conseil_interruption": "Ask again to get a complete answer.",
    },
}
# Fins dues a une interruption, et non a la longueur.
_FINS_INTERROMPUES = frozenset({FIN_INTERROMPUE, "ERROR"})


def mention_de_fin(fin: Optional[str], langue: str = "fr", conseil: bool = True) -> str:
    """
    La mention a ajouter a une reponse selon sa fin, ou "" si elle est
    complete. `fin` suit le vocabulaire commun (« STOP », « MAX_TOKENS »...) ;
    Gemini et Mistral le partagent.
    """
    if fin not in FINS_INCOMPLETES:
        return ""
    textes = _MENTIONS.get(langue, _MENTIONS["fr"])
    cause = "interruption" if fin in _FINS_INTERROMPUES else "longueur"
    phrase = textes[cause] + (" " + textes[f"conseil_{cause}"] if conseil else "")
    return f"\n\n*({phrase})*"


def mention_de_reponse(reponse: Dict[str, Any], langue: str = "fr", conseil: bool = True) -> str:
    """La mention d'une reponse de `generate` : coupee si elle le dit."""
    if not (reponse or {}).get("tronquee"):
        return ""
    return mention_de_fin(reponse.get("fin") or "MAX_TOKENS", langue, conseil)


# Les mentions francaises completes, telles que le chat les affiche.
MENTION_REPONSE_TRONQUEE = mention_de_fin("MAX_TOKENS")
MENTION_REPONSE_INTERROMPUE = mention_de_fin(FIN_INTERROMPUE)


def get_llm_service() -> Any:
    """Le service de generation retenu par LLM_PROVIDER."""
    if settings.LLM_PROVIDER == "mistral":
        return get_mistral_service()
    return get_gemini_service()
