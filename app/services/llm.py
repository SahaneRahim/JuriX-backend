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

# Ajoutee a une reponse que le modele n'a pas pu finir : sans elle, une phrase
# coupee ressemble a une reponse complete.
MENTION_REPONSE_TRONQUEE = (
    "\n\n*(Réponse interrompue : longueur maximale atteinte. "
    "Posez une question plus précise pour obtenir la suite.)*"
)
# Le flux s'est rompu en cours de reponse, et la suite demandee a echoue :
# incomplete, mais pas a cause de sa longueur.
MENTION_REPONSE_INTERROMPUE = (
    "\n\n*(Réponse interrompue : la connexion avec le modèle a été coupée. "
    "Reposez la question pour obtenir une réponse complète.)*"
)


def mention_de_fin(fin: Optional[str]) -> str:
    """
    La mention a ajouter a une reponse selon sa fin, ou "" si elle est
    complete. `fin` suit le vocabulaire commun (« STOP », « MAX_TOKENS »...) ;
    Gemini et Mistral le partagent.
    """
    if fin == FIN_INTERROMPUE:
        return MENTION_REPONSE_INTERROMPUE
    if fin in FINS_INCOMPLETES:
        return MENTION_REPONSE_TRONQUEE
    return ""


def mention_de_reponse(reponse: Dict[str, Any]) -> str:
    """La mention d'une reponse de `generate` : coupee si elle le dit."""
    if not (reponse or {}).get("tronquee"):
        return ""
    return mention_de_fin(reponse.get("fin") or "MAX_TOKENS")


def get_llm_service() -> Any:
    """Le service de generation retenu par LLM_PROVIDER."""
    if settings.LLM_PROVIDER == "mistral":
        return get_mistral_service()
    return get_gemini_service()
