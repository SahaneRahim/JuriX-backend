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

from typing import Any

from app.core.config import settings
from app.services.gemini_service import (
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiServiceError,
    get_gemini_service,
)
from app.services.mistral_service import (
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


def get_llm_service() -> Any:
    """Le service de generation retenu par LLM_PROVIDER."""
    if settings.LLM_PROVIDER == "mistral":
        return get_mistral_service()
    return get_gemini_service()
