"""
Dependency injection pour FastAPI.

Ce module fournit des factory functions pour les services partagés.
Utilise le pattern singleton avec @lru_cache pour efficacité mémoire.

Usage dans routes:
    @router.post("/detect")
    async def detect_language(
        detector: LanguageDetector = Depends(get_language_detector)
    ):
        result = detector.detect(text)
"""

import logging
from functools import lru_cache

from app.services.embedding_service import get_embedding_service
from app.services.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


@lru_cache()
def get_language_detector() -> LanguageDetector:
    """
    Factory pour LanguageDetector (singleton).

    Utilise @lru_cache pour créer une seule instance partagée
    entre toutes les requêtes. Évite le rechargement des modèles
    NLP (~200MB) à chaque requête.

    Returns:
        Instance singleton de LanguageDetector

    Raises:
        LanguageDetectionError: Si les modèles ne peuvent pas être chargés

    Example:
        >>> from fastapi import Depends
        >>> detector = Depends(get_language_detector)
    """
    logger.info("📦 Création du singleton LanguageDetector")
    return LanguageDetector()




# clear_detector_cache() et clear_embedding_service_cache() ont ete retirees :
# aucun appelant, ni en production ni dans les tests. Les caches @lru_cache
# qu'elles vidaient vivent le temps du processus.


# Exports for dependency injection
__all__ = [
    "get_language_detector",
    "get_embedding_service",
]
