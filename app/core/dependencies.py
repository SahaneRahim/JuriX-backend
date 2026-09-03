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

from app.services.document_classifier import DocumentClassifier
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.language_detector import LanguageDetector
from app.services.search_service import SearchService

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


def clear_detector_cache():
    """
    Efface le cache du detector (force rechargement).

    Utile pour:
    - Tests (isolation entre tests)
    - Mise à jour des modèles
    - Debugging

    Example:
        >>> clear_detector_cache()
        >>> detector = get_language_detector()  # Nouvelle instance
    """
    get_language_detector.cache_clear()
    logger.info("🗑️  Cache LanguageDetector effacé")


@lru_cache()
def get_document_classifier() -> DocumentClassifier:
    """
    Factory pour DocumentClassifier (singleton).

    Utilise @lru_cache pour créer une seule instance partagée
    entre toutes les requêtes. Évite le rechargement des modèles
    ML à chaque requête (Phase 3).

    Returns:
        Instance singleton de DocumentClassifier

    Example:
        >>> from fastapi import Depends
        >>> classifier = Depends(get_document_classifier)
    """
    logger.info("📦 Création du singleton DocumentClassifier")
    return DocumentClassifier()


def clear_classifier_cache():
    """
    Efface le cache du classifier (force rechargement).

    Utile pour:
    - Tests (isolation entre tests)
    - Mise à jour des modèles ML
    - Debugging

    Example:
        >>> clear_classifier_cache()
        >>> classifier = get_document_classifier()  # Nouvelle instance
    """
    get_document_classifier.cache_clear()
    logger.info("🗑️  Cache DocumentClassifier effacé")


# La fabrique vit dans app/services/embedding_service.py et est simplement
# re-exportee ici : il y avait deux singletons concurrents, celui-ci et le
# _embedding_service_instance prive de SearchService. Elle ne peut pas remonter
# dans ce module — dependencies.py importe SearchService, qui importe
# EmbeddingService : le cycle serait immediat.


def clear_embedding_service_cache():
    """
    Efface le cache du service d'embeddings (force rechargement).

    Utile pour:
    - Tests (isolation entre tests)
    - Mise à jour des modèles
    - Debugging

    Example:
        >>> clear_embedding_service_cache()
        >>> service = get_embedding_service()  # Nouvelle instance
    """
    get_embedding_service.cache_clear()
    logger.info("🗑️  Cache EmbeddingService effacé")


# Exports for dependency injection
__all__ = [
    "get_language_detector",
    "get_document_classifier",
    "get_embedding_service",
    "clear_detector_cache",
    "clear_classifier_cache",
    "clear_embedding_service_cache",
]
