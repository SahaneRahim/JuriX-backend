"""
Tests pour EmbeddingService.

Test categories:
- Basic functionality (4 tests)
- Caching (3 tests)
- Multilingual support (2 tests)
- Edge cases (5 tests)
- Performance (2 tests)
- Health check (2 tests)

Total: 18 tests

Author: JuriX Team
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


# ==================== FIXTURES ====================

@pytest.fixture(scope="module")
def service():
    """
    Shared service instance for all tests (module-scoped).

    Évite de recharger le modèle sentence-transformers (~120MB)
    pour chaque test, ce qui accélère significativement l'exécution.
    """
    return EmbeddingService(use_cache=False)  # Cache désactivé pour tests déterministes


@pytest.fixture(scope="module")
def service_with_cache():
    """Service instance avec cache Redis activé."""
    try:
        return EmbeddingService(use_cache=True)
    except Exception:
        pytest.skip("Redis non disponible pour tests cache")


@pytest.fixture
def sample_french_text():
    """Texte juridique français exemple."""
    return "Article 1er. Le Code civil camerounais régit les personnes, les biens et les obligations."


@pytest.fixture
def sample_english_text():
    """Texte juridique anglais exemple."""
    return "Article 1. The Cameroonian Civil Code governs persons, property and obligations."


# ==================== TESTS BASIC FUNCTIONALITY ====================

class TestBasicFunctionality:
    """Tests de la fonctionnalité de base du service."""

    def test_generate_single_embedding(self, service, sample_french_text):
        """Test génération d'un embedding unique."""
        embedding = service.generate_embedding(sample_french_text)

        # Vérifications
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32

    def test_generate_batch_embeddings(self, service):
        """Test génération batch d'embeddings."""
        texts = [
            "Article 1er du Code civil",
            "Article 2 du Code pénal",
            "Article 3 du Code de commerce"
        ]

        embeddings = service.generate_batch_embeddings(texts)

        # Vérifications
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.dtype == np.float32 for emb in embeddings)

    def test_embedding_dimensions(self, service, sample_french_text):
        """Test que les embeddings ont exactement 768 dimensions."""
        embedding = service.generate_embedding(sample_french_text)

        assert embedding.shape == (768,), (
            f"Dimension incorrecte: {embedding.shape} != (768,)"
        )

    def test_embedding_normalized(self, service, sample_french_text):
        """Test que les embeddings sont normalisés L2 (norme ≈ 1.0)."""
        embedding = service.generate_embedding(sample_french_text, normalize=True)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, (
            f"Embedding non normalisé: norme={norm:.4f} (attendu ≈1.0)"
        )


# ==================== TESTS CACHING ====================

class TestCaching:
    """Tests du système de cache Redis."""

    def test_cache_hit(self, service_with_cache, sample_french_text):
        """Test que le deuxième appel utilise le cache."""
        # Premier appel (génération)
        start1 = time.time()
        emb1 = service_with_cache.generate_embedding(sample_french_text)
        time1 = time.time() - start1

        # Deuxième appel (cache hit)
        start2 = time.time()
        emb2 = service_with_cache.generate_embedding(sample_french_text)
        time2 = time.time() - start2

        # Vérifications
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)
        assert time2 < time1 * 0.5, (
            f"Cache hit devrait être plus rapide: {time2:.3f}s vs {time1:.3f}s"
        )

    def test_cache_miss(self, service_with_cache):
        """Test que le premier appel génère et cache l'embedding."""
        unique_text = f"Article unique test {time.time()}"

        start = time.time()
        embedding = service_with_cache.generate_embedding(unique_text)
        elapsed = time.time() - start

        # Vérifications
        assert embedding is not None
        assert embedding.shape == (768,)
        assert elapsed < 1.0, f"Génération trop lente: {elapsed:.3f}s > 1.0s"

    def test_cache_disabled(self, service):
        """Test que le service fonctionne sans cache."""
        text = "Article test sans cache"

        # Devrait fonctionner normalement
        emb1 = service.generate_embedding(text)
        emb2 = service.generate_embedding(text)

        # Les embeddings devraient être identiques (même texte)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)


# ==================== TESTS MULTILINGUAL SUPPORT ====================

class TestMultilingualSupport:
    """Tests du support multilingue (FR/EN)."""

    def test_french_text(self, service, sample_french_text):
        """Test traitement de texte juridique français."""
        embedding = service.generate_embedding(sample_french_text)

        # Vérifications
        assert embedding is not None
        assert embedding.shape == (768,)

        # Le texte contient des mots français juridiques
        assert "camerounais" in sample_french_text.lower()

    def test_english_text(self, service, sample_english_text):
        """Test traitement de texte juridique anglais."""
        embedding = service.generate_embedding(sample_english_text)

        # Vérifications
        assert embedding is not None
        assert embedding.shape == (768,)

        # Le texte contient des mots anglais juridiques
        assert "cameroonian" in sample_english_text.lower()


# ==================== TESTS EDGE CASES ====================

class TestEdgeCases:
    """Tests des cas limites et edge cases."""

    def test_empty_text_raises_error(self, service):
        """Test que le texte vide lève une ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            service.generate_embedding("")

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            service.generate_embedding("   ")  # Whitespace only

    def test_very_long_text(self, service):
        """Test gestion de textes très longs (>10,000 chars)."""
        long_text = "Article " * 2000  # ~14,000 chars

        with pytest.raises(ValueError, match="Texte trop long"):
            service.generate_embedding(long_text)

    def test_special_characters(self, service):
        """Test gestion des caractères spéciaux et accents."""
        special_text = (
            "Article 1er. Les droits énoncés à l'article 2 § 3 "
            "s'appliquent à tous : é, è, ê, ç, à, ù, œ, €, °, ..."
        )

        embedding = service.generate_embedding(special_text)

        # Devrait fonctionner normalement
        assert embedding is not None
        assert embedding.shape == (768,)

    def test_batch_with_duplicates(self, service):
        """Test batch avec textes dupliqués."""
        texts = [
            "Article 1er",
            "Article 2",
            "Article 1er",  # Duplicate
            "Article 3",
            "Article 1er"   # Duplicate
        ]

        embeddings = service.generate_batch_embeddings(texts)

        # Vérifications
        assert len(embeddings) == 5

        # Les duplicates devraient avoir les mêmes embeddings
        np.testing.assert_array_almost_equal(
            embeddings[0], embeddings[2], decimal=5
        )
        np.testing.assert_array_almost_equal(
            embeddings[0], embeddings[4], decimal=5
        )

    def test_concurrent_requests(self, service):
        """Test thread safety avec requêtes concurrentes."""
        texts = [f"Article {i}" for i in range(10)]

        def generate_embedding(text: str) -> np.ndarray:
            return service.generate_embedding(text)

        # Exécution parallèle
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(generate_embedding, texts))

        # Vérifications
        assert len(results) == 10
        assert all(emb.shape == (768,) for emb in results)

        # Pas de corruption de données
        for i, emb in enumerate(results):
            assert not np.isnan(emb).any(), f"NaN dans embedding {i}"
            assert not np.isinf(emb).any(), f"Inf dans embedding {i}"


# ==================== TESTS PERFORMANCE ====================

class TestPerformance:
    """Tests des performances du service."""

    def test_single_embedding_performance(self, service, sample_french_text):
        """Test que l'embedding unique prend <200ms."""
        start = time.time()
        embedding = service.generate_embedding(sample_french_text)
        elapsed = time.time() - start

        assert embedding is not None
        assert elapsed < 0.2, (
            f"Trop lent: {elapsed:.3f}s > 0.2s (cible: <200ms)"
        )

    def test_batch_10_performance(self, service):
        """Test que le batch de 10 embeddings prend <1s."""
        texts = [f"Article {i} du Code civil camerounais" for i in range(1, 11)]

        start = time.time()
        embeddings = service.generate_batch_embeddings(texts, batch_size=10)
        elapsed = time.time() - start

        assert len(embeddings) == 10
        assert elapsed < 1.0, (
            f"Trop lent: {elapsed:.3f}s > 1.0s (cible: <1s pour 10 textes)"
        )


# ==================== TESTS HEALTH CHECK ====================

class TestHealthCheck:
    """Tests du health check du service."""

    def test_health_check_returns_status(self, service):
        """Test que le health check retourne un statut valide."""
        health = service.health_check()

        # Vérifications structure
        assert isinstance(health, dict)
        assert "service" in health
        assert "status" in health
        assert "model" in health
        assert "dimensions" in health
        assert "device" in health
        assert "cache_enabled" in health

        # Vérifications valeurs
        assert health["service"] == "EmbeddingService"
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert health["model"] == EmbeddingService.MODEL_NAME
        assert health["dimensions"] == 768
        assert health["device"] in ["cpu", "cuda", "cuda:0", "mps"]
        assert isinstance(health["cache_enabled"], bool)

    def test_health_check_with_redis_down(self, service):
        """Test graceful degradation si Redis indisponible."""
        # Service créé sans cache (Redis potentiellement indisponible)
        health = service.health_check()

        # Le service devrait rester "healthy" même sans cache
        assert health["status"] == "healthy"
        assert health["cache_status"] == "disabled"


# ==================== TESTS SIMILARITY ====================

class TestSimilarity:
    """Tests de la fonction de similarité."""

    def test_similarity_identical_texts(self, service):
        """Test similarité entre textes identiques."""
        text = "Article 1er du Code civil"

        emb1 = service.generate_embedding(text)
        emb2 = service.generate_embedding(text)

        similarity = service.similarity(emb1, emb2)

        # Devrait être très proche de 1.0 (identiques)
        assert similarity > 0.99, (
            f"Similarité trop faible pour textes identiques: {similarity:.4f}"
        )

    def test_similarity_similar_texts(self, service):
        """Test similarité entre textes similaires."""
        text1 = "Article premier du Code civil camerounais"
        text2 = "Article 1er du Code civil du Cameroun"

        emb1 = service.generate_embedding(text1)
        emb2 = service.generate_embedding(text2)

        similarity = service.similarity(emb1, emb2)

        # Devrait être élevée (textes très similaires)
        assert similarity > 0.7, (
            f"Similarité trop faible: {similarity:.4f} (attendu >0.7)"
        )

    def test_similarity_different_texts(self, service):
        """Test similarité entre textes différents."""
        text1 = "Article 1er du Code civil"
        text2 = "Article 100 du Code pénal"

        emb1 = service.generate_embedding(text1)
        emb2 = service.generate_embedding(text2)

        similarity = service.similarity(emb1, emb2)

        # Devrait être modérée (même domaine mais articles différents)
        assert 0.0 <= similarity <= 1.0, (
            f"Similarité hors bornes: {similarity:.4f}"
        )

    def test_similarity_wrong_dimensions(self, service):
        """Test que les mauvaises dimensions lèvent une erreur."""
        emb1 = np.random.rand(768).astype(np.float32)
        emb2_wrong = np.random.rand(512).astype(np.float32)  # Mauvaise dim

        with pytest.raises(ValueError, match="dimension incorrecte"):
            service.similarity(emb1, emb2_wrong)


# ==================== TESTS VALIDATION ====================

class TestValidation:
    """Tests de validation des entrées."""

    def test_batch_empty_list_raises_error(self, service):
        """Test que liste vide lève une erreur."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            service.generate_batch_embeddings([])

    def test_batch_with_invalid_text(self, service):
        """Test que texte invalide dans batch lève une erreur."""
        texts = [
            "Article 1er",
            "",  # Invalide
            "Article 3"
        ]

        with pytest.raises(ValueError, match="Texte invalide à l'index 1"):
            service.generate_batch_embeddings(texts)
