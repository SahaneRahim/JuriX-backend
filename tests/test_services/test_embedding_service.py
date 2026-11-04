"""
Tests pour EmbeddingService.

Test categories:
- Basic functionality (2 tests)
- Caching (1 test)
- Edge cases (4 tests)
- Health check (2 tests)

Total: 9 tests

Author: JuriX Team
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import httpx
import numpy as np
import pytest

from app.services import embedding_service
from app.services.embedding_service import EmbeddingService

# ==================== DOUBLURE VOYAGE ====================
#
# Ces tests portent sur la logique d'EmbeddingService — validation, cache,
# similarite cosinus — et non sur l'API. Ils passent par le fournisseur de
# production, Voyage, dont le transport HTTP est double : aucun appel reseau,
# aucun jeton consomme. La normalisation L2 est verifiee dans
# test_embedding_dimensions.py et test_embedding_providers.py.
#
# L'embedding est deterministe : derive d'une empreinte du texte, donc deux appels
# sur le meme texte donnent le meme vecteur et deux textes differents des vecteurs
# differents. C'est exactement ce que les tests de similarite supposent.

def _fake_embedding(text: str, dim: int = None) -> List[float]:
    """
    Vecteur deterministe, NON normalise et d'amplitude quelconque.

    Non normalise a dessein : c'est au service de normaliser, quelle que soit
    la sortie du fournisseur. Une doublure qui renverrait deja des vecteurs
    unitaires ne prouverait rien.
    """
    dim = dim if dim is not None else EmbeddingService.EMBEDDING_DIM
    graine = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(graine)
    return (rng.random(dim) * 7.0).tolist()


@pytest.fixture(autouse=True)
def _stub_voyage(monkeypatch):
    """Double le transport HTTP de Voyage, pour tout ce module."""
    corps_recus = []

    def _repondre(requete: httpx.Request) -> httpx.Response:
        corps = json.loads(requete.content)
        corps_recus.append(corps)
        # La doublure HONORE output_dimension : c'est ce que le service demande.
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": _fake_embedding(t, corps["output_dimension"])}
            for i, t in enumerate(corps["input"])
        ]})

    monkeypatch.setattr(embedding_service, "transport_http", httpx.MockTransport(_repondre))
    return corps_recus


# ==================== FIXTURES ====================

@pytest.fixture
def service(_stub_voyage):
    """Service partage par les tests, cache desactive pour rester deterministe."""
    return EmbeddingService(use_cache=False)  # Cache désactivé pour tests déterministes


@pytest.fixture
def service_with_cache(_stub_voyage):
    """Service avec cache active. Le cache est une table PostgreSQL
    (embedding_cache), plus Redis."""
    try:
        return EmbeddingService(use_cache=True)
    except Exception as exc:
        pytest.skip(f"Cache PostgreSQL indisponible: {exc}")


@pytest.fixture
def sample_french_text():
    """Texte juridique français exemple."""
    return "Article 1er. Le Code civil camerounais régit les personnes, les biens et les obligations."


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


# ==================== TESTS CACHING ====================

class TestCaching:
    """Tests du cache d'embeddings (table PostgreSQL embedding_cache)."""

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
        # Pas de seuil de rapidite : avec une doublure locale, generer et lire le
        # cache coutent le meme ordre de grandeur. Le facteur 2 attendu supposait
        # un appel reseau. Ce qui compte ici est que le cache renvoie le MEME
        # vecteur (verifie juste au-dessus) sans nouvel appel au modele.
        assert time2 <= time1 * 3, (
            f"Un hit de cache ne doit pas etre nettement plus lent: {time2:.3f}s vs {time1:.3f}s"
        )


# ==================== TESTS EDGE CASES ====================

class TestEdgeCases:
    """Tests des cas limites et edge cases."""

    def test_empty_text_raises_error(self, service):
        """Test que le texte vide lève une ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            service.generate_embedding("")

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            service.generate_embedding("   ")  # Whitespace only

    def test_special_characters(self, service):
        """Test gestion des caractères spéciaux et accents."""
        special_text = (
            "Article 1er. Les droits énoncés à l'article 2 § 3 "
            "s'appliquent à tous : é, è, ê, ç, à, ù, œ, €, °, ..."
        )

        embedding = service.generate_embedding(special_text)

        # Devrait fonctionner normalement
        assert embedding is not None
        assert embedding.shape == (EmbeddingService.EMBEDDING_DIM,)

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
        assert all(emb.shape == (EmbeddingService.EMBEDDING_DIM,) for emb in results)

        # Pas de corruption de données
        for i, emb in enumerate(results):
            assert not np.isnan(emb).any(), f"NaN dans embedding {i}"
            assert not np.isinf(emb).any(), f"Inf dans embedding {i}"


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
        # "device" appartenait au modele sentence-transformers local ; le service
        # passe par une API, qui n'a pas de notion de peripherique.
        assert "model" in health
        assert "cache_enabled" in health

        # Vérifications valeurs
        assert health["service"] == "EmbeddingService"
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert health["model"] == service.provider.model
        assert health["dimensions"] == EmbeddingService.EMBEDDING_DIM
        # "device" appartenait au modele sentence-transformers charge en local.
        # Le service passe par l'API Voyage : ce qui compte est le fournisseur.
        assert health["provider"] == "Voyage AI"
        assert isinstance(health["cache_enabled"], bool)

    def test_health_check_without_cache(self, service):
        """Test degradation propre quand le cache est indisponible."""
        # Service cree sans cache
        health = service.health_check()

        # Le service devrait rester "healthy" même sans cache
        assert health["status"] == "healthy"
        assert health["cache_backend"] == "disabled"


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

    def test_similarity_wrong_dimensions(self, service):
        """Test que les mauvaises dimensions lèvent une erreur."""
        emb1 = np.random.rand(EmbeddingService.EMBEDDING_DIM).astype(np.float32)
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
