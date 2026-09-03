"""
Tests de la dimension, de la normalisation et de la cle de cache des embeddings.

Chaque test correspond a un mecanisme qui n'existait pas :

- output_dimensionality n'etait pas demande a l'API ;
- la renormalisation des sorties tronquees (Matryoshka) n'existait pas, alors
  que l'operateur <=> de pgvector suppose des vecteurs unitaires ;
- le chemin batch n'avait ni garde de dimension ni controle du nombre de
  vecteurs rendus ;
- la cle de cache ignorait la dimension (un vecteur 3072 aurait ete resservi
  sous configuration 1536) et minusculait le texte.
"""

import json

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class _Emb:
    def __init__(self, values):
        self.values = values


class _Result:
    def __init__(self, embeddings):
        self.embeddings = embeddings


@pytest.fixture
def recorder(monkeypatch):
    """Doublure Gemini qui enregistre la configuration recue."""
    state = {"configs": [], "n_returned": None, "dim": None, "scale": 9.0}

    class _Models:
        def embed_content(self, model=None, contents=None, config=None):
            state["configs"].append(config)
            items = contents if isinstance(contents, list) else [contents]
            dim = state["dim"] or getattr(config, "output_dimensionality", None)
            n = state["n_returned"] if state["n_returned"] is not None else len(items)
            rng = np.random.default_rng(7)
            # Vecteurs NON unitaires : c'est le service qui doit normaliser.
            return _Result([
                _Emb((rng.random(dim) * state["scale"]).tolist())
                for _ in range(n)
            ])

    class _Client:
        def __init__(self, *a, **k):
            self.models = _Models()

    monkeypatch.setattr("app.services.embedding_service.genai.Client", _Client)
    return state


@pytest.fixture
def service(recorder):
    return EmbeddingService(use_cache=False)


class TestDimension:

    def test_output_dimensionality_is_requested(self, service, recorder):
        service.generate_embedding("Article premier du code civil")

        config = recorder["configs"][-1]
        assert config.output_dimensionality == EmbeddingService.EMBEDDING_DIM
        assert EmbeddingService.EMBEDDING_DIM == 1536

    def test_embedding_has_configured_dimension(self, service):
        embedding = service.generate_embedding("Article premier")

        assert embedding.shape == (EmbeddingService.EMBEDDING_DIM,)

    def test_batch_rejects_wrong_dimension(self, service, recorder):
        recorder["dim"] = 768  # l'API renvoie autre chose que ce qui est demande

        with pytest.raises(EmbeddingServiceError, match="Dimension"):
            service.generate_batch_embeddings(["a", "b"])

    def test_batch_rejects_truncated_response(self, service, recorder):
        """
        Deux vecteurs pour trois textes : zip() perdait silencieusement la
        queue du lot, puis un KeyError tombait bien plus loin.
        """
        recorder["n_returned"] = 2

        with pytest.raises(EmbeddingServiceError, match="incomplète"):
            service.generate_batch_embeddings(["a", "b", "c"])


class TestNormalisation:

    def test_embedding_is_unit_norm(self, service):
        embedding = service.generate_embedding("Responsabilité des dirigeants")

        assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5

    def test_batch_embeddings_are_unit_norm(self, service):
        embeddings = service.generate_batch_embeddings(["un", "deux", "trois"])

        for embedding in embeddings:
            assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5

    def test_normalisation_applies_even_when_not_requested(self, service):
        """
        normalize=False ne doit PAS produire un vecteur non unitaire sous la
        dimension native : l'API ne normalise que la sortie pleine, et tout le
        calcul de distance en aval suppose une norme de 1.
        """
        embedding = service.generate_embedding("Test", normalize=False)

        assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5


class TestCacheKey:

    def test_key_isolates_dimension(self, service):
        key_1536 = service._cache_key("texte", EmbeddingService.TASK_DOCUMENT)
        original = EmbeddingService.EMBEDDING_DIM
        try:
            service.EMBEDDING_DIM = 3072
            key_3072 = service._cache_key("texte", EmbeddingService.TASK_DOCUMENT)
        finally:
            service.EMBEDDING_DIM = original

        assert key_1536 != key_3072

    def test_key_isolates_task_type(self, service):
        as_document = service._cache_key("texte", EmbeddingService.TASK_DOCUMENT)
        as_query = service._cache_key("texte", EmbeddingService.TASK_QUERY)

        assert as_document != as_query

    def test_key_is_case_sensitive(self, service):
        """
        La cle minusculait le texte : "Article PREMIER" et "article premier"
        partageaient une entree alors que Gemini les encode differemment.
        """
        upper = service._cache_key("Article PREMIER", EmbeddingService.TASK_DOCUMENT)
        lower = service._cache_key("article premier", EmbeddingService.TASK_DOCUMENT)

        assert upper != lower


class TestCacheRead:

    @pytest.mark.asyncio
    async def test_cache_read_rejects_wrong_dimension(self, recorder, db_session):
        """
        Une entree ecrite sous l'ancienne dimension doit etre ignoree, pas
        servie : elle serait rejetee par la colonne vector(1536), ou pire,
        comparee de travers.
        """
        from sqlalchemy import text as sql_text

        service = EmbeddingService(use_cache=True)
        key = service._cache_key("texte périmé", EmbeddingService.TASK_DOCUMENT)

        await db_session.execute(
            sql_text(
                "INSERT INTO embedding_cache (text_hash, embedding_json, expires_at) "
                "VALUES (:key, :data, now() + interval '1 day')"
            ),
            {"key": key, "data": json.dumps([0.1] * 3072)},
        )
        await db_session.commit()

        assert service._get_from_pg_cache(key) is None
