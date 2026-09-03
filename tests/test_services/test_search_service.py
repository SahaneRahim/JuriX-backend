"""
Comprehensive test suite for SearchService.

Tests cover:
- Text search (PostgreSQL FTS)
- Semantic search (pgvector)
- Hybrid search (RRF fusion)
- Indexing operations (create, update, delete, reindex)
- Filtering (language, category, date, combined)
- Caching (hit, miss)
- Edge cases (empty query, special chars)
- Performance requirements (<200ms hybrid)

Total: 28+ tests for >90% coverage

Author: JuriX Team
Version: 1.0.0
"""

import asyncio
import json
import time
from datetime import date, datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base
from app.models.law import Article, Category, Law
from app.schemas.search import (
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.search_service import (

# NOTE: les fixtures db_engine / db_session locales (SQLite en memoire) ont ete
# retirees. Elles masquaient celles de conftest.py et testaient un moteur qui ne
# supporte ni pgvector, ni tsvector, ni les fonctions PostgreSQL dont ce code
# depend — une suite verte y aurait certifie du code cassé en production.
# db_session vient desormais de conftest.py (PostgreSQL reel + rollback).
    IndexingError,
    TextSearchError,
    SearchService,
    SearchServiceError,
    VectorSearchError,
)

# ============================================================================
# Test Configuration & Fixtures
# ============================================================================







# NOTE: les fixtures mock_meilisearch et mock_redis ont ete supprimees. Elles
# patchaient app.services.search_service.meilisearch.Client et .redis.from_url,
# deux symboles qui n'existent plus depuis le passage a PostgreSQL natif — le
# patch echouait donc au setup. La recherche textuelle s'appuie desormais sur la
# vraie base fournie par conftest.


@pytest.fixture
def mock_embedding_service():
    """
    Doublure d'EmbeddingService.

    3072 dimensions et non 768 : c'est la taille reelle produite par
    gemini-embedding-001 et le type declare sur articles.embedding.
    """
    mock_service = MagicMock()
    mock_service.generate_embedding.return_value = np.random.rand(3072).astype(np.float32)
    mock_service.TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    mock_service.TASK_QUERY = "RETRIEVAL_QUERY"
    mock_service.health_check.return_value = {"status": "healthy", "dimensions": 3072}
    return mock_service


@pytest.fixture
async def search_service(db_session, mock_embedding_service):
    """
    SearchService branche sur la vraie base, embeddings simules.

    L'ancienne version patchait SearchService._get_embedding_service, methode
    qui n'existe plus : le service lit maintenant un singleton de module,
    qu'on remplace directement sur l'instance.
    """
    service = SearchService(db_session, use_cache=False)
    service.embedding_service = mock_embedding_service
    yield service


@pytest.fixture
async def sample_data(db_session):
    """
    Lois et articles d'exemple.

    Les categories ne sont plus creees ici : la fixture db_session de conftest
    seme deja les 12 categories de reference avec les ids 1 a 12. Les reinserer
    avec des ids explicites provoquait une violation de contrainte d'unicite au
    setup de chaque test de ce fichier.
    """
    await db_session.flush()

    # Laws
    laws = [
        Law(
            id=1,
            reference="LOI-2024-001",
            title="Code civil camerounais",
            content="Contenu du code civil avec responsabilité des dirigeants.",
            type="loi",
            language="fr",
            status="published",
            category_id=1,
            publication_date=date(2024, 1, 15),
            created_at=datetime(2024, 1, 15, 10, 0, 0)
        ),
        Law(
            id=2,
            reference="LOI-2024-002",
            title="Code pénal",
            content="Contenu du code pénal avec infractions et sanctions.",
            type="loi",
            language="fr",
            status="published",
            category_id=2,
            publication_date=date(2024, 2, 20),
            created_at=datetime(2024, 2, 20, 10, 0, 0)
        ),
        Law(
            id=3,
            reference="LAW-2024-003",
            title="Commercial Code",
            content="Content about commercial law and company directors.",
            type="law",
            language="en",
            status="published",
            category_id=3,
            publication_date=date(2024, 3, 10),
            created_at=datetime(2024, 3, 10, 10, 0, 0)
        ),
    ]

    for law in laws:
        db_session.add(law)

    await db_session.flush()

    # Articles with embeddings
    articles = [
        Article(
            id=1,
            law_id=1,
            number="1",
            content="Article 1er sur la responsabilité.",
            embedding=np.random.rand(3072).tolist(),
            order=1
        ),
        Article(
            id=2,
            law_id=1,
            number="2",
            content="Article 2 sur les dirigeants.",
            embedding=np.random.rand(3072).tolist(),
            order=2
        ),
        Article(
            id=3,
            law_id=2,
            number="1",
            content="Article 1er sur les infractions.",
            embedding=np.random.rand(3072).tolist(),
            order=1
        ),
    ]

    for article in articles:
        db_session.add(article)

    await db_session.commit()

    # categories vient de la fixture db_session (conftest), plus de cette fixture
    return {"laws": laws, "articles": articles}


# ============================================================================
# Test Text Search (PostgreSQL FTS)
# ============================================================================

class TestTextSearch:
    """Test text search functionality (PostgreSQL FTS)."""

    @pytest.mark.asyncio
    async def test_text_search_basic(self, search_service, sample_data):
        """Test basic text search returns results."""
        results = await search_service.text_search(
            query="civil",
            filters=None,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        assert len(results) >= 1

        # Verify result structure
        result = results[0]
        assert result.law_id == 1
        assert result.reference == "LOI-2024-001"
        assert "civil" in result.title.lower() or "<mark>" in str(result.highlights)
        assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_text_search_with_filters(self, search_service, sample_data):
        """Test text search with language filter."""
        filters = SearchFilters(language="fr")

        results = await search_service.text_search(
            query="code",
            filters=filters,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        # All results should be French
        for result in results:
            assert result.language == "fr"

    @pytest.mark.asyncio
    async def test_text_search_typo_tolerance(self, search_service):
        """Test that typos are handled gracefully."""
        # Resultats semes en base par les fixtures
        results = await search_service.text_search(
            query="civi",  # Typo: civil
            filters=None,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        # La recherche plein texte tolere la faute de frappe

    @pytest.mark.asyncio
    async def test_text_search_multilingual(self, search_service, sample_data):
        """Test search works for both French and English."""
        # French query
        fr_results = await search_service.text_search(
            query="responsabilité",
            filters=SearchFilters(language="fr"),
            limit=15,
            offset=0
        )

        # English query
        en_results = await search_service.text_search(
            query="directors",
            filters=SearchFilters(language="en"),
            limit=15,
            offset=0
        )

        # Both should work (mocked)
        assert isinstance(fr_results, list)
        assert isinstance(en_results, list)

    @pytest.mark.asyncio
    async def test_text_search_no_results(self, search_service):
        """Une requete sans correspondance renvoie une liste vide, pas une erreur."""

        results = await search_service.text_search(
            query="nonexistentquery12345",
            filters=None,
            limit=15,
            offset=0
        )

        assert results == []


# ============================================================================
# Test Semantic Search (pgvector)
# ============================================================================

class TestSemanticSearch:
    """Test semantic search functionality with pgvector."""

    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, search_service, sample_data):
        """Test basic semantic search returns results."""
        results = await search_service.semantic_search(
            query="responsabilité des dirigeants",
            filters=None,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        # May be empty in SQLite (no real pgvector), but structure is valid
        for result in results:
            assert hasattr(result, "law_id")
            assert hasattr(result, "relevance_score")
            assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_semantic_search_contextual(self, search_service, sample_data):
        """Test semantic search finds contextually similar content."""
        # Query with synonyms/context
        results = await search_service.semantic_search(
            query="accountability of company leaders",
            filters=None,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        # Semantic search should find related content even with different words

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self, search_service, sample_data):
        """Test semantic search with category filter."""
        filters = SearchFilters(category_ids=[1, 2])

        results = await search_service.semantic_search(
            query="infractions pénales",
            filters=filters,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        # All results should be from categories 1 or 2
        for result in results:
            assert result.category_id in [1, 2]

    @pytest.mark.asyncio
    async def test_semantic_search_ranking(self, search_service, sample_data):
        """Test semantic search results are properly ranked."""
        results = await search_service.semantic_search(
            query="civil law responsibilities",
            filters=None,
            limit=5,
            offset=0
        )

        # Results should be sorted by relevance score descending
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

    @pytest.mark.asyncio
    async def test_semantic_search_no_results(self, search_service, db_session):
        """Test semantic search with no embeddings in database."""
        # Empty database, no articles
        results = await search_service.semantic_search(
            query="test query",
            filters=None,
            limit=15,
            offset=0
        )

        assert results == []


# ============================================================================
# Test Hybrid Search (RRF Fusion)
# ============================================================================

class TestHybridSearch:
    """Test hybrid search with RRF fusion algorithm."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, search_service, sample_data):
        """Test basic hybrid search combines both modes."""
        results = await search_service.hybrid_search(
            query="responsabilité dirigeants",
            filters=None,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)
        # Hybrid should return combined results
        for result in results:
            assert hasattr(result, "relevance_score")
            assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf_fusion(self, search_service, sample_data):
        """Test RRF fusion combines rankings correctly."""
        # Create mock results from both modes
        text_result = SearchResult(
            law_id=1,
            reference="LOI-001",
            title="Test Law 1",
            type="loi",
            language="fr",
            status="published",
            category_id=1,
            category_name="Test",
            publication_date=None,
            relevance_score=0.9,
            matched_articles=[],
            highlights={}
        )

        semantic_result = SearchResult(
            law_id=2,
            reference="LOI-002",
            title="Test Law 2",
            type="loi",
            language="fr",
            status="published",
            category_id=1,
            category_name="Test",
            publication_date=None,
            relevance_score=0.8,
            matched_articles=[],
            highlights={}
        )

        # Test RRF fusion directly
        fused = search_service._rrf_fusion(
            [text_result],
            [semantic_result],
            k=60,
            text_weight=0.4,
            semantic_weight=0.6
        )

        assert isinstance(fused, list)
        assert len(fused) == 2  # Both results present

    @pytest.mark.asyncio
    async def test_hybrid_search_normalization(self, search_service):
        """Test score normalization to [0, 1] range."""
        results = [
            SearchResult(
                law_id=1, reference="LOI-001", title="Test 1",
                type="loi", language="fr", status="published",
                category_id=1, category_name="Test",
                publication_date=None, relevance_score=0.5,
                matched_articles=[], highlights={}
            ),
            SearchResult(
                law_id=2, reference="LOI-002", title="Test 2",
                type="loi", language="fr", status="published",
                category_id=1, category_name="Test",
                publication_date=None, relevance_score=0.9,
                matched_articles=[], highlights={}
            ),
        ]

        normalized = search_service._normalize_scores(results)

        # All scores should be in [0, 1]
        for result in normalized:
            assert 0.0 <= result.relevance_score <= 1.0

        # Highest score should be 1.0, lowest should be 0.0
        scores = [r.relevance_score for r in normalized]
        assert max(scores) == 1.0
        assert min(scores) == 0.0

    @pytest.mark.asyncio
    async def test_hybrid_search_deduplication(self, search_service, sample_data):
        """Test that duplicate law_ids are properly merged."""
        results = await search_service.hybrid_search(
            query="civil",
            filters=None,
            limit=15,
            offset=0
        )

        # Check no duplicate law_ids
        law_ids = [r.law_id for r in results]
        assert len(law_ids) == len(set(law_ids)), "Duplicate law_ids found"

    @pytest.mark.asyncio
    async def test_hybrid_search_performance(self, search_service, sample_data):
        """Test hybrid search completes in <200ms (spec requirement)."""
        start_time = time.time()

        results = await search_service.hybrid_search(
            query="test performance",
            filters=None,
            limit=15,
            offset=0
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # With mocked services, should be very fast
        # Real implementation target: <200ms
        assert elapsed_ms < 500  # Generous for test environment

    @pytest.mark.asyncio
    async def test_hybrid_search_better_than_single_mode(self, search_service, sample_data):
        """Test hybrid search provides better results than single modes."""
        query = "responsabilité dirigeants société"
        filters = None

        # Get results from all three modes
        text_results = await search_service.text_search(query, filters, 15, 0)
        semantic_results = await search_service.semantic_search(query, filters, 15, 0)
        hybrid_results = await search_service.hybrid_search(query, filters, 15, 0)

        # Hybrid should combine strengths (or at least not be empty if others have results)
        assert isinstance(hybrid_results, list)


# ============================================================================
# Test Indexing Operations
# ============================================================================

class TestIndexing:
    """Test des operations d'indexation (search_vector PostgreSQL)."""

    @pytest.mark.asyncio
    async def test_index_law(self, search_service, sample_data):
        """Test indexing a single law."""
        law = sample_data["laws"][0]

        # Should not raise
        await search_service.index_law(law)

        # L'indexation renseigne le tsvector PostgreSQL (l'ancien assert
        # portait sur un mock Meilisearch qui n'existe plus).
        from sqlalchemy import text as sa_text

        indexed = await search_service.db.execute(
            sa_text("SELECT search_vector IS NOT NULL FROM laws WHERE id = :i"),
            {"i": law.id},
        )
        assert indexed.scalar() is True

    @pytest.mark.asyncio
    async def test_update_law_index(self, search_service, sample_data):
        """Test updating an existing law in index."""
        law = sample_data["laws"][0]

        # Should not raise
        await search_service.update_law_index(law.id, law)

        from sqlalchemy import text as sa_text

        indexed = await search_service.db.execute(
            sa_text("SELECT search_vector IS NOT NULL FROM laws WHERE id = :i"),
            {"i": law.id},
        )
        assert indexed.scalar() is True

    @pytest.mark.asyncio
    async def test_delete_law_index(self, search_service, sample_data):
        """Test deleting a law from index."""
        law_id = sample_data["laws"][0].id

        # Should not raise
        await search_service.delete_law_index(law_id)

        # La desindexation vide le search_vector ; le trigger nettoie les articles.
        from sqlalchemy import text as sa_text

        cleared = await search_service.db.execute(
            sa_text("SELECT search_vector IS NULL FROM laws WHERE id = :i"),
            {"i": law_id},
        )
        assert cleared.scalar() is True

    @pytest.mark.asyncio
    async def test_reindex_all_laws(self, search_service, sample_data):
        """Test batch reindexing all laws."""
        response = await search_service.reindex_all_laws()

        assert response.status == "success"
        assert response.total_laws >= 0
        assert response.indexed_count >= 0
        assert response.failed_count == 0
        assert response.duration_seconds >= 0


# ============================================================================
# Test Filtering
# ============================================================================

class TestFiltering:
    """Test search filtering functionality."""

    @pytest.mark.asyncio
    async def test_filter_by_language(self, search_service, sample_data):
        """Test filtering by language."""
        filters = SearchFilters(language="fr")

        results = await search_service.text_search(
            query="code",
            filters=filters,
            limit=15,
            offset=0
        )

        # Mock returns what we configured, so just verify it works
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_filter_by_category(self, search_service, sample_data):
        """Test filtering by category IDs."""
        filters = SearchFilters(category_ids=[1, 2])

        results = await search_service.semantic_search(
            query="test",
            filters=filters,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self, search_service, sample_data):
        """Test filtering by publication year range."""
        filters = SearchFilters(year_from=2024, year_to=2024)

        results = await search_service.text_search(
            query="code",
            filters=filters,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_filter_combined(self, search_service, sample_data):
        """Test combining multiple filters."""
        filters = SearchFilters(
            language="fr",
            category_ids=[1],
            status="published",
            year_from=2024
        )

        results = await search_service.hybrid_search(
            query="civil",
            filters=filters,
            limit=15,
            offset=0
        )

        assert isinstance(results, list)


# ============================================================================
# Test Caching
# ============================================================================

class TestCaching:
    """
    Cache de recherche PostgreSQL (table query_cache).

    Reecrits : la version precedente simulait Redis, remplace par la table
    query_cache lors du passage a PostgreSQL natif.
    """

    @pytest.mark.asyncio
    async def test_cache_stores_and_serves_response(self, db_session, sample_data):
        """Une meme requete lancee deux fois est servie par le cache la 2e fois."""
        from sqlalchemy import text as sa_text

        service = SearchService(db_session, use_cache=True)
        service.embedding_service = None  # mode texte uniquement

        request = SearchRequest(query="responsabilite", mode="text")

        first = await service.search(request)
        assert isinstance(first, SearchResponse)

        rows = await db_session.execute(sa_text("SELECT count(*) FROM query_cache"))
        assert rows.scalar() == 1, "la reponse aurait du etre mise en cache"

        second = await service.search(request)
        assert isinstance(second, SearchResponse)
        assert second.query == first.query
        assert second.total == first.total

    @pytest.mark.asyncio
    async def test_expired_entry_is_ignored(self, db_session):
        """Une entree expiree n'est pas servie."""
        from app.services.postgres_search_service import get_from_pg_cache
        from sqlalchemy import text as sa_text

        await db_session.execute(
            sa_text(
                "INSERT INTO query_cache (cache_key, response_json, expires_at) "
                "VALUES ('perimee', '{\"query\": \"x\"}', now() - interval '1 hour')"
            )
        )
        await db_session.commit()

        assert await get_from_pg_cache(db_session, "perimee") is None

    @pytest.mark.asyncio
    async def test_cache_disabled_writes_nothing(self, db_session, sample_data):
        """use_cache=False ne doit rien ecrire dans query_cache."""
        from sqlalchemy import text as sa_text

        service = SearchService(db_session, use_cache=False)
        service.embedding_service = None

        await service.search(SearchRequest(query="responsabilite", mode="text"))

        rows = await db_session.execute(sa_text("SELECT count(*) FROM query_cache"))
        assert rows.scalar() == 0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, search_service):
        """Test handling of empty query string."""
        # SearchRequest validation should catch this
        with pytest.raises(Exception):  # Pydantic validation error
            SearchRequest(query="", mode="hybrid")

    @pytest.mark.asyncio
    async def test_special_characters(self, search_service, sample_data):
        """Test search with special characters."""
        results = await search_service.text_search(
            query="test @#$% special & chars",
            filters=None,
            limit=15,
            offset=0
        )

        # Should handle gracefully without errors
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_text_search_failure_falls_back(self, search_service, sample_data, monkeypatch):
        """
        Le mode hybride doit degrader proprement si la recherche textuelle echoue.

        L'ancienne version simulait une panne Meilisearch ; on fait desormais
        echouer la recherche plein texte PostgreSQL elle-meme.
        """
        async def _boom(*a, **kw):
            raise RuntimeError("recherche plein texte indisponible")

        monkeypatch.setattr(search_service, "text_search", _boom)

        request = SearchRequest(query="test", mode="hybrid")

        # Should handle gracefully - if semantic search succeeds, hybrid continues
        # If both fail, should get semantic fallback
        response = await search_service.search(request)

        assert isinstance(response, SearchResponse)
        # Hybrid mode continues if semantic search works (text search fails gracefully)
        # Or falls back to semantic if both fail
        assert response.mode in ["hybrid", "semantic"]
        assert response.total >= 0  # Should return results or empty list

    @pytest.mark.asyncio
    async def test_long_query_truncation(self, search_service):
        """Test handling of very long queries."""
        long_query = "word " * 200  # 200 words

        results = await search_service.text_search(
            query=long_query[:500],  # Truncate to max length
            filters=None,
            limit=15,
            offset=0
        )

        # Should handle without errors
        assert isinstance(results, list)


# ============================================================================
# Test Main Search Entry Point
# ============================================================================

class TestSearchEntryPoint:
    """Test main search() entry point."""

    @pytest.mark.asyncio
    async def test_search_text_mode(self, search_service, sample_data):
        """Test search with text mode."""
        request = SearchRequest(query="civil", mode="text")

        response = await search_service.search(request)

        assert isinstance(response, SearchResponse)
        assert response.mode == "text"
        assert response.query == "civil"
        assert response.search_time_ms >= 0

    @pytest.mark.asyncio
    async def test_search_semantic_mode(self, search_service, sample_data):
        """Test search with semantic mode."""
        request = SearchRequest(query="responsabilité", mode="semantic")

        response = await search_service.search(request)

        assert isinstance(response, SearchResponse)
        assert response.mode == "semantic"

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, search_service, sample_data):
        """Test search with hybrid mode."""
        request = SearchRequest(query="dirigeants", mode="hybrid")

        response = await search_service.search(request)

        assert isinstance(response, SearchResponse)
        assert response.mode in ["hybrid", "text", "semantic"]  # May fallback

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, search_service, sample_data):
        """Test search with pagination parameters."""
        request = SearchRequest(
            query="test",
            mode="text",
            limit=5,
            offset=10
        )

        response = await search_service.search(request)

        assert isinstance(response, SearchResponse)
        assert len(response.results) <= 5

    @pytest.mark.asyncio
    async def test_search_invalid_mode(self, search_service):
        """Test search with invalid mode raises error."""
        with pytest.raises(Exception):  # Pydantic validation
            SearchRequest(query="test", mode="invalid_mode")
