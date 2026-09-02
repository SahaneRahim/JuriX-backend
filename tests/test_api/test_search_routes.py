"""
API integration tests for search routes.

Tests cover:
- POST /api/v1/search - Main search endpoint
- POST /api/v1/search/reindex - Batch reindex (admin)
- GET /api/v1/search/stats - Statistics (admin)
- GET /api/v1/search/health - Health check

Total: 15+ API tests

Author: JuriX Team
Version: 1.0.0
"""

import json
from datetime import date, datetime
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base
from app.main import app
from app.models.law import Article, Category, Law

# NOTE: les fixtures db_engine / db_session locales (SQLite en memoire) ont ete
# retirees. Elles masquaient celles de conftest.py et testaient un moteur qui ne
# supporte ni pgvector, ni tsvector, ni les fonctions PostgreSQL dont ce code
# depend — une suite verte y aurait certifie du code cassé en production.
# db_session vient desormais de conftest.py (PostgreSQL reel + rollback).

# ============================================================================
# Test Configuration & Fixtures
# ============================================================================







@pytest.fixture
async def sample_laws(db_session):
    """Create sample data for testing."""
    # Categories
    categories = [
        Category(id=1, name="Droit Civil", description="Droit civil"),
        Category(id=2, name="Droit Pénal", description="Droit pénal"),
    ]

    for cat in categories:
        db_session.add(cat)

    await db_session.flush()

    # Laws
    laws = [
        Law(
            id=1,
            reference="LOI-2024-001",
            title="Code civil camerounais",
            content="Contenu sur la responsabilité des dirigeants.",
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
            content="Contenu sur les infractions.",
            type="loi",
            language="fr",
            status="published",
            category_id=2,
            publication_date=date(2024, 2, 20),
            created_at=datetime(2024, 2, 20, 10, 0, 0)
        ),
    ]

    for law in laws:
        db_session.add(law)

    await db_session.flush()

    # Articles
    articles = [
        Article(
            id=1,
            law_id=1,
            number="1",
            content="Article 1er.",
            embedding=json.dumps(np.random.rand(768).tolist()),
            order=1
        ),
    ]

    for article in articles:
        db_session.add(article)

    await db_session.commit()

    return laws


@pytest.fixture
def mock_search_service():
    """Mock SearchService for API tests."""
    with patch("app.api.routes.search.get_search_service") as mock_get_service:
        mock_service = MagicMock()

        # Mock search response
        from app.schemas.search import SearchResponse, SearchResult

        mock_service.search.return_value = SearchResponse(
            query="test",
            mode="hybrid",
            results=[
                SearchResult(
                    law_id=1,
                    reference="LOI-2024-001",
                    title="Code civil",
                    type="loi",
                    language="fr",
                    status="published",
                    category_id=1,
                    category_name="Droit Civil",
                    publication_date=date(2024, 1, 15),
                    relevance_score=0.95,
                    matched_articles=[],
                    highlights={"title": "Code <mark>civil</mark>"}
                )
            ],
            total=1,
            search_time_ms=150,
            filters_applied=None
        )

        # Mock reindex response
        from app.schemas.search import ReindexResponse

        mock_service.reindex_all_laws.return_value = ReindexResponse(
            status="success",
            total_laws=10,
            indexed_count=10,
            failed_count=0,
            duration_seconds=3
        )

        # Mock stats response
        from app.schemas.search import SearchStats

        mock_service.meilisearch_client.index.return_value.get_stats.return_value = {
            "numberOfDocuments": 10
        }

        mock_service.redis_client = MagicMock()
        mock_service.embedding_service.health_check.return_value = {"status": "healthy"}

        mock_get_service.return_value = mock_service

        yield mock_service


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ============================================================================
# Test POST /api/v1/search
# ============================================================================

class TestSearchEndpoint:
    """Test main search endpoint."""

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, client, mock_search_service):
        """Test search with hybrid mode."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "responsabilité dirigeants",
                "mode": "hybrid",
                "limit": 15,
                "offset": 0
            }
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["query"] == "test"  # From mock
        assert data["mode"] == "hybrid"
        assert isinstance(data["results"], list)
        assert data["total"] >= 0
        assert data["search_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_search_text_mode(self, client, mock_search_service):
        """Test search with text mode."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "code civil",
                "mode": "text",
                "limit": 10
            }
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_search_semantic_mode(self, client, mock_search_service):
        """Test search with semantic mode."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "accountability of leaders",
                "mode": "semantic",
                "limit": 10
            }
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_search_with_filters(self, client, mock_search_service):
        """Test search with filters."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "mode": "hybrid",
                "filters": {
                    "language": "fr",
                    "category_ids": [1, 2],
                    "status": "published"
                },
                "limit": 15
            }
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "filters_applied" in data

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, client, mock_search_service):
        """Test search with pagination parameters."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "mode": "text",
                "limit": 5,
                "offset": 10
            }
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) <= 5

    @pytest.mark.asyncio
    async def test_search_invalid_mode(self, client, mock_search_service):
        """Test search with invalid mode returns 400."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "mode": "invalid_mode"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_empty_query(self, client, mock_search_service):
        """Test search with empty query returns 422."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "",
                "mode": "hybrid"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_query_too_long(self, client, mock_search_service):
        """Test search with too long query returns 422."""
        long_query = "a" * 501  # Max is 500

        response = await client.post(
            "/api/v1/search/",
            json={
                "query": long_query,
                "mode": "hybrid"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_invalid_limit(self, client, mock_search_service):
        """Test search with invalid limit returns 422."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "mode": "hybrid",
                "limit": 100  # Max is 50
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_service_error(self, client, mock_search_service):
        """Test search service error returns 500."""
        # Configure mock to raise exception
        from app.services.search_service import SearchServiceError
        mock_search_service.search.side_effect = SearchServiceError("Test error")

        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "mode": "hybrid"
            }
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data


# ============================================================================
# Test POST /api/v1/search/reindex
# ============================================================================

class TestReindexEndpoint:
    """Test batch reindex endpoint (admin)."""

    @pytest.mark.asyncio
    async def test_reindex_success(self, client, mock_search_service):
        """Test successful reindex operation."""
        response = await client.post("/api/v1/search/reindex")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "success"
        assert "total_laws" in data
        assert "indexed_count" in data
        assert "failed_count" in data
        assert "duration_seconds" in data
        assert data["indexed_count"] == data["total_laws"]

    @pytest.mark.asyncio
    async def test_reindex_service_error(self, client, mock_search_service):
        """Test reindex with service error returns 500."""
        from app.services.search_service import IndexingError
        mock_search_service.reindex_all_laws.side_effect = IndexingError("Reindex failed")

        response = await client.post("/api/v1/search/reindex")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data


# ============================================================================
# Test GET /api/v1/search/stats
# ============================================================================

class TestStatsEndpoint:
    """Test search statistics endpoint (admin)."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, client, mock_search_service):
        """Test successful stats retrieval."""
        response = await client.get("/api/v1/search/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "total_documents" in data
        assert "by_language" in data
        assert "by_category" in data
        assert "by_status" in data
        assert "index_health" in data
        assert "cache_status" in data

    @pytest.mark.asyncio
    async def test_get_stats_error(self, client, mock_search_service):
        """Test stats with error returns 500."""
        mock_search_service.meilisearch_client.index.side_effect = Exception("Connection failed")

        response = await client.get("/api/v1/search/stats")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ============================================================================
# Test GET /api/v1/search/health
# ============================================================================

class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client, mock_search_service):
        """Test health check returns healthy status."""
        # Configure mocks for healthy state
        mock_search_service.meilisearch_client.health.return_value = {"status": "available"}
        mock_search_service.redis_client.ping.return_value = True
        mock_search_service.embedding_service.health_check.return_value = {"status": "healthy"}

        response = await client.get("/api/v1/search/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "meilisearch" in data
        assert "redis" in data
        assert "embedding_service" in data

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, client, mock_search_service):
        """Test health check returns degraded when Redis is down."""
        # Configure mocks for degraded state
        mock_search_service.meilisearch_client.health.return_value = {"status": "available"}
        mock_search_service.redis_client.ping.side_effect = Exception("Redis down")
        mock_search_service.embedding_service.health_check.return_value = {"status": "healthy"}

        response = await client.get("/api/v1/search/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["status"] == "degraded"
        assert "error" in data["redis"]


# ============================================================================
# Test Error Responses
# ============================================================================

class TestErrorResponses:
    """Test various error response scenarios."""

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client, mock_search_service):
        """Test request with missing required fields returns 422."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "mode": "hybrid"
                # Missing "query"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_invalid_json(self, client, mock_search_service):
        """Test request with invalid JSON returns 422."""
        response = await client.post(
            "/api/v1/search/",
            content="invalid json{",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_invalid_filter_language(self, client, mock_search_service):
        """Test search with invalid language filter returns 422."""
        response = await client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "mode": "hybrid",
                "filters": {
                    "language": "invalid"  # Must be fr or en
                }
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
