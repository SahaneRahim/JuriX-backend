"""
Tests for RAG API routes.

Test categories:
- POST /ask endpoint (4 tests)
- POST /ask/stream endpoint (2 tests)
- GET /conversations/{session_id} endpoint (2 tests)
- DELETE /conversations/{session_id} endpoint (1 test)
- GET /health endpoint (1 test)

Total: 10 tests

Author: JuriX Team
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.rag import RAGResponse, Citation


# ==================== FIXTURES ====================

@pytest.fixture
def sync_client():
    """Client synchrone SANS base : reserve aux tests dont le service est
    entierement double. Les tests qui touchent la base utilisent la fixture
    `client` de conftest, qui redirige get_db vers la base de test."""
    return TestClient(app)


@pytest.fixture
def mock_rag_response():
    """Mock RAG response."""
    return RAGResponse(
        answer="Selon l'article 161 du Code OHADA, les dirigeants sont responsables...",
        confidence=0.85,
        sources=[
            Citation(
                law_id=1,
                law_reference="LOI-2024-001",
                law_title="Code OHADA",
                article_number="161",
                excerpt="Les dirigeants sont responsables...",
                relevance_score=0.92
            )
        ],
        session_id="test-session-123",
        retrieval_time_ms=180,
        generation_time_ms=2400,
        total_time_ms=2600,
        persona="citoyen"
    )


# ==================== TESTS POST /ask ====================

class TestAskEndpoint:
    """Tests for POST /api/v1/rag/ask endpoint."""

    @patch('app.api.routes.rag.RAGService')
    def test_ask_success(self, mock_rag_service_class, sync_client, mock_rag_response):
        """Test successful ask request."""
        # Mock service
        mock_service = AsyncMock()
        mock_service.ask = AsyncMock(return_value=mock_rag_response)
        mock_rag_service_class.return_value = mock_service

        # Request
        response = sync_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Quelle est la responsabilité des dirigeants?",
                "persona": "citoyen",
                "language": "fr",
                "session_id": None,
                "stream": False
            }
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["persona"] == "citoyen"

    def test_ask_invalid_persona(self, sync_client):
        """Test ask with invalid persona."""
        response = sync_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Test question?",
                "persona": "invalid_persona",
                "language": "fr"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_ask_invalid_language(self, sync_client):
        """Test ask with invalid language."""
        response = sync_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Test question?",
                "persona": "citoyen",
                "language": "invalid"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_ask_question_too_short(self, sync_client):
        """Test ask with question too short."""
        response = sync_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Hi",  # Too short (< 5 chars)
                "persona": "citoyen",
                "language": "fr"
            }
        )

        assert response.status_code == 422  # Validation error


# ==================== TESTS POST /ask/stream ====================

class TestAskStreamEndpoint:
    """Tests for POST /api/v1/rag/ask/stream endpoint."""

    @patch('app.api.routes.rag.RAGService')
    def test_ask_stream_returns_sse(self, mock_rag_service_class, sync_client):
        """Test streaming response uses SSE format."""
        # Mock streaming chunks
        async def mock_stream():
            yield json.dumps({"chunk": "Test ", "done": False})
            yield json.dumps({"chunk": "answer", "done": False})
            yield json.dumps({
                "chunk": "",
                "done": True,
                "sources": [],
                "confidence": 0.8,
                "session_id": "test"
            })

        mock_service = AsyncMock()
        mock_service.ask_stream = AsyncMock(return_value=mock_stream())
        mock_rag_service_class.return_value = mock_service

        # Request
        response = sync_client.post(
            "/api/v1/rag/ask/stream",
            json={
                "question": "Test question?",
                "persona": "citoyen",
                "language": "fr"
            }
        )

        # Assertions
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    @patch('app.api.routes.rag.RAGService')
    def test_ask_stream_chunks_format(self, mock_rag_service_class, sync_client):
        """Test streaming chunks are properly formatted."""
        async def mock_stream():
            yield json.dumps({"chunk": "Test", "done": False})
            yield json.dumps({"chunk": "", "done": True})

        mock_service = AsyncMock()
        mock_service.ask_stream = AsyncMock(return_value=mock_stream())
        mock_rag_service_class.return_value = mock_service

        response = sync_client.post(
            "/api/v1/rag/ask/stream",
            json={
                "question": "Test?",
                "persona": "citoyen"
            }
        )

        # Check SSE format
        content = response.text
        assert "data: " in content


# ==================== TESTS GET /conversations/{session_id} ====================

class TestGetConversationEndpoint:
    """Tests for GET /api/v1/rag/conversations/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, client):
        """Test getting non-existent conversation."""
        response = await client.get("/api/v1/rag/conversations/nonexistent-session-id")

        # 404 strictement, sur la base de test : "404 ou 500" validait la panne
        # autant que le fonctionnement, et c'est bien 500 que renvoyait la route
        # avec le client sans base.
        assert response.status_code == 404


# ==================== TESTS DELETE /conversations/{session_id} ====================

class TestDeleteConversationEndpoint:
    """Tests for DELETE /api/v1/rag/conversations/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, client):
        """Test deleting non-existent conversation."""
        response = await client.delete("/api/v1/rag/conversations/nonexistent-id")

        assert response.status_code == 404


# ==================== TESTS GET /health ====================

class TestHealthEndpoint:
    """Tests for GET /api/v1/rag/health endpoint."""

    @patch('app.api.routes.rag.RAGService')
    def test_health_check(self, mock_rag_service_class, sync_client):
        """Test health check endpoint."""
        # Mock healthy service
        # L'attribut est llm (GeminiService), plus ollama : la doublure posee
        # sur un nom disparu laissait un AsyncMock nu sur llm, dont .get()
        # renvoyait une coroutine placee telle quelle dans la reponse
        # ("Unable to serialize unknown type: <class 'coroutine'>").
        mock_service = AsyncMock()
        mock_service.llm = AsyncMock()
        mock_service.llm.health_check = AsyncMock(return_value={
            "status": "healthy",
            "model": "gemini-2.0-flash",
            "available": True
        })
        mock_service.db = AsyncMock()
        mock_service.db.execute = AsyncMock()
        mock_service.search_service = AsyncMock()
        mock_service.search_service.search = AsyncMock()
        mock_rag_service_class.return_value = mock_service

        response = sync_client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "llm" in data
        assert data["llm"] == "healthy"
        assert "database" in data
        assert "search_service" in data
