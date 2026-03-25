"""
Integration tests for RAG system.

Test categories:
- End-to-end RAG pipeline (2 tests)
- Conversation persistence (1 test)
- Multi-turn conversations (1 test)
- Performance benchmarks (1 test)

Total: 5 tests

Author: JuriX Team

Note: These tests require:
- Database connection
- Ollama service running (can be skipped if unavailable)
- SearchService with indexed data
"""

import pytest
import time
from httpx import AsyncClient

from app.main import app
from app.core.config import settings


# ==================== FIXTURES ====================

@pytest.fixture
async def async_client():
    """Async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def skip_if_ollama_unavailable():
    """Skip test if Ollama is not available."""
    import httpx
    try:
        response = httpx.get(f"{settings.OLLAMA_URL}/api/tags", timeout=2.0)
        if response.status_code != 200:
            pytest.skip("Ollama service not available")
    except Exception:
        pytest.skip("Ollama service not available")


# ==================== INTEGRATION TESTS ====================

class TestEndToEndRAG:
    """End-to-end RAG pipeline tests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_rag_flow_with_real_services(
        self,
        async_client,
        skip_if_ollama_unavailable
    ):
        """
        Test complete RAG flow with real services.

        This test requires:
        - Ollama running with mistral:7b
        - Database with indexed laws
        - SearchService operational
        """
        # Send question
        response = await async_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Qu'est-ce qu'un contrat?",
                "persona": "étudiant",
                "language": "fr",
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
        assert "total_time_ms" in data

        # Verify response quality
        assert len(data["answer"]) > 50  # Meaningful answer
        assert 0.0 <= data["confidence"] <= 1.0

        # Performance check
        assert data["total_time_ms"] < 10000  # < 10s (generous for integration)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_with_no_relevant_documents(
        self,
        async_client,
        skip_if_ollama_unavailable
    ):
        """Test RAG when no relevant documents are found."""
        # Ask nonsense question
        response = await async_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Quel est le prix du café à Tokyo?",
                "persona": "citoyen",
                "language": "fr"
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return "no results" message
        assert data["confidence"] == 0.0
        assert len(data["sources"]) == 0
        assert "pas trouvé" in data["answer"].lower()


class TestConversationPersistence:
    """Tests for conversation persistence."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_conversation_saved_and_retrievable(
        self,
        async_client,
        skip_if_ollama_unavailable
    ):
        """Test that conversations are saved and can be retrieved."""
        # Ask first question
        response1 = await async_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Qu'est-ce qu'un contrat?",
                "persona": "citoyen",
                "language": "fr"
            }
        )

        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # Retrieve conversation
        response2 = await async_client.get(
            f"/api/v1/rag/conversations/{session_id}"
        )

        assert response2.status_code == 200
        conv_data = response2.json()

        assert conv_data["session_id"] == session_id
        assert "messages" in conv_data
        assert len(conv_data["messages"]) >= 2  # User + assistant


class TestMultiTurnConversations:
    """Tests for multi-turn conversation context."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_maintained_across_turns(
        self,
        async_client,
        skip_if_ollama_unavailable
    ):
        """Test that conversation context is maintained."""
        # First question
        response1 = await async_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Qu'est-ce qu'un contrat?",
                "persona": "citoyen",
                "language": "fr"
            }
        )

        session_id = response1.json()["session_id"]

        # Follow-up question (requires context)
        response2 = await async_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Quelles sont les conditions pour qu'il soit valide?",
                "persona": "citoyen",
                "language": "fr",
                "session_id": session_id
            }
        )

        assert response2.status_code == 200
        data = response2.json()

        # Should have answer (context understood)
        assert len(data["answer"]) > 50
        assert data["session_id"] == session_id


class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_rag_performance_under_5_seconds(
        self,
        async_client,
        skip_if_ollama_unavailable
    ):
        """
        Test that RAG responds in < 5 seconds (spec requirement).

        This is a critical performance test.
        """
        start_time = time.time()

        response = await async_client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Quelle est la responsabilité des dirigeants?",
                "persona": "avocat",
                "language": "fr"
            }
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Assertions
        assert response.status_code == 200
        data = response.json()

        # Critical performance requirement
        assert data["total_time_ms"] < 5000, (
            f"RAG took {data['total_time_ms']}ms, "
            f"exceeds 5000ms requirement"
        )

        # Log performance breakdown
        print(f"\n Performance breakdown:")
        print(f"  Retrieval: {data['retrieval_time_ms']}ms")
        print(f"  Generation: {data['generation_time_ms']}ms")
        print(f"  Total: {data['total_time_ms']}ms")
        print(f"  Actual elapsed: {elapsed_ms:.0f}ms")
