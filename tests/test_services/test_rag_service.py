"""
Tests for RAGService.

Test categories:
- Core functionality (5 tests)
- Context retrieval (2 tests)
- Citation extraction (3 tests)
- Confidence calculation (2 tests)
- Conversation management (3 tests)

Total: 15 tests

Author: JuriX Team
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.conversation import Conversation, Message
from app.schemas.rag import RAGRequest, RAGResponse, Citation
from app.schemas.search import SearchResult
from app.services.rag_service import RAGService, RAGServiceError


# ==================== FIXTURES ====================

@pytest.fixture
def mock_db_session():
    """Mock async database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def rag_service(mock_db_session):
    """Create RAGService instance with mocked dependencies."""
    service = RAGService(mock_db_session)

    # Mock OllamaService
    service.ollama = AsyncMock()
    service.ollama.generate = AsyncMock()
    service.ollama.generate_stream = AsyncMock()
    service.ollama.health_check = AsyncMock()

    # Mock SearchService
    service.search_service = AsyncMock()
    service.search_service.search = AsyncMock()

    return service


@pytest.fixture
def sample_rag_request():
    """Sample RAG request."""
    return RAGRequest(
        question="Quelle est la responsabilité des dirigeants de société?",
        persona="citoyen",
        language="fr",
        session_id=None,
        stream=False
    )


@pytest.fixture
def mock_search_results():
    """Mock search results from SearchService."""
    return [
        SearchResult(
            law_id=1,
            reference="LOI-2024-001",
            title="Code OHADA",
            content="Les dirigeants sont responsables...",
            relevance_score=0.92,
            category_name="Droit commercial",
            highlights={"content": "Article 161: Les dirigeants sont responsables..."},
            matched_articles=[]
        ),
        SearchResult(
            law_id=2,
            reference="LOI-2023-015",
            title="Code des sociétés",
            content="Obligations des dirigeants...",
            relevance_score=0.87,
            category_name="Droit commercial",
            highlights={"content": "Article 5: Obligations des dirigeants..."},
            matched_articles=[]
        )
    ]


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama generation response."""
    return {
        "response": "Selon l'article 161 du Code OHADA, les dirigeants de société sont responsables civilement et pénalement de leurs actes de gestion.",
        "done": True,
        "total_duration": 2500000000
    }


# ==================== TESTS CORE FUNCTIONALITY ====================

class TestCoreFunctionality:
    """Tests for core RAG functionality."""

    @pytest.mark.asyncio
    async def test_ask_complete_pipeline(
        self,
        rag_service,
        sample_rag_request,
        mock_search_results,
        mock_ollama_response,
        mock_db_session
    ):
        """Test complete RAG pipeline from question to answer."""
        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = mock_search_results
        mock_search_response.search_time_ms = 150
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock Ollama response
        rag_service.ollama.generate.return_value = mock_ollama_response

        # Execute
        response = await rag_service.ask(sample_rag_request)

        # Assertions
        assert isinstance(response, RAGResponse)
        assert response.answer == mock_ollama_response["response"]
        assert response.confidence > 0
        assert response.total_time_ms > 0
        assert response.retrieval_time_ms > 0
        assert response.generation_time_ms > 0
        assert response.persona == "citoyen"

        # Verify search was called
        rag_service.search_service.search.assert_called_once()

        # Verify Ollama was called
        rag_service.ollama.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_with_no_search_results(
        self,
        rag_service,
        sample_rag_request,
        mock_db_session
    ):
        """Test handling of no search results."""
        # Mock empty search results
        mock_search_response = MagicMock()
        mock_search_response.results = []
        mock_search_response.search_time_ms = 100
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Execute
        response = await rag_service.ask(sample_rag_request)

        # Assertions
        assert response.confidence == 0.0
        assert len(response.sources) == 0
        assert "pas trouvé" in response.answer.lower() or "couldn't find" in response.answer.lower()

        # Ollama should not be called when no results
        rag_service.ollama.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_stream_yields_chunks(
        self,
        rag_service,
        sample_rag_request,
        mock_search_results,
        mock_db_session
    ):
        """Test streaming response yields chunks."""
        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = mock_search_results
        mock_search_response.search_time_ms = 150
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock streaming chunks
        async def mock_stream():
            yield "Selon "
            yield "l'article "
            yield "161"

        rag_service.ollama.generate_stream.return_value = mock_stream()

        # Execute
        chunks = []
        async for chunk_json in rag_service.ask_stream(sample_rag_request):
            chunks.append(chunk_json)

        # Assertions
        assert len(chunks) > 0

        # Last chunk should have done=True
        import json
        last_chunk = json.loads(chunks[-1])
        assert last_chunk["done"] is True

    @pytest.mark.asyncio
    async def test_ask_handles_ollama_error(
        self,
        rag_service,
        sample_rag_request,
        mock_search_results,
        mock_db_session
    ):
        """Test handling of Ollama service errors."""
        from app.services.ollama_service import OllamaServiceError

        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = mock_search_results
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock Ollama error
        rag_service.ollama.generate.side_effect = OllamaServiceError("Service unavailable")

        # Execute and expect error
        with pytest.raises(RAGServiceError) as exc_info:
            await rag_service.ask(sample_rag_request)

        assert "génération" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ask_saves_interaction_to_database(
        self,
        rag_service,
        sample_rag_request,
        mock_search_results,
        mock_ollama_response,
        mock_db_session
    ):
        """Test that interaction is saved to database."""
        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = mock_search_results
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock Ollama response
        rag_service.ollama.generate.return_value = mock_ollama_response

        # Execute
        await rag_service.ask(sample_rag_request)

        # Verify database operations
        assert mock_db_session.add.call_count >= 2  # Conversation + 2 messages
        mock_db_session.commit.assert_called_once()


# ==================== TESTS CONTEXT RETRIEVAL ====================

class TestContextRetrieval:
    """Tests for document retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_context_uses_hybrid_search(
        self,
        rag_service,
        mock_search_results
    ):
        """Test that context retrieval uses hybrid search mode."""
        mock_search_response = MagicMock()
        mock_search_response.results = mock_search_results
        rag_service.search_service.search.return_value = mock_search_response

        # Call retrieve
        results = await rag_service._retrieve_context("Test question", "fr")

        # Verify hybrid search called
        call_args = rag_service.search_service.search.call_args
        search_request = call_args[0][0]
        assert search_request.mode == "hybrid"
        assert search_request.limit == 5

    @pytest.mark.asyncio
    async def test_retrieve_context_respects_language_filter(
        self,
        rag_service,
        mock_search_results
    ):
        """Test that language filter is applied."""
        mock_search_response = MagicMock()
        mock_search_response.results = mock_search_results
        rag_service.search_service.search.return_value = mock_search_response

        await rag_service._retrieve_context("Test", "en")

        call_args = rag_service.search_service.search.call_args
        search_request = call_args[0][0]
        assert search_request.filters.language == "en"


# ==================== TESTS CITATION EXTRACTION ====================

class TestCitationExtraction:
    """Tests for citation extraction and validation."""

    def test_extract_citations_from_answer(self, rag_service, mock_search_results):
        """Test extracting citations from answer text."""
        answer = "Selon l'article 161 du Code OHADA, les dirigeants sont responsables. L'article 5 du Code des sociétés précise leurs obligations."

        citations = rag_service._extract_citations(answer, mock_search_results)

        # Should find 2 citations
        assert len(citations) >= 1
        assert all(isinstance(c, Citation) for c in citations)

    def test_extract_citations_validates_against_results(
        self,
        rag_service,
        mock_search_results
    ):
        """Test that citations are validated against search results."""
        # Citation for non-existent law
        answer = "Selon l'article 999 de la Loi Inexistante"

        citations = rag_service._extract_citations(answer, mock_search_results)

        # Should not extract citation for non-existent law
        assert len(citations) == 0

    def test_extract_citations_deduplicates(
        self,
        rag_service,
        mock_search_results
    ):
        """Test that duplicate citations are removed."""
        answer = "L'article 161 du Code OHADA stipule... Comme mentionné dans l'article 161 du Code OHADA..."

        citations = rag_service._extract_citations(answer, mock_search_results)

        # Should have only one citation despite two mentions
        article_161_count = sum(1 for c in citations if c.article_number == "161")
        assert article_161_count <= 1


# ==================== TESTS CONFIDENCE CALCULATION ====================

class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_calculate_confidence_with_citations(
        self,
        rag_service,
        mock_search_results
    ):
        """Test confidence calculation with citations."""
        answer = "Detailed answer with good length " * 20  # ~100 words
        citations = [
            Citation(
                law_id=1,
                law_reference="LOI-2024-001",
                law_title="Test Law",
                article_number="1",
                excerpt="Test excerpt",
                relevance_score=0.9
            )
        ]

        confidence = rag_service._calculate_confidence(
            answer, citations, mock_search_results
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.3  # Should have reasonable confidence with citations

    def test_calculate_confidence_without_citations(
        self,
        rag_service,
        mock_search_results
    ):
        """Test confidence calculation without citations."""
        answer = "Answer without citations " * 20
        citations = []

        confidence = rag_service._calculate_confidence(
            answer, citations, mock_search_results
        )

        assert 0.0 <= confidence <= 1.0
        # Lower confidence without citations
        assert confidence < 0.5


# ==================== TESTS CONVERSATION MANAGEMENT ====================

class TestConversationManagement:
    """Tests for conversation and message management."""

    @pytest.mark.asyncio
    async def test_load_existing_conversation(
        self,
        rag_service,
        mock_db_session
    ):
        """Test loading existing conversation by session_id."""
        # Mock existing conversation
        existing_conv = Conversation(
            id=1,
            session_id="test-session",
            persona="citoyen",
            language="fr"
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_conv
        mock_db_session.execute.return_value = mock_result

        # Mock messages
        mock_msg_result = MagicMock()
        mock_msg_result.scalars.return_value.all.return_value = []

        # Setup to return existing conv first, then messages
        mock_db_session.execute.side_effect = [mock_result, mock_msg_result]

        conv, messages = await rag_service._load_or_create_conversation(
            "test-session", "citoyen", "fr"
        )

        assert conv.session_id == "test-session"
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_create_new_conversation(
        self,
        rag_service,
        mock_db_session
    ):
        """Test creating new conversation when none exists."""
        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        conv, messages = await rag_service._load_or_create_conversation(
            None, "avocat", "fr"
        )

        assert conv.persona == "avocat"
        assert conv.language == "fr"
        assert len(messages) == 0
        mock_db_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_interaction_creates_messages(
        self,
        rag_service,
        mock_db_session
    ):
        """Test that save_interaction creates user and assistant messages."""
        conversation = Conversation(
            id=1,
            session_id="test",
            persona="citoyen",
            language="fr"
        )

        await rag_service._save_interaction(
            conversation=conversation,
            question="Test question?",
            answer="Test answer",
            citations=[],
            confidence=0.8,
            retrieval_time_ms=100,
            generation_time_ms=2000
        )

        # Should add 2 messages (user + assistant)
        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called_once()
