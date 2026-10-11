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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.conversation import Conversation, Message
from app.schemas.rag import Citation, RAGRequest, RAGResponse
from app.schemas.search import ChunkResult
from app.services.intent_classifier import IntentResult
from app.services.prompts import get_conversational_prompt, get_system_prompt
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

    # Doublure du LLM. L'attribut s'appelle `llm` (GeminiService) : la fixture
    # simulait `service.llm`, disparu avec l'ancienne architecture, si bien
    # que les tests appelaient la VRAIE API Gemini et echouaient en 400.
    service.llm = AsyncMock()
    service.llm.generate = AsyncMock(
        return_value="Selon l'article 161 du Code OHADA, les dirigeants sont "
        "responsables civilement et penalement de leurs actes de gestion."
    )
    service.llm.health_check = AsyncMock(return_value={"status": "healthy"})

    # Mock SearchService
    service.search_service = AsyncMock()
    service.search_service.search = AsyncMock()

    return service


@pytest.fixture(autouse=True)
def routage_juridique(request):
    """
    Epingle l'intention a « juridique » pour tout ce module.

    Sans cela, `ask()` appelle le modele DEUX fois — classification puis
    generation — et toute assertion `llm.generate.assert_called_once()` tombe,
    a commencer par test_ask_complete_pipeline. Pire : `assert_not_called()`
    dans test_ask_with_no_search_results deviendrait faux alors que le
    comportement teste, lui, n'a pas bouge.

    L'epingler ici plutot que dans chaque test dit aussi ce que ces tests
    couvrent : le chemin juridique, pas le routage. Les tests de routage
    reaffectent `return_value` sur cette meme doublure.
    """
    with patch(
        "app.services.rag_service.classify_intent",
        new=AsyncMock(return_value=IntentResult("juridique", 1.0, "test")),
    ) as double:
        yield double


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
    """
    Chunks renvoyes par SearchService.

    Le RAG consomme desormais des ChunkResult : un article, avec son numero,
    sa section, sa page et son CONTENU INTEGRAL. La doublure precedente ne
    portait qu'un extrait dans `highlights`, ce qui laissait croire que le
    modele recevait un texte alors qu'il ne voyait que 400 caracteres.
    """
    return [
        ChunkResult(
            article_id=161,
            law_id=1,
            number="161",
            article_title="Responsabilité des dirigeants",
            section="TITRE III — DES DIRIGEANTS",
            page_number=47,
            content=(
                "Les dirigeants sociaux sont responsables, individuellement ou "
                "solidairement selon le cas, envers la société ou envers les tiers, "
                "des fautes commises dans l'exercice de leurs fonctions."
            ),
            excerpt="Article 161: Les dirigeants sont responsables...",
            reference="LOI-2024-001",
            law_title="Code OHADA",
            type="loi",
            language="fr",
            status="published",
            category_name="Droit commercial",
            relevance_score=0.92,
            source="fts",
        ),
        ChunkResult(
            article_id=5,
            law_id=2,
            number="5",
            article_title="Obligations des dirigeants",
            content="Les dirigeants sont tenus d'une obligation de loyauté envers la société.",
            excerpt="Article 5: Obligations des dirigeants...",
            reference="LOI-2023-015",
            law_title="Code des sociétés",
            type="loi",
            language="fr",
            status="published",
            category_name="Droit commercial",
            relevance_score=0.87,
            source="fts",
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """Mock Gemini generation response."""
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
        mock_llm_response,
        mock_db_session
    ):
        """Test complete RAG pipeline from question to answer."""
        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = []
        mock_search_response.chunks = mock_search_results
        mock_search_response.search_time_ms = 150
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock Gemini response
        rag_service.llm.generate.return_value = mock_llm_response

        # Execute
        response = await rag_service.ask(sample_rag_request)

        # Assertions
        assert isinstance(response, RAGResponse)
        assert response.answer == mock_llm_response["response"]
        assert response.confidence > 0
        assert response.total_time_ms > 0
        # >= 0 et non > 0 : avec des doublures la recherche prend moins d'une
        # milliseconde et l'arrondi entier donne 0.
        assert response.retrieval_time_ms >= 0
        # >= 0 : avec une doublure la generation prend moins d'une milliseconde
        # et l'arrondi entier donne 0.
        assert response.generation_time_ms >= 0
        assert response.persona == "citoyen"

        # Verify search was called
        rag_service.search_service.search.assert_called_once()

        # Verify Gemini was called
        rag_service.llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_reponse_tronquee_le_dit_au_lecteur(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        """Une phrase coupee ne doit pas passer pour une reponse complete."""
        from app.services.rag_service import MENTION_REPONSE_TRONQUEE

        recherche = MagicMock(results=[], chunks=mock_search_results, search_time_ms=1)
        rag_service.search_service.search.return_value = recherche
        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        rag_service.llm.generate.return_value = {
            "response": "Un militaire qui abuse de sa position pour", "tronquee": True,
        }

        response = await rag_service.ask(sample_rag_request)

        assert response.answer.endswith(MENTION_REPONSE_TRONQUEE)
        assert rag_service.llm.generate.call_args.kwargs["max_tokens"] == 8192
        assert rag_service.llm.generate.call_args.kwargs["reflexion"] is None

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
        mock_search_response.chunks = []
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

        # Gemini should not be called when no results
        rag_service.llm.generate.assert_not_called()

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
        mock_search_response.results = []
        mock_search_response.chunks = mock_search_results
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

        rag_service.llm.generate_stream.return_value = mock_stream()

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
    async def test_ask_handles_llm_error(
        self,
        rag_service,
        sample_rag_request,
        mock_search_results,
        mock_db_session
    ):
        """Test handling of Gemini service errors."""
        from app.services.gemini_service import GeminiServiceError

        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = []
        mock_search_response.chunks = mock_search_results
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock Gemini error
        rag_service.llm.generate.side_effect = GeminiServiceError("Service unavailable")

        # Execute and expect error
        with pytest.raises(RAGServiceError) as exc_info:
            await rag_service.ask(sample_rag_request)

        # Le service enveloppe l'erreur du LLM dans RAGServiceError en
        # conservant le message d'origine ; c'est ce message qui est verifie,
        # et non un libelle fixe qui figerait la formulation.
        assert "service unavailable" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ask_saves_interaction_to_database(
        self,
        rag_service,
        sample_rag_request,
        mock_search_results,
        mock_llm_response,
        mock_db_session
    ):
        """Test that interaction is saved to database."""
        # Mock search results
        mock_search_response = MagicMock()
        mock_search_response.results = []
        mock_search_response.chunks = mock_search_results
        rag_service.search_service.search.return_value = mock_search_response

        # Mock no existing conversation
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock Gemini response
        rag_service.llm.generate.return_value = mock_llm_response

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
        mock_search_response.chunks = mock_search_results
        mock_search_response.results = []
        rag_service.search_service.search.return_value = mock_search_response

        chunks = await rag_service._retrieve_chunks("Test question", "fr")

        call_args = rag_service.search_service.search.call_args
        search_request = call_args[0][0]
        # Mode hybride : il etait epingle sur "text" tant que la recherche
        # semantique ne rendait rien d'exploitable (niveau loi, sans texte).
        assert search_request.mode == "hybrid"
        assert search_request.limit == RAGService.TOP_K_CHUNKS
        # Ce sont bien les chunks qui sont consommes, pas les resultats loi.
        assert chunks == mock_search_results

    @pytest.mark.asyncio
    async def test_retrieve_context_respects_language_filter(
        self,
        rag_service,
        mock_search_results
    ):
        """Test that language filter is applied."""
        mock_search_response = MagicMock()
        mock_search_response.chunks = mock_search_results
        mock_search_response.results = []
        rag_service.search_service.search.return_value = mock_search_response

        await rag_service._retrieve_chunks("Test", "en")

        call_args = rag_service.search_service.search.call_args
        search_request = call_args[0][0]
        # Aucun filtre de langue n'est applique : le service cherche dans
        # toutes les langues pour maximiser le rappel (rag_service.py,
        # "Searches all languages to maximize results"). Le corpus camerounais
        # est bilingue et un meme texte existe souvent dans une seule langue.
        assert search_request.filters.language is None
        assert search_request.filters.status == "published"


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
        # La citation pointe une LIGNE d'article, pas seulement un numero
        # extrait du texte de la reponse : le front peut l'ouvrir directement.
        assert citations[0].article_id == 161
        # L'extrait vient du chunk envoye au modele, sans requete supplementaire.
        assert "responsables" in citations[0].excerpt

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
        # .unique() est desormais appele avant scalar_one_or_none : obligatoire
        # apres un joinedload sur une collection, sans quoi SQLAlchemy leve
        # InvalidRequestError.
        mock_result = MagicMock()
        mock_result.unique.return_value.scalar_one_or_none.return_value = existing_conv
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


# ==================== ROUTAGE D'INTENTION ====================

class TestRoutageIntention:
    """
    Le chemin conversationnel : ce qu'il fait, et surtout ce qu'il NE fait pas.

    La fixture `routage_juridique` (autouse) epingle l'intention a
    « juridique » ; ces tests reaffectent son `return_value`.
    """

    @pytest.fixture
    def conversation_neuve(self, mock_db_session):
        """Aucune conversation existante : `ask` en cree une."""
        resultat = MagicMock()
        resultat.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = resultat
        return resultat

    @pytest.fixture
    def smalltalk(self, routage_juridique, rag_service):
        routage_juridique.return_value = IntentResult("smalltalk", 0.97, "llm")
        rag_service.llm.generate.return_value = {
            "response": "Je vais bien, merci. Je suis là pour vos questions de droit camerounais."
        }
        return routage_juridique

    @pytest.mark.asyncio
    async def test_smalltalk_ne_declenche_aucune_recherche(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        """Le gain qui justifie tout : ni embedding facture, ni recherche hybride."""
        await rag_service.ask(sample_rag_request)

        rag_service.search_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_smalltalk_ne_renvoie_aucune_source(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        """Le bug d'origine : un article du Code Minier sous « comment vas tu ? »."""
        reponse = await rag_service.ask(sample_rag_request)

        assert reponse.sources == []

    @pytest.mark.asyncio
    async def test_smalltalk_ne_fait_aucune_recuperation(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        reponse = await rag_service.ask(sample_rag_request)

        assert reponse.retrieval_time_ms == 0

    @pytest.mark.asyncio
    async def test_smalltalk_renvoie_son_intention(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        reponse = await rag_service.ask(sample_rag_request)

        assert reponse.intent == "smalltalk"

    @pytest.mark.asyncio
    async def test_reponse_conversationnelle_coupee_le_dit(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        """Le budget conversationnel est court : une phrase coupee le dit aussi ici."""
        from app.services.rag_service import MENTION_REPONSE_TRONQUEE

        rag_service.llm.generate.return_value = {
            "response": "Je vais bien, merci. Je suis là pour", "tronquee": True,
            "fin": "MAX_TOKENS",
        }

        reponse = await rag_service.ask(sample_rag_request)

        assert reponse.answer.endswith(MENTION_REPONSE_TRONQUEE)

    @pytest.mark.asyncio
    async def test_smalltalk_utilise_le_prompt_conversationnel(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        await rag_service.ask(sample_rag_request)

        systeme = rag_service.llm.generate.await_args.kwargs["system"]
        # La requete est en citoyen : le prompt porte sa ligne d'audience.
        assert systeme == get_conversational_prompt("smalltalk", "fr", "citoyen")
        assert systeme != get_system_prompt("citoyen", "fr")

    @pytest.mark.asyncio
    async def test_smalltalk_n_envoie_aucun_document_au_modele(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk
    ):
        await rag_service.ask(sample_rag_request)

        prompt = rag_service.llm.generate.await_args.kwargs["prompt"]
        assert "Documents juridiques pertinents" not in prompt

    @pytest.mark.asyncio
    async def test_smalltalk_est_enregistre_dans_la_conversation(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk, mock_db_session
    ):
        """L'historique doit rester continu : la question suivante voit la salutation."""
        await rag_service.ask(sample_rag_request)

        messages = [
            appel.args[0] for appel in mock_db_session.add.call_args_list
            if isinstance(appel.args[0], Message)
        ]
        assert [m.role for m in messages] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_smalltalk_ne_titre_pas_la_conversation(
        self, rag_service, sample_rag_request, conversation_neuve, smalltalk, mock_db_session
    ):
        """Sinon la barre laterale se remplit de « bonjour » et « comment vas tu ? »."""
        await rag_service.ask(sample_rag_request)

        conversations = [
            appel.args[0] for appel in mock_db_session.add.call_args_list
            if isinstance(appel.args[0], Conversation)
        ]
        assert conversations and conversations[0].title is None

    @pytest.mark.asyncio
    async def test_meta_titre_la_conversation(
        self, rag_service, sample_rag_request, conversation_neuve,
        routage_juridique, mock_db_session
    ):
        """Une question sur le produit en dit assez pour s'y retrouver."""
        routage_juridique.return_value = IntentResult("meta", 0.95, "llm")
        rag_service.llm.generate.return_value = {"response": "Je suis JuriX."}

        await rag_service.ask(sample_rag_request)

        conversations = [
            appel.args[0] for appel in mock_db_session.add.call_args_list
            if isinstance(appel.args[0], Conversation)
        ]
        assert conversations and conversations[0].title is not None

    @pytest.mark.asyncio
    async def test_law_id_court_circuite_le_classificateur(
        self, rag_service, conversation_neuve, routage_juridique, mock_search_results
    ):
        """
        L'utilisateur lit un document precis : la preuve contextuelle vaut mieux
        qu'un verdict de modele, et l'appel est economise.
        """
        reponse_recherche = MagicMock()
        reponse_recherche.results = []
        reponse_recherche.chunks = mock_search_results
        reponse_recherche.search_time_ms = 10
        rag_service.search_service.search.return_value = reponse_recherche
        rag_service.llm.generate.return_value = {
            "response": "Selon l'article 161 du Code OHADA, les dirigeants sont responsables."
        }

        await rag_service.ask(
            RAGRequest(question="et ce document, il dit quoi ?", law_id=1)
        )

        routage_juridique.assert_not_called()

    @pytest.mark.asyncio
    async def test_intention_juridique_conserve_le_pipeline(
        self, rag_service, sample_rag_request, conversation_neuve,
        mock_search_results, mock_llm_response
    ):
        """Non-regression : le chemin qui compte n'a pas bouge."""
        reponse_recherche = MagicMock()
        reponse_recherche.results = []
        reponse_recherche.chunks = mock_search_results
        reponse_recherche.search_time_ms = 150
        rag_service.search_service.search.return_value = reponse_recherche
        rag_service.llm.generate.return_value = mock_llm_response

        reponse = await rag_service.ask(sample_rag_request)

        rag_service.search_service.search.assert_called_once()
        assert reponse.intent == "juridique"
        assert reponse.sources

    @pytest.mark.asyncio
    async def test_echec_du_classificateur_conserve_le_pipeline(
        self, rag_service, sample_rag_request, conversation_neuve, routage_juridique,
        mock_search_results, mock_llm_response
    ):
        """
        Le defaut sur panne est « juridique », donc le comportement anterieur :
        une saturation du fournisseur ne fait pas basculer la plateforme en
        mode bavardage.
        """
        routage_juridique.return_value = IntentResult("juridique", 0.0, "defaut-erreur")
        reponse_recherche = MagicMock()
        reponse_recherche.results = []
        reponse_recherche.chunks = mock_search_results
        reponse_recherche.search_time_ms = 150
        rag_service.search_service.search.return_value = reponse_recherche
        rag_service.llm.generate.return_value = mock_llm_response

        await rag_service.ask(sample_rag_request)

        rag_service.search_service.search.assert_called_once()


# ==================== SOURCES FABRIQUEES : LE GARDE-FOU ====================

class TestGardeFouDesSources:
    """
    `_create_sources_from_results` fabrique une source quand le modele n'a rien
    cite. Utile, mais applique sans condition il accroche la loi la mieux
    classee a une reponse qui dit elle-meme n'avoir rien trouve.
    """

    @pytest.mark.parametrize(
        "aveu",
        [
            # Cas fondateur, mesure en conditions reelles sur « que dit
            # l'article 33 du Code Minier ? » : la reponse avouait l'absence et
            # portait quand meme une source vers l'article 1.
            "Les documents fournis ne contiennent pas l'article 33 de la Loi N°2023/014.",
            "Cette information ne figure pas dans les documents fournis.",
            "Les documents ne contiennent aucune information sur ce point.",
            "The documents provided do not contain this information.",
        ],
    )
    def test_pas_de_source_quand_la_reponse_avoue_ne_pas_savoir(
        self, rag_service, mock_search_results, aveu
    ):
        assert not rag_service._peut_fabriquer_une_source(aveu, mock_search_results)

    @pytest.mark.parametrize(
        "reponse",
        [
            # L'aveu porte sur LES DOCUMENTS ; ici la negation porte sur la LOI,
            # ce qui est une vraie reponse juridique et garde sa source.
            "La loi ne contient pas de disposition sur ce point, mais l'article 5 encadre le cas voisin.",
            "Le Code ne comporte pas d'exception pour ce cas.",
            "Les dirigeants sont responsables de leurs actes de gestion.",
        ],
    )
    def test_une_negation_portant_sur_la_loi_n_est_pas_un_aveu(
        self, rag_service, mock_search_results, reponse
    ):
        assert rag_service._peut_fabriquer_une_source(reponse, mock_search_results)

    def test_pas_de_source_quand_la_reponse_avoue_en_anglais(
        self, rag_service, mock_search_results
    ):
        assert not rag_service._peut_fabriquer_une_source(
            "The supplied documents do not contain this information.",
            mock_search_results,
        )

    def test_pas_de_source_sous_le_seuil_de_pertinence(
        self, rag_service, mock_search_results
    ):
        """Le moins mauvais resultat d'une recherche ratee n'est pas une source."""
        mock_search_results[0].relevance_score = 0.1

        assert not rag_service._peut_fabriquer_une_source(
            "Les dirigeants sont responsables.", mock_search_results
        )

    def test_source_fabriquee_au_dessus_du_seuil(self, rag_service, mock_search_results):
        """Non-regression du cas utile : le repli sert encore."""
        assert rag_service._peut_fabriquer_une_source(
            "Les dirigeants sont responsables.", mock_search_results
        )

    def test_pas_de_source_sans_resultat(self, rag_service):
        assert not rag_service._peut_fabriquer_une_source("Peu importe.", [])


# ==================== EXTRACTION DES CITATIONS ====================

class TestCitationsMultilingues:

    def test_forme_anglaise_reconnue(self, rag_service, mock_search_results):
        """
        Avant l'ajout des connecteurs `of the` / `of`, AUCUNE reponse anglaise
        n'a jamais produit une citation extraite : toutes les sources anglaises
        affichees venaient du repli, donc n'etaient pas des citations mais des
        devinettes.
        """
        mock_search_results[0].law_title = "OHADA Code"

        citations = rag_service._extract_citations(
            "According to Article 161 of the OHADA Code, directors are liable.",
            mock_search_results,
        )

        assert len(citations) == 1
        assert citations[0].article_id == 161

    def test_pluriel_non_reconnu(self, rag_service, mock_search_results):
        """
        Limitation connue, epinglee ici plutot que laissee en commentaire :
        c'est la raison d'etre de la contrainte « au singulier » des prompts.
        Si ce test se met a echouer, la regex a gagne le pluriel et les prompts
        peuvent se relacher.
        """
        citations = rag_service._extract_citations(
            "Les articles 161 et 5 du Code OHADA le prevoient.", mock_search_results
        )

        assert citations == []

    @pytest.mark.parametrize("reponse", [
        "Les dirigeants sont responsables (art. 161 du Code OHADA).",
        "Selon l'article 161, alinéa 2, du Code OHADA, les dirigeants repondent.",
        "Selon l'article 161 alinéa 2 du Code OHADA, les dirigeants repondent.",
        "Directors are liable (Art. 161 of the OHADA Code).",
    ])
    def test_abreviation_et_alinea_reconnus(self, rag_service, mock_search_results, reponse):
        """
        Mesure du 2026-10-11 : « (art. 15 du Décret…) » et « l'article 5,
        alinéa 1, du Décret… » faisaient l'essentiel des citations illisibles
        de l'avocat. Illisibles, elles ne devenaient pas des sources.
        """
        mock_search_results[0].law_title = "Code OHADA"

        citations = rag_service._extract_citations(reponse, mock_search_results)

        assert [c.article_id for c in citations] == [161]

    def test_un_mot_en_art_n_est_pas_une_citation(self, rag_service, mock_search_results):
        """\\b devant « art. » : « départ. 161 » ne cite rien."""
        assert rag_service._extract_citations("Le départ. 161 du train.", mock_search_results) == []

    @pytest.mark.parametrize("reponse", [
        "Selon l'article 161 du Code OHADA et l'article 5 du Code des sociétés, ils repondent.",
        "Ils repondent (l'article 161 du Code OHADA) et l'article 5 du Code des sociétés le precise.",
        "Directors are liable under Article 161 of the Code OHADA and Article 5 of the Code des sociétés.",
    ])
    def test_deux_citations_dans_la_meme_phrase(self, rag_service, mock_search_results, reponse):
        """
        La forme que les prompts demandent pour plusieurs articles. Le texte
        cite de la premiere avalait la seconde, qui ne devenait pas une source.
        """
        citations = rag_service._extract_citations(reponse, mock_search_results)

        assert [c.article_id for c in citations] == [161, 5]

    @pytest.mark.parametrize("reponse", [
        "Selon l'article 161, alinéa 1er, du Code OHADA, ils repondent.",
        "Selon l'article 161, alinéa premier, du Code OHADA, ils repondent.",
        "Selon l'article 161, paragraphe 2, du Code OHADA, ils repondent.",
        "Directors are liable under Article 161(2) of the Code OHADA.",
    ])
    def test_alinea_ordinal_et_paragraphe(self, rag_service, mock_search_results, reponse):
        """« alinéa 1er » est la forme usuelle : elle rendait toute la citation illisible."""
        assert [c.article_id for c in rag_service._extract_citations(reponse, mock_search_results)] == [161]

    def test_la_loi_citee_departage_les_articles_de_meme_numero(
        self, rag_service, mock_search_results
    ):
        """
        Deux lois exposent un article 5 : la citation va a celle qu'elle nomme,
        pas au premier chunk qui porte le numero.
        """
        decret = mock_search_results[1].model_copy(update={
            "article_id": 501, "law_id": 9, "law_title": "Décret N°2019/143",
            "reference": "DEC-2019-143",
        })
        chunks = [decret, *mock_search_results]

        citations = rag_service._extract_citations(
            "Selon l'article 5 du Code des sociétés, ils sont loyaux.", chunks
        )

        assert [c.article_id for c in citations] == [5]

    def test_jamais_l_article_d_une_autre_loi_presente(self, rag_service, mock_search_results):
        """
        Le Code OHADA est dans le contexte, mais pas son article 5 : l'article 5
        d'une autre loi n'est pas une source de cette citation.
        """
        citations = rag_service._extract_citations(
            "Selon l'article 5 du Code OHADA, ils sont loyaux.", mock_search_results
        )

        assert citations == []

    def test_loi_nommee_autrement_le_numero_suffit(self, rag_service, mock_search_results):
        """Repli conserve : la loi citee n'est reconnue nulle part dans le contexte."""
        citations = rag_service._extract_citations(
            "Selon l'article 5 de l'Acte uniforme, ils sont loyaux.", mock_search_results
        )

        assert [c.article_id for c in citations] == [5]


class TestFluxAligneSurAsk:
    """
    Le flux ne faisait pas ce que fait `ask` : document ouvert ignore, aucune
    source de repli, reponse coupee non signalee, quota confondu avec une
    panne. L'interface l'emprunte desormais.
    """

    @staticmethod
    def _preparer(rag_service, mock_db_session, mock_search_results, morceaux, raison="STOP"):
        recherche = MagicMock(results=[], chunks=mock_search_results, search_time_ms=1)
        rag_service.search_service.search.return_value = recherche
        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )

        def _flux(**kwargs):
            async def _gen():
                for m in morceaux:
                    yield m
                if kwargs.get("fin"):
                    kwargs["fin"](raison)
            return _gen()

        rag_service.llm.generate_stream = MagicMock(side_effect=_flux)

    @staticmethod
    async def _evenements(rag_service, requete):
        import json

        return [json.loads(e) async for e in rag_service.ask_stream(requete)]

    @pytest.mark.asyncio
    async def test_reponse_coupee_signalee(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        from app.services.rag_service import MENTION_REPONSE_TRONQUEE

        self._preparer(rag_service, mock_db_session, mock_search_results,
                       ["Un militaire qui abuse ", "de sa position pour"], raison="MAX_TOKENS")

        evenements = await self._evenements(rag_service, sample_rag_request)

        texte = "".join(e["chunk"] for e in evenements)
        assert texte.endswith(MENTION_REPONSE_TRONQUEE)
        assert evenements[-1]["done"] is True

    @pytest.mark.asyncio
    async def test_flux_rompu_signale_autrement(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        """Coupee par la connexion, pas par sa longueur : la mention le dit."""
        from app.services.llm import FIN_INTERROMPUE
        from app.services.rag_service import MENTION_REPONSE_INTERROMPUE

        self._preparer(rag_service, mock_db_session, mock_search_results,
                       ["Un militaire qui abuse "], raison=FIN_INTERROMPUE)

        evenements = await self._evenements(rag_service, sample_rag_request)

        texte = "".join(e["chunk"] for e in evenements)
        assert texte.endswith(MENTION_REPONSE_INTERROMPUE)

    @pytest.mark.asyncio
    async def test_mention_dans_la_langue_de_la_question(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        """Une reponse en anglais ne se termine pas par une mention en francais."""
        requete = sample_rag_request.model_copy(update={"language": "en"})
        self._preparer(rag_service, mock_db_session, mock_search_results,
                       ["A soldier who abuses "], raison="MAX_TOKENS")

        evenements = await self._evenements(rag_service, requete)

        texte = "".join(e["chunk"] for e in evenements)
        assert texte.endswith("*(Answer cut short: maximum length reached. "
                              "Ask a more specific question to get the rest.)*")

    @pytest.mark.asyncio
    async def test_reponse_complete_sans_mention(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        self._preparer(rag_service, mock_db_session, mock_search_results, ["Réponse."])

        evenements = await self._evenements(rag_service, sample_rag_request)

        assert "".join(e["chunk"] for e in evenements) == "Réponse."

    @pytest.mark.asyncio
    async def test_meme_budget_et_reflexion_que_ask(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        self._preparer(rag_service, mock_db_session, mock_search_results, ["Réponse."])

        await self._evenements(rag_service, sample_rag_request)

        kwargs = rag_service.llm.generate_stream.call_args.kwargs
        assert kwargs["max_tokens"] == 8192
        assert kwargs["reflexion"] is None

    @pytest.mark.asyncio
    async def test_source_de_repli_comme_ask(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session, monkeypatch
    ):
        """Une reponse juste sans citation reconnue garde ses sources."""
        self._preparer(rag_service, mock_db_session, mock_search_results,
                       ["Les dirigeants répondent de leurs fautes de gestion."])
        monkeypatch.setattr(rag_service, "_peut_fabriquer_une_source", lambda *a: True)

        evenements = await self._evenements(rag_service, sample_rag_request)

        assert evenements[-1]["sources"], "la source de repli manquait au flux"

    @pytest.mark.asyncio
    async def test_document_ouvert_et_article_absent(
        self, rag_service, sample_rag_request, mock_db_session, monkeypatch
    ):
        """Sur la page d'un document, la reponse directe « article absent »."""
        absent = RAGResponse(
            answer="Ce document ne contient pas d'article 99.", confidence=1.0, sources=[],
            session_id="s", retrieval_time_ms=0, generation_time_ms=0, total_time_ms=0,
            persona="citoyen",
        )

        async def _recuperation(requete, conversation):
            return [], absent

        monkeypatch.setattr(rag_service, "_retrieve_and_merge_context", _recuperation)
        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        requete = sample_rag_request.model_copy(update={"law_id": 7, "question": "Que dit l'article 99 ?"})

        evenements = await self._evenements(rag_service, requete)

        assert len(evenements) == 1
        assert evenements[0]["chunk"] == "Ce document ne contient pas d'article 99."
        assert evenements[0]["done"] is True

    @pytest.mark.asyncio
    async def test_quota_epuise_a_son_code(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        from app.services.gemini_service import GeminiQuotaError

        self._preparer(rag_service, mock_db_session, mock_search_results, [])

        def _quota(**kwargs):
            async def _gen():
                raise GeminiQuotaError("Quota de generation epuise. Reessaie dans 60 secondes.")
                yield  # fait de _gen un generateur asynchrone
            return _gen()

        rag_service.llm.generate_stream = MagicMock(side_effect=_quota)

        evenements = await self._evenements(rag_service, sample_rag_request)

        assert evenements[-1]["done"] is True
        assert evenements[-1]["error_code"] == "quota"


# ==================== UN PROFIL PAR CONVERSATION ====================

class TestUnProfilParConversation:
    """
    Le profil choisi pilote la reponse, et une conversation garde le sien :
    un autre profil en ouvre une nouvelle, l'ancienne n'est pas touchee.
    """

    @staticmethod
    def _existante(mock_db_session, persona="citoyen", user_id=None):
        existante = Conversation(
            id=7, session_id="sid-ancienne", persona=persona, language="fr", user_id=user_id
        )
        trouvee = MagicMock()
        trouvee.unique.return_value.scalar_one_or_none.return_value = existante
        messages = MagicMock()
        messages.scalars.return_value.all.return_value = []
        mock_db_session.execute.side_effect = [trouvee, messages]
        return existante

    @pytest.mark.asyncio
    async def test_meme_profil_la_conversation_continue(self, rag_service, mock_db_session):
        existante = self._existante(mock_db_session, "citoyen")

        conversation, _ = await rag_service._load_or_create_conversation("sid-ancienne", "citoyen", "fr")

        assert conversation is existante
        mock_db_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_autre_profil_ouvre_une_nouvelle_conversation(self, rag_service, mock_db_session):
        rag_service.user_id = 5
        existante = self._existante(mock_db_session, "citoyen")

        conversation, historique = await rag_service._load_or_create_conversation(
            "sid-ancienne", "avocat", "fr"
        )

        assert conversation is not existante
        assert conversation.persona == "avocat"
        assert conversation.session_id != "sid-ancienne"
        assert historique == []
        # L'ancienne n'est ni modifiee ni adoptee par le compte.
        assert existante.persona == "citoyen"
        assert existante.user_id is None
        mock_db_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_la_conversation_d_autrui_reste_refusee(self, rag_service, mock_db_session):
        """L'appartenance passe AVANT le changement de profil : 404, pas de conversation neuve."""
        from app.services.rag_service import ConversationInterdite

        rag_service.user_id = 1
        self._existante(mock_db_session, "citoyen", user_id=2)

        with pytest.raises(ConversationInterdite):
            await rag_service._load_or_create_conversation("sid-ancienne", "avocat", "fr")
        mock_db_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_le_profil_atteint_le_prompt_et_le_dernier_evenement(
        self, rag_service, sample_rag_request, mock_search_results, mock_db_session
    ):
        TestFluxAligneSurAsk._preparer(rag_service, mock_db_session, mock_search_results, ["Réponse."])
        requete = sample_rag_request.model_copy(update={"persona": "avocat"})

        evenements = await TestFluxAligneSurAsk._evenements(rag_service, requete)

        systeme = rag_service.llm.generate_stream.call_args.kwargs["system"]
        assert systeme == get_system_prompt("avocat", "fr")
        assert evenements[-1]["persona"] == "avocat"
        # Le profil n'est porte que par l'evenement final.
        assert all("persona" not in e for e in evenements[:-1])

    @pytest.mark.asyncio
    async def test_sans_profil_le_profil_general(
        self, rag_service, mock_search_results, mock_db_session
    ):
        TestFluxAligneSurAsk._preparer(rag_service, mock_db_session, mock_search_results, ["Réponse."])
        requete = RAGRequest(question="Quelle est la responsabilité des dirigeants ?", language="fr")

        evenements = await TestFluxAligneSurAsk._evenements(rag_service, requete)

        assert requete.persona == "general"
        assert rag_service.llm.generate_stream.call_args.kwargs["system"] == get_system_prompt(
            "general", "fr"
        )
        assert evenements[-1]["persona"] == "general"

    @pytest.mark.asyncio
    async def test_sans_resultat_une_seule_conversation_et_le_conseil_du_profil(
        self, rag_service, mock_db_session
    ):
        """
        `_handle_no_results` rechargeait la conversation : sans session_id, il
        en creait une deuxieme, vide. Et il conseillait a un avocat de
        consulter un avocat.
        """
        rag_service.search_service.search.return_value = MagicMock(
            results=[], chunks=[], search_time_ms=1
        )
        rag_service._fallback_search = AsyncMock(return_value=[])
        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        )
        requete = RAGRequest(question="Une question sans réponse ici ?", persona="avocat", language="fr")

        reponse = await rag_service.ask(requete)

        ajoutees = [c.args[0] for c in mock_db_session.add.call_args_list]
        assert sum(isinstance(objet, Conversation) for objet in ajoutees) == 1
        assert "Consultez un avocat" not in reponse.answer
        assert reponse.persona == "avocat"
