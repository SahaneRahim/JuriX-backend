"""
Tests du contexte envoye au modele et des citations.

Le contexte ne contenait que highlights['content'], soit les 400 premiers
caracteres du texte de la LOI : le modele devait citer des articles dont il
n'avait jamais lu le texte. Et format_matched_articles, seule fonction censee
ajouter ce texte, lisait un attribut inexistant (`snippet` au lieu de
`content_snippet`) : sa branche ne s'est jamais declenchee.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.search import ChunkResult
from app.services.prompts import (
    CONTEXT_MAX_CHARS,
    CONTEXT_TRUNCATION_MARK,
    build_context_string,
)
from app.services.rag_service import RAGService, _normalize_article_number


def _chunk(article_id: int, number: str, content: str, **kwargs) -> ChunkResult:
    defaults = dict(
        law_id=kwargs.pop("law_id", 1),
        reference=kwargs.pop("reference", "LOI-2024-001"),
        law_title=kwargs.pop("law_title", "Code OHADA"),
        relevance_score=kwargs.pop("relevance_score", 0.9),
    )
    return ChunkResult(
        article_id=article_id,
        number=number,
        content=content,
        excerpt=content[:120],
        **defaults,
        **kwargs,
    )


@pytest.fixture
def rag_service():
    service = RAGService.__new__(RAGService)
    service.db = MagicMock()
    service.llm = AsyncMock()
    service.search_service = MagicMock()
    return service


class TestContextBuilding:

    def test_context_contains_article_number_and_full_content(self):
        chunk = _chunk(
            161, "161",
            "Les dirigeants sociaux sont responsables des fautes commises "
            "dans l'exercice de leurs fonctions envers la société et les tiers.",
            article_title="Responsabilité des dirigeants",
            section="TITRE III",
            page_number=47,
        )

        context = build_context_string([chunk])

        assert "Article 161" in context
        assert "Responsabilité des dirigeants" in context
        assert "TITRE III" in context
        assert "Page 47" in context
        # Le CONTENU, pas seulement un extrait de 400 caracteres de la loi.
        assert "envers la société et les tiers" in context

    def test_context_respects_global_budget(self):
        chunks = [
            _chunk(i, str(i), "A" * 5000) for i in range(1, 21)
        ]

        context = build_context_string(chunks)

        assert len(context) <= CONTEXT_MAX_CHARS * 1.1
        assert "Article 1" in context

    def test_first_chunk_survives_even_if_oversized(self):
        """Un contexte tronque vaut mieux qu'un contexte vide."""
        chunks = [_chunk(1, "1", "B" * 100_000)]

        context = build_context_string(chunks)

        assert context
        assert "Article 1" in context
        assert CONTEXT_TRUNCATION_MARK in context

    def test_truncation_falls_on_a_paragraph_boundary(self):
        content = ("Paragraphe un." + "\n\n" + "x" * 200) * 60
        chunks = [_chunk(1, "1", content)]

        context = build_context_string(chunks)

        assert CONTEXT_TRUNCATION_MARK in context

    def test_multilingual_context_gets_a_translation_note(self):
        chunks = [
            _chunk(1, "1", "Texte en français", language="fr"),
            _chunk(2, "1", "English text", language="en", law_id=2,
                   reference="LAW-2024-002", law_title="Commercial Code"),
        ]

        context = build_context_string(chunks)

        assert "traduis" in context.lower()

    def test_empty_chunks_give_empty_context(self):
        assert build_context_string([]) == ""


class TestArticlePinning:

    def test_requested_article_moves_to_the_front(self, rag_service):
        chunks = [
            _chunk(1, "1", "Premier"),
            _chunk(2, "42", "Quarante-deux"),
        ]

        pinned = rag_service._pin_requested_article("Que dit l'article 42 ?", chunks)

        assert pinned[0].number == "42"
        assert len(pinned) == 2

    def test_pinning_does_not_touch_the_database(self, rag_service):
        """
        Le texte est deja dans le contexte : aller le rechercher ouvrait un
        moteur synchrone de plus a chaque question.
        """
        chunks = [_chunk(1, "1", "Premier"), _chunk(2, "42", "Quarante-deux")]

        rag_service._pin_requested_article("article 42", chunks)

        rag_service.db.execute.assert_not_called()

    def test_unknown_article_leaves_order_unchanged(self, rag_service):
        chunks = [_chunk(1, "1", "Premier"), _chunk(2, "2", "Deux")]

        assert rag_service._pin_requested_article("article 99", chunks) == chunks


class TestArticleNumberNormalisation:

    @pytest.mark.parametrize("raw,expected", [
        ("1", "1"),
        ("1er", "1"),
        ("PREMIER", "1"),
        ("Article 5", "5"),
        ("  12 ", "12"),
    ])
    def test_normalisation(self, raw, expected):
        assert _normalize_article_number(raw) == expected

    def test_does_not_confuse_1_with_10(self):
        """
        Les numeros etaient parfois compares par inclusion : "1" correspondait
        alors a "10", "11" et "100", et la mauvaise citation etait rattachee.
        """
        assert _normalize_article_number("1") != _normalize_article_number("10")


class TestCitations:

    def test_citation_binds_article_id(self, rag_service):
        chunks = [_chunk(161, "161", "Les dirigeants sont responsables.")]
        answer = "Selon l'article 161 du Code OHADA, les dirigeants sont responsables."

        citations = rag_service._extract_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0].article_id == 161
        assert citations[0].law_id == 1

    def test_citation_excerpt_comes_from_the_chunk(self, rag_service):
        chunks = [_chunk(161, "161", "Contenu unique et reconnaissable.")]
        answer = "Selon l'article 161 du Code OHADA."

        citations = rag_service._extract_citations(answer, chunks)

        assert "reconnaissable" in citations[0].excerpt
        rag_service.db.execute.assert_not_called()

    def test_citation_rejects_an_article_absent_from_the_context(self, rag_service):
        """
        La validation ne portait que sur le titre de la loi, jamais sur le
        numero : un article invente par le modele passait.
        """
        chunks = [_chunk(161, "161", "Les dirigeants sont responsables.")]
        answer = "Selon l'article 999 du Code OHADA, tout est permis."

        assert rag_service._extract_citations(answer, chunks) == []

    def test_sources_fallback_uses_chunk_content(self, rag_service):
        chunks = [_chunk(161, "161", "Contenu de repli.")]

        sources = rag_service._create_sources_from_results(chunks, "Une question")

        assert len(sources) == 1
        assert sources[0].article_id == 161
        assert "repli" in sources[0].excerpt


class TestEventLoop:
    """Le chemin RAG ne doit rien appeler de bloquant sur la boucle."""

    @pytest.mark.asyncio
    async def test_gemini_generate_runs_off_the_event_loop(self, monkeypatch):
        """
        GeminiService.generate est `async def` mais appelle un client
        SYNCHRONE : sans deport, la boucle d'evenements etait gelee pendant
        tout l'aller-retour avec le modele, a chaque question.
        """
        import asyncio
        import threading

        from app.services.gemini_service import GeminiService

        service = GeminiService.__new__(GeminiService)
        service.model_name = "test-model"
        service.SYSTEM_INSTRUCTION = "sys"

        loop_thread = threading.get_ident()
        seen = {}

        class _Response:
            text = "réponse"

        class _Models:
            def generate_content(self, **kwargs):
                seen["thread"] = threading.get_ident()
                return _Response()

        service.client = type("C", (), {"models": _Models()})()

        result = await service.generate(prompt="Question ?")

        assert result["response"] == "réponse"
        assert seen["thread"] != loop_thread
