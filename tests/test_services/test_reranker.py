"""
Tests du re-ranking, etage 1 (traits lexicaux).

Le classement issu de la fusion RRF ne connait que deux signaux. Ces tests
verifient que l'etage 1 rattrape les cas ou ils se trompent, et surtout qu'il
reste une fonction pure : la fusion, elle, ecrase relevance_score en place, et
c'est ce qui rendait impossible d'attribuer un ecart au re-ranking.
"""

import pytest

from app.schemas.search import ChunkResult
from app.services.reranker import rerank_chunks, score_chunk


def _chunk(article_id, number, content, score=0.5, **kwargs):
    return ChunkResult(
        article_id=article_id,
        law_id=kwargs.pop("law_id", 1),
        number=number,
        content=content,
        excerpt=content[:120],
        reference=kwargs.pop("reference", "LOI-2024-001"),
        law_title=kwargs.pop("law_title", "Code OHADA"),
        relevance_score=score,
        **kwargs,
    )


class TestArticleNumber:

    def test_promotes_the_requested_article(self):
        """
        "article 161 du Code OHADA" est la forme dominante des questions, et le
        seul cas ou la bonne reponse est connaissable sans semantique.
        """
        chunks = [
            _chunk(1, "12", "Dispositions relatives aux dirigeants sociaux.", 0.95),
            _chunk(2, "45", "Autres dispositions sur la gestion.", 0.90),
            _chunk(3, "161", "Les dirigeants sont responsables de leur gestion.", 0.40),
        ]

        ranked = rerank_chunks("Que dit l'article 161 du Code OHADA ?", chunks)

        assert ranked[0].article_id == 3

    def test_does_not_confuse_161_with_16(self):
        chunks = [
            _chunk(1, "16", "Article seize.", 0.9),
            _chunk(2, "161", "Article cent soixante et un.", 0.4),
        ]

        ranked = rerank_chunks("article 161", chunks)

        assert ranked[0].number == "161"

    def test_ordinal_forms_match(self):
        chunks = [
            _chunk(1, "5", "Article cinq.", 0.9),
            _chunk(2, "1er", "Article premier de la loi.", 0.3),
        ]

        ranked = rerank_chunks("que dit l'article 1 ?", chunks)

        assert ranked[0].number == "1er"


class TestLexicalSignals:

    def test_promotes_exact_phrase_over_higher_fusion_score(self):
        chunks = [
            _chunk(1, "1", "La gestion des sociétés obéit à des règles.", 0.90),
            _chunk(2, "2", "La responsabilité des dirigeants est engagée en cas de faute.", 0.60),
        ]

        ranked = rerank_chunks("responsabilité des dirigeants", chunks)

        assert ranked[0].article_id == 2

    def test_penalises_boilerplate_execution_articles(self):
        """
        52,6 % du corpus sont des listes nominatives dont l'article d'execution
        est identique d'un decret a l'autre : il s'encode au meme endroit et
        sature le haut du classement.
        """
        chunks = [
            _chunk(1, "2", "Le présent décret sera enregistré, publié au Journal Officiel.", 0.95),
            _chunk(2, "1", "Les dirigeants sociaux répondent des fautes de gestion "
                           "commises dans l'exercice de leurs fonctions, individuellement "
                           "ou solidairement selon le cas, envers la société ou les tiers.", 0.70),
        ]

        ranked = rerank_chunks("responsabilité des dirigeants pour faute de gestion", chunks)

        assert ranked[0].article_id == 2

    def test_accent_insensitive(self):
        chunks = [
            _chunk(1, "1", "Texte sans rapport aucun.", 0.8),
            _chunk(2, "2", "La responsabilité civile des dirigeants sociaux.", 0.5),
        ]

        ranked = rerank_chunks("responsabilite civile", chunks)

        assert ranked[0].article_id == 2

    def test_law_title_match_helps(self):
        chunks = [
            _chunk(1, "1", "Dispositions générales.", 0.6,
                   law_id=1, reference="LOI-001", law_title="Code du travail"),
            _chunk(2, "1", "Dispositions générales.", 0.6,
                   law_id=2, reference="LOI-002", law_title="Code pénal camerounais"),
        ]

        ranked = rerank_chunks("que dit le code pénal ?", chunks)

        assert ranked[0].law_id == 2


class TestPurity:

    def test_does_not_mutate_the_input(self):
        chunks = [
            _chunk(1, "1", "Premier article.", 0.8),
            _chunk(2, "2", "Deuxième article.", 0.4),
        ]
        before = [(c.article_id, c.relevance_score, c.rerank_score) for c in chunks]

        rerank_chunks("une question", chunks)

        assert [(c.article_id, c.relevance_score, c.rerank_score) for c in chunks] == before

    def test_output_is_a_permutation_of_the_input(self):
        chunks = [_chunk(i, str(i), f"Contenu {i}.", 0.5) for i in range(1, 8)]

        ranked = rerank_chunks("une question", chunks)

        assert len(ranked) == len(chunks)
        assert {c.fusion_key for c in ranked} == {c.fusion_key for c in chunks}

    def test_retrieval_score_is_preserved(self):
        """
        Le score de recuperation reste lisible : sans lui, impossible de dire si
        un ecart vient de la recherche ou du re-ranking.
        """
        chunks = [_chunk(1, "1", "Contenu.", 0.42)]

        ranked = rerank_chunks("question", chunks)

        assert ranked[0].relevance_score == 0.42
        assert ranked[0].rerank_score is not None

    def test_scores_stay_in_range(self):
        chunks = [
            _chunk(1, "161", "Les dirigeants sont responsables.", 1.0),
            _chunk(2, "2", "x", 0.0),
        ]

        ranked = rerank_chunks("article 161 responsabilité des dirigeants", chunks)

        for chunk in ranked:
            assert 0.0 <= chunk.rerank_score <= 1.0

    def test_empty_input(self):
        assert rerank_chunks("question", []) == []

    def test_uniform_batch_keeps_original_order(self):
        """Lot homogene : rien ne departage, l'ordre d'entree doit survivre."""
        chunks = [_chunk(i, str(i), "Contenu identique.", 0.5) for i in range(1, 5)]

        ranked = rerank_chunks("aucun rapport", chunks)

        assert [c.article_id for c in ranked] == [1, 2, 3, 4]


class TestScoring:

    def test_score_is_deterministic(self):
        chunk = _chunk(1, "1", "Les dirigeants sont responsables.", 0.5)

        assert score_chunk(chunk, "responsabilité") == score_chunk(chunk, "responsabilité")
