"""
Tests de l'etage 2 du re-ranking (notation par le modele).

Ce qui est verifie ici n'est pas la qualite du classement — elle depend du
modele — mais le fait qu'AUCUNE defaillance ne remonte. Le point d'appel est
dans le chemin RAG, dont les exceptions deviennent des HTTP 500 : une reponse
JSON mal formee ne doit jamais couter une reponse a l'utilisateur.
"""

import asyncio
import json

import pytest

from app.schemas.search import ChunkResult
from app.services.reranker import rerank_with_llm


def _chunks(n=3):
    return [
        ChunkResult(
            article_id=i, law_id=1, number=str(i),
            content=f"Contenu de l'article {i}.", excerpt=f"Contenu {i}",
            reference="LOI-2024-001", law_title="Code OHADA",
            # Scores decroissants mais bornes : relevance_score est declare
            # ge=0.0, un lot de 30 passerait sous zero avec un pas de 0.1.
            relevance_score=max(0.0, 1.0 - i * 0.02),
            rerank_score=max(0.0, 1.0 - i * 0.02),
        )
        for i in range(1, n + 1)
    ]


class _LLM:
    """Doublure : renvoie ce qu'on lui dit, ou leve ce qu'on lui dit."""

    def __init__(self, payload=None, raises=None, delay=0.0):
        self.payload = payload
        self.raises = raises
        self.delay = delay
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        return {"response": self.payload}


class TestHappyPath:

    @pytest.mark.asyncio
    async def test_reorders_according_to_the_scores(self):
        chunks = _chunks(3)
        llm = _LLM(json.dumps({"scores": [
            {"id": 0, "score": 1.0},
            {"id": 1, "score": 2.0},
            {"id": 2, "score": 10.0},
        ]}))

        ranked = await rerank_with_llm("question", chunks, llm=llm)

        # Melange moitie-moitie : les notes du modele (1, 2, 10) dominent
        # l'ecart d'etage 1 (0.98 / 0.96 / 0.94).
        assert [c.article_id for c in ranked] == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_asks_for_structured_output(self):
        llm = _LLM(json.dumps({"scores": []}))

        await rerank_with_llm("question", _chunks(2), llm=llm)

        assert llm.calls[0]["response_mime_type"] == "application/json"
        assert "scores" in llm.calls[0]["response_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_only_the_head_is_sent(self):
        chunks = _chunks(30)
        llm = _LLM(json.dumps({"scores": [{"id": 0, "score": 5.0}]}))

        ranked = await rerank_with_llm("question", chunks, llm=llm, top_n=5)

        assert llm.calls[0]["prompt"].count("[4]") == 1
        assert "[5]" not in llm.calls[0]["prompt"]
        # La queue est conservee telle quelle, jamais perdue.
        assert len(ranked) == 30

    @pytest.mark.asyncio
    async def test_blend_keeps_a_strong_stage_one_chunk(self):
        """
        Une note basse du modele ne doit pas enterrer un chunk sur lequel tous
        les signaux lexicaux s'accordent.
        """
        chunks = _chunks(2)
        chunks[0] = chunks[0].model_copy(update={"rerank_score": 1.0})
        chunks[1] = chunks[1].model_copy(update={"rerank_score": 0.0})
        llm = _LLM(json.dumps({"scores": [
            {"id": 0, "score": 4.0},
            {"id": 1, "score": 6.0},
        ]}))

        ranked = await rerank_with_llm("question", chunks, llm=llm)

        # 0.5*1.0 + 0.5*0.4 = 0.70  contre  0.5*0.0 + 0.5*0.6 = 0.30
        assert ranked[0].article_id == 1


class TestDegradation:
    """Chaque mode de defaillance renvoie l'ordre recu, sans lever."""

    @pytest.mark.asyncio
    async def test_exception(self):
        chunks = _chunks(3)
        llm = _LLM(raises=RuntimeError("API indisponible"))

        assert await rerank_with_llm("q", chunks, llm=llm) == chunks

    @pytest.mark.asyncio
    async def test_timeout(self):
        chunks = _chunks(3)
        llm = _LLM(json.dumps({"scores": []}), delay=0.5)

        assert await rerank_with_llm("q", chunks, llm=llm, timeout=0.05) == chunks

    @pytest.mark.asyncio
    async def test_empty_response(self):
        chunks = _chunks(3)

        assert await rerank_with_llm("q", chunks, llm=_LLM("")) == chunks

    @pytest.mark.asyncio
    async def test_non_json_response(self):
        chunks = _chunks(3)
        llm = _LLM("Voici mon classement : d'abord l'article 3...")

        assert await rerank_with_llm("q", chunks, llm=llm) == chunks

    @pytest.mark.asyncio
    async def test_unknown_ids_are_dropped(self):
        chunks = _chunks(2)
        llm = _LLM(json.dumps({"scores": [
            {"id": 99, "score": 10.0},
            {"id": -1, "score": 10.0},
        ]}))

        assert await rerank_with_llm("q", chunks, llm=llm) == chunks

    @pytest.mark.asyncio
    async def test_out_of_range_scores_are_clamped(self):
        chunks = _chunks(2)
        llm = _LLM(json.dumps({"scores": [
            {"id": 0, "score": -50.0},
            {"id": 1, "score": 900.0},
        ]}))

        ranked = await rerank_with_llm("q", chunks, llm=llm)

        assert all(0.0 <= c.rerank_score <= 1.0 for c in ranked)
        assert ranked[0].article_id == 2

    @pytest.mark.asyncio
    async def test_malformed_entries_are_skipped(self):
        """
        Les entrees illisibles sont ignorees, les autres appliquees.

        Un chunk envoye mais non note conserve son score d'etage 1 : le modele
        qui l'omet n'est pas une raison de le degrader, l'omission relevant
        plus souvent du formatage que du jugement.
        """
        chunks = _chunks(2)
        llm = _LLM(json.dumps({"scores": [
            {"id": "pas un entier", "score": 5.0},
            {"score": 5.0},
            {"id": 1, "score": 10.0},
        ]}))

        ranked = await rerank_with_llm("q", chunks, llm=llm)

        assert len(ranked) == 2
        # Le chunk 0, non note, garde son score d'etage 1 (0.98).
        by_id = {c.article_id: c.rerank_score for c in ranked}
        assert by_id[1] == pytest.approx(0.98)
        # Le chunk 1 est melange : 0.5 * 0.96 + 0.5 * 1.0
        assert by_id[2] == pytest.approx(0.98, abs=1e-3)

    @pytest.mark.asyncio
    async def test_no_llm_is_a_no_op(self):
        chunks = _chunks(3)

        assert await rerank_with_llm("q", chunks, llm=None) == chunks

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        assert await rerank_with_llm("q", [], llm=_LLM("{}")) == []
