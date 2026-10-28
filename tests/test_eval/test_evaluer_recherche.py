"""
scripts/eval/evaluer_recherche.py rejoue la requete de la page de recherche.

Il sautait le re-ranking que SearchService.search() applique : il mesurait un
classement que la production ne sert pas, et n'aurait vu aucun changement qui
ne touche que lui — la borne des lignes niveau loi d'octobre 2026, notamment.
"""

import pytest

from app.schemas.search import SearchResponse
from scripts.eval import evaluer_recherche as module


class _Service:
    """Les deux etapes de search() que le script enchaine, sans base."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.recus = None

    async def _execute_search_by_mode(self, demande):
        return list(self.chunks)

    async def _build_search_response(self, demande, chunks, elapsed_ms):
        self.recus = chunks
        return SearchResponse(
            query=demande.query, mode=demande.mode, results=[], total=0, search_time_ms=0
        )


@pytest.fixture
def reclassement(monkeypatch):
    """Un re-ranking factice qui inverse l'ordre, et garde ses appels."""
    appels = []

    def _inverser(question, chunks):
        appels.append(question)
        return list(reversed(chunks))

    monkeypatch.setattr(module, "rerank_chunks", _inverser)
    return appels


@pytest.mark.asyncio
async def test_le_re_ranking_s_applique_comme_dans_search(monkeypatch, reclassement):
    monkeypatch.setattr(module.settings, "RERANK_ENABLED", True)
    service = _Service(["a", "b"])

    await module.rejouer(service, "bail commercial", "text")

    assert reclassement == ["bail commercial"]
    assert service.recus == ["b", "a"]


@pytest.mark.asyncio
async def test_sans_re_ranking_en_production_le_script_n_en_fait_pas(monkeypatch, reclassement):
    monkeypatch.setattr(module.settings, "RERANK_ENABLED", False)
    service = _Service(["a", "b"])

    await module.rejouer(service, "bail commercial", "text")

    assert reclassement == []
    assert service.recus == ["a", "b"]
