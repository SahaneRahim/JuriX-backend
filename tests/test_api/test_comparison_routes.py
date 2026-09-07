"""
Tests de POST /api/v1/compare.

Comme pour l'explication d'article, ce qui est protege en priorite est la
TRADUCTION des echecs en codes HTTP : un quota rendu en 500 ferait cesser le
client de reessayer, et une erreur inattendue recopiee dans `detail` ferait
fuiter une cle sur une route publique.

Usage:
    pytest tests/test_api/test_comparison_routes.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.schemas.comparison import (
    ComparisonCell,
    ComparisonResponse,
    ComparisonRow,
    SourceRef,
)
from app.services.comparison_service import (
    ComparisonError,
    ComparisonOverloadedError,
    ComparisonQuotaError,
    NoContextError,
)

CHEMIN = "/api/v1/compare"


@pytest.fixture
def sync_client():
    return TestClient(app)


@pytest.fixture
def comparaison():
    source = SourceRef(
        article_id=7, law_id=16, law_title="Loi portant Code Minier",
        reference="LOI-2016-017", number="33", page_number=9,
        content="Le permis de recherche est delivre pour trois ans.",
    )
    return ComparisonResponse(
        subject_a="permis de recherche",
        subject_b="permis d'exploitation",
        language="fr",
        rows=[ComparisonRow(
            criterion="Duree et renouvellement",
            a=ComparisonCell(value="Trois ans", sources=[source]),
            b=ComparisonCell(value="Vingt ans", sources=[]),
        )],
        key_differences=["La duree differe."],
        blind_spots=["Les sanctions ne sont pas couvertes."],
        unmatched_citations=["9999"],
        retrieval_time_ms=120,
        generation_time_ms=4300,
    )


def _double(classe, *, retour=None, erreur=None):
    service = AsyncMock()
    service.compare = AsyncMock(return_value=retour, side_effect=erreur)
    classe.return_value = service
    return service


class TestCompareSucces:
    @patch("app.api.routes.comparison.ComparisonService")
    def test_200_et_forme(self, classe, sync_client, comparaison):
        _double(classe, retour=comparaison)

        r = sync_client.post(CHEMIN, json={
            "subject_a": "permis de recherche", "subject_b": "permis d'exploitation"
        })

        assert r.status_code == 200
        corps = r.json()
        ligne = corps["rows"][0]
        assert ligne["a"]["sources"][0]["number"] == "33"
        # Le texte integral voyage : c'est lui qu'on deplie sous la cellule.
        assert ligne["a"]["sources"][0]["content"].startswith("Le permis")
        assert ligne["b"]["sources"] == []
        assert corps["unmatched_citations"] == ["9999"]

    @patch("app.api.routes.comparison.ComparisonService")
    def test_options_transmises(self, classe, sync_client, comparaison):
        service = _double(classe, retour=comparaison)

        sync_client.post(CHEMIN, json={
            "subject_a": "SARL", "subject_b": "SA", "language": "en",
            "criteria": ["Capital", "Gerance"], "law_id": 16, "top_k": 12,
        })

        kw = service.compare.await_args.kwargs
        assert kw["language"] == "en"
        assert kw["criteria"] == ["Capital", "Gerance"]
        assert kw["law_id"] == 16
        assert kw["top_k"] == 12


class TestCompareErreurs:
    @patch("app.api.routes.comparison.ComparisonService")
    def test_404_sujet_sans_matiere(self, classe, sync_client):
        _double(classe, erreur=NoContextError("Aucun texte trouve pour « x »."))

        r = sync_client.post(CHEMIN, json={"subject_a": "x y", "subject_b": "z w"})

        assert r.status_code == 404
        assert "Aucun texte" in r.json()["detail"]

    @patch("app.api.routes.comparison.ComparisonService")
    def test_429_porte_retry_after(self, classe, sync_client):
        _double(classe, erreur=ComparisonQuotaError("Quota epuise."))

        r = sync_client.post(CHEMIN, json={"subject_a": "aa", "subject_b": "bb"})

        assert r.status_code == 429
        assert r.headers["Retry-After"] == "60"

    @patch("app.api.routes.comparison.ComparisonService")
    def test_503_porte_retry_after(self, classe, sync_client):
        _double(classe, erreur=ComparisonOverloadedError("Sature."))

        r = sync_client.post(CHEMIN, json={"subject_a": "aa", "subject_b": "bb"})

        assert r.status_code == 503
        assert r.headers["Retry-After"] == "10"

    @patch("app.api.routes.comparison.ComparisonService")
    def test_500_echec_de_generation(self, classe, sync_client):
        _double(classe, erreur=ComparisonError("Reponse incomplete."))

        r = sync_client.post(CHEMIN, json={"subject_a": "aa", "subject_b": "bb"})

        assert r.status_code == 500

    @patch("app.api.routes.comparison.ComparisonService")
    def test_500_inattendu_ne_fuit_rien(self, classe, sync_client):
        secret = "AIzaSyFAUSSECLEDETEST123456789"
        _double(classe, erreur=RuntimeError(f"boom {secret}"))

        r = sync_client.post(CHEMIN, json={"subject_a": "aa", "subject_b": "bb"})

        assert r.status_code == 500
        assert r.json()["detail"] == "Erreur interne du serveur"
        assert secret not in r.text


class TestCompareValidation:
    @pytest.mark.parametrize("corps", [
        {},                                                    # sujets manquants
        {"subject_a": "a"},                                    # un seul sujet
        {"subject_a": "a", "subject_b": "bb"},                 # trop court
        {"subject_a": "aa", "subject_b": "bb", "language": "de"},
        {"subject_a": "aa", "subject_b": "bb", "top_k": 99},   # hors bornes
        {"subject_a": "aa", "subject_b": "bb", "criteria": []},
        {"subject_a": "   ", "subject_b": "bb"},               # blanc
    ])
    def test_422(self, sync_client, corps):
        assert sync_client.post(CHEMIN, json=corps).status_code == 422


class TestCompareSurLaBase:
    """Le seul cas qui traverse le vrai service : corpus vide."""

    @pytest.mark.asyncio
    async def test_404_corpus_vide(self, client: AsyncClient):
        r = await client.post(CHEMIN, json={
            "subject_a": "zzzqqq inexistant", "subject_b": "wwwyyy inexistant",
        })

        assert r.status_code == 404
