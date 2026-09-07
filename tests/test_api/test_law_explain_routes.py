"""
Tests de POST /api/v1/laws/{law_id}/explain-article.

Ce que ces tests protegent en priorite : la TRADUCTION des echecs Gemini en
codes HTTP. Un quota epuise rendu en 500 ferait cesser le client de reessayer,
et une erreur inattendue recopiee dans `detail` ferait fuiter une cle sur une
route publique. La meme correspondance existe dans rag.py et n'y est couverte
par aucun test — c'est le trou que ceux-ci comblent.

Usage:
    pytest tests/test_api/test_law_explain_routes.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.schemas.law import ArticleExplanationResponse
from app.services.explanation_service import (
    ArticleNotFoundError,
    ExplanationError,
    ExplanationOverloadedError,
    ExplanationQuotaError,
)

CHEMIN = "/api/v1/laws/1/explain-article"


@pytest.fixture
def sync_client():
    """Client sans base : reserve aux tests dont le service est double."""
    return TestClient(app)


@pytest.fixture
def explication():
    return ArticleExplanationResponse(
        law_id=1,
        article_id=42,
        number="35",
        explanation="**En clair**\n\nCet article soumet toute transaction à autorisation.",
        language="fr",
        persona="citoyen",
        resolved_from="database",
        generation_time_ms=1234,
    )


def _double(mock_classe, *, retour=None, erreur=None):
    """Branche une doublure d'ExplanationService sur la classe patchee."""
    service = AsyncMock()
    service.explain = AsyncMock(return_value=retour, side_effect=erreur)
    mock_classe.return_value = service
    return service


# ==================== CAS NOMINAL ====================


class TestExplainSucces:
    @patch("app.api.routes.laws.ExplanationService")
    def test_200_et_forme_de_la_reponse(self, mock_classe, sync_client, explication):
        _double(mock_classe, retour=explication)

        reponse = sync_client.post(CHEMIN, json={"number": "35", "language": "fr"})

        assert reponse.status_code == 200
        corps = reponse.json()
        assert corps["explanation"].startswith("**En clair**")
        assert corps["persona"] == "citoyen"
        assert corps["resolved_from"] == "database"
        assert corps["article_id"] == 42
        assert corps["number"] == "35"

    @patch("app.api.routes.laws.ExplanationService")
    def test_langue_et_extrait_transmis(self, mock_classe, sync_client, explication):
        service = _double(mock_classe, retour=explication)

        sync_client.post(
            CHEMIN,
            json={"number": "35", "language": "en", "excerpt": "Texte affiché."},
        )

        kwargs = service.explain.await_args.kwargs
        assert kwargs["law_id"] == 1
        assert kwargs["number"] == "35"
        assert kwargs["language"] == "en"
        assert kwargs["excerpt"] == "Texte affiché."

    @patch("app.api.routes.laws.ExplanationService")
    def test_le_client_ne_choisit_pas_le_persona(
        self, mock_classe, sync_client, explication
    ):
        """
        Le ton est fixe cote serveur.

        Un `persona` dans le corps doit rester sans effet : sinon n'importe qui
        changerait la voix du produit depuis son navigateur.
        """
        service = _double(mock_classe, retour=explication)

        reponse = sync_client.post(
            CHEMIN, json={"number": "35", "persona": "avocat"}
        )

        assert reponse.status_code == 200
        assert "persona" not in service.explain.await_args.kwargs
        assert reponse.json()["persona"] == "citoyen"


# ==================== ERREURS ====================


class TestExplainErreurs:
    @patch("app.api.routes.laws.ExplanationService")
    def test_404_article_absent(self, mock_classe, sync_client):
        _double(mock_classe, erreur=ArticleNotFoundError("Article 404 introuvable."))

        reponse = sync_client.post(CHEMIN, json={"number": "404"})

        assert reponse.status_code == 404
        assert "introuvable" in reponse.json()["detail"]

    @patch("app.api.routes.laws.ExplanationService")
    def test_429_porte_retry_after(self, mock_classe, sync_client):
        """Le client doit savoir QUAND réessayer, pas seulement que ça a raté."""
        _double(mock_classe, erreur=ExplanationQuotaError("Quota epuise."))

        reponse = sync_client.post(CHEMIN, json={"number": "35"})

        assert reponse.status_code == 429
        assert reponse.headers["Retry-After"] == "60"

    @patch("app.api.routes.laws.ExplanationService")
    def test_503_porte_retry_after(self, mock_classe, sync_client):
        _double(mock_classe, erreur=ExplanationOverloadedError("Sature."))

        reponse = sync_client.post(CHEMIN, json={"number": "35"})

        assert reponse.status_code == 503
        assert reponse.headers["Retry-After"] == "10"

    @patch("app.api.routes.laws.ExplanationService")
    def test_500_sur_echec_de_generation(self, mock_classe, sync_client):
        _double(mock_classe, erreur=ExplanationError("Réponse vide du modèle."))

        reponse = sync_client.post(CHEMIN, json={"number": "35"})

        assert reponse.status_code == 500

    @patch("app.api.routes.laws.ExplanationService")
    def test_500_inattendu_ne_fuit_rien(self, mock_classe, sync_client):
        """
        Route publique : une erreur inattendue peut porter une clé.

        `detail` doit rester générique, quoi que dise l'exception d'origine.
        """
        secret = "AIzaSyFAUSSECLEDETEST123456789"
        _double(mock_classe, erreur=RuntimeError(f"boom {secret}"))

        reponse = sync_client.post(CHEMIN, json={"number": "35"})

        assert reponse.status_code == 500
        assert reponse.json()["detail"] == "Erreur interne du serveur"
        assert secret not in reponse.text


class TestExplainValidation:
    @pytest.mark.parametrize(
        "corps",
        [
            {},                                        # numéro manquant
            {"number": ""},                            # numéro vide
            {"number": "   "},                         # numéro blanc
            {"number": "35", "language": "de"},        # langue hors corpus
            {"number": "x" * 65},                      # au-delà de String(64)
            {"number": "35", "excerpt": "x" * 12_001},  # extrait démesuré
        ],
    )
    def test_422(self, sync_client, corps):
        assert sync_client.post(CHEMIN, json=corps).status_code == 422


# ==================== AVEC LA VRAIE BASE ====================


class TestExplainSurLaBase:
    """Le seul cas qui traverse le vrai service : la loi n'existe pas."""

    @pytest.mark.asyncio
    async def test_404_loi_absente(self, client: AsyncClient):
        reponse = await client.post(
            "/api/v1/laws/999999/explain-article", json={"number": "1"}
        )

        assert reponse.status_code == 404
