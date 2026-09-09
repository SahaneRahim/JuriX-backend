"""Tests de base pour vérifier setup."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_root():
    """Test root endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "JuriX API v2.1" in data["message"]
    assert data["status"] == "running"
    assert data["version"] == "2.1.0"


@pytest.mark.asyncio
async def test_health():
    """Test health endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_ne_touche_pas_la_base():
    """
    `/health` doit rester un littéral, sans le moindre accès à la base.

    C'est la cible du ping de maintien en éveil, appelé toutes les dix minutes,
    soit environ 4 300 fois par mois. Sur une base infogérée qui s'endort et
    facture le temps de calcul, chacun de ces appels la réveillerait. Le jour où
    quelqu'un ajoute ici un `health_check_db()` — la fonction existe — le coût
    changerait de nature sans que rien ne le signale.

    La sonde qui a le droit de réveiller la base est une seconde tâche, bien plus
    espacée, qui vise une vraie route de lecture.
    """
    from app.core import database

    class _BaseInterdite:
        def __call__(self, *a, **k):
            raise AssertionError("/health a ouvert une session vers la base")

    original = database.AsyncSessionLocal
    database.AsyncSessionLocal = _BaseInterdite()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    finally:
        database.AsyncSessionLocal = original

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_secret_key_par_defaut_refuse_le_demarrage_hors_developpement():
    """
    Le garde-fou de `lifespan` n'était couvert par aucun test.

    S'il disparaissait, une instance de production signerait ses jetons avec une
    clé publiée dans le dépôt : n'importe qui pourrait forger un jeton
    d'administration.
    """
    from contextlib import asynccontextmanager

    from app.core.config import settings
    from app.main import _DEV_SECRET_KEY, lifespan

    environnement = settings.ENVIRONMENT
    secret = settings.SECRET_KEY
    settings.ENVIRONMENT = "production"
    settings.SECRET_KEY = _DEV_SECRET_KEY
    try:
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            async with lifespan(app):
                pass
    finally:
        settings.ENVIRONMENT = environnement
        settings.SECRET_KEY = secret
