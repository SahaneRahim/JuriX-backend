"""
Tests de l'API de classification.

L'ancienne route `GET /api/v1/classifier/categories` renvoyait un dictionnaire
code en dur qui contredisait `GET /api/v1/categories`, servi depuis la base.
Deux listes de categories concurrentes sur la meme API garantissaient qu'un
client se fie a la mauvaise ; ces tests figent sa disparition.

Usage:
    pytest tests/test_api/test_classifier_routes.py -v
"""

import pytest
from httpx import AsyncClient

from app.services.legal_domain_classifier import CANONICAL_DOMAINS


class TestClassifyEndpoint:
    @pytest.mark.asyncio
    async def test_classifies_from_the_title_alone(self, client: AsyncClient, db_session):
        response = await client.post(
            "/api/v1/classifier/classify",
            json={"title": "Loi N°2015/019 portant Loi de finances pour l'exercice 2016"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["domain"] == "Finances Publiques et Fiscalité"
        assert body["source"] == "title"
        assert body["rule"].startswith("A5")

    @pytest.mark.asyncio
    async def test_category_id_is_resolved_from_the_table(self, client: AsyncClient, db_session):
        """
        L'identifiant rendu doit designer, en base, la ligne portant ce nom.
        C'est precisement ce que l'ancien code ne faisait pas : il rendait une
        position dans un dictionnaire Python.
        """
        from sqlalchemy import select

        from app.models.law import Category

        response = await client.post(
            "/api/v1/classifier/classify",
            json={"title": "Décret portant nomination d'un Inspecteur Général"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["domain"] == "Fonction Publique"

        name = (await db_session.execute(
            select(Category.name).where(Category.id == body["category_id"])
        )).scalar_one()
        assert name == body["domain"]

    @pytest.mark.asyncio
    async def test_domain_is_always_canonical(self, client: AsyncClient, db_session):
        response = await client.post(
            "/api/v1/classifier/classify", json={"title": "Texte quelconque"}
        )
        assert response.status_code == 200
        assert response.json()["domain"] in CANONICAL_DOMAINS

    @pytest.mark.asyncio
    async def test_empty_request_is_rejected(self, client: AsyncClient, db_session):
        response = await client.post(
            "/api/v1/classifier/classify", json={"title": "", "text": "   "}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_response_carries_the_deciding_rule(self, client: AsyncClient, db_session):
        """Sans le nom de la regle, une erreur de classement n'est pas diagnosticable."""
        response = await client.post(
            "/api/v1/classifier/classify", json={"title": "Loi portant Code de Procédure Pénale"}
        )
        body = response.json()
        assert body["rule"] == "A1:procedure-penale"


class TestRemovedEndpoint:
    @pytest.mark.asyncio
    async def test_hardcoded_categories_endpoint_is_gone(self, client: AsyncClient, db_session):
        response = await client.get("/api/v1/classifier/categories")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_categories_come_from_the_database(self, client: AsyncClient, db_session):
        """La seule liste de categories servie par l'API est celle de la base."""
        response = await client.get("/api/v1/categories")
        assert response.status_code == 200
        names = {row["name"] for row in response.json()}
        assert names == set(CANONICAL_DOMAINS)


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_lists_the_canonical_domains(self, client: AsyncClient):
        response = await client.get("/api/v1/classifier/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["canonical_domains"] == list(CANONICAL_DOMAINS)
