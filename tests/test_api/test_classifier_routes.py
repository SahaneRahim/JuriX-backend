"""
Tests de l'API de classification.

L'ancienne route `GET /api/v1/classifier/categories` renvoyait un dictionnaire
code en dur qui contredisait `GET /api/v1/categories`, servi depuis la base.
Deux listes de categories concurrentes sur la meme API garantissaient qu'un
client se fie a la mauvaise ; ces tests figent sa disparition.

`POST /classify` coute une requete Groq : elle est reservee aux
administrateurs, et dit 503 (avec Retry-After) quand le modele ne peut pas
repondre, au lieu d'inventer un domaine. Le classifieur est double : ces
tests portent sur la route, pas sur le modele.

Usage:
    pytest tests/test_api/test_classifier_routes.py -v
"""

import pytest
from httpx import AsyncClient

from app.main import app
from app.services.legal_domain_classifier import (
    CANONICAL_DOMAINS,
    FINANCES,
    FONCTION_PUBLIQUE,
    ClassementIndisponible,
    DomainResult,
    get_legal_domain_classifier,
)

URL = "/api/v1/classifier/classify"


class _Classeur:
    """Doublure : rend `domaine`, ou leve `erreur`."""

    def __init__(self, domaine=FINANCES, erreur=None):
        self.domaine = domaine
        self.erreur = erreur
        self.titres = []

    async def classify_async(self, title, content="", doc_type=None):
        self.titres.append(title)
        if self.erreur:
            raise self.erreur
        return DomainResult(
            domain=self.domaine, confidence=0.9, rule="groq:doublure", source="groq",
            runners_up=((FONCTION_PUBLIQUE, 0.72),),
        )

    def health_check(self):
        return {"status": "healthy", "domains": len(CANONICAL_DOMAINS)}


@pytest.fixture
def classeur():
    """Installe la doublure derriere la dependance de la route."""
    doublure = _Classeur()
    app.dependency_overrides[get_legal_domain_classifier] = lambda: doublure
    try:
        yield doublure
    finally:
        app.dependency_overrides.pop(get_legal_domain_classifier, None)


class TestAcces:
    async def test_anonyme_refuse(self, client: AsyncClient, classeur):
        response = await client.post(URL, json={"title": "Loi de finances"})

        assert response.status_code == 401
        assert classeur.titres == []

    async def test_utilisateur_sans_role_refuse(self, client: AsyncClient, auth_headers, classeur):
        response = await client.post(URL, json={"title": "Loi de finances"}, headers=auth_headers)

        assert response.status_code == 403
        assert classeur.titres == []


class TestClassify:
    async def test_domaine_et_identifiant_resolu_en_base(
        self, admin_client: AsyncClient, db_session, classeur
    ):
        """
        L'identifiant rendu doit designer, en base, la ligne portant ce nom.
        C'est precisement ce que l'ancien code ne faisait pas : il rendait une
        position dans un dictionnaire Python.
        """
        from sqlalchemy import select

        from app.models.law import Category

        response = await admin_client.post(
            URL, json={"title": "Loi N°2015/019 portant Loi de finances pour l'exercice 2016"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["domain"] == FINANCES
        assert body["rule"] == "groq:doublure"
        assert body["runners_up"] == [{"domain": FONCTION_PUBLIQUE, "score": 0.72}]
        name = (await db_session.execute(
            select(Category.name).where(Category.id == body["category_id"])
        )).scalar_one()
        assert name == body["domain"]

    async def test_requete_vide_refusee(self, admin_client: AsyncClient, classeur):
        response = await admin_client.post(URL, json={"title": "", "text": "   "})

        assert response.status_code == 422
        assert classeur.titres == []

    async def test_classement_indisponible_rend_503(self, admin_client: AsyncClient, classeur):
        """Jamais de domaine invente : l'ancien code rendait Droit Administratif."""
        classeur.erreur = ClassementIndisponible("quota Groq", retry_after=41.2, quota=True)

        response = await admin_client.post(URL, json={"title": "Loi de finances"})

        assert response.status_code == 503
        assert response.headers["retry-after"] == "42"

    async def test_503_sans_delai_connu(self, admin_client: AsyncClient, classeur):
        classeur.erreur = ClassementIndisponible("panne")

        response = await admin_client.post(URL, json={"title": "Loi de finances"})

        assert response.status_code == 503
        assert "retry-after" not in response.headers


class TestRemovedEndpoint:
    async def test_hardcoded_categories_endpoint_is_gone(self, client: AsyncClient, db_session):
        response = await client.get("/api/v1/classifier/categories")
        assert response.status_code == 404

    async def test_categories_come_from_the_database(self, client: AsyncClient, db_session):
        """La seule liste de categories servie par l'API est celle de la base."""
        response = await client.get("/api/v1/categories")
        assert response.status_code == 200
        names = {row["name"] for row in response.json()}
        assert names == set(CANONICAL_DOMAINS)


class TestHealth:
    async def test_health_lists_the_canonical_domains(self, client: AsyncClient):
        response = await client.get("/api/v1/classifier/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["canonical_domains"] == list(CANONICAL_DOMAINS)
