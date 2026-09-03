"""
Tests des routes d'authentification.

Aucun test ne couvrait l'authentification : elle n'existait pas (les endpoints
d'administration passaient par des stubs renvoyant un dictionnaire en dur).
"""

import pytest
from httpx import AsyncClient

PASSWORD = "MotDePasseTest123"


class TestLogin:
    """Connexion."""

    @pytest.mark.asyncio
    async def test_login_json_succeeds(self, client: AsyncClient, test_admin_user):
        r = await client.post(
            "/api/v1/auth/login/json",
            json={"email": test_admin_user.email, "password": PASSWORD},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["access_token"]
        assert body["token_type"] == "bearer"
        assert body["role"] == "admin"

    @pytest.mark.asyncio
    async def test_login_form_succeeds(self, client: AsyncClient, test_admin_user):
        """Le formulaire OAuth2 alimente le bouton Authorize de /docs."""
        r = await client.post(
            "/api/v1/auth/login",
            data={"username": test_admin_user.email, "password": PASSWORD},
        )
        assert r.status_code == 200
        assert r.json()["access_token"]

    @pytest.mark.asyncio
    async def test_wrong_password_is_rejected(self, client: AsyncClient, test_user):
        r = await client.post(
            "/api/v1/auth/login/json",
            json={"email": test_user.email, "password": "mauvais"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_email_gives_same_message(
        self, client: AsyncClient, test_user
    ):
        """
        Message identique pour email inconnu et mot de passe faux.

        Les distinguer permettrait d'énumérer les comptes existants.
        """
        wrong_pw = await client.post(
            "/api/v1/auth/login/json",
            json={"email": test_user.email, "password": "mauvais"},
        )
        unknown = await client.post(
            "/api/v1/auth/login/json",
            json={"email": "personne@nulle-part.cm", "password": "mauvais"},
        )
        assert wrong_pw.status_code == unknown.status_code == 401
        assert wrong_pw.json()["detail"] == unknown.json()["detail"]

    @pytest.mark.asyncio
    async def test_last_login_is_recorded(
        self, client: AsyncClient, test_admin_user, db_session
    ):
        assert test_admin_user.last_login_at is None
        await client.post(
            "/api/v1/auth/login/json",
            json={"email": test_admin_user.email, "password": PASSWORD},
        )
        await db_session.refresh(test_admin_user)
        assert test_admin_user.last_login_at is not None


class TestMe:
    """Identité associée au jeton."""

    @pytest.mark.asyncio
    async def test_me_requires_a_token(self, client: AsyncClient):
        assert (await client.get("/api/v1/auth/me")).status_code == 401

    @pytest.mark.asyncio
    async def test_me_rejects_a_bogus_token(self, client: AsyncClient):
        r = await client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer pas-un-jeton"}
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_me_returns_the_account(
        self, client: AsyncClient, test_user, auth_headers
    ):
        r = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["email"] == test_user.email


class TestAuthorization:
    """Contrôle d'accès par rôle."""

    # (méthode, chemin) — endpoints qui ne doivent jamais être publics
    PROTECTED = [
        ("get", "/api/v1/admin/users"),
        ("post", "/api/v1/search/reindex"),
        ("get", "/api/v1/search/stats"),
        ("post", "/api/v1/upload/cleanup"),
        ("get", "/api/v1/batch-upload/status"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method,path", PROTECTED)
    async def test_anonymous_access_is_refused(self, client: AsyncClient, method, path):
        r = await getattr(client, method)(path)
        assert r.status_code == 401, f"{path} accessible sans jeton"

    @pytest.mark.asyncio
    async def test_plain_user_cannot_reach_admin(
        self, client: AsyncClient, auth_headers
    ):
        """Un jeton valide mais sans privilège doit donner 403, pas 200."""
        r = await client.get("/api/v1/admin/users", headers=auth_headers)
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_reaches_admin(self, client: AsyncClient, admin_headers):
        r = await client.get("/api/v1/admin/users", headers=admin_headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_no_protected_route_lacks_a_guard(self):
        """
        Filet de sécurité : parcourt les routes montées et échoue si l'une des
        préfixes protégés n'a aucune dépendance d'authentification.

        C'est ce test qui rattrapera le prochain endpoint ajouté sans garde.
        """
        from fastapi.routing import APIRoute

        from app.main import app

        prefixes = ("/api/v1/admin/", "/api/v1/ocr/")
        exempt = {"/api/v1/admin/health"}

        unguarded = []
        for route in _iter_routes(app):
            if not isinstance(route, APIRoute):
                continue
            if route.path in exempt or not route.path.startswith(prefixes):
                continue
            names = {d.name for d in route.dependant.dependencies}
            if not any("current" in (n or "") for n in names):
                unguarded.append(f"{sorted(route.methods)} {route.path}")

        assert not unguarded, f"routes sans garde : {unguarded}"


def _iter_routes(app):
    """Parcourt les routes, y compris celles imbriquées dans les routeurs inclus."""
    stack = list(app.routes)
    while stack:
        route = stack.pop()
        yield route
        stack.extend(getattr(route, "routes", []) or [])
