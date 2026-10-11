"""
PATCH /api/v1/auth/me : le profil du compte, en libre-service.

Le seul champ modifiable. La route est publique dans le sens ou tout compte
connecte y accede : elle ne doit donc JAMAIS accepter role, email ou mot de
passe, sinon n'importe quel compte se donnerait l'administration.

Usage:
    pytest tests/test_api/test_profil_du_compte_routes.py -v
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.user import User

ME = "/api/v1/auth/me"


@pytest.mark.asyncio
async def test_sans_jeton_401(client: AsyncClient):
    assert (await client.patch(ME, json={"persona": "avocat"})).status_code == 401


@pytest.mark.asyncio
async def test_le_profil_est_enregistre_puis_relu(client: AsyncClient, auth_headers):
    r = await client.patch(ME, json={"persona": "avocat"}, headers=auth_headers)

    assert r.status_code == 200
    assert r.json()["persona"] == "avocat"
    assert (await client.get(ME, headers=auth_headers)).json()["persona"] == "avocat"


@pytest.mark.asyncio
async def test_les_alias_sont_ramenes_a_la_forme_canonique(client: AsyncClient, auth_headers):
    r = await client.patch(ME, json={"persona": "Etudiant"}, headers=auth_headers)

    assert r.json()["persona"] == "étudiant"


@pytest.mark.asyncio
async def test_null_efface_et_vide_ne_change_rien(client: AsyncClient, auth_headers):
    await client.patch(ME, json={"persona": "entrepreneur"}, headers=auth_headers)

    assert (await client.patch(ME, json={}, headers=auth_headers)).json()["persona"] == "entrepreneur"
    assert (await client.patch(ME, json={"persona": None}, headers=auth_headers)).json()["persona"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", ["general", "astronaute", ""])
async def test_un_refus_ou_une_valeur_inconnue_422(client: AsyncClient, auth_headers, persona):
    """Un refus de repondre ne s'enregistre jamais sur le compte."""
    r = await client.patch(ME, json={"persona": persona}, headers=auth_headers)

    assert r.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("corps", [
    {"persona": "avocat", "role": "superadmin"},
    {"email": "autre@essai.cm"},
    {"password": "NouveauMotDePasse1"},
    {"is_active": False},
])
async def test_aucun_autre_champ(client: AsyncClient, auth_headers, db_session, test_user, corps):
    """L'escalade de privileges que UserUpdate aurait permise."""
    user_id = test_user.id

    r = await client.patch(ME, json=corps, headers=auth_headers)

    assert r.status_code == 422
    db_session.expire_all()
    compte = (await db_session.execute(select(User).where(User.id == user_id))).scalar_one()
    assert compte.role == "user"
    assert compte.persona is None


@pytest.mark.asyncio
async def test_la_connexion_rend_le_profil(client: AsyncClient, auth_headers, test_user):
    """C'est ce qui fait suivre le profil d'un appareil a l'autre."""
    await client.patch(ME, json={"persona": "avocat"}, headers=auth_headers)

    r = await client.post("/api/v1/auth/login/json", json={
        "email": test_user.email, "password": "MotDePasseTest123",
    })

    assert r.status_code == 200
    assert r.json()["persona"] == "avocat"
