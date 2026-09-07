"""
Tests de POST /api/v1/auth/signup — l'inscription publique.

Le test qui compte le plus est `test_role_injecte_refuse` : `UserCreate` porte
un champ `role` fourni par le client, et reutiliser ce schema pour une route
publique aurait laisse n'importe qui s'inscrire en superadmin. Ce fichier
verifie que la porte est fermee PAR LE SCHEMA, pas seulement par une garde
imperative qu'un refactoring pourrait retirer sans bruit.

Le second est `test_email_majuscules_peut_se_connecter` : `_authenticate`
cherche l'adresse en minuscules alors qu'`admin.create_user` l'ecrivait telle
quelle — un compte cree avec une majuscule ne pouvait jamais se connecter. Ce
test aurait attrape ce defaut.

Usage:
    pytest tests/test_api/test_signup_routes.py -v
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

CHEMIN = "/api/v1/auth/signup"
MDP = "MotDePasse1"


def corps(**surcharges):
    base = {"full_name": "Rahim Sahane", "email": "rahim@jurix.cm", "password": MDP}
    base.update(surcharges)
    return base


class TestInscriptionNominale:
    @pytest.mark.asyncio
    async def test_creation_et_jeton_utilisable(self, client: AsyncClient):
        r = await client.post(CHEMIN, json=corps())

        assert r.status_code == 201
        donnees = r.json()
        assert donnees["email"] == "rahim@jurix.cm"
        assert donnees["full_name"] == "Rahim Sahane"
        assert donnees["role"] == "user"
        assert donnees["is_verified"] is False, "aucun e-mail envoye : rien n'est prouve"

        # Le jeton rendu ouvre bien la session.
        moi = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {donnees['access_token']}"},
        )
        assert moi.status_code == 200
        assert moi.json()["email"] == "rahim@jurix.cm"

    @pytest.mark.asyncio
    async def test_la_reponse_ne_fuit_aucun_secret(self, client: AsyncClient):
        r = await client.post(CHEMIN, json=corps())

        assert "hashed_password" not in r.text
        assert "google_sub" not in r.text

    @pytest.mark.asyncio
    async def test_nom_normalise(self, client: AsyncClient):
        r = await client.post(CHEMIN, json=corps(full_name="  Rahim   Sahane  "))

        assert r.json()["full_name"] == "Rahim Sahane"


class TestEscaladeDePrivilege:
    """Le fichier existe surtout pour cette classe."""

    @pytest.mark.asyncio
    async def test_role_injecte_refuse(self, client: AsyncClient, db_session: AsyncSession):
        """
        `{"role": "superadmin"}` doit produire un 422, PAS un silence.

        `SignupRequest` n'expose pas ce champ et porte `extra="forbid"` : la
        tentative laisse une trace dans les journaux au lieu d'etre ignoree
        sans que personne ne le sache jamais.
        """
        r = await client.post(CHEMIN, json=corps(role="superadmin"))

        assert r.status_code == 422
        # Et la preuve en base : aucun compte privilegie n'a ete cree.
        combien = (
            await db_session.execute(
                select(func.count()).select_from(User).where(User.role != "user")
            )
        ).scalar()
        assert combien == 0

    @pytest.mark.asyncio
    async def test_is_active_injecte_refuse(self, client: AsyncClient):
        assert (await client.post(CHEMIN, json=corps(is_active=False))).status_code == 422

    @pytest.mark.asyncio
    async def test_champ_inconnu_refuse(self, client: AsyncClient):
        """`extra="forbid"` : toute clé inattendue est rejetée, pas ignorée."""
        assert (await client.post(CHEMIN, json=corps(is_verified=True))).status_code == 422


class TestNormalisationEmail:
    @pytest.mark.asyncio
    async def test_email_majuscules_peut_se_connecter(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        """
        Le test qui aurait attrape le defaut d'`admin.create_user`.

        `_authenticate` cherche avec `email.lower().strip()`. Ecrire l'adresse
        telle quelle rendait le compte definitivement inconnectable.
        """
        r = await client.post(CHEMIN, json=corps(email="  Rahim@JuriX.CM  "))
        assert r.status_code == 201
        assert r.json()["email"] == "rahim@jurix.cm"

        for saisie in ("rahim@jurix.cm", "Rahim@JuriX.CM"):
            connexion = await client.post(
                "/api/v1/auth/login/json", json={"email": saisie, "password": MDP}
            )
            assert connexion.status_code == 200, f"connexion refusee pour {saisie!r}"

    @pytest.mark.asyncio
    async def test_doublon_en_casse_differente(self, client: AsyncClient):
        assert (await client.post(CHEMIN, json=corps())).status_code == 201

        r = await client.post(CHEMIN, json=corps(email="RAHIM@JURIX.CM"))

        assert r.status_code == 409
        # Message neutre : ne jamais reveler qu'il s'agit d'un compte Google.
        assert "Google" not in r.json()["detail"]


class TestUsernameDerive:
    @pytest.mark.asyncio
    async def test_point_dans_ladresse(self, client: AsyncClient):
        """
        `jean.dupont@x.cm` doit rendre 201, pas 500.

        `UserResponse` herite de `UserBase`, donc `validate_username` s'execute
        AUSSI a la sortie : un username portant un point franchirait l'insert
        puis ferait echouer la serialisation de la reponse.
        """
        r = await client.post(CHEMIN, json=corps(email="jean.dupont@jurix.cm"))

        assert r.status_code == 201, r.text
        assert "." not in r.json()["username"]

    @pytest.mark.asyncio
    async def test_adresse_tres_courte(self, client: AsyncClient):
        """`min_length=3` cote schema : le username doit etre complete."""
        r = await client.post(CHEMIN, json=corps(email="ab@jurix.cm"))

        assert r.status_code == 201
        assert len(r.json()["username"]) >= 3

    @pytest.mark.asyncio
    async def test_deux_adresses_meme_partie_locale(self, client: AsyncClient):
        a = await client.post(CHEMIN, json=corps(email="jean@a.cm"))
        b = await client.post(CHEMIN, json=corps(email="jean@b.cm"))

        assert a.status_code == 201 and b.status_code == 201
        assert a.json()["username"] != b.json()["username"]

    @pytest.mark.asyncio
    async def test_accents_deplies(self, client: AsyncClient):
        r = await client.post(CHEMIN, json=corps(email="josé@jurix.cm"))

        assert r.status_code == 201
        assert r.json()["username"].isascii()


class TestValidation:
    @pytest.mark.parametrize(
        "champ,valeur",
        [
            ("password", "motdepasse1"),   # sans majuscule
            ("password", "MOTDEPASSE1"),   # sans minuscule
            ("password", "MotDePasse"),    # sans chiffre
            ("password", "MotDeP1"),       # trop court
            ("email", "pas-une-adresse"),
            ("full_name", "R"),            # trop court
            ("full_name", "   "),          # blanc
        ],
    )
    @pytest.mark.asyncio
    async def test_422(self, client: AsyncClient, champ, valeur):
        assert (await client.post(CHEMIN, json=corps(**{champ: valeur}))).status_code == 422

    @pytest.mark.asyncio
    async def test_champs_manquants(self, client: AsyncClient):
        assert (await client.post(CHEMIN, json={"email": "a@b.cm"})).status_code == 422
