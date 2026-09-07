"""
Tests de POST /api/v1/auth/google.

AUCUN APPEL RESEAU : `verifier_jeton_google` est double par monkeypatch. Le
vrai module fait un GET HTTPS vers Google pour ses certificats — en dependre
rendrait la suite lente et instable, et la ferait echouer hors ligne.

Le test le plus important est `test_client_id_vide_repond_503_sans_verifier` :
`verify_oauth2_token(..., audience=None)` DESACTIVE la verification de l'`aud`,
ce qui ferait accepter n'importe quel jeton Google emis pour n'importe quelle
application. La route doit donc refuser AVANT d'appeler la verification quand
l'identifiant client manque.

Usage:
    pytest tests/test_api/test_google_auth_routes.py -v
"""

import pytest
from google.auth import exceptions as google_exceptions
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import app.api.routes.auth as routes_auth
from app.core.config import settings
from app.models.user import User

CHEMIN = "/api/v1/auth/google"
JETON = "j" * 200  # `credential` est borne a 100 caracteres minimum


def claims(**surcharges):
    base = {
        "sub": "108234567890123456789",
        "email": "rahim@gmail.com",
        "email_verified": True,
        "name": "Rahim Sahane",
    }
    base.update(surcharges)
    return base


@pytest.fixture
def google_configure(monkeypatch):
    """Un identifiant client present, sinon la route refuse en 503."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "id-client-de-test.apps.googleusercontent.com")


@pytest.fixture
def google_ok(monkeypatch, google_configure):
    """Double la verification par un succes, et compte les appels."""
    appels = []

    async def _double(credential):
        appels.append(credential)
        return claims()

    monkeypatch.setattr(routes_auth, "verifier_jeton_google", _double)
    return appels


def double_avec(monkeypatch, resultat=None, erreur=None):
    appels = []

    async def _double(credential):
        appels.append(credential)
        if erreur is not None:
            raise erreur
        return resultat

    monkeypatch.setattr(routes_auth, "verifier_jeton_google", _double)
    return appels


class TestCreationEtLiaison:
    @pytest.mark.asyncio
    async def test_premiere_connexion_cree_le_compte(
        self, client: AsyncClient, db_session: AsyncSession, google_ok
    ):
        r = await client.post(CHEMIN, json={"credential": JETON})

        assert r.status_code == 200
        donnees = r.json()
        assert donnees["email"] == "rahim@gmail.com"
        assert donnees["full_name"] == "Rahim Sahane"
        assert donnees["role"] == "user"
        assert donnees["is_verified"] is True, "Google a affirme email_verified"

        user = (
            await db_session.execute(select(User).where(User.email == "rahim@gmail.com"))
        ).scalar_one()
        assert user.google_sub == claims()["sub"]
        assert user.hashed_password is None, "un compte Google n'a pas de mot de passe"

    @pytest.mark.asyncio
    async def test_seconde_connexion_ne_duplique_pas(
        self, client: AsyncClient, db_session: AsyncSession, google_ok
    ):
        await client.post(CHEMIN, json={"credential": JETON})
        await client.post(CHEMIN, json={"credential": JETON})

        combien = (
            await db_session.execute(
                select(func.count()).select_from(User).where(User.email == "rahim@gmail.com")
            )
        ).scalar()
        assert combien == 1

    @pytest.mark.asyncio
    async def test_liaison_conserve_le_mot_de_passe(
        self, client: AsyncClient, db_session: AsyncSession, google_ok
    ):
        """
        Un compte cree par mot de passe qui se connecte ensuite par Google gagne
        une porte, il n'en perd pas. Google prouve le controle de la boite —
        preuve plus forte que notre inscription, qui n'envoie aucun e-mail.
        """
        inscription = await client.post(
            "/api/v1/auth/signup",
            json={
                "full_name": "Rahim",
                "email": "rahim@gmail.com",
                "password": "MotDePasse1",
            },
        )
        assert inscription.status_code == 201

        r = await client.post(CHEMIN, json={"credential": JETON})
        assert r.status_code == 200

        combien = (
            await db_session.execute(
                select(func.count()).select_from(User).where(User.email == "rahim@gmail.com")
            )
        ).scalar()
        assert combien == 1, "un second compte a ete cree au lieu d'une liaison"

        db_session.expire_all()
        user = (
            await db_session.execute(select(User).where(User.email == "rahim@gmail.com"))
        ).scalar_one()
        assert user.google_sub is not None
        assert user.hashed_password is not None, "le mot de passe a ete perdu"

        # Et il fonctionne toujours.
        connexion = await client.post(
            "/api/v1/auth/login/json",
            json={"email": "rahim@gmail.com", "password": "MotDePasse1"},
        )
        assert connexion.status_code == 200

    @pytest.mark.asyncio
    async def test_inscription_sur_un_compte_google_refusee(
        self, client: AsyncClient, db_session: AsyncSession, google_ok
    ):
        """
        L'ASYMETRIE. L'inverse serait une prise de controle : sans verification
        d'adresse de notre cote, quiconque connait l'adresse Gmail de quelqu'un
        pourrait se greffer un mot de passe sur son compte.
        """
        await client.post(CHEMIN, json={"credential": JETON})

        r = await client.post(
            "/api/v1/auth/signup",
            json={
                "full_name": "Usurpateur",
                "email": "rahim@gmail.com",
                "password": "MotDePasse1",
            },
        )

        assert r.status_code == 409
        assert "Google" not in r.json()["detail"], "le message revele le fournisseur"
        db_session.expire_all()
        user = (
            await db_session.execute(select(User).where(User.email == "rahim@gmail.com"))
        ).scalar_one()
        assert user.hashed_password is None, "un mot de passe a ete greffe"

    @pytest.mark.asyncio
    async def test_adresse_normalisee(self, client: AsyncClient, monkeypatch, google_configure):
        double_avec(monkeypatch, resultat=claims(email="  Rahim@GMAIL.com  "))

        r = await client.post(CHEMIN, json={"credential": JETON})

        assert r.json()["email"] == "rahim@gmail.com"


class TestRefus:
    @pytest.mark.asyncio
    async def test_client_id_vide_repond_503_sans_verifier(
        self, client: AsyncClient, monkeypatch, db_session: AsyncSession
    ):
        """
        LE TEST LE PLUS IMPORTANT DU FICHIER.

        `verify_oauth2_token(..., audience=None)` DESACTIVE la verification de
        l'`aud` : n'importe quel jeton Google, emis pour n'importe quelle
        application tierce, serait alors accepte — contournement complet de
        l'authentification. La garde doit donc refuser AVANT d'appeler la
        bibliotheque.

        On espionne ici `verify_oauth2_token` LUI-MEME, et non
        `verifier_jeton_google` : doubler ce dernier remplacerait justement la
        garde qu'on veut eprouver, et le test passerait sans rien prouver.
        """
        from app.core import google_identity

        monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "")
        appels = []

        def _espion(*a, **k):
            appels.append(a)
            return claims()

        monkeypatch.setattr(google_identity.id_token, "verify_oauth2_token", _espion)

        r = await client.post(CHEMIN, json={"credential": JETON})

        assert r.status_code == 503
        assert appels == [], "la verification a ete appelee sans identifiant client"
        combien = (await db_session.execute(select(func.count()).select_from(User))).scalar()
        assert combien == 0, "un compte a ete cree sans verification"

    @pytest.mark.asyncio
    async def test_jeton_invalide_401(self, client: AsyncClient, monkeypatch, google_configure):
        from app.core.google_identity import JetonGoogleInvalide

        double_avec(monkeypatch, erreur=JetonGoogleInvalide("adresse non verifiee"))

        r = await client.post(CHEMIN, json={"credential": JETON})

        assert r.status_code == 401
        # Message unique : detailler la cause aiderait qui cherche a fabriquer
        # un jeton acceptable.
        assert "verifiee" not in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_google_injoignable_503(self, client: AsyncClient, monkeypatch, google_configure):
        """
        NOTRE reseau a echoue, pas celui de l'utilisateur. Un 401 l'accuserait
        d'un defaut qui n'est pas le sien.
        """
        from app.core.google_identity import GoogleInjoignable

        double_avec(monkeypatch, erreur=GoogleInjoignable("certificats injoignables"))

        r = await client.post(CHEMIN, json={"credential": JETON})

        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "30"

    @pytest.mark.asyncio
    async def test_compte_desactive_403(
        self, client: AsyncClient, db_session: AsyncSession, google_ok
    ):
        await client.post(CHEMIN, json={"credential": JETON})
        user = (
            await db_session.execute(select(User).where(User.email == "rahim@gmail.com"))
        ).scalar_one()
        user.is_active = False
        await db_session.commit()

        r = await client.post(CHEMIN, json={"credential": JETON})

        assert r.status_code == 403

    @pytest.mark.parametrize(
        "corps", [{}, {"credential": "trop-court"}, {"credential": "x" * 5000}, {"credential": JETON, "role": "superadmin"}]
    )
    @pytest.mark.asyncio
    async def test_422(self, client: AsyncClient, google_configure, corps):
        assert (await client.post(CHEMIN, json=corps)).status_code == 422


class TestVerificationReelle:
    """La fonction de verification elle-meme, sans passer par la route."""

    @pytest.mark.asyncio
    async def test_email_non_verifie_refuse(self, monkeypatch):
        """
        `verify_oauth2_token` ne verifie PAS `email_verified`. Sans ce controle,
        un compte Google dont l'adresse n'a jamais ete prouvee pourrait se lier
        a un compte JuriX existant portant la meme adresse.
        """
        from app.core import google_identity

        monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "id-de-test")
        monkeypatch.setattr(
            google_identity.id_token,
            "verify_oauth2_token",
            lambda *a, **k: claims(email_verified=False),
        )

        with pytest.raises(google_identity.JetonGoogleInvalide):
            await google_identity.verifier_jeton_google(JETON)

    @pytest.mark.asyncio
    async def test_erreur_de_transport_distincte(self, monkeypatch):
        from app.core import google_identity

        monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "id-de-test")

        def _lever(*a, **k):
            raise google_exceptions.TransportError("reseau")

        monkeypatch.setattr(google_identity.id_token, "verify_oauth2_token", _lever)

        with pytest.raises(google_identity.GoogleInjoignable):
            await google_identity.verifier_jeton_google(JETON)

    @pytest.mark.asyncio
    async def test_jeton_illisible(self, monkeypatch):
        from app.core import google_identity

        monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "id-de-test")

        def _lever(*a, **k):
            raise ValueError("Token expired")

        monkeypatch.setattr(google_identity.id_token, "verify_oauth2_token", _lever)

        with pytest.raises(google_identity.JetonGoogleInvalide):
            await google_identity.verifier_jeton_google(JETON)
