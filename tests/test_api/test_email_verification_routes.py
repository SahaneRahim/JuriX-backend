"""
Verification d'adresse a l'inscription.

CE QUE CE FICHIER PROTEGE. Brancher un envoi de courriel sur `/signup` cree deux
facons de casser une route qui marchait :

1. RENDRE L'INSCRIPTION IMPOSSIBLE SANS CLE. Le depot tourne sans cle Brevo en
   developpement, en test, et chez quiconque clone le projet. Si l'absence de
   configuration faisait echouer `/signup`, plus personne ne pourrait creer de
   compte — et les centaines de tests existants tomberaient tous.
   → `test_sans_cle_brevo_l_inscription_est_inchangee`

2. RENDRE L'INSCRIPTION TRIBUTAIRE D'UN TIERS. Une panne de Brevo — 500, delai
   depasse, quota — ne doit pas empecher de creer un compte. Le compte existe,
   la session est ouverte, l'utilisateur demandera un renvoi.
   → `test_inscription_reussit_meme_si_brevo_echoue`

Le troisieme test couvre l'invariant de la table unique, dans le sens
symetrique de celui de `test_password_reset_routes.py` : un jeton de
REINITIALISATION ne doit pas verifier une adresse.

Usage:
    pytest tests/test_api/test_email_verification_routes.py -v
"""

import re

import httpx
import pytest
import respx
from sqlalchemy import func, select

from app.core.config import settings
from app.models.email_token import VERIFICATION_ADRESSE, EmailToken
from app.models.user import User
from app.services.email_service import API_BREVO
from app.services.email_tokens_service import reinitialiser_letranglement_ip

SIGNUP = "/api/v1/auth/signup"
VERIFY = "/api/v1/auth/verify-email"


@pytest.fixture
def brevo(monkeypatch):
    monkeypatch.setattr(settings, "BREVO_API_KEY", "xkeysib-cle-de-test")
    monkeypatch.setattr(settings, "BREVO_SENDER_EMAIL", "expediteur@example.cm")
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://jurix.test")
    reinitialiser_letranglement_ip()
    from app.services import email_service

    email_service._budget["jour"] = None
    yield
    reinitialiser_letranglement_ip()


async def _inscrire(client, email="rahim@example.cm"):
    return await client.post(
        SIGNUP, json={"full_name": "Rahim Sahane", "email": email, "password": "MotDePasse1"}
    )


class TestSansConfiguration:
    @pytest.mark.asyncio
    @respx.mock
    async def test_sans_cle_brevo_l_inscription_est_inchangee(self, client, db_session):
        """
        LA PREUVE QUE RIEN NE BOUGE. Voir le point 1 du docstring.

        Aucune requete sortante, aucun jeton cree, et la reponse est celle
        d'avant : 201 avec un jeton de session.
        """
        assert settings.BREVO_API_KEY == ""
        route = respx.post(API_BREVO).mock(return_value=httpx.Response(201))

        r = await _inscrire(client)

        assert r.status_code == 201
        assert r.json()["access_token"]
        assert route.call_count == 0, "un POST est parti sans cle"
        combien = (
            await db_session.execute(select(func.count()).select_from(EmailToken))
        ).scalar()
        assert combien == 0, "un jeton a ete cree pour un message qui ne partira jamais"


class TestPanneDeLExpediteur:
    @pytest.mark.asyncio
    @respx.mock
    async def test_inscription_reussit_meme_si_brevo_echoue(self, client, brevo):
        """Voir le point 2 du docstring : le compte prime sur le courriel."""
        respx.post(API_BREVO).mock(return_value=httpx.Response(500))

        r = await _inscrire(client)

        assert r.status_code == 201, "une panne Brevo a fait echouer une inscription"
        assert r.json()["access_token"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_inscription_reussit_meme_si_brevo_est_injoignable(self, client, brevo):
        respx.post(API_BREVO).mock(side_effect=httpx.ConnectError("reseau coupe"))

        r = await _inscrire(client)

        assert r.status_code == 201


class TestParcoursNominal:
    @pytest.mark.asyncio
    @respx.mock
    async def test_inscription_envoie_un_lien_et_cree_un_jeton(self, client, db_session, brevo):
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m1"})
        )

        r = await _inscrire(client)

        assert r.status_code == 201
        assert route.call_count == 1
        corps = route.calls[-1].request.content.decode()
        assert "/verify-email/" in corps
        ligne = (
            await db_session.execute(
                select(EmailToken).where(EmailToken.purpose == VERIFICATION_ADRESSE)
            )
        ).scalar_one()
        assert ligne.consumed_at is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_lien_passe_is_verified_a_vrai(self, client, db_session, brevo):
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m1"})
        )
        donnees = (await _inscrire(client)).json()
        assert donnees["is_verified"] is False

        corps = route.calls[-1].request.content.decode()
        jeton = re.search(r"/verify-email/([A-Za-z0-9_-]+)", corps).group(1)

        r = await client.post(VERIFY, json={"token": jeton})

        assert r.status_code == 200
        db_session.expire_all()
        user = (
            await db_session.execute(select(User).where(User.id == donnees["id"]))
        ).scalar_one()
        assert user.is_verified is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_lien_ne_sert_qu_une_fois(self, client, brevo):
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m1"})
        )
        await _inscrire(client)
        corps = route.calls[-1].request.content.decode()
        jeton = re.search(r"/verify-email/([A-Za-z0-9_-]+)", corps).group(1)

        assert (await client.post(VERIFY, json={"token": jeton})).status_code == 200
        assert (await client.post(VERIFY, json={"token": jeton})).status_code == 400


class TestEtranglementDesInscriptions:
    @pytest.mark.asyncio
    @respx.mock
    async def test_inscrire_cent_adresses_n_envoie_pas_cent_messages(self, client, brevo):
        """
        LE TROU QUE CETTE GARDE FERME.

        L'etranglement par COMPTE ne protege rien sur ce chemin : chaque
        inscription cree un compte neuf, dont le compteur repart de zero. Sans
        garde par IP, inscrire des adresses toutes differentes enverrait autant
        de messages et epuiserait le quota quotidien — un plafond dur, partage
        entre tous les usages, qui empecherait ensuite le moindre courriel de
        reinitialisation d'arriver a qui que ce soit.

        L'inscription elle-meme n'est PAS bloquee : le compte est cree, la
        session est ouverte. Seul le message ne part pas.
        """
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m"})
        )

        codes = []
        for i in range(20):
            r = await _inscrire(client, f"vague{i}@example.cm")
            codes.append(r.status_code)

        assert codes == [201] * 20, "l'etranglement a fait echouer des inscriptions"
        assert route.call_count <= 10, (
            f"{route.call_count} messages pour 20 inscriptions : "
            "le quota quotidien peut etre epuise par un tiers"
        )


class TestRefus:
    @pytest.mark.parametrize("corps", [{}, {"token": "court"}, {"token": "x" * 40, "role": "admin"}])
    @pytest.mark.asyncio
    async def test_422(self, client, corps):
        assert (await client.post(VERIFY, json=corps)).status_code == 422

    @pytest.mark.asyncio
    async def test_jeton_inconnu_donne_400(self, client):
        r = await client.post(VERIFY, json={"token": "j" * 43})

        assert r.status_code == 400

    @pytest.mark.asyncio
    @respx.mock
    async def test_jeton_de_reinitialisation_refuse_par_la_verification(
        self, client, db_session, brevo
    ):
        """
        Le symetrique de `test_jeton_de_verification_refuse_par_la_reinitialisation`.

        La cloison de la table unique doit tenir dans les DEUX sens, sinon elle
        ne tient dans aucun.
        """
        import datetime

        from app.models.email_token import (
            REINITIALISATION_MOT_DE_PASSE,
            hachage,
            maintenant_naif,
            nouveau_jeton,
        )

        respx.post(API_BREVO).mock(return_value=httpx.Response(201, json={"messageId": "m"}))
        donnees = (await _inscrire(client)).json()

        jeton = nouveau_jeton()
        db_session.add(
            EmailToken(
                user_id=donnees["id"],
                purpose=REINITIALISATION_MOT_DE_PASSE,
                token_hash=hachage(jeton),
                expires_at=maintenant_naif() + datetime.timedelta(minutes=30),
                created_at=maintenant_naif(),
            )
        )
        await db_session.commit()

        r = await client.post(VERIFY, json={"token": jeton})

        assert r.status_code == 400
