"""
Reinitialisation de mot de passe — le fichier de securite du courriel.

CE QU'IL PROTEGE. Cette fonctionnalite donne, par un simple message, le pouvoir
de changer le mot de passe d'un compte. Trois choses peuvent mal tourner, et
chacune a son test :

1. UN JETON VIVANT LISIBLE EN BASE. Si le jeton etait stocke en clair, toute
   lecture de la base — une sauvegarde qui traine, un acces en lecture accorde a
   un tiers, une injection SQL en lecture seule — livrerait de quoi prendre
   n'importe quel compte ayant une demande en cours.
   → `test_le_jeton_en_base_est_un_hachage_pas_le_jeton_du_lien`

2. UN JETON ACCEPTE HORS DE SON USAGE. La table est unique et porte une colonne
   `purpose`. Si une lecture oubliait la clause, un jeton de VERIFICATION
   D'ADRESSE serait accepte par la reinitialisation : quiconque recoit un
   courriel de confirmation pourrait changer le mot de passe du compte.
   → `test_jeton_de_verification_refuse_par_la_reinitialisation`

3. UN ORACLE D'ENUMERATION. Si la reponse differait entre adresse connue et
   adresse inconnue, cette route dirait au monde entier qui possede un compte.
   → `test_adresse_connue_et_inconnue_repondent_a_l_identique`

Et une quatrieme, propre au palier gratuit : le quota d'envoi est un plafond DUR
et partage. Sans etranglement, un attaquant l'epuise en quelques minutes et
personne ne recoit plus rien de la journee.
   → `test_letranglement_ip_ignore_le_premier_element_de_x_forwarded_for`

AUCUN RESEAU REEL : `respx` intercepte httpx et permet d'affirmer sur le CORPS
envoye a Brevo — c'est ainsi qu'on recupere le jeton en clair, comme le ferait
le destinataire.

Usage:
    pytest tests/test_api/test_password_reset_routes.py -v
"""

import hashlib
import re

import httpx
import pytest
import respx
from sqlalchemy import func, select

from app.core.config import settings
from app.models.email_token import (
    REINITIALISATION_MOT_DE_PASSE,
    VERIFICATION_ADRESSE,
    EmailToken,
    hachage,
    maintenant_naif,
    nouveau_jeton,
)
from app.models.user import User
from app.services.email_service import API_BREVO
from app.services.email_tokens_service import reinitialiser_letranglement_ip

FORGOT = "/api/v1/auth/password/forgot"
RESET = "/api/v1/auth/password/reset"
SIGNUP = "/api/v1/auth/signup"

MDP_ORIGINE = "MotDePasse1"
MDP_NOUVEAU = "NouveauMdp2"


@pytest.fixture
def brevo(monkeypatch):
    """Envoi configure, et chaque message intercepte plutot qu'expedie."""
    monkeypatch.setattr(settings, "BREVO_API_KEY", "xkeysib-cle-de-test")
    monkeypatch.setattr(settings, "BREVO_SENDER_EMAIL", "expediteur@example.cm")
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://jurix.test")
    reinitialiser_letranglement_ip()
    from app.services import email_service

    email_service._budget["jour"] = None
    yield
    reinitialiser_letranglement_ip()


@pytest.fixture
def envoi(brevo):
    """Route Brevo doublée, qui garde les corps envoyés."""
    with respx.mock:
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "msg-1"})
        )
        yield route


async def _inscrire(client, email: str = "rahim@example.cm") -> dict:
    r = await client.post(
        SIGNUP, json={"full_name": "Rahim Sahane", "email": email, "password": MDP_ORIGINE}
    )
    assert r.status_code == 201, r.text
    return r.json()


def _jeton_du_dernier_message(route, motif: str) -> str:
    """Extrait le jeton du lien, exactement comme le ferait le destinataire."""
    corps = route.calls[-1].request.content.decode()
    trouve = re.search(rf"{motif}/([A-Za-z0-9_-]+)", corps)
    assert trouve, f"aucun lien {motif} dans le message :\n{corps[:400]}"
    return trouve.group(1)


class TestSecretDuJeton:
    @pytest.mark.asyncio
    async def test_le_jeton_en_base_est_un_hachage_pas_le_jeton_du_lien(
        self, client, db_session, envoi
    ):
        """LE TEST LE PLUS IMPORTANT DU FICHIER. Voir le point 1 du docstring."""
        await _inscrire(client)
        envoi.reset()

        await client.post(FORGOT, json={"email": "rahim@example.cm"})
        jeton = _jeton_du_dernier_message(envoi, "/reset-password")

        ligne = (
            await db_session.execute(
                select(EmailToken).where(EmailToken.purpose == REINITIALISATION_MOT_DE_PASSE)
            )
        ).scalar_one()

        assert ligne.token_hash == hashlib.sha256(jeton.encode()).hexdigest()
        assert ligne.token_hash != jeton
        # Et le jeton en clair n'apparait dans AUCUNE colonne de la ligne.
        valeurs = [str(getattr(ligne, c.name)) for c in ligne.__table__.columns]
        assert jeton not in valeurs, "le jeton en clair est stocke quelque part"


class TestCloisonnementDesUsages:
    @pytest.mark.asyncio
    async def test_jeton_de_verification_refuse_par_la_reinitialisation(
        self, client, db_session, envoi
    ):
        """
        CE TEST JUSTIFIE LA TABLE UNIQUE. Voir le point 2 du docstring.

        Le jeton est pose a la main avec `purpose='verify_email'`, puis presente
        a la route de reinitialisation. Elle doit le refuser, et le mot de passe
        ne doit PAS avoir bouge.
        """
        donnees = await _inscrire(client)
        user_id = donnees["id"]

        jeton = nouveau_jeton()
        db_session.add(
            EmailToken(
                user_id=user_id,
                purpose=VERIFICATION_ADRESSE,  # <-- le mauvais usage
                token_hash=hachage(jeton),
                expires_at=maintenant_naif() + __import__("datetime").timedelta(hours=1),
                created_at=maintenant_naif(),
            )
        )
        await db_session.commit()

        r = await client.post(RESET, json={"token": jeton, "password": MDP_NOUVEAU})

        assert r.status_code == 400, "un jeton de verification a change un mot de passe"

        # Le mot de passe d'origine fonctionne toujours : rien n'a bouge.
        connexion = await client.post(
            "/api/v1/auth/login/json",
            json={"email": "rahim@example.cm", "password": MDP_ORIGINE},
        )
        assert connexion.status_code == 200

    @pytest.mark.asyncio
    async def test_jeton_de_reinitialisation_refuse_par_la_verification(
        self, client, db_session, envoi
    ):
        """Le symetrique : la cloison tient dans les deux sens."""
        await _inscrire(client)
        envoi.reset()
        await client.post(FORGOT, json={"email": "rahim@example.cm"})
        jeton = _jeton_du_dernier_message(envoi, "/reset-password")

        r = await client.post("/api/v1/auth/verify-email", json={"token": jeton})

        assert r.status_code == 400


class TestNonEnumeration:
    @pytest.mark.asyncio
    async def test_adresse_connue_et_inconnue_repondent_a_l_identique(
        self, client, db_session, envoi
    ):
        """Voir le point 3 du docstring. Meme statut, meme corps octet pour octet."""
        await _inscrire(client, "connue@example.cm")
        reinitialiser_letranglement_ip()

        connue = await client.post(FORGOT, json={"email": "connue@example.cm"})
        inconnue = await client.post(FORGOT, json={"email": "jamais-vue@example.cm"})

        assert connue.status_code == inconnue.status_code == 202
        assert connue.content == inconnue.content

        # Et l'adresse inconnue ne laisse AUCUNE trace exploitable.
        # Filtre sur le `purpose` : l'inscription cree, elle, un jeton de
        # VERIFICATION, qui n'a rien a voir avec ce qu'on mesure ici.
        reinitialisations = (
            await db_session.execute(
                select(func.count())
                .select_from(EmailToken)
                .where(EmailToken.purpose == REINITIALISATION_MOT_DE_PASSE)
            )
        ).scalar()
        assert reinitialisations == 1, "un jeton a ete cree pour une adresse inconnue"

    @pytest.mark.asyncio
    async def test_compte_desactive_repond_202_sans_rien_envoyer(
        self, client, db_session, envoi
    ):
        donnees = await _inscrire(client)
        user = (
            await db_session.execute(select(User).where(User.id == donnees["id"]))
        ).scalar_one()
        user.is_active = False
        await db_session.commit()
        envoi.reset()

        r = await client.post(FORGOT, json={"email": "rahim@example.cm"})

        assert r.status_code == 202
        assert envoi.call_count == 0

    @pytest.mark.asyncio
    async def test_compte_google_ne_recoit_aucun_jeton(self, client, db_session, envoi):
        """
        Un compte sans mot de passe n'a rien a reinitialiser.

        Il recoit quand meme un message — sans quoi l'ABSENCE de courriel
        distinguerait un compte Google d'une adresse inconnue — mais ce message
        ne porte aucun lien de reinitialisation, et aucun jeton n'est cree.
        """
        user = User(
            email="google@example.cm",
            username="googleuser",
            hashed_password=None,
            google_sub="1082345678901234567890",
            full_name="Compte Google",
            role="user",
            is_active=True,
            is_verified=True,
        )
        db_session.add(user)
        await db_session.commit()
        envoi.reset()

        r = await client.post(FORGOT, json={"email": "google@example.cm"})

        assert r.status_code == 202
        combien = (
            await db_session.execute(
                select(func.count())
                .select_from(EmailToken)
                .where(EmailToken.purpose == REINITIALISATION_MOT_DE_PASSE)
            )
        ).scalar()
        assert combien == 0, "un jeton a ete cree pour un compte sans mot de passe"
        corps = envoi.calls[-1].request.content.decode()
        assert "/reset-password/" not in corps, "un lien de reinitialisation a ete envoye"


class TestCycleDeVieDuJeton:
    @pytest.mark.asyncio
    async def test_le_nouveau_mot_de_passe_marche_et_l_ancien_non(
        self, client, db_session, envoi
    ):
        await _inscrire(client)
        envoi.reset()
        await client.post(FORGOT, json={"email": "rahim@example.cm"})
        jeton = _jeton_du_dernier_message(envoi, "/reset-password")

        r = await client.post(RESET, json={"token": jeton, "password": MDP_NOUVEAU})
        assert r.status_code == 200

        ancien = await client.post(
            "/api/v1/auth/login/json",
            json={"email": "rahim@example.cm", "password": MDP_ORIGINE},
        )
        nouveau = await client.post(
            "/api/v1/auth/login/json",
            json={"email": "rahim@example.cm", "password": MDP_NOUVEAU},
        )

        assert ancien.status_code == 401, "l'ancien mot de passe marche encore"
        assert nouveau.status_code == 200

    @pytest.mark.asyncio
    async def test_jeton_a_usage_unique(self, client, envoi):
        await _inscrire(client)
        envoi.reset()
        await client.post(FORGOT, json={"email": "rahim@example.cm"})
        jeton = _jeton_du_dernier_message(envoi, "/reset-password")

        assert (await client.post(RESET, json={"token": jeton, "password": MDP_NOUVEAU})).status_code == 200
        rejoue = await client.post(RESET, json={"token": jeton, "password": "EncoreAutre3"})

        assert rejoue.status_code == 400

    @pytest.mark.asyncio
    async def test_jeton_expire_refuse(self, client, db_session, envoi):
        import datetime

        donnees = await _inscrire(client)
        jeton = nouveau_jeton()
        db_session.add(
            EmailToken(
                user_id=donnees["id"],
                purpose=REINITIALISATION_MOT_DE_PASSE,
                token_hash=hachage(jeton),
                expires_at=maintenant_naif() - datetime.timedelta(minutes=1),
                created_at=maintenant_naif() - datetime.timedelta(hours=1),
            )
        )
        await db_session.commit()

        r = await client.post(RESET, json={"token": jeton, "password": MDP_NOUVEAU})

        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_la_reinitialisation_invalide_les_autres_jetons(
        self, client, db_session, envoi
    ):
        """
        Un lien plus ancien encore valide serait une seconde porte ouverte
        pendant trente minutes — y compris pour qui aurait intercepte le premier.
        """
        import datetime

        donnees = await _inscrire(client)
        vieux = nouveau_jeton()
        db_session.add(
            EmailToken(
                user_id=donnees["id"],
                purpose=REINITIALISATION_MOT_DE_PASSE,
                token_hash=hachage(vieux),
                expires_at=maintenant_naif() + datetime.timedelta(minutes=30),
                created_at=maintenant_naif() - datetime.timedelta(minutes=10),
            )
        )
        await db_session.commit()
        envoi.reset()
        reinitialiser_letranglement_ip()

        await client.post(FORGOT, json={"email": "rahim@example.cm"})
        recent = _jeton_du_dernier_message(envoi, "/reset-password")
        assert (await client.post(RESET, json={"token": recent, "password": MDP_NOUVEAU})).status_code == 200

        r = await client.post(RESET, json={"token": vieux, "password": "EncoreAutre3"})

        assert r.status_code == 400, "un ancien lien reste utilisable apres reinitialisation"

    @pytest.mark.asyncio
    async def test_la_reinitialisation_verifie_le_compte(self, client, db_session, envoi):
        """Recevoir ce courriel prouve le controle de la boite, comme Google le fait."""
        donnees = await _inscrire(client)
        assert donnees["is_verified"] is False
        envoi.reset()
        await client.post(FORGOT, json={"email": "rahim@example.cm"})
        jeton = _jeton_du_dernier_message(envoi, "/reset-password")

        await client.post(RESET, json={"token": jeton, "password": MDP_NOUVEAU})

        db_session.expire_all()
        user = (
            await db_session.execute(select(User).where(User.id == donnees["id"]))
        ).scalar_one()
        assert user.is_verified is True

    @pytest.mark.parametrize("mdp", ["court", "sansmajuscule1", "SANSMINUSCULE1", "SansChiffre"])
    @pytest.mark.asyncio
    async def test_mot_de_passe_faible_refuse_en_422(self, client, envoi, mdp):
        """Prouve la reutilisation de `valider_force_du_mot_de_passe`, non sa recopie."""
        r = await client.post(RESET, json={"token": "x" * 40, "password": mdp})

        assert r.status_code == 422


class TestEtranglement:
    @pytest.mark.asyncio
    async def test_letranglement_ip_ignore_le_premier_element_de_x_forwarded_for(
        self, client, db_session, envoi
    ):
        """
        LE TEST QUI ATTRAPE LE CONTOURNEMENT TRIVIAL.

        Un proxy AJOUTE l'adresse observee a DROITE de `X-Forwarded-For` ; la
        valeur envoyee par le client se retrouve donc a GAUCHE. Une lecture qui
        prendrait `split(",")[0]` — la faute qu'on lit partout — laisserait
        l'attaquant declarer une IP fictive differente a chaque requete, et
        l'etranglement ne mordrait jamais.

        LE MONTAGE COMPTE AUTANT QUE L'ASSERTION. Une premiere version de ce
        test visait des adresses INCONNUES : elle passait avec la faute
        introduite, parce qu'une adresse inconnue ne declenche aucun envoi de
        toute facon. Le compteur restait a zero dans les deux cas, et le test ne
        mesurait rien.

        Ici, quinze comptes REELS et DISTINCTS : l'etranglement par compte ne
        peut donc pas se declencher (une demande chacun), et seule la couche par
        IP peut arreter les envois. La PREMIERE entree de l'en-tete change a
        chaque appel ; la derniere, celle que le proxy a reellement observee,
        reste identique.

        Verifie par experience : avec `entrees[0]`, ce test echoue a 15 envois.
        """
        for i in range(15):
            db_session.add(
                User(
                    email=f"cible{i}@example.cm",
                    username=f"cible{i}",
                    # Empreinte factice : ces comptes ne se connectent jamais.
                    # Elle doit seulement etre NON NULLE, sans quoi la route les
                    # traiterait comme des comptes Google et n'enverrait pas de
                    # lien de reinitialisation.
                    hashed_password="$2b$12$empreinte.factice.pour.le.test.uniquement.xxxxxxxxxxx",
                    full_name=f"Cible {i}",
                    role="user",
                    is_active=True,
                    is_verified=True,
                )
            )
        await db_session.commit()
        envoi.reset()
        reinitialiser_letranglement_ip()

        for i in range(15):
            r = await client.post(
                FORGOT,
                json={"email": f"cible{i}@example.cm"},
                headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"},
            )
            assert r.status_code == 202, "l'etranglement doit rester invisible du client"

        assert envoi.call_count <= 10, (
            f"{envoi.call_count} messages envoyes sur 15 : l'etranglement par IP est "
            "contourne en faisant varier la premiere entree de X-Forwarded-For"
        )

    @pytest.mark.asyncio
    async def test_demandes_repetees_sur_un_compte_cessent_d_envoyer(
        self, client, db_session, envoi
    ):
        """Le quota Brevo est un plafond dur : un seul compte ne doit pas l'epuiser."""
        import datetime

        await _inscrire(client)
        envoi.reset()

        for _ in range(6):
            reinitialiser_letranglement_ip()
            await client.post(FORGOT, json={"email": "rahim@example.cm"})
            # Contourner le delai de garde de deux minutes pour eprouver le
            # plafond horaire, qui est la limite testee ici.
            await db_session.execute(
                EmailToken.__table__.update().values(
                    created_at=maintenant_naif() - datetime.timedelta(seconds=150)
                )
            )
            await db_session.commit()

        assert envoi.call_count <= 3, (
            f"{envoi.call_count} messages : l'etranglement par compte ne mord pas"
        )
