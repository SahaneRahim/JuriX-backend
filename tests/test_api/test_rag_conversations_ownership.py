"""
Appartenance des conversations — le fichier de securite du mode « compte ».

CE QU'IL PROTEGE. Avant ce travail, `GET` et `DELETE /rag/conversations/{sid}`
n'avaient aucune authentification ni controle d'appartenance, et `POST
/rag/ask` avec le `session_id` d'un tiers chargeait sa conversation puis en
reinjectait les cinq derniers messages dans le prompt — le contenu d'autrui
ressortait donc dans la reponse generee.

Promettre « votre compte conserve vos conversations » sans fermer cela aurait
ete un mensonge sur la confidentialite. Ces tests sont la garantie que la porte
reste fermee.

La regle testee, en une phrase : `user_id` est le proprietaire ; NULL signifie
« anonyme, appartient a qui detient le session_id ». Un tiers recoit 404 et non
403 — un 403 confirmerait l'existence du session_id et permettrait de les
enumerer.

Usage:
    pytest tests/test_api/test_rag_conversations_ownership.py -v
"""

from datetime import datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import create_access_token
from app.models.conversation import Conversation, Message

LISTE = "/api/v1/rag/conversations"


def entetes(user):
    return {"Authorization": f"Bearer {create_access_token({'sub': user.email})}"}


async def _conversation(
    db: AsyncSession, session_id: str, user_id=None, titre=None, question="Ma question"
) -> Conversation:
    conv = Conversation(
        session_id=session_id,
        user_id=user_id,
        title=titre,
        persona="citoyen",
        language="fr",
    )
    db.add(conv)
    await db.flush()
    db.add(Message(conversation_id=conv.id, role="user", content=question))
    db.add(Message(conversation_id=conv.id, role="assistant", content="Ma reponse"))
    await db.commit()
    await db.refresh(conv)
    return conv


class TestLectureEtSuppression:
    @pytest.mark.asyncio
    async def test_conversation_dautrui_repond_404(
        self, client: AsyncClient, db_session, test_user, test_admin_user
    ):
        conv = await _conversation(db_session, "sid-de-test-user", user_id=test_user.id)

        r = await client.get(f"{LISTE}/{conv.session_id}", headers=entetes(test_admin_user))

        assert r.status_code == 404, "un 403 confirmerait l'existence du session_id"

    @pytest.mark.asyncio
    async def test_suppression_dautrui_refusee_et_sans_effet(
        self, client: AsyncClient, db_session, test_user, test_admin_user
    ):
        """Le 404 ne doit pas etre qu'un message : la conversation doit survivre."""
        conv = await _conversation(db_session, "sid-a-proteger", user_id=test_user.id)
        # Identifiants relus AVANT `expire_all()` : apres, y acceder declenche
        # un rafraichissement paresseux synchrone, donc un MissingGreenlet.
        conv_id, session_id = conv.id, conv.session_id

        r = await client.delete(f"{LISTE}/{session_id}", headers=entetes(test_admin_user))

        assert r.status_code == 404
        db_session.expire_all()
        reste = (
            await db_session.execute(
                select(func.count()).select_from(Conversation).where(Conversation.id == conv_id)
            )
        ).scalar()
        assert reste == 1, "la conversation a ete supprimee malgre le refus"

    @pytest.mark.asyncio
    async def test_anonyme_reste_accessible_sans_jeton(self, client: AsyncClient, db_session):
        """
        NON-REGRESSION. Toutes les conversations anterieures ont `user_id` NULL.
        Les rendre inaccessibles aurait casse l'existant.
        """
        conv = await _conversation(db_session, "sid-anonyme")

        assert (await client.get(f"{LISTE}/{conv.session_id}")).status_code == 200
        assert (await client.delete(f"{LISTE}/{conv.session_id}")).status_code == 204

    @pytest.mark.asyncio
    async def test_ma_conversation_est_lisible(
        self, client: AsyncClient, db_session, test_user
    ):
        conv = await _conversation(db_session, "sid-a-moi", user_id=test_user.id)

        r = await client.get(f"{LISTE}/{conv.session_id}", headers=entetes(test_user))

        assert r.status_code == 200
        assert len(r.json()["messages"]) == 2

    @pytest.mark.asyncio
    async def test_conversation_inexistante(self, client: AsyncClient, test_user):
        r = await client.get(f"{LISTE}/sid-qui-nexiste-pas", headers=entetes(test_user))

        assert r.status_code == 404


class TestListe:
    @pytest.mark.asyncio
    async def test_sans_jeton_401(self, client: AsyncClient):
        assert (await client.get(LISTE)).status_code == 401

    @pytest.mark.asyncio
    async def test_ne_rend_que_les_miennes(
        self, client: AsyncClient, db_session, test_user, test_admin_user
    ):
        await _conversation(db_session, "sid-moi-1", user_id=test_user.id, titre="A moi 1")
        await _conversation(db_session, "sid-moi-2", user_id=test_user.id, titre="A moi 2")
        await _conversation(db_session, "sid-autre", user_id=test_admin_user.id, titre="Pas a moi")
        await _conversation(db_session, "sid-anon", titre="Anonyme")

        r = await client.get(LISTE, headers=entetes(test_user))

        assert r.status_code == 200
        titres = {c["title"] for c in r.json()}
        assert titres == {"A moi 1", "A moi 2"}, (
            "une conversation d'autrui ou anonyme s'est glissee dans la liste"
        )

    @pytest.mark.asyncio
    async def test_la_plus_recente_en_premier(
        self, client: AsyncClient, db_session, test_user
    ):
        vieille = await _conversation(db_session, "sid-vieux", user_id=test_user.id, titre="Vieille")
        recente = await _conversation(db_session, "sid-neuf", user_id=test_user.id, titre="Recente")
        vieille.updated_at = datetime.utcnow() - timedelta(days=3)
        recente.updated_at = datetime.utcnow()
        await db_session.commit()

        r = await client.get(LISTE, headers=entetes(test_user))

        assert [c["title"] for c in r.json()] == ["Recente", "Vieille"]

    @pytest.mark.asyncio
    async def test_la_liste_ne_porte_pas_les_messages(
        self, client: AsyncClient, db_session, test_user
    ):
        """Sinon l'ouverture du panneau ferait transiter tout l'historique."""
        await _conversation(db_session, "sid-leger", user_id=test_user.id, titre="Titre")

        r = await client.get(LISTE, headers=entetes(test_user))

        assert "messages" not in r.json()[0]


class TestAskAvecSessionDautrui:
    """Le chemin le plus dangereux : l'historique d'autrui dans le prompt."""

    @pytest.mark.asyncio
    async def test_ask_refuse_et_ne_touche_a_rien(
        self, client: AsyncClient, db_session, test_user, test_admin_user
    ):
        conv = await _conversation(
            db_session, "sid-victime", user_id=test_user.id, question="Question privee"
        )
        conv_id, session_id = conv.id, conv.session_id
        avant = (
            await db_session.execute(
                select(func.count())
                .select_from(Message)
                .where(Message.conversation_id == conv_id)
            )
        ).scalar()

        r = await client.post(
            "/api/v1/rag/ask",
            headers=entetes(test_admin_user),
            json={
                "question": "Quelles sont les conditions ?",
                "persona": "citoyen",
                "language": "fr",
                "session_id": session_id,
            },
        )

        assert r.status_code == 404
        db_session.expire_all()
        apres = (
            await db_session.execute(
                select(func.count())
                .select_from(Message)
                .where(Message.conversation_id == conv_id)
            )
        ).scalar()
        # L'assertion qui prouve que l'historique d'autrui n'est pas reparti
        # dans le prompt : rien n'a ete lu, rien n'a ete ajoute.
        assert apres == avant, "des messages ont ete ajoutes chez la victime"

    @pytest.mark.asyncio
    async def test_session_id_trop_long_est_un_422(self, client: AsyncClient):
        """
        La colonne fait `String(100)`. Sans borne au schema, une chaine plus
        longue faisait echouer l'insert en 500 — et ce champ porte desormais
        une decision d'autorisation.
        """
        r = await client.post(
            "/api/v1/rag/ask",
            json={
                "question": "Une question valide",
                "persona": "citoyen",
                "language": "fr",
                "session_id": "x" * 150,
            },
        )

        assert r.status_code == 422
