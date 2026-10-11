"""
Une conversation garde son profil — sur une vraie base.

Les doublures de test_rag_service.py verifient la logique ; celles-ci, ce qui
est vraiment ecrit en base : un changement de profil cree une SECONDE ligne,
et laisse la premiere telle quelle (profil, titre, proprietaire).

Usage:
    pytest tests/test_services/test_conversation_un_profil.py -v
"""

import pytest
from sqlalchemy import func, select

from app.models.conversation import Conversation, Message
from app.services.rag_service import RAGService


async def _conversation(db, session_id, persona="citoyen"):
    conversation = Conversation(
        session_id=session_id, persona=persona, language="fr", title="Ma question"
    )
    db.add(conversation)
    await db.flush()
    db.add(Message(conversation_id=conversation.id, role="user", content="Ma question"))
    await db.commit()
    return conversation


@pytest.mark.asyncio
async def test_meme_profil_meme_ligne(db_session):
    await _conversation(db_session, "sid-meme-profil")

    conversation, historique = await RAGService(db_session)._load_or_create_conversation(
        "sid-meme-profil", "citoyen", "fr"
    )
    await db_session.commit()

    assert conversation.session_id == "sid-meme-profil"
    assert len(historique) == 1
    total = await db_session.scalar(select(func.count()).select_from(Conversation))
    assert total == 1


@pytest.mark.asyncio
async def test_autre_profil_seconde_ligne_et_l_ancienne_intacte(db_session):
    ancienne = await _conversation(db_session, "sid-autre-profil")
    ancienne_id = ancienne.id

    nouvelle, historique = await RAGService(db_session, user_id=None)._load_or_create_conversation(
        "sid-autre-profil", "avocat", "fr"
    )
    await db_session.commit()

    assert nouvelle.id != ancienne_id
    assert nouvelle.persona == "avocat"
    assert nouvelle.session_id != "sid-autre-profil"
    assert historique == []

    db_session.expire_all()
    relue = await db_session.get(Conversation, ancienne_id)
    assert relue.persona == "citoyen"
    assert relue.title == "Ma question"
    assert relue.user_id is None
    total = await db_session.scalar(select(func.count()).select_from(Conversation))
    assert total == 2
