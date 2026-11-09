"""
Supprimer une conversation supprime ses lignes d'analytique et les avis de ses
messages. Sans cascade, SQLAlchemy mettait leur cle a NULL, refusee par la
colonne : DELETE /rag/conversations rendait 500 en production (08/11/2026).
"""
from sqlalchemy.orm import configure_mappers

from app.models.conversation import Conversation, Message

# Les backrefs n'existent qu'une fois les mappers configures.
configure_mappers()


def test_les_interactions_suivent_la_conversation():
    relation = Conversation.persona_interactions.property
    assert relation.cascade.delete and relation.cascade.delete_orphan


def test_les_avis_suivent_le_message():
    relation = Message.feedback.property
    assert relation.cascade.delete and relation.cascade.delete_orphan
