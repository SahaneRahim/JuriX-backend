"""
Database models for JuriX application.

Exports all SQLAlchemy models for easy import.
"""

from app.models.conversation import (
    Conversation,
    Message,
    MessageFeedback,
    PersonaInteraction,
    PersonaStat,
)
from app.models.law import Article, Category, Law
from app.models.user import User

__all__ = [
    # Law models
    "Law",
    "Article",
    "Category",
    # Conversation models
    "Conversation",
    "Message",
    "PersonaStat",
    "MessageFeedback",
    "PersonaInteraction",
    # User models
    "User",
]
