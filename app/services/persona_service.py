"""
Service 9: PersonaService - Gestion des personas utilisateurs + analytics.

Features:
- Persona operations (info, list, validate)
- Real-time analytics per persona
- Daily aggregation for historical trends
- Message feedback system (👍/👎)
- Engagement metrics and popular questions
- Persona comparison and trend analysis

Author: JuriX Team
Version: 2.1.0
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta

from sqlalchemy import select, func, and_, or_, desc, distinct
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.models.conversation import (
    Conversation, Message, PersonaStat, MessageFeedback, PersonaInteraction
)
from app.services.prompts import SYSTEM_PROMPTS

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class PersonaServiceError(Exception):
    """Base exception for PersonaService operations."""
    pass


class InvalidPersonaError(PersonaServiceError):
    """Raised when persona is not one of the 4 valid types."""
    pass


class MessageNotFoundError(PersonaServiceError):
    """Raised when message ID doesn't exist."""
    pass


class FeedbackAlreadyExistsError(PersonaServiceError):
    """Raised when trying to add duplicate feedback."""
    pass


# ============================================================================
# PERSONA SERVICE
# ============================================================================

class PersonaService:
    """
    Service for managing user personas and analytics.

    Supports 4 personas:
    - citoyen: Simplified, empathetic responses for citizens
    - avocat: Technical, precise legal citations for lawyers
    - entrepreneur: Practical, compliance-focused for business owners
    - étudiant: Pedagogical, step-by-step for students

    Features:
    - Persona information and validation
    - Real-time statistics per persona
    - Daily aggregation for historical data
    - User feedback system (helpful/unhelpful, 1-5 rating)
    - Engagement metrics (messages per conversation, session duration)
    - Popular questions tracking
    - Persona comparison across metrics
    - Trend analysis over time

    Usage:
        service = PersonaService(db_session)
        personas = await service.list_personas()
        stats = await service.get_persona_stats("citoyen")
        await service.add_message_feedback(msg_id, helpful=True, rating=5)
    """

    VALID_PERSONAS = ["citoyen", "avocat", "entrepreneur", "étudiant"]

    # Persona metadata
    PERSONA_INFO = {
        "citoyen": {
            "display_name": "Citoyen",
            "description": "Assistant bienveillant qui aide les citoyens à comprendre leurs droits en termes simples",
            "icon": "👤",
            "tone": "Empathique et rassurant",
            "example_questions": [
                "Quels sont mes droits en tant que locataire?",
                "Comment porter plainte au commissariat?",
                "Que faire en cas de licenciement abusif?"
            ]
        },
        "avocat": {
            "display_name": "Avocat",
            "description": "Assistant expert fournissant des analyses juridiques précises avec références complètes",
            "icon": "⚖️",
            "tone": "Technique et nuancé",
            "example_questions": [
                "Quelle est la jurisprudence sur l'article 5 du Code pénal?",
                "Analyse des vices de consentement en droit camerounais",
                "Procédure d'appel en matière civile"
            ]
        },
        "entrepreneur": {
            "display_name": "Entrepreneur",
            "description": "Consultant juridique spécialisé en droit des affaires avec conseils actionnables",
            "icon": "💼",
            "tone": "Professionnel et pragmatique",
            "example_questions": [
                "Comment créer une SARL au Cameroun?",
                "Obligations fiscales d'une entreprise",
                "Réglementation des contrats commerciaux"
            ]
        },
        "étudiant": {
            "display_name": "Étudiant",
            "description": "Professeur patient qui explique les concepts juridiques de manière pédagogique",
            "icon": "🎓",
            "tone": "Pédagogique et encourageant",
            "example_questions": [
                "Qu'est-ce que la hiérarchie des normes?",
                "Expliquer le principe de légalité",
                "Comment rédiger un cas pratique?"
            ]
        }
    }

    def __init__(self, db: AsyncSession):
        """
        Initialize PersonaService.

        Args:
            db: Async database session
        """
        self.db = db
        logger.info("👥 PersonaService initialized")

    # ========================================================================
    # PERSONA OPERATIONS
    # ========================================================================

    async def get_persona_info(self, persona: str) -> Dict[str, Any]:
        """
        Get information about a specific persona.

        Args:
            persona: Persona name (citoyen|avocat|entrepreneur|étudiant)

        Returns:
            Dict with persona metadata

        Raises:
            InvalidPersonaError: If persona is invalid
        """
        await self.validate_persona(persona)

        info = self.PERSONA_INFO[persona].copy()
        info["name"] = persona

        logger.debug(f"📋 Retrieved info for persona: {persona}")
        return info

    async def list_personas(self) -> List[Dict[str, Any]]:
        """
        List all available personas with metadata.

        Returns:
            List of persona info dicts
        """
        personas = []
        for persona_name in self.VALID_PERSONAS:
            info = self.PERSONA_INFO[persona_name].copy()
            info["name"] = persona_name
            personas.append(info)

        logger.debug(f"📋 Listed {len(personas)} personas")
        return personas

    async def validate_persona(self, persona: str) -> bool:
        """
        Validate that persona is one of the 4 supported types.

        Args:
            persona: Persona name to validate

        Returns:
            True if valid

        Raises:
            InvalidPersonaError: If persona is not valid
        """
        if persona not in self.VALID_PERSONAS:
            logger.warning(f"❌ Invalid persona: {persona}")
            raise InvalidPersonaError(
                f"Invalid persona '{persona}'. Must be one of: {', '.join(self.VALID_PERSONAS)}"
            )
        return True

    # ========================================================================
    # ANALYTICS - REAL-TIME
    # ========================================================================

    async def get_persona_stats(
        self,
        persona: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific persona over a date range.

        Queries the persona_stats table for aggregated data.

        Args:
            persona: Persona name
            start_date: Start date (inclusive), defaults to 30 days ago
            end_date: End date (inclusive), defaults to today

        Returns:
            Dict with usage, performance, engagement, and quality metrics

        Raises:
            InvalidPersonaError: If persona is invalid
        """
        await self.validate_persona(persona)

        # Default date range: last 30 days
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Query persona_stats table
        stmt = select(PersonaStat).where(
            and_(
                PersonaStat.persona == persona,
                PersonaStat.date >= start_date,
                PersonaStat.date <= end_date
            )
        ).order_by(PersonaStat.date)

        result = await self.db.execute(stmt)
        stats = result.scalars().all()

        if not stats:
            logger.debug(f"📊 No stats found for {persona} between {start_date} and {end_date}")
            return self._empty_stats_response(persona, start_date, end_date)

        # Aggregate stats across date range
        aggregated = self._aggregate_stats(stats)

        response = {
            "persona": persona,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "usage": {
                "total_questions": aggregated["total_questions"],
                "total_conversations": aggregated["total_conversations"],
                "unique_sessions": aggregated["unique_sessions"]
            },
            "performance": {
                "avg_confidence": round(aggregated["avg_confidence"], 3),
                "avg_retrieval_time_ms": aggregated["avg_retrieval_time_ms"],
                "avg_generation_time_ms": aggregated["avg_generation_time_ms"],
                "avg_total_time_ms": aggregated["avg_total_time_ms"]
            },
            "engagement": {
                "avg_messages_per_conversation": round(aggregated["avg_messages_per_conversation"], 2),
                "avg_session_duration_seconds": aggregated["avg_session_duration_seconds"]
            },
            "quality": {
                "helpful_count": aggregated["helpful_count"],
                "unhelpful_count": aggregated["unhelpful_count"],
                "satisfaction_rate": round(aggregated["satisfaction_rate"], 3)
            }
        }

        logger.info(f"📊 Retrieved stats for {persona}: {aggregated['total_questions']} questions")
        return response

    async def get_all_persona_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get statistics for all personas.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of stats dicts for each persona
        """
        all_stats = []
        for persona in self.VALID_PERSONAS:
            stats = await self.get_persona_stats(persona, start_date, end_date)
            all_stats.append(stats)

        logger.info(f"📊 Retrieved stats for all {len(all_stats)} personas")
        return all_stats

    async def get_persona_usage_breakdown(self) -> Dict[str, float]:
        """
        Get percentage breakdown of persona usage (last 30 days).

        Returns:
            Dict mapping persona name to usage percentage
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        # Query conversation counts per persona
        stmt = select(
            Conversation.persona,
            func.count(Conversation.id).label('count')
        ).where(
            Conversation.created_at >= datetime.combine(start_date, datetime.min.time())
        ).group_by(Conversation.persona)

        result = await self.db.execute(stmt)
        counts = dict(result.all())

        total = sum(counts.values())
        if total == 0:
            logger.debug("📊 No usage data found")
            return {p: 0.0 for p in self.VALID_PERSONAS}

        breakdown = {}
        for persona in self.VALID_PERSONAS:
            count = counts.get(persona, 0)
            percentage = (count / total) * 100
            breakdown[persona] = round(percentage, 2)

        logger.info(f"📊 Usage breakdown: {breakdown}")
        return breakdown

    # ========================================================================
    # ANALYTICS - AGGREGATION
    # ========================================================================

    async def aggregate_daily_stats(self, target_date: Optional[date] = None) -> None:
        """
        Aggregate conversation/message data into persona_stats table.

        Called by Celery task daily at midnight to process previous day's data.

        Args:
            target_date: Date to aggregate (defaults to yesterday)
        """
        if not target_date:
            target_date = date.today() - timedelta(days=1)

        logger.info(f"📊 Starting daily aggregation for {target_date}")

        start_datetime = datetime.combine(target_date, datetime.min.time())
        end_datetime = datetime.combine(target_date, datetime.max.time())

        for persona in self.VALID_PERSONAS:
            await self._aggregate_persona_day(persona, target_date, start_datetime, end_datetime)

        await self.db.commit()
        logger.info(f"✅ Completed daily aggregation for {target_date}")

    async def _aggregate_persona_day(
        self,
        persona: str,
        target_date: date,
        start_datetime: datetime,
        end_datetime: datetime
    ) -> None:
        """
        Aggregate stats for one persona for one day.

        Args:
            persona: Persona name
            target_date: Date to aggregate
            start_datetime: Start of day
            end_datetime: End of day
        """
        assert isinstance(persona, str) and len(persona) > 0, "Persona must be a non-empty string"
        assert start_datetime < end_datetime, "Start datetime must be before end datetime"

        # Get conversations for this persona on this day
        conv_stmt = select(Conversation).where(
            and_(
                Conversation.persona == persona,
                Conversation.created_at >= start_datetime,
                Conversation.created_at <= end_datetime
            )
        ).options(selectinload(Conversation.messages))

        conv_result = await self.db.execute(conv_stmt)
        conversations = conv_result.scalars().all()

        if not conversations:
            logger.debug(f"No conversations for {persona} on {target_date}")
            return

        # Calculate metrics
        metrics = self._calculate_conversation_metrics(conversations)

        # Fetch feedback quality metrics
        message_ids = [m.id for m in metrics["all_messages"]]
        feedback_stmt = select(MessageFeedback).where(
            and_(
                MessageFeedback.message_id.in_(message_ids),
                MessageFeedback.created_at >= start_datetime,
                MessageFeedback.created_at <= end_datetime
            )
        )
        feedback_result = await self.db.execute(feedback_stmt)
        feedbacks = feedback_result.scalars().all()

        helpful_count = sum(1 for f in feedbacks if f.helpful)
        unhelpful_count = sum(1 for f in feedbacks if not f.helpful)
        total_feedback = len(feedbacks)
        satisfaction_rate = helpful_count / total_feedback if total_feedback > 0 else 0.0

        metrics.update({
            "helpful_count": helpful_count,
            "unhelpful_count": unhelpful_count,
            "satisfaction_rate": satisfaction_rate,
        })

        # Upsert persona_stats
        await self._upsert_persona_stat(persona, target_date, metrics)

    def _calculate_conversation_metrics(self, conversations: list) -> dict:
        """
        Calculate usage/performance/engagement metrics from conversations.

        Args:
            conversations: List of Conversation ORM objects with messages loaded

        Returns:
            Dict with total_questions, total_conversations, unique_sessions,
            avg_confidence, avg_retrieval, avg_generation, avg_total,
            messages_per_conv, avg_duration, and all_messages list.
        """
        total_conversations = len(conversations)
        unique_sessions = len(set(c.session_id for c in conversations))

        all_messages = []
        for conv in conversations:
            all_messages.extend([m for m in conv.messages if m.role == 'assistant'])

        total_questions = sum(
            len([m for m in conv.messages if m.role == 'user'])
            for conv in conversations
        )

        confidences = [m.confidence for m in all_messages if m.confidence is not None]
        retrieval_times = [m.retrieval_time_ms for m in all_messages if m.retrieval_time_ms is not None]
        generation_times = [m.generation_time_ms for m in all_messages if m.generation_time_ms is not None]

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_retrieval = sum(retrieval_times) // len(retrieval_times) if retrieval_times else 0
        avg_generation = sum(generation_times) // len(generation_times) if generation_times else 0

        messages_per_conv = total_questions / total_conversations if total_conversations else 0.0

        session_durations = []
        for conv in conversations:
            if len(conv.messages) >= 2:
                duration = (conv.messages[-1].created_at - conv.messages[0].created_at).total_seconds()
                session_durations.append(duration)
        avg_duration = int(sum(session_durations) / len(session_durations)) if session_durations else 0

        return {
            "total_questions": total_questions,
            "total_conversations": total_conversations,
            "unique_sessions": unique_sessions,
            "avg_confidence": avg_confidence,
            "avg_retrieval_time_ms": avg_retrieval,
            "avg_generation_time_ms": avg_generation,
            "avg_total_time_ms": avg_retrieval + avg_generation,
            "avg_messages_per_conversation": messages_per_conv,
            "avg_session_duration_seconds": avg_duration,
            "all_messages": all_messages,
        }

    async def _upsert_persona_stat(
        self, persona: str, target_date: date, metrics: dict
    ) -> None:
        """
        Create or update a PersonaStat record for a persona on a given date.

        Args:
            persona: Persona name
            target_date: Date for the stats record
            metrics: Dict with all computed metric fields
        """
        existing_stmt = select(PersonaStat).where(
            and_(
                PersonaStat.persona == persona,
                PersonaStat.date == target_date
            )
        )
        existing_result = await self.db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        stat_fields = {k: v for k, v in metrics.items() if k != "all_messages"}

        if existing:
            for key, value in stat_fields.items():
                setattr(existing, key, value)
            existing.updated_at = datetime.utcnow()
            logger.debug(f"Updated stats for {persona} on {target_date}")
        else:
            new_stat = PersonaStat(
                persona=persona,
                date=target_date,
                **stat_fields
            )
            self.db.add(new_stat)
            logger.debug(f"Created stats for {persona} on {target_date}")

    async def update_interaction_metrics(
        self,
        conversation_id: int,
        persona: str,
        language: str
    ) -> None:
        """
        Update or create persona_interactions record for a conversation.

        Called by RAGService after each message exchange.

        Args:
            conversation_id: ID of conversation
            persona: Current persona
            language: Current language
        """
        assert isinstance(conversation_id, int) and conversation_id > 0, "Conversation ID must be a positive integer"
        assert isinstance(persona, str) and len(persona) > 0, "Persona must be a non-empty string"

        # Check if interaction record exists
        stmt = select(PersonaInteraction).where(
            PersonaInteraction.conversation_id == conversation_id
        )
        result = await self.db.execute(stmt)
        interaction = result.scalar_one_or_none()

        # Get conversation with messages
        conv_stmt = select(Conversation).where(
            Conversation.id == conversation_id
        ).options(selectinload(Conversation.messages))
        conv_result = await self.db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            logger.warning(f"❌ Conversation {conversation_id} not found")
            return

        # Calculate metrics
        user_messages = [m for m in conversation.messages if m.role == 'user']
        assistant_messages = [m for m in conversation.messages if m.role == 'assistant']

        question_count = len(user_messages)
        answer_count = len(assistant_messages)

        confidences = [m.confidence for m in assistant_messages if m.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        response_times = [
            (m.retrieval_time_ms or 0) + (m.generation_time_ms or 0)
            for m in assistant_messages
        ]
        total_response_time = sum(response_times)

        # Track persona switches
        personas_used = [persona] if not interaction else interaction.personas_used or [persona]
        if persona not in personas_used:
            personas_used.append(persona)
        persona_switches = len(personas_used) - 1

        if interaction:
            # Update existing
            interaction.persona = persona
            interaction.language = language
            interaction.question_count = question_count
            interaction.answer_count = answer_count
            interaction.avg_confidence = avg_confidence
            interaction.total_response_time_ms = total_response_time
            interaction.personas_used = personas_used
            interaction.persona_switches = persona_switches
            interaction.updated_at = datetime.utcnow()
            logger.debug(f"Updated interaction for conversation {conversation_id}")
        else:
            # Create new
            new_interaction = PersonaInteraction(
                conversation_id=conversation_id,
                persona=persona,
                language=language,
                question_count=question_count,
                answer_count=answer_count,
                avg_confidence=avg_confidence,
                total_response_time_ms=total_response_time,
                personas_used=personas_used,
                persona_switches=persona_switches
            )
            self.db.add(new_interaction)
            logger.debug(f"Created interaction for conversation {conversation_id}")

        await self.db.commit()

    # ========================================================================
    # MESSAGE FEEDBACK
    # ========================================================================

    async def add_message_feedback(
        self,
        message_id: int,
        helpful: bool,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record user feedback on a message.

        Args:
            message_id: ID of message
            helpful: Whether message was helpful (👍/👎)
            rating: Optional 1-5 star rating
            comment: Optional text comment

        Returns:
            Created feedback dict

        Raises:
            MessageNotFoundError: If message doesn't exist
            FeedbackAlreadyExistsError: If feedback already exists for this message
        """
        # Verify message exists
        msg_stmt = select(Message).where(Message.id == message_id)
        msg_result = await self.db.execute(msg_stmt)
        message = msg_result.scalar_one_or_none()

        if not message:
            logger.warning(f"❌ Message {message_id} not found")
            raise MessageNotFoundError(f"Message {message_id} not found")

        # Check for existing feedback
        existing_stmt = select(MessageFeedback).where(
            MessageFeedback.message_id == message_id
        )
        existing_result = await self.db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            logger.warning(f"❌ Feedback already exists for message {message_id}")
            raise FeedbackAlreadyExistsError(f"Feedback already exists for message {message_id}")

        # Validate rating if provided
        if rating is not None and (rating < 1 or rating > 5):
            raise ValueError("Rating must be between 1 and 5")

        # Create feedback
        feedback = MessageFeedback(
            message_id=message_id,
            helpful=helpful,
            rating=rating,
            comment=comment
        )
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)

        logger.info(f"✅ Added feedback for message {message_id}: helpful={helpful}, rating={rating}")

        return {
            "id": feedback.id,
            "message_id": feedback.message_id,
            "helpful": feedback.helpful,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "created_at": feedback.created_at.isoformat()
        }

    async def get_message_feedback(self, message_id: int) -> Optional[Dict[str, Any]]:
        """
        Get feedback for a specific message.

        Args:
            message_id: ID of message

        Returns:
            Feedback dict or None if no feedback exists
        """
        stmt = select(MessageFeedback).where(
            MessageFeedback.message_id == message_id
        )
        result = await self.db.execute(stmt)
        feedback = result.scalar_one_or_none()

        if not feedback:
            return None

        return {
            "id": feedback.id,
            "message_id": feedback.message_id,
            "helpful": feedback.helpful,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "created_at": feedback.created_at.isoformat()
        }

    async def get_feedback_stats(
        self,
        persona: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.

        Args:
            persona: Optional persona filter
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dict with feedback statistics

        Raises:
            InvalidPersonaError: If persona is invalid
        """
        if persona:
            await self.validate_persona(persona)

        # Build query
        query = select(MessageFeedback)

        filters = []
        if start_date:
            filters.append(MessageFeedback.created_at >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            filters.append(MessageFeedback.created_at <= datetime.combine(end_date, datetime.max.time()))

        if persona:
            # Join with messages and conversations to filter by persona
            query = query.join(Message, MessageFeedback.message_id == Message.id)\
                         .join(Conversation, Message.conversation_id == Conversation.id)\
                         .where(Conversation.persona == persona)

        if filters:
            query = query.where(and_(*filters))

        result = await self.db.execute(query)
        feedbacks = result.scalars().all()

        # Calculate stats
        total_feedback = len(feedbacks)
        helpful_count = sum(1 for f in feedbacks if f.helpful)
        unhelpful_count = total_feedback - helpful_count
        satisfaction_rate = helpful_count / total_feedback if total_feedback > 0 else 0.0

        ratings = [f.rating for f in feedbacks if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        # By persona breakdown
        by_persona = {}
        if not persona:
            for p in self.VALID_PERSONAS:
                p_feedbacks = await self._get_persona_feedbacks(p, start_date, end_date)
                by_persona[p] = {
                    "total": len(p_feedbacks),
                    "helpful": sum(1 for f in p_feedbacks if f.helpful),
                    "unhelpful": len(p_feedbacks) - sum(1 for f in p_feedbacks if f.helpful)
                }

        stats = {
            "total_feedback": total_feedback,
            "helpful_count": helpful_count,
            "unhelpful_count": unhelpful_count,
            "satisfaction_rate": round(satisfaction_rate, 3),
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "by_persona": by_persona
        }

        logger.info(f"📊 Feedback stats: {total_feedback} total, {satisfaction_rate:.1%} satisfaction")
        return stats

    async def _get_persona_feedbacks(
        self,
        persona: str,
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> List[MessageFeedback]:
        """Helper to get feedbacks for a specific persona."""
        query = select(MessageFeedback)\
            .join(Message, MessageFeedback.message_id == Message.id)\
            .join(Conversation, Message.conversation_id == Conversation.id)\
            .where(Conversation.persona == persona)

        filters = []
        if start_date:
            filters.append(MessageFeedback.created_at >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            filters.append(MessageFeedback.created_at <= datetime.combine(end_date, datetime.max.time()))

        if filters:
            query = query.where(and_(*filters))

        result = await self.db.execute(query)
        return result.scalars().all()

    # ========================================================================
    # ENGAGEMENT METRICS
    # ========================================================================

    async def get_engagement_metrics(
        self,
        persona: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get engagement metrics for a persona over N days.

        Args:
            persona: Persona name
            days: Number of days to analyze (default 30)

        Returns:
            Dict with engagement metrics

        Raises:
            InvalidPersonaError: If persona is invalid
        """
        await self.validate_persona(persona)

        start_date = date.today() - timedelta(days=days)
        end_date = date.today()

        # Get conversations
        start_datetime = datetime.combine(start_date, datetime.min.time())

        conv_stmt = select(Conversation).where(
            and_(
                Conversation.persona == persona,
                Conversation.created_at >= start_datetime
            )
        ).options(selectinload(Conversation.messages))

        conv_result = await self.db.execute(conv_stmt)
        conversations = conv_result.scalars().all()

        if not conversations:
            return {
                "persona": persona,
                "days": days,
                "metrics": {
                    "total_conversations": 0,
                    "total_messages": 0,
                    "avg_messages_per_conversation": 0.0,
                    "avg_session_duration_seconds": 0,
                    "active_days": 0
                }
            }

        # Calculate metrics
        total_conversations = len(conversations)
        total_messages = sum(len(c.messages) for c in conversations)
        avg_messages = total_messages / total_conversations

        # Session durations
        durations = []
        for conv in conversations:
            if len(conv.messages) >= 2:
                duration = (conv.messages[-1].created_at - conv.messages[0].created_at).total_seconds()
                durations.append(duration)
        avg_duration = int(sum(durations) / len(durations)) if durations else 0

        # Active days (days with at least one conversation)
        active_dates = set(c.created_at.date() for c in conversations)
        active_days = len(active_dates)

        metrics = {
            "persona": persona,
            "days": days,
            "metrics": {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "avg_messages_per_conversation": round(avg_messages, 2),
                "avg_session_duration_seconds": avg_duration,
                "active_days": active_days,
                "conversations_per_active_day": round(total_conversations / active_days, 2) if active_days > 0 else 0.0
            }
        }

        logger.info(f"📊 Engagement metrics for {persona}: {total_conversations} conversations over {days} days")
        return metrics

    async def get_popular_questions(
        self,
        persona: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most frequently asked questions for a persona.

        Uses fuzzy matching to group similar questions.

        Args:
            persona: Persona name
            limit: Max number of questions to return

        Returns:
            List of dicts with question and count

        Raises:
            InvalidPersonaError: If persona is invalid
        """
        await self.validate_persona(persona)

        # Get user messages for this persona (last 90 days)
        start_date = datetime.now() - timedelta(days=90)

        stmt = select(Message.content, func.count(Message.id).label('count'))\
            .join(Conversation, Message.conversation_id == Conversation.id)\
            .where(
                and_(
                    Conversation.persona == persona,
                    Message.role == 'user',
                    Message.created_at >= start_date
                )
            )\
            .group_by(Message.content)\
            .order_by(desc('count'))\
            .limit(limit)

        result = await self.db.execute(stmt)
        questions = result.all()

        popular = [
            {
                "question": q.content,
                "count": q.count
            }
            for q in questions
        ]

        logger.info(f"📊 Found {len(popular)} popular questions for {persona}")
        return popular

    # ========================================================================
    # COMPARISON & TRENDS
    # ========================================================================

    async def compare_personas(
        self,
        metric: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare personas across a specific metric.

        Args:
            metric: Metric to compare (usage|confidence|satisfaction)
            days: Number of days to analyze

        Returns:
            Dict with comparison data for all personas

        Raises:
            ValueError: If metric is invalid
        """
        valid_metrics = ["usage", "confidence", "satisfaction"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of: {', '.join(valid_metrics)}")

        start_date = date.today() - timedelta(days=days)
        end_date = date.today()

        comparison = {
            "metric": metric,
            "days": days,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "personas": {}
        }

        for persona in self.VALID_PERSONAS:
            stats = await self.get_persona_stats(persona, start_date, end_date)

            if metric == "usage":
                value = stats["usage"]["total_questions"]
            elif metric == "confidence":
                value = stats["performance"]["avg_confidence"]
            elif metric == "satisfaction":
                value = stats["quality"]["satisfaction_rate"]
            else:
                value = 0

            comparison["personas"][persona] = {
                "value": value,
                "display_name": self.PERSONA_INFO[persona]["display_name"]
            }

        logger.info(f"📊 Compared personas by {metric} over {days} days")
        return comparison

    async def get_trends(
        self,
        persona: str,
        metric: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get time-series trend data for a metric.

        Args:
            persona: Persona name
            metric: Metric to track (questions|confidence|satisfaction)
            days: Number of days

        Returns:
            List of dicts with date and value

        Raises:
            InvalidPersonaError: If persona is invalid
            ValueError: If metric is invalid
        """
        await self.validate_persona(persona)

        valid_metrics = ["questions", "confidence", "satisfaction"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of: {', '.join(valid_metrics)}")

        start_date = date.today() - timedelta(days=days)
        end_date = date.today()

        # Query persona_stats
        stmt = select(PersonaStat).where(
            and_(
                PersonaStat.persona == persona,
                PersonaStat.date >= start_date,
                PersonaStat.date <= end_date
            )
        ).order_by(PersonaStat.date)

        result = await self.db.execute(stmt)
        stats = result.scalars().all()

        trends = []
        for stat in stats:
            if metric == "questions":
                value = stat.total_questions
            elif metric == "confidence":
                value = stat.avg_confidence
            elif metric == "satisfaction":
                value = stat.satisfaction_rate
            else:
                value = 0

            trends.append({
                "date": stat.date.isoformat(),
                "value": value
            })

        logger.info(f"📊 Retrieved {len(trends)} trend points for {persona} ({metric})")
        return trends

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Service health check.

        Returns:
            Dict with service status
        """
        return {
            "service": "PersonaService",
            "status": "healthy",
            "personas_count": len(self.VALID_PERSONAS),
            "personas": self.VALID_PERSONAS,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _empty_stats_response(
        self,
        persona: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Generate empty stats response when no data exists."""
        return {
            "persona": persona,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "usage": {
                "total_questions": 0,
                "total_conversations": 0,
                "unique_sessions": 0
            },
            "performance": {
                "avg_confidence": 0.0,
                "avg_retrieval_time_ms": 0,
                "avg_generation_time_ms": 0,
                "avg_total_time_ms": 0
            },
            "engagement": {
                "avg_messages_per_conversation": 0.0,
                "avg_session_duration_seconds": 0
            },
            "quality": {
                "helpful_count": 0,
                "unhelpful_count": 0,
                "satisfaction_rate": 0.0
            }
        }

    def _aggregate_stats(self, stats: List[PersonaStat]) -> Dict[str, Any]:
        """Aggregate multiple PersonaStat records into summary metrics."""
        if not stats:
            return {
                "total_questions": 0,
                "total_conversations": 0,
                "unique_sessions": 0,
                "avg_confidence": 0.0,
                "avg_retrieval_time_ms": 0,
                "avg_generation_time_ms": 0,
                "avg_total_time_ms": 0,
                "avg_messages_per_conversation": 0.0,
                "avg_session_duration_seconds": 0,
                "helpful_count": 0,
                "unhelpful_count": 0,
                "satisfaction_rate": 0.0
            }

        # Sum usage metrics
        total_questions = sum(s.total_questions for s in stats)
        total_conversations = sum(s.total_conversations for s in stats)
        unique_sessions = sum(s.unique_sessions for s in stats)

        # Average performance metrics (weighted by question count)
        total_weight = sum(s.total_questions for s in stats if s.total_questions > 0)

        if total_weight > 0:
            avg_confidence = sum(
                s.avg_confidence * s.total_questions for s in stats if s.total_questions > 0
            ) / total_weight
            avg_retrieval = sum(
                s.avg_retrieval_time_ms * s.total_questions for s in stats if s.total_questions > 0
            ) / total_weight
            avg_generation = sum(
                s.avg_generation_time_ms * s.total_questions for s in stats if s.total_questions > 0
            ) / total_weight
            avg_total = sum(
                s.avg_total_time_ms * s.total_questions for s in stats if s.total_questions > 0
            ) / total_weight
        else:
            avg_confidence = avg_retrieval = avg_generation = avg_total = 0.0

        # Average engagement metrics
        avg_messages = sum(s.avg_messages_per_conversation for s in stats) / len(stats)
        avg_duration = sum(s.avg_session_duration_seconds for s in stats) // len(stats)

        # Sum quality metrics
        helpful_count = sum(s.helpful_count for s in stats)
        unhelpful_count = sum(s.unhelpful_count for s in stats)
        total_feedback = helpful_count + unhelpful_count
        satisfaction_rate = helpful_count / total_feedback if total_feedback > 0 else 0.0

        return {
            "total_questions": total_questions,
            "total_conversations": total_conversations,
            "unique_sessions": unique_sessions,
            "avg_confidence": avg_confidence,
            "avg_retrieval_time_ms": int(avg_retrieval),
            "avg_generation_time_ms": int(avg_generation),
            "avg_total_time_ms": int(avg_total),
            "avg_messages_per_conversation": avg_messages,
            "avg_session_duration_seconds": avg_duration,
            "helpful_count": helpful_count,
            "unhelpful_count": unhelpful_count,
            "satisfaction_rate": satisfaction_rate
        }
