"""Tests for PersonaService."""

import pytest
from datetime import date, datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.persona_service import (
    PersonaService,
    InvalidPersonaError,
    MessageNotFoundError,
    FeedbackAlreadyExistsError
)
from app.models.conversation import (
    Conversation, Message, PersonaStat, MessageFeedback, PersonaInteraction
)


# ============================================================================
# TEST PERSONA OPERATIONS
# ============================================================================

class TestPersonaOperations:
    """Tests for persona information and validation."""

    @pytest.mark.asyncio
    async def test_get_persona_info_success(self, db_session: AsyncSession):
        """Test getting info for valid persona."""
        service = PersonaService(db_session)

        info = await service.get_persona_info("citoyen")

        assert info["name"] == "citoyen"
        assert info["display_name"] == "Citoyen"
        assert "description" in info
        assert info["icon"] == "👤"
        assert "tone" in info
        assert isinstance(info["example_questions"], list)
        assert len(info["example_questions"]) > 0

    @pytest.mark.asyncio
    async def test_get_persona_info_invalid(self, db_session: AsyncSession):
        """Test getting info for invalid persona raises error."""
        service = PersonaService(db_session)

        with pytest.raises(InvalidPersonaError) as exc_info:
            await service.get_persona_info("invalid_persona")

        assert "invalid_persona" in str(exc_info.value).lower()
        assert "citoyen" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_personas(self, db_session: AsyncSession):
        """Test listing all personas."""
        service = PersonaService(db_session)

        personas = await service.list_personas()

        assert len(personas) == 4
        persona_names = [p["name"] for p in personas]
        assert "citoyen" in persona_names
        assert "avocat" in persona_names
        assert "entrepreneur" in persona_names
        assert "étudiant" in persona_names

        # Check each has required fields
        for persona in personas:
            assert "name" in persona
            assert "display_name" in persona
            assert "description" in persona
            assert "icon" in persona
            assert "tone" in persona
            assert "example_questions" in persona

    @pytest.mark.asyncio
    async def test_validate_persona_valid(self, db_session: AsyncSession):
        """Test validating valid persona returns True."""
        service = PersonaService(db_session)

        assert await service.validate_persona("citoyen") is True
        assert await service.validate_persona("avocat") is True
        assert await service.validate_persona("entrepreneur") is True
        assert await service.validate_persona("étudiant") is True

    @pytest.mark.asyncio
    async def test_validate_persona_invalid(self, db_session: AsyncSession):
        """Test validating invalid persona raises error."""
        service = PersonaService(db_session)

        with pytest.raises(InvalidPersonaError):
            await service.validate_persona("unknown")

        with pytest.raises(InvalidPersonaError):
            await service.validate_persona("")


# ============================================================================
# TEST ANALYTICS STATS
# ============================================================================

class TestAnalyticsStats:
    """Tests for statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_persona_stats_empty(self, db_session: AsyncSession):
        """Test getting stats when no data exists."""
        service = PersonaService(db_session)

        stats = await service.get_persona_stats("citoyen")

        assert stats["persona"] == "citoyen"
        assert "date_range" in stats
        assert stats["usage"]["total_questions"] == 0
        assert stats["usage"]["total_conversations"] == 0
        assert stats["performance"]["avg_confidence"] == 0.0
        assert stats["engagement"]["avg_messages_per_conversation"] == 0.0
        assert stats["quality"]["satisfaction_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_persona_stats_with_data(self, db_session: AsyncSession):
        """Test getting stats when data exists."""
        service = PersonaService(db_session)

        # Create test data
        test_date = date.today()
        stat = PersonaStat(
            persona="citoyen",
            date=test_date,
            total_questions=50,
            total_conversations=30,
            unique_sessions=28,
            avg_confidence=0.85,
            avg_retrieval_time_ms=450,
            avg_generation_time_ms=1200,
            avg_total_time_ms=1650,
            avg_messages_per_conversation=1.67,
            avg_session_duration_seconds=180,
            helpful_count=25,
            unhelpful_count=5,
            satisfaction_rate=0.833
        )
        db_session.add(stat)
        await db_session.commit()

        # Get stats
        stats = await service.get_persona_stats(
            "citoyen",
            start_date=test_date,
            end_date=test_date
        )

        assert stats["persona"] == "citoyen"
        assert stats["usage"]["total_questions"] == 50
        assert stats["usage"]["total_conversations"] == 30
        assert stats["performance"]["avg_confidence"] == 0.85
        assert stats["quality"]["helpful_count"] == 25
        assert stats["quality"]["unhelpful_count"] == 5

    @pytest.mark.asyncio
    async def test_get_all_persona_stats(self, db_session: AsyncSession):
        """Test getting stats for all personas."""
        service = PersonaService(db_session)

        all_stats = await service.get_all_persona_stats()

        assert len(all_stats) == 4
        personas = [s["persona"] for s in all_stats]
        assert "citoyen" in personas
        assert "avocat" in personas
        assert "entrepreneur" in personas
        assert "étudiant" in personas

    @pytest.mark.asyncio
    async def test_get_persona_usage_breakdown_empty(self, db_session: AsyncSession):
        """Test usage breakdown with no data."""
        service = PersonaService(db_session)

        breakdown = await service.get_persona_usage_breakdown()

        assert breakdown["citoyen"] == 0.0
        assert breakdown["avocat"] == 0.0
        assert breakdown["entrepreneur"] == 0.0
        assert breakdown["étudiant"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_with_date_range(self, db_session: AsyncSession):
        """Test stats filtering by date range."""
        service = PersonaService(db_session)

        # Create stats for different dates
        today = date.today()
        yesterday = today - timedelta(days=1)

        stat1 = PersonaStat(persona="citoyen", date=yesterday, total_questions=10)
        stat2 = PersonaStat(persona="citoyen", date=today, total_questions=20)
        db_session.add_all([stat1, stat2])
        await db_session.commit()

        # Get only today's stats
        stats = await service.get_persona_stats(
            "citoyen",
            start_date=today,
            end_date=today
        )

        assert stats["usage"]["total_questions"] == 20


# ============================================================================
# TEST MESSAGE FEEDBACK
# ============================================================================

class TestMessageFeedback:
    """Tests for message feedback functionality."""

    @pytest.mark.asyncio
    async def test_add_feedback_success(self, db_session: AsyncSession):
        """Test successfully adding feedback."""
        service = PersonaService(db_session)

        # Create test conversation and message
        conv = Conversation(
            session_id="test-session",
            persona="citoyen",
            language="fr"
        )
        db_session.add(conv)
        await db_session.flush()

        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content="Test answer",
            confidence=0.9
        )
        db_session.add(msg)
        await db_session.commit()

        # Add feedback
        feedback = await service.add_message_feedback(
            message_id=msg.id,
            helpful=True,
            rating=5,
            comment="Very helpful!"
        )

        assert feedback["message_id"] == msg.id
        assert feedback["helpful"] is True
        assert feedback["rating"] == 5
        assert feedback["comment"] == "Very helpful!"
        assert "created_at" in feedback

    @pytest.mark.asyncio
    async def test_add_feedback_duplicate(self, db_session: AsyncSession):
        """Test adding duplicate feedback raises error."""
        service = PersonaService(db_session)

        # Create test message
        conv = Conversation(session_id="test-session2", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        msg = Message(conversation_id=conv.id, role="assistant", content="Test")
        db_session.add(msg)
        await db_session.commit()

        # Add first feedback
        await service.add_message_feedback(msg.id, helpful=True)

        # Try to add second feedback
        with pytest.raises(FeedbackAlreadyExistsError):
            await service.add_message_feedback(msg.id, helpful=False)

    @pytest.mark.asyncio
    async def test_add_feedback_message_not_found(self, db_session: AsyncSession):
        """Test adding feedback to non-existent message."""
        service = PersonaService(db_session)

        with pytest.raises(MessageNotFoundError):
            await service.add_message_feedback(
                message_id=999999,
                helpful=True
            )

    @pytest.mark.asyncio
    async def test_get_feedback(self, db_session: AsyncSession):
        """Test retrieving feedback for a message."""
        service = PersonaService(db_session)

        # Create test data
        conv = Conversation(session_id="test-session3", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        msg = Message(conversation_id=conv.id, role="assistant", content="Test")
        db_session.add(msg)
        await db_session.commit()

        # Add feedback
        created = await service.add_message_feedback(msg.id, helpful=True, rating=4)

        # Get feedback
        feedback = await service.get_message_feedback(msg.id)

        assert feedback is not None
        assert feedback["message_id"] == msg.id
        assert feedback["helpful"] is True
        assert feedback["rating"] == 4

    @pytest.mark.asyncio
    async def test_get_feedback_stats(self, db_session: AsyncSession):
        """Test aggregated feedback statistics."""
        service = PersonaService(db_session)

        # Create test data
        conv = Conversation(session_id="test-session4", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        # Create multiple messages with feedback
        for i in range(5):
            msg = Message(conversation_id=conv.id, role="assistant", content=f"Test {i}")
            db_session.add(msg)
            await db_session.flush()

            helpful = i < 4  # 4 helpful, 1 unhelpful
            fb = MessageFeedback(message_id=msg.id, helpful=helpful, rating=5 if helpful else 2)
            db_session.add(fb)

        await db_session.commit()

        # Get stats
        stats = await service.get_feedback_stats()

        assert stats["total_feedback"] == 5
        assert stats["helpful_count"] == 4
        assert stats["unhelpful_count"] == 1
        assert stats["satisfaction_rate"] == 0.8


# ============================================================================
# TEST ENGAGEMENT METRICS
# ============================================================================

class TestEngagementMetrics:
    """Tests for engagement metrics."""

    @pytest.mark.asyncio
    async def test_get_engagement_metrics_empty(self, db_session: AsyncSession):
        """Test engagement metrics with no data."""
        service = PersonaService(db_session)

        metrics = await service.get_engagement_metrics("citoyen", days=30)

        assert metrics["persona"] == "citoyen"
        assert metrics["days"] == 30
        assert metrics["metrics"]["total_conversations"] == 0
        assert metrics["metrics"]["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_get_engagement_metrics_with_data(self, db_session: AsyncSession):
        """Test engagement metrics with conversation data."""
        service = PersonaService(db_session)

        # Create test conversations
        for i in range(3):
            conv = Conversation(
                session_id=f"engagement-test-{i}",
                persona="citoyen",
                language="fr"
            )
            db_session.add(conv)
            await db_session.flush()

            # Add messages
            for j in range(4):  # 2 user + 2 assistant
                role = "user" if j % 2 == 0 else "assistant"
                msg = Message(
                    conversation_id=conv.id,
                    role=role,
                    content=f"Message {j}"
                )
                db_session.add(msg)

        await db_session.commit()

        # Get metrics
        metrics = await service.get_engagement_metrics("citoyen", days=30)

        assert metrics["metrics"]["total_conversations"] == 3
        assert metrics["metrics"]["total_messages"] == 12

    @pytest.mark.asyncio
    async def test_get_popular_questions(self, db_session: AsyncSession):
        """Test retrieving popular questions."""
        service = PersonaService(db_session)

        # Create test data
        conv = Conversation(session_id="popular-q-test", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        # Add same question multiple times
        for i in range(3):
            msg = Message(
                conversation_id=conv.id,
                role="user",
                content="Quels sont mes droits?"
            )
            db_session.add(msg)

        await db_session.commit()

        # Get popular questions
        questions = await service.get_popular_questions("citoyen", limit=10)

        assert len(questions) > 0
        assert questions[0]["question"] == "Quels sont mes droits?"
        assert questions[0]["count"] == 3

    @pytest.mark.asyncio
    async def test_compare_personas(self, db_session: AsyncSession):
        """Test comparing personas across a metric."""
        service = PersonaService(db_session)

        # Create test stats
        today = date.today()
        for persona in ["citoyen", "avocat"]:
            stat = PersonaStat(
                persona=persona,
                date=today,
                total_questions=50 if persona == "citoyen" else 30
            )
            db_session.add(stat)
        await db_session.commit()

        # Compare
        comparison = await service.compare_personas("usage", days=1)

        assert comparison["metric"] == "usage"
        assert "citoyen" in comparison["personas"]
        assert "avocat" in comparison["personas"]

    @pytest.mark.asyncio
    async def test_get_trends(self, db_session: AsyncSession):
        """Test getting trend data."""
        service = PersonaService(db_session)

        # Create stats for multiple days
        today = date.today()
        for i in range(3):
            stat_date = today - timedelta(days=i)
            stat = PersonaStat(
                persona="citoyen",
                date=stat_date,
                total_questions=10 + i
            )
            db_session.add(stat)
        await db_session.commit()

        # Get trends
        trends = await service.get_trends("citoyen", "questions", days=3)

        assert len(trends) == 3
        assert all("date" in t and "value" in t for t in trends)


# ============================================================================
# TEST AGGREGATION
# ============================================================================

class TestAggregation:
    """Tests for daily aggregation functionality."""

    @pytest.mark.asyncio
    async def test_aggregate_daily_stats_empty(self, db_session: AsyncSession):
        """Test aggregating when no conversations exist."""
        service = PersonaService(db_session)

        target_date = date.today() - timedelta(days=1)

        # Should not raise error
        await service.aggregate_daily_stats(target_date)
        await db_session.commit()

    @pytest.mark.asyncio
    async def test_update_interaction_metrics(self, db_session: AsyncSession):
        """Test updating interaction metrics for a conversation."""
        service = PersonaService(db_session)

        # Create conversation with messages
        conv = Conversation(
            session_id="interaction-test",
            persona="citoyen",
            language="fr"
        )
        db_session.add(conv)
        await db_session.flush()

        msg1 = Message(conversation_id=conv.id, role="user", content="Question")
        msg2 = Message(
            conversation_id=conv.id,
            role="assistant",
            content="Answer",
            confidence=0.9,
            retrieval_time_ms=500,
            generation_time_ms=1500
        )
        db_session.add_all([msg1, msg2])
        await db_session.commit()

        # Update metrics
        await service.update_interaction_metrics(
            conversation_id=conv.id,
            persona="citoyen",
            language="fr"
        )
        await db_session.commit()

        # Verify interaction was created
        from sqlalchemy import select
        stmt = select(PersonaInteraction).where(
            PersonaInteraction.conversation_id == conv.id
        )
        result = await db_session.execute(stmt)
        interaction = result.scalar_one_or_none()

        assert interaction is not None
        assert interaction.persona == "citoyen"
        assert interaction.question_count == 1
        assert interaction.answer_count == 1
        assert interaction.avg_confidence == 0.9


# ============================================================================
# TEST HEALTH CHECK
# ============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check(self, db_session: AsyncSession):
        """Test service health check."""
        service = PersonaService(db_session)

        health = service.health_check()

        assert health["service"] == "PersonaService"
        assert health["status"] == "healthy"
        assert health["personas_count"] == 4
        assert len(health["personas"]) == 4
        assert "citoyen" in health["personas"]
        assert "timestamp" in health
