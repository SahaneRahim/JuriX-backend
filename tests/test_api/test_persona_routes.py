"""Integration tests for persona API routes."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date, timedelta

from app.models.conversation import (
    Conversation, Message, PersonaStat, MessageFeedback
)


# ============================================================================
# TEST PERSONA INFORMATION ENDPOINTS
# ============================================================================

class TestListPersonasAPI:
    """Tests for GET /api/v1/personas endpoint."""

    @pytest.mark.asyncio
    async def test_list_personas_success(self, client: AsyncClient):
        """Test listing all personas."""
        response = await client.get("/api/v1/personas")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 4

        # Check all expected personas present
        persona_names = [p["name"] for p in data]
        assert "citoyen" in persona_names
        assert "avocat" in persona_names
        assert "entrepreneur" in persona_names
        assert "étudiant" in persona_names

        # Check structure
        for persona in data:
            assert "name" in persona
            assert "display_name" in persona
            assert "description" in persona
            assert "icon" in persona
            assert "tone" in persona
            assert "example_questions" in persona
            assert isinstance(persona["example_questions"], list)


class TestGetPersonaInfoAPI:
    """Tests for GET /api/v1/personas/{persona} endpoint."""

    @pytest.mark.asyncio
    async def test_get_persona_info_success(self, client: AsyncClient):
        """Test getting info for valid persona."""
        response = await client.get("/api/v1/personas/citoyen")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "citoyen"
        assert data["display_name"] == "Citoyen"
        assert data["icon"] == "👤"
        assert "description" in data
        assert "tone" in data
        assert len(data["example_questions"]) > 0

    @pytest.mark.asyncio
    async def test_get_persona_info_invalid(self, client: AsyncClient):
        """Test getting info for invalid persona returns 400."""
        response = await client.get("/api/v1/personas/invalid_persona")

        assert response.status_code == 400
        data = response.json()
        assert "InvalidPersonaError" in data["detail"]["error"]

    @pytest.mark.asyncio
    async def test_get_all_personas_individually(self, client: AsyncClient):
        """Test getting each persona individually."""
        personas = ["citoyen", "avocat", "entrepreneur", "étudiant"]

        for persona in personas:
            response = await client.get(f"/api/v1/personas/{persona}")
            assert response.status_code == 200
            assert response.json()["name"] == persona


class TestHealthCheckAPI:
    """Tests for GET /api/v1/personas/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: AsyncClient):
        """Test health check endpoint."""
        response = await client.get("/api/v1/personas/health")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "PersonaService"
        assert data["status"] == "healthy"
        assert data["personas_count"] == 4
        assert len(data["personas"]) == 4
        assert "timestamp" in data


# ============================================================================
# TEST ANALYTICS - STATISTICS ENDPOINTS
# ============================================================================

class TestGetPersonaStatsAPI:
    """Tests for GET /api/v1/personas/{persona}/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_no_data(self, client: AsyncClient):
        """Test getting stats when no data exists."""
        response = await client.get("/api/v1/personas/citoyen/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["persona"] == "citoyen"
        assert "date_range" in data
        assert data["usage"]["total_questions"] == 0
        assert data["performance"]["avg_confidence"] == 0.0
        assert data["engagement"]["avg_messages_per_conversation"] == 0.0
        assert data["quality"]["satisfaction_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_with_date_range(self, client: AsyncClient):
        """Test getting stats with date range parameters."""
        start = (date.today() - timedelta(days=7)).isoformat()
        end = date.today().isoformat()

        response = await client.get(
            f"/api/v1/personas/citoyen/stats?start_date={start}&end_date={end}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["date_range"]["start"] == start
        assert data["date_range"]["end"] == end

    @pytest.mark.asyncio
    async def test_get_stats_invalid_persona(self, client: AsyncClient):
        """Test getting stats for invalid persona."""
        response = await client.get("/api/v1/personas/invalid/stats")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_stats_invalid_date_format(self, client: AsyncClient):
        """Test getting stats with invalid date format."""
        response = await client.get(
            "/api/v1/personas/citoyen/stats?start_date=invalid-date"
        )

        assert response.status_code == 400


class TestGetAllPersonaStatsAPI:
    """Tests for GET /api/v1/personas/stats/all endpoint."""

    @pytest.mark.asyncio
    async def test_get_all_stats_success(self, client: AsyncClient):
        """Test getting stats for all personas."""
        response = await client.get("/api/v1/personas/stats/all")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 4

        personas = [s["persona"] for s in data]
        assert "citoyen" in personas
        assert "avocat" in personas
        assert "entrepreneur" in personas
        assert "étudiant" in personas

    @pytest.mark.asyncio
    async def test_get_all_stats_with_dates(self, client: AsyncClient):
        """Test getting all stats with date filters."""
        start = (date.today() - timedelta(days=7)).isoformat()
        end = date.today().isoformat()

        response = await client.get(
            f"/api/v1/personas/stats/all?start_date={start}&end_date={end}"
        )

        assert response.status_code == 200
        assert len(response.json()) == 4


class TestGetUsageBreakdownAPI:
    """Tests for GET /api/v1/personas/stats/usage endpoint."""

    @pytest.mark.asyncio
    async def test_get_usage_breakdown(self, client: AsyncClient):
        """Test getting usage breakdown."""
        response = await client.get("/api/v1/personas/stats/usage")

        assert response.status_code == 200
        data = response.json()
        assert "percentages" in data
        assert len(data["percentages"]) == 4
        assert "citoyen" in data["percentages"]
        assert all(isinstance(v, float) for v in data["percentages"].values())


# ============================================================================
# TEST ANALYTICS - ENGAGEMENT ENDPOINTS
# ============================================================================

class TestGetEngagementMetricsAPI:
    """Tests for GET /api/v1/personas/{persona}/engagement endpoint."""

    @pytest.mark.asyncio
    async def test_get_engagement_default(self, client: AsyncClient):
        """Test getting engagement metrics with default days."""
        response = await client.get("/api/v1/personas/citoyen/engagement")

        assert response.status_code == 200
        data = response.json()
        assert data["persona"] == "citoyen"
        assert data["days"] == 30
        assert "metrics" in data
        assert "total_conversations" in data["metrics"]
        assert "total_messages" in data["metrics"]
        assert "avg_messages_per_conversation" in data["metrics"]

    @pytest.mark.asyncio
    async def test_get_engagement_custom_days(self, client: AsyncClient):
        """Test getting engagement metrics with custom days parameter."""
        response = await client.get("/api/v1/personas/citoyen/engagement?days=7")

        assert response.status_code == 200
        data = response.json()
        assert data["days"] == 7

    @pytest.mark.asyncio
    async def test_get_engagement_invalid_persona(self, client: AsyncClient):
        """Test engagement for invalid persona."""
        response = await client.get("/api/v1/personas/invalid/engagement")

        assert response.status_code == 400


class TestGetPopularQuestionsAPI:
    """Tests for GET /api/v1/personas/{persona}/questions endpoint."""

    @pytest.mark.asyncio
    async def test_get_popular_questions_default(self, client: AsyncClient):
        """Test getting popular questions with default limit."""
        response = await client.get("/api/v1/personas/citoyen/questions")

        assert response.status_code == 200
        data = response.json()
        assert data["persona"] == "citoyen"
        assert "questions" in data
        assert isinstance(data["questions"], list)
        assert "total_count" in data

    @pytest.mark.asyncio
    async def test_get_popular_questions_custom_limit(self, client: AsyncClient):
        """Test getting popular questions with custom limit."""
        response = await client.get("/api/v1/personas/citoyen/questions?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["questions"]) <= 5

    @pytest.mark.asyncio
    async def test_get_popular_questions_invalid_persona(self, client: AsyncClient):
        """Test popular questions for invalid persona."""
        response = await client.get("/api/v1/personas/invalid/questions")

        assert response.status_code == 400


# ============================================================================
# TEST ANALYTICS - COMPARISON & TRENDS ENDPOINTS
# ============================================================================

class TestComparePersonasAPI:
    """Tests for GET /api/v1/personas/compare/{metric} endpoint."""

    @pytest.mark.asyncio
    async def test_compare_usage(self, client: AsyncClient):
        """Test comparing personas by usage metric."""
        response = await client.get("/api/v1/personas/compare/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "usage"
        assert data["days"] == 30
        assert "personas" in data
        assert len(data["personas"]) == 4
        assert "citoyen" in data["personas"]

    @pytest.mark.asyncio
    async def test_compare_confidence(self, client: AsyncClient):
        """Test comparing personas by confidence metric."""
        response = await client.get("/api/v1/personas/compare/confidence")

        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "confidence"

    @pytest.mark.asyncio
    async def test_compare_satisfaction(self, client: AsyncClient):
        """Test comparing personas by satisfaction metric."""
        response = await client.get("/api/v1/personas/compare/satisfaction")

        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "satisfaction"

    @pytest.mark.asyncio
    async def test_compare_invalid_metric(self, client: AsyncClient):
        """Test comparing with invalid metric."""
        response = await client.get("/api/v1/personas/compare/invalid_metric")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_compare_custom_days(self, client: AsyncClient):
        """Test comparing with custom days parameter."""
        response = await client.get("/api/v1/personas/compare/usage?days=7")

        assert response.status_code == 200
        data = response.json()
        assert data["days"] == 7


class TestGetTrendsAPI:
    """Tests for GET /api/v1/personas/{persona}/trends/{metric} endpoint."""

    @pytest.mark.asyncio
    async def test_get_trends_questions(self, client: AsyncClient):
        """Test getting trend data for questions metric."""
        response = await client.get("/api/v1/personas/citoyen/trends/questions")

        assert response.status_code == 200
        data = response.json()
        assert data["persona"] == "citoyen"
        assert data["metric"] == "questions"
        assert data["days"] == 30
        assert isinstance(data["data_points"], list)

    @pytest.mark.asyncio
    async def test_get_trends_confidence(self, client: AsyncClient):
        """Test getting trend data for confidence metric."""
        response = await client.get("/api/v1/personas/avocat/trends/confidence")

        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "confidence"

    @pytest.mark.asyncio
    async def test_get_trends_satisfaction(self, client: AsyncClient):
        """Test getting trend data for satisfaction metric."""
        response = await client.get("/api/v1/personas/entrepreneur/trends/satisfaction")

        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "satisfaction"

    @pytest.mark.asyncio
    async def test_get_trends_custom_days(self, client: AsyncClient):
        """Test getting trends with custom days."""
        response = await client.get("/api/v1/personas/citoyen/trends/questions?days=14")

        assert response.status_code == 200
        data = response.json()
        assert data["days"] == 14

    @pytest.mark.asyncio
    async def test_get_trends_invalid_persona(self, client: AsyncClient):
        """Test trends for invalid persona."""
        response = await client.get("/api/v1/personas/invalid/trends/questions")

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_trends_invalid_metric(self, client: AsyncClient):
        """Test trends with invalid metric."""
        response = await client.get("/api/v1/personas/citoyen/trends/invalid")

        assert response.status_code == 400


# ============================================================================
# TEST FEEDBACK ENDPOINTS
# ============================================================================

class TestAddFeedbackAPI:
    """Tests for POST /api/v1/personas/feedback endpoint."""

    @pytest.mark.asyncio
    async def test_add_feedback_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession
    ):
        """Test successfully adding feedback."""
        # Create test message
        conv = Conversation(session_id="feedback-test", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        msg = Message(conversation_id=conv.id, role="assistant", content="Test")
        db_session.add(msg)
        await db_session.commit()

        # Add feedback via API
        response = await client.post(
            "/api/v1/personas/feedback",
            json={
                "message_id": msg.id,
                "helpful": True,
                "rating": 5,
                "comment": "Excellent!"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["message_id"] == msg.id
        assert data["helpful"] is True
        assert data["rating"] == 5
        assert data["comment"] == "Excellent!"
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_add_feedback_without_rating(
        self,
        client: AsyncClient,
        db_session: AsyncSession
    ):
        """Test adding feedback without optional rating."""
        conv = Conversation(session_id="feedback-test2", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        msg = Message(conversation_id=conv.id, role="assistant", content="Test")
        db_session.add(msg)
        await db_session.commit()

        response = await client.post(
            "/api/v1/personas/feedback",
            json={"message_id": msg.id, "helpful": False}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["helpful"] is False
        assert data["rating"] is None

    @pytest.mark.asyncio
    async def test_add_feedback_duplicate(
        self,
        client: AsyncClient,
        db_session: AsyncSession
    ):
        """Test adding duplicate feedback returns 409."""
        conv = Conversation(session_id="feedback-test3", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        msg = Message(conversation_id=conv.id, role="assistant", content="Test")
        db_session.add(msg)
        await db_session.commit()

        # Add first feedback
        await client.post(
            "/api/v1/personas/feedback",
            json={"message_id": msg.id, "helpful": True}
        )

        # Try to add second feedback
        response = await client.post(
            "/api/v1/personas/feedback",
            json={"message_id": msg.id, "helpful": False}
        )

        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_add_feedback_message_not_found(self, client: AsyncClient):
        """Test adding feedback for non-existent message."""
        response = await client.post(
            "/api/v1/personas/feedback",
            json={"message_id": 999999, "helpful": True}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_add_feedback_invalid_rating(self, client: AsyncClient):
        """Test adding feedback with invalid rating."""
        response = await client.post(
            "/api/v1/personas/feedback",
            json={"message_id": 1, "helpful": True, "rating": 10}  # Invalid: must be 1-5
        )

        assert response.status_code == 422  # Validation error


class TestGetFeedbackAPI:
    """Tests for GET /api/v1/personas/feedback/{message_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_feedback_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession
    ):
        """Test getting existing feedback."""
        # Create test data
        conv = Conversation(session_id="get-feedback-test", persona="citoyen", language="fr")
        db_session.add(conv)
        await db_session.flush()

        msg = Message(conversation_id=conv.id, role="assistant", content="Test")
        db_session.add(msg)
        await db_session.flush()

        feedback = MessageFeedback(message_id=msg.id, helpful=True, rating=4)
        db_session.add(feedback)
        await db_session.commit()

        # Get via API
        response = await client.get(f"/api/v1/personas/feedback/{msg.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message_id"] == msg.id
        assert data["helpful"] is True
        assert data["rating"] == 4

    @pytest.mark.asyncio
    async def test_get_feedback_not_found(self, client: AsyncClient):
        """Test getting feedback for message without feedback."""
        response = await client.get("/api/v1/personas/feedback/999999")

        assert response.status_code == 404


class TestGetFeedbackStatsAPI:
    """Tests for GET /api/v1/personas/feedback/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_feedback_stats_all(self, client: AsyncClient):
        """Test getting feedback stats for all personas."""
        response = await client.get("/api/v1/personas/feedback/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data
        assert "helpful_count" in data
        assert "unhelpful_count" in data
        assert "satisfaction_rate" in data
        assert "by_persona" in data

    @pytest.mark.asyncio
    async def test_get_feedback_stats_filtered_by_persona(self, client: AsyncClient):
        """Test getting feedback stats for specific persona."""
        response = await client.get("/api/v1/personas/feedback/stats?persona=citoyen")

        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data

    @pytest.mark.asyncio
    async def test_get_feedback_stats_with_dates(self, client: AsyncClient):
        """Test getting feedback stats with date range."""
        start = (date.today() - timedelta(days=7)).isoformat()
        end = date.today().isoformat()

        response = await client.get(
            f"/api/v1/personas/feedback/stats?start_date={start}&end_date={end}"
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_feedback_stats_invalid_persona(self, client: AsyncClient):
        """Test feedback stats with invalid persona."""
        response = await client.get("/api/v1/personas/feedback/stats?persona=invalid")

        assert response.status_code == 400
