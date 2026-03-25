"""
Pydantic schemas for PersonaService API.

Request/response models for persona management and analytics endpoints.
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class MessageFeedbackCreate(BaseModel):
    """
    Schema for creating message feedback.

    Users can rate messages as helpful/unhelpful with optional 1-5 rating.
    """
    message_id: int = Field(..., description="ID of the message to rate", gt=0)
    helpful: bool = Field(..., description="Whether the message was helpful (👍/👎)")
    rating: Optional[int] = Field(None, description="Optional 1-5 star rating", ge=1, le=5)
    comment: Optional[str] = Field(None, description="Optional comment", max_length=500)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": 123,
                    "helpful": True,
                    "rating": 5,
                    "comment": "Très claire et utile!"
                }
            ]
        }
    }


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class PersonaInfo(BaseModel):
    """
    Information about a specific persona.

    Includes name, display name, description, icon, tone, and example questions.
    """
    name: str = Field(..., description="Persona identifier (citoyen|avocat|entrepreneur|étudiant)")
    display_name: str = Field(..., description="Display name (e.g., 'Citoyen', 'Avocat')")
    description: str = Field(..., description="Description of persona's approach")
    icon: str = Field(..., description="Emoji icon representing persona")
    tone: str = Field(..., description="Communication tone of persona")
    example_questions: List[str] = Field(..., description="Example questions for this persona")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "citoyen",
                    "display_name": "Citoyen",
                    "description": "Assistant bienveillant qui aide les citoyens à comprendre leurs droits",
                    "icon": "👤",
                    "tone": "Empathique et rassurant",
                    "example_questions": [
                        "Quels sont mes droits en tant que locataire?",
                        "Comment porter plainte au commissariat?"
                    ]
                }
            ]
        }
    }


class UsageMetrics(BaseModel):
    """Usage statistics metrics."""
    total_questions: int = Field(..., description="Total number of questions asked")
    total_conversations: int = Field(..., description="Total number of conversations")
    unique_sessions: int = Field(..., description="Number of unique session IDs")


class PerformanceMetrics(BaseModel):
    """Performance statistics metrics."""
    avg_confidence: float = Field(..., description="Average confidence score (0.0-1.0)")
    avg_retrieval_time_ms: int = Field(..., description="Average retrieval time in milliseconds")
    avg_generation_time_ms: int = Field(..., description="Average generation time in milliseconds")
    avg_total_time_ms: int = Field(..., description="Average total response time in milliseconds")


class EngagementMetrics(BaseModel):
    """Engagement statistics metrics."""
    avg_messages_per_conversation: float = Field(..., description="Average messages per conversation")
    avg_session_duration_seconds: int = Field(..., description="Average session duration in seconds")


class QualityMetrics(BaseModel):
    """Quality statistics metrics from user feedback."""
    helpful_count: int = Field(..., description="Number of helpful ratings (👍)")
    unhelpful_count: int = Field(..., description="Number of unhelpful ratings (👎)")
    satisfaction_rate: float = Field(..., description="Satisfaction rate (0.0-1.0)")


class DateRange(BaseModel):
    """Date range for statistics."""
    start: str = Field(..., description="Start date (ISO format)")
    end: str = Field(..., description="End date (ISO format)")


class PersonaStatsResponse(BaseModel):
    """
    Comprehensive statistics for a persona over a date range.

    Aggregated from persona_stats table.
    """
    persona: str = Field(..., description="Persona name")
    date_range: DateRange = Field(..., description="Date range for statistics")
    usage: UsageMetrics = Field(..., description="Usage metrics")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    engagement: EngagementMetrics = Field(..., description="Engagement metrics")
    quality: QualityMetrics = Field(..., description="Quality metrics from feedback")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "persona": "citoyen",
                    "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
                    "usage": {
                        "total_questions": 150,
                        "total_conversations": 80,
                        "unique_sessions": 75
                    },
                    "performance": {
                        "avg_confidence": 0.85,
                        "avg_retrieval_time_ms": 450,
                        "avg_generation_time_ms": 1200,
                        "avg_total_time_ms": 1650
                    },
                    "engagement": {
                        "avg_messages_per_conversation": 1.88,
                        "avg_session_duration_seconds": 180
                    },
                    "quality": {
                        "helpful_count": 45,
                        "unhelpful_count": 5,
                        "satisfaction_rate": 0.9
                    }
                }
            ]
        }
    }


class MessageFeedbackResponse(BaseModel):
    """Response schema for message feedback."""
    id: int = Field(..., description="Feedback ID")
    message_id: int = Field(..., description="Message ID")
    helpful: bool = Field(..., description="Whether message was helpful")
    rating: Optional[int] = Field(None, description="Optional 1-5 rating")
    comment: Optional[str] = Field(None, description="Optional comment")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "message_id": 123,
                    "helpful": True,
                    "rating": 5,
                    "comment": "Très utile!",
                    "created_at": "2024-01-15T10:30:00"
                }
            ]
        }
    }


class PersonaFeedbackBreakdown(BaseModel):
    """Feedback breakdown for a single persona."""
    total: int = Field(..., description="Total feedback count")
    helpful: int = Field(..., description="Helpful count")
    unhelpful: int = Field(..., description="Unhelpful count")


class FeedbackStats(BaseModel):
    """
    Aggregated feedback statistics.

    Can be filtered by persona and date range.
    """
    total_feedback: int = Field(..., description="Total number of feedback entries")
    helpful_count: int = Field(..., description="Number of helpful ratings")
    unhelpful_count: int = Field(..., description="Number of unhelpful ratings")
    satisfaction_rate: float = Field(..., description="Satisfaction rate (0.0-1.0)")
    avg_rating: Optional[float] = Field(None, description="Average star rating (1-5)")
    by_persona: Dict[str, PersonaFeedbackBreakdown] = Field(
        default_factory=dict,
        description="Breakdown by persona (empty if filtered by persona)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total_feedback": 100,
                    "helpful_count": 85,
                    "unhelpful_count": 15,
                    "satisfaction_rate": 0.85,
                    "avg_rating": 4.2,
                    "by_persona": {
                        "citoyen": {"total": 50, "helpful": 45, "unhelpful": 5},
                        "avocat": {"total": 30, "helpful": 25, "unhelpful": 5},
                        "entrepreneur": {"total": 15, "helpful": 12, "unhelpful": 3},
                        "étudiant": {"total": 5, "helpful": 3, "unhelpful": 2}
                    }
                }
            ]
        }
    }


class DetailedEngagementMetrics(BaseModel):
    """Detailed engagement metrics for a persona."""
    total_conversations: int = Field(..., description="Total conversations in period")
    total_messages: int = Field(..., description="Total messages exchanged")
    avg_messages_per_conversation: float = Field(..., description="Average messages per conversation")
    avg_session_duration_seconds: int = Field(..., description="Average session duration")
    active_days: int = Field(..., description="Number of days with activity")
    conversations_per_active_day: float = Field(..., description="Average conversations per active day")


class EngagementMetricsResponse(BaseModel):
    """
    Response schema for engagement metrics endpoint.
    """
    persona: str = Field(..., description="Persona name")
    days: int = Field(..., description="Number of days analyzed")
    metrics: DetailedEngagementMetrics = Field(..., description="Engagement metrics")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "persona": "citoyen",
                    "days": 30,
                    "metrics": {
                        "total_conversations": 120,
                        "total_messages": 240,
                        "avg_messages_per_conversation": 2.0,
                        "avg_session_duration_seconds": 180,
                        "active_days": 25,
                        "conversations_per_active_day": 4.8
                    }
                }
            ]
        }
    }


class PopularQuestion(BaseModel):
    """A popular question with its frequency count."""
    question: str = Field(..., description="Question text")
    count: int = Field(..., description="Number of times asked")


class PopularQuestionsResponse(BaseModel):
    """Response schema for popular questions endpoint."""
    persona: str = Field(..., description="Persona name")
    questions: List[PopularQuestion] = Field(..., description="List of popular questions")
    total_count: int = Field(..., description="Total number of questions found")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "persona": "citoyen",
                    "questions": [
                        {"question": "Quels sont mes droits en tant que locataire?", "count": 15},
                        {"question": "Comment porter plainte?", "count": 12}
                    ],
                    "total_count": 2
                }
            ]
        }
    }


class PersonaValue(BaseModel):
    """Value and display name for a persona in comparison."""
    value: float = Field(..., description="Metric value")
    display_name: str = Field(..., description="Display name of persona")


class PersonaComparisonResponse(BaseModel):
    """
    Response schema for persona comparison endpoint.
    """
    metric: str = Field(..., description="Metric being compared (usage|confidence|satisfaction)")
    days: int = Field(..., description="Number of days analyzed")
    date_range: DateRange = Field(..., description="Date range for comparison")
    personas: Dict[str, PersonaValue] = Field(..., description="Comparison values per persona")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "metric": "usage",
                    "days": 30,
                    "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
                    "personas": {
                        "citoyen": {"value": 150, "display_name": "Citoyen"},
                        "avocat": {"value": 80, "display_name": "Avocat"},
                        "entrepreneur": {"value": 45, "display_name": "Entrepreneur"},
                        "étudiant": {"value": 25, "display_name": "Étudiant"}
                    }
                }
            ]
        }
    }


class TrendPoint(BaseModel):
    """A single point in a trend time series."""
    date: str = Field(..., description="Date (ISO format)")
    value: float = Field(..., description="Metric value for this date")


class TrendsResponse(BaseModel):
    """Response schema for trends endpoint."""
    persona: str = Field(..., description="Persona name")
    metric: str = Field(..., description="Metric tracked (questions|confidence|satisfaction)")
    days: int = Field(..., description="Number of days analyzed")
    data_points: List[TrendPoint] = Field(..., description="Time series data points")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "persona": "citoyen",
                    "metric": "questions",
                    "days": 7,
                    "data_points": [
                        {"date": "2024-01-01", "value": 5},
                        {"date": "2024-01-02", "value": 8},
                        {"date": "2024-01-03", "value": 12}
                    ]
                }
            ]
        }
    }


class UsageBreakdownResponse(BaseModel):
    """Response schema for persona usage breakdown."""
    total_conversations: int = Field(..., description="Total conversations across all personas")
    percentages: Dict[str, float] = Field(..., description="Usage percentage per persona")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total_conversations": 300,
                    "percentages": {
                        "citoyen": 50.0,
                        "avocat": 26.67,
                        "entrepreneur": 15.0,
                        "étudiant": 8.33
                    }
                }
            ]
        }
    }


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    personas_count: int = Field(..., description="Number of supported personas")
    personas: List[str] = Field(..., description="List of persona names")
    timestamp: str = Field(..., description="Timestamp (ISO format)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "service": "PersonaService",
                    "status": "healthy",
                    "personas_count": 4,
                    "personas": ["citoyen", "avocat", "entrepreneur", "étudiant"],
                    "timestamp": "2024-01-15T10:30:00"
                }
            ]
        }
    }


# ============================================================================
# ERROR RESPONSE SCHEMAS
# ============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: ErrorDetail = Field(..., description="Error details")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": {
                        "error": "InvalidPersonaError",
                        "message": "Invalid persona 'unknown'. Must be one of: citoyen, avocat, entrepreneur, étudiant",
                        "details": {"provided": "unknown", "valid": ["citoyen", "avocat", "entrepreneur", "étudiant"]}
                    }
                }
            ]
        }
    }
