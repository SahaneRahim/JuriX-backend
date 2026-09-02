"""
API Routes for Persona Management and Analytics (Service 9).

Endpoints:
- GET /api/v1/personas - List all personas
- GET /api/v1/personas/{persona} - Get persona info
- GET /api/v1/personas/health - Health check
- GET /api/v1/personas/{persona}/stats - Get stats for persona
- GET /api/v1/personas/stats/all - Get stats for all personas
- GET /api/v1/personas/stats/usage - Get usage breakdown
- GET /api/v1/personas/{persona}/engagement - Get engagement metrics
- GET /api/v1/personas/{persona}/questions - Get popular questions
- GET /api/v1/personas/compare/{metric} - Compare personas
- GET /api/v1/personas/{persona}/trends/{metric} - Get trend data
- POST /api/v1/personas/feedback - Add message feedback
- GET /api/v1/personas/feedback/{message_id} - Get feedback
- GET /api/v1/personas/feedback/stats - Get feedback statistics

Author: JuriX Team
Version: 2.1.0
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.persona_service import (
    PersonaService,
    InvalidPersonaError,
    MessageNotFoundError,
    FeedbackAlreadyExistsError,
    PersonaServiceError
)
from app.schemas.persona import (
    PersonaInfo,
    PersonaStatsResponse,
    MessageFeedbackCreate,
    MessageFeedbackResponse,
    FeedbackStats,
    EngagementMetricsResponse,
    PopularQuestionsResponse,
    PopularQuestion,
    PersonaComparisonResponse,
    TrendsResponse,
    TrendPoint,
    UsageBreakdownResponse,
    HealthCheckResponse,
    ErrorResponse
)

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["personas"],
    responses={
        400: {"description": "Bad Request - Invalid persona or validation error"},
        404: {"description": "Not Found - Message or resource doesn't exist"},
        409: {"description": "Conflict - Feedback already exists"},
        500: {"description": "Internal Server Error"}
    }
)


# ============================================================================
# PERSONA INFORMATION ENDPOINTS
# ============================================================================

@router.get(
    "",
    response_model=List[PersonaInfo],
    summary="List all available personas",
    description="Get information about all 4 supported personas with metadata"
)
async def list_personas(
    db: AsyncSession = Depends(get_db)
) -> List[PersonaInfo]:
    """
    List all available personas.

    Returns metadata for all 4 personas:
    - citoyen: Simplified responses for citizens
    - avocat: Technical legal analysis for lawyers
    - entrepreneur: Business-focused compliance advice
    - étudiant: Pedagogical explanations for students

    **Returns:**
    - List of persona info with name, display name, description, icon, tone, example questions
    """
    try:
        service = PersonaService(db)
        personas = await service.list_personas()
        logger.info(f"📋 Listed {len(personas)} personas")
        return personas
    except Exception as e:
        logger.error(f"❌ Error listing personas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


# NOTE: declaree avant "/{persona}" — FastAPI resout dans l'ordre de
# declaration, la route parametree capturait sinon ce chemin.
@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service health check"
)
def health_check(
    db: AsyncSession = Depends(get_db)
) -> HealthCheckResponse:
    """
    Check PersonaService health status.

    **Returns:**
    - Service status, persona count, and timestamp
    """
    try:
        service = PersonaService(db)
        health = service.health_check()
        return health
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


# ============================================================================
# ANALYTICS - STATISTICS ENDPOINTS


@router.get(
    "/{persona}",
    response_model=PersonaInfo,
    summary="Get info for specific persona",
    responses={
        404: {"model": ErrorResponse, "description": "Persona not found"}
    }
)
async def get_persona_info(
    persona: str = Path(..., description="Persona name (citoyen|avocat|entrepreneur|étudiant)"),
    db: AsyncSession = Depends(get_db)
) -> PersonaInfo:
    """
    Get information about a specific persona.

    **Parameters:**
    - **persona**: Persona name (citoyen, avocat, entrepreneur, étudiant)

    **Returns:**
    - Persona metadata including description, tone, and example questions

    **Raises:**
    - 400: Invalid persona name
    """
    assert isinstance(persona, str) and len(persona) > 0, "Persona must be a non-empty string"

    try:
        service = PersonaService(db)
        info = await service.get_persona_info(persona)
        logger.info(f"📋 Retrieved info for persona: {persona}")
        return info
    except InvalidPersonaError as e:
        logger.warning(f"❌ Invalid persona: {persona}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidPersonaError",
                "message": str(e),
                "details": {"provided": persona, "valid": PersonaService.VALID_PERSONAS}
            }
        )
    except Exception as e:
        logger.error(f"❌ Error getting persona info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


# ============================================================================

# NOTE: declaree avant "/{persona}/stats" — FastAPI resout dans l'ordre de
# declaration, la route parametree capturait sinon ce chemin.
@router.get(
    "/feedback/stats",
    response_model=FeedbackStats,
    summary="Get feedback statistics"
)
async def get_feedback_stats(
    persona: Optional[str] = Query(None, description="Filter by persona"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db)
) -> FeedbackStats:
    """
    Get aggregated feedback statistics.

    **Parameters:**
    - **persona**: Optional persona filter
    - **start_date**: Optional start date (ISO format)
    - **end_date**: Optional end date (ISO format)

    **Returns:**
    - Total feedback count
    - Helpful/unhelpful counts
    - Satisfaction rate
    - Average rating
    - Breakdown by persona (if not filtered)

    **Raises:**
    - 400: Invalid persona or date format
    """
    try:
        # Parse dates
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None

        service = PersonaService(db)
        stats = await service.get_feedback_stats(persona, start, end)
        logger.info(f"📊 Feedback stats retrieved")
        return stats
    except InvalidPersonaError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidPersonaError", "message": str(e)}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": f"Invalid date format: {e}"}
        )
    except Exception as e:
        logger.error(f"❌ Error getting feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


@router.get(
    "/{persona}/stats",
    response_model=PersonaStatsResponse,
    summary="Get statistics for a persona",
    description="Retrieve usage, performance, engagement, and quality metrics for a specific persona"
)
async def get_persona_stats(
    persona: str = Path(..., description="Persona name"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD), defaults to 30 days ago"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
    db: AsyncSession = Depends(get_db)
) -> PersonaStatsResponse:
    """
    Get statistics for a specific persona over a date range.

    Queries aggregated data from persona_stats table.

    **Parameters:**
    - **persona**: Persona name
    - **start_date**: Start date (ISO format), defaults to 30 days ago
    - **end_date**: End date (ISO format), defaults to today

    **Returns:**
    - Usage metrics (questions, conversations, sessions)
    - Performance metrics (confidence, retrieval/generation time)
    - Engagement metrics (messages per conversation, session duration)
    - Quality metrics (helpful/unhelpful counts, satisfaction rate)

    **Raises:**
    - 400: Invalid persona or date format
    """
    assert isinstance(persona, str) and len(persona) > 0, "Persona must be a non-empty string"
    assert start_date is None or isinstance(start_date, str), "Start date must be a string or None"

    try:
        # Parse dates
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None

        service = PersonaService(db)
        stats = await service.get_persona_stats(persona, start, end)
        logger.info(f"📊 Retrieved stats for {persona}")
        return stats
    except InvalidPersonaError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidPersonaError", "message": str(e)}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": f"Invalid date format: {e}"}
        )
    except Exception as e:
        logger.error(f"❌ Error getting persona stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


@router.get(
    "/stats/all",
    response_model=List[PersonaStatsResponse],
    summary="Get statistics for all personas"
)
async def get_all_persona_stats(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db)
) -> List[PersonaStatsResponse]:
    """
    Get statistics for all 4 personas.

    **Parameters:**
    - **start_date**: Optional start date (ISO format)
    - **end_date**: Optional end date (ISO format)

    **Returns:**
    - List of stats for each persona

    **Raises:**
    - 400: Invalid date format
    """
    try:
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None

        service = PersonaService(db)
        all_stats = await service.get_all_persona_stats(start, end)
        logger.info(f"📊 Retrieved stats for all personas")
        return all_stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": f"Invalid date format: {e}"}
        )
    except Exception as e:
        logger.error(f"❌ Error getting all persona stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


@router.get(
    "/stats/usage",
    response_model=UsageBreakdownResponse,
    summary="Get persona usage breakdown",
    description="Get percentage breakdown of persona usage over last 30 days"
)
async def get_usage_breakdown(
    db: AsyncSession = Depends(get_db)
) -> UsageBreakdownResponse:
    """
    Get percentage breakdown of persona usage.

    Analyzes last 30 days of conversations.

    **Returns:**
    - Total conversations across all personas
    - Usage percentage for each persona

    """
    try:
        service = PersonaService(db)
        breakdown = await service.get_persona_usage_breakdown()

        # Calculate total from percentages (approximate)
        total = sum(breakdown.values())

        response = {
            "total_conversations": int(total),
            "percentages": breakdown
        }

        logger.info(f"📊 Usage breakdown retrieved")
        return response
    except Exception as e:
        logger.error(f"❌ Error getting usage breakdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


# ============================================================================
# ANALYTICS - ENGAGEMENT ENDPOINTS
# ============================================================================

@router.get(
    "/{persona}/engagement",
    response_model=EngagementMetricsResponse,
    summary="Get engagement metrics for persona"
)
async def get_engagement_metrics(
    persona: str = Path(..., description="Persona name"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze (1-365)"),
    db: AsyncSession = Depends(get_db)
) -> EngagementMetricsResponse:
    """
    Get engagement metrics for a persona over N days.

    **Parameters:**
    - **persona**: Persona name
    - **days**: Number of days to analyze (default 30, max 365)

    **Returns:**
    - Total conversations and messages
    - Average messages per conversation
    - Average session duration
    - Active days
    - Conversations per active day

    **Raises:**
    - 400: Invalid persona
    """
    assert isinstance(persona, str) and len(persona) > 0, "Persona must be a non-empty string"
    assert isinstance(days, int) and days > 0, "Days must be a positive integer"

    try:
        service = PersonaService(db)
        metrics = await service.get_engagement_metrics(persona, days)
        logger.info(f"📊 Engagement metrics for {persona} over {days} days")
        return metrics
    except InvalidPersonaError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidPersonaError", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Error getting engagement metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


@router.get(
    "/{persona}/questions",
    response_model=PopularQuestionsResponse,
    summary="Get popular questions for persona"
)
async def get_popular_questions(
    persona: str = Path(..., description="Persona name"),
    limit: int = Query(10, ge=1, le=50, description="Max number of questions (1-50)"),
    db: AsyncSession = Depends(get_db)
) -> PopularQuestionsResponse:
    """
    Get most frequently asked questions for a persona.

    Analyzes last 90 days of questions.

    **Parameters:**
    - **persona**: Persona name
    - **limit**: Max number of questions to return (default 10, max 50)

    **Returns:**
    - List of popular questions with frequency counts

    **Raises:**
    - 400: Invalid persona
    """
    assert isinstance(persona, str) and len(persona) > 0, "Persona must be a non-empty string"
    assert isinstance(limit, int) and limit > 0, "Limit must be a positive integer"

    try:
        service = PersonaService(db)
        questions = await service.get_popular_questions(persona, limit)

        response = {
            "persona": persona,
            "questions": questions,
            "total_count": len(questions)
        }

        logger.info(f"📊 Popular questions for {persona}: {len(questions)} found")
        return response
    except InvalidPersonaError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidPersonaError", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Error getting popular questions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


# ============================================================================
# ANALYTICS - COMPARISON & TRENDS ENDPOINTS
# ============================================================================

@router.get(
    "/compare/{metric}",
    response_model=PersonaComparisonResponse,
    summary="Compare personas by metric"
)
async def compare_personas(
    metric: str = Path(..., description="Metric to compare (usage|confidence|satisfaction)"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
) -> PersonaComparisonResponse:
    """
    Compare all personas across a specific metric.

    **Parameters:**
    - **metric**: Metric to compare (usage, confidence, satisfaction)
    - **days**: Number of days to analyze (default 30)

    **Returns:**
    - Comparison values for each persona
    - Date range analyzed

    **Raises:**
    - 400: Invalid metric
    """
    try:
        service = PersonaService(db)
        comparison = await service.compare_personas(metric, days)
        logger.info(f"📊 Compared personas by {metric} over {days} days")
        return comparison
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Error comparing personas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


@router.get(
    "/{persona}/trends/{metric}",
    response_model=TrendsResponse,
    summary="Get trend data for persona"
)
async def get_trends(
    persona: str = Path(..., description="Persona name"),
    metric: str = Path(..., description="Metric to track (questions|confidence|satisfaction)"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
) -> TrendsResponse:
    """
    Get time-series trend data for a persona metric.

    **Parameters:**
    - **persona**: Persona name
    - **metric**: Metric to track (questions, confidence, satisfaction)
    - **days**: Number of days to analyze (default 30)

    **Returns:**
    - List of data points with date and value
    - Metadata about persona, metric, and time range

    **Raises:**
    - 400: Invalid persona or metric
    """
    try:
        service = PersonaService(db)
        trends = await service.get_trends(persona, metric, days)

        response = {
            "persona": persona,
            "metric": metric,
            "days": days,
            "data_points": trends
        }

        logger.info(f"📊 Trends for {persona} ({metric}) over {days} days: {len(trends)} points")
        return response
    except InvalidPersonaError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidPersonaError", "message": str(e)}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Error getting trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@router.post(
    "/feedback",
    response_model=MessageFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add message feedback",
    responses={
        404: {"model": ErrorResponse, "description": "Message not found"},
        409: {"model": ErrorResponse, "description": "Feedback already exists"}
    }
)
async def add_message_feedback(
    feedback_data: MessageFeedbackCreate,
    db: AsyncSession = Depends(get_db)
) -> MessageFeedbackResponse:
    """
    Record user feedback on a message.

    Users can rate messages as helpful/unhelpful (👍/👎) with optional 1-5 star rating.

    **Parameters:**
    - **message_id**: ID of message to rate
    - **helpful**: Whether message was helpful (boolean)
    - **rating**: Optional 1-5 star rating (integer)
    - **comment**: Optional text comment (max 500 chars)

    **Returns:**
    - Created feedback with ID and timestamp

    **Raises:**
    - 404: Message not found
    - 409: Feedback already exists for this message
    - 400: Invalid rating (must be 1-5)
    """
    try:
        service = PersonaService(db)
        feedback = await service.add_message_feedback(
            message_id=feedback_data.message_id,
            helpful=feedback_data.helpful,
            rating=feedback_data.rating,
            comment=feedback_data.comment
        )
        logger.info(f"✅ Added feedback for message {feedback_data.message_id}")
        return feedback
    except MessageNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "MessageNotFoundError", "message": str(e)}
        )
    except FeedbackAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "FeedbackAlreadyExistsError", "message": str(e)}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Error adding feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


@router.get(
    "/feedback/{message_id}",
    response_model=Optional[MessageFeedbackResponse],
    summary="Get feedback for message",
    responses={
        200: {"description": "Feedback found"},
        404: {"description": "No feedback exists for this message"}
    }
)
async def get_message_feedback(
    message_id: int = Path(..., description="Message ID", gt=0),
    db: AsyncSession = Depends(get_db)
) -> Optional[MessageFeedbackResponse]:
    """
    Get feedback for a specific message.

    **Parameters:**
    - **message_id**: ID of message

    **Returns:**
    - Feedback data if exists, null otherwise
    """
    try:
        service = PersonaService(db)
        feedback = await service.get_message_feedback(message_id)

        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "NotFound", "message": f"No feedback found for message {message_id}"}
            )

        logger.info(f"📋 Retrieved feedback for message {message_id}")
        return feedback
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalServerError", "message": str(e)}
        )


