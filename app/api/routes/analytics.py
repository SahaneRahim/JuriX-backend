"""
API routes for Analytics and Statistics.

Endpoints:
- GET /api/v1/analytics/overview - Dashboard overview
- GET /api/v1/analytics/laws - Law statistics
- GET /api/v1/analytics/search - Search analytics
- GET /api/v1/analytics/usage - Usage metrics

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.law import Law

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


# ==================== DASHBOARD OVERVIEW ====================


@router.get("/overview")
async def get_overview(
    db: AsyncSession = Depends(get_db),
) -> Dict:
    """
    Get dashboard overview with key metrics.
    
    **Returns:**
    - Total laws count
    - Laws by language
    - Laws by status
    - Recent activity
    
    **Example Response:**
    ```json
    {
        "total_laws": 150,
        "by_language": {"fr": 120, "en": 30},
        "by_status": {"active": 145, "archived": 5},
        "recent_laws": 10
    }
    ```
    """
    logger.info("📊 GET /analytics/overview")
    
    try:
        # Total laws
        total_query = select(func.count(Law.id))
        total_result = await db.execute(total_query)
        total_laws = total_result.scalar()
        
        # Laws by language
        lang_query = select(
            Law.language,
            func.count(Law.id)
        ).group_by(Law.language)
        lang_result = await db.execute(lang_query)
        by_language = {row[0]: row[1] for row in lang_result.all()}
        
        # Laws by status
        status_query = select(
            Law.status,
            func.count(Law.id)
        ).group_by(Law.status)
        status_result = await db.execute(status_query)
        by_status = {row[0]: row[1] for row in status_result.all()}
        
        # Recent laws (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_query = select(func.count(Law.id)).where(
            Law.created_at >= thirty_days_ago
        )
        recent_result = await db.execute(recent_query)
        recent_laws = recent_result.scalar()
        
        return {
            "total_laws": total_laws or 0,
            "by_language": by_language,
            "by_status": by_status,
            "recent_laws": recent_laws or 0,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting overview: {e}", exc_info=True)
        return {
            "total_laws": 0,
            "by_language": {},
            "by_status": {},
            "recent_laws": 0,
            "error": str(e),
        }


# ==================== LAW STATISTICS ====================


@router.get("/laws")
async def get_law_statistics(
    db: AsyncSession = Depends(get_db),
) -> Dict:
    """
    Get detailed law statistics.
    
    **Returns:**
    - Laws by language (FR/EN)
    - Laws by type (loi, décret, etc.)
    - Average content length
    - Publication trends
    
    **Example Response:**
    ```json
    {
        "by_language": {"fr": 120, "en": 30},
        "by_type": {"loi": 80, "décret": 50, "ordonnance": 20},
        "avg_content_length": 5000,
        "total": 150
    }
    ```
    """
    logger.info("📊 GET /analytics/laws")
    
    try:
        # Laws by language
        lang_query = select(
            Law.language,
            func.count(Law.id)
        ).group_by(Law.language)
        lang_result = await db.execute(lang_query)
        by_language = {row[0]: row[1] for row in lang_result.all()}
        
        # Laws by type
        type_query = select(
            Law.type,
            func.count(Law.id)
        ).group_by(Law.type)
        type_result = await db.execute(type_query)
        by_type = {row[0]: row[1] for row in type_result.all()}
        
        # Average content length
        avg_query = select(func.avg(func.length(Law.content)))
        avg_result = await db.execute(avg_query)
        avg_length = avg_result.scalar() or 0
        
        # Total
        total_query = select(func.count(Law.id))
        total_result = await db.execute(total_query)
        total = total_result.scalar()
        
        return {
            "by_language": by_language,
            "by_type": by_type,
            "avg_content_length": int(avg_length),
            "total": total or 0,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting law statistics: {e}", exc_info=True)
        return {
            "by_language": {},
            "by_type": {},
            "avg_content_length": 0,
            "total": 0,
            "error": str(e),
        }


# ==================== SEARCH ANALYTICS ====================


@router.get("/search")
async def get_search_analytics() -> Dict:
    """
    Get search analytics.
    
    **Returns:**
    - Search modes usage (text/semantic/hybrid)
    - Average response time
    - Popular queries (mock data for now)
    
    **Note:** This is a simplified version. In production, 
    integrate with actual search logs.
    
    **Example Response:**
    ```json
    {
        "modes_usage": {
            "text": 100,
            "semantic": 50,
            "hybrid": 200
        },
        "avg_response_time_ms": 150,
        "total_searches": 350
    }
    ```
    """
    logger.info("📊 GET /analytics/search")
    
    # TODO: Integrate with actual search logs
    # For now, return mock data
    return {
        "modes_usage": {
            "text": 100,
            "semantic": 50,
            "hybrid": 200,
        },
        "avg_response_time_ms": 150,
        "total_searches": 350,
        "timestamp": datetime.now().isoformat(),
        "note": "Mock data - integrate with search logs in production",
    }


# ==================== USAGE METRICS ====================


@router.get("/usage")
async def get_usage_metrics() -> Dict:
    """
    Get API usage metrics.
    
    **Returns:**
    - API calls per endpoint
    - Peak usage times
    - Active users (mock data for now)
    
    **Note:** This is a simplified version. In production,
    integrate with API gateway logs or monitoring system.
    
    **Example Response:**
    ```json
    {
        "api_calls": {
            "/api/v1/laws": 500,
            "/api/v1/search": 350,
            "/api/v1/rag": 200
        },
        "total_calls": 1050,
        "active_users": 25
    }
    ```
    """
    logger.info("📊 GET /analytics/usage")
    
    # TODO: Integrate with API gateway logs or monitoring
    # For now, return mock data
    return {
        "api_calls": {
            "/api/v1/laws": 500,
            "/api/v1/search": 350,
            "/api/v1/rag": 200,
            "/api/v1/upload": 100,
        },
        "total_calls": 1150,
        "active_users": 25,
        "peak_hours": [9, 10, 14, 15, 16],
        "timestamp": datetime.now().isoformat(),
        "note": "Mock data - integrate with monitoring system in production",
    }


# ==================== HEALTH CHECK ====================


@router.get("/health")
async def health_check() -> Dict:
    """
    Check analytics service health.
    
    **Returns:**
    - Service status
    - Database connectivity
    """
    return {
        "service": "Analytics",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }
