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
from typing import Dict

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_admin_user
from app.core.database import get_db
from app.models.law import Law
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analytics"])


# ==================== DASHBOARD OVERVIEW ====================


@router.get("/overview")
async def get_overview(
    db: AsyncSession = Depends(get_db),
    # Le tableau de bord est reserve aux administrateurs, et le front envoyait
    # deja un jeton. La route, elle, repondait 200 a n'importe qui et exposait
    # la composition du corpus et les dernieres lois traitees.
    _: User = Depends(get_current_admin_user),
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
    # Le tableau de bord est reserve aux administrateurs, et le front envoyait
    # deja un jeton. La route, elle, repondait 200 a n'importe qui et exposait
    # la composition du corpus et les dernieres lois traitees.
    _: User = Depends(get_current_admin_user),
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
async def get_search_analytics(
    days: int = Query(7, ge=1, le=90, description="Fenetre d'observation, en jours"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
) -> Dict:
    """
    Statistiques de recherche, MESUREES.

    Cette route rendait des constantes codees en dur — 350 recherches, 150 ms,
    une repartition 100/50/200 entre les modes — que le tableau de bord admin
    affichait comme des mesures. Le champ `note: "Mock data"` qui les
    accompagnait n'etait lu par personne.

    Les chiffres viennent maintenant de `search_events`, alimentee a chaque
    recherche. Ils seront petits au debut : c'est le but.
    """
    fenetre = {"days": days}

    total = (await db.execute(text(
        "SELECT count(*) FROM search_events WHERE created_at >= now() - make_interval(days => :days)"
    ), fenetre)).scalar_one()

    modes = {
        ligne.mode: ligne.n
        for ligne in (await db.execute(text("""
            SELECT mode, count(*) AS n FROM search_events
            WHERE created_at >= now() - make_interval(days => :days)
            GROUP BY mode ORDER BY n DESC
        """), fenetre)).all()
    }

    # Mediane et non moyenne : une seule requete froide a 3 secondes deplace la
    # moyenne et ne dit rien de l'experience courante.
    latences = (await db.execute(text("""
        SELECT
            percentile_disc(0.5) WITHIN GROUP (ORDER BY duration_ms)  AS mediane,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95,
            max(duration_ms)                                          AS maximum
        FROM search_events
        WHERE created_at >= now() - make_interval(days => :days) AND NOT cached
    """), fenetre)).first()

    caches = (await db.execute(text("""
        SELECT count(*) FILTER (WHERE cached) AS depuis_cache, count(*) AS total
        FROM search_events WHERE created_at >= now() - make_interval(days => :days)
    """), fenetre)).first()

    # Les requetes SANS resultat sont la statistique la plus utile du lot :
    # elles disent ce que les gens cherchent et que le corpus ne contient pas.
    sans_resultat = [
        {"query": r.query, "count": r.n}
        for r in (await db.execute(text("""
            SELECT query, count(*) AS n FROM search_events
            WHERE created_at >= now() - make_interval(days => :days) AND results_count = 0
            GROUP BY query ORDER BY n DESC, query LIMIT 10
        """), fenetre)).all()
    ]

    populaires = [
        {"query": r.query, "count": r.n}
        for r in (await db.execute(text("""
            SELECT query, count(*) AS n FROM search_events
            WHERE created_at >= now() - make_interval(days => :days)
            GROUP BY query ORDER BY n DESC, query LIMIT 10
        """), fenetre)).all()
    ]

    taux_cache = round(caches.depuis_cache / caches.total * 100, 1) if caches.total else 0.0

    return {
        "window_days": days,
        "total_searches": total,
        "modes_usage": modes,
        "median_response_time_ms": latences.mediane or 0,
        "p95_response_time_ms": latences.p95 or 0,
        "max_response_time_ms": latences.maximum or 0,
        "cache_hit_rate_percent": taux_cache,
        "popular_queries": populaires,
        "queries_without_results": sans_resultat,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/usage")
async def get_usage_metrics(
    days: int = Query(7, ge=1, le=90, description="Fenetre d'observation, en jours"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin_user),
) -> Dict:
    """
    Metriques d'usage, MESUREES.

    Cette route rendait `total_calls: 1150`, `active_users: 25` et des heures de
    pointe inventees. Tout vient desormais de `conversations`, `messages`,
    `users` et `search_events`, qui portaient deja ces informations.
    """
    fenetre = {"days": days}

    conversations = (await db.execute(text(
        "SELECT count(*) FROM conversations WHERE created_at >= now() - make_interval(days => :days)"
    ), fenetre)).scalar_one()

    messages = (await db.execute(text("""
        SELECT count(*) FILTER (WHERE role = 'user')      AS questions,
               count(*) FILTER (WHERE role = 'assistant') AS reponses
        FROM messages WHERE created_at >= now() - make_interval(days => :days)
    """), fenetre)).first()

    recherches = (await db.execute(text(
        "SELECT count(*) FROM search_events WHERE created_at >= now() - make_interval(days => :days)"
    ), fenetre)).scalar_one()

    # Un « utilisateur actif » est ici quelqu'un qui a ouvert une conversation.
    # Les visiteurs anonymes ne sont pas comptes : rien ne les identifie, et en
    # inventer un compte serait revenir au probleme que cette route corrige.
    actifs = (await db.execute(text("""
        SELECT count(DISTINCT user_id) FROM conversations
        WHERE user_id IS NOT NULL AND created_at >= now() - make_interval(days => :days)
    """), fenetre)).scalar_one()

    heures = [
        {"hour": int(r.heure), "count": r.n}
        for r in (await db.execute(text("""
            SELECT extract(hour FROM created_at) AS heure, count(*) AS n
            FROM (
                SELECT created_at FROM messages
                WHERE created_at >= now() - make_interval(days => :days)
                UNION ALL
                SELECT created_at FROM search_events
                WHERE created_at >= now() - make_interval(days => :days)
            ) activite
            GROUP BY heure ORDER BY n DESC, heure LIMIT 5
        """), fenetre)).all()
    ]

    latences = (await db.execute(text("""
        SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY retrieval_time_ms + generation_time_ms)
        FROM messages
        WHERE role = 'assistant' AND created_at >= now() - make_interval(days => :days)
          AND retrieval_time_ms IS NOT NULL AND generation_time_ms IS NOT NULL
    """), fenetre)).scalar()

    personas = {
        r.persona: r.n
        for r in (await db.execute(text("""
            SELECT persona, count(*) AS n FROM conversations
            WHERE created_at >= now() - make_interval(days => :days)
            GROUP BY persona ORDER BY n DESC
        """), fenetre)).all()
    }

    return {
        "window_days": days,
        "conversations": conversations,
        "questions_asked": messages.questions or 0,
        "answers_generated": messages.reponses or 0,
        "searches": recherches,
        "active_users": actifs,
        "peak_hours": heures,
        "median_answer_time_ms": latences or 0,
        "personas_usage": personas,
        "timestamp": datetime.now().isoformat(),
    }


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
