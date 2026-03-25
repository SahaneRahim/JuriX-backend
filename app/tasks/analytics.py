"""
Celery tasks for persona analytics aggregation.

Daily task to aggregate conversation and message data into persona_stats table.
Runs at 00:05 UTC daily to process previous day's data.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from celery import Celery
from celery.schedules import crontab

from app.core.database import AsyncSessionLocal
from app.services.persona_service import PersonaService

# Configure logger
logger = logging.getLogger(__name__)

# Initialize Celery app
# Note: This should be configured with your broker URL (Redis/RabbitMQ)
celery_app = Celery(
    'jurix_tasks',
    broker='redis://localhost:6379/0',  # Update with your broker URL
    backend='redis://localhost:6379/0'
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


# ============================================================================
# PERSONA ANALYTICS AGGREGATION TASK
# ============================================================================

@celery_app.task(name="aggregate_persona_stats", bind=True, max_retries=3)
def aggregate_persona_stats_task(self, target_date: Optional[str] = None):
    """
    Aggregate persona statistics for a specific date.

    Processes conversation and message data to calculate:
    - Usage metrics (questions, conversations, sessions)
    - Performance metrics (confidence, response times)
    - Engagement metrics (messages per conversation, session duration)
    - Quality metrics (helpful/unhelpful feedback, satisfaction rate)

    Args:
        target_date: Date to aggregate (YYYY-MM-DD). Defaults to yesterday.

    Returns:
        Dict with aggregation summary

    Scheduled: Daily at 00:05 UTC via Celery Beat

    Example:
        # Manual execution
        aggregate_persona_stats_task.delay()
        aggregate_persona_stats_task.delay("2024-01-15")
    """
    try:
        logger.info("📊 Starting persona stats aggregation task")

        # Parse target date
        if target_date:
            try:
                date_obj = date.fromisoformat(target_date)
                logger.info(f"📅 Aggregating data for: {date_obj}")
            except ValueError as e:
                logger.error(f"❌ Invalid date format: {target_date}")
                raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {target_date}")
        else:
            # Default to yesterday
            date_obj = date.today() - timedelta(days=1)
            logger.info(f"📅 No date specified, aggregating yesterday: {date_obj}")

        # Run async aggregation
        result = asyncio.run(_run_aggregation(date_obj))

        logger.info(f"✅ Aggregation complete: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Aggregation task failed: {e}", exc_info=True)
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


async def _run_aggregation(target_date: date) -> dict:
    """
    Execute aggregation within async context.

    Args:
        target_date: Date to aggregate

    Returns:
        Dict with aggregation results
    """
    start_time = datetime.utcnow()

    async with AsyncSessionLocal() as db:
        try:
            persona_service = PersonaService(db)

            # Aggregate stats for target date
            await persona_service.aggregate_daily_stats(target_date)

            await db.commit()

            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            logger.info(f"✅ Aggregation completed in {duration_ms}ms")

            return {
                "status": "success",
                "date": target_date.isoformat(),
                "duration_ms": duration_ms,
                "timestamp": end_time.isoformat()
            }

        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Aggregation failed: {e}")
            raise


# ============================================================================
# CELERY BEAT SCHEDULE
# ============================================================================

celery_app.conf.beat_schedule = {
    'aggregate-persona-stats-daily': {
        'task': 'aggregate_persona_stats',
        'schedule': crontab(hour=0, minute=5),  # Run at 00:05 UTC daily
        'options': {
            'expires': 3600,  # Task expires after 1 hour if not picked up
        }
    },
}


# ============================================================================
# MANUAL EXECUTION UTILITIES
# ============================================================================

def aggregate_date_range(start_date: date, end_date: date):
    """
    Manually aggregate a range of dates (for backfilling).

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Example:
        from app.tasks.analytics import aggregate_date_range
        from datetime import date

        aggregate_date_range(
            date(2024, 1, 1),
            date(2024, 1, 31)
        )
    """
    current_date = start_date
    tasks = []

    logger.info(f"📊 Backfilling persona stats from {start_date} to {end_date}")

    while current_date <= end_date:
        logger.info(f"📅 Queuing aggregation for {current_date}")
        task = aggregate_persona_stats_task.delay(current_date.isoformat())
        tasks.append(task)
        current_date += timedelta(days=1)

    logger.info(f"✅ Queued {len(tasks)} aggregation tasks")
    return tasks


def aggregate_yesterday():
    """
    Manually trigger aggregation for yesterday.

    Useful for testing or manual runs.

    Example:
        from app.tasks.analytics import aggregate_yesterday
        aggregate_yesterday()
    """
    yesterday = date.today() - timedelta(days=1)
    logger.info(f"📊 Manually aggregating data for {yesterday}")
    return aggregate_persona_stats_task.delay(yesterday.isoformat())


# ============================================================================
# HEALTH CHECK
# ============================================================================

@celery_app.task(name="analytics_health_check")
def health_check_task():
    """
    Health check task for analytics worker.

    Returns:
        Dict with worker status
    """
    return {
        "status": "healthy",
        "service": "analytics_worker",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# TASK MONITORING
# ============================================================================

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery configuration."""
    logger.info(f"Request: {self.request!r}")
    return {
        "status": "ok",
        "worker_id": self.request.id,
        "hostname": self.request.hostname
    }


if __name__ == "__main__":
    """
    For testing/debugging aggregation locally without Celery.

    Run: python -m app.tasks.analytics
    """
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = None

    print(f"🧪 Testing aggregation task locally")
    result = aggregate_persona_stats_task(target)
    print(f"✅ Result: {result}")
