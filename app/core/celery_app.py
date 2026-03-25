"""
Celery application configuration for JuriX.

Configures Celery with Redis broker for asynchronous task processing.

Usage:
    # Start worker
    celery -A app.core.celery_app worker --loglevel=info

    # Monitor tasks
    celery -A app.core.celery_app flower

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import os

from celery import Celery

from app.core.config import settings

os.environ.setdefault("FORKED_BY_MULTIPROCESSING", "1")

logger = logging.getLogger(__name__)

# Create Celery app with Redis broker
celery_app = Celery(
    "jurix",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    include=["app.tasks.process_law"],  # Explicitly include task modules
)

logger.info("✅ Celery app configured")
