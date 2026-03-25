"""
Celery tasks package for JuriX.

Contains asynchronous tasks for:
- Document processing pipeline
- Background jobs
- Scheduled tasks

Author: JuriX Development Team
Date: 2026-01-11
"""

from app.core.celery_app import celery_app

__all__ = ["celery_app"]
