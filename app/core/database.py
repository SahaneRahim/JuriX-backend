"""
Gestion des sessions de base de données asynchrones.

Ce module configure SQLAlchemy 2.0 avec sessions asynchrones pour PostgreSQL.
Utilise le pattern AsyncSession pour toutes les opérations DB.

Usage dans routes FastAPI:
    @router.get("/laws")
    async def list_laws(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Law))
        return result.scalars().all()

Author: JuriX Team
"""

import logging
from typing import AsyncGenerator

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


# Base class for all SQLAlchemy models
class Base(DeclarativeBase):
    """Base class pour tous les modèles SQLAlchemy."""
    pass


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG if hasattr(settings, 'DEBUG') else False,  # Log SQL queries in debug mode
    pool_size=20,  # Connection pool size
    max_overflow=10,  # Additional connections beyond pool_size
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autoflush=False,  # Manual flush control
    autocommit=False,  # Manual commit control
)


# ---------------------------------------------------------------------------
# Moteur SYNCHRONE partage
# ---------------------------------------------------------------------------
# Le pipeline de traitement (app/tasks/process_law.py) et le cache
# d'embeddings tournent dans un thread d'executor, hors boucle d'evenements :
# ils ont besoin d'un moteur synchrone. Onze appels a create_engine() etaient
# dissemines dans le code, un par fonction ou par instance, aucun n'appelait
# jamais dispose() : chaque loi traitee laissait un pool de connexions ouvert
# jusqu'a l'arret du processus.
#
# Les consommateurs doivent utiliser SyncSessionLocal(), JAMAIS sync_engine
# directement : c'est ce qui permet de rebrancher toute l'application sur une
# autre base avec sessionmaker.configure(bind=...), ce dont les tests dependent.
SYNC_DATABASE_URL = settings.DATABASE_URL.replace("+asyncpg", "")

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency pour obtenir une session DB asynchrone.

    Crée une nouvelle session pour chaque requête et garantit
    sa fermeture propre même en cas d'erreur.

    Yields:
        AsyncSession: Session de base de données asynchrone

    Example:
        >>> @router.get("/laws")
        >>> async def list_laws(db: AsyncSession = Depends(get_db)):
        ...     result = await db.execute(select(Law))
        ...     return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("📦 Nouvelle session DB créée")
            yield session
        except HTTPException:
            # Une HTTPException n'est pas une erreur de base : c'est une
            # reponse deliberee de la route. La journaliser en « Erreur session
            # DB » avec sa trace complete noyait les vraies erreurs SQL sous des
            # 404 et des 429 parfaitement normaux. On annule quand meme la
            # transaction, sans mentir sur la cause.
            await session.rollback()
            raise
        except Exception as e:
            logger.error(f"❌ Erreur session DB: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
            logger.debug("🗑️  Session DB fermée")




# init_db() et drop_db() ont ete retirees : elles appelaient
# Base.metadata.create_all/drop_all, et n'etaient appelees par personne. Le
# schema appartient a Alembic — deux sources de verite pour une meme structure
# finissent toujours par diverger.


async def close_db() -> None:
    """
    Ferme le moteur de base de données.

    Ferme toutes les connexions actives et libère les ressources.
    Doit être appelé lors de l'arrêt de l'application.

    Example:
        >>> # Dans le shutdown event de FastAPI
        >>> @app.on_event("shutdown")
        >>> async def shutdown():
        ...     await close_db()
    """
    await engine.dispose()
    close_sync_db()
    logger.info("✅ Moteur de base de données fermé")


def close_sync_db() -> None:
    """Ferme le moteur synchrone partage. Appele par close_db()."""
    sync_engine.dispose()
    logger.info("✅ Moteur synchrone fermé")


async def health_check_db() -> bool:
    """
    Vérifie la santé de la connexion à la base de données.

    Returns:
        bool: True si la connexion fonctionne, False sinon

    Example:
        >>> is_healthy = await health_check_db()
        >>> if not is_healthy:
        ...     logger.error("DB connection failed")
    """
    try:
        async with AsyncSessionLocal() as session:
            # text() obligatoire : SQLAlchemy 2.0 refuse une chaine brute
            # (ObjectNotExecutableError). L'exception etait avalee par le except
            # ci-dessous, donc health_check_db renvoyait TOUJOURS False et
            # GET /api/v1/admin/system declarait la base en panne en permanence.
            await session.execute(text("SELECT 1"))
            logger.debug("✅ DB health check OK")
            return True
    except Exception as e:
        logger.error(f"❌ DB health check failed: {e}")
        return False
