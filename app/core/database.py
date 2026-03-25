"""
Gestion des sessions de base de données asynchrones.

Ce module configure SQLAlchemy 2.0 avec sessions asynchrones pour PostgreSQL.
Utilise le pattern AsyncSession pour toutes les opérations DB.

Usage dans routes FastAPI:
    @router.get("/laws")
    async def list_laws(db: AsyncSession = Depends(get_db)):
        service = LawService(db)
        return await service.list_laws()

Author: JuriX Team
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

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
        ...     service = LawService(db)
        ...     return await service.list_laws()
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("📦 Nouvelle session DB créée")
            yield session
        except Exception as e:
            logger.error(f"❌ Erreur session DB: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
            logger.debug("🗑️  Session DB fermée")


async def init_db() -> None:
    """
    Initialise les tables de la base de données.

    Crée toutes les tables définies dans les modèles SQLAlchemy
    si elles n'existent pas déjà.

    Note:
        En production, utiliser Alembic pour les migrations.
        Cette fonction est principalement pour les tests.

    Example:
        >>> # Dans les tests
        >>> await init_db()
    """
    async with engine.begin() as conn:
        logger.info("🚀 Initialisation des tables de la base de données...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Tables créées avec succès")


async def drop_db() -> None:
    """
    Supprime toutes les tables de la base de données.

    ⚠️  ATTENTION: Cette fonction est destructive!
    Utilisée uniquement pour les tests.

    Example:
        >>> # Nettoyage après tests
        >>> await drop_db()
    """
    async with engine.begin() as conn:
        logger.warning("⚠️  Suppression de toutes les tables...")
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("✅ Tables supprimées")


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
    logger.info("✅ Moteur de base de données fermé")


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
            await session.execute("SELECT 1")
            logger.debug("✅ DB health check OK")
            return True
    except Exception as e:
        logger.error(f"❌ DB health check failed: {e}")
        return False
