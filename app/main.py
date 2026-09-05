"""Point d'entrée FastAPI."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    admin,
    analytics,
    articles,
    auth,
    batch_upload,
    categories,
    classifier,
    language,
    laws,
    ocr,
    personas,
    rag,
    search,
    upload,
)
from app.core.config import settings
from app.core.database import close_db

logger = logging.getLogger(__name__)

# Valeur de repli declaree dans config.py. Si elle survit hors developpement,
# les JWT seraient signes avec une clé publiée dans le dépôt.
_DEV_SECRET_KEY = "dev_secret_key_change_in_production_with_openssl_rand_hex_32"


# Intervalle de purge. Le cache de recherche vit cinq minutes ; passer plus
# souvent ne libererait rien de plus, passer beaucoup moins souvent laisserait
# s'accumuler une heure de lignes mortes.
CACHE_CLEANUP_INTERVAL_S = 15 * 60


async def _purger_les_caches() -> None:
    """Boucle de menage. Ne doit jamais interrompre le service."""
    from app.core.database import AsyncSessionLocal
    from app.services.postgres_search_service import cleanup_expired_cache

    while True:
        try:
            await asyncio.sleep(CACHE_CLEANUP_INTERVAL_S)
            async with AsyncSessionLocal() as session:
                supprimees = await cleanup_expired_cache(session)
            if supprimees:
                logger.info(f"🧹 {supprimees} entrees de cache expirees supprimees")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"⚠️ Purge des caches impossible: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application."""
    if settings.ENVIRONMENT != "development" and settings.SECRET_KEY == _DEV_SECRET_KEY:
        raise RuntimeError(
            "SECRET_KEY est resté à sa valeur de développement alors que "
            f"ENVIRONMENT={settings.ENVIRONMENT!r}. Générez-en une avec "
            "`openssl rand -hex 32` et placez-la dans .env — sinon n'importe qui "
            "peut forger un jeton d'administration."
        )
    logger.info(f"🚀 {settings.APP_NAME} v{settings.VERSION} ({settings.ENVIRONMENT})")

    # Purge des caches expires, en tache de fond.
    #
    # `cleanup_expired_cache` existait, etait importee par search_service, et
    # n'etait APPELEE NULLE PART. `query_cache` et `embedding_cache` portent une
    # colonne `expires_at` que rien n'appliquait : les deux tables grossissaient
    # sans fin. Une colonne d'expiration non appliquee est pire que pas de
    # colonne du tout — elle laisse croire que le menage est fait.
    tache_menage = asyncio.create_task(_purger_les_caches())
    try:
        yield
    finally:
        tache_menage.cancel()
        with suppress(asyncio.CancelledError):
            await tache_menage
        # close_db() existait mais n'etait jamais appele : le pool de
        # connexions asyncpg n'etait jamais libere a l'arret. Dans le `finally`
        # pour qu'il s'execute meme si l'arret vient d'une exception.
        await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# CORS — allow localhost in dev, Vercel domain in prod
# Set ALLOWED_ORIGINS in your env for production (comma-separated)
_extra_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
_cors_origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
] + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "JuriX API v2.1",
        "status": "running",
        "version": settings.VERSION
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Include routers with global /api/v1 prefix
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(language.router, prefix="/api/v1/language", tags=["language"])
app.include_router(classifier.router, prefix="/api/v1/classifier", tags=["classifier"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(categories.router, prefix="/api/v1/categories", tags=["categories"])
app.include_router(articles.router, prefix="/api/v1/articles", tags=["articles"])
app.include_router(personas.router, prefix="/api/v1/personas", tags=["personas"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag", "chatbot"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["ocr"])
app.include_router(laws.router, prefix="/api/v1/laws", tags=["laws"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(batch_upload.router, prefix="/api/v1/batch-upload", tags=["batch"])
# Trigger reload

