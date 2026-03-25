"""Point d'entrée FastAPI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import language, classifier, articles, search, rag, categories, personas, upload, ocr, laws, analytics, admin, batch_upload

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG
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


# Include routers
app.include_router(language.router)
app.include_router(classifier.router)
app.include_router(articles.router)
app.include_router(
    search.router,
    prefix="/api/v1/search",
    tags=["search"]
)
app.include_router(
    rag.router,
    prefix="/api/v1/rag",
    tags=["rag", "chatbot"]
)
app.include_router(
    categories.router,
    tags=["categories"]
)
app.include_router(
    personas.router,
    tags=["personas"]
)
app.include_router(upload.router)
app.include_router(ocr.router)
app.include_router(laws.router)
app.include_router(analytics.router)
app.include_router(admin.router)
app.include_router(batch_upload.router)  # Batch upload with WebSocket
# Trigger reload

