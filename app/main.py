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


# Include routers with global /api/v1 prefix
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

