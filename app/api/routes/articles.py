"""
Article extraction API endpoint (utility).

For testing/debugging article extraction on arbitrary text.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time

from app.utils.text_chunker import extract_articles, ArticleExtractionError


router = APIRouter(prefix="/api/v1/articles", tags=["articles"])


class ExtractRequest(BaseModel):
    """Request model for article extraction."""
    text: str = Field(..., min_length=200, max_length=1_000_000, description="Legal document text")
    min_article_length: int = Field(default=10, ge=5, le=1000, description="Minimum article length in characters")
    preserve_formatting: bool = Field(default=False, description="Preserve original formatting")
    strict: bool = Field(default=True, description="Strict validation (require minimum 3 articles)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Article 1. Dispositions générales\nLa présente loi régit...\n\nArticle 2. Champ d'application\nLes dispositions s'appliquent...\n\nArticle 3. Définitions\nAu sens de la présente loi...",
                "min_article_length": 10,
                "preserve_formatting": False,
                "strict": True
            }
        }


class ArticleResult(BaseModel):
    """Single extracted article."""
    number: str
    title: str | None
    content: str
    position: int
    parent_id: str | None
    section: str | None
    word_count: int
    char_count: int


class ExtractResponse(BaseModel):
    """Response model for article extraction."""
    articles: List[Dict[str, Any]]
    count: int
    processing_time_ms: int

    class Config:
        json_schema_extra = {
            "example": {
                "articles": [
                    {
                        "number": "1",
                        "title": "Dispositions générales",
                        "content": "La présente loi régit...",
                        "position": 0,
                        "parent_id": None,
                        "section": None,
                        "word_count": 45,
                        "char_count": 234
                    }
                ],
                "count": 3,
                "processing_time_ms": 125
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    service: str
    status: str
    version: str


@router.post("/extract", response_model=ExtractResponse, summary="Extract articles from legal text")
async def extract_articles_endpoint(request: ExtractRequest) -> ExtractResponse:
    """
    Extract articles from legal document text.

    This utility endpoint allows testing/debugging of article extraction
    on arbitrary legal text without uploading full documents.

    **Supported Patterns:**
    - Article X. / Article X:
    - Art. X / Art X.
    - Article premier / Article première
    - Hierarchical: Article 1.1, Article 1.2.3

    **Validation:**
    - Text length: 200 characters - 1MB
    - Minimum 3 articles (if strict=True)
    - Articles must have minimum length

    **Returns:**
    - List of extracted articles with metadata
    - Processing time in milliseconds
    - Total article count
    """
    assert request is not None, "Extract request must not be None"
    assert isinstance(request.text, str) and len(request.text) >= 200, "Text must be at least 200 characters"

    try:
        start = time.time()

        articles = extract_articles(
            text=request.text,
            min_article_length=request.min_article_length,
            preserve_formatting=request.preserve_formatting,
            strict=request.strict
        )

        elapsed_ms = int((time.time() - start) * 1000)

        return ExtractResponse(
            articles=articles,
            count=len(articles),
            processing_time_ms=elapsed_ms
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid input",
                "message": str(e),
                "type": "ValueError"
            }
        )
    except ArticleExtractionError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Extraction failed",
                "message": str(e),
                "type": "ArticleExtractionError"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"Unexpected error: {str(e)}",
                "type": type(e).__name__
            }
        )


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """
    Health check for article extraction service.

    Returns service status and version information.
    """
    return HealthResponse(
        service="ArticleExtractor",
        status="healthy",
        version="1.0.0"
    )
