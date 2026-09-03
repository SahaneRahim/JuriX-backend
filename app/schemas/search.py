"""
Pydantic schemas for Search operations.

Provides request/response models for hybrid search (text + semantic + RRF):
- SearchFilters: Filter parameters for search queries
- SearchRequest: Main search request with query, mode, filters
- ArticleMatch: Matched article with snippet and relevance
- SearchResult: Single search result with highlights
- SearchResponse: Complete search response with metadata
- SearchStats: Admin statistics for search operations

Author: JuriX Development Team
Date: 2026-01-10
"""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Filter Schemas
# ============================================================================

class SearchFilters(BaseModel):
    """
    Filter parameters for search operations.

    Supports filtering by:
    - Language (fr/en)
    - Categories (multiple IDs)
    - Document types
    - Status (draft, published, archived)
    - Publication date range
    """

    language: Optional[str] = Field(
        None,
        description="Filter by language code (fr or en)"
    )
    category_ids: Optional[List[int]] = Field(
        None,
        description="Filter by category IDs"
    )
    types: Optional[List[str]] = Field(
        None,
        description="Filter by law types (loi, décret, ordonnance, arrêté)"
    )
    status: Optional[str] = Field(
        None,
        description="Filter by status (draft, published, archived)"
    )
    year_from: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="Filter laws from this publication year"
    )
    year_to: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="Filter laws up to this publication year"
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language code."""
        if v is None:
            return v
        if v.lower() not in {"fr", "en"}:
            raise ValueError(f"Language must be 'fr' or 'en'. Got: {v}")
        return v.lower()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate status."""
        if v is None:
            return v
        allowed = {"draft", "published", "archived"}
        if v.lower() not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}. Got: {v}")
        return v.lower()

    @field_validator("year_to")
    @classmethod
    def validate_year_range(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure year_to >= year_from."""
        if v is None:
            return v
        year_from = info.data.get("year_from")
        if year_from is not None and v < year_from:
            raise ValueError(f"year_to ({v}) must be >= year_from ({year_from})")
        return v


# ============================================================================
# Request Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """
    Request schema for search endpoint.

    Supports three search modes:
    - text: Full-text search only (Meilisearch)
    - semantic: Vector search only (pgvector)
    - hybrid: Combined search with RRF fusion (default, recommended)
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query text"
    )
    mode: str = Field(
        "hybrid",
        description="Search mode: text, semantic, or hybrid"
    )
    filters: Optional[SearchFilters] = Field(
        None,
        description="Optional filters to apply"
    )
    limit: int = Field(
        15,
        ge=1,
        le=50,
        description="Maximum number of results to return (1-50)"
    )
    offset: int = Field(
        0,
        ge=0,
        description="Number of results to skip (for pagination)"
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate search mode."""
        allowed_modes = {"text", "semantic", "hybrid"}
        if v.lower() not in allowed_modes:
            raise ValueError(
                f"Mode must be one of: {', '.join(allowed_modes)}. Got: {v}"
            )
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "responsabilité dirigeants société",
                "mode": "hybrid",
                "filters": {
                    "category_ids": [3, 7],
                    "types": ["loi", "décret"],
                    "language": "fr",
                    "status": "published",
                    "year_from": 2020
                },
                "limit": 15,
                "offset": 0
            }
        }


# ============================================================================
# Result Schemas
# ============================================================================

class ArticleMatch(BaseModel):
    """
    Matched article within a law with highlighted snippet.

    Represents an article that matches the search query
    with its relevance score and content snippet.
    """

    article_id: int = Field(..., description="Article ID")
    number: str = Field(..., description="Article number (e.g., '1er', '42')")
    title: Optional[str] = Field(None, description="Article title (optional)")
    content_snippet: str = Field(..., description="Highlighted content snippet")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0-1.0)"
    )


class SearchResult(BaseModel):
    """
    Single search result with law information and highlights.

    Contains complete law metadata, relevance score,
    matched articles, and highlighted text snippets.
    """

    law_id: int = Field(..., description="Law ID")
    reference: str = Field(..., description="Law reference (e.g., LOI-2024-001)")
    title: str = Field(..., description="Law title")
    type: str = Field(..., description="Law type")
    language: Optional[str] = Field(None, description="Language code (fr/en)")
    status: str = Field(..., description="Status")
    category_id: Optional[int] = Field(None, description="Category ID")
    category_name: Optional[str] = Field(None, description="Category name")
    publication_date: Optional[date] = Field(None, description="Publication date")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall relevance score (0.0-1.0)"
    )
    matched_articles: List[ArticleMatch] = Field(
        default_factory=list,
        description="List of matched articles within this law"
    )
    highlights: Dict[str, str] = Field(
        default_factory=dict,
        description="Highlighted text snippets (title, content)"
    )
    content: Optional[str] = Field(
        None,
        description="Full article content for RAG context"
    )

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """
    Complete search response with results and metadata.

    Contains search results, total count, execution time,
    and applied filters for client display.
    """

    query: str = Field(..., description="Original search query")
    mode: str = Field(..., description="Search mode used")
    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results"
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of results found"
    )
    search_time_ms: int = Field(
        ...,
        ge=0,
        description="Search execution time in milliseconds"
    )
    filters_applied: Optional[Dict] = Field(
        None,
        description="Filters that were applied to this search"
    )
    # Article-specific navigation fields
    target_article: Optional[str] = Field(
        None,
        description="Article number/reference to scroll to (e.g., '5', 'PREMIER')"
    )
    direct_navigation: bool = Field(
        False,
        description="If true, client should redirect directly to the document"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "responsabilité dirigeants",
                "mode": "hybrid",
                "results": [
                    {
                        "law_id": 156,
                        "reference": "Code OHADA",
                        "title": "Acte uniforme relatif au droit des sociétés",
                        "type": "code",
                        "language": "fr",
                        "status": "published",
                        "category_id": 3,
                        "category_name": "Droit Commercial OHADA",
                        "relevance_score": 0.92,
                        "matched_articles": [],
                        "highlights": {
                            "title": "Acte uniforme relatif au droit des <mark>sociétés</mark>",
                            "content": "...la <mark>responsabilité</mark> des <mark>dirigeants</mark>..."
                        }
                    }
                ],
                "total": 15,
                "search_time_ms": 187,
                "filters_applied": {
                    "language": "fr",
                    "status": "published"
                }
            }
        }


# ============================================================================
# Admin Statistics Schemas
# ============================================================================

class SearchStats(BaseModel):
    """
    Statistiques de l'index de recherche (reserve aux administrateurs).

    Reecrit pour PostgreSQL : les champs decrivaient Meilisearch et Redis,
    supprimes de l'architecture. Les quatre champs ajoutes (indexed_documents,
    total_articles, articles_with_embeddings, cache_entries) sont ceux qui
    disent reellement si la chaine d'ingestion a fonctionne.
    """

    total_documents: int = Field(0, ge=0, description="Nombre total de lois")
    indexed_documents: int = Field(
        0, ge=0, description="Lois dont le search_vector est renseigné"
    )
    total_articles: int = Field(0, ge=0, description="Nombre total d'articles")
    articles_with_embeddings: int = Field(
        0, ge=0, description="Articles disposant d'un vecteur d'embedding"
    )
    by_language: Dict[str, int] = Field(
        default_factory=dict, description="Répartition par langue"
    )
    by_category: Dict[str, int] = Field(
        default_factory=dict, description="Répartition par catégorie"
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict, description="Répartition par statut"
    )
    index_health: str = Field(
        "unknown",
        description="empty | healthy (tout est indexé) | degraded (indexation partielle)",
    )
    cache_entries: int = Field(
        0, ge=0, description="Entrées non expirées dans query_cache"
    )
    cache_status: str = Field(
        "unknown", description="active | empty | unavailable"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_documents": 2143,
                "indexed_documents": 2143,
                "total_articles": 18742,
                "articles_with_embeddings": 18742,
                "by_language": {"fr": 2100, "en": 43},
                "by_category": {"Finances publiques": 512, "Sans catégorie": 88},
                "by_status": {"published": 2050, "refused": 61, "processing": 32},
                "index_health": "healthy",
                "cache_entries": 14,
                "cache_status": "active",
            }
        }
    }


class ReindexResponse(BaseModel):
    """Response for reindex operations."""

    status: str = Field(..., description="Reindex status (success/failed)")
    total_laws: int = Field(..., description="Total laws processed")
    indexed_count: int = Field(..., description="Number of laws successfully indexed")
    failed_count: int = Field(0, description="Number of laws that failed to index")
    duration_seconds: int = Field(..., description="Time elapsed in seconds")
