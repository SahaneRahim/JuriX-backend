"""
Pydantic schemas for Law CRUD operations.

Provides request/response models with validation for:
- Law creation, update, and response
- Category operations
- Article management
- Filtering and pagination
- v2.1 features: language filtering, category suggestions, confidence scores

Author: JuriX Development Team
Date: 2026-01-10
"""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Category Schemas
# ============================================================================


class CategoryBase(BaseModel):
    """Base schema for Category with shared fields."""

    name: str = Field(..., min_length=1, max_length=100, description="Category name")
    description: Optional[str] = Field(None, description="Category description")
    icon: Optional[str] = Field(None, max_length=10, description="Emoji icon for category")


class CategoryCreate(CategoryBase):
    """Schema for creating a new category."""

    pass


class CategoryUpdate(BaseModel):
    """Schema for updating an existing category."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None


class CategoryResponse(CategoryBase):
    """Schema for category responses."""

    id: int
    created_at: datetime
    law_count: int = Field(0, description="Number of laws in this category")

    class Config:
        from_attributes = True  # SQLAlchemy 2.0 (orm_mode replacement)


# ============================================================================
# Article Schemas
# ============================================================================


class ArticleBase(BaseModel):
    """Base schema for Article with shared fields."""

    number: str = Field(
        ..., min_length=1, max_length=20, description="Article number (e.g., 'Art. 1', '42')"
    )
    title: Optional[str] = Field(None, max_length=200, description="Article title (optional)")
    content: str = Field(..., min_length=1, description="Article text content")
    order: int = Field(..., ge=1, description="Display order within the law")


class ArticleCreate(ArticleBase):
    """Schema for creating a new article."""

    pass


class ArticleUpdate(BaseModel):
    """Schema for updating an existing article."""

    number: Optional[str] = Field(None, min_length=1, max_length=20)
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    order: Optional[int] = Field(None, ge=1)


class ArticleResponse(ArticleBase):
    """Schema for article responses."""

    id: int
    law_id: int
    created_at: datetime
    has_embedding: bool = Field(False, description="Whether article has semantic embedding")

    class Config:
        from_attributes = True


# ============================================================================
# Law Schemas
# ============================================================================


class LawBase(BaseModel):
    """Base schema for Law with shared fields."""

    reference: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Unique law reference (e.g., 'LOI-2024-001', 'DECRET-2023-045')",
    )
    title: str = Field(..., min_length=1, max_length=500, description="Law title")
    type: str = Field(..., description="Law type (loi, décret, ordonnance, arrêté, etc.)")
    content: str = Field(..., min_length=10, description="Full text content of the law")
    language: Optional[str] = Field(
        None, description="Language code (fr or en). Auto-detected if not provided."
    )
    category_id: Optional[int] = Field(
        None, ge=1, description="Category ID. Auto-suggested if not provided."
    )
    status: Optional[str] = Field(
        "draft", description="Publication status (draft, published, archived)"
    )
    publication_date: Optional[date] = Field(None, description="Official publication date")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate law type against allowed values."""
        allowed_types = {
            "loi",
            "décret",
            "ordonnance",
            "arrêté",
            "circulaire",
            "instruction",
            "décision",
            "autre",
            "acte uniforme",  # Added for OHADA acts
        }
        if v.lower() not in allowed_types:
            raise ValueError(f"Type must be one of: {', '.join(allowed_types)}. Got: {v}")
        return v.lower()

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
    def validate_status(cls, v: Optional[str]) -> str:
        """Validate status against allowed values."""
        if v is None:
            return "draft"
        # Statuts du cycle de vie d'ingestion inclus : un document passe par
        # pending -> processing -> published | refused. Sans eux, LawResponse
        # rejetait sa propre reponse (erreur 500) des qu'un document etait en
        # cours de traitement ou en echec — donc invisible dans l'admin, qui
        # est justement l'endroit ou il faut le suivre.
        allowed_statuses = {
            "draft", "published", "archived",
            "pending", "processing", "refused",
        }
        if v.lower() not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}. Got: {v}")
        return v.lower()


class LawCreate(LawBase):
    """Schema for creating a new law."""

    pass


class LawUpdate(BaseModel):
    """
    Schema for updating an existing law.

    All fields are optional to support partial updates.
    If content is updated, language and categories will be re-detected.
    """

    reference: Optional[str] = Field(None, min_length=1, max_length=500)
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    type: Optional[str] = None
    content: Optional[str] = Field(None, min_length=10)
    language: Optional[str] = None
    category_id: Optional[int] = Field(None, ge=1)
    status: Optional[str] = None
    publication_date: Optional[date] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate law type if provided."""
        if v is None:
            return v
        allowed_types = {
            "loi",
            "décret",
            "ordonnance",
            "arrêté",
            "circulaire",
            "instruction",
            "décision",
            "autre",
        }
        if v.lower() not in allowed_types:
            raise ValueError(f"Type must be one of: {', '.join(allowed_types)}. Got: {v}")
        return v.lower()

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language code if provided."""
        if v is None:
            return v
        if v.lower() not in {"fr", "en"}:
            raise ValueError(f"Language must be 'fr' or 'en'. Got: {v}")
        return v.lower()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate status if provided."""
        if v is None:
            return v
        # Statuts du cycle de vie d'ingestion inclus : un document passe par
        # pending -> processing -> published | refused. Sans eux, LawResponse
        # rejetait sa propre reponse (erreur 500) des qu'un document etait en
        # cours de traitement ou en echec — donc invisible dans l'admin, qui
        # est justement l'endroit ou il faut le suivre.
        allowed_statuses = {
            "draft", "published", "archived",
            "pending", "processing", "refused",
        }
        if v.lower() not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}. Got: {v}")
        return v.lower()


class LawResponse(LawBase):
    """
    Schema for law responses with v2.1 auto-detection fields.

    Includes:
    - Basic law metadata
    - v2.1 language detection (detected_language, language_confidence)
    - v2.1 category suggestions (suggested_categories, category_confidence)
    - Article count
    - Timestamps
    """

    id: int

    # v2.1 Auto-detection fields
    detected_language: Optional[str] = Field(None, description="Auto-detected language (fr or en)")
    language_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for language detection (0.0-1.0)"
    )
    suggested_categories: Optional[List[int]] = Field(
        None, description="Top 3 suggested category IDs from DocumentClassifier"
    )
    category_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for top category suggestion (0.0-1.0)"
    )

    # File tracking
    file_id: Optional[str] = Field(None, description="ID of the uploaded file")
    original_filename: Optional[str] = Field(None, description="Original filename of the uploaded file")

    # Metadata
    article_count: int = Field(0, description="Number of articles in this law")
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Optional nested category
    category: Optional[CategoryResponse] = None

    class Config:
        from_attributes = True


class LawDetailResponse(LawResponse):
    """
    Detailed law response including articles.

    Used for GET /laws/{id}?include_articles=true
    """

    articles: List[ArticleResponse] = Field(default_factory=list)


# ============================================================================
# Filtering & Pagination Schemas
# ============================================================================


class LawFilters(BaseModel):
    """
    Query parameters for filtering and paginating law lists.

    Supports:
    - Language filtering (v2.1 feature)
    - Category filtering
    - Status filtering
    - Type filtering
    - Date range filtering
    - Pagination
    """

    # v2.1 Language filtering
    language: Optional[str] = Field(None, description="Filter by language (fr or en)")

    # Category filtering
    category_id: Optional[int] = Field(None, ge=1, description="Filter by category ID")

    # Status filtering
    status: Optional[str] = Field(None, description="Filter by status (draft, published, archived)")

    # Type filtering
    type: Optional[str] = Field(None, description="Filter by law type")

    # Date range filtering
    year_from: Optional[int] = Field(
        None, ge=1900, le=2100, description="Filter laws published from this year"
    )
    year_to: Optional[int] = Field(
        None, ge=1900, le=2100, description="Filter laws published up to this year"
    )

    # Pagination
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    per_page: int = Field(20, ge=1, le=100, description="Items per page (max 100)")

    # Sorting
    sort_by: Optional[str] = Field(
        "created_at", description="Sort field (created_at, publication_date, title, reference)"
    )
    sort_order: Optional[str] = Field("desc", description="Sort order (asc or desc)")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language filter."""
        if v is None:
            return v
        if v.lower() not in {"fr", "en"}:
            raise ValueError(f"Language must be 'fr' or 'en'. Got: {v}")
        return v.lower()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate status filter."""
        if v is None:
            return v
        # Statuts du cycle de vie d'ingestion inclus : un document passe par
        # pending -> processing -> published | refused. Sans eux, LawResponse
        # rejetait sa propre reponse (erreur 500) des qu'un document etait en
        # cours de traitement ou en echec — donc invisible dans l'admin, qui
        # est justement l'endroit ou il faut le suivre.
        allowed_statuses = {
            "draft", "published", "archived",
            "pending", "processing", "refused",
        }
        if v.lower() not in allowed_statuses:
            raise ValueError(f"Status must be one of: {', '.join(allowed_statuses)}. Got: {v}")
        return v.lower()

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, v: Optional[str]) -> str:
        """Validate sort field."""
        if v is None:
            return "created_at"
        allowed_fields = {"created_at", "publication_date", "title", "reference", "updated_at"}
        if v.lower() not in allowed_fields:
            raise ValueError(f"sort_by must be one of: {', '.join(allowed_fields)}. Got: {v}")
        return v.lower()

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, v: Optional[str]) -> str:
        """Validate sort order."""
        if v is None:
            return "desc"
        if v.lower() not in {"asc", "desc"}:
            raise ValueError(f"sort_order must be 'asc' or 'desc'. Got: {v}")
        return v.lower()

    @field_validator("year_to")
    @classmethod
    def validate_year_range(cls, v: Optional[int], info) -> Optional[int]:
        """Ensure year_to >= year_from if both provided."""
        if v is None:
            return v
        year_from = info.data.get("year_from")
        if year_from is not None and v < year_from:
            raise ValueError(f"year_to ({v}) must be >= year_from ({year_from})")
        return v


class LawListResponse(BaseModel):
    """
    Paginated list of laws with metadata.

    Used for GET /laws with filtering and pagination.
    """

    items: List[LawResponse] = Field(default_factory=list)
    total: int = Field(0, description="Total number of laws matching filters")
    page: int = Field(1, description="Current page number")
    per_page: int = Field(20, description="Items per page")
    pages: int = Field(0, description="Total number of pages")

    # Optional filter summary
    filters_applied: Optional[dict] = Field(None, description="Summary of applied filters")


# Les schemas de recherche vivaient ici en double de app/schemas/search.py, avec
# les MEMES noms (SearchRequest / SearchResult / SearchResponse). Ils ne
# servaient qu'a law_service.search_laws, supprime ; la recherche passe
# entierement par app/schemas/search.py.


# ============================================================================
# Statistics Schemas
# ============================================================================


class LanguageStats(BaseModel):
    """Statistics on law distribution by language (v2.1 feature)."""

    french: int = Field(0, description="Number of French laws")
    english: int = Field(0, description="Number of English laws")
    unknown: int = Field(0, description="Number of laws with unknown language")
    total: int = Field(0, description="Total number of laws")


class CategoryStats(BaseModel):
    """Statistics on law distribution by category."""

    category_id: int
    category_name: str
    law_count: int
    percentage: float = Field(0.0, ge=0.0, le=100.0)


class LawStats(BaseModel):
    """Overall statistics for the law database."""

    total_laws: int = Field(0)
    total_articles: int = Field(0)
    by_language: LanguageStats
    by_status: dict = Field(default_factory=dict)  # {"draft": 10, "published": 50, ...}
    by_type: dict = Field(default_factory=dict)  # {"loi": 30, "décret": 20, ...}
    top_categories: List[CategoryStats] = Field(default_factory=list)
    avg_articles_per_law: float = Field(0.0)
    latest_publication: Optional[date] = None
