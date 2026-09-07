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

    # 64 et non 20 : la colonne a ete elargie (migration c8d9e0f1a2b3) pour
    # accueillir les ordinaux ecrits en toutes lettres, du type
    # « QUATRE-VINGT-DIX-SEPTIEME ». Le schema etait reste en arriere, et aurait
    # rejete a la validation ce que la base accepte.
    number: str = Field(
        ..., min_length=1, max_length=64, description="Article number (e.g., 'Art. 1', '42')"
    )
    title: Optional[str] = Field(None, max_length=200, description="Article title (optional)")
    content: str = Field(..., min_length=1, description="Article text content")
    order: int = Field(..., ge=1, description="Display order within the law")


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


class ArticleSummary(BaseModel):
    """
    Un article tel qu'il apparait dans le sommaire d'un document.

    Existe parce que la page de lecture reconstruisait le sommaire par une
    EXPRESSION REGULIERE sur `law.content`, cote navigateur. Deux consequences :
    elle trouvait 193 articles la ou la base en compte 230 pour le meme
    document, et elle n'avait aucun moyen de connaitre la page du PDF — donc un
    lien `?article=35` deplacait le sommaire mais laissait le PDF a la page 1.
    """

    id: int
    number: str
    title: Optional[str] = None
    section: Optional[str] = Field(None, description="En-tete de section (TITRE/CHAPITRE)")
    page_number: Optional[int] = Field(None, description="Page du PDF, 1-indexee")
    kind: Optional[str] = Field(None, description="article | legal_basis | preamble | ...")

    class Config:
        from_attributes = True


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


# Un premier LawDetailResponse vivait ici, exposant `List[ArticleResponse]`.
# Il n'a jamais ete branche sur aucune route, et il etait doublement
# inutilisable : ArticleResponse porte le CONTENU integral de chaque article —
# soit, pour le Code Minier, 166 Ko servis deux fois, dans `law.content` puis
# article par article — et son `number` etait plafonne a 20 caracteres alors
# que la colonne en fait 64 depuis qu'elle accueille les ordinaux ecrits en
# toutes lettres. Le schema reellement servi est plus bas, et ne porte que le
# sommaire.


# ============================================================================
# Filtering & Pagination Schemas
# ============================================================================


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


class LawDetailResponse(LawResponse):
    """
    Detail d'un document, sommaire compris.

    Volontairement SEPAREE de LawResponse : la route de LISTE charge deja les
    articles par `selectinload`, et les exposer la ferait transiter tout le
    corpus a chaque affichage de liste. Seul `GET /laws/{id}` rend ce schema.
    """

    articles: List[ArticleSummary] = Field(
        default_factory=list,
        description="Articles du document, dans l'ordre, avec leur page",
    )


# ============================================================================
# Explication d'article par le modele
# ============================================================================


class ArticleExplanationRequest(BaseModel):
    """
    Demande d'explication d'un article, envoyee par la page de lecture.

    Le numero voyage dans le CORPS et non dans le chemin : `articles.number`
    est un String(64) qui contient « 1er », « L 94 bis » ou « PREAMBULE ». En
    segment d'URL il faudrait l'encoder, et un numero portant « / » ou « . »
    produirait des surprises de routage. Ici il gagne en plus une borne.

    `persona` n'est PAS un champ : le ton est fixe cote serveur. L'exposer
    laisserait n'importe quel client changer la voix du produit depuis son
    navigateur.
    """

    number: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Numero de l'article, tel qu'affiche (« 35 », « 1er »)",
    )
    language: str = Field("fr", description="Langue de la reponse : fr ou en")
    excerpt: Optional[str] = Field(
        None,
        max_length=12_000,
        description=(
            "Texte de l'article tel qu'affiche. Sert de repli quand la base n'a "
            "pas de ligne pour ce numero ; sa provenance est verifiee."
        ),
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Seules les deux langues du corpus sont acceptees."""
        if v.lower() not in {"fr", "en"}:
            raise ValueError(f"Language must be 'fr' or 'en'. Got: {v}")
        return v.lower()

    @field_validator("number")
    @classmethod
    def validate_number(cls, v: str) -> str:
        """Un numero fait d'espaces n'en est pas un."""
        if not v.strip():
            raise ValueError("Article number must not be blank")
        return v.strip()


class ArticleExplanationResponse(BaseModel):
    """
    Explication d'un article, rendue sur la page du document.

    `resolved_from` dit par quelle voie l'article a ete retrouve : « database »
    quand la ligne existe (cas nominal, autoritaire), « excerpt » quand seule
    la page l'avait — un test peut ainsi prouver quel chemin s'est execute sans
    lire les journaux.
    """

    law_id: int
    article_id: Optional[int] = Field(None, description="None si resolu depuis l'extrait")
    number: str = Field(..., description="Numero normalise reellement explique")
    explanation: str
    language: str
    persona: str = Field(..., description="Toujours « citoyen »")
    resolved_from: str = Field(..., description="database | excerpt")
    generation_time_ms: int
