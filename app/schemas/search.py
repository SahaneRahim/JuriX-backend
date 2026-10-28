"""
Pydantic schemas for Search operations.

Provides request/response models for hybrid search (text + semantic + RRF):
- SearchFilters: Filter parameters for search queries
- SearchRequest: Main search request with query, mode, filters
- ArticleMatch: Matched article with snippet and relevance
- ChunkResult: Un article (chunk) tel qu'il sort de la recherche — l'unite du RAG
- SearchResult: Single search result with highlights
- SearchResponse: Complete search response with metadata
- SearchStats: Admin statistics for search operations

Author: JuriX Development Team
Date: 2026-01-10
"""

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.reperes import ReperesDuTexte

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


class SearchRefine(BaseModel):
    """
    Filtres appliques a l'ENSEMBLE CANDIDAT d'une requete, apres le classement
    et avant la pagination (voir app/services/recherche_affinee.py).

    A ne pas confondre avec SearchFilters, dont les filtres entrent dans le SQL
    de chaque branche et delimitent l'ensemble lui-meme. Ceux-ci le
    restreignent sans le recalculer : leurs resultats sont exactement ceux que
    comptent les effectifs (`facets`) de la meme requete.
    """

    types: Optional[List[str]] = Field(
        None,
        description=(
            "Natures retenues, valeurs de `type` (« décret », « loi », « arrêté », "
            "« acte uniforme »…). Accents, casse, « _ » et « - » ignores : "
            "« decret » et « acte_uniforme » conviennent."
        ),
    )
    category_ids: Optional[List[int]] = Field(None, description="Domaines retenus (ids)")
    year_from: Optional[int] = Field(None, ge=1900, le=2100, description="Annee minimale du texte")
    year_to: Optional[int] = Field(None, ge=1900, le=2100, description="Annee maximale du texte")
    date_from: Optional[date] = Field(None, description="Date minimale du texte (incluse)")
    date_to: Optional[date] = Field(None, description="Date maximale du texte (incluse)")

    @model_validator(mode="after")
    def _bornes_ordonnees(self) -> "SearchRefine":
        if self.year_from is not None and self.year_to is not None and self.year_to < self.year_from:
            raise ValueError(f"year_to ({self.year_to}) must be >= year_from ({self.year_from})")
        if self.date_from is not None and self.date_to is not None and self.date_to < self.date_from:
            raise ValueError(f"date_to ({self.date_to}) must be >= date_from ({self.date_from})")
        return self


# ============================================================================
# Request Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """
    Request schema for search endpoint.

    Supports three search modes:
    - text: Full-text search only (la recherche plein texte)
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
    # Le chat cherche en plein texte sur les MOTS-CLES de la question, mais un
    # modele d'embeddings comprend mieux la question entiere : il la recoit
    # ici. Absent, le volet semantique vectorise `query`.
    semantic_query: Optional[str] = Field(
        None,
        min_length=1,
        max_length=1000,
        description="Texte vectorise par le volet semantique, s'il differe de query"
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
    # Les trois champs suivants sont ADDITIFS et facultatifs. Sans aucun d'eux,
    # la recherche suit le parcours classique, a l'identique : le front deja
    # deploye, le chat et la comparaison n'en envoient jamais. Avec l'un d'eux,
    # elle passe par la recherche affinee (ensemble candidat fixe, filtres puis
    # tri puis page, effectifs) : voir app/services/recherche_affinee.py.
    sort: Optional[Literal["pertinence", "date_desc", "date_asc"]] = Field(
        None,
        description=(
            "Tri de l'ensemble filtre : pertinence (defaut), date_desc ou date_asc. "
            "Les textes sans date passent en dernier."
        ),
    )
    refine: Optional[SearchRefine] = Field(
        None,
        description="Filtres sur l'ensemble candidat, avant la pagination, coherents avec `facets`",
    )
    facets: bool = Field(
        False,
        description="Rendre les effectifs par nature, domaine et annee (recherche affinee)",
    )

    def recherche_affinee(self) -> bool:
        """Vrai si l'appelant a envoye l'un des parametres de la recherche affinee."""
        return self.sort is not None or self.refine is not None or self.facets

    @field_validator("semantic_query", mode="before")
    @classmethod
    def _semantic_query_blanc_absent(cls, v):
        """Un texte blanc ne se vectorise pas : il vaut absence, pas erreur 500."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

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


class ChunkResult(BaseModel):
    """
    Un chunk — c'est-a-dire un article — tel qu'il sort de la recherche.

    C'est l'unite que consomme le RAG. La recherche travaillait au niveau loi :
    les articles etaient ecrases a un par loi (DISTINCT ON), et sur le chemin
    semantique l'article gagnant etait jete par un GROUP BY. Le modele ne
    recevait donc jamais qu'un prefixe de 400 caracteres de texte de loi.

    SearchResult reste l'unite document-niveau exposee au front ; elle est
    desormais DERIVEE de ces chunks (voir SearchService._chunks_to_search_results).

    Les metadonnees de la loi sont denormalisees ici a dessein : un chunk doit
    se suffire a lui-meme pour etre cite sans second aller-retour en base.
    """

    # Identite du chunk
    article_id: Optional[int] = Field(
        None,
        description="ID de l'article. None pour une ligne de repli niveau-loi."
    )
    law_id: int = Field(..., description="ID de la loi")
    number: Optional[str] = Field(None, description="Numero d'article (ex: '1er', '42')")
    article_title: Optional[str] = Field(None, description="Titre de l'article")
    section: Optional[str] = Field(None, description="En-tete de section (TITRE/CHAPITRE)")
    page_number: Optional[int] = Field(None, description="Page du PDF (1-indexee)")
    content: str = Field("", description="Contenu integral de l'article")
    excerpt: str = Field("", description="Extrait mis en evidence (balises <mark>)")

    # Metadonnees de la loi, denormalisees
    reference: str = Field(..., description="Reference de la loi")
    law_title: str = Field(..., description="Titre de la loi")
    type: str = Field("loi", description="Type de texte")
    language: Optional[str] = Field(None, description="Code langue (fr/en)")
    status: str = Field("published", description="Statut de la loi")
    category_id: Optional[int] = Field(None, description="ID de categorie")
    category_name: Optional[str] = Field(None, description="Nom de categorie")
    publication_date: Optional[date] = Field(None, description="Date de publication")

    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Score de pertinence")
    source: str = Field(
        "fts",
        description=(
            "Origine: fts | trigram | title_fts | title_trigram | semantic | "
            "law_fts | priority | fallback"
        )
    )
    # Par quoi la correspondance passe. Calcule en SQL sur le POIDS des lexemes
    # (poids A = titre), pas devine apres coup : une correspondance de titre et
    # une correspondance de corps ne doivent pas se melanger dans la liste de
    # resultats. Defaut "body" — un chunk dont on ne sait rien n'est pas un
    # titre.
    match_scope: Literal["title", "body"] = Field(
        "body", description="Origine de la correspondance : title | body"
    )

    # Score du re-ranking, s'il a eu lieu. Champ SEPARE et non ecrasement de
    # relevance_score : le score de recuperation reste inspectable, ce qui
    # permet d'attribuer un ecart au re-ranking plutot qu'a la recherche.
    rerank_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Score apres re-ranking, si applique"
    )

    @property
    def fusion_key(self) -> tuple:
        """
        Cle de fusion RRF.

        Prefixee par le type : une ligne de repli niveau-loi porte l'id de la
        loi, qui peut coincider avec l'id d'un article sans aucun rapport. La
        fusion les confondait.
        """
        if self.article_id is not None:
            return ("a", self.article_id)
        return ("l", self.law_id)

    class Config:
        from_attributes = True


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
    # Par quoi la correspondance passe. Calcule en SQL sur le POIDS des lexemes
    # (poids A = titre), pas devine apres coup : une correspondance de titre et
    # une correspondance de corps ne doivent pas se melanger dans la liste de
    # resultats. Defaut "body" — un chunk dont on ne sait rien n'est pas un
    # titre.
    match_scope: Literal["title", "body"] = Field(
        "body", description="Origine de la correspondance : title | body"
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


# ============================================================================
# Effectifs de la recherche affinee
# ============================================================================


class FacetteValeur(BaseModel):
    """Une nature et son effectif : « décret », 77."""

    value: str = Field(..., description="Valeur de `type`, telle qu'en base")
    count: int = Field(..., ge=1, description="Resultats de cette nature")


class FacetteCategorie(BaseModel):
    """Un domaine et son effectif."""

    id: Optional[int] = Field(None, description="Id du domaine ; null : sans domaine")
    name: Optional[str] = Field(None, description="Nom du domaine, tel qu'en base")
    count: int = Field(..., ge=1, description="Resultats de ce domaine")


class FacetteAnnee(BaseModel):
    """Une annee et son effectif."""

    value: Optional[int] = Field(None, description="Annee du texte ; null : sans date")
    count: int = Field(..., ge=1, description="Resultats de cette annee")


class SearchFacets(BaseModel):
    """
    Effectifs sur l'ensemble candidat de la requete. Chaque facette compte les
    resultats qui passent tous les filtres de `refine` SAUF le sien (facettes
    disjonctives) ; une valeur absente a un effectif nul.
    """

    types: List[FacetteValeur] = Field(default_factory=list)
    categories: List[FacetteCategorie] = Field(default_factory=list)
    years: List[FacetteAnnee] = Field(default_factory=list)


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
    target_law_id: Optional[int] = Field(
        None,
        description=(
            "Loi a ouvrir quand direct_navigation est vrai. Champ explicite "
            "plutot que results[0] : les resultats sont partitionnes titre/corps, "
            "et lier la navigation a cet ordre la casserait au premier changement."
        ),
    )
    target_article: Optional[str] = Field(
        None,
        description="Article number/reference to scroll to (e.g., '5', 'PREMIER')"
    )
    direct_navigation: bool = Field(
        False,
        description="If true, client should redirect directly to the document"
    )
    # Champ ADDITIF, en dernier et avec un defaut : les reponses deja en cache
    # sous l'ancienne forme se deserialisent encore, et le front qui ne le
    # connait pas l'ignore. C'est ce que consomme le RAG.
    chunks: List[ChunkResult] = Field(
        default_factory=list,
        description="Resultats au niveau chunk (article), ordonnes par pertinence"
    )
    # Champs de la recherche affinee, ADDITIFS et en dernier comme `chunks` :
    # null sur le parcours classique. `total` y compte alors les resultats
    # filtres, ceux que la pagination parcourt.
    sort: Optional[str] = Field(
        None, description="Tri applique (recherche affinee) ; null sur le parcours classique"
    )
    candidates: Optional[int] = Field(
        None,
        ge=0,
        description="Taille de l'ensemble candidat avant les filtres de `refine` (recherche affinee)",
    )
    facets: Optional[SearchFacets] = Field(
        None, description="Effectifs par nature, domaine et annee (recherche affinee)"
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
                        "category_id": 1010,
                        "category_name": "Droit des Affaires, Banque et OHADA",
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
# Reponse HTTP publique
# ============================================================================

# Longueur de `results[].content` dans la reponse de POST /api/v1/search. Le
# front en lit au plus 200 caracteres (SearchResultCard.svelte, quand
# highlights.content manque) et ajoute lui-meme « ... » : 300 lui laissent de
# quoi savoir que le texte continue.
APERCU_CONTENU_CARACTERES = 300


class SearchResultPublic(SearchResult, ReperesDuTexte):
    """
    Un resultat tel que POST /api/v1/search le renvoie : `content` reduit a un
    apercu, tout le reste tel quel, et les reperes d'une ligne de liste dense
    (objet, numero, date_signature, annee, statut : voir app/schemas/reperes.py),
    sur les deux parcours, classique et affine. Ce sont des champs de plus, que
    le front deja deploye ignore.

    `content` portait le texte INTEGRAL de l'article, ou de la loi entiere pour
    une ligne niveau loi : mesure, 7,13 M de caracteres sur une reponse de
    8,56 Mo, envoyes a un telephone qui n'en affichait que 200. Les extraits
    (`highlights`, `matched_articles`) sont deja courts et restent entiers.
    """

    @field_validator("content")
    @classmethod
    def _apercu(cls, v: Optional[str]) -> Optional[str]:
        return v[:APERCU_CONTENU_CARACTERES] if v else v


class SearchResponsePublic(BaseModel):
    """
    Reponse de POST /api/v1/search : celle de SearchService, SANS `chunks` et
    avec des `content` reduits a un apercu.

    Seule la route HTTP passe par ce modele. Le chat et la comparaison
    appellent SearchService dans le processus et gardent la reponse complete :
    ce sont eux, et eux seuls, qui lisent `chunks`. Le front ne les lisait
    jamais, et les chunks niveau loi portaient le texte de la loi entiere.
    """

    query: str = Field(..., description="Original search query")
    mode: str = Field(..., description="Search mode used")
    results: List[SearchResultPublic] = Field(
        default_factory=list,
        description="List of search results (content reduit a un apercu)"
    )
    total: int = Field(..., ge=0, description="Total number of results found")
    search_time_ms: int = Field(
        ..., ge=0, description="Search execution time in milliseconds"
    )
    filters_applied: Optional[Dict] = Field(
        None, description="Filters that were applied to this search"
    )
    target_law_id: Optional[int] = Field(
        None, description="Loi a ouvrir quand direct_navigation est vrai"
    )
    target_article: Optional[str] = Field(
        None, description="Article number/reference to scroll to (e.g., '5', 'PREMIER')"
    )
    direct_navigation: bool = Field(
        False, description="If true, client should redirect directly to the document"
    )
    sort: Optional[str] = Field(
        None, description="Tri applique (recherche affinee) ; null sur le parcours classique"
    )
    candidates: Optional[int] = Field(
        None,
        ge=0,
        description="Taille de l'ensemble candidat avant les filtres de `refine` (recherche affinee)",
    )
    facets: Optional[SearchFacets] = Field(
        None, description="Effectifs par nature, domaine et annee (recherche affinee)"
    )

    @classmethod
    def depuis(cls, reponse: SearchResponse) -> "SearchResponsePublic":
        """
        La version publique d'une reponse du service.

        Lue par attributs : les chunks ne sont ni copies ni serialises, et
        chaque resultat passe par SearchResultPublic, qui coupe `content`.
        """
        return cls.model_validate(reponse, from_attributes=True)


# ============================================================================
# Admin Statistics Schemas
# ============================================================================

class SearchStats(BaseModel):
    """
    Statistiques de l'index de recherche (reserve aux administrateurs).

    Reecrit pour PostgreSQL : les champs decrivaient l'ancienne pile de recherche,
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
