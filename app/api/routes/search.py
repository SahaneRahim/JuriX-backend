"""
Routes API pour la recherche hybride.

Endpoints:
- POST /api/v1/search - Recherche principale (text/semantic/hybrid)
- POST /api/v1/search/reindex - Réindexation complète (admin)
- GET /api/v1/search/stats - Statistiques recherche (admin)

Author: JuriX Team
Version: 1.0.0
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.search import (
    ReindexResponse,
    SearchRequest,
    SearchResponse,
    SearchStats,
)
from app.services.search_service import SearchService, SearchServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== DEPENDENCIES ====================


def get_search_service(db: AsyncSession = Depends(get_db)) -> SearchService:
    """
    Dependency injection pour SearchService.

    Args:
        db: Session database async

    Returns:
        Instance SearchService
    """
    return SearchService(db, use_cache=True)


# ==================== ENDPOINTS ====================


@router.post("/", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search(
    request: SearchRequest, search_service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    """
    Point d'entrée principal pour la recherche.

    Effectue une recherche selon le mode spécifié:
    - **text**: Recherche textuelle full-text (Meilisearch, <50ms)
    - **semantic**: Recherche sémantique par similarité (pgvector, <200ms)
    - **hybrid**: Combinaison optimale via RRF fusion (40/60, <200ms)

    **Filtres disponibles**:
    - language: Code langue (fr/en)
    - category_ids: Liste IDs catégories
    - types: Types de lois
    - status: Statut publication
    - year_from/year_to: Plage années

    **Pagination**:
    - limit: Nombre résultats (1-50, défaut: 15)
    - offset: Offset pagination (défaut: 0)

    **Performance**:
    - Text: <50ms
    - Semantic: <200ms
    - Hybrid: <200ms (spécification)

    Args:
        request: SearchRequest avec query, mode, filters, pagination
        search_service: Service injection

    Returns:
        SearchResponse avec résultats et métadonnées

    Raises:
        HTTPException 400: Si requête invalide
        HTTPException 500: Si recherche échoue

    Example:
        ```json
        POST /api/v1/search
        {
            "query": "responsabilité dirigeants société",
            "mode": "hybrid",
            "filters": {
                "language": "fr",
                "category_ids": [1, 3]
            },
            "limit": 15,
            "offset": 0
        }
        ```

        Response:
        ```json
        {
            "query": "responsabilité dirigeants société",
            "mode": "hybrid",
            "results": [
                {
                    "law_id": 156,
                    "reference": "LOI-2024-001",
                    "title": "Code des sociétés commerciales",
                    "relevance_score": 0.95,
                    "highlights": {
                        "title": "Code des <mark>sociétés</mark> commerciales",
                        "content": "...la <mark>responsabilité</mark> des <mark>dirigeants</mark>..."
                    }
                }
            ],
            "total": 42,
            "search_time_ms": 187,
            "filters_applied": {"language": "fr", "category_ids": [1, 3]}
        }
        ```
    """
    assert request is not None, "SearchRequest must not be None"
    assert isinstance(request.query, str) and len(request.query) > 0, "Query must be a non-empty string"
    assert request.mode in ("text", "semantic", "hybrid"), f"Invalid search mode: {request.mode}"

    try:
        logger.info(f'📥 Search request: mode={request.mode}, query="{request.query[:50]}"')

        # Exécution recherche
        response = await search_service.search(request)

        logger.info(f"📤 Search response: {response.total} results in {response.search_time_ms}ms")

        return response

    except ValueError as e:
        # Validation errors (query vide, mode invalide, etc.)
        logger.warning(f"⚠️ Invalid search request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Requête invalide: {str(e)}"
        ) from e

    except SearchServiceError as e:
        # Erreurs service (Meilisearch down, pgvector error, etc.)
        logger.error(f"❌ Search service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la recherche: {str(e)}",
        ) from e

    except Exception as e:
        # Erreurs inattendues
        logger.error(f"❌ Unexpected error during search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur interne du serveur"
        ) from e


@router.get("/suggest", status_code=status.HTTP_200_OK)
async def suggest(
    q: str,
    limit: int = 5,
    search_service: SearchService = Depends(get_search_service)
) -> dict:
    """
    Autocomplétion rapide pour suggestions de recherche.
    
    Utilise uniquement Meilisearch (mode text) pour des réponses ultra-rapides (<50ms).
    Retourne titres et références de lois correspondant à la requête.
    
    Args:
        q: Query de recherche (min 2 caractères)
        limit: Nombre de suggestions (défaut: 5, max: 10)
        
    Returns:
        Liste de suggestions avec titre, référence et ID
        
    Example:
        GET /api/v1/search/suggest?q=const&limit=5
        
        Response:
        {
            "suggestions": [
                {"id": 1, "title": "Constitution du Cameroun", "reference": "CONST-1996"},
                {"id": 42, "title": "Loi constitutionnelle", "reference": "LOI-2008-001"}
            ],
            "query": "const",
            "search_time_ms": 12
        }
    """
    assert isinstance(q, str), "Query must be a string"
    assert isinstance(limit, int) and limit > 0, "Limit must be a positive integer"

    if len(q) < 2:
        return {"suggestions": [], "query": q, "search_time_ms": 0}
    
    limit = min(limit, 10)  # Max 10 suggestions
    
    try:
        start_time = time.time()
        
        # Recherche Meilisearch directe (ultra-rapide)
        index = search_service.meilisearch_client.index(SearchService.MEILISEARCH_INDEX)
        
        results = index.search(
            q,
            {
                "limit": limit,
                "attributesToRetrieve": ["id", "title", "reference"],
                "attributesToHighlight": ["title"],
                "highlightPreTag": "<mark>",
                "highlightPostTag": "</mark>"
            }
        )
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        suggestions = [
            {
                "id": hit.get("id"),
                "title": hit.get("_formatted", {}).get("title", hit.get("title", "")),
                "reference": hit.get("reference", "")
            }
            for hit in results.get("hits", [])
        ]
        
        logger.debug(f"📋 Suggest: {len(suggestions)} results for '{q}' in {search_time_ms}ms")
        
        return {
            "suggestions": suggestions,
            "query": q,
            "search_time_ms": search_time_ms
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Suggest error: {e}")
        return {"suggestions": [], "query": q, "search_time_ms": 0, "error": str(e)}


def _parse_article_reference(query: str) -> tuple:
    """
    Parse article number and document hint from a query string.

    Supports French (article, art.) and English (section) patterns.
    Normalizes 'premier', '1er', 'first' etc. to '1'.

    Args:
        query: Lowercased query string

    Returns:
        (article_num: str | None, doc_hint: str | None)
    """
    import re

    patterns = [
        # "article 5 de la constitution"
        r"article\s+(premier|1er|1ère|\d+)\s+(?:de\s+)?(?:la\s+|le\s+|l[''']|du\s+|des\s+)?(.+)",
        # "art. 12 de la loi"
        r"art\.?\s*(\d+)\s+(?:de\s+)?(?:la\s+|le\s+|l[''']|du\s+)?(.+)",
        # English: "section 5"
        r"section\s+(\d+|one|first)\s+(?:of\s+)?(?:the\s+)?(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            article_num = match.group(1).strip()
            doc_hint = match.group(2).strip() if match.group(2) else ""

            # Normalize
            if article_num.lower() in ("premier", "1er", "1ère", "first", "one"):
                article_num = "1"
            return article_num, doc_hint

    return None, None


def _estimate_page_number(article) -> int:
    """
    Estimate the PDF page number where an article appears.

    Uses stored page_number if available; otherwise estimates from
    the article number (roughly 2 articles per page).

    Args:
        article: Article ORM object with .page_number and .number

    Returns:
        Estimated page number (>= 2)
    """
    if article.page_number is not None:
        return article.page_number

    try:
        art_num = int(article.number.replace("er", "").replace("ère", "").strip())
        return max(2, 1 + (art_num // 2))
    except ValueError:
        return 2  # Default to page 2


@router.get("/article", status_code=status.HTTP_200_OK)
async def find_article(
    q: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Recherche rapide d'article pour navigation directe (<2s).
    
    Détecte les références d'articles dans la query et retourne:
    - law_id: ID du document contenant l'article
    - article_num: Numéro de l'article trouvé
    - direct_url: URL pour navigation directe
    
    Performance cible: <500ms
    
    Args:
        q: Query contenant une référence d'article (ex: "article 5 de la constitution")
        
    Returns:
        Résultat avec law_id et article_num pour navigation directe
    """
    from sqlalchemy import select, or_
    from app.models.law import Law, Article
    
    assert isinstance(q, str) and len(q) > 0, "Query must be a non-empty string"

    start_time = time.time()
    
    # Parse article reference from query
    article_num, doc_hint = _parse_article_reference(q.lower().strip())
    
    if not article_num:
        return {
            "found": False,
            "error": "Aucune référence d'article détectée",
            "query": q,
            "search_time_ms": int((time.time() - start_time) * 1000)
        }
    
    try:
        # Search for the law by title hint
        law_query = select(Law).where(Law.status == "published")
        
        if doc_hint:
            law_query = law_query.where(
                or_(
                    Law.title.ilike(f"%{doc_hint}%"),
                    Law.reference.ilike(f"%{doc_hint}%")
                )
            )
        
        result = await db.execute(law_query.limit(5))
        laws = result.scalars().all()
        
        # Find article in these laws
        for law in laws:
            article_query = select(Article).where(
                Article.law_id == law.id,
                Article.number == article_num
            ).limit(1)
            
            article_result = await db.execute(article_query)
            article = article_result.scalar_one_or_none()
            
            if article:
                search_time_ms = int((time.time() - start_time) * 1000)
                logger.info(f"🎯 Found article {article.number} in law {law.id} in {search_time_ms}ms")
                
                page_number = _estimate_page_number(article)
                
                return {
                    "found": True,
                    "law_id": law.id,
                    "law_title": law.title,
                    "law_reference": law.reference,
                    "article_num": article.number,
                    "article_id": article.id,
                    "article_order": article.order,
                    "estimated_page": page_number,
                    "direct_url": f"/laws/{law.id}?article={article.number}&page={page_number}",
                    "query": q,
                    "search_time_ms": search_time_ms
                }
        
        # Article not found
        search_time_ms = int((time.time() - start_time) * 1000)
        return {
            "found": False,
            "error": f"Article {article_num} non trouvé dans '{doc_hint or 'tous les documents'}'",
            "query": q,
            "search_time_ms": search_time_ms
        }
        
    except Exception as e:
        logger.error(f"❌ Article search error: {e}")
        return {
            "found": False,
            "error": str(e),
            "query": q,
            "search_time_ms": int((time.time() - start_time) * 1000)
        }



@router.post(
    "/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_200_OK,
    # TODO: Add admin authentication dependency
    # dependencies=[Depends(require_admin)]
)
async def reindex_all(
    background_tasks: BackgroundTasks, search_service: SearchService = Depends(get_search_service)
) -> ReindexResponse:
    """
    Réindexe toutes les lois publiées dans Meilisearch (admin uniquement).

    **Opération longue** (~5s pour 100 lois):
    - Récupère toutes les lois publiées depuis DB
    - Supprime ancien index Meilisearch
    - Recrée index avec configuration optimale
    - Indexe documents en batch (1000 par batch)

    **Configuration index**:
    - Searchable: title, reference, content
    - Filterable: id, category_id, type, language, status, year
    - Sortable: publication_year, created_at
    - Typo tolerance: 1-2 typos selon longueur
    - Ranking: words > typo > proximity > attribute > sort > exactness

    **Quand utiliser**:
    - Après changements massifs de contenu
    - Après mise à jour configuration Meilisearch
    - Résolution corruption index
    - Tests/développement

    **⚠️ Attention**:
    - Recherche indisponible pendant réindexation
    - Opération CPU/IO intensive
    - Invalidation cache nécessaire

    Args:
        background_tasks: FastAPI background tasks (future async support)
        search_service: Service injection

    Returns:
        ReindexResponse avec statistiques (total, indexed, failed, duration)

    Raises:
        HTTPException 500: Si réindexation échoue

    Example:
        ```
        POST /api/v1/search/reindex
        Authorization: Bearer <admin_token>
        ```

        Response:
        ```json
        {
            "status": "success",
            "total_laws": 156,
            "indexed_count": 156,
            "failed_count": 0,
            "duration_seconds": 4
        }
        ```
    """
    try:
        logger.info("🔄 Starting full reindex (admin operation)...")

        # Exécution réindexation (blocking pour l'instant)
        # TODO: Implémenter via background_tasks pour async
        response = await search_service.reindex_all_laws()

        logger.info(
            f"✅ Reindex complete: {response.indexed_count}/{response.total_laws} "
            f"in {response.duration_seconds}s"
        )

        return response

    except SearchServiceError as e:
        logger.error(f"❌ Reindex failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec réindexation: {str(e)}",
        ) from e

    except Exception as e:
        logger.error(f"❌ Unexpected error during reindex: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur interne du serveur"
        ) from e


@router.get(
    "/stats",
    response_model=SearchStats,
    status_code=status.HTTP_200_OK,
    # TODO: Add admin authentication dependency
    # dependencies=[Depends(require_admin)]
)
async def get_search_stats(
    search_service: SearchService = Depends(get_search_service),
) -> SearchStats:
    """
    Récupère les statistiques de recherche (admin uniquement).

    **Métriques fournies**:
    - Nombre total de documents indexés
    - Répartition par langue (fr/en)
    - Répartition par catégorie
    - Répartition par statut
    - État de santé index Meilisearch
    - État cache Redis

    **Cas d'usage**:
    - Monitoring santé système recherche
    - Analytics utilisation
    - Debugging problèmes indexation
    - Validation après réindexation

    Args:
        search_service: Service injection

    Returns:
        SearchStats avec métriques système

    Raises:
        HTTPException 500: Si récupération stats échoue

    Example:
        ```
        GET /api/v1/search/stats
        Authorization: Bearer <admin_token>
        ```

        Response:
        ```json
        {
            "total_documents": 156,
            "by_language": {"fr": 142, "en": 14},
            "by_category": {"Droit Civil": 45, "Droit Pénal": 38, ...},
            "by_status": {"published": 156, "draft": 0},
            "index_health": "healthy",
            "cache_status": "connected"
        }
        ```
    """
    try:
        logger.info("📊 Fetching search stats (admin operation)...")

        # TODO: Implémenter récupération stats depuis Meilisearch + DB
        # Pour l'instant retourne mock data

        # Récupération stats Meilisearch
        index = search_service.meilisearch_client.index(SearchService.MEILISEARCH_INDEX)

        try:
            index_stats = index.get_stats()
            total_docs = index_stats.get("numberOfDocuments", 0)
        except Exception:
            total_docs = 0

        # Mock stats (TODO: Vraies stats depuis DB)
        stats = SearchStats(
            total_documents=total_docs,
            by_language={"fr": 0, "en": 0},  # TODO: Query DB
            by_category={},  # TODO: Query DB
            by_status={"published": total_docs},
            index_health="healthy" if total_docs > 0 else "empty",
            cache_status=("connected" if search_service.redis_client else "disabled"),
        )

        logger.info(f"✅ Stats retrieved: {total_docs} documents indexed")

        return stats

    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des statistiques",
        ) from e


# ==================== HEALTH CHECK ====================


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(search_service: SearchService = Depends(get_search_service)) -> dict:
    """
    Vérifie l'état de santé du service de recherche.

    Teste:
    - Connexion Meilisearch
    - Existence index
    - Connexion Redis (si cache activé)
    - Disponibilité EmbeddingService

    Returns:
        Statut de santé avec détails composants

    Example:
        ```
        GET /api/v1/search/health
        ```

        Response:
        ```json
        {
            "status": "healthy",
            "meilisearch": "connected",
            "redis": "connected",
            "embedding_service": "ready"
        }
        ```
    """
    health_status = {
        "status": "healthy",
        "meilisearch": "unknown",
        "redis": "unknown",
        "embedding_service": "unknown",
    }

    # Test Meilisearch
    try:
        search_service.meilisearch_client.health()
        health_status["meilisearch"] = "connected"
    except Exception as e:
        health_status["meilisearch"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Test Redis
    if search_service.redis_client:
        try:
            search_service.redis_client.ping()
            health_status["redis"] = "connected"
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["redis"] = "disabled"

    # Test EmbeddingService
    if search_service.embedding_service is not None:
        try:
            emb_health = search_service.embedding_service.health_check()
            health_status["embedding_service"] = emb_health["status"]
            if emb_health["status"] != "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["embedding_service"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["embedding_service"] = "unavailable (text search only)"
        health_status["status"] = "degraded"

    return health_status
