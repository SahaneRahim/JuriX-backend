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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.search import (
    ReindexResponse,
    SearchRequest,
    SearchResponse,
    SearchStats,
)
from app.services.postgres_search_service import escape_like
from app.services.search_service import SearchService, SearchServiceError
from app.core.auth import get_current_admin_user
from app.models.user import User

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
    - **text**: Recherche plein texte PostgreSQL (tsvector + GIN, <50ms)
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
    # Les trois assertions posees ici ont ete retirees : SearchRequest declare
    # deja query avec min_length=1 et valide mode, donc FastAPI repond 422 pour
    # chacun de ces cas. Et un `assert` disparait sous python -O : la garde
    # etait au mieux redondante, au pire absente en production.

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
        # Erreurs service (FTS PostgreSQL, pgvector, etc.)
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
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Autocomplétion sur les titres et références de lois.

    Réécrit en PostgreSQL : la version précédente interrogeait
    `search_service.meilisearch_client`, attribut supprimé lors du passage à
    PostgreSQL natif. L'AttributeError était avalée par un `except`, si bien que
    l'endpoint renvoyait **200 avec une liste vide** — cassé en silence.

    Point de performance : l'opérateur `%` est accéléré par les index
    gin_trgm_ops (migration c2d3e4f5a6b7) ; `similarity(a,b) > seuil` ne l'est
    pas. Le filtre utilise donc `%` et `ILIKE 'q%'`, et `similarity()` ne sert
    qu'au tri. Les correspondances par préfixe passent devant les approximatives
    pour que « const » propose « Constitution… » avant une correspondance floue.

    Pas de mise en cache : une écriture dans query_cache à chaque frappe serait
    de l'amplification d'écriture pour une requête déjà sous les 20 ms.
    """
    if not isinstance(limit, int) or limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="limit doit être un entier positif",
        )

    q = (q or "").strip()
    if len(q) < 2:
        return {"suggestions": [], "query": q, "search_time_ms": 0}

    limit = min(limit, 10)
    start_time = time.time()

    sql = text("""
        SELECT id, title, reference,
               GREATEST(similarity(title, :q), similarity(reference, :q)) AS score,
               (title ILIKE :prefix OR reference ILIKE :prefix) AS is_prefix
        FROM laws
        WHERE status = 'published'
          AND (title ILIKE :prefix OR reference ILIKE :prefix
               OR title % :q OR reference % :q)
        ORDER BY is_prefix DESC, score DESC, title ASC
        LIMIT :limit
    """)

    try:
        result = await db.execute(
            sql, {"q": q, "prefix": f"{q}%", "limit": limit}
        )
        suggestions = [
            {"id": row.id, "title": row.title, "reference": row.reference}
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.error(f"❌ Autocomplétion en échec pour '{q[:40]}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'autocomplétion",
        )

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"📋 Suggest : {len(suggestions)} résultats pour '{q}' en {elapsed_ms}ms")
    return {"suggestions": suggestions, "query": q, "search_time_ms": elapsed_ms}


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
            # Metacaracteres echappes : un % ou un _ dans la saisie elargit
            # sinon le motif au lieu d'etre cherche litteralement.
            pattern = f"%{escape_like(doc_hint)}%"
            law_query = law_query.where(
                or_(
                    Law.title.ilike(pattern, escape="\\"),
                    Law.reference.ilike(pattern, escape="\\")
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
    background_tasks: BackgroundTasks, search_service: SearchService = Depends(get_search_service),
    current_admin: User = Depends(get_current_admin_user),
) -> ReindexResponse:
    """
    Reconstruit les tsvector de toutes les lois publiées (admin uniquement).

    **Opération longue** (~5s pour 100 lois):
    - Récupère toutes les lois publiées depuis DB
    - Recalcule laws.search_vector et articles.search_vector
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
    - Après un changement de configuration de la recherche plein texte
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
)
async def search_stats(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> SearchStats:
    """
    Statistiques d'indexation (administrateurs).

    Réécrit en PostgreSQL : la version précédente interrogeait
    `search_service.meilisearch_client` puis `redis_client`, deux attributs
    supprimés — l'endpoint renvoyait **systématiquement 500**. Les chiffres
    « par langue » et « par catégorie » étaient par ailleurs codés en dur à zéro.

    `indexed_documents` et `articles_with_embeddings` sont les deux nombres qui
    disent si la chaîne d'ingestion a réellement abouti.
    """
    try:
        totals = (await db.execute(text("""
            SELECT
                (SELECT count(*) FROM laws)                                  AS total_documents,
                (SELECT count(*) FROM laws WHERE search_vector IS NOT NULL)  AS indexed_documents,
                (SELECT count(*) FROM articles)                              AS total_articles,
                (SELECT count(*) FROM articles WHERE embedding IS NOT NULL)  AS articles_with_embeddings,
                (SELECT count(*) FROM query_cache WHERE expires_at > now())  AS cache_entries
        """))).one()

        by_language = {
            row.k: row.n
            for row in (await db.execute(text(
                "SELECT coalesce(language,'inconnue') AS k, count(*) AS n "
                "FROM laws GROUP BY 1 ORDER BY 2 DESC"
            ))).fetchall()
        }
        by_status = {
            row.k: row.n
            for row in (await db.execute(text(
                "SELECT coalesce(status,'inconnu') AS k, count(*) AS n "
                "FROM laws GROUP BY 1 ORDER BY 2 DESC"
            ))).fetchall()
        }
        by_category = {
            row.k: row.n
            for row in (await db.execute(text(
                "SELECT coalesce(c.name,'Sans catégorie') AS k, count(l.id) AS n "
                "FROM laws l LEFT JOIN categories c ON c.id = l.category_id "
                "GROUP BY 1 ORDER BY 2 DESC"
            ))).fetchall()
        }

        if totals.total_documents == 0:
            index_health = "empty"
        elif totals.indexed_documents == totals.total_documents:
            index_health = "healthy"
        else:
            index_health = "degraded"

        return SearchStats(
            total_documents=totals.total_documents,
            indexed_documents=totals.indexed_documents,
            total_articles=totals.total_articles,
            articles_with_embeddings=totals.articles_with_embeddings,
            by_language=by_language,
            by_category=by_category,
            by_status=by_status,
            index_health=index_health,
            cache_entries=totals.cache_entries,
            cache_status="active" if totals.cache_entries else "empty",
        )

    except Exception as e:
        logger.error(f"❌ Calcul des statistiques en échec : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des statistiques",
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def search_health(db: AsyncSession = Depends(get_db)) -> dict:
    """
    État de santé de la recherche.

    Réécrit : la version précédente testait `meilisearch_client` puis
    `redis_client`. La seconde vérification était **hors de tout `try`**, si bien
    que l'endpoint renvoyait toujours 500.

    Chaque sonde est isolée : une sonde en échec dégrade le rapport, elle ne le
    fait pas échouer.
    """
    report = {
        "status": "healthy",
        "database": "unknown",
        "fts_index": "unknown",
        "cache": "unknown",
        "embedding_service": "unknown",
    }

    try:
        await db.execute(text("SELECT 1"))
        report["database"] = "connected"
    except Exception as e:
        report["database"] = f"error: {type(e).__name__}"
        report["status"] = "degraded"

    try:
        present = (await db.execute(text(
            "SELECT to_regclass('idx_articles_search_vector') IS NOT NULL"
        ))).scalar()
        report["fts_index"] = "ready" if present else "missing"
        if not present:
            report["status"] = "degraded"
    except Exception as e:
        report["fts_index"] = f"error: {type(e).__name__}"
        report["status"] = "degraded"

    try:
        await db.execute(text("SELECT count(*) FROM query_cache"))
        report["cache"] = "active"
    except Exception:
        report["cache"] = "unavailable"

    try:
        from app.services.search_service import _embedding_service_instance

        report["embedding_service"] = (
            "ready" if _embedding_service_instance is not None else "unavailable"
        )
    except Exception:
        report["embedding_service"] = "unavailable"

    return report
