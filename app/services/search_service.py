"""
Service de recherche hybride (textuelle + sémantique) pour JuriX.

Ce service implémente 3 modes de recherche:
1. Textuelle (PostgreSQL FTS + pg_trgm)
2. Sémantique (pgvector) - Recherche contextuelle par similarité
3. Hybrid (RRF fusion 40/60) - Combinaison optimale

Architecture (PostgreSQL native):
- FTS: tsvector / websearch_to_tsquery + pg_trgm (trigrammes)
- pgvector: Vector similarity avec HNSW index
- RRF Fusion: Weighted Reciprocal Rank Fusion (40% text + 60% semantic)
- Cache: Table query_cache PostgreSQL avec TTL (cache en base)

Author: JuriX Team
Version: 3.0.0 (PostgreSQL natif)
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.models.law import Article, Category, Law
from app.schemas.search import (
    ArticleMatch,
    ReindexResponse,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchStats,
)
from app.services.embedding_service import EmbeddingService
from app.services.postgres_search_service import (
    get_from_pg_cache,
    store_in_pg_cache,
    search_articles_pg,
    search_laws_pg,
    update_law_search_vector,
    remove_law_search_index,
    cleanup_expired_cache,
    _make_cache_key,
)

logger = logging.getLogger(__name__)

# ==================== GLOBAL SINGLETONS ====================

_embedding_service_instance: Optional[EmbeddingService] = None
_singletons_initialized: bool = False


def _init_global_singletons() -> None:
    """
    Initialize global singleton instances for performance.
    Called once at first SearchService instantiation.
    """
    global _embedding_service_instance, _singletons_initialized

    if _singletons_initialized:
        return

    logger.info("🚀 Initializing global search singletons (one-time)...")

    try:
        _embedding_service_instance = EmbeddingService(use_cache=True)
        logger.info("✅ EmbeddingService initialized")
    except Exception as e:
        logger.error(f"❌ EmbeddingService init failed: {e}")
        _embedding_service_instance = None

    _singletons_initialized = True
    logger.info("✅ All global singletons ready")


# ==================== EXCEPTIONS ====================


class SearchServiceError(Exception):
    """Exception de base pour SearchService."""
    pass


class TextSearchError(SearchServiceError):
    """Erreur liée à la recherche textuelle PostgreSQL FTS."""
    pass


class VectorSearchError(SearchServiceError):
    """Erreur liée à la recherche vectorielle pgvector."""
    pass


class IndexingError(SearchServiceError):
    """Erreur lors des opérations d'indexation."""
    pass




# ==================== SEARCH SERVICE ====================


class SearchService:
    """
    Service de recherche hybride pour lois camerounaises.

    Fournit 3 modes de recherche avec fusion RRF et caching PostgreSQL:
    - Text: PostgreSQL FTS full-text (tsvector + pg_trgm)
    - Semantic: pgvector cosine similarity (contexte, synonymes)
    - Hybrid: RRF fusion 40% text + 60% semantic (optimal)

    Caractéristiques:
    - Performance: <200ms hybrid search
    - Filters: language, category, type, status, dates
    - Pagination: limit/offset
    - Caching: PostgreSQL query_cache table avec TTL 5min
    - Auto-indexing: mise à jour tsvector via SQL triggers

    Attributes:
        db: Session database async SQLAlchemy
        embedding_service: Service embeddings (singleton)
        use_cache: Flag activation cache
    """

    # Configuration
    CACHE_TTL_SECONDS = 300   # 5 minutes
    RRF_K = 60                # Constante RRF fusion
    TEXT_WEIGHT = 0.4         # Poids recherche textuelle
    SEMANTIC_WEIGHT = 0.6     # Poids recherche sémantique
    MAX_RESULTS_PER_MODE = 20 # Limite avant fusion

    def __init__(self, db: AsyncSession, use_cache: bool = True):
        """
        Initialise le service de recherche.

        Args:
            db: Session database async SQLAlchemy
            use_cache: Active le cache PostgreSQL (défaut: True)
        """
        _init_global_singletons()

        self.db = db
        self.use_cache = use_cache
        self.embedding_service = _embedding_service_instance

        logger.debug(f"SearchService ready (cache={use_cache}, pg_fts=True)")

    # ==================== ARTICLE REFERENCE PARSING ====================

    def _parse_article_reference(self, query: str) -> Optional[Dict[str, str]]:
        """
        Détecte et extrait les références d'articles dans la query.
        Ex: "article 5 de la constitution" → {'article_num': '5', 'doc_hint': 'constitution'}
        """
        import re

        query_lower = query.lower().strip()

        patterns = [
            r"article\s+(premier|1er|1ère|\d+)\s+(?:de\s+)?(?:la\s+|le\s+|l[''']|du\s+|des\s+)?(.+)",
            r"art\.?\s*(\d+)\s+(?:de\s+)?(?:la\s+|le\s+|l[''']|du\s+)?(.+)",
            r"section\s+(\d+|one|first)\s+(?:of\s+)?(?:the\s+)?(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                article_num = match.group(1).strip()
                doc_hint = match.group(2).strip() if match.group(2) else ""
                article_num_normalized = article_num.upper()
                if article_num_normalized in ("PREMIER", "1ER", "1ÈRE"):
                    article_num_normalized = "PREMIER"
                return {"article_num": article_num_normalized, "doc_hint": doc_hint}

        return None

    # ==================== PUBLIC API ====================

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Point d'entrée principal pour la recherche.

        Args:
            request: Requête de recherche (query, mode, filters, pagination)

        Returns:
            SearchResponse avec résultats et métadonnées
        """
        assert request and request.query, "SearchRequest with query is required"
        start_time = time.time()

        logger.info(
            f"🔍 Search request: mode={request.mode}, "
            f"query=\"{request.query[:50]}\", filters={request.filters}"
        )

        cache_key = ""
        # Check PostgreSQL cache
        if self.use_cache:
            cache_key = _make_cache_key(
                request.query, request.filters, request.limit, request.offset
            )
            # Le cache est une optimisation, pas une dependance : si query_cache
            # est absente ou injoignable, on doit degrader, pas renvoyer 500.
            # La lecture n'etait pas protegee alors que l'ecriture l'etait deja.
            try:
                cached_data = await get_from_pg_cache(self.db, cache_key)
            except Exception as e:
                logger.warning(f"⚠️ Lecture du cache impossible, ignoree: {e}")
                cached_data = None
            if cached_data:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(f"🎯 Cache HIT ({elapsed_ms}ms)")
                return SearchResponse(**cached_data)

        # Execute search with fallback handling
        try:
            results = await self._execute_search_by_mode(request)
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = self._build_search_response(request, results, elapsed_ms)

            # Store in PostgreSQL cache
            if self.use_cache:
                try:
                    await store_in_pg_cache(
                        self.db,
                        cache_key,
                        response.model_dump(),
                        ttl_seconds=self.CACHE_TTL_SECONDS,
                    )
                except Exception as cache_err:
                    logger.warning(f"⚠️ Cache write failed (non-fatal): {cache_err}")

            logger.info(
                f"✅ Search completed: {response.total} results in {elapsed_ms}ms "
                f"(mode={request.mode})"
            )
            return response

        except TextSearchError as e:
            return await self._handle_search_fallback(
                request, start_time, "semantic", e, "PostgreSQL FTS"
            )
        except VectorSearchError as e:
            return await self._handle_search_fallback(
                request, start_time, "text", e, "Vector search"
            )

    async def _execute_search_by_mode(self, request: SearchRequest):
        """Dispatch search to the correct mode handler."""
        if request.mode == "text":
            return await self.text_search(
                request.query, request.filters, request.limit, request.offset
            )
        elif request.mode == "semantic":
            return await self.semantic_search(
                request.query, request.filters, request.limit, request.offset
            )
        elif request.mode == "hybrid":
            return await self.hybrid_search(
                request.query, request.filters, request.limit, request.offset
            )
        else:
            raise ValueError(f"Mode invalide: {request.mode}")

    def _build_search_response(self, request, results, elapsed_ms):
        """Build SearchResponse with article reference detection."""
        total = len(results)
        article_ref = self._parse_article_reference(request.query)
        target_article = None
        direct_navigation = False

        if article_ref and total == 1:
            target_article = article_ref["article_num"]
            direct_navigation = True
        elif article_ref and total > 1:
            target_article = article_ref["article_num"]

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            results=results,
            total=total,
            search_time_ms=elapsed_ms,
            filters_applied=request.filters.model_dump() if request.filters else None,
            target_article=target_article,
            direct_navigation=direct_navigation,
        )

    async def _handle_search_fallback(self, request, start_time, fallback_mode, error, source_name):
        """Handle search error by falling back to alternative mode."""
        logger.warning(f"⚠️ {source_name} failed, fallback to {fallback_mode}: {error}")
        if request.mode != "hybrid":
            raise error

        if fallback_mode == "semantic":
            results = await self.semantic_search(
                request.query, request.filters, request.limit, request.offset
            )
        else:
            results = await self.text_search(
                request.query, request.filters, request.limit, request.offset
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        return SearchResponse(
            query=request.query,
            mode=fallback_mode,
            results=results,
            total=len(results),
            search_time_ms=elapsed_ms,
            filters_applied=request.filters.model_dump() if request.filters else None,
        )

    # ==================== SEARCH MODES ====================

    async def text_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[SearchResult]:
        """
        Recherche textuelle via PostgreSQL FTS (remplace la recherche plein texte).

        Cherche d'abord dans les articles, puis dans les lois en fallback.

        Args:
            query: Requête textuelle
            filters: Filtres optionnels
            limit: Nombre max résultats
            offset: Offset pagination

        Returns:
            Liste de SearchResult triée par relevance_score

        Raises:
            TextSearchError: Si recherche échoue
        """
        assert query and isinstance(query, str), "query must be a non-empty string"
        assert limit > 0, "limit must be positive"
        start_time = time.time()

        try:
            results = await search_articles_pg(self.db, query, limit, offset)
            if not results:
                results = await search_laws_pg(self.db, query, filters, limit, offset)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"📝 Text search total: {len(results)} results in {elapsed_ms}ms")
            return results

        except Exception as e:
            logger.error(f"❌ Text search failed: {e}")
            raise TextSearchError(f"Échec recherche textuelle: {e}") from e

    async def semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[SearchResult]:
        """
        Recherche sémantique via pgvector (inchangée).

        Args:
            query: Requête textuelle
            filters: Filtres optionnels
            limit: Nombre max résultats
            offset: Offset pagination

        Returns:
            Liste de SearchResult triée par relevance_score (similarity)

        Raises:
            VectorSearchError: Si recherche échoue
        """
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(limit, int) and limit > 0
        start_time = time.time()

        try:
            if self.embedding_service is None:
                logger.warning("⚠️ Semantic search unavailable: EmbeddingService not loaded")
                return []

            # TASK_QUERY et non TASK_DOCUMENT : encoder une question comme un
            # document dégrade la pertinence en recherche asymétrique.
            query_embedding = self.embedding_service.generate_embedding(
                query, task_type=self.embedding_service.TASK_QUERY
            )
            query_embedding_str = f"[{','.join(map(str, query_embedding.tolist()))}]"

            from sqlalchemy import cast
            from pgvector.sqlalchemy import Vector

            stmt = (
                select(
                    Law.id,
                    Law.reference,
                    Law.title,
                    Law.type,
                    Law.language,
                    Law.status,
                    Law.category_id,
                    Category.name.label("category_name"),
                    func.max(
                        1 - func.cosine_distance(
                            Article.embedding,
                            # cast vers Vector, pas String : Postgres rejette
                            # cosine_distance(vector, text) et la recherche
                            # sémantique échouait systématiquement.
                            cast(query_embedding_str, Vector(3072))
                        )
                    ).label("similarity")
                )
                .join(Article, Law.id == Article.law_id)
                .outerjoin(Category, Law.category_id == Category.id)
                .where(Article.embedding.isnot(None))
            )

            stmt = self._apply_filters_pgvector(stmt, filters)

            stmt = (
                stmt
                .group_by(
                    Law.id, Law.reference, Law.title, Law.type,
                    Law.language, Law.status, Law.category_id, Category.name
                )
                .order_by(text("similarity DESC"))
                .limit(min(limit, self.MAX_RESULTS_PER_MODE))
                .offset(offset)
            )

            result = await self.db.execute(stmt)
            rows = result.all()

            results = []
            for row in rows:
                results.append(SearchResult(
                    law_id=row.id,
                    reference=row.reference,
                    title=row.title,
                    type=row.type,
                    language=row.language,
                    status=row.status,
                    category_id=row.category_id,
                    category_name=row.category_name,
                    publication_date=None,
                    relevance_score=float(row.similarity),
                    matched_articles=[],
                    highlights={},
                ))

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"🧠 Semantic search: {len(results)} results in {elapsed_ms}ms")
            return results

        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            raise VectorSearchError(f"Échec recherche sémantique: {e}") from e

    async def hybrid_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[SearchResult]:
        """
        Recherche hybride via RRF fusion (40% text + 60% semantic).

        Args:
            query: Requête textuelle
            filters: Filtres optionnels
            limit: Nombre max résultats
            offset: Offset pagination

        Returns:
            Liste de SearchResult triée par RRF score
        """
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(limit, int) and limit > 0
        start_time = time.time()

        try:
            text_results, semantic_results = await asyncio.gather(
                self.text_search(query, filters, limit=self.MAX_RESULTS_PER_MODE, offset=0),
                self.semantic_search(query, filters, limit=self.MAX_RESULTS_PER_MODE, offset=0),
                return_exceptions=True,
            )

            if not isinstance(text_results, list):
                logger.warning(f"⚠️ Text search failed in hybrid: {text_results}")
                text_res_list: List[SearchResult] = []
            else:
                text_res_list = text_results

            if not isinstance(semantic_results, list):
                logger.warning(f"⚠️ Semantic search failed in hybrid: {semantic_results}")
                sem_res_list: List[SearchResult] = []
            else:
                sem_res_list = semantic_results

            if not text_res_list and not sem_res_list:
                return []

            fused_results = self._rrf_fusion(
                text_res_list, sem_res_list,
                k=self.RRF_K,
                text_weight=self.TEXT_WEIGHT,
                semantic_weight=self.SEMANTIC_WEIGHT,
            )
            fused_results = self._normalize_scores(fused_results)
            paginated_results = fused_results[offset:offset + limit]

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"🔀 Hybrid search: {len(paginated_results)} results in {elapsed_ms}ms "
                f"(text={len(text_res_list)}, semantic={len(sem_res_list)})"
            )
            return paginated_results

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise SearchServiceError(f"Échec recherche hybride: {e}") from e

    # ==================== INDEXING (PostgreSQL native) ====================

    async def index_law(self, law: Law) -> None:
        """
        Indexe une loi via mise à jour du tsvector PostgreSQL.
        Met a jour les tsvector PostgreSQL.

        Args:
            law: Instance Law à indexer
        """
        try:
            from typing import cast
            await update_law_search_vector(self.db, cast(int, law.id))
            logger.info(f"✅ PG FTS indexed law {law.id} ({law.reference})")
        except Exception as e:
            logger.error(f"❌ Failed to index law {law.id}: {e}")
            raise IndexingError(f"Échec indexation law {law.id}: {e}") from e

    async def update_law_index(self, law_id: int, law: Law) -> None:
        """Met à jour l'index FTS d'une loi existante."""
        await self.index_law(law)
        logger.info(f"✅ Updated PG FTS index for law {law_id}")

    async def delete_law_index(self, law_id: int) -> None:
        """
        Supprime/vide le search_vector d'une loi.
        Les articles sont nettoyés en cascade par le trigger.

        Args:
            law_id: ID de la loi à désindexer
        """
        try:
            await remove_law_search_index(self.db, law_id)
            logger.info(f"✅ PG FTS deindexed law {law_id}")
        except Exception as e:
            logger.error(f"❌ Failed to deindex law {law_id}: {e}")
            raise IndexingError(f"Échec désindexation law {law_id}: {e}") from e

    async def reindex_all_laws(self) -> ReindexResponse:
        """
        Réindexe toutes les lois publiées (opération admin).
        Met à jour les tsvector de toutes les lois et articles.

        Returns:
            ReindexResponse avec statistiques
        """
        assert self.db is not None, "Database session must be initialized"
        start_time = time.time()

        try:
            # Recalcule tous les tsvectors en masse
            result = await self.db.execute(text("UPDATE laws SET search_vector = "
                "to_tsvector('french', coalesce(title,'') || ' ' || coalesce(content,''))"
                " || to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))"))
            laws_updated = getattr(result, "rowcount", 0)

            result2 = await self.db.execute(text("UPDATE articles SET search_vector = "
                "to_tsvector('french', coalesce(content,''))"
                " || to_tsvector('english', coalesce(content,''))"
                " || to_tsvector('simple', coalesce(number,''))"))
            articles_updated = getattr(result2, "rowcount", 0)

            await self.db.commit()
            elapsed_seconds = time.time() - start_time

            logger.info(
                f"✅ Reindex complete: {laws_updated} laws, "
                f"{articles_updated} articles in {elapsed_seconds:.1f}s"
            )

            return ReindexResponse(
                status="success",
                total_laws=laws_updated,
                indexed_count=laws_updated,
                failed_count=0,
                duration_seconds=int(elapsed_seconds),
            )

        except Exception as e:
            logger.error(f"❌ Reindexing failed: {e}")
            raise IndexingError(f"Échec réindexation: {e}") from e

    async def invalidate_cache(self) -> int:
        """
        Invalide tout le cache PostgreSQL de recherche.

        Returns:
            Nombre d'entrées supprimées
        """
        try:
            result = await self.db.execute(text("DELETE FROM query_cache"))
            deleted = getattr(result, "rowcount", 0)
            await self.db.commit()
            logger.info(f"🗑️ PG cache invalidated: {deleted} entries deleted")
            return deleted
        except Exception as e:
            logger.warning(f"⚠️ Cache invalidation error: {e}")
            return 0

    # ==================== PRIVATE HELPERS ====================

    def _apply_filters_pgvector(self, stmt: Any, filters: Optional[SearchFilters]) -> Any:
        """Applique les filtres SQL à la query pgvector."""
        if not filters:
            return stmt

        if filters.language:
            stmt = stmt.where(Law.language == filters.language)
        if filters.category_ids:
            stmt = stmt.where(Law.category_id.in_(filters.category_ids))
        if filters.types:
            stmt = stmt.where(Law.type.in_(filters.types))
        if filters.status:
            stmt = stmt.where(Law.status == filters.status)

        return stmt

    def _rrf_fusion(
        self,
        text_results: List[SearchResult],
        semantic_results: List[SearchResult],
        k: int = 60,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[SearchResult]:
        """Applique l'algorithme RRF (Reciprocal Rank Fusion) avec poids."""
        scores: Dict[int, Dict] = {}

        for rank, result in enumerate(text_results, start=1):
            scores[result.law_id] = {
                "text_rank": rank,
                "semantic_rank": None,
                "result": result,
            }

        for rank, result in enumerate(semantic_results, start=1):
            if result.law_id in scores:
                scores[result.law_id]["semantic_rank"] = rank
                scores[result.law_id]["result"].highlights.update(result.highlights)
            else:
                scores[result.law_id] = {
                    "text_rank": None,
                    "semantic_rank": rank,
                    "result": result,
                }

        for law_id, data in scores.items():
            rrf_score = 0.0
            if data["text_rank"]:
                rrf_score += text_weight * (1.0 / (k + data["text_rank"]))
            if data["semantic_rank"]:
                rrf_score += semantic_weight * (1.0 / (k + data["semantic_rank"]))
            data["rrf_score"] = rrf_score
            data["result"].relevance_score = rrf_score

        sorted_results = sorted(
            scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )
        return [item["result"] for item in sorted_results]

    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalise les scores à [0, 1]."""
        if not results:
            return results

        scores = [r.relevance_score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score < 1e-9:
            for result in results:
                result.relevance_score = 1.0
        else:
            for result in results:
                result.relevance_score = (
                    (result.relevance_score - min_score) / (max_score - min_score)
                )

        return results

    def _extract_highlight_snippet(self, content: str, max_length: int = 300) -> str:
        """Extrait un snippet de contenu."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."


# ==================== COMPATIBILITY FUNCTION ====================


async def invalidate_search_cache(db: AsyncSession) -> int:
    """
    Helper global pour invalider le cache depuis n'importe où.
    À appeler quand des lois sont ajoutées/modifiées/supprimées.

    Args:
        db: Session async SQLAlchemy

    Returns:
        Nombre de clés supprimées
    """
    try:
        result = await db.execute(text("DELETE FROM query_cache"))
        deleted = getattr(result, "rowcount", 0)
        await db.commit()
        logger.info(f"🗑️ Global PG cache invalidation: {deleted} search entries deleted")
        return deleted
    except Exception as e:
        logger.warning(f"⚠️ Global cache invalidation error: {e}")
        return 0
