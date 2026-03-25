"""
Service de recherche hybride (textuelle + sémantique) pour JuriX.

Ce service implémente 3 modes de recherche:
1. Textuelle (Meilisearch) - Recherche full-text rapide (<50ms)
2. Sémantique (pgvector) - Recherche contextuelle par similarité (<200ms)
3. Hybrid (RRF fusion 40/60) - Combinaison optimale (<200ms)

Architecture:
- Meilisearch: Full-text search avec typo tolerance
- pgvector: Vector similarity avec HNSW index
- RRF Fusion: Weighted Reciprocal Rank Fusion (40% text + 60% semantic)
- Redis: Cache résultats avec TTL 1h

Performance cible:
- Text search: <50ms
- Semantic search: <200ms
- Hybrid search: <200ms (spec requirement)
- Indexing: <100ms per law

Usage:
    service = SearchService(db_session)

    # Recherche hybrid
    response = await service.search(SearchRequest(
        query="responsabilité dirigeants société",
        mode="hybrid",
        filters=SearchFilters(language="fr"),
        limit=15
    ))

    # Indexation automatique
    await service.index_law(law)

Author: JuriX Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import meilisearch
import numpy as np
import redis
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

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

logger = logging.getLogger(__name__)

# ==================== GLOBAL SINGLETONS ====================
# These are initialized once at module load / first access to avoid
# reloading the heavy embedding model on every request.

_embedding_service_instance: Optional[EmbeddingService] = None
_meilisearch_client_instance: Optional[meilisearch.Client] = None
_redis_client_instance: Optional[redis.Redis] = None
_singletons_initialized: bool = False


def _init_global_singletons() -> None:
    """
    Initialize global singleton instances for performance.
    Called once at first SearchService instantiation.
    """
    global _embedding_service_instance, _meilisearch_client_instance, _redis_client_instance, _singletons_initialized
    
    if _singletons_initialized:
        return
    
    logger.info("🚀 Initializing global search singletons (one-time)...")
    
    # 1. Meilisearch client (fast, lightweight)
    try:
        _meilisearch_client_instance = meilisearch.Client(
            settings.MEILISEARCH_URL,
            settings.MEILISEARCH_KEY
        )
        _meilisearch_client_instance.health()
        
        # Ensure index exists
        try:
            _meilisearch_client_instance.get_index(SearchService.MEILISEARCH_INDEX)
        except Exception:
            _meilisearch_client_instance.create_index(
                SearchService.MEILISEARCH_INDEX,
                {"primaryKey": "id"}
            )
        logger.info("✅ Meilisearch client initialized")
    except Exception as e:
        logger.error(f"❌ Meilisearch init failed: {e}")
        _meilisearch_client_instance = None
    
    # 2. Redis client (fast)
    try:
        _redis_client_instance = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        _redis_client_instance.ping()
        logger.info("✅ Redis client initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis init failed (cache disabled): {e}")
        _redis_client_instance = None
    
    # 3. EmbeddingService (SLOW - loads 120MB model)
    try:
        _embedding_service_instance = EmbeddingService(
            redis_url=settings.REDIS_URL,
            use_cache=True
        )
        logger.info("✅ EmbeddingService initialized")
    except Exception as e:
        logger.error(f"❌ EmbeddingService init failed: {e}")
        _embedding_service_instance = None
    
    _singletons_initialized = True
    logger.info("✅ All global singletons ready")


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get the global Redis client instance.
    
    Returns:
        Redis client or None if not initialized
    """
    global _redis_client_instance, _singletons_initialized
    
    if not _singletons_initialized:
        _init_global_singletons()
    
    return _redis_client_instance


# ==================== EXCEPTIONS ====================


class SearchServiceError(Exception):
    """Exception de base pour SearchService."""
    pass


class MeilisearchError(SearchServiceError):
    """Erreur liée à Meilisearch (connexion, query, indexing)."""
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

    Fournit 3 modes de recherche avec fusion RRF et caching Redis:
    - Text: Meilisearch full-text (mots-clés, typos, proximité)
    - Semantic: pgvector cosine similarity (contexte, synonymes)
    - Hybrid: RRF fusion 40% text + 60% semantic (optimal)

    Caractéristiques:
    - Performance: <200ms hybrid search (spec requirement)
    - Filters: language, category, type, status, dates
    - Pagination: limit/offset
    - Highlights: Extraits avec <mark> tags
    - Caching: Redis TTL 1h avec invalidation
    - Auto-indexing: Hooks dans LawService

    Attributes:
        db: Session database async SQLAlchemy
        embedding_service: Service embeddings (singleton)
        meilisearch_client: Client Meilisearch
        redis_client: Client Redis (optional)
        use_cache: Flag activation cache
    """

    # Configuration
    MEILISEARCH_INDEX = "laws"  # Laws index
    MEILISEARCH_ARTICLES_INDEX = "articles"  # Articles index for article-level search
    CACHE_TTL_SECONDS = 300  # 5 minutes (reduced from 1 hour to prevent stale data)
    RRF_K = 60  # Constante RRF fusion
    TEXT_WEIGHT = 0.4  # Poids recherche textuelle
    SEMANTIC_WEIGHT = 0.6  # Poids recherche sémantique
    MAX_RESULTS_PER_MODE = 20  # Limite avant fusion
    CONTENT_TRUNCATE_LENGTH = 10000  # Chars pour Meilisearch

    def __init__(self, db: AsyncSession, use_cache: bool = True):
        """
        Initialise le service de recherche.

        Args:
            db: Session database async SQLAlchemy
            use_cache: Active le cache Redis (défaut: True)

        Raises:
            MeilisearchError: Si connexion Meilisearch échoue
        """
        # Initialize global singletons once (avoids reloading model every request)
        _init_global_singletons()
        
        self.db = db
        self.use_cache = use_cache

        # Embedding service (singleton)
        self.embedding_service = self._get_embedding_service()

        # Meilisearch client
        self.meilisearch_client = self._init_meilisearch()

        # Redis client (optional)
        self.redis_client = self._init_redis() if use_cache else None

        logger.debug(
            f"SearchService ready (cache={use_cache})"
        )

    # ==================== ARTICLE REFERENCE PARSING ====================

    def _parse_article_reference(self, query: str) -> Optional[Dict[str, str]]:
        """
        Détecte et extrait les références d'articles dans la query.
        
        Patterns supportés:
        - "article 5 de la constitution"
        - "article premier du code civil"
        - "art. 12 de la loi sur..."
        - "article 1er du décret..."
        
        Args:
            query: Requête de recherche originale
            
        Returns:
            Dict avec 'article_num' et 'doc_hint', ou None si pas de référence
            
        Example:
            >>> _parse_article_reference("article 5 de la constitution")
            {'article_num': '5', 'doc_hint': 'constitution'}
        """
        import re
        
        query_lower = query.lower().strip()
        
        # Patterns pour détecter les références d'articles (FR + EN)
        patterns = [
            # French: "article 5 de la constitution" / "article premier du code"
            r"article\s+(premier|1er|1ère|\d+)\s+(?:de\s+)?(?:la\s+|le\s+|l[''']|du\s+|des\s+)?(.+)",
            # French: "art. 12 de la loi"
            r"art\.?\s*(\d+)\s+(?:de\s+)?(?:la\s+|le\s+|l[''']|du\s+)?(.+)",
            # English: "section 5 of the constitution"
            r"section\s+(\d+|one|first)\s+(?:of\s+)?(?:the\s+)?(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                article_num = match.group(1).strip()
                doc_hint = match.group(2).strip() if match.group(2) else ""
                
                # Normaliser le numéro d'article
                article_num_normalized = article_num.upper()
                if article_num_normalized in ("PREMIER", "1ER", "1ÈRE"):
                    article_num_normalized = "PREMIER"
                
                logger.info(
                    f"📌 Detected article reference: article={article_num_normalized}, "
                    f"doc_hint='{doc_hint}'"
                )
                
                return {
                    "article_num": article_num_normalized,
                    "doc_hint": doc_hint
                }
        
        return None

    # ==================== PUBLIC API ====================

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Point d'entrée principal pour la recherche.

        Dispatches to text/semantic/hybrid search, builds response,
        handles fallbacks on errors.

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

        # Check cache
        if self.use_cache and self.redis_client:
            cached_response = self._get_from_cache(request)
            if cached_response:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(f"🎯 Cache HIT ({elapsed_ms}ms)")
                return cached_response

        # Execute search with fallback handling
        try:
            results = await self._execute_search_by_mode(request)
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = self._build_search_response(request, results, elapsed_ms)

            if self.use_cache and self.redis_client:
                self._store_in_cache(request, response)

            logger.info(
                f"✅ Search completed: {response.total} results in {elapsed_ms}ms "
                f"(mode={request.mode})"
            )
            return response

        except MeilisearchError as e:
            return await self._handle_search_fallback(
                request, start_time, "semantic", e, "Meilisearch"
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
            logger.info(f"🎯 Direct navigation enabled: article {target_article}")
        elif article_ref and total > 1:
            target_article = article_ref["article_num"]
            logger.info(f"📌 Article reference detected: {target_article} (multiple results)")

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            results=results,
            total=total,
            search_time_ms=elapsed_ms,
            filters_applied=request.filters.model_dump() if request.filters else None,
            target_article=target_article,
            direct_navigation=direct_navigation
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
            filters_applied=request.filters.model_dump() if request.filters else None
        )

    async def text_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Recherche textuelle via Meilisearch.

        Searches articles index first, falls back to laws index.

        Args:
            query: Requête textuelle
            filters: Filtres optionnels
            limit: Nombre max résultats (défaut: 15)
            offset: Offset pagination (défaut: 0)

        Returns:
            Liste de SearchResult triée par relevance_score

        Raises:
            MeilisearchError: Si recherche échoue
        """
        assert query and isinstance(query, str), "query must be a non-empty string"
        assert limit > 0, "limit must be positive"
        start_time = time.time()

        try:
            # Priority: articles index, then laws index as fallback
            results = self._search_articles_index(query, limit, offset)
            if not results:
                results = self._search_laws_index(query, filters, limit, offset)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"📝 Text search total: {len(results)} results in {elapsed_ms}ms")
            return results

        except Exception as e:
            logger.error(f"❌ Text search failed: {e}")
            raise MeilisearchError(f"Échec recherche textuelle: {e}") from e

    def _search_articles_index(
        self, query: str, limit: int, offset: int
    ) -> List[SearchResult]:
        """Search articles Meilisearch index for article-level content."""
        try:
            articles_index = self.meilisearch_client.index(self.MEILISEARCH_ARTICLES_INDEX)
            articles_params = {
                "limit": min(limit, self.MAX_RESULTS_PER_MODE),
                "offset": offset,
                "attributesToRetrieve": [
                    "id", "number", "title", "section", "content",
                    "law_id", "law_title", "law_reference", "category"
                ],
                "attributesToHighlight": ["content", "title"],
                "highlightPreTag": "<mark>",
                "highlightPostTag": "</mark>",
            }

            articles_response = articles_index.search(query, articles_params)
            results = []
            seen_law_ids = set()

            for rank, hit in enumerate(articles_response.get("hits", []), start=1):
                law_id = hit.get("law_id")
                if law_id in seen_law_ids:
                    continue
                seen_law_ids.add(law_id)

                relevance_score = 1.0 / (1.0 + rank * 0.1)
                content = hit.get("content", "")
                highlights = {}
                if "_formatted" in hit:
                    formatted = hit["_formatted"]
                    if "content" in formatted:
                        highlights["content"] = self._extract_highlight_snippet(
                            formatted["content"], max_length=500
                        )

                results.append(SearchResult(
                    law_id=law_id,
                    reference=hit.get("law_reference", ""),
                    title=hit.get("law_title", ""),
                    type="loi",
                    language="fr",
                    status="published",
                    category_id=None,
                    category_name=hit.get("category"),
                    publication_date=None,
                    relevance_score=relevance_score,
                    matched_articles=[ArticleMatch(
                        article_id=hit.get("id", 0),
                        number=hit.get("number", ""),
                        title=hit.get("title"),
                        content_snippet=content[:300] if content else "",
                        relevance_score=relevance_score
                    )],
                    highlights=highlights,
                    content=content
                ))

            logger.info(f"📝 Articles search: {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"⚠️ Articles index search failed: {e}")
            return []

    def _search_laws_index(
        self, query: str, filters: Optional[SearchFilters],
        limit: int, offset: int
    ) -> List[SearchResult]:
        """Search laws Meilisearch index as fallback."""
        try:
            filter_string = self._apply_filters_meilisearch(filters)
            index = self.meilisearch_client.index(self.MEILISEARCH_INDEX)

            search_params = {
                "limit": min(limit, self.MAX_RESULTS_PER_MODE),
                "offset": offset,
                "attributesToRetrieve": [
                    "id", "reference", "title", "type", "language",
                    "status", "category_id", "category_name",
                    "publication_year", "content"
                ],
                "attributesToHighlight": ["title", "content"],
                "highlightPreTag": "<mark>",
                "highlightPostTag": "</mark>",
            }
            if filter_string:
                search_params["filter"] = filter_string

            meili_response = index.search(query, search_params)
            results = []

            for rank, hit in enumerate(meili_response["hits"], start=1):
                relevance_score = 1.0 / (1.0 + rank * 0.1)
                highlights = {}
                if "_formatted" in hit:
                    formatted = hit["_formatted"]
                    if "title" in formatted:
                        highlights["title"] = formatted["title"]
                    if "content" in formatted:
                        highlights["content"] = self._extract_highlight_snippet(
                            formatted["content"], max_length=300
                        )

                results.append(SearchResult(
                    law_id=hit["id"],
                    reference=hit["reference"],
                    title=hit["title"],
                    type=hit["type"],
                    language=hit.get("language"),
                    status=hit["status"],
                    category_id=hit.get("category_id"),
                    category_name=hit.get("category_name"),
                    publication_date=None,
                    relevance_score=relevance_score,
                    matched_articles=[],
                    highlights=highlights,
                    content=hit.get("content", "")
                ))

            logger.info(f"📝 Laws search: {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"⚠️ Laws index search failed: {e}")
            return []

    async def semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Recherche sémantique via pgvector.

        Effectue une recherche par similarité cosine sur embeddings:
        1. Génère embedding de la query (3072-dim)
        2. Recherche KNN dans articles.embedding
        3. Agrège résultats par law_id (score max)
        4. Applique filtres et pagination

        Args:
            query: Requête textuelle
            filters: Filtres optionnels
            limit: Nombre max résultats (défaut: 15)
            offset: Offset pagination (défaut: 0)

        Returns:
            Liste de SearchResult triée par relevance_score (similarity)

        Raises:
            VectorSearchError: Si recherche échoue

        Performance: <200ms avec HNSW index
        """
        assert isinstance(query, str) and len(query) > 0, "Query must be a non-empty string"
        assert isinstance(limit, int) and limit > 0, "Limit must be a positive integer"

        start_time = time.time()

        try:
            # Check if embedding service is available
            if self.embedding_service is None:
                logger.warning("⚠️ Semantic search unavailable: EmbeddingService not loaded")
                return []
            
            # Génération embedding query
            query_embedding = self.embedding_service.generate_embedding(query)
            query_embedding_str = f"[{','.join(map(str, query_embedding.tolist()))}]"

            # Construction query pgvector avec similarité cosine réelle
            # Using pgvector's cosine_distance: 1 - cosine_distance = cosine similarity
            # Lower distance = higher similarity (0 = identical)
            from sqlalchemy import cast
            from sqlalchemy.types import String
            
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
                        # Cosine similarity: 1 - cosine_distance
                        # Higher score = more similar (0.0 to 1.0)
                        1 - func.cosine_distance(
                            Article.embedding,
                            cast(query_embedding_str, String)
                        )
                    ).label("similarity")
                )
                .join(Article, Law.id == Article.law_id)
                .outerjoin(Category, Law.category_id == Category.id)
                .where(Article.embedding.isnot(None))
            )

            # Application filtres
            stmt = self._apply_filters_pgvector(stmt, filters)

            # Group by law, order by similarity
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

            # Exécution
            result = await self.db.execute(stmt)
            rows = result.all()

            # Conversion vers SearchResult
            results = []
            for row in rows:
                result_obj = SearchResult(
                    law_id=row.id,
                    reference=row.reference,
                    title=row.title,
                    type=row.type,
                    language=row.language,
                    status=row.status,
                    category_id=row.category_id,
                    category_name=row.category_name,
                    publication_date=None,
                    relevance_score=float(row.similarity),  # Déjà normalisé 0-1
                    matched_articles=[],  # TODO: Récupérer articles matchés
                    highlights={}  # Semantic search ne produit pas highlights
                )
                results.append(result_obj)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"🧠 Semantic search: {len(results)} results in {elapsed_ms}ms"
            )

            return results

        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            raise VectorSearchError(f"Échec recherche sémantique: {e}") from e

    async def hybrid_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Recherche hybride via RRF fusion (40% text + 60% semantic).

        Stratégie:
        1. Exécute text_search et semantic_search en parallèle (asyncio.gather)
        2. Applique RRF fusion avec poids 40/60
        3. Normalise scores finaux à [0, 1]
        4. Déduplique par law_id (garde meilleur score)
        5. Trie par RRF score décroissant
        6. Applique pagination

        Formule RRF:
            rrf_score = 0.4 × (1/(60 + text_rank)) + 0.6 × (1/(60 + semantic_rank))

        Args:
            query: Requête textuelle
            filters: Filtres optionnels
            limit: Nombre max résultats (défaut: 15)
            offset: Offset pagination (défaut: 0)

        Returns:
            Liste de SearchResult triée par RRF score

        Raises:
            SearchServiceError: Si les deux modes échouent

        Performance: <200ms (spec requirement)
        """
        assert isinstance(query, str) and len(query) > 0, "Query must be a non-empty string"
        assert isinstance(limit, int) and limit > 0, "Limit must be a positive integer"

        start_time = time.time()

        try:
            # Exécution parallèle des deux modes
            text_results, semantic_results = await asyncio.gather(
                self.text_search(
                    query,
                    filters,
                    limit=self.MAX_RESULTS_PER_MODE,
                    offset=0  # Fusion puis pagination
                ),
                self.semantic_search(
                    query,
                    filters,
                    limit=self.MAX_RESULTS_PER_MODE,
                    offset=0
                ),
                return_exceptions=True
            )

            # Gestion erreurs
            if isinstance(text_results, Exception):
                logger.warning(f"⚠️ Text search failed in hybrid: {text_results}")
                text_results = []

            if isinstance(semantic_results, Exception):
                logger.warning(f"⚠️ Semantic search failed in hybrid: {semantic_results}")
                semantic_results = []

            if not text_results and not semantic_results:
                logger.info("🔀 Hybrid search: No results found in either text or semantic search")
                return []

            # Application RRF fusion
            fused_results = self._rrf_fusion(
                text_results,
                semantic_results,
                k=self.RRF_K,
                text_weight=self.TEXT_WEIGHT,
                semantic_weight=self.SEMANTIC_WEIGHT
            )

            # Normalisation scores
            fused_results = self._normalize_scores(fused_results)

            # Pagination
            paginated_results = fused_results[offset:offset + limit]

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"🔀 Hybrid search: {len(paginated_results)} results in {elapsed_ms}ms "
                f"(text={len(text_results)}, semantic={len(semantic_results)}, "
                f"fused={len(fused_results)})"
            )

            return paginated_results

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise SearchServiceError(f"Échec recherche hybride: {e}") from e

    # ==================== INDEXING METHODS ====================

    async def index_law(self, law: Law) -> None:
        """
        Indexe une loi dans Meilisearch.

        Appelé automatiquement lors de la création d'une loi (hook LawService).
        Crée un document Meilisearch avec:
        - Champs searchable: title, reference, content (tronqué 10k)
        - Champs filterable: id, category_id, type, language, status, year
        - Metadata: category_name, timestamps

        Args:
            law: Instance Law à indexer

        Raises:
            IndexingError: Si indexation échoue

        Performance: <100ms
        """
        try:
            # Construction document Meilisearch
            document = {
                "id": law.id,
                "reference": law.reference,
                "title": law.title,
                "content": law.content[:self.CONTENT_TRUNCATE_LENGTH],
                "type": law.type,
                "language": law.language or "unknown",
                "status": law.status,
                "category_id": law.category_id,
                "category_name": law.category.name if law.category else None,
                "publication_year": (
                    law.publication_date.year
                    if law.publication_date else None
                ),
                "created_at_timestamp": int(law.created_at.timestamp())
            }

            # Indexation
            index = self.meilisearch_client.index(self.MEILISEARCH_INDEX)
            index.add_documents([document])

            logger.info(f"✅ Indexed law {law.id} ({law.reference})")

        except Exception as e:
            logger.error(f"❌ Failed to index law {law.id}: {e}")
            raise IndexingError(f"Échec indexation law {law.id}: {e}") from e

    async def update_law_index(self, law_id: int, law: Law) -> None:
        """
        Met à jour l'index d'une loi existante.

        Meilisearch effectue automatiquement un upsert basé sur l'ID.

        Args:
            law_id: ID de la loi (pour cohérence API)
            law: Instance Law mise à jour

        Raises:
            IndexingError: Si mise à jour échoue
        """
        # Meilisearch upsert automatique via add_documents
        await self.index_law(law)
        logger.info(f"✅ Updated index for law {law_id}")

    async def delete_law_index(self, law_id: int) -> None:
        """
        Supprime une loi de l'index Meilisearch.

        Args:
            law_id: ID de la loi à supprimer

        Raises:
            IndexingError: Si suppression échoue
        """
        try:
            index = self.meilisearch_client.index(self.MEILISEARCH_INDEX)
            index.delete_document(law_id)
            logger.info(f"✅ Deleted law {law_id} from index")

        except Exception as e:
            logger.error(f"❌ Failed to delete law {law_id} from index: {e}")
            raise IndexingError(f"Échec suppression index law {law_id}: {e}") from e

    async def reindex_all_laws(self) -> ReindexResponse:
        """
        Réindexe toutes les lois publiées (opération admin).

        Stratégie:
        1. Récupère toutes les lois publiées depuis DB
        2. Construit documents batch
        3. Supprime ancien index
        4. Recrée index avec configuration
        5. Indexe documents en batch (1000 par batch)

        Returns:
            ReindexResponse avec statistiques

        Raises:
            IndexingError: Si réindexation échoue

        Performance: <5s pour 100 lois
        """
        assert self.meilisearch_client is not None, "Meilisearch client must be initialized"
        assert self.db is not None, "Database session must be initialized"

        start_time = time.time()

        try:
            # Récupération lois publiées
            stmt = (
                select(Law)
                .where(Law.status == "published")
                .options(joinedload(Law.category))
            )
            result = await self.db.execute(stmt)
            laws = result.scalars().all()

            total_laws = len(laws)
            logger.info(f"🔄 Reindexing {total_laws} laws...")

            # Construction documents
            documents = []
            for law in laws:
                doc = {
                    "id": law.id,
                    "reference": law.reference,
                    "title": law.title,
                    "content": law.content[:self.CONTENT_TRUNCATE_LENGTH],
                    "type": law.type,
                    "language": law.language or "unknown",
                    "status": law.status,
                    "category_id": law.category_id,
                    "category_name": law.category.name if law.category else None,
                    "publication_year": (
                        law.publication_date.year
                        if law.publication_date else None
                    ),
                    "created_at_timestamp": int(law.created_at.timestamp())
                }
                documents.append(doc)

            # Suppression ancien index (si existe)
            try:
                self.meilisearch_client.delete_index(self.MEILISEARCH_INDEX)
                logger.info("Deleted old index")
            except Exception:
                pass  # Index n'existait pas

            # Création index avec configuration
            self.meilisearch_client.create_index(
                self.MEILISEARCH_INDEX,
                {"primaryKey": "id"}
            )

            # Get the index object
            index = self.meilisearch_client.index(self.MEILISEARCH_INDEX)

            # Configuration ranking rules
            index.update_settings({
                "searchableAttributes": ["title", "reference", "content"],
                "filterableAttributes": [
                    "id", "category_id", "type", "language",
                    "status", "publication_year"
                ],
                "sortableAttributes": ["publication_year", "created_at_timestamp"],
                "rankingRules": [
                    "words",
                    "typo",
                    "proximity",
                    "attribute",
                    "sort",
                    "exactness"
                ],
                "typoTolerance": {
                    "enabled": True,
                    "minWordSizeForTypos": {
                        "oneTypo": 4,
                        "twoTypos": 8
                    }
                }
            })

            # Indexation batch (1000 par batch)
            batch_size = 1000
            indexed_count = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                index.add_documents(batch)
                indexed_count += len(batch)
                logger.info(f"📦 Indexed batch {i//batch_size + 1}: {indexed_count}/{total_laws}")

            elapsed_seconds = time.time() - start_time

            logger.info(
                f"✅ Reindexing complete: {indexed_count} laws in {elapsed_seconds:.1f}s"
            )

            return ReindexResponse(
                status="success",
                total_laws=total_laws,
                indexed_count=indexed_count,
                failed_count=0,
                duration_seconds=int(elapsed_seconds)
            )

        except Exception as e:
            logger.error(f"❌ Reindexing failed: {e}")
            raise IndexingError(f"Échec réindexation: {e}") from e

    # ==================== PRIVATE METHODS ====================

    def _get_embedding_service(self) -> Optional[EmbeddingService]:
        """
        Récupère le singleton EmbeddingService.
        Returns None if embedding service is unavailable (text search still works).

        Returns:
            Instance EmbeddingService or None
        """
        global _embedding_service_instance
        
        if _embedding_service_instance is None:
            # Try to create instance, but don't fail if unavailable
            try:
                from app.services.embedding_service import EmbeddingService
                _embedding_service_instance = EmbeddingService(
                    redis_url=settings.REDIS_URL,
                    use_cache=True
                )
            except Exception as e:
                logger.warning(f"⚠️ EmbeddingService unavailable (text search only): {e}")
                return None
        
        return _embedding_service_instance

    def _init_meilisearch(self) -> meilisearch.Client:
        """
        Récupère le singleton Meilisearch client.

        Returns:
            Client Meilisearch

        Raises:
            MeilisearchError: Si connexion échoue
        """
        global _meilisearch_client_instance
        
        if _meilisearch_client_instance is not None:
            return _meilisearch_client_instance
        
        # Fallback: create if not initialized
        try:
            client = meilisearch.Client(
                settings.MEILISEARCH_URL,
                settings.MEILISEARCH_KEY
            )
            client.health()
            
            try:
                client.get_index(self.MEILISEARCH_INDEX)
            except Exception:
                client.create_index(
                    self.MEILISEARCH_INDEX,
                    {"primaryKey": "id"}
                )
            
            _meilisearch_client_instance = client
            return client

        except Exception as e:
            logger.error(f"❌ Meilisearch connection failed: {e}")
            raise MeilisearchError(f"Connexion Meilisearch échouée: {e}") from e

    def _init_redis(self) -> Optional[redis.Redis]:
        """
        Récupère le singleton Redis client pour le cache.

        Returns:
            Client Redis ou None si connexion échoue
        """
        global _redis_client_instance
        
        if _redis_client_instance is not None:
            return _redis_client_instance
        
        # Fallback: create if not initialized
        try:
            client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            client.ping()
            _redis_client_instance = client
            return client

        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, cache disabled: {e}")
            return None

    def _apply_filters_meilisearch(
        self,
        filters: Optional[SearchFilters]
    ) -> Optional[str]:
        """
        Construit la chaîne de filtres pour Meilisearch.

        Format Meilisearch: "language = fr AND (category_id = 1 OR category_id = 2)"

        Args:
            filters: Filtres SearchFilters

        Returns:
            Chaîne de filtres ou None si aucun filtre
        """
        if not filters:
            return None

        conditions = []

        if filters.language:
            conditions.append(f"language = {filters.language}")

        if filters.category_ids:
            cat_conditions = " OR ".join(
                f"category_id = {cid}" for cid in filters.category_ids
            )
            conditions.append(f"({cat_conditions})")

        if filters.types:
            type_conditions = " OR ".join(
                f"type = {typ}" for typ in filters.types
            )
            conditions.append(f"({type_conditions})")

        if filters.status:
            conditions.append(f"status = {filters.status}")

        if filters.year_from:
            conditions.append(f"publication_year >= {filters.year_from}")

        if filters.year_to:
            conditions.append(f"publication_year <= {filters.year_to}")

        return " AND ".join(conditions) if conditions else None

    def _apply_filters_pgvector(
        self,
        stmt: Any,
        filters: Optional[SearchFilters]
    ) -> Any:
        """
        Applique les filtres SQL à la query pgvector.

        Args:
            stmt: Statement SQLAlchemy
            filters: Filtres SearchFilters

        Returns:
            Statement modifié avec WHERE clauses
        """
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

        if filters.year_from or filters.year_to:
            # TODO: Filtrer par année publication_date
            pass

        return stmt

    def _rrf_fusion(
        self,
        text_results: List[SearchResult],
        semantic_results: List[SearchResult],
        k: int = 60,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6
    ) -> List[SearchResult]:
        """
        Applique l'algorithme RRF (Reciprocal Rank Fusion) avec poids.

        Formule: rrf_score = text_weight × (1/(k + text_rank)) +
                            semantic_weight × (1/(k + semantic_rank))

        Args:
            text_results: Résultats recherche textuelle
            semantic_results: Résultats recherche sémantique
            k: Constante RRF (défaut: 60)
            text_weight: Poids résultats textuels (défaut: 0.4)
            semantic_weight: Poids résultats sémantiques (défaut: 0.6)

        Returns:
            Liste fusionnée triée par RRF score décroissant
        """
        scores = {}

        # Indexation résultats textuels (rank-based)
        for rank, result in enumerate(text_results, start=1):
            scores[result.law_id] = {
                "text_rank": rank,
                "semantic_rank": None,
                "result": result
            }

        # Indexation résultats sémantiques (rank-based)
        for rank, result in enumerate(semantic_results, start=1):
            if result.law_id in scores:
                scores[result.law_id]["semantic_rank"] = rank
                # Merge highlights et matched_articles
                scores[result.law_id]["result"].highlights.update(result.highlights)
            else:
                scores[result.law_id] = {
                    "text_rank": None,
                    "semantic_rank": rank,
                    "result": result
                }

        # Calcul RRF scores
        for law_id, data in scores.items():
            rrf_score = 0.0

            if data["text_rank"]:
                rrf_score += text_weight * (1.0 / (k + data["text_rank"]))

            if data["semantic_rank"]:
                rrf_score += semantic_weight * (1.0 / (k + data["semantic_rank"]))

            data["rrf_score"] = rrf_score
            data["result"].relevance_score = rrf_score  # Temporaire, sera normalisé

        # Tri par RRF score décroissant
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return [item["result"] for item in sorted_results]

    def _normalize_scores(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Normalise les scores à [0, 1].

        Args:
            results: Liste SearchResult avec scores bruts

        Returns:
            Liste avec scores normalisés
        """
        if not results:
            return results

        # Min-max normalization
        scores = [r.relevance_score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score < 1e-9:  # Tous les scores identiques
            for result in results:
                result.relevance_score = 1.0
        else:
            for result in results:
                result.relevance_score = (
                    (result.relevance_score - min_score) /
                    (max_score - min_score)
                )

        return results

    def _extract_highlight_snippet(
        self,
        content: str,
        max_length: int = 300
    ) -> str:
        """
        Extrait un snippet de contenu centré sur les <mark> tags.

        Args:
            content: Contenu avec <mark> tags
            max_length: Longueur max snippet (défaut: 300)

        Returns:
            Snippet tronqué avec "..." si nécessaire
        """
        if len(content) <= max_length:
            return content

        # Recherche premier <mark>
        mark_pos = content.find("<mark>")

        if mark_pos == -1:
            # Pas de highlight, retourne début
            return content[:max_length] + "..."

        # Centrer snippet autour du highlight
        start = max(0, mark_pos - max_length // 2)
        end = start + max_length

        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def _get_cache_key(self, request: SearchRequest) -> str:
        """
        Génère la clé de cache pour une requête.

        Args:
            request: Requête de recherche

        Returns:
            Clé cache format "search:{mode}:{md5_hash}"
        """
        request_dict = {
            "query": request.query,
            "mode": request.mode,
            "filters": request.filters.model_dump() if request.filters else None,
            "limit": request.limit,
            "offset": request.offset
        }

        request_str = json.dumps(request_dict, sort_keys=True)
        hash_digest = hashlib.md5(request_str.encode()).hexdigest()

        return f"search:{request.mode}:{hash_digest}"

    def _get_from_cache(
        self,
        request: SearchRequest
    ) -> Optional[SearchResponse]:
        """
        Récupère une réponse du cache Redis.

        Args:
            request: Requête de recherche

        Returns:
            SearchResponse si trouvé, None sinon
        """
        if not self.redis_client:
            return None

        try:
            cache_key = self._get_cache_key(request)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                # Désérialisation JSON -> Pydantic
                response_dict = json.loads(cached_data)
                return SearchResponse(**response_dict)

        except Exception as e:
            logger.warning(f"⚠️ Cache read error: {e}")

        return None

    def _store_in_cache(
        self,
        request: SearchRequest,
        response: SearchResponse
    ) -> None:
        """
        Stocke une réponse dans le cache Redis.

        Args:
            request: Requête de recherche
            response: Réponse à cacher
        """
        if not self.redis_client:
            return

        try:
            cache_key = self._get_cache_key(request)

            # Sérialisation Pydantic -> JSON
            response_dict = response.model_dump()
            cached_data = json.dumps(response_dict)

            # Stockage avec TTL
            self.redis_client.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                cached_data
            )

        except Exception as e:
            logger.warning(f"⚠️ Cache write error: {e}")

    def invalidate_cache(self) -> int:
        """
        Invalide tout le cache de recherche.
        
        À appeler quand les lois sont ajoutées/modifiées/supprimées.
        
        Returns:
            Nombre de clés supprimées
        """
        if not self.redis_client:
            return 0
        
        try:
            # Find all search-related keys
            keys = self.redis_client.keys("search:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"🗑️ Cache invalidated: {deleted} keys deleted")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"⚠️ Cache invalidation error: {e}")
            return 0


# Global function to invalidate cache from anywhere
def invalidate_search_cache() -> int:
    """
    Invalidate all search cache entries.
    
    Call this when laws are added/updated/deleted to ensure
    search results are up-to-date.
    
    Returns:
        Number of cache keys deleted
    """
    try:
        redis_client = get_redis_client()
        if redis_client:
            keys = redis_client.keys("search:*")
            if keys:
                deleted = redis_client.delete(*keys)
                logger.info(f"🗑️ Global cache invalidation: {deleted} search keys deleted")
                return deleted
        return 0
    except Exception as e:
        logger.warning(f"⚠️ Global cache invalidation error: {e}")
        return 0

