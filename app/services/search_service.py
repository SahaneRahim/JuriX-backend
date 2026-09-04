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
    ChunkResult,
    ReindexResponse,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchStats,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.reranker import rerank_chunks
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
        # Passe par la fabrique partagee : construire l'instance ici en creait
        # une seconde, distincte de celle des dependances FastAPI.
        _embedding_service_instance = get_embedding_service()
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
    # Lues dans settings : ces trois valeurs seront calibrees sur le jeu
    # d'evaluation, et l'exploitation doit pouvoir les changer sans redeployer.
    # RRF_K = 60 vient du papier RRF d'origine, sur des runs TREC — ce n'est
    # pas une mesure sur ce corpus.
    RRF_K = settings.RRF_K
    TEXT_WEIGHT = settings.TEXT_WEIGHT
    SEMANTIC_WEIGHT = settings.SEMANTIC_WEIGHT
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

        chunks = await self._execute_search_by_mode(request)

        # Re-ranking AVANT la construction de la reponse, donc avant la
        # troncature a `limit` ET avant l'ecriture du cache : reclasser apres
        # coup laisserait cinq minutes de reponses pre-reranking en circulation,
        # et reclasser apres troncature ne pourrait plus rien remonter.
        if settings.RERANK_ENABLED and chunks:
            chunks = rerank_chunks(request.query, chunks)

        elapsed_ms = int((time.time() - start_time) * 1000)
        response = self._build_search_response(request, chunks, elapsed_ms)

        # Store in PostgreSQL cache
        if self.use_cache:
            try:
                await store_in_pg_cache(
                    self.db,
                    cache_key,
                    # mode="json" et non model_dump() nu : la reponse porte une
                    # publication_date, et json.dumps refuse un objet date. Sans
                    # cela l'ecriture levait a chaque fois, l'exception etait
                    # avalee juste en dessous, et le cache ne servait plus jamais.
                    response.model_dump(mode="json"),
                    ttl_seconds=self.CACHE_TTL_SECONDS,
                )
            except Exception as cache_err:
                logger.warning(f"⚠️ Cache write failed (non-fatal): {cache_err}")

        logger.info(
            f"✅ Search completed: {response.total} results in {elapsed_ms}ms "
            f"(mode={request.mode})"
        )
        return response

    async def _execute_search_by_mode(self, request: SearchRequest) -> List[ChunkResult]:
        """Dispatch search to the correct mode handler. Renvoie des chunks."""
        budget = self._chunk_budget(request.limit)

        if request.mode == "text":
            return await self.text_chunks(request.query, request.filters, budget, 0)
        elif request.mode == "semantic":
            return await self.semantic_chunks(request.query, request.filters, budget, 0)
        elif request.mode == "hybrid":
            return await self.hybrid_chunks(request.query, request.filters, budget, 0)
        else:
            raise ValueError(f"Mode invalide: {request.mode}")

    def _build_search_response(self, request, chunks: List[ChunkResult], elapsed_ms):
        """
        Construit la reponse : documents pour le front, chunks pour le RAG.

        Les documents sont derives de la LISTE COMPLETE de chunks, pour que le
        front recoive toujours jusqu'a `limit` lois ; les chunks exposes sont
        eux tronques a `limit`, c'est ce que consomme le RAG.
        """
        results = self._chunks_to_search_results(chunks)
        results = results[request.offset:request.offset + request.limit]
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
            chunks=chunks[:request.limit],
        )

    # _handle_search_fallback a ete supprimee : elle etait injoignable.
    # hybrid_chunks rattrape deja ses deux pannes en interne, et pour un mode
    # non hybride la methode se contentait de relever l'erreur recue.

    # ==================== SEARCH MODES ====================

    # ==================== SEARCH MODES — niveau chunk ====================

    def _chunk_budget(self, limit: int) -> int:
        """
        Nombre de chunks a recuperer pour produire `limit` documents.

        Sur-echantillonnage x4 : plusieurs chunks retombent sur la meme loi, et
        sous filtre PostgreSQL elague APRES le parcours ANN, ce qui peut rendre
        moins de lignes que demande.
        """
        return max(limit * 4, 20)

    async def text_chunks(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[ChunkResult]:
        """
        Recherche textuelle PostgreSQL FTS, au niveau article.

        Cherche d'abord dans les articles, puis dans les lois en repli. Les
        filtres sont desormais transmis a la branche article : ils ne l'etaient
        pas, si bien qu'un status="published" demande par l'appelant — le RAG le
        fait — n'avait aucun effet des que des articles correspondaient.

        Raises:
            TextSearchError: Si la recherche echoue
        """
        assert query and isinstance(query, str), "query must be a non-empty string"
        assert limit > 0, "limit must be positive"
        start_time = time.time()

        try:
            chunks = await search_articles_pg(self.db, query, filters, limit, offset)
            if not chunks:
                chunks = await search_laws_pg(self.db, query, filters, limit, offset)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"📝 Text search total: {len(chunks)} chunks in {elapsed_ms}ms")
            return chunks

        except Exception as e:
            logger.error(f"❌ Text search failed: {e}")
            raise TextSearchError(f"Échec recherche textuelle: {e}") from e

    # Sur-echantillonnage avant le reclassement exact. L'etage ANN travaille en
    # fp16 (halfvec) : l'erreur induite sur un score cosinus est de l'ordre de
    # 1e-3, assez pour permuter des quasi-ex-aequo, pas pour deplacer un chunk
    # pertinent de cent places. Un facteur 8 avec un plancher a 100 couvre
    # largement. Valeurs a calibrer avec scripts/eval/run_eval.py.
    ANN_CANDIDATE_MULTIPLIER = 8
    ANN_MIN_CANDIDATES = 100
    ANN_MAX_CANDIDATES = 500

    def _ann_candidates(self, limit: int) -> int:
        """Nombre de candidats a extraire de l'index avant reclassement exact."""
        return min(
            max(limit * self.ANN_CANDIDATE_MULTIPLIER, self.ANN_MIN_CANDIDATES),
            self.ANN_MAX_CANDIDATES,
        )

    def _build_semantic_statement(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters],
        limit: int,
        offset: int,
    ):
        """
        Enonce de la recherche semantique, en DEUX etages.

        Etage 1, dans la sous-requete : parcours de l'index HNSW, qui est pose
        sur l'expression `embedding::halfvec(3072)` — le type `vector` n'est
        indexable que jusqu'a 2000 dimensions, `halfvec` jusqu'a 4000. On en
        tire `candidates` lignes.

        Etage 2, a l'exterieur : reclassement de ces candidats a la distance
        fp32 EXACTE. Le fp16 ne sert qu'a selectionner, jamais a classer.

        Quatre details portent tout le dispositif, et chacun a une variante
        fausse qui a l'air correcte :

        - La distance exacte est projetee DANS la sous-requete. Calculee a
          l'exterieur, la sous-requete devrait remonter articles.embedding,
          soit 12 Ko par candidat traversant le plan.
        - ORDER BY porte sur l'EXPRESSION, jamais sur l'alias : le
          planificateur apparie l'index sur l'arbre d'expression.
        - Le typmod doit etre present des deux cotes. `embedding::halfvec` sans
          `(3072)` est un noeud d'expression different et n'apparie aucun index.
        - L'offset ne s'applique qu'a l'exterieur : le poser a l'interieur
          jetterait les meilleurs candidats avant le reclassement.

        Extrait dans une methode pour qu'un test puisse le compiler sans base.
        """
        from sqlalchemy import cast, literal
        from pgvector.sqlalchemy import HALFVEC, Vector

        dim = EmbeddingService.EMBEDDING_DIM
        candidates = self._ann_candidates(limit)

        ann_distance = cast(Article.embedding, HALFVEC(dim)).cosine_distance(
            literal(query_vector, HALFVEC(dim))
        )
        exact_distance = Article.embedding.cosine_distance(
            literal(query_vector, Vector(dim))
        )

        inner = (
            select(
                Article.id.label("article_id"),
                Article.number,
                Article.title.label("article_title"),
                Article.section,
                Article.page_number,
                Article.content,
                Article.law_id,
                Law.reference,
                Law.title.label("law_title"),
                Law.type,
                Law.language,
                Law.status,
                Law.category_id,
                Law.publication_date,
                Category.name.label("category_name"),
                exact_distance.label("distance"),
            )
            .select_from(Article)
            .join(Law, Law.id == Article.law_id)
            .outerjoin(Category, Category.id == Law.category_id)
            .where(Article.embedding.isnot(None))
        )
        inner = self._apply_filters_pgvector(inner, filters)
        inner = inner.order_by(ann_distance).limit(candidates).subquery("ann")

        return select(inner).order_by(inner.c.distance).limit(limit).offset(offset)

    async def _apply_hnsw_settings(self, candidates: int, filtered: bool) -> None:
        """
        Reglages de session du parcours HNSW.

        hnsw.ef_search vaut 40 par defaut : un parcours non iteratif ne rend
        JAMAIS plus de lignes que cette valeur. Demander 200 candidats sans le
        relever revient a en reclasser 40 — le sur-echantillonnage devient
        decoratif.

        SET n'accepte pas de parametre lie : la valeur est interpolee APRES
        passage par int(), jamais une saisie.

        Les deux reglages sont sous try : ils n'existent qu'a partir de
        pgvector 0.8, et un serveur plus ancien doit degrader, pas renvoyer 500.
        SET LOCAL exige une transaction ouverte — l'AsyncSession en ouvre une au
        premier execute — et ne vaut que jusqu'a la fin de celle-ci.
        """
        ef_search = int(min(max(40, candidates), 1000))
        try:
            await self.db.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
        except Exception as err:
            logger.debug(f"hnsw.ef_search indisponible: {err}")

        if filtered:
            # Sous filtre, PostgreSQL elague APRES le parcours du graphe et peut
            # rendre moins de lignes que demande. iterative_scan rescanne.
            try:
                await self.db.execute(
                    text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                )
            except Exception as err:
                logger.debug(f"hnsw.iterative_scan indisponible: {err}")

    async def semantic_chunks(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[ChunkResult]:
        """
        Recherche semantique pgvector, au niveau article, en deux etages.

        Trois differences avec la version d'origine, toutes necessaires :

        - Une ligne par ARTICLE. Le `func.max(...) GROUP BY Law.*` d'avant
          calculait bien la distance article par article, puis jetait l'article
          gagnant : le resultat ne portait ni son identite ni son texte.
        - L'index est parcouru via `embedding::halfvec(3072)` puis les
          candidats sont reclasses a la distance fp32 exacte. La forme
          `func.cosine_distance(...)` d'origine compilait en un APPEL DE
          FONCTION que le planificateur ne peut rattacher a aucun index.
        - Le score est borne. `1 - distance` peut etre negatif avec de vrais
          vecteurs (composantes negatives), et relevance_score est declare
          ge=0.0 : la reponse levait alors une ValidationError. Les fixtures de
          test, tirees avec np.random.rand donc toutes positives, le masquaient.

        Raises:
            VectorSearchError: Si la recherche echoue
        """
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(limit, int) and limit > 0
        start_time = time.time()

        try:
            if self.embedding_service is None:
                logger.warning("⚠️ Semantic search unavailable: EmbeddingService not loaded")
                return []

            # TASK_QUERY et non TASK_DOCUMENT : encoder une question comme un
            # document degrade la pertinence en recherche asymetrique.
            # Version async : le client Gemini est synchrone et gelait la boucle
            # d'evenements pour tout l'aller-retour reseau.
            query_embedding = await self.embedding_service.generate_embedding_async(
                query, task_type=self.embedding_service.TASK_QUERY
            )
            # Liste passee en parametre lie, PAS en chaine castee : le
            # convertisseur pgvector attend une liste ou un ndarray.
            query_vector = query_embedding.tolist()

            candidates = self._ann_candidates(limit)
            await self._apply_hnsw_settings(candidates, filters is not None)

            stmt = self._build_semantic_statement(query_vector, filters, limit, offset)
            result = await self.db.execute(stmt)
            rows = result.all()

            chunks = [
                ChunkResult(
                    article_id=row.article_id,
                    law_id=row.law_id,
                    number=str(row.number) if row.number else None,
                    article_title=row.article_title,
                    section=row.section,
                    page_number=row.page_number,
                    content=row.content or "",
                    excerpt=(row.content or "")[:400],
                    reference=row.reference or "",
                    law_title=row.law_title or "",
                    type=row.type or "loi",
                    language=row.language,
                    status=row.status or "published",
                    category_id=row.category_id,
                    category_name=row.category_name,
                    publication_date=row.publication_date,
                    # Score issu de la distance EXACTE, celle qui a servi au
                    # tri : un appelant qui re-trierait sur le score doit
                    # retrouver le meme ordre.
                    relevance_score=max(0.0, min(1.0, 1.0 - float(row.distance))),
                    source="semantic",
                )
                for row in rows
            ]

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"🧠 Semantic search: {len(chunks)} chunks "
                f"({candidates} candidats ANN) in {elapsed_ms}ms"
            )
            return chunks

        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            raise VectorSearchError(f"Échec recherche sémantique: {e}") from e

    async def hybrid_chunks(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[ChunkResult]:
        """
        Recherche hybride au niveau article, fusion RRF (40% texte + 60% semantique).
        """
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(limit, int) and limit > 0
        start_time = time.time()

        try:
            # SEQUENTIEL, pas asyncio.gather : les deux recherches partagent
            # self.db, et une AsyncSession n'est pas utilisable par deux
            # coroutines a la fois ("another operation is in progress"). Le
            # parallelisme ne gagnait rien : une seule connexion, donc les
            # requetes se serialisent cote serveur de toute facon.
            #
            # Chaque branche est isolee pour qu'une panne de l'une n'emporte
            # pas l'autre.
            budget = max(limit, self.MAX_RESULTS_PER_MODE)

            try:
                text_res_list = await self.text_chunks(query, filters, budget, 0)
            except Exception as e:
                logger.warning(f"⚠️ Text search failed in hybrid: {e}")
                text_res_list = []

            try:
                sem_res_list = await self.semantic_chunks(query, filters, budget, 0)
            except Exception as e:
                logger.warning(f"⚠️ Semantic search failed in hybrid: {e}")
                sem_res_list = []

            if not text_res_list and not sem_res_list:
                return []

            fused = self._rrf_fusion(
                text_res_list, sem_res_list,
                k=self.RRF_K,
                text_weight=self.TEXT_WEIGHT,
                semantic_weight=self.SEMANTIC_WEIGHT,
            )
            fused = self._normalize_scores(fused)
            paginated = fused[offset:offset + limit]

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"🔀 Hybrid search: {len(paginated)} chunks in {elapsed_ms}ms "
                f"(text={len(text_res_list)}, semantic={len(sem_res_list)})"
            )
            return paginated

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise SearchServiceError(f"Échec recherche hybride: {e}") from e

    # ============ SEARCH MODES — niveau document, derive des chunks ============
    # Ces trois methodes conservent la signature et le type de retour d'origine
    # (List[SearchResult]) : le front, les routes et les tests existants n'ont
    # rien a changer. Elles ne sont plus que des adaptateurs.

    async def text_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[SearchResult]:
        """Recherche textuelle, resultats regroupes par loi."""
        chunks = await self.text_chunks(query, filters, self._chunk_budget(limit), 0)
        return self._chunks_to_search_results(chunks)[offset:offset + limit]

    async def semantic_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[SearchResult]:
        """Recherche semantique, resultats regroupes par loi."""
        chunks = await self.semantic_chunks(query, filters, self._chunk_budget(limit), 0)
        return self._chunks_to_search_results(chunks)[offset:offset + limit]

    async def hybrid_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 15,
        offset: int = 0,
    ) -> List[SearchResult]:
        """Recherche hybride, resultats regroupes par loi."""
        chunks = await self.hybrid_chunks(query, filters, self._chunk_budget(limit), 0)
        return self._chunks_to_search_results(chunks)[offset:offset + limit]

    @staticmethod
    def _chunks_to_search_results(chunks: List[ChunkResult]) -> List[SearchResult]:
        """
        Regroupe des chunks en resultats niveau document.

        L'ordre des lois suit celui de leur MEILLEUR chunk (premiere apparition
        dans une liste deja triee par pertinence). Chaque loi porte jusqu'a
        trois articles correspondants : matched_articles, highlights et content
        etaient vides sur le chemin semantique, ou remplis d'un prefixe brut de
        400 caracteres sur le chemin textuel.
        """
        by_law: Dict[int, Dict[str, Any]] = {}

        for chunk in chunks:
            entry = by_law.get(chunk.law_id)
            if entry is None:
                by_law[chunk.law_id] = {"best": chunk, "chunks": [chunk]}
            else:
                entry["chunks"].append(chunk)

        results: List[SearchResult] = []
        for law_id, entry in by_law.items():
            best: ChunkResult = entry["best"]
            matched = [
                ArticleMatch(
                    article_id=c.article_id,
                    number=c.number or "",
                    title=c.article_title,
                    content_snippet=c.excerpt,
                    relevance_score=c.relevance_score,
                )
                for c in entry["chunks"][:3]
                # PREAMBULE et LEGAL_BASIS sont des pseudo-numeros donnes par le
                # decoupeur au texte hors articles : les exposer comme des
                # articles ferait afficher "Article LEGAL_BASIS" au front.
                if c.article_id is not None and c.number
                and c.number.upper() not in ("PREAMBULE", "LEGAL_BASIS")
            ]

            results.append(SearchResult(
                law_id=law_id,
                reference=best.reference,
                title=best.law_title,
                type=best.type,
                language=best.language,
                status=best.status,
                category_id=best.category_id,
                category_name=best.category_name,
                publication_date=best.publication_date,
                relevance_score=best.relevance_score,
                matched_articles=matched,
                highlights={"content": best.excerpt} if best.excerpt else {},
                content=best.content or None,
            ))

        return results

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
        # year_from / year_to n'etaient appliques nulle part, ni ici ni dans le
        # SQL plein texte : les filtres de periode etaient silencieusement sans
        # effet.
        if getattr(filters, "year_from", None):
            stmt = stmt.where(
                func.extract("year", Law.publication_date) >= filters.year_from
            )
        if getattr(filters, "year_to", None):
            stmt = stmt.where(
                func.extract("year", Law.publication_date) <= filters.year_to
            )

        return stmt

    def _rrf_fusion(
        self,
        text_results: List[ChunkResult],
        semantic_results: List[ChunkResult],
        k: int = 60,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[ChunkResult]:
        """
        Applique l'algorithme RRF (Reciprocal Rank Fusion) avec poids.

        La cle est celle du CHUNK (ChunkResult.fusion_key) et non le law_id :
        deux articles distincts d'une meme loi, l'un trouve par le texte et
        l'autre par le vecteur, etaient fusionnes en un seul et l'un des deux
        etait perdu. Le prefixe de la cle evite en plus qu'une ligne de repli
        niveau-loi entre en collision avec un article de meme identifiant.
        """
        scores: Dict[tuple, Dict] = {}

        for rank, result in enumerate(text_results, start=1):
            scores[result.fusion_key] = {
                "text_rank": rank,
                "semantic_rank": None,
                "result": result,
            }

        for rank, result in enumerate(semantic_results, start=1):
            key = result.fusion_key
            if key in scores:
                scores[key]["semantic_rank"] = rank
                # Le chunk textuel est conserve : il porte un extrait balise
                # par ts_headline, la ou le chemin semantique n'a qu'un debut
                # d'article. On ne recupere l'extrait semantique que si l'autre
                # est vide.
                if not scores[key]["result"].excerpt:
                    scores[key]["result"].excerpt = result.excerpt
            else:
                scores[key] = {
                    "text_rank": None,
                    "semantic_rank": rank,
                    "result": result,
                }

        for _key, data in scores.items():
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

    def _normalize_scores(self, results: List[ChunkResult]) -> List[ChunkResult]:
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

    # _extract_highlight_snippet a ete supprimee : elle n'etait appelee nulle
    # part. Les extraits viennent desormais de ts_headline, cote SQL.


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
