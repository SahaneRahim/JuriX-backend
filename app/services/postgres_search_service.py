"""
Service de recherche full-text natif PostgreSQL pour JuriX.

Remplace Meilisearch en utilisant :
- tsvector / tsquery pour la recherche full-text (FR + EN)
- pg_trgm pour la tolérance aux fautes d'orthographe
- Table query_cache pour le cache des résultats (remplace Redis)

Architecture :
- search_articles_pg()  : recherche article par article (index GIN sur search_vector)
- search_laws_pg()      : recherche par loi (fallback)
- get_from_pg_cache()   : lecture cache PostgreSQL
- store_in_pg_cache()   : écriture cache PostgreSQL
- cleanup_expired_cache(): nettoyage des entrées expirées

Performance cible :
- FTS avec GIN index : < 50ms
- Similarité trigramme pg_trgm : < 100ms
- Cache hit PostgreSQL : < 5ms

Author: JuriX Team
Version: 3.0.0 (PostgreSQL native)
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.search import ArticleMatch, SearchFilters, SearchResult

logger = logging.getLogger(__name__)

# TTL du cache des résultats de recherche (5 minutes)
CACHE_TTL_SECONDS = 300

# Nombre maximal de résultats retournés par mode FTS
MAX_FTS_RESULTS = 20


# ==================== CACHE PostgreSQL ====================


def _make_cache_key(query: str, filters: Optional[SearchFilters], limit: int, offset: int) -> str:
    """Génère une clé de cache SHA-256 déterministe."""
    filters_dict = filters.model_dump() if filters else {}
    raw = f"{query}|{json.dumps(filters_dict, sort_keys=True)}|{limit}|{offset}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


async def get_from_pg_cache(
    db: AsyncSession, cache_key: str
) -> Optional[Any]:
    """
    Lit une réponse depuis la table query_cache.

    Args:
        db: Session async SQLAlchemy
        cache_key: Clé SHA-256

    Returns:
        L'objet JSON désérialisé ou None si absent/expiré
    """
    result = await db.execute(
        text(
            "SELECT response_json FROM query_cache "
            "WHERE cache_key = :key AND expires_at > now()"
        ),
        {"key": cache_key},
    )
    row = result.fetchone()
    if row:
        logger.debug(f"🎯 PG Cache HIT: {cache_key[:16]}...")
        return json.loads(row[0])
    return None


async def store_in_pg_cache(
    db: AsyncSession,
    cache_key: str,
    response_data: Any,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> None:
    """
    Stocke une réponse dans query_cache avec TTL.

    Args:
        db: Session async SQLAlchemy
        cache_key: Clé SHA-256
        response_data: Données sérialisables en JSON
        ttl_seconds: Durée de vie en secondes (défaut: 5 minutes)
    """
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=ttl_seconds)
    await db.execute(
        text(
            "INSERT INTO query_cache (cache_key, response_json, expires_at) "
            "VALUES (:key, :data, :exp) "
            "ON CONFLICT (cache_key) DO UPDATE "
            "SET response_json = :data, expires_at = :exp"
        ),
        {
            "key": cache_key,
            "data": json.dumps(response_data),
            "exp": expires_at,
        },
    )
    await db.commit()
    logger.debug(f"💾 PG Cache stored: {cache_key[:16]}... (TTL={ttl_seconds}s)")


async def cleanup_expired_cache(db: AsyncSession) -> int:
    """
    Supprime les entrées expirées de query_cache et embedding_cache.

    Returns:
        Nombre total d'entrées supprimées
    """
    result = await db.execute(
        text("DELETE FROM query_cache WHERE expires_at <= now()")
    )
    count_q = getattr(result, "rowcount", 0)

    result2 = await db.execute(
        text("DELETE FROM embedding_cache WHERE expires_at <= now()")
    )
    count_e = getattr(result2, "rowcount", 0)

    await db.commit()
    total = (count_q or 0) + (count_e or 0)
    if total > 0:
        logger.info(f"🧹 Cleaned {total} expired cache entries (search={count_q}, embed={count_e})")
    return total


# ==================== RECHERCHE FTS ====================


async def search_articles_pg(
    db: AsyncSession,
    query: str,
    limit: int = 15,
    offset: int = 0,
) -> List[SearchResult]:
    """
    Recherche full-text dans les articles via PostgreSQL tsvector + pg_trgm.

    Stratégie en deux passes :
    1. websearch_to_tsquery sur search_vector (exact + ranking ts_rank_cd)
    2. Fallback trigramme pg_trgm si tsvector retourne 0 résultats

    Args:
        db: Session async SQLAlchemy
        query: Requête de l'utilisateur (ex: "responsabilité civile")
        limit: Nombre max de résultats
        offset: Décalage pour pagination

    Returns:
        Liste de SearchResult triée par pertinence décroissante
    """
    assert query and isinstance(query, str), "query must be a non-empty string"
    assert limit > 0, "limit must be positive"

    results = await _fts_articles_query(db, query, limit, offset)

    if not results:
        logger.debug(f"FTS returned 0, trying trigram fallback for: {query[:40]}")
        results = await _trigram_articles_query(db, query, limit, offset)

    logger.info(f"📝 PG articles search: {len(results)} results for '{query[:40]}'")
    return results


async def _fts_articles_query(
    db: AsyncSession,
    query: str,
    limit: int,
    offset: int,
) -> List[SearchResult]:
    """
    Recherche tsvector principale avec websearch_to_tsquery.
    Supporte: "expression exacte", OR, -, mots simples.
    """
    sql = text("""
        WITH ranked AS (
            SELECT
                a.id              AS article_id,
                a.number,
                a.title           AS article_title,
                a.section,
                a.content,
                a.law_id,
                l.reference,
                l.title           AS law_title,
                l.type,
                l.language,
                l.status,
                l.category_id,
                c.name            AS category_name,
                ts_rank_cd(
                    a.search_vector,
                    websearch_to_tsquery('french', :query)
                ) + ts_rank_cd(
                    a.search_vector,
                    websearch_to_tsquery('english', :query)
                ) AS rank
            FROM articles a
            JOIN laws l ON l.id = a.law_id
            LEFT JOIN categories c ON c.id = l.category_id
            WHERE
                a.search_vector @@ websearch_to_tsquery('french', :query)
                OR a.search_vector @@ websearch_to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit OFFSET :offset
        )
        SELECT DISTINCT ON (law_id) * FROM ranked ORDER BY law_id, rank DESC
    """)

    try:
        result = await db.execute(sql, {"query": query, "limit": min(limit, MAX_FTS_RESULTS), "offset": offset})
        rows = result.fetchall()
        return _rows_to_search_results(rows)
    except Exception as e:
        logger.warning(f"⚠️ FTS query failed: {e}")
        return []


async def _trigram_articles_query(
    db: AsyncSession,
    query: str,
    limit: int,
    offset: int,
) -> List[SearchResult]:
    """
    Fallback: recherche par similarité trigramme pg_trgm.
    Tolère les fautes d'orthographe.

    Point de performance : `similarity(a, b) > seuil` n'est PAS accéléré par un
    index gin_trgm_ops — seul l'opérateur `%` l'est. Le filtre utilise donc `%`
    et `similarity()` ne sert plus qu'au tri. Le `ILIKE '%q%'` précédent forçait
    de toute façon un balayage séquentiel : il est retiré, `%` le couvre.
    Le seuil est celui de pg_trgm.similarity_threshold (0.3 par defaut). Il se
    regle au niveau base — `ALTER DATABASE <db> SET pg_trgm.similarity_threshold
    = 0.2` — et non par requete : un `SET LOCAL` ici emettrait un second execute
    sur une session deja occupee ("another operation is in progress").
    """
    sql = text("""
        SELECT
            a.id              AS article_id,
            a.number,
            a.title           AS article_title,
            a.section,
            a.content,
            a.law_id,
            l.reference,
            l.title           AS law_title,
            l.type,
            l.language,
            l.status,
            l.category_id,
            c.name            AS category_name,
            similarity(a.content, :query) AS rank
        FROM articles a
        JOIN laws l ON l.id = a.law_id
        LEFT JOIN categories c ON c.id = l.category_id
        WHERE a.content % :query
        ORDER BY rank DESC
        LIMIT :limit OFFSET :offset
    """)

    try:
        result = await db.execute(
            sql,
            {
                "query": query,
                "limit": min(limit, MAX_FTS_RESULTS),
                "offset": offset,
            },
        )
        rows = result.fetchall()
        return _rows_to_search_results(rows)
    except Exception as e:
        logger.warning(f"⚠️ Trigram query failed: {e}")
        return []


async def search_laws_pg(
    db: AsyncSession,
    query: str,
    filters: Optional[SearchFilters] = None,
    limit: int = 15,
    offset: int = 0,
) -> List[SearchResult]:
    """
    Recherche full-text dans les lois (niveau document, pas article).
    Utilisé comme fallback si search_articles_pg retourne 0 résultats.

    Args:
        db: Session async SQLAlchemy
        query: Requête de l'utilisateur
        filters: Filtres optionnels (language, category, status, type)
        limit: Nombre max de résultats
        offset: Décalage pour pagination

    Returns:
        Liste de SearchResult
    """
    assert query and isinstance(query, str), "query must be a non-empty string"

    # Build filter clauses dynamically
    filter_clauses = []
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, MAX_FTS_RESULTS),
        "offset": offset,
    }

    if filters:
        if filters.language:
            filter_clauses.append("l.language = :language")
            params["language"] = filters.language
        if filters.category_ids:
            filter_clauses.append("l.category_id = ANY(:category_ids)")
            params["category_ids"] = filters.category_ids
        if filters.status:
            filter_clauses.append("l.status = :status")
            params["status"] = filters.status
        if filters.types:
            filter_clauses.append("l.type = ANY(:types)")
            params["types"] = filters.types

    where_extra = ("AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

    sql = text(f"""
        SELECT
            l.id              AS article_id,
            NULL              AS number,
            NULL              AS article_title,
            NULL              AS section,
            l.content         AS content,
            l.id              AS law_id,
            l.reference,
            l.title           AS law_title,
            l.type,
            l.language,
            l.status,
            l.category_id,
            c.name            AS category_name,
            ts_rank_cd(
                l.search_vector,
                websearch_to_tsquery('french', :query)
            ) + ts_rank_cd(
                l.search_vector,
                websearch_to_tsquery('english', :query)
            ) AS rank
        FROM laws l
        LEFT JOIN categories c ON c.id = l.category_id
        WHERE (
            l.search_vector @@ websearch_to_tsquery('french', :query)
            OR l.search_vector @@ websearch_to_tsquery('english', :query)
            OR l.title ILIKE :ilike_query
        )
        {where_extra}
        ORDER BY rank DESC
        LIMIT :limit OFFSET :offset
    """)
    params["ilike_query"] = f"%{query}%"

    try:
        result = await db.execute(sql, params)
        rows = result.fetchall()
        logger.info(f"📚 PG laws search: {len(rows)} results for '{query[:40]}'")
        return _rows_to_search_results(rows)
    except Exception as e:
        logger.warning(f"⚠️ Laws FTS query failed: {e}")
        return []


# ==================== HELPERS ====================


def _rows_to_search_results(rows: Sequence[Any]) -> List[SearchResult]:
    """
    Convertit les lignes de résultats SQL en objets SearchResult.

    Calcule le relevance_score depuis le rank SQL (normalisation min-max).
    Génère un snippet de contenu pour les highlights.
    """
    if not rows:
        return []

    # Normalize ranks to [0, 1]
    ranks = [float(row.rank) for row in rows if row.rank is not None]
    max_rank = max(ranks) if ranks else 1.0
    max_rank = max(max_rank, 0.001)  # Avoid division by zero

    results = []
    seen_law_ids: set = set()

    for row in rows:
        law_id = row.law_id
        if law_id in seen_law_ids:
            continue
        seen_law_ids.add(law_id)

        rank = float(row.rank) if row.rank else 0.0
        relevance_score = min(rank / max_rank, 1.0)

        content = row.content or ""
        snippet = content[:400] if content else ""

        matched_articles = []
        if row.number:  # Article-level result
            matched_articles = [
                ArticleMatch(
                    article_id=int(row.article_id),
                    number=str(row.number),
                    title=row.article_title,
                    content_snippet=snippet,
                    relevance_score=relevance_score,
                )
            ]

        results.append(
            SearchResult(
                law_id=int(law_id),
                reference=str(row.reference or ""),
                title=str(row.law_title or ""),
                type=str(row.type or "loi"),
                language=row.language,
                status=str(row.status or "published"),
                category_id=row.category_id,
                category_name=row.category_name,
                publication_date=None,
                relevance_score=relevance_score,
                matched_articles=matched_articles,
                highlights={"content": snippet},
                content=content,
            )
        )

    return results


async def update_law_search_vector(db: AsyncSession, law_id: int) -> None:
    """
    Met à jour manuellement le search_vector d'une loi et de ses articles.
    À appeler après indexation (remplace Meilisearch.index_law).

    Args:
        db: Session async SQLAlchemy
        law_id: ID de la loi à réindexer
    """
    await db.execute(
        text("""
            UPDATE laws
            SET search_vector =
                to_tsvector('french', coalesce(title, '') || ' ' || coalesce(content, ''))
                || to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
            WHERE id = :law_id
        """),
        {"law_id": law_id},
    )
    await db.execute(
        text("""
            UPDATE articles
            SET search_vector =
                to_tsvector('french', coalesce(content, ''))
                || to_tsvector('english', coalesce(content, ''))
                || to_tsvector('simple', coalesce(number, ''))
            WHERE law_id = :law_id
        """),
        {"law_id": law_id},
    )
    await db.commit()
    logger.info(f"✅ FTS search_vector updated for law {law_id}")


async def remove_law_search_index(db: AsyncSession, law_id: int) -> None:
    """
    'Désindexe' une loi en vidant ses search_vector.
    À appeler lors d'une suppression (remplace Meilisearch.remove_law).
    Les articles sont supprimés en cascade par le modèle (ondelete=CASCADE).

    Args:
        db: Session async SQLAlchemy
        law_id: ID de la loi à désindexer
    """
    await db.execute(
        text("UPDATE laws SET search_vector = null WHERE id = :law_id"),
        {"law_id": law_id},
    )
    await db.commit()
    logger.info(f"🗑️ FTS search_vector cleared for law {law_id}")
