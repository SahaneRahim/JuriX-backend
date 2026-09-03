"""
Service de recherche full-text natif PostgreSQL pour JuriX.

Recherche plein texte native PostgreSQL :
- tsvector / tsquery pour la recherche full-text (FR + EN)
- pg_trgm pour la tolérance aux fautes d'orthographe
- Table query_cache pour le cache des résultats (cache en base)

Architecture :
- search_articles_pg()  : recherche article par article, renvoie des ChunkResult
- search_laws_pg()      : recherche par loi (fallback), article_id = None
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

from app.schemas.search import ChunkResult, SearchFilters

logger = logging.getLogger(__name__)

# TTL du cache des résultats de recherche (5 minutes)
CACHE_TTL_SECONDS = 300

# Nombre maximal de resultats retournes par la branche LOI
MAX_FTS_RESULTS = 20

# Budget de chunks pour la branche ARTICLE. Plus large que la branche loi :
# 20 articles couvrent bien moins de documents distincts que 20 lois, et la
# fusion comme le regroupement par loi ont besoin de matiere.
MAX_FTS_CHUNKS = 60


# ==================== CACHE PostgreSQL ====================


# Version du schema de reponse mise en cache. A incrementer des que la forme de
# SearchResponse change : sans elle, des reponses serialisees sous l'ancienne
# forme (sans `chunks`) continueraient a etre servies pendant tout le TTL.
CACHE_SCHEMA_VERSION = "v2"


def _make_cache_key(query: str, filters: Optional[SearchFilters], limit: int, offset: int) -> str:
    """Génère une clé de cache SHA-256 déterministe."""
    filters_dict = filters.model_dump() if filters else {}
    raw = (
        f"{CACHE_SCHEMA_VERSION}|{query}|"
        f"{json.dumps(filters_dict, sort_keys=True)}|{limit}|{offset}"
    )
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
    filters: Optional[SearchFilters] = None,
    limit: int = 15,
    offset: int = 0,
) -> List[ChunkResult]:
    """
    Recherche full-text dans les articles via PostgreSQL tsvector + pg_trgm.

    Renvoie des CHUNKS : un article par ligne, dans l'ordre de pertinence.
    Auparavant un DISTINCT ON (law_id) ecrasait les articles a un par loi et
    reordonnait par law_id — deux articles pertinents d'un meme code etaient
    donc reduits a un seul, et l'ordre transmis a la fusion RRF n'etait plus
    celui de la pertinence mais celui des identifiants.

    Strategie en deux passes :
    1. websearch_to_tsquery sur search_vector (exact + ranking ts_rank_cd)
    2. Fallback trigramme pg_trgm si tsvector retourne 0 resultats

    Args:
        db: Session async SQLAlchemy
        query: Requete de l'utilisateur (ex: "responsabilite civile")
        filters: Filtres optionnels. Ils sont appliques DANS la requete, avant
            le LIMIT : ce parametre n'existait pas, si bien qu'un
            status="published" demande par l'appelant etait ignore des que la
            branche article renvoyait quelque chose.
        limit: Nombre max de chunks
        offset: Decalage pour pagination

    Returns:
        Liste de ChunkResult triee par pertinence decroissante
    """
    assert query and isinstance(query, str), "query must be a non-empty string"
    assert limit > 0, "limit must be positive"

    results = await _fts_articles_query(db, query, filters, limit, offset)

    if not results:
        logger.debug(f"FTS returned 0, trying trigram fallback for: {query[:40]}")
        results = await _trigram_articles_query(db, query, filters, limit, offset)

    logger.info(f"📝 PG articles search: {len(results)} chunks for '{query[:40]}'")
    return results


async def _fts_articles_query(
    db: AsyncSession,
    query: str,
    filters: Optional[SearchFilters],
    limit: int,
    offset: int,
) -> List[ChunkResult]:
    """
    Recherche tsvector principale avec websearch_to_tsquery.
    Supporte: "expression exacte", OR, -, mots simples.

    Trois choix a expliciter :

    - Les filtres entrent DANS la CTE, avant le LIMIT. Places apres, le budget
      de lignes serait depense sur des lignes destinees a etre jetees.
    - ts_headline est calcule a l'EXTERIEUR de la CTE. Il reparse le texte
      integral de l'article a chaque ligne : c'est de loin la partie chere. En
      le laissant dehors, il ne s'execute que sur les lignes deja retenues.
    - La CTE est MATERIALIZED pour empecher PostgreSQL 16 de l'inliner, ce qui
      repousserait ts_headline sur chaque ligne candidate.

    La configuration de dictionnaire suit la langue de la loi : passer un texte
    anglais au stemmer francais degrade l'extrait.
    """
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, MAX_FTS_CHUNKS),
        "offset": offset,
    }
    where_extra = _law_filter_clauses(filters, params)

    sql = text(f"""
        WITH ranked AS MATERIALIZED (
            SELECT
                a.id              AS article_id,
                a.number,
                a.title           AS article_title,
                a.section,
                a.page_number,
                a.content,
                a.law_id,
                l.reference,
                l.title           AS law_title,
                l.type,
                l.language,
                l.status,
                l.category_id,
                l.publication_date,
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
            WHERE (
                a.search_vector @@ websearch_to_tsquery('french', :query)
                OR a.search_vector @@ websearch_to_tsquery('english', :query)
            )
            {where_extra}
            ORDER BY rank DESC
            LIMIT :limit OFFSET :offset
        )
        SELECT
            ranked.*,
            ts_headline(
                (CASE WHEN ranked.language = 'en' THEN 'english' ELSE 'french' END)::regconfig,
                ranked.content,
                websearch_to_tsquery(
                    (CASE WHEN ranked.language = 'en' THEN 'english' ELSE 'french' END)::regconfig,
                    :query
                ),
                'StartSel=<mark>,StopSel=</mark>,MaxFragments=2,MaxWords=40,MinWords=20'
            ) AS excerpt
        FROM ranked
        ORDER BY rank DESC
    """)

    try:
        result = await db.execute(sql, params)
        return _rows_to_chunks(result.fetchall(), source="fts")
    except Exception as e:
        logger.warning(f"⚠️ FTS query failed: {e}")
        return []


async def _trigram_articles_query(
    db: AsyncSession,
    query: str,
    filters: Optional[SearchFilters],
    limit: int,
    offset: int,
) -> List[ChunkResult]:
    """
    Fallback: recherche par similarite trigramme pg_trgm.
    Tolere les fautes d'orthographe.

    Point de performance : `similarity(a, b) > seuil` n'est PAS accelere par un
    index gin_trgm_ops — seul l'operateur `%` l'est. Le filtre utilise donc `%`
    et `similarity()` ne sert plus qu'au tri. Le seuil est celui de
    pg_trgm.similarity_threshold (0.3 par defaut), reglable au niveau base et
    non par requete : un `SET LOCAL` ici emettrait un second execute sur une
    session deja occupee ("another operation is in progress").

    Pas de ts_headline sur ce chemin : la correspondance est floue, il n'y a pas
    de lexeme a mettre en evidence. L'extrait est le debut de l'article.
    """
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, MAX_FTS_CHUNKS),
        "offset": offset,
    }
    where_extra = _law_filter_clauses(filters, params)

    sql = text(f"""
        SELECT
            a.id              AS article_id,
            a.number,
            a.title           AS article_title,
            a.section,
            a.page_number,
            a.content,
            a.law_id,
            l.reference,
            l.title           AS law_title,
            l.type,
            l.language,
            l.status,
            l.category_id,
            l.publication_date,
            c.name            AS category_name,
            similarity(a.content, :query) AS rank
        FROM articles a
        JOIN laws l ON l.id = a.law_id
        LEFT JOIN categories c ON c.id = l.category_id
        WHERE a.content % :query
        {where_extra}
        ORDER BY rank DESC
        LIMIT :limit OFFSET :offset
    """)

    try:
        result = await db.execute(sql, params)
        return _rows_to_chunks(result.fetchall(), source="trigram")
    except Exception as e:
        logger.warning(f"⚠️ Trigram query failed: {e}")
        return []


async def search_laws_pg(
    db: AsyncSession,
    query: str,
    filters: Optional[SearchFilters] = None,
    limit: int = 15,
    offset: int = 0,
) -> List[ChunkResult]:
    """
    Recherche full-text dans les lois (niveau document, pas article).
    Utilise comme repli si search_articles_pg retourne 0 resultats.

    Les lignes produites portent `article_id = NULL` : elles ne designent aucun
    article. La colonne valait auparavant `l.id`, ce qui faisait entrer en
    collision une loi et l'article portant le meme identifiant des lors que la
    fusion se fait sur l'article.
    """
    assert query and isinstance(query, str), "query must be a non-empty string"

    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, MAX_FTS_RESULTS),
        "offset": offset,
    }
    where_extra = _law_filter_clauses(filters, params)

    sql = text(f"""
        SELECT
            NULL::integer     AS article_id,
            NULL::varchar     AS number,
            NULL::varchar     AS article_title,
            NULL::varchar     AS section,
            NULL::integer     AS page_number,
            l.content         AS content,
            l.id              AS law_id,
            l.reference,
            l.title           AS law_title,
            l.type,
            l.language,
            l.status,
            l.category_id,
            l.publication_date,
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
            OR l.title ILIKE :ilike_query ESCAPE '\\'
        )
        {where_extra}
        ORDER BY rank DESC
        LIMIT :limit OFFSET :offset
    """)
    # Echappement des metacaracteres : sans lui, une requete contenant % ou _
    # elargit le motif au lieu d'etre cherchee litteralement.
    params["ilike_query"] = f"%{escape_like(query)}%"

    try:
        result = await db.execute(sql, params)
        rows = result.fetchall()
        logger.info(f"📚 PG laws search: {len(rows)} results for '{query[:40]}'")
        return _rows_to_chunks(rows, source="law_fts")
    except Exception as e:
        logger.warning(f"⚠️ Laws FTS query failed: {e}")
        return []


# ==================== HELPERS ====================


def escape_like(value: str) -> str:
    """
    Echappe les metacaracteres LIKE/ILIKE (\\, %, _).

    Un `%` dans la saisie de l'utilisateur elargit sinon le motif a tout, et un
    `_` en fait un joker : ce n'est pas une injection, mais la requete ne
    cherche plus ce qui a ete demande.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _law_filter_clauses(
    filters: Optional[SearchFilters],
    params: Dict[str, Any],
    alias: str = "l",
) -> str:
    """
    Construit le fragment WHERE des filtres et alimente `params`.

    Seule la FORME du SQL est construite ici, a partir de litteraux ; toutes les
    valeurs restent liees. Partage entre les trois requetes : la branche article
    n'appliquait aucun filtre, et year_from/year_to n'etaient appliques nulle
    part.
    """
    if not filters:
        return ""

    clauses: List[str] = []

    if filters.language:
        clauses.append(f"{alias}.language = :language")
        params["language"] = filters.language
    if filters.category_ids:
        clauses.append(f"{alias}.category_id = ANY(:category_ids)")
        params["category_ids"] = filters.category_ids
    if filters.status:
        clauses.append(f"{alias}.status = :status")
        params["status"] = filters.status
    if filters.types:
        clauses.append(f"{alias}.type = ANY(:types)")
        params["types"] = filters.types
    if getattr(filters, "year_from", None):
        clauses.append(f"EXTRACT(YEAR FROM {alias}.publication_date) >= :year_from")
        params["year_from"] = filters.year_from
    if getattr(filters, "year_to", None):
        clauses.append(f"EXTRACT(YEAR FROM {alias}.publication_date) <= :year_to")
        params["year_to"] = filters.year_to

    return ("AND " + " AND ".join(clauses)) if clauses else ""


def _rows_to_chunks(rows: Sequence[Any], source: str) -> List[ChunkResult]:
    """
    Convertit les lignes SQL en ChunkResult, un par article.

    Le score est normalise min-max sur le lot. Le dedoublonnage par law_id qui
    se trouvait ici a ete retire : c'etait le second ecrasement des articles,
    apres le DISTINCT ON de la requete.
    """
    if not rows:
        return []

    ranks = [float(row.rank) for row in rows if row.rank is not None]
    max_rank = max(ranks) if ranks else 1.0
    max_rank = max(max_rank, 0.001)  # evite la division par zero

    chunks: List[ChunkResult] = []

    for row in rows:
        rank = float(row.rank) if row.rank else 0.0
        relevance_score = min(rank / max_rank, 1.0)

        content = row.content or ""
        excerpt = getattr(row, "excerpt", None) or content[:400]

        chunks.append(
            ChunkResult(
                article_id=int(row.article_id) if row.article_id is not None else None,
                law_id=int(row.law_id),
                number=str(row.number) if row.number else None,
                article_title=row.article_title,
                section=row.section,
                page_number=row.page_number,
                content=content,
                excerpt=excerpt,
                reference=str(row.reference or ""),
                law_title=str(row.law_title or ""),
                type=str(row.type or "loi"),
                language=row.language,
                status=str(row.status or "published"),
                category_id=row.category_id,
                category_name=row.category_name,
                publication_date=getattr(row, "publication_date", None),
                relevance_score=relevance_score,
                source=source,
            )
        )

    return chunks


async def update_law_search_vector(db: AsyncSession, law_id: int) -> None:
    """
    Met à jour manuellement le search_vector d'une loi et de ses articles.
    À appeler après indexation (remplace la recherche plein texte.index_law).

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
    À appeler lors d'une suppression (remplace la recherche plein texte.remove_law).
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
