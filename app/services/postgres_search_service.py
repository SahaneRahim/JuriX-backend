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
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.search import ChunkResult, SearchFilters
from app.services.prompts import CONTEXT_MAX_CHARS_PER_CHUNK
from app.services.search_vectors import (
    RANK_WEIGHTS,
    REINDEX_ARTICLES_SQL,
    REINDEX_LAWS_SQL,
    TITLE_ONLY_WEIGHTS,
)

logger = logging.getLogger(__name__)

# TTL du cache des résultats de recherche (5 minutes)
CACHE_TTL_SECONDS = 300

# Nombre maximal de resultats retournes par la branche LOI
MAX_FTS_RESULTS = 20

# Texte porte par une ligne NIVEAU LOI (branche titre, passe floue sur les
# titres) : un extrait borne, et non plus le texte integral. Les deux requetes
# selectionnaient `l.content` tel quel, jusqu'a 1,76 M de caracteres par ligne
# (mesure) : vingt lois pesaient de 1 a 8,6 Mo, tires de la base a chaque
# recherche, re-classes en Python caractere par caractere (0,2 a 3 s), ecrits
# dans query_cache, puis envoyes a un front qui en lisait 200.
#
# CONTEXT_MAX_CHARS_PER_CHUNK + 1, et non une valeur ronde : le chat ne lit
# jamais plus de CONTEXT_MAX_CHARS_PER_CHUNK caracteres d'un extrait
# (prompts.build_context_string). Avec ce caractere de plus,
# _truncate_on_boundary voit encore un texte trop long, et coupe au meme
# endroit, avec la meme marque « […] », que sur le texte entier : le contexte
# du chat ne change pas d'un octet. Borne a CONTEXT_MAX_CHARS_PER_CHUNK pile,
# l'extrait passerait tel quel, coupe en pleine phrase et sans marque.
LAW_CHUNK_MAX_CHARS = CONTEXT_MAX_CHARS_PER_CHUNK + 1

# Passe trigramme sur le TEXTE des articles (mode texte seul, quand le plein
# texte n'a rien trouve). Voir _trigram_articles_query pour les deux bornes.
TRIGRAM_FALLBACK_TIMEOUT_MS = 1000
TRIGRAM_LONGUEUR_MARGE = 4

# Budget de chunks pour la branche ARTICLE. Plus large que la branche loi :
# 20 articles couvrent bien moins de documents distincts que 20 lois, et la
# fusion comme le regroupement par loi ont besoin de matiere.
MAX_FTS_CHUNKS = 60

# Nombre maximal de chunks retenus PAR LOI. Cale sur ce que
# `_chunks_to_search_results` expose reellement (`entry["chunks"][:3]`) : au-dela,
# les lignes etaient recuperees, transportees, re-classees, puis jetees.
MAX_CHUNKS_PER_LAW = 3


# ==================== CACHE PostgreSQL ====================


# Version du schema de reponse mise en cache. A incrementer des que la forme de
# SearchResponse change : sans elle, des reponses serialisees sous l'ancienne
# forme (sans `chunks`) continueraient a etre servies pendant tout le TTL.
# Aussi quand l'espace des vecteurs change, ce qui change les resultats
# semantiques et hybrides. v5 : passage a Voyage (voyage-4-large, 1024).
# v6 : lignes niveau loi bornees a LAW_CHUNK_MAX_CHARS (le re-ranking les note
# autrement) et `results[].content` reduit a un apercu dans le cache.
CACHE_SCHEMA_VERSION = "v6"


def _make_cache_key(
    query: str, filters: Optional[SearchFilters], limit: int, offset: int, mode: str,
    semantic_query: Optional[str] = None,
) -> str:
    """
    Génère une clé de cache SHA-256 déterministe.

    Le mode en fait partie : sans lui, une recherche « text » et une recherche
    « hybrid » de la meme requete se servaient l'une l'autre pendant tout le TTL.
    Le texte vectorise aussi, quand il differe de la requete : deux questions
    aux memes mots-cles n'ont pas le meme volet semantique.
    """
    filters_dict = filters.model_dump() if filters else {}
    raw = (
        f"{CACHE_SCHEMA_VERSION}|{mode}|{query}|"
        f"{json.dumps(filters_dict, sort_keys=True)}|{limit}|{offset}"
    )
    if semantic_query is not None and mode != "text":
        raw += f"|semantique:{semantic_query}"
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
    repli_trigramme: bool = True,
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
        repli_trigramme: tenter la passe trigramme sur le texte des articles
            quand le plein texte ne rend rien. False en mode hybride : le volet
            semantique y rattrape deja les fautes de frappe, et cette passe
            coutait 5 a 13 s par faute (mesure : 15,9 s pour « licensiement »).

    Returns:
        Liste de ChunkResult triee par pertinence decroissante
    """
    assert query and isinstance(query, str), "query must be a non-empty string"
    assert limit > 0, "limit must be positive"

    results = await _fts_articles_query(db, query, filters, limit, offset)

    if not results:
        # Aucun article ne porte TOUS les mots : on retire d'abord les plus
        # FREQUENTS. ts_rank_cd ne connait pas la rarete d'un terme ; en OU,
        # le mot banal l'emportait. Mesure sur le corpus : « sanctions
        # harcelement sexuel » placait en tete les sanctions disciplinaires
        # des etudiants (« sanctions » cinq fois, aucun harcelement), devant
        # l'article 302 du Code penal. Sans « sanctions », les deux mots rares
        # designent exactement les bons articles.
        if offset == 0:
            for sous_requete in await _sous_requetes_par_rarete(db, query):
                results = await _fts_articles_query(db, sous_requete, filters, limit, 0)
                if results:
                    logger.debug(f"FTS ET a 0 resultat, termes rares retenus : {sous_requete}")
                    break

        # websearch_to_tsquery relie les mots par ET : "obligations dirigeants
        # societe" exige les TROIS dans le meme article, ce qu'aucun article ne
        # satisfait souvent — alors que 56 articles parlent de societes. Sans ce
        # second essai en OU, la recherche tombait directement au niveau LOI, et
        # le RAG recevait un document entier sans identite d'article : la
        # citation sortait donc sans numero d'article. Le OU COMPLETE desormais
        # les termes rares, a leur suite : il ne passe plus devant.
        any_terms = _as_any_terms(query)
        if any_terms != query and len(results) < limit:
            logger.debug(f"FTS ET a 0 resultat, essai en OU pour: {query[:40]}")
            complement = await _fts_articles_query(db, any_terms, filters, limit, offset)
            vus = {c.article_id for c in results}
            results = (results + [c for c in complement if c.article_id not in vus])[:limit]

    if not results and repli_trigramme:
        logger.debug(f"FTS returned 0, trying trigram fallback for: {query[:40]}")
        results = await _trigram_articles_query(db, query, filters, limit, offset)

    logger.info(f"📝 PG articles search: {len(results)} chunks for '{query[:40]}'")
    return results


def _as_any_terms(query: str) -> str:
    """
    Reecrit "a b c" en "a OR b OR c" pour websearch_to_tsquery.

    Le classement reste assure par ts_rank_cd, qui favorise naturellement les
    articles portant le plus de termes : passer en OU elargit le rappel sans
    mettre au meme niveau un article qui contient tout et un qui contient un
    seul mot.

    Les operateurs deja presents dans la saisie (guillemets, OR, -) sont
    respectes : on ne reecrit qu'une suite de mots simples.
    """
    cleaned = query.strip()
    if any(ch in cleaned for ch in '"-') or " OR " in cleaned.upper():
        return cleaned

    terms = [t for t in cleaned.split() if len(t) > 1]
    if len(terms) < 2:
        return cleaned
    return " OR ".join(terms)


async def _sous_requetes_par_rarete(db: AsyncSession, query: str) -> List[str]:
    """
    Requetes de repli, des plus exigeantes aux moins exigeantes, en retirant
    a chaque etape le terme le plus FREQUENT du corpus.

    « sanctions harcelement sexuel » donne « harcelement sexuel », puis
    « harcelement ». Un terme absent du corpus (faute de frappe) est retire
    d'emblee. Rien pour une saisie a operateurs ou d'un seul mot : _as_any_terms
    la laisse telle quelle, et on n'y touche pas davantage.

    Un seul aller-retour : le nombre d'articles de chaque terme, compte sur les
    seuls articles qui en portent au moins un (index GIN).
    """
    any_terms = _as_any_terms(query)
    if any_terms == query.strip():
        return []
    termes = [t for t in query.split() if len(t) > 1]
    params: Dict[str, Any] = {f"t{i}": t for i, t in enumerate(termes)}
    params["ou"] = any_terms
    comptes = ", ".join(
        f"count(*) FILTER (WHERE a.search_vector @@ plainto_tsquery('french', :t{i})"
        f" OR a.search_vector @@ plainto_tsquery('english', :t{i})) AS n{i}"
        for i in range(len(termes))
    )
    try:
        ligne = (await db.execute(text(
            f"SELECT {comptes} FROM articles a "
            "WHERE a.search_vector @@ websearch_to_tsquery('french', :ou) "
            "OR a.search_vector @@ websearch_to_tsquery('english', :ou)"
        ), params)).one()
    except Exception as e:
        logger.warning(f"⚠️ Frequence des termes indisponible : {e}")
        return []

    presents = sorted(
        (ligne[i], i) for i in range(len(termes)) if ligne[i] and ligne[i] > 0
    )
    depart = len(presents) if len(presents) < len(termes) else len(presents) - 1
    requetes = []
    for n in range(depart, 0, -1):
        retenus = sorted(i for _, i in presents[:n])
        requetes.append(" ".join(termes[i] for i in retenus))
    return requetes


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
        "per_law_cap": MAX_CHUNKS_PER_LAW,
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
                    '{RANK_WEIGHTS}',
                    a.search_vector,
                    websearch_to_tsquery('french', :query)
                ) + ts_rank_cd(
                    '{RANK_WEIGHTS}',
                    a.search_vector,
                    websearch_to_tsquery('english', :query)
                ) AS rank,
                -- Vrai si et seulement si un lexeme de POIDS A participe a la
                -- correspondance, c'est-a-dire si le TITRE de la loi matche.
                -- Ce n'est pas une heuristique : mesure sur le corpus, 5 titres
                -- sur 5 detectes, 3 correspondances de corps sur 3 ecartees.
                (
                    ts_rank_cd('{TITLE_ONLY_WEIGHTS}', a.search_vector,
                               websearch_to_tsquery('french', :query)) > 0
                    OR
                    ts_rank_cd('{TITLE_ONLY_WEIGHTS}', a.search_vector,
                               websearch_to_tsquery('english', :query)) > 0
                ) AS title_match,
                -- PLAFOND PAR LOI. Sans lui, une seule loi prend tout le budget :
                -- mesure sur le corpus, « avancement de grade » rendait 60 chunks
                -- issus de la MEME loi, donc UN SEUL document affiche. Et le
                -- probleme empire avec le titre de la loi en poids A, puisque
                -- tous les articles d'une loi dont le titre correspond marquent
                -- alors le meme score.
                ROW_NUMBER() OVER (
                    PARTITION BY a.law_id
                    ORDER BY ts_rank_cd(
                        '{RANK_WEIGHTS}', a.search_vector,
                        websearch_to_tsquery('french', :query)
                    ) + ts_rank_cd(
                        '{RANK_WEIGHTS}', a.search_vector,
                        websearch_to_tsquery('english', :query)
                    ) DESC
                ) AS law_rank
            FROM articles a
            JOIN laws l ON l.id = a.law_id
            LEFT JOIN categories c ON c.id = l.category_id
            WHERE (
                a.search_vector @@ websearch_to_tsquery('french', :query)
                OR a.search_vector @@ websearch_to_tsquery('english', :query)
            )
            {where_extra}
        ),
        capped AS (
            SELECT * FROM ranked
            WHERE law_rank <= :per_law_cap
            ORDER BY rank DESC
            LIMIT :limit OFFSET :offset
        )
        SELECT
            capped.*,
            ts_headline(
                (CASE WHEN capped.language = 'en' THEN 'english' ELSE 'french' END)::regconfig,
                capped.content,
                websearch_to_tsquery(
                    (CASE WHEN capped.language = 'en' THEN 'english' ELSE 'french' END)::regconfig,
                    :query
                ),
                'StartSel=<mark>,StopSel=</mark>,MaxFragments=2,MaxWords=40,MinWords=20'
            ) AS excerpt
        FROM capped
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

    DEUX BORNES, parce que cette passe prenait des secondes. L'index GIN
    rendait 74 a 81 % des articles comme candidats, et la reverification de
    chacun rejetait presque tout : 5,8 a 7,5 s en local, 4,9 a 12,8 s sur
    Supabase, pour 0 ou 1 resultat.

    1. Un article trop long ne peut PAS passer le seuil, et il est ecarte
       avant tout calcul de similarite. similarity = c / (|Tq| + |Ta| - c),
       ou c <= |Tq| compte les trigrammes communs : elle ne depasse donc pas
       |Tq| / |Ta|, et le seuil s exige |Ta| < |Tq| / s. Une requete d'un mot
       (|Tq| ~ 13) ne peut toucher qu'un article d'une quarantaine de
       trigrammes distincts, soit quelques dizaines de caracteres. La borne
       retenue, en octets, multiplie cette limite par TRIGRAM_LONGUEUR_MARGE
       (4) : un texte de prose a au moins un trigramme distinct pour quatre
       octets, et seul un texte tres repetitif (une ligne de pointilles) y
       echapperait. octet_length se lit dans l'en-tete, sans decompresser le
       texte. Le `OFFSET 0` empeche PostgreSQL de remettre l'index trigramme
       devant ce filtre. Mesure en local : 0,23 s au lieu de 6 a 7,5 s, memes
       resultats (« Paine de mort » : 1 article de 29 octets ; « licensiement »
       : aucun).
    2. Le balayage qui reste lit tout le tas des articles (82 Mo) : rapide en
       memoire, plus lent sur une base froide. Un delai d'execution
       (TRIGRAM_FALLBACK_TIMEOUT_MS) le plafonne : au-dela, la passe rend
       « rien », ce qu'elle rendait presque toujours. Le delai est pose dans
       un SAVEPOINT puis ANNULE avec lui : sans cela, il survivrait jusqu'a la
       fin de la transaction et couperait les requetes suivantes de
       l'appelant. Cette passe est rare (mode texte, plein texte vide) : ses
       quatre allers-retours de plus n'ont lieu qu'alors.
    """
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, MAX_FTS_CHUNKS),
        "offset": offset,
        "marge": TRIGRAM_LONGUEUR_MARGE,
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
        FROM (
            SELECT a0.id, a0.number, a0.title, a0.section, a0.page_number,
                   a0.content, a0.law_id
            FROM articles a0
            -- show_limit() : le seuil meme de l'operateur %.
            WHERE octet_length(a0.content)
                  <= :marge * array_length(show_trgm(:query), 1) / show_limit()
            OFFSET 0
        ) a
        JOIN laws l ON l.id = a.law_id
        LEFT JOIN categories c ON c.id = l.category_id
        WHERE a.content % :query
        {where_extra}
        ORDER BY rank DESC
        LIMIT :limit OFFSET :offset
    """)

    try:
        async with db.begin_nested() as point:
            await db.execute(
                text("SELECT set_config('statement_timeout', :delai, true)"),
                {"delai": f"{TRIGRAM_FALLBACK_TIMEOUT_MS}ms"},
            )
            rows = (await db.execute(sql, params)).fetchall()
            # ROLLBACK TO SAVEPOINT, et non RELEASE : il retire le delai.
            await point.rollback()
        return _rows_to_chunks(rows, source="trigram")
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

    Cette branche etait un REPLI, appele seulement quand la recherche d'articles
    ne rendait rien. C'etait la cause du symptome « je cherche un mot du titre et
    je vois des documents qui ne l'ont pas » : elle etait donc la seule a voir
    `laws.title`, et elle ne s'executait jamais. Elle tourne desormais a chaque
    recherche, et ses lignes portent `match_scope = "title"`.

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
        "law_chunk_max_chars": LAW_CHUNK_MAX_CHARS,
    }
    where_extra = _law_filter_clauses(filters, params)

    sql = text(f"""
        SELECT
            NULL::integer     AS article_id,
            NULL::varchar     AS number,
            NULL::varchar     AS article_title,
            NULL::varchar     AS section,
            NULL::integer     AS page_number,
            -- Extrait borne, pas le texte integral : voir LAW_CHUNK_MAX_CHARS.
            left(l.content, :law_chunk_max_chars) AS content,
            l.id              AS law_id,
            l.reference,
            l.title           AS law_title,
            l.type,
            l.language,
            l.status,
            l.category_id,
            l.publication_date,
            c.name            AS category_name,
            -- COALESCE : une ligne trouvee par le seul ILIKE a un rank NULL, et
            -- PostgreSQL place les NULL EN TETE sous ORDER BY ... DESC. Sans ce
            -- garde-fou, la correspondance la plus faible passait premiere.
            COALESCE(
                ts_rank_cd(
                    '{RANK_WEIGHTS}',
                    l.search_vector,
                    websearch_to_tsquery('french', :query)
                ) + ts_rank_cd(
                    '{RANK_WEIGHTS}',
                    l.search_vector,
                    websearch_to_tsquery('english', :query)
                ),
                0.0
            ) AS rank,
            -- Cette branche interroge le titre : une correspondance de poids A,
            -- ou a defaut un ILIKE sur le titre, est une correspondance de titre.
            (
                ts_rank_cd('{TITLE_ONLY_WEIGHTS}', l.search_vector,
                           websearch_to_tsquery('french', :query)) > 0
                OR ts_rank_cd('{TITLE_ONLY_WEIGHTS}', l.search_vector,
                              websearch_to_tsquery('english', :query)) > 0
                OR l.title ILIKE :ilike_query ESCAPE '\\'
            ) AS title_match
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
        return _rows_to_chunks(rows, source="title_fts")
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
                # `title_match` n'est produit que par les requetes qui savent le
                # calculer ; ailleurs le defaut "body" s'applique, ce qui est la
                # bonne reponse pour une correspondance de contenu.
                match_scope="title" if getattr(row, "title_match", False) else "body",
            )
        )

    return chunks


# Mots ignores par la passe floue sur les titres : les mots vides, et ceux que
# presque tous les titres du corpus contiennent. Ils ne designent aucun texte.
MOTS_SANS_POIDS_TITRE = frozenset({
    "les", "des", "aux", "une", "par", "pour", "sur", "dans", "avec", "sans",
    "que", "qui", "quoi", "est", "sont", "son", "ses", "leur", "leurs", "entre",
    "cameroun", "camerounais", "camerounaise", "republique",
    "loi", "lois", "decret", "decrets", "ordonnance", "arrete", "portant",
    "relatif", "relative", "the", "and", "law",
})


def mots_significatifs(query: str) -> List[str]:
    """
    Les mots de la requete qui doivent se retrouver dans un titre : sans
    accents, en minuscules, d'au moins trois caracteres, hors
    MOTS_SANS_POIDS_TITRE, sans doublon.
    """
    sans_accents = "".join(
        c for c in unicodedata.normalize("NFD", query.lower())
        if unicodedata.category(c) != "Mn"
    )
    mots: List[str] = []
    for mot in re.findall(r"[a-z0-9]+", sans_accents):
        if len(mot) >= 3 and mot not in MOTS_SANS_POIDS_TITRE and mot not in mots:
            mots.append(mot)
    return mots


async def search_titles_trgm_pg(
    db: AsyncSession,
    query: str,
    filters: Optional[SearchFilters],
    limit: int,
    threshold: float,
) -> List[ChunkResult]:
    """
    Recherche floue sur le TITRE et la reference d'une loi.

    C'est la tolerance aux fautes de frappe. Elle reprend exactement le SQL de
    `/search/suggest` — meme operateur `%>`, meme `immutable_unaccent` des deux
    cotes — de sorte que les index `idx_laws_title_unaccent_trgm` et
    `idx_laws_reference_unaccent_trgm`, jusqu'ici au service de la seule
    autocompletion, servent aussi la recherche.

    Mesure sur le corpus : « nominaton » ne rend RIEN en plein texte et rend
    exactement les 5 bons documents ici, sans un faux positif. Autres scores
    releves : « code miner » 0,727 · « cod minier » 0,769 · « avancment » 0,615 ·
    « fonciere » 0,333 (aucune correspondance, correctement).

    Le seuil est pose DANS l'enonce, avant le parcours : l'operateur `%>`
    filtre sur le reglage pg_trgm.word_similarity_threshold, si bien qu'un
    predicat `word_similarity(...) >= 0.5` ajoute ici serait du code mort —
    rien sous le defaut de 0,6 ne lui parviendrait. Il etait pose par un
    `SET LOCAL` envoye juste avant : deux allers-retours de plus, et une
    dependance a l'etat de la transaction. Ici, `(SELECT set_config(...)) IS NOT
    NULL` ne porte sur aucune colonne : PostgreSQL en fait un « One-Time
    Filter », evalue UNE fois avant de lire la moindre ligne (verifie par
    EXPLAIN ANALYZE : 849 titres pour « nominasion » a 0,5, 0 sans reglage).
    set_config(..., true) ne vaut que pour la transaction : rien ne passe a la
    requete suivante d'un pooler en mode transaction.

    MOT PAR MOT, ET NON LA REQUETE ENTIERE. Comparee d'un bloc, une requete de
    plusieurs mots passait le seuil sur ses mots les plus banals : dans
    « peine de mort au cameroun », « de », « au » et « cameroun » font plus de
    la moitie des trigrammes, et tout titre en « ... de ... au Cameroun »
    sortait a 0,516 — des decrets de pret, classes en tete comme
    correspondances de titre, devant les 11 articles du Code penal sur la
    peine de mort. Chaque mot significatif (`mots_significatifs`) doit
    desormais se retrouver, meme mal orthographie, dans le titre. Sans mot
    significatif, la passe ne rend rien. La reference, elle, garde la
    comparaison d'un bloc : c'est un identifiant, pas une phrase.
    """
    mots = mots_significatifs(query)
    if not mots:
        return []
    params: Dict[str, Any] = {
        "query": query,
        "limit": limit,
        "law_chunk_max_chars": LAW_CHUNK_MAX_CHARS,
        "seuil": _seuil_trigramme(threshold),
    }
    for i, mot in enumerate(mots):
        params[f"mot_{i}"] = mot
    titre = "immutable_unaccent(l.title)"
    chaque_mot = " AND ".join(f"{titre} %> :mot_{i}" for i in range(len(mots)))
    similarite_titre = " + ".join(
        f"word_similarity(:mot_{i}, {titre})" for i in range(len(mots))
    )
    where_extra = _law_filter_clauses(filters, params)

    sql = text(f"""
        SELECT
            NULL::integer     AS article_id,
            NULL::text        AS number,
            NULL::text        AS article_title,
            NULL::text        AS section,
            NULL::integer     AS page_number,
            -- Extrait borne, pas le texte integral : voir LAW_CHUNK_MAX_CHARS.
            left(l.content, :law_chunk_max_chars) AS content,
            l.id              AS law_id,
            l.reference,
            l.title           AS law_title,
            l.type,
            l.language,
            l.status,
            l.category_id,
            l.publication_date,
            c.name            AS category_name,
            TRUE              AS title_match,
            GREATEST(
                ({similarite_titre}) / {len(mots)},
                word_similarity(immutable_unaccent(:query), immutable_unaccent(l.reference))
            ) AS rank
        FROM laws l
        LEFT JOIN categories c ON c.id = l.category_id
        WHERE (SELECT set_config('pg_trgm.word_similarity_threshold', :seuil, true)) IS NOT NULL
        AND (
            ({chaque_mot})
            OR immutable_unaccent(l.reference) %> immutable_unaccent(:query)
        )
        {where_extra}
        ORDER BY rank DESC
        LIMIT :limit
    """)

    try:
        result = await db.execute(sql, params)
        return _rows_to_chunks(result.fetchall(), source="title_trigram")
    except Exception as e:
        logger.warning(f"⚠️ Title trigram query failed: {e}")
        return []


async def resolve_law_by_hint(
    db: AsyncSession,
    hint: str,
    filters: Optional[SearchFilters] = None,
    limit: int = 3,
) -> List[int]:
    """
    Lois designees par un indice de titre : « code minier », « constitution ».

    Trois passes, la premiere qui rend quelque chose gagne :
      1. ILIKE — l'utilisateur a ecrit le titre tel quel
      2. FTS de poids A — les mots du titre, dans le desordre, stemmes
      3. word_similarity — faute de frappe (« code miner » vaut 0,727)

    L'ordre est celui de la certitude decroissante. Une correspondance exacte ne
    doit jamais etre departagee par une correspondance floue.
    """
    hint = (hint or "").strip()
    if len(hint) < 3:
        return []

    params: Dict[str, Any] = {"hint": hint, "limit": limit}
    where_extra = _law_filter_clauses(filters, params, alias="l")

    passes = (
        "immutable_unaccent(l.title) ILIKE immutable_unaccent(:pattern) ESCAPE '\\'",
        # f-string : sans elle, {TITLE_ONLY_WEIGHTS} partait litteralement dans
        # le SQL et PostgreSQL rendait « invalid input syntax for type real ».
        f"ts_rank_cd('{TITLE_ONLY_WEIGHTS}', l.search_vector,"
        " websearch_to_tsquery('french', :hint)) > 0",
        "word_similarity(immutable_unaccent(:hint), immutable_unaccent(l.title))"
        " >= :threshold",
    )
    params["pattern"] = f"%{escape_like(hint)}%"
    params["threshold"] = 0.4

    for predicate in passes:
        sql = text(f"""
            SELECT l.id,
                   word_similarity(immutable_unaccent(:hint), immutable_unaccent(l.title)) AS score
            FROM laws l
            WHERE ({predicate})
            {where_extra}
            ORDER BY score DESC, length(l.title) ASC
            LIMIT :limit
        """)
        try:
            rows = (await db.execute(sql, params)).fetchall()
        except Exception as e:
            logger.warning(f"⚠️ resolve_law_by_hint failed: {e}")
            return []
        if rows:
            return [int(r.id) for r in rows]

    return []


async def find_article_in_laws(
    db: AsyncSession, law_ids: List[int], number: str
) -> Optional[int]:
    """
    Rend la premiere loi de `law_ids` qui contient un article de ce numero.

    L'ordre de `law_ids` est celui de la certitude rendue par
    `resolve_law_by_hint` : `array_position` le respecte au lieu de laisser
    PostgreSQL choisir.
    """
    if not law_ids or not number:
        return None
    sql = text("""
        SELECT a.law_id
        FROM articles a
        WHERE a.law_id = ANY(:law_ids)
          AND upper(btrim(a.number, ' .:-')) = :number
        ORDER BY array_position(:law_ids, a.law_id)
        LIMIT 1
    """)
    try:
        row = (await db.execute(sql, {"law_ids": law_ids, "number": number.upper()})).first()
    except Exception as e:
        logger.warning(f"⚠️ find_article_in_laws failed: {e}")
        return None
    return int(row.law_id) if row else None


async def record_search_event(
    db: AsyncSession,
    query: str,
    mode: str,
    results_count: int,
    duration_ms: int,
    cached: bool,
) -> None:
    """
    Journalise une recherche.

    Ecrit dans `search_events`, la table qui permet a
    `GET /api/v1/analytics/search` de rendre des mesures plutot que des
    constantes. `query_cache` ne pouvait pas servir a cela : il ne garde qu'un
    hachage, avec cinq minutes de duree de vie, et une requete servie depuis le
    cache n'y cree aucune ligne.

    NE LEVE JAMAIS. Une statistique manquante ne doit pas faire echouer la
    recherche de l'utilisateur — c'est exactement l'inversion de priorite qu'il
    faut eviter ici.

    Mais l'echec se VOIT : WARNING, et non plus DEBUG. Le journal part
    desormais apres la reponse, dans une tache (search_service) : une erreur
    avalee au niveau DEBUG n'y laissait aucune trace en production, et la
    statistique de recherche pouvait s'arreter sans que rien ne le dise.
    Premiere ligne seulement : une ligne DETAIL de PostgreSQL peut citer la
    question de l'utilisateur.
    """
    try:
        await db.execute(
            text("""
                INSERT INTO search_events (query, mode, results_count, duration_ms, cached)
                VALUES (:query, :mode, :results_count, :duration_ms, :cached)
            """),
            {
                "query": query[:500],
                "mode": mode,
                "results_count": results_count,
                "duration_ms": duration_ms,
                "cached": cached,
            },
        )
        await db.commit()
    except Exception as e:
        premiere_ligne = (str(e).strip().splitlines() or [""])[0][:200]
        logger.warning(
            f"⚠️ Journalisation de recherche impossible — {type(e).__name__}: {premiere_ligne}"
        )


def _seuil_trigramme(threshold: float) -> str:
    """Seuil de word_similarity borne a [0,1 ; 1], en texte pour set_config."""
    return str(min(max(float(threshold), 0.1), 1.0))


async def apply_trigram_threshold(db: AsyncSession, threshold: float) -> None:
    """
    Pose le seuil de `word_similarity` pour la transaction en cours.

    Le defaut de session est 0,6 : sans ce reglage, « nominasion » (0,571) est
    perdu alors que « nominaton » (0,700) passe — une tolerance aux fautes a
    moitie.

    La recherche ne l'appelle plus : search_titles_trgm_pg pose son seuil dans
    son propre enonce, sans aller-retour de plus. Gardee pour qui voudrait le
    poser sur toute une transaction. set_config et non SET : la valeur passe en
    parametre lie, et ne vaut que jusqu'a la fin de la transaction.
    """
    try:
        await db.execute(
            text("SELECT set_config('pg_trgm.word_similarity_threshold', :seuil, true)"),
            {"seuil": _seuil_trigramme(threshold)},
        )
    except Exception as err:  # pragma: no cover - depend du serveur
        logger.debug(f"pg_trgm.word_similarity_threshold indisponible: {err}")


async def update_law_search_vector(db: AsyncSession, law_id: int) -> None:
    """
    Met à jour manuellement le search_vector d'une loi et de ses articles.
    À appeler après indexation (remplace la recherche plein texte.index_law).

    Args:
        db: Session async SQLAlchemy
        law_id: ID de la loi à réindexer
    """
    await db.execute(
        text(f"{REINDEX_LAWS_SQL} WHERE id = :law_id"),
        {"law_id": law_id},
    )
    await db.execute(
        text(f"{REINDEX_ARTICLES_SQL} AND a.law_id = :law_id"),
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
