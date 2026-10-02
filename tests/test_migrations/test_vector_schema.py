"""
Tests du schema vectoriel.

La colonne est en vector(768), indexee par HNSW directement, sans le detour par
une expression halfvec qu'imposaient les 3072 dimensions d'avant (migration
c4d5e6f7a8b9). Ces tests verifient la dimension, l'index — un seul, le bon —,
et surtout le PLAN de l'enonce que produit l'ORM : qu'il passe par l'index, et
que le planificateur n'y supprime pas le tri final.
"""

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import asyncpg as pg_asyncpg

from app.models.law import Article, Law
from app.schemas.search import SearchFilters
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService

INDEX = "idx_articles_embedding_hnsw_vector"


@pytest.mark.asyncio
async def test_embedding_column_has_configured_dimension(db_session):
    result = await db_session.execute(text("""
        SELECT format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'articles'::regclass AND attname = 'embedding'
    """))
    declared = result.scalar()

    assert declared == f"vector({EmbeddingService.EMBEDDING_DIM})"


@pytest.mark.asyncio
async def test_exactly_one_index_on_the_embedding_column(db_session):
    """
    Un seul index sur la colonne, et le bon.

    Un ancien index halfvec oublie par la migration survit a la conversion de
    la colonne : il ne sert plus aucune requete mais coute a chaque ecriture,
    et un test qui chercherait seulement « un index HNSW » passerait contre
    lui. D'ou l'egalite stricte sur la liste.
    """
    # Les index dont la CLE porte sur la colonne : en colonne (indkey) ou dans
    # une expression (indexprs). Pas ceux qui ne la citent qu'en predicat,
    # comme idx_articles_embed_pending (`WHERE embedding IS NULL`), qui sert a
    # trouver les articles en attente et n'a rien d'un index vectoriel.
    # \m et \M : bornes de mot, pour ne pas prendre `embedding_model`.
    rows = (await db_session.execute(text(r"""
        SELECT c.relname AS indexname, pg_get_indexdef(i.indexrelid) AS indexdef
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attname = 'embedding'
        WHERE i.indrelid = 'articles'::regclass
          AND (a.attnum = ANY(i.indkey)
               OR pg_get_expr(i.indexprs, i.indrelid) ~ '\membedding\M')
    """))).all()

    assert [r.indexname for r in rows] == [INDEX], rows
    indexdef = rows[0].indexdef.lower()
    assert "using hnsw (embedding vector_cosine_ops)" in indexdef
    assert "halfvec" not in indexdef


def _statement(filters=None, limit=8):
    service = SearchService.__new__(SearchService)
    return service._build_semantic_statement(
        [0.001] * EmbeddingService.EMBEDDING_DIM, filters, limit, 0
    )


def _compiled_semantic_sql() -> str:
    return str(_statement().compile(dialect=postgresql.dialect())).replace("\n", " ")


def test_orm_emits_the_distance_operator():
    """
    Garde durable contre le retour a func.cosine_distance().

    Cette forme compile en un APPEL DE FONCTION `cosine_distance(a, b)`, que le
    planificateur ne peut rattacher a aucun index. Seule la forme operateur
    `<=>` est indexable — l'index aurait donc existe sans jamais servir.
    """
    sql = _compiled_semantic_sql()

    assert "<=>" in sql
    assert "cosine_distance(" not in sql


def test_semantic_query_is_two_stage():
    """
    L'etage 1 trie sur l'operateur indexable, l'etage 2 sur `distance + 0`.

    Le `+ 0` est ce que casserait une simplification d'apparence anodine : sur
    `ann.distance` nue, le planificateur sait la sous-requete deja ordonnee et
    supprime le tri externe — verifie sur plan reel plus bas.
    """
    sql = _compiled_semantic_sql()
    inner, _, outer = sql.partition(") AS ann")

    assert "ORDER BY articles.embedding <=>" in inner
    assert "ORDER BY ann.distance + 0" in outer
    # Plus aucun cast : l'index est pose sur la colonne elle-meme.
    assert "HALFVEC" not in sql.upper()
    # L'offset ne s'applique qu'apres le tri strict.
    assert "OFFSET" not in inner
    assert "OFFSET" in outer


# ==================== LE PLAN REEL ====================


@pytest.fixture
async def corpus_vectorise(db_session):
    """200 articles vectorises, repartis sur deux langues, statistiques a jour."""
    db_session.add_all([
        Law(id=900, reference="LOI-INDEX-FR", title="Loi de test index",
            content="Contenu.", type="loi", language="fr", status="published"),
        Law(id=901, reference="LOI-INDEX-EN", title="Index test law",
            content="Content.", type="loi", language="en", status="published"),
    ])
    await db_session.flush()

    rng = np.random.default_rng(3)
    for i in range(200):
        vec = rng.normal(size=EmbeddingService.EMBEDDING_DIM)
        db_session.add(Article(
            law_id=900 + i % 2,
            number=str(i + 1),
            content=f"Article de test numero {i + 1}.",
            order=i + 1,
            embedding=(vec / np.linalg.norm(vec)).tolist(),
        ))
    await db_session.commit()

    await db_session.execute(text("ANALYZE articles"))
    await db_session.execute(text("ANALYZE laws"))


async def _prepare_session(db_session, filters, limit):
    """
    Les reglages de session de la vraie recherche, plus deux penalites.

    Sur 200 lignes, le planificateur prefere legitimement lire les articles
    par un autre chemin — balayage, ou index sur law_id — puis TRIER par
    distance : c'est moins cher que l'index HNSW a cette echelle. A 20 000
    articles et sans penalite, il choisit l'index (mesure). Ce qui est
    verifie ici n'est pas ce choix de cout, mais que la FORME de l'enonce
    permet l'index. D'ou enable_seqscan et enable_sort a off : le seul chemin
    sans penalite est alors l'index qui rend les lignes deja ordonnees — il
    n'existe que si le tri interne lui est rattachable.

    Le tri EXTERNE, lui, n'a aucune alternative ordonnee : `distance + 0` ne
    correspond a aucun index. Penalise ou non, il doit rester dans le plan, et
    le planificateur, qui l'eliminerait plus volontiers encore sous cette
    penalite, ne le peut pas.
    """
    service = SearchService(db_session, use_cache=False)
    await service._apply_hnsw_settings(service._ann_candidates(limit), filters is not None)
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))
    await db_session.execute(text("SET LOCAL enable_sort = off"))


def _assert_indexed_and_sorted(plan: str) -> None:
    assert f"Index Scan using {INDEX}" in plan, plan
    assert "Seq Scan on articles" not in plan, plan
    # Le second etage existe encore : un noeud Sort sur `distance + 0`. Sans le
    # `+ 0`, ce noeud disparait et le plan va directement de l'index au Limit.
    sort_keys = [line for line in plan.splitlines() if "Sort Key:" in line]
    assert any("ann.distance +" in line for line in sort_keys), plan


FILTRES = [
    pytest.param(None, id="sans-filtre"),
    pytest.param(SearchFilters(language="fr"), id="filtre-relaxed-order"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", FILTRES)
async def test_plan_uses_the_index_and_keeps_the_final_sort(
    db_session, corpus_vectorise, filters
):
    """EXPLAIN de l'enonce que l'ORM produit, et non d'une requete recopiee."""
    limit = 8
    await _prepare_session(db_session, filters, limit)

    sql = str(_statement(filters, limit).compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
    ))
    conn = await db_session.connection()
    plan = "\n".join(row[0] for row in (await conn.exec_driver_sql("EXPLAIN " + sql)))

    _assert_indexed_and_sorted(plan)


def _litteral(valeur) -> str:
    if isinstance(valeur, list):
        return "'[" + ",".join(str(v) for v in valeur) + "]'"
    if isinstance(valeur, str):
        return "'" + valeur.replace("'", "''") + "'"
    return str(int(valeur))


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", FILTRES)
async def test_generic_plan_too(db_session, corpus_vectorise, filters):
    """
    Le meme controle sur le plan GENERIQUE, force plutot qu'attendu.

    asyncpg prepare ses requetes. Apres cinq executions, PostgreSQL compare le
    plan generique — etabli sans connaitre ni le vecteur ni la limite — au
    cout moyen des plans personnalises, et l'adopte s'il n'est pas plus cher.
    Mesure sur 20 000 articles : il est estime six a sept fois plus cher, les
    plans personnalises restent. Ce test verifie donc que l'enonce resterait
    indexable avec `$1` a la place du vecteur, pas que le planificateur
    choisirait l'index.
    """
    limit = 8
    await _prepare_session(db_session, filters, limit)

    compiled = _statement(filters, limit).compile(dialect=pg_asyncpg.dialect())
    arguments = ", ".join(_litteral(compiled.params[nom]) for nom in compiled.positiontup)

    conn = await db_session.connection()
    brute = (await conn.get_raw_connection()).driver_connection
    await brute.execute(f"PREPARE semantique AS {compiled.string}")
    try:
        await brute.execute("SET LOCAL plan_cache_mode = force_generic_plan")
        rows = await brute.fetch(f"EXPLAIN EXECUTE semantique({arguments})")
    finally:
        await brute.execute("DEALLOCATE semantique")
    plan = "\n".join(row[0] for row in rows)

    assert "$1" in plan, "le plan n'est pas generique"
    _assert_indexed_and_sorted(plan)
