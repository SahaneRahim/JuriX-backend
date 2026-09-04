"""
Tests du schema vectoriel.

La colonne faisait 3072 dimensions, au-dessus du plafond de 2000 de pgvector
pour HNSW et IVFFlat : aucun index n'etait possible et chaque recherche
semantique balayait la table. Ces tests verifient la dimension, l'existence de
l'index, et surtout que la requete produite par l'ORM est de la forme que le
planificateur sait y rattacher.
"""

import numpy as np
import pytest
from sqlalchemy import literal, select, text
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

from app.models.law import Article
from app.services.embedding_service import EmbeddingService


@pytest.mark.asyncio
async def test_embedding_column_has_configured_dimension(db_session):
    result = await db_session.execute(text("""
        SELECT format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'articles'::regclass AND attname = 'embedding'
    """))
    declared = result.scalar()

    assert declared == f"vector({EmbeddingService.EMBEDDING_DIM})"
    # Et surtout PAS halfvec : le stockage reste en fp32, seule l'indexation
    # passe en fp16. C'est tout l'interet du montage.
    assert not declared.startswith("halfvec")


@pytest.mark.asyncio
async def test_hnsw_index_exists(db_session):
    result = await db_session.execute(text("""
        SELECT indexdef FROM pg_indexes
        WHERE tablename = 'articles'
          AND indexname = 'idx_articles_embedding_hnsw_halfvec'
    """))

    indexdef = result.scalar()
    assert indexdef is not None, "l'index HNSW halfvec est absent"
    assert "hnsw" in indexdef.lower()
    assert "halfvec_cosine_ops" in indexdef.lower()
    # Le TYPMOD, pas seulement le type : `embedding::halfvec` sans dimension
    # est un noeud d'expression different, qui n'apparierait jamais la requete.
    assert f"halfvec({EmbeddingService.EMBEDDING_DIM})" in indexdef.lower()


@pytest.mark.asyncio
async def test_old_vector_index_is_gone(db_session):
    """L'ancien index sur `vector` ne doit pas survivre a la migration."""
    result = await db_session.execute(text("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'articles' AND indexname = 'idx_articles_embedding_hnsw'
    """))

    assert result.scalar() is None


def _compiled_semantic_sql() -> str:
    from app.services.search_service import SearchService

    service = SearchService.__new__(SearchService)
    stmt = service._build_semantic_statement(
        [0.0] * EmbeddingService.EMBEDDING_DIM, None, 8, 0
    )
    return str(stmt.compile(dialect=postgresql.dialect())).replace("\n", " ")


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
    L'etage ANN passe par le cast halfvec, le classement final par la distance
    exacte.

    C'est ce qui casse si quelqu'un "simplifie" la requete en un seul etage :
    on perdrait soit l'index (tri sur la distance fp32, non indexee), soit la
    precision du classement (tri sur le fp16).
    """
    sql = _compiled_semantic_sql()
    dim = EmbeddingService.EMBEDDING_DIM

    inner, _, outer = sql.partition(") AS ann")

    # Etage 1 : le cast avec son typmod, dans le ORDER BY interne
    assert f"CAST(articles.embedding AS HALFVEC({dim}))" in inner
    assert "ORDER BY CAST(articles.embedding AS HALFVEC" in inner

    # Etage 2 : classement externe sur la colonne de distance exacte, sans cast
    assert "ORDER BY ann.distance" in outer
    assert "HALFVEC" not in outer


@pytest.mark.asyncio
async def test_hnsw_index_is_usable_for_the_query_shape(db_session):
    """
    Le plan doit nommer l'index.

    enable_seqscan=off est necessaire : sur quelques centaines de lignes le
    planificateur prefere legitimement un balayage sequentiel. Ce qui est teste
    ici, c'est que la FORME de la requete est indexable — precisement ce que la
    forme fonction interdisait.
    """
    from app.models.law import Law

    law = Law(
        reference="LOI-INDEX-TEST",
        title="Loi de test index",
        content="Contenu de test.",
        type="loi",
        language="fr",
        status="published",
    )
    db_session.add(law)
    await db_session.flush()

    rng = np.random.default_rng(3)
    for i in range(200):
        vec = rng.normal(size=EmbeddingService.EMBEDDING_DIM)
        vec = vec / np.linalg.norm(vec)
        db_session.add(Article(
            law_id=law.id,
            number=str(i + 1),
            content=f"Article de test numero {i + 1}.",
            order=i + 1,
            embedding=vec.tolist(),
        ))
    await db_session.commit()

    await db_session.execute(text("ANALYZE articles"))
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))

    dim = EmbeddingService.EMBEDDING_DIM
    probe = "[" + ",".join("0.001" for _ in range(dim)) + "]"
    result = await db_session.execute(text(
        "EXPLAIN SELECT id FROM articles WHERE embedding IS NOT NULL "
        f"ORDER BY (embedding::halfvec({dim})) <=> '{probe}'::halfvec({dim}) LIMIT 5"
    ))
    plan = "\n".join(row[0] for row in result.fetchall())

    assert "idx_articles_embedding_hnsw_halfvec" in plan, plan
