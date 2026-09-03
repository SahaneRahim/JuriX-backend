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

    assert result.scalar() == f"vector({EmbeddingService.EMBEDDING_DIM})"


@pytest.mark.asyncio
async def test_hnsw_index_exists(db_session):
    result = await db_session.execute(text("""
        SELECT indexdef FROM pg_indexes
        WHERE tablename = 'articles' AND indexname = 'idx_articles_embedding_hnsw'
    """))

    indexdef = result.scalar()
    assert indexdef is not None, "l'index HNSW est absent"
    assert "hnsw" in indexdef.lower()
    assert "vector_cosine_ops" in indexdef.lower()


def test_orm_emits_the_distance_operator():
    """
    Garde durable contre le retour a func.cosine_distance().

    Cette forme compile en un APPEL DE FONCTION `cosine_distance(a, b)`, que le
    planificateur ne peut rattacher a aucun index. Seule la forme operateur
    `<=>` est indexable — l'index aurait donc existe sans jamais servir.
    """
    query_vector = [0.0] * EmbeddingService.EMBEDDING_DIM
    distance = Article.embedding.cosine_distance(
        literal(query_vector, Vector(EmbeddingService.EMBEDDING_DIM))
    )
    stmt = select(Article.id).order_by(distance).limit(5)

    sql = str(stmt.compile(dialect=postgresql.dialect()))

    assert "<=>" in sql
    assert "cosine_distance(" not in sql


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

    probe = "[" + ",".join("0.001" for _ in range(EmbeddingService.EMBEDDING_DIM)) + "]"
    result = await db_session.execute(text(
        "EXPLAIN SELECT id FROM articles WHERE embedding IS NOT NULL "
        f"ORDER BY embedding <=> '{probe}'::vector LIMIT 5"
    ))
    plan = "\n".join(row[0] for row in result.fetchall())

    assert "idx_articles_embedding_hnsw" in plan, plan
