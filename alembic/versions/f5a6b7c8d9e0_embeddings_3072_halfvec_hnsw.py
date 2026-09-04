"""Embeddings 3072 dimensions, index HNSW sur une expression halfvec

La migration precedente (e4f5a6b7c8d9) est passee a 1536 dimensions en partant
du principe que 3072 etait inindexable. C'est vrai du type `vector`, dont le
plafond d'indexation HNSW et IVFFlat est de 2000 dimensions. C'est FAUX du type
`halfvec` (pgvector >= 0.7), qui monte a 4000.

On garde donc la pleine precision du modele :
  - la colonne reste en `vector(3072)`, fp32, valeurs exactes ;
  - l'index HNSW est pose sur l'EXPRESSION `embedding::halfvec(3072)`, donc en
    fp16, ce qui divise sa taille par deux et le rend possible ;
  - la requete parcourt l'index pour selectionner des candidats, puis les
    reclasse a la distance fp32 exacte (voir SearchService._build_semantic_
    statement). Le fp16 ne sert qu'au tri grossier.

`USING NULL` est inevitable. `vector(1536)::vector(3072)` echoue, et completer
par des zeros serait pire : les 1536 premieres composantes d'une sortie 3072 ne
sont le vecteur 1536 qu'a un facteur d'echelle pres (le 1536 est ce prefixe
RENORMALISE), et les 1536 suivantes ne sont pas nulles. Un vecteur complete
tomberait au mauvais endroit sur la sphere unite et toutes les distances
seraient fausses en silence. Il faut re-embedder.

Les deux caches sont purges : embedding_cache contient des vecteurs 1536, et
query_cache des reponses serialisees sous l'ancienne forme.

Le nom de l'index change (_halfvec) pour qu'une base a moitie migree se
reconnaisse a l'oeil et pour qu'un test ne puisse pas passer par accident
contre un ancien index survivant.

Ni CONCURRENTLY ni maintenance_work_mem ici : la colonne vient d'etre videe,
l'index se construit sur zero ligne. La construction en masse appartient a
scripts/regenerate_embeddings.py --reindex.

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-09-04

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "f5a6b7c8d9e0"
down_revision = "e4f5a6b7c8d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_halfvec")

    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE vector(3072) USING NULL::vector(3072)"
    )

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")

    op.execute(
        "CREATE INDEX idx_articles_embedding_hnsw_halfvec ON articles "
        "USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_halfvec")

    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE vector(1536) USING NULL::vector(1536)"
    )

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")

    op.execute(
        "CREATE INDEX idx_articles_embedding_hnsw ON articles "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
