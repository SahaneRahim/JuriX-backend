"""Embeddings 1536 dimensions et index HNSW

Les embeddings faisaient 3072 dimensions, la sortie native de
gemini-embedding-001. C'est au-dessus du plafond de 2000 dimensions de pgvector
pour HNSW et IVFFlat : aucun index vectoriel n'etait donc possible, et la
migration 7a8b9c0d1e2f l'assumait explicitement ("Sequential scan is acceptable
for <20k articles"). Chaque recherche semantique balayait la table.

gemini-embedding-001 accepte output_dimensionality : le modele est entraine en
Matryoshka, les 1536 premieres dimensions portent l'essentiel du signal. A 1536
l'index HNSW redevient possible, l'empreinte disque est divisee par deux, et la
perte de qualite mesuree sur ce type de modele est marginale.

Consequence assumee : les vecteurs existants sont INUTILISABLES et la colonne
est videe (USING NULL). Il faut relancer scripts/regenerate_embeddings.py apres
cette migration. Entre les deux, la recherche semantique ne renvoie rien et
l'hybride degrade en texte seul.

Les deux caches sont purges pour la meme raison :
  - embedding_cache contient des vecteurs 3072 (la cle de cache inclut desormais
    la dimension, ceci est la ceinture en plus des bretelles) ;
  - query_cache contient des SearchResponse serialises sous l'ancienne forme,
    sans le champ `chunks`.

m=16 / ef_construction=64 sont les defauts pgvector, corrects en-dessous d'une
centaine de milliers de vecteurs. Ni CONCURRENTLY ni maintenance_work_mem ici :
la colonne vient d'etre videe, l'index se construit sur zero ligne et
instantanement. La reconstruction en masse appartient au backfill
(regenerate_embeddings.py --reindex).

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-09-03

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "e4f5a6b7c8d9"
down_revision = "d3e4f5a6b7c8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw")

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


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw")

    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE vector(3072) USING NULL::vector(3072)"
    )

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")
