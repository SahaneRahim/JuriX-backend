"""Fusionne les deux heads : pgvector et FTS PostgreSQL

Deux migrations derivaient toutes les deux de 668e3740d3b5 sans jamais etre
fusionnees :

    668e3740d3b5 ─┬─ 7a8b9c0d1e2f  (enable_pgvector)
                  └─ a1b2c3d4e5f6  (add_postgres_fts_and_cache)

`alembic upgrade head` echouait donc avec "Multiple head revisions are present",
et personne ne pouvait provisionner la base. Consequence indirecte : les objets
crees par a1b2c3d4e5f6 — search_vector, les index GIN, les triggers, pg_trgm,
query_cache et embedding_cache — pouvaient n'avoir jamais ete crees, alors que
toute la recherche en depend.

Revision ID: b1c2d3e4f5a6
Revises: 7a8b9c0d1e2f, a1b2c3d4e5f6
Create Date: 2026-09-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b1c2d3e4f5a6"
down_revision = ("7a8b9c0d1e2f", "a1b2c3d4e5f6")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Point de convergence : aucune operation de schema."""
    pass


def downgrade() -> None:
    """Re-scinde l'historique en deux branches."""
    pass
