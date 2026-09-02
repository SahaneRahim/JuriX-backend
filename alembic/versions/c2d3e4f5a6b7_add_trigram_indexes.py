"""Index trigrammes pour l'autocompletion et la tolerance aux fautes

La migration a1b2c3d4e5f6 installe l'extension pg_trgm mais ne cree AUCUN index
gin_trgm_ops. Consequence : chaque appel a similarity() ou a l'operateur % fait
un balayage sequentiel complet.

Deux chemins de code en dependent :
  - postgres_search_service._trigram_articles_query (repli quand la recherche
    plein texte ne renvoie rien : c'est ce qui rattrape les fautes de frappe)
  - la reecriture de /search/suggest (autocompletion, appelee a chaque frappe)

Note sur l'operateur : `similarity(a, b) > seuil` n'est PAS accelere par l'index.
Seul l'operateur `%` l'est. Les requetes doivent donc filtrer avec `%` dans le
WHERE et ne garder similarity() que dans le ORDER BY.

idx_articles_content_trgm est long a construire (texte integral des articles) et
volumineux, mais c'est le seul moyen de rendre le repli tolerant aux fautes
utilisable sur un corpus de plusieurs milliers de documents.

CREATE INDEX simple et non CONCURRENTLY : Alembic execute dans une transaction.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-09-02

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "c2d3e4f5a6b7"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent : a1b2c3d4e5f6 l'a normalement deja creee
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Autocompletion : titre et reference de loi
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_laws_title_trgm "
        "ON laws USING gin (title gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_laws_reference_trgm "
        "ON laws USING gin (reference gin_trgm_ops)"
    )

    # Repli tolerant aux fautes sur le contenu des articles
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_articles_content_trgm "
        "ON articles USING gin (content gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_content_trgm")
    op.execute("DROP INDEX IF EXISTS idx_laws_reference_trgm")
    op.execute("DROP INDEX IF EXISTS idx_laws_title_trgm")
    # L'extension pg_trgm n'est pas supprimee : elle appartient a a1b2c3d4e5f6.
