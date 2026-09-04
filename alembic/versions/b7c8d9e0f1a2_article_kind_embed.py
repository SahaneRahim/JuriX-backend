"""Classification des chunks : kind, embed, embed_text

app/utils/chunk_refiner.py existait depuis longtemps, testé, et n'était appelé
par personne. Il post-traite la sortie de text_chunker et produit trois
informations que la table ne savait pas stocker :

  - `kind` : la nature du chunk (article, legal_basis, preamble, boilerplate,
    roster, table, fragment, continuation). Tout ce qui n'est pas un article
    normatif encombrait l'index vectoriel : les visas, les formules
    d'exécution, les listes nominatives.
  - `embed` : faut-il vectoriser ce chunk. Rien n'est supprimé — un visa reste
    consultable et cherchable en plein texte — mais il ne consomme plus d'appel
    d'embedding et ne pollue plus les résultats sémantiques.
  - `embed_text` : le texte réellement envoyé au modèle d'embedding, préfixé de
    l'en-tête du document (référence, titre, date, catégorie, section, page).
    Sans lui, "Article 3.- La dépense résultant des présentes dispositions sera
    imputée sur le budget de l'État" est un chunk orphelin, indistinguable des
    milliers d'articles identiques du corpus. `content` reste intact : ce qui
    est affiché et cité ne change pas.

Défaut de `embed` : true. Les articles déjà en base gardent donc le
comportement actuel, et le retraitement d'un document les reclasse.

Index partiel sur `embed` : la requête de génération d'embeddings filtre
dessus, et la majorité des chunks d'un corpus de décrets de nomination est
non-embeddable.

Revision ID: b7c8d9e0f1a2
Revises: a6b7c8d9e0f1
Create Date: 2026-09-04

"""
import sqlalchemy as sa
from alembic import op


revision = "b7c8d9e0f1a2"
down_revision = "a6b7c8d9e0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("kind", sa.String(30), nullable=True))
    op.add_column(
        "articles",
        sa.Column("embed", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column("articles", sa.Column("embed_text", sa.Text(), nullable=True))

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_articles_embed_pending "
        "ON articles (law_id) WHERE embed AND embedding IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embed_pending")
    op.drop_column("articles", "embed_text")
    op.drop_column("articles", "embed")
    op.drop_column("articles", "kind")
