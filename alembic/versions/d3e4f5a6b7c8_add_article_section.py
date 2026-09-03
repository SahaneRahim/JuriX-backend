"""Ajoute articles.section, present dans le modele mais jamais cree

Le modele ORM `Article` declare `section = Column(String(300))` depuis
l'introduction du decoupage par TITRE/CHAPITRE (`text_chunker`), mais aucune
migration ne l'a jamais creee.

Consequence : `process_law._split_and_save_articles` construit chaque Article
avec `section=article_data.get("section")`, donc TOUTE insertion d'article
echouait avec UndefinedColumnError sur une base correctement migree.
L'ingestion de documents n'a jamais pu fonctionner.

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-09-03

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "d3e4f5a6b7c8"
down_revision = "c2d3e4f5a6b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column(
            "section",
            sa.String(length=300),
            nullable=True,
            comment="En-tete de section (TITRE / CHAPITRE) dont releve l'article",
        ),
    )


def downgrade() -> None:
    op.drop_column("articles", "section")
