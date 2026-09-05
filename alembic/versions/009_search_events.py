"""Journal des recherches, pour que les statistiques cessent d'etre inventees.

`GET /api/v1/analytics/search` renvoyait des constantes codees en dur — 350
recherches, 150 ms, une repartition 100/50/200 entre les modes — que le tableau
de bord admin affichait comme des mesures. Le champ `note: "Mock data"` qui les
accompagnait n'etait lu par personne.

Rien en base ne permettait de faire mieux : `query_cache` ne garde qu'un
hachage, avec cinq minutes de duree de vie, et une requete servie depuis le
cache n'y cree aucune ligne. Ce n'est pas un journal.

Cette table en est un. Une ligne par recherche, ecrite hors du chemin critique.

Revision ID: f1a2b3c4d5e6
Revises: e0f1a2b3c4d5
Create Date: 2026-09-05
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "e0f1a2b3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "search_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        # La requete est conservee : c'est elle qui dit ce que les gens
        # cherchent et ne trouvent pas. Bornee a 500 caracteres, la meme limite
        # que SearchRequest.query.
        sa.Column("query", sa.String(500), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("results_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cached", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    # Toutes les lectures d'analytics sont « les N derniers jours, groupees » :
    # l'index porte donc sur la date.
    op.create_index("idx_search_events_created", "search_events", ["created_at"])
    op.create_index("idx_search_events_mode", "search_events", ["mode"])


def downgrade() -> None:
    op.drop_index("idx_search_events_mode", table_name="search_events")
    op.drop_index("idx_search_events_created", table_name="search_events")
    op.drop_table("search_events")
