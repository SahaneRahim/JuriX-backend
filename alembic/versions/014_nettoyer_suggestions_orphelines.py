"""Nettoie les suggestions de categorie orphelines.

La premiere version de 013 supprimait « Procédure Pénale » et « Procédure
Civile » sans faire suivre `laws.suggested_categories`, un tableau
d'identifiants sans cle etrangere : 5 lois de jurix_dev y gardent les
identifiants 10 et 11, qui ne designent plus rien. 013 a ete corrigee depuis,
mais une base deja passee par l'ancienne version est a la revision
d5e6f7a8b9c0 : `alembic upgrade head` n'y rejouerait pas la correction.

Cette migration fait le nettoyage, sur toute base, et ne fait que lui :
identifiants inconnus et doublons retires des suggestions, ordre garde.
Les suggestions ne se laissent pas recoller (on ne sait plus ce que 10 et 11
designaient) ; le reclassement par lots les reecrit de toute facon.

Idempotente : sur une base deja propre, aucune ligne n'est reecrite.

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-10-07
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        WITH propres AS (
            SELECT l.id, ARRAY(
                SELECT u.x
                FROM unnest(l.suggested_categories) WITH ORDINALITY AS u(x, rang)
                WHERE u.x IN (SELECT id FROM categories)
                GROUP BY u.x
                ORDER BY min(u.rang)
            ) AS suggestions
            FROM laws l
            WHERE l.suggested_categories IS NOT NULL
        )
        UPDATE laws SET suggested_categories = propres.suggestions
        FROM propres
        WHERE laws.id = propres.id
          AND laws.suggested_categories IS DISTINCT FROM propres.suggestions
    """))
    restantes = conn.execute(sa.text("""
        SELECT count(*) FROM laws l, unnest(l.suggested_categories) AS x
        WHERE x NOT IN (SELECT id FROM categories)
    """)).scalar_one()
    if restantes:
        raise RuntimeError(f"{restantes} suggestion(s) designent encore une categorie inexistante")


def downgrade() -> None:
    """Rien a defaire : remettre des identifiants qui ne designent rien serait un defaut."""
