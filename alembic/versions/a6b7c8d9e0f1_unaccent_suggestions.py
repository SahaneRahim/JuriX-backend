"""Recherche d'autocompletion insensible aux accents

/search/suggest ne trouvait rien pour une saisie sans accents. Le francais
juridique en est plein — "societe", "arrete", "decret" — et les utilisateurs les
omettent presque toujours. Mesure sur le corpus : word_similarity('societe', ...)
vaut 0,500 contre un titre contenant "societe" accentue, sous le seuil de 0,6,
alors que la version accentuee vaut 1,000.

unaccent() est declaree STABLE et non IMMUTABLE — elle depend d'un dictionnaire
qui peut changer — donc inutilisable telle quelle dans un index. Le contournement
standard est un enrobage IMMUTABLE qui fige le dictionnaire utilise.

Les deux index trigrammes portent sur l'expression depliee, sans quoi chaque
autocompletion balaierait la table entiere.

Revision ID: a6b7c8d9e0f1
Revises: f5a6b7c8d9e0
Create Date: 2026-09-04

"""
from alembic import op


revision = "a6b7c8d9e0f1"
down_revision = "f5a6b7c8d9e0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # PARALLEL SAFE et STRICT : sans elles, l'expression bloque le parallelisme
    # et se comporte mal sur NULL.
    op.execute("""
        CREATE OR REPLACE FUNCTION immutable_unaccent(text)
        RETURNS text
        LANGUAGE sql
        IMMUTABLE STRICT PARALLEL SAFE
        AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, $1) $$
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_laws_title_unaccent_trgm
        ON laws USING gin (immutable_unaccent(title) gin_trgm_ops)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_laws_reference_unaccent_trgm
        ON laws USING gin (immutable_unaccent(reference) gin_trgm_ops)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_laws_reference_unaccent_trgm")
    op.execute("DROP INDEX IF EXISTS idx_laws_title_unaccent_trgm")
    op.execute("DROP FUNCTION IF EXISTS immutable_unaccent(text)")
    # L'extension unaccent n'est PAS supprimee : d'autres objets peuvent en
    # dependre, et sa presence est sans effet de bord.
