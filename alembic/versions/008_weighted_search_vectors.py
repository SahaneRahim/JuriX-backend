"""Weighted search vectors: law title in weight A, article number in B.

Les tsvector concatenaient titre et contenu dans une seule chaine, sans un seul
`setweight` : un mot du titre pesait exactement autant qu'un mot du corps.
Mesure sur le corpus avant migration, requete « nomination » — le document dont
le titre ne porte PAS le mot sortait premier (ts_rank_cd 0,4000) devant les cinq
documents dont le titre le porte (0,2000), parce qu'il le repete dans son corps.

Deux changements :

1. Pondération A/B/D (voir app/services/search_vectors.py pour le detail).
2. Le titre de la loi entre dans `articles.search_vector`. La recherche
   interroge les ARTICLES ; ce vecteur ne contenait que le contenu et le numero,
   donc aucune requete ne pouvait atteindre un titre par cette branche.

Le point 2 impose une jointure dans le trigger des articles, et donc un second
trigger sur `laws` qui reindexe les articles quand un titre change.

Revision ID: e0f1a2b3c4d5
Revises: d9e0f1a2b3c4
Create Date: 2026-09-05
"""

from typing import Sequence, Union

from alembic import op

from app.services.search_vectors import (
    ARTICLES_TRIGGER_FUNCTION,
    LAWS_TITLE_CASCADE_FUNCTION,
    LAWS_TRIGGER_FUNCTION,
    REINDEX_ARTICLES_SQL,
    REINDEX_LAWS_SQL,
)

revision: str = "e0f1a2b3c4d5"
down_revision: Union[str, None] = "d9e0f1a2b3c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(LAWS_TRIGGER_FUNCTION)
    op.execute(ARTICLES_TRIGGER_FUNCTION)
    op.execute(LAWS_TITLE_CASCADE_FUNCTION)

    # La garde `IS DISTINCT FROM` vit dans le WHEN et non dans le corps :
    # PostgreSQL saute alors l'appel de fonction au lieu d'y entrer pour en
    # ressortir aussitot. AFTER et non BEFORE : les lignes filles doivent voir
    # le titre valide.
    op.execute("""
        DROP TRIGGER IF EXISTS laws_title_cascade_trigger ON laws;
        CREATE TRIGGER laws_title_cascade_trigger
        AFTER UPDATE OF title ON laws
        FOR EACH ROW
        WHEN (NEW.title IS DISTINCT FROM OLD.title)
        EXECUTE FUNCTION laws_title_reindex_articles()
    """)

    # Le trigger des articles doit aussi se declencher sur `title` : il entre
    # desormais dans le vecteur, et la clause UPDATE OF ne le listait pas.
    op.execute("""
        DROP TRIGGER IF EXISTS articles_search_vector_trigger ON articles;
        CREATE TRIGGER articles_search_vector_trigger
        BEFORE INSERT OR UPDATE OF content, number, title, law_id ON articles
        FOR EACH ROW EXECUTE FUNCTION articles_search_vector_update()
    """)

    # Reindexation en masse. On ecrit `search_vector` directement plutot que de
    # declencher les triggers par un UPDATE bidon : c'est une seule passe au
    # lieu d'une par ligne, et le resultat est identique par construction.
    op.execute(REINDEX_LAWS_SQL)
    op.execute(REINDEX_ARTICLES_SQL)


def downgrade() -> None:
    """Retablit les vecteurs plats, sans ponderation."""
    op.execute("DROP TRIGGER IF EXISTS laws_title_cascade_trigger ON laws")
    op.execute("DROP FUNCTION IF EXISTS laws_title_reindex_articles()")

    op.execute("""
        CREATE OR REPLACE FUNCTION laws_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                to_tsvector('french', coalesce(NEW.title, '') || ' ' || coalesce(NEW.content, ''))
                || to_tsvector('english', coalesce(NEW.title, '') || ' ' || coalesce(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION articles_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                to_tsvector('french', coalesce(NEW.content, ''))
                || to_tsvector('english', coalesce(NEW.content, ''))
                || to_tsvector('simple', coalesce(NEW.number, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS articles_search_vector_trigger ON articles;
        CREATE TRIGGER articles_search_vector_trigger
        BEFORE INSERT OR UPDATE OF content, number ON articles
        FOR EACH ROW EXECUTE FUNCTION articles_search_vector_update()
    """)
    op.execute("""
        UPDATE laws SET search_vector =
            to_tsvector('french', coalesce(title, '') || ' ' || coalesce(content, ''))
            || to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    """)
    op.execute("""
        UPDATE articles SET search_vector =
            to_tsvector('french', coalesce(content, ''))
            || to_tsvector('english', coalesce(content, ''))
            || to_tsvector('simple', coalesce(number, ''))
    """)
