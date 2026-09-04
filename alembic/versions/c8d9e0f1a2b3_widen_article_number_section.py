"""Elargit articles.number et articles.section

Ces deux colonnes etaient trop etroites pour ce que le chunker produit, et le
depassement ne tronquait pas : il faisait perdre TOUS les articles de la loi.

  - `number` etait String(20). Le motif d'ordinaux composes de
    app/utils/text_chunker.py produit des numeros comme
    'QUATRE-VINGT-DIX-SEPTIEME' (25 caracteres). normalize_article_number ne
    convertit que jusqu'a 'dixieme' et rend la chaine telle quelle au-dela.

  - `section` etait String(300). Les SECTION_PATTERNS capturent `[^\\n]*`,
    c'est-a-dire toute la fin de ligne, et le `\\s*` qui precede peut franchir
    un saut de ligne : la capture s'etend donc souvent sur deux lignes. Sur du
    markdown, ou un paragraphe entier tient frequemment sur une seule ligne,
    des sections de plus de 450 caracteres sont courantes.

POURQUOI C'ETAIT GRAVE. PostgreSQL ne tronque pas un varchar(n) : il leve
StringDataRightTruncation au commit. Cette exception etait capturee par le
`except Exception: return 0` de _split_and_save_articles, qui avait deja
supprime les articles existants quelques lignes plus haut. Resultat : la loi
perdait la totalite de ses articles, et etait publiee malgre tout. Seul le log
en gardait trace.

`section` passe en Text plutot qu'en varchar plus large : il n'existe aucune
borne superieure defendable a la longueur d'un en-tete de titre capture par une
expression gourmande. `number` reste borne — un numero d'article est court par
nature, et 64 laisse la place a tous les ordinaux composes du corpus.

Revision ID: c8d9e0f1a2b3
Revises: b7c8d9e0f1a2
Create Date: 2026-09-04

"""
import sqlalchemy as sa
from alembic import op


revision = "c8d9e0f1a2b3"
down_revision = "b7c8d9e0f1a2"
branch_labels = None
depends_on = None


# PostgreSQL refuse d'alterer le type d'une colonne dont depend un trigger
# ("cannot alter type of a column used in a trigger definition"). Le trigger FTS
# est declare BEFORE INSERT OR UPDATE OF content, number : il faut donc le
# retirer, alterer, puis le recreer a l'identique. La FONCTION n'est pas
# touchee, seulement le declencheur.
_DROP_TRIGGER = "DROP TRIGGER IF EXISTS articles_search_vector_trigger ON articles"
_CREATE_TRIGGER = """
    CREATE TRIGGER articles_search_vector_trigger
    BEFORE INSERT OR UPDATE OF content, number ON articles
    FOR EACH ROW EXECUTE FUNCTION articles_search_vector_update()
"""


def upgrade() -> None:
    op.execute(_DROP_TRIGGER)
    op.alter_column(
        "articles",
        "number",
        existing_type=sa.String(20),
        type_=sa.String(64),
        existing_nullable=False,
    )
    op.alter_column(
        "articles",
        "section",
        existing_type=sa.String(300),
        type_=sa.Text(),
        existing_nullable=True,
    )
    op.execute(_CREATE_TRIGGER)


def downgrade() -> None:
    # Retour en arriere destructeur : toute valeur plus longue que l'ancienne
    # borne serait refusee par PostgreSQL. On tronque explicitement d'abord,
    # sinon le downgrade echoue sur un corpus deja reingere.
    op.execute("UPDATE articles SET number = left(number, 20) WHERE length(number) > 20")
    op.execute("UPDATE articles SET section = left(section, 300) WHERE length(section) > 300")
    op.execute(_DROP_TRIGGER)
    op.alter_column(
        "articles",
        "section",
        existing_type=sa.Text(),
        type_=sa.String(300),
        existing_nullable=True,
    )
    op.alter_column(
        "articles",
        "number",
        existing_type=sa.String(64),
        type_=sa.String(20),
        existing_nullable=False,
    )
    op.execute(_CREATE_TRIGGER)
