"""Initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # L'extension vector doit exister AVANT la creation de articles.embedding
    # (type VECTOR(3072) plus bas). Elle n'etait installee que par la migration
    # 7a8b9c0d1e2f, bien plus loin dans la chaine : un `alembic upgrade head`
    # sur une base vierge echouait donc des la premiere migration avec
    # 'type "vector" does not exist'. IF NOT EXISTS rend l'operation idempotente
    # et compatible avec les bases ou 7a8b9c0d1e2f a deja tourne.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Categories table
    op.create_table(
        'categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Laws table avec colonnes v2.1
    op.create_table(
        'laws',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('reference', sa.String(100), nullable=False, unique=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('category_id', sa.Integer(), nullable=True),
        # v2.1 Colonnes détection automatique
        sa.Column('language', sa.String(2), nullable=True),  # 'fr' | 'en'
        sa.Column('language_confidence', sa.Float(), nullable=True),
        sa.Column('detected_language', sa.String(2), nullable=True),
        sa.Column('suggested_categories', postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column('category_confidence', sa.Float(), nullable=True),
        # Autres colonnes
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('publication_date', sa.Date(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='draft'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['category_id'], ['categories.id'])
    )

    # Articles table
    op.create_table(
        'articles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('law_id', sa.Integer(), nullable=False),
        sa.Column('number', sa.String(20), nullable=False),
        sa.Column('title', sa.String(200), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(3072), nullable=True),  # pgvector 3072-dim Gemini
        sa.Column('order', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['law_id'], ['laws.id'], ondelete='CASCADE')
    )

    # Index
    op.create_index('idx_laws_language', 'laws', ['language'])
    op.create_index('idx_laws_category', 'laws', ['category_id'])
    op.create_index('idx_articles_law', 'articles', ['law_id'])

    # Insertion catégories initiales
    op.execute("""
        INSERT INTO categories (name, description) VALUES
        ('Droit Civil', 'Lois relatives au droit civil'),
        ('Droit Pénal', 'Lois relatives au droit pénal'),
        ('Droit Commercial', 'Lois relatives au commerce'),
        ('Droit du Travail', 'Lois relatives au travail'),
        ('Droit Fiscal', 'Lois relatives à la fiscalité'),
        ('Droit Administratif', 'Lois relatives à l''administration'),
        ('Droit de la Famille', 'Lois relatives à la famille'),
        ('Droit des Affaires', 'Lois relatives aux affaires'),
        ('Droit Constitutionnel', 'Constitution et lois constitutionnelles'),
        ('Procédure Civile', 'Procédures judiciaires civiles'),
        ('Procédure Pénale', 'Procédures judiciaires pénales'),
        ('Droit OHADA', 'Actes uniformes OHADA')
    """)


def downgrade() -> None:
    op.drop_table('articles')
    op.drop_table('laws')
    op.drop_table('categories')
