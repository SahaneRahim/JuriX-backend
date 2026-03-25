"""Add page_number to articles

Revision ID: 668e3740d3b5
Revises: 3f0760423953
Create Date: 2026-01-21 03:44:42.433227

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '668e3740d3b5'
down_revision = '3f0760423953'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add page_number column to articles table
    op.add_column('articles', sa.Column('page_number', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('articles', 'page_number')
