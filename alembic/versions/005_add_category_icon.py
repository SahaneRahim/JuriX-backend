"""Add icon field to categories table

Revision ID: 005
Revises: 1e5feee465ba
Create Date: 2026-01-13

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005'
down_revision = '1e5feee465ba'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add icon field to categories table
    op.add_column('categories', sa.Column('icon', sa.String(10), nullable=True))


def downgrade() -> None:
    # Drop icon column
    op.drop_column('categories', 'icon')
