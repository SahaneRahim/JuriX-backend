"""Add display_order to categories

Revision ID: 006
Revises: 005
Create Date: 2026-01-14

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('categories', sa.Column('display_order', sa.Integer(), nullable=True, server_default='0'))
    # Make it non-nullable after setting defaults
    op.execute("UPDATE categories SET display_order = id WHERE display_order IS NULL")
    op.alter_column('categories', 'display_order', nullable=False)


def downgrade() -> None:
    op.drop_column('categories', 'display_order')
