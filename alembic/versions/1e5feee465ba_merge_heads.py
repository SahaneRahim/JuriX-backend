"""merge_heads

Revision ID: 1e5feee465ba
Revises: 004, 0b21fb6a3651
Create Date: 2026-01-13 16:01:59.279535

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1e5feee465ba'
down_revision = ('004', '0b21fb6a3651')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
