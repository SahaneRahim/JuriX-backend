"""Add processing status fields to laws table

Revision ID: 004
Revises: 003
Create Date: 2026-01-13

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add processing status tracking fields to laws table
    op.add_column('laws', sa.Column('processing_progress', sa.Integer(), nullable=True))
    op.add_column('laws', sa.Column('processing_error', sa.Text(), nullable=True))
    op.add_column('laws', sa.Column('processing_started_at', sa.DateTime(), nullable=True))
    
    # Create index for processing_progress for efficient filtering
    op.create_index('idx_laws_processing_progress', 'laws', ['processing_progress'])


def downgrade() -> None:
    # Drop index
    op.drop_index('idx_laws_processing_progress', table_name='laws')
    
    # Drop columns
    op.drop_column('laws', 'processing_started_at')
    op.drop_column('laws', 'processing_error')
    op.drop_column('laws', 'processing_progress')
