"""Enable pgvector and migrate embeddings column

Revision ID: 7a8b9c0d1e2f
Revises: 668e3740d3b5
Create Date: 2026-01-27 13:10:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '7a8b9c0d1e2f'
down_revision = '668e3740d3b5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Enable pgvector extension and migrate embedding column.
    
    Steps:
    1. Enable pgvector extension
    2. Convert embedding column to Vector(3072) for Gemini embeddings
    
    Note: No vector index is created because 3072 dimensions exceeds
    pgvector's 2000-dimension limit for HNSW/IVFFlat indexes.
    Sequential scan is acceptable for <20k articles.
    """
    # Step 1: Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Step 2: Convert embedding column type to vector(3072)
    # 3072 dimensions = native Gemini gemini-embedding-001 output
    op.execute("""
        ALTER TABLE articles 
        ALTER COLUMN embedding 
        TYPE vector(3072) 
        USING NULL::vector(3072)
    """)


def downgrade() -> None:
    """
    Rollback pgvector changes.
    
    Steps:
    1. Drop HNSW index
    2. Convert embedding column back to JSON
    3. Drop pgvector extension (if no other tables use it)
    """
    # Step 1: Drop index
    op.execute('DROP INDEX IF EXISTS idx_articles_embedding_cosine')
    
    # Step 2: Convert back to JSON
    op.execute("""
        ALTER TABLE articles 
        ALTER COLUMN embedding 
        TYPE json 
        USING NULL::json
    """)
    
    # Step 3: Drop extension (commented out for safety - may affect other tables)
    # op.execute('DROP EXTENSION IF EXISTS vector')
