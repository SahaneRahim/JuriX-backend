"""Add PostgreSQL FTS and cache tables for Meilisearch/Redis replacement

Revision ID: a1b2c3d4e5f6
Revises: 668e3740d3b5
Create Date: 2026-04-17

Summary:
- Add search_vector (tsvector) to articles for full-text search
- Add search_vector (tsvector) to laws for full-text search  
- Add GIN indexes for fast FTS queries
- Create pg_trgm extension for fuzzy matching
- Create query_cache table (replaces Redis search cache)
- Create embedding_cache table (replaces Redis embedding cache)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '668e3740d3b5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Enable pg_trgm extension (fuzzy matching, similarity search)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # 2. Add tsvector column to articles for full-text search
    op.add_column(
        'articles',
        sa.Column('search_vector', postgresql.TSVECTOR, nullable=True)
    )

    # 3. Add tsvector column to laws for full-text search
    op.add_column(
        'laws',
        sa.Column('search_vector', postgresql.TSVECTOR, nullable=True)
    )

    # 4. Create GIN indexes for fast FTS
    op.create_index(
        'idx_articles_search_vector',
        'articles',
        ['search_vector'],
        postgresql_using='gin'
    )
    op.create_index(
        'idx_laws_search_vector',
        'laws',
        ['search_vector'],
        postgresql_using='gin'
    )

    # 5. Populate search_vector for existing articles
    op.execute("""
        UPDATE articles 
        SET search_vector = 
            to_tsvector('french', coalesce(content, ''))
            || to_tsvector('english', coalesce(content, ''))
            || to_tsvector('simple', coalesce(number, ''))
        WHERE content IS NOT NULL
    """)

    # 6. Populate search_vector for existing laws
    op.execute("""
        UPDATE laws
        SET search_vector =
            to_tsvector('french', coalesce(title, '') || ' ' || coalesce(content, ''))
            || to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
        WHERE content IS NOT NULL
    """)

    # 7. Create trigger function to auto-update articles.search_vector on insert/update
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

    # 8. Create trigger function for laws.search_vector
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
        DROP TRIGGER IF EXISTS laws_search_vector_trigger ON laws;
        CREATE TRIGGER laws_search_vector_trigger
        BEFORE INSERT OR UPDATE OF title, content ON laws
        FOR EACH ROW EXECUTE FUNCTION laws_search_vector_update()
    """)

    # 9. Create query_cache table (replaces Redis for search result caching)
    op.create_table(
        'query_cache',
        sa.Column('cache_key', sa.String(64), primary_key=True),
        sa.Column('response_json', sa.Text, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime, nullable=False),
    )
    op.create_index('idx_query_cache_expires', 'query_cache', ['expires_at'])

    # 10. Create embedding_cache table (replaces Redis for embedding caching)
    op.create_table(
        'embedding_cache',
        sa.Column('text_hash', sa.String(64), primary_key=True),
        sa.Column('embedding_json', sa.Text, nullable=False),  # JSON array of floats
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime, nullable=False),
    )
    op.create_index('idx_embedding_cache_expires', 'embedding_cache', ['expires_at'])


def downgrade() -> None:
    # Remove embedding_cache
    op.drop_index('idx_embedding_cache_expires', table_name='embedding_cache')
    op.drop_table('embedding_cache')

    # Remove query_cache
    op.drop_index('idx_query_cache_expires', table_name='query_cache')
    op.drop_table('query_cache')

    # Remove triggers and functions
    op.execute("DROP TRIGGER IF EXISTS laws_search_vector_trigger ON laws")
    op.execute("DROP TRIGGER IF EXISTS articles_search_vector_trigger ON articles")
    op.execute("DROP FUNCTION IF EXISTS laws_search_vector_update()")
    op.execute("DROP FUNCTION IF EXISTS articles_search_vector_update()")

    # Remove indexes
    op.drop_index('idx_laws_search_vector', table_name='laws')
    op.drop_index('idx_articles_search_vector', table_name='articles')

    # Remove columns
    op.drop_column('laws', 'search_vector')
    op.drop_column('articles', 'search_vector')
