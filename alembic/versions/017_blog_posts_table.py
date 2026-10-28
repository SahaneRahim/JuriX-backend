"""create blog_posts table

Revision ID: 017_blog_posts_table
Revises: a8b9c0d1e2f3
Create Date: 2026-10-26 10:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '017_blog_posts_table'
down_revision = 'a8b9c0d1e2f3'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS blog_posts (
            id SERIAL PRIMARY KEY,
            slug VARCHAR(255) NOT NULL UNIQUE,
            title VARCHAR(500) NOT NULL,
            category VARCHAR(100) NOT NULL,
            published_at DATE NOT NULL DEFAULT CURRENT_DATE,
            reading_time VARCHAR(50) NOT NULL DEFAULT '4 min',
            context TEXT NOT NULL,
            summary TEXT NOT NULL,
            key_points JSONB NOT NULL DEFAULT '[]'::jsonb,
            citations JSONB NOT NULL DEFAULT '[]'::jsonb,
            citizen_impact TEXT NOT NULL,
            suggested_question VARCHAR(500),
            tags JSONB NOT NULL DEFAULT '[]'::jsonb,
            is_published BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_blog_posts_slug ON blog_posts(slug);
        CREATE INDEX IF NOT EXISTS idx_blog_posts_category ON blog_posts(category);
        CREATE INDEX IF NOT EXISTS idx_blog_posts_published ON blog_posts(is_published, published_at DESC);
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS blog_posts CASCADE;")
