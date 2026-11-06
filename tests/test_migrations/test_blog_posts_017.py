"""
Le modèle BlogPost est celui de la migration 017, colonne pour colonne.

La table de production vient de alembic/versions/017_blog_posts_table.py (SQL
brut). Le premier modèle en différait : `published_at` en DateTime pour une
DATE, `suggested_question` en Text NOT NULL pour un VARCHAR(500) nullable, des
index `ix_blog_posts_*` pour les `idx_blog_posts_*` de production. Rien ne
cassait à l'exécution, mais `alembic revision --autogenerate` proposait quinze
changements, dont la suppression des trois index de production, et une base
créée par `create_all` n'avait pas le schéma de production.

Ce test fait la comparaison même d'autogenerate (types et défauts compris),
sur la base migrée par `alembic upgrade head` : aucun écart.

Usage:
    pytest tests/test_migrations/test_blog_posts_017.py -v
"""

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import MetaData, text

from app.models.blog import BlogPost


def test_autogenerate_ne_voit_aucun_ecart(sync_db_session):
    metadonnees = MetaData()
    BlogPost.__table__.to_metadata(metadonnees)
    contexte = MigrationContext.configure(
        sync_db_session.connection(),
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "include_name": lambda nom, genre, parent: genre != "table" or nom == "blog_posts",
        },
    )

    assert compare_metadata(contexte, metadonnees) == []


def test_les_index_de_production(sync_db_session):
    """Les noms de 017 : un autogenerate qui les ignorerait proposerait de les supprimer."""
    index = dict(sync_db_session.execute(text(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'blog_posts'"
    )).all())

    assert {"idx_blog_posts_slug", "idx_blog_posts_category", "idx_blog_posts_published"} <= set(index)
    assert "(is_published, published_at DESC)" in index["idx_blog_posts_published"]
    assert {i.name for i in BlogPost.__table__.indexes} == {
        "idx_blog_posts_slug", "idx_blog_posts_category", "idx_blog_posts_published"
    }
