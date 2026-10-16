"""
Migration 016 : embeddings Voyage, halfvec(1024), index HNSW sur la colonne.

Ce que ces tests figent :
- l'aller-retour : la descente rend vector(768) et l'index d'avant, la montee
  halfvec(1024) et le nouvel index, un seul a chaque fois ;
- la montee VIDE les vecteurs ET leur provenance, et purge les deux caches :
  un vecteur EmbeddingGemma n'a pas de projection dans l'espace de Voyage ;
- l'ordre des operations : l'ancien index est supprime avant l'ALTER, sinon
  l'opclass vector_cosine_ops refuse le type halfvec.

Chaque test passe par Alembic, puis remet la base de test a `head` quoi qu'il
arrive.

Usage:
    pytest tests/test_migrations/test_embeddings_voyage_016.py -v
"""

import os
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from tests.conftest import TEST_DATABASE_URL

RACINE = Path(__file__).resolve().parents[2]
AVANT_016 = "f7a8b9c0d1e2"


def _alembic(action: str, cible: str) -> None:
    """`alembic <action> <cible>` sur la base de test, dans ce processus."""
    from alembic.config import Config

    from alembic import command

    precedente = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    try:
        getattr(command, action)(Config(str(RACINE / "alembic.ini")), cible)
    finally:
        if precedente is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = precedente


@pytest.fixture
def session(sync_db_session):
    """Base de test videe ; remise a `head` en sortie, meme apres un echec."""
    try:
        yield sync_db_session
    finally:
        sync_db_session.rollback()
        _alembic("upgrade", "head")


def _type_de_la_colonne(session) -> str:
    return session.execute(text(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'articles'::regclass AND attname = 'embedding'"
    )).scalar()


def _index_vectoriels(session) -> list:
    return session.execute(text(
        "SELECT c.relname, pg_get_indexdef(i.indexrelid) FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = 'articles'::regclass AND pg_get_indexdef(i.indexrelid) LIKE '%hnsw%'"
    )).all()


def test_aller_retour(session):
    session.commit()
    _alembic("downgrade", AVANT_016)

    assert _type_de_la_colonne(session) == "vector(768)"
    ((nom, definition),) = _index_vectoriels(session)
    assert nom == "idx_articles_embedding_hnsw_vector"
    assert "vector_cosine_ops" in definition
    session.commit()

    _alembic("upgrade", "head")

    assert _type_de_la_colonne(session) == "halfvec(1024)"
    ((nom, definition),) = _index_vectoriels(session)
    assert nom == "idx_articles_embedding_hnsw_halfvec1024"
    assert "USING hnsw (embedding halfvec_cosine_ops)" in definition


def test_la_montee_vide_vecteurs_provenance_et_caches(session):
    """
    Un vecteur Gemma garde avec son empreinte passerait pour « foreign » ;
    sans empreinte, pour d'origine inconnue : dans les deux cas a refaire. La
    colonne le refuserait de toute facon (768 contre 1024) : la montee le vide.
    """
    session.commit()
    _alembic("downgrade", AVANT_016)

    vecteur = np.ones(768) / np.sqrt(768)
    session.execute(text(
        "INSERT INTO laws (id, reference, title, content, type, language, status) "
        "VALUES (1, 'LOI-016', 'Loi', 'Contenu.', 'loi', 'fr', 'published')"
    ))
    session.execute(text(
        "INSERT INTO articles (id, law_id, number, content, \"order\", embed, embedding, embedding_model) "
        "VALUES (1, 1, '1', 'Article premier.', 1, true, CAST(:v AS vector), 'gemma|ancien')"
    ), {"v": "[" + ",".join(f"{x:.6g}" for x in vecteur) + "]"})
    session.execute(text(
        "INSERT INTO embedding_cache (text_hash, embedding_json, expires_at) "
        "VALUES ('cle', '[0.1]', now() + interval '1 day')"
    ))
    session.execute(text(
        "INSERT INTO query_cache (cache_key, response_json, expires_at) "
        "VALUES ('cle', '{}', now() + interval '1 day')"
    ))
    session.commit()

    _alembic("upgrade", "head")

    article = session.execute(text(
        "SELECT embedding IS NULL AS vide, embedding_model, content FROM articles WHERE id = 1"
    )).one()
    assert article.vide
    assert article.embedding_model is None
    assert article.content == "Article premier.", "seuls les vecteurs sont touches"
    assert session.execute(text("SELECT count(*) FROM embedding_cache")).scalar() == 0
    assert session.execute(text("SELECT count(*) FROM query_cache")).scalar() == 0
    session.commit()


def test_une_copie_ccnew_laissee_par_un_reindex_interrompu_ne_bloque_pas(session):
    """
    Un REINDEX CONCURRENTLY coupe (redemarrage de la base) laisse une copie
    _ccnew de l'ancien index. L'ALTER la reconstruirait, et vector_cosine_ops
    refuse le type halfvec : l'API ne demarrerait plus.
    """
    session.commit()
    _alembic("downgrade", AVANT_016)
    session.execute(text(
        "CREATE INDEX idx_articles_embedding_hnsw_vector_ccnew ON articles "
        "USING hnsw (embedding vector_cosine_ops)"
    ))
    session.commit()

    _alembic("upgrade", "head")

    noms = [nom for nom, _ in _index_vectoriels(session)]
    assert noms == ["idx_articles_embedding_hnsw_halfvec1024"]
