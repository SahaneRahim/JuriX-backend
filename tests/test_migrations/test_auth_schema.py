"""
Le schema que la migration 010 doit avoir pose.

Ces tests portent sur la BASE, pas sur les modeles : un modele peut declarer ce
qu'il veut, c'est PostgreSQL qui tranche. Ils attrapent le cas ou la migration
n'a pas ete rejouee, ou a divergé du modele.

Usage:
    pytest tests/test_migrations/test_auth_schema.py -v
"""

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_hashed_password_est_nullable(db_session):
    """Un compte cree par Google n'a pas de mot de passe."""
    nullable = (
        await db_session.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_name = 'users' AND column_name = 'hashed_password'"
            )
        )
    ).scalar()

    assert nullable == "YES"


@pytest.mark.asyncio
async def test_index_unique_sur_google_sub(db_session):
    """
    Unique ET nullable : PostgreSQL autorise plusieurs NULL sur un index
    unique, donc les comptes par mot de passe cohabitent sans contrainte.
    """
    definition = (
        await db_session.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_users_google_sub'")
        )
    ).scalar()

    assert definition is not None, "index absent : la migration 010 n'est pas appliquee"
    assert "UNIQUE" in definition


@pytest.mark.asyncio
async def test_contrainte_au_moins_une_identite(db_session):
    """Un compte doit garder au moins une porte d'entree."""
    nom = (
        await db_session.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conname = 'ck_users_au_moins_une_identite'"
            )
        )
    ).scalar()

    assert nom is not None


@pytest.mark.asyncio
async def test_la_contrainte_refuse_un_compte_sans_identite(db_session):
    """La contrainte doit MORDRE, pas seulement exister."""
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await db_session.execute(
            text(
                "INSERT INTO users (email, username, hashed_password, google_sub, "
                "role, is_active, is_verified, created_at, updated_at) "
                "VALUES ('fantome@x.cm', 'fantome', NULL, NULL, 'user', true, false, "
                "now(), now())"
            )
        )
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_colonne_title_sur_conversations(db_session):
    longueur = (
        await db_session.execute(
            text(
                "SELECT character_maximum_length FROM information_schema.columns "
                "WHERE table_name = 'conversations' AND column_name = 'title'"
            )
        )
    ).scalar()

    assert longueur == 120


@pytest.mark.asyncio
async def test_index_composite_pour_la_liste(db_session):
    """
    C'est lui qui rend `GET /rag/conversations` servable sans jointure : filtre
    sur user_id, tri sur updated_at decroissant.
    """
    definition = (
        await db_session.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'idx_conversations_user_updated'"
            )
        )
    ).scalar()

    assert definition is not None
    assert "user_id" in definition
    assert "updated_at DESC" in definition
