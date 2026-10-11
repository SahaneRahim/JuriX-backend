"""
Le schema que la migration 015 doit avoir pose : users.persona.

Ces tests portent sur la BASE : c'est PostgreSQL qui tranche. Le dernier
verifie que la liste figee dans la contrainte accepte chaque profil de
app/core/personas.py : une divergence (un profil ajoute d'un seul cote, ou
« étudiant » enregistre en NFD dans la migration) y echoue.

Usage:
    pytest tests/test_migrations/test_persona_du_compte_015.py -v
"""

import unicodedata

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.personas import PERSONAS


async def _inserer_compte(db, persona):
    await db.execute(
        text(
            "INSERT INTO users (email, username, hashed_password, role, is_active, is_verified, "
            "created_at, updated_at, persona) VALUES ('p@essai.cm', 'p-essai', 'x', 'user', true, "
            "true, now(), now(), :persona)"
        ),
        {"persona": persona},
    )


@pytest.mark.asyncio
async def test_la_colonne_est_nullable(db_session):
    ligne = (await db_session.execute(text(
        "SELECT is_nullable, data_type, character_maximum_length FROM information_schema.columns "
        "WHERE table_name = 'users' AND column_name = 'persona'"
    ))).one_or_none()

    assert ligne is not None, "colonne absente : la migration 015 n'est pas appliquee"
    assert ligne == ("YES", "character varying", 50)


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", [None, *PERSONAS])
async def test_la_contrainte_accepte_chaque_profil(db_session, persona):
    await _inserer_compte(db_session, persona)
    await db_session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", [
    "astronaute",
    "general",
    unicodedata.normalize("NFD", "étudiant"),
])
async def test_la_contrainte_mord(db_session, persona):
    """« general » n'est jamais un profil de compte : un refus ne s'enregistre pas."""
    with pytest.raises(IntegrityError):
        await _inserer_compte(db_session, persona)
        await db_session.commit()
    await db_session.rollback()
