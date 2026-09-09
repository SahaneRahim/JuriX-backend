"""
Le schema que la migration 011 doit avoir pose.

Ces tests portent sur la BASE, pas sur les modeles : un modele peut declarer ce
qu'il veut, c'est PostgreSQL qui tranche. Ils attrapent le cas ou la migration
n'a pas ete rejouee, ou a diverge du modele.

Le plus important est `test_supprimer_un_compte_supprime_ses_jetons` : sans
`ON DELETE CASCADE`, supprimer un utilisateur violerait la cle etrangere et
`DELETE /admin/users/{id}` repondrait 500 des qu'un compte a demande une
reinitialisation. Cette route supprime deja les conversations explicitement,
pour une raison voisine — le defaut est donc du genre a se reproduire.

Usage:
    pytest tests/test_migrations/test_email_tokens_schema.py -v
"""

import pytest
from sqlalchemy import text

from app.models.user import User


@pytest.mark.asyncio
async def test_index_unique_sur_token_hash(db_session):
    """
    C'est l'index de RECHERCHE, pas une decoration.

    Le jeton presente est hache puis cherche ici : sans index, chaque validation
    ferait un parcours complet de la table. Et sans unicite, rien n'empecherait
    deux lignes de porter la meme empreinte.
    """
    unique = (
        await db_session.execute(
            text(
                "SELECT indexdef ~ 'UNIQUE' FROM pg_indexes "
                "WHERE tablename = 'email_tokens' AND indexname = 'ix_email_tokens_token_hash'"
            )
        )
    ).scalar_one_or_none()

    assert unique is True


@pytest.mark.asyncio
async def test_les_index_de_service_existent(db_session):
    """(user_id, purpose) sert l'etranglement ; expires_at sert la purge."""
    noms = {
        r[0]
        for r in (
            await db_session.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'email_tokens'")
            )
        ).all()
    }

    assert "ix_email_tokens_user_purpose" in noms
    assert "ix_email_tokens_expires_at" in noms


@pytest.mark.asyncio
async def test_la_contrainte_check_refuse_un_usage_invente(db_session, test_user):
    """
    Une valeur de `purpose` hors liste doit echouer A L'INSERTION.

    Sinon elle se retrouverait dans une table ou personne ne la cherche : les
    lectures filtrent toutes sur un `purpose` connu, donc la ligne serait
    invisible et le jeton correspondant, mort sans explication.
    """
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await db_session.execute(
            text(
                "INSERT INTO email_tokens (user_id, purpose, token_hash, expires_at) "
                "VALUES (:uid, 'autre_chose', 'empreinte', now())"
            ),
            {"uid": test_user.id},
        )
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_supprimer_un_compte_supprime_ses_jetons(db_session, test_user):
    """
    LE TEST LE PLUS IMPORTANT DU FICHIER. Voir le docstring du module.

    Sans ON DELETE CASCADE, cette suppression leverait une violation de cle
    etrangere et la route d'administration repondrait 500.
    """
    user_id = test_user.id
    await db_session.execute(
        text(
            "INSERT INTO email_tokens (user_id, purpose, token_hash, expires_at) "
            "VALUES (:uid, 'reset_password', 'empreinte-de-test', now() + interval '1 hour')"
        ),
        {"uid": user_id},
    )
    await db_session.commit()

    await db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})
    await db_session.commit()

    restants = (
        await db_session.execute(
            text("SELECT count(*) FROM email_tokens WHERE user_id = :uid"), {"uid": user_id}
        )
    ).scalar_one()
    assert restants == 0


def test_une_seule_tete_alembic():
    """
    Un `down_revision` mal recopie cree une seconde tete.

    `alembic upgrade head` devient alors ambigu et echoue — or le conteneur le
    lance a chaque demarrage, donc l'image ne demarrerait plus du tout.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    tetes = ScriptDirectory.from_config(Config("alembic.ini")).get_heads()

    assert len(tetes) == 1, f"plusieurs tetes alembic : {tetes}"
