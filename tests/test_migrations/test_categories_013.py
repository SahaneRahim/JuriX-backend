"""
Migration 013 : 14 domaines, procedures fusionnees, Santé et Éducation.

La premiere version de 013 avait trois defauts que ces tests figent :
- un retour arriere vide (`pass`) : `alembic downgrade` annoncait un succes
  et laissait les 14 nouveaux noms en place ;
- les categories supprimees sans faire suivre `laws.suggested_categories`
  (un tableau d'identifiants, sans cle etrangere) : 5 lois de jurix_dev
  gardaient des identifiants disparus ;
- les categories cibles inserees APRES le re-pointage des lois.

Chaque test part d'un etat construit a la main, passe par Alembic, puis remet
la base de test a `head` quoi qu'il arrive.

Usage:
    pytest tests/test_migrations/test_categories_013.py -v
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from app.services.legal_domain_classifier import (
    CANONICAL_DOMAINS,
    CIVIL,
    EDUCATION,
    FAMILLE,
    FONCTION_PUBLIQUE,
    PENAL,
    SANTE,
)
from tests.conftest import TEST_DATABASE_URL

RACINE = Path(__file__).resolve().parents[2]
AVANT_013 = "c4d5e6f7a8b9"


def _charger_013():
    chemin = RACINE / "alembic" / "versions" / "013_harmonize_categories.py"
    spec = importlib.util.spec_from_file_location("migration_013", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M013 = _charger_013()


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


def _ids(session) -> dict:
    return dict(session.execute(text("SELECT name, id FROM categories")).all())


def _seme(session, noms) -> dict:
    for ordre, nom in enumerate(noms):
        session.execute(
            text("""
                INSERT INTO categories (name, description, display_order)
                SELECT :nom, :nom, :ordre
                WHERE NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:nom))
            """),
            {"nom": nom, "ordre": ordre},
        )
    session.commit()
    return _ids(session)


def _loi(session, reference, category_id=None, suggestions=None) -> int:
    return session.execute(
        text("""
            INSERT INTO laws (reference, title, type, content, language, status,
                              category_id, suggested_categories)
            VALUES (:ref, :ref, 'loi', 'Contenu.', 'fr', 'published', :cat, :sug)
            RETURNING id
        """),
        {"ref": reference, "cat": category_id, "sug": suggestions},
    ).scalar_one()


def _lire(session, law_id) -> tuple:
    """(category_id, suggested_categories) d'une loi."""
    return tuple(session.execute(
        text("SELECT category_id, suggested_categories FROM laws WHERE id = :id"),
        {"id": law_id},
    ).one())


def _etat_de_007(session) -> dict:
    """Descend avant 013, puis seme les 14 noms de 007."""
    _alembic("downgrade", AVANT_013)
    return _seme(session, [nom for nom, _, _ in M013.CANONICAL_007])


def test_les_noms_de_la_migration_sont_ceux_du_code():
    """Le classement ecrit des NOMS : la table et le code doivent s'accorder."""
    noms = tuple(nom for nom, _, _ in M013.CANONICAL_CATEGORIES)

    assert noms == CANONICAL_DOMAINS


class TestMontee:
    def test_depuis_l_etat_de_007(self, session):
        avant = _etat_de_007(session)
        civil = _loi(session, "L-CIVIL", avant["Droit Civil"])
        procedure = _loi(session, "L-PROC", avant["Procédure Pénale"])
        famille = _loi(session, "L-FAM", avant["Droit de la Famille"])
        # Fusion, doublon cree par la fusion, et identifiant deja disparu.
        sugg_civil = _loi(session, "L-S1", None, [
            avant["Procédure Civile"], avant["Droit Civil"], 99999, avant["Droit Civil"],
        ])
        sugg_penal = _loi(session, "L-S2", None, [
            avant["Droit Pénal"], avant["Procédure Pénale"], avant["Fonction Publique"],
        ])
        session.commit()

        _alembic("upgrade", "head")
        session.expire_all()
        apres = _ids(session)

        assert set(apres) == set(CANONICAL_DOMAINS)
        # Renommages SUR PLACE : l'identifiant est celui d'avant.
        assert apres[CIVIL] == avant["Droit Civil"]
        assert apres[PENAL] == avant["Droit Pénal"]
        assert apres[FAMILLE] == avant["Droit de la Famille"]
        assert _lire(session, civil)[0] == apres[CIVIL]
        assert _lire(session, famille)[0] == apres[FAMILLE]
        # Fusion : la loi de procedure rejoint le domaine de fond.
        assert _lire(session, procedure)[0] == apres[PENAL]
        # Suggestions : suivies, dedoublonnees, ordre garde, orphelin retire.
        assert _lire(session, sugg_civil)[1] == [apres[CIVIL]]
        assert _lire(session, sugg_penal)[1] == [apres[PENAL], apres[FONCTION_PUBLIQUE]]

    def test_ordre_icones_et_descriptions(self, session):
        _etat_de_007(session)

        _alembic("upgrade", "head")
        lignes = session.execute(text(
            "SELECT name, icon, description FROM categories ORDER BY display_order"
        )).all()

        assert tuple(tuple(ligne) for ligne in lignes) == M013.CANONICAL_CATEGORIES

    def test_quand_la_cible_existe_deja(self, session):
        """
        Le renommage est saute, la ligne source devient une fusion : ses lois
        rejoignent la cible au lieu de passer a NULL.
        """
        avant = _etat_de_007(session)
        cible = _seme(session, [CIVIL])[CIVIL]
        loi = _loi(session, "L-CIVIL", avant["Droit Civil"], [avant["Droit Civil"]])
        session.commit()

        _alembic("upgrade", "head")
        session.expire_all()

        assert set(_ids(session)) == set(CANONICAL_DOMAINS)
        assert _lire(session, loi) == (cible, [cible])

    def test_est_idempotente(self, session):
        """Rejouee sur une base deja migree, la montee ne change rien."""
        avant = _etat_de_007(session)
        loi = _loi(session, "L-PROC", avant["Procédure Pénale"], [avant["Procédure Pénale"]])
        session.commit()
        _alembic("upgrade", "head")
        session.expire_all()
        premiere = (_ids(session), _lire(session, loi))

        # `stamp` recule la version sans rien executer : 013 repasse en entier.
        _alembic("stamp", AVANT_013)
        _alembic("upgrade", "head")
        session.expire_all()

        assert (_ids(session), _lire(session, loi)) == premiere


class TestDescente:
    def test_defait_013(self, session):
        ids = _seme(session, CANONICAL_DOMAINS)
        sante = _loi(session, "L-SANTE", ids[SANTE], [ids[SANTE], ids[CIVIL]])
        civil = _loi(session, "L-CIVIL", ids[CIVIL], [ids[EDUCATION], ids[CIVIL]])
        session.commit()

        _alembic("downgrade", AVANT_013)
        session.expire_all()
        apres = _ids(session)

        assert set(apres) == {nom for nom, _, _ in M013.CANONICAL_007}
        # Santé n'existe pas en 007 : la loi passe a NULL, rien n'est invente.
        assert _lire(session, sante) == (None, [ids[CIVIL]])
        # Renommage inverse, sur place.
        assert apres["Droit Civil"] == ids[CIVIL]
        assert _lire(session, civil) == (ids[CIVIL], [ids[CIVIL]])
        # Les procedures reviennent, vides : la perte est documentee.
        for procedure in M013.PROCEDURES:
            assert session.execute(
                text("SELECT count(*) FROM laws WHERE category_id = :id"),
                {"id": apres[procedure]},
            ).scalar_one() == 0
        descriptions = dict(session.execute(
            text("SELECT name, description FROM categories")
        ).all())
        assert descriptions == {nom: desc for nom, _, desc in M013.CANONICAL_007}

    def test_aller_retour(self, session):
        ids = _seme(session, CANONICAL_DOMAINS)
        loi = _loi(session, "L-PENAL", ids[PENAL], [ids[PENAL], ids[FAMILLE]])
        session.commit()

        _alembic("downgrade", AVANT_013)
        _alembic("upgrade", "head")
        session.expire_all()
        apres = _ids(session)

        assert set(apres) == set(CANONICAL_DOMAINS)
        assert apres[PENAL] == ids[PENAL]
        assert _lire(session, loi) == (ids[PENAL], [ids[PENAL], ids[FAMILLE]])


def test_alembic_sans_database_url_echoue_et_dit_quoi_faire():
    """
    Le repli sur le .env faisait migrer en silence la base qu'il designait.
    Sans DATABASE_URL, la commande doit echouer, et nommer la variable.
    """
    environnement = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}

    resultat = subprocess.run(
        [sys.executable, "-m", "alembic", "current"],
        cwd=RACINE,
        env=environnement,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert resultat.returncode != 0
    assert "DATABASE_URL absente" in resultat.stderr
    assert "alembic upgrade head" in resultat.stderr
