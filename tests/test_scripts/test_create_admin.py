"""
Tests de scripts/create_admin.py : un mot de passe refuse ne s'affiche jamais.

demarrer.sh lance ce script a chaque premier deploiement, avec ADMIN_PASSWORD
du .env de production. Un mot de passe hors politique (sans majuscule, par
exemple) faisait afficher l'erreur Pydantic entiere, valeur refusee comprise :
le mot de passe en clair dans le terminal et dans la sortie du deploiement.

Usage:
    pytest tests/test_scripts/test_create_admin.py -v
"""

import asyncio
import sys

import pytest

from scripts import create_admin

REFUSE = "sansmajuscule123"


@pytest.fixture
def lancer(monkeypatch):
    def _lancer(*arguments):
        monkeypatch.setattr(sys, "argv", ["create_admin.py", *arguments])
        return asyncio.run(create_admin.main())
    return _lancer


def test_un_mot_de_passe_refuse_n_est_jamais_affiche(lancer, capsys):
    code = lancer("--email", "admin@jurix.cm", "--password", REFUSE)

    sortie = capsys.readouterr()
    assert code == 1
    assert REFUSE not in sortie.out + sortie.err
    assert "password" in sortie.err
    assert "uppercase" in sortie.err


def test_le_refus_precede_tout_acces_a_la_base(lancer, monkeypatch):
    """La validation echoue avant d'ouvrir une session : rien n'est ecrit."""
    def interdit(*args, **kwargs):
        raise AssertionError("la base ne devait pas etre ouverte")

    monkeypatch.setattr(create_admin, "AsyncSessionLocal", interdit)

    assert lancer("--email", "admin@jurix.cm", "--password", REFUSE) == 1
