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
    """
    La validation echoue avant d'ouvrir une session : rien n'est ecrit.

    Les appels sont COMPTES : lever une erreur depuis la doublure ne prouvait
    rien, le `except Exception` du script l'avalait et rendait le meme code 1.
    """
    appels = []

    def interdit(*args, **kwargs):
        appels.append(1)
        raise AssertionError("la base ne devait pas etre ouverte")

    monkeypatch.setattr(create_admin, "AsyncSessionLocal", interdit)

    assert lancer("--email", "admin@jurix.cm", "--password", REFUSE) == 1
    assert appels == []


VALIDE = "Valide-Essai-42"


class _CompteExistant:
    def __init__(self, role):
        self.role, self.is_active, self.hashed_password = role, True, "hachage-d-origine"


@pytest.fixture
def base_avec(monkeypatch):
    """Une base dont l'adresse demandee appartient deja a un compte de ce role."""
    def installer(role):
        compte = _CompteExistant(role)
        commits = []

        class Resultat:
            def scalar_one_or_none(self):
                return compte

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, *args, **kwargs):
                return Resultat()

            async def commit(self):
                commits.append(1)

        monkeypatch.setattr(create_admin, "AsyncSessionLocal", Session)
        monkeypatch.setattr(create_admin, "hash_password", lambda mdp: f"hache:{mdp}")
        return compte, commits
    return installer


def test_un_compte_ordinaire_n_est_jamais_promu_en_silence(lancer, base_avec, capsys):
    """
    Le site est public avant l'etape 6 de demarrer.sh : quiconque s'inscrivait
    sous ADMIN_EMAIL devenait superadmin, avec son propre mot de passe.
    """
    compte, commits = base_avec("user")

    code = lancer("--email", "admin@jurix.cm", "--password", VALIDE)

    assert code == 1
    assert compte.role == "user"
    assert commits == []
    assert "--promouvoir" in capsys.readouterr().err


def test_promouvoir_remplace_le_mot_de_passe(lancer, base_avec):
    compte, commits = base_avec("user")

    code = lancer("--email", "admin@jurix.cm", "--password", VALIDE, "--promouvoir")

    assert code == 0
    assert compte.role == "superadmin"
    assert compte.hashed_password == f"hache:{VALIDE}"
    assert commits == [1]


def test_un_administrateur_existant_garde_son_mot_de_passe(lancer, base_avec):
    """create_admin.py ne remet jamais un mot de passe sans --reset-password."""
    compte, _ = base_avec("admin")

    assert lancer("--email", "admin@jurix.cm", "--password", VALIDE) == 0
    assert compte.role == "superadmin"
    assert compte.hashed_password == "hachage-d-origine"
