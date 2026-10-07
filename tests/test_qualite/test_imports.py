"""
Chaque fichier importe ce qu'il utilise : une garde permanente.

La bascule vers Mistral avait ajoute `settings.LLM_PROVIDER` a la comparaison
et a l'explication d'article sans importer `settings`. Rien n'echouait a
l'import : le nom n'etait resolu qu'a l'appel, et chaque `/compare` et chaque
`/explain-article` repondait 500. ruff le signalait (F821), mais personne ne
lancait ruff. Ces tests le lancent, et importent chaque module de `app` : un
import casse ou circulaire echoue ici, pas au premier appel en production.

Ils tournent sans base et sans reseau.

Usage:
    pytest tests/test_qualite/test_imports.py -v
"""

import importlib
import importlib.util
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import app

RACINE = Path(__file__).resolve().parents[2]
DOSSIERS_CONTROLES = ["app", "scripts", "tests", "alembic"]


def test_ruff_ne_signale_rien():
    """Noms inconnus, imports inutiles ou mal ordonnes : zero, partout."""
    if importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff n'est pas installe dans cet environnement")

    resultat = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *DOSSIERS_CONTROLES,
         "--output-format", "concise", "--no-cache"],
        cwd=RACINE,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert resultat.returncode == 0, (
        "ruff signale des defauts (corriger, ou `ruff check --fix` pour les "
        "imports) :\n" + resultat.stdout + resultat.stderr
    )


def _modules_de_app():
    return sorted(
        module.name
        for module in pkgutil.walk_packages(app.__path__, prefix="app.")
    )


def test_le_parcours_trouve_les_modules():
    """Sans ce plancher, un parcours vide ferait passer le test suivant."""
    modules = _modules_de_app()

    assert "app.services.llm" in modules
    assert len(modules) > 50


@pytest.mark.parametrize("nom", _modules_de_app())
def test_chaque_module_de_app_s_importe(nom):
    importlib.import_module(nom)
