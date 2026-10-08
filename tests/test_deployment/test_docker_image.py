"""
Tests du Dockerfile et du contexte de build.

Ils ne construisent pas l'image — trop lent pour une suite — mais verifient les
proprietes dont l'absence se paie en production :

- sans poppler, GET /laws/{id}/page/{n} repond 500 alors qu'il fonctionne en
  developpement, ou le binaire est installe sur la machine ;
- sans .dockerignore, le COPY . . embarque .env, donc les cles d'API, dans une
  couche de l'image ;
- sans migration au demarrage, le conteneur sert un schema incomplet : les
  colonnes search_vector, les index GIN et les declencheurs n'existent que dans
  alembic, jamais dans Base.metadata ;
- sans versions figees, l'image prend les dernieres versions du jour de sa
  construction, que la suite n'a jamais vues.
"""

import pathlib
import re
from importlib import metadata

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE_PATH = ROOT / ".dockerignore"
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
CONSTRAINTS_PATH = ROOT / "constraints.txt"
COPIE_DU_CODE = "COPY --chown=jurix:jurix . ."


def _nom(nom: str) -> str:
    """Nom de paquet normalise (PEP 503) : SQLAlchemy, sqlalchemy et SQL_Alchemy se valent."""
    return re.sub(r"[-_.]+", "-", nom).lower()


def _requirements():
    from packaging.requirements import Requirement

    lignes = (ligne.split("#")[0].strip() for ligne in REQUIREMENTS.splitlines())
    return [Requirement(ligne) for ligne in lignes if ligne]


def _contraintes() -> dict:
    lignes = CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines()
    paires = (ligne.split("==", 1) for ligne in lignes if ligne and not ligne.startswith("#"))
    return {_nom(nom): version for nom, version in paires}


class TestSystemDependencies:

    @pytest.mark.parametrize("package", [
        "poppler-utils",      # pdftoppm, dont depend pdf2image
        "tesseract-ocr",
        "tesseract-ocr-fra",
        "tesseract-ocr-eng",
    ])
    def test_package_is_installed(self, package):
        assert package in DOCKERFILE, (
            f"{package} absent du Dockerfile : la fonctionnalite qui en depend "
            f"marchera en developpement et echouera en production."
        )

    def test_tesseract_path_is_set_for_linux(self):
        assert "TESSERACT_PATH=/usr/bin/tesseract" in DOCKERFILE


class TestBuildContext:

    def test_dockerignore_exists(self):
        assert DOCKERIGNORE_PATH.is_file(), (
            "Sans .dockerignore, COPY . . embarque .env dans l'image."
        )

    @pytest.mark.parametrize("pattern", [".env", ".git", ".venv", "data/", "tests/", "deploiement/"])
    def test_excludes_what_must_not_ship(self, pattern):
        content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
        lines = {line.strip() for line in content.splitlines()}

        assert pattern in lines, f"{pattern} devrait etre exclu du contexte de build"

    def test_dockerignore_est_versionne(self):
        """
        Il etait ignore par le .gitignore (*.dockerignore) : absent de tout
        checkout et de `git archive`, il ne protegeait que la machine ou il
        avait ete ecrit. Le build sur la VM aurait embarque le .env de
        production.
        """
        import subprocess

        try:
            resultat = subprocess.run(
                ["git", "check-ignore", "-q", ".dockerignore"], cwd=ROOT, capture_output=True
            )
        except FileNotFoundError:
            pytest.skip("git absent")
        if resultat.returncode == 128:
            pytest.skip("hors d'un depot git")

        assert resultat.returncode == 1, ".dockerignore est ignore par git"

    def test_env_example_stays_included(self):
        """Le modele de configuration, lui, doit rester : il documente les cles."""
        content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")

        assert "!.env.example" in content


class TestStartup:

    def test_migrations_run_before_the_server(self):
        assert "alembic upgrade head" in DOCKERFILE
        upgrade_at = DOCKERFILE.index("alembic upgrade head")
        uvicorn_at = DOCKERFILE.index("uvicorn app.main:app", upgrade_at)
        assert upgrade_at < uvicorn_at

    def test_runs_as_a_non_root_user(self):
        assert "USER jurix" in DOCKERFILE

    def test_pas_de_journal_d_acces(self):
        """Il ecrivait le jeton d'administration du WebSocket (?token=...) en clair."""
        assert "--no-access-log" in DOCKERFILE


class TestCouches:

    def test_le_modele_fasttext_precede_le_code(self):
        """
        Telecharge apres la copie du code, le modele (126 Mo) l'etait a chaque
        mise a jour ; et un echec donnait, sans erreur, une image sans
        detection de langue.
        """
        assert DOCKERFILE.index("lid.176.bin") < DOCKERFILE.index(COPIE_DU_CODE)
        assert "skip language detection" not in DOCKERFILE

    def test_pas_de_chown_recursif_apres_le_code(self):
        """Il recopiait tout /app, modele compris, dans une couche de plus."""
        assert "chown -R" not in DOCKERFILE.split(COPIE_DU_CODE, 1)[1]


class TestVersionsFigees:
    """
    requirements.txt donne des fourchettes. Sans contraintes, l'image prenait
    les dernieres versions du jour de sa construction : SQLAlchemy 2.1, sorti
    entre deux, n'installait plus greenlet, et l'API plantait au demarrage
    alors que tout passait en developpement, en 2.0.52.
    """

    # Outils d'installation, et non code de l'application : le venv peut avoir
    # les siens (conda), que scripts/figer_versions.sh laisse libres.
    OUTILS = {"pip", "setuptools", "wheel", "packaging"}

    def test_l_image_installe_les_versions_figees(self):
        assert "COPY requirements.txt constraints.txt ./" in DOCKERFILE
        assert "-r requirements.txt -c constraints.txt" in DOCKERFILE

    def test_chaque_dependance_est_figee(self):
        contraintes = _contraintes()
        absentes = [r.name for r in _requirements() if _nom(r.name) not in contraintes]

        assert not absentes, (
            f"Absentes de constraints.txt : {absentes}. Lancer scripts/figer_versions.sh."
        )

    def test_les_versions_figees_respectent_requirements(self):
        contraintes = _contraintes()
        hors = [
            f"{r.name}=={contraintes[_nom(r.name)]} hors de {r.specifier}"
            for r in _requirements()
            if _nom(r.name) in contraintes and contraintes[_nom(r.name)] not in r.specifier
        ]

        assert not hors, f"{hors}. Lancer scripts/figer_versions.sh."

    def test_greenlet_est_installe(self):
        """sqlalchemy.ext.asyncio en a besoin, et SQLAlchemy 2.1 ne l'installe plus."""
        sqlalchemy = next(r for r in _requirements() if _nom(r.name) == "sqlalchemy")

        assert "asyncio" in sqlalchemy.extras
        assert "greenlet" in _contraintes()

    def test_la_suite_tourne_sur_les_versions_de_l_image(self):
        """Ce que les tests valident doit etre ce que l'image installe."""
        ecarts = []
        for nom, version in _contraintes().items():
            if nom in self.OUTILS:
                continue
            try:
                installee = metadata.version(nom)
            except metadata.PackageNotFoundError:
                continue
            if installee != version:
                ecarts.append(f"{nom} : venv {installee}, image {version}")

        assert not ecarts, (
            f"{ecarts}. Aligner le venv (pip install -r requirements.txt -c constraints.txt) "
            f"ou, apres une mise a jour voulue, lancer scripts/figer_versions.sh."
        )
