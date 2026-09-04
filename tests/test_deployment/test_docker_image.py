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
  alembic, jamais dans Base.metadata.
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE_PATH = ROOT / ".dockerignore"


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

    @pytest.mark.parametrize("pattern", [".env", ".git", ".venv", "data/", "tests/"])
    def test_excludes_what_must_not_ship(self, pattern):
        content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
        lines = {line.strip() for line in content.splitlines()}

        assert pattern in lines, f"{pattern} devrait etre exclu du contexte de build"

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
