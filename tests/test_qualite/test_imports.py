"""
Chaque fichier importe ce qu'il utilise : une garde permanente.

La bascule vers Mistral avait ajoute `settings.LLM_PROVIDER` a la comparaison
et a l'explication d'article sans importer `settings`. Rien n'echouait a
l'import : le nom n'etait resolu qu'a l'appel, et chaque `/compare` et chaque
`/explain-article` repondait 500. ruff le signalait (F821), mais personne ne
lancait ruff. Ces tests le lancent, et importent chaque module de `app` : un
import casse ou circulaire echoue ici, pas au premier appel en production.

Les modules que tests/conftest.py charge deja n'ont pas de cas ici (voir
DEJA_CHARGES_PAR_LE_CONFTEST) : un import casse y fait deja echouer toute la
suite, et les reimporter ne ferait que les relire dans sys.modules.

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


# Charges par tests/conftest.py avant le premier test : app.main et tout ce
# qu'il importe, plus groq_service, que la garde reseau (_garde_reseau_llm)
# importe a la mise en place de chaque test. import_module les rendrait depuis
# sys.modules sans rien executer : ces cas ne pouvaient pas echouer.
DEJA_CHARGES_PAR_LE_CONFTEST = frozenset({
    "app.api", "app.api.routes", "app.api.routes.admin", "app.api.routes.analytics",
    "app.api.routes.articles", "app.api.routes.auth", "app.api.routes.batch_upload",
    "app.api.routes.blog", "app.api.routes.categories", "app.api.routes.classifier",
    "app.api.routes.comparison", "app.api.routes.controle", "app.api.routes.corpus",
    "app.api.routes.language", "app.api.routes.laws", "app.api.routes.ocr",
    "app.api.routes.personas", "app.api.routes.rag", "app.api.routes.search",
    "app.api.routes.upload",
    "app.core", "app.core.auth", "app.core.config", "app.core.database", "app.core.dependencies",
    "app.core.google_identity", "app.core.limiteur", "app.core.mesure_sql", "app.core.personas",
    "app.main",
    "app.models", "app.models.blog", "app.models.conversation", "app.models.email_token",
    "app.models.law", "app.models.user",
    "app.services", "app.services.article_reference", "app.services.blog_generator_service",
    "app.services.category_service", "app.services.chiffres_du_corpus",
    "app.services.comparison_service", "app.services.controle_du_chat",
    "app.services.document_storage", "app.services.email_service", "app.services.email_templates",
    "app.services.email_tokens_service", "app.services.embedding_service",
    "app.services.explanation_service", "app.services.file_upload_service",
    "app.services.gemini_service", "app.services.groq_service", "app.services.intent_classifier",
    "app.services.language_detector", "app.services.lecture_des_citations",
    "app.services.legal_domain_classifier", "app.services.liens_entre_textes", "app.services.llm",
    "app.services.mistral_service", "app.services.ocr_service", "app.services.persona_service",
    "app.services.postgres_search_service", "app.services.prompts",
    "app.services.questions_de_reference", "app.services.rag_service",
    "app.services.recherche_affinee", "app.services.reperes_textes", "app.services.reranker",
    "app.services.search_service", "app.services.search_vectors",
    "app.services.statut_des_sources", "app.services.statut_en_bref",
    "app.services.statut_juridique", "app.services.text_features",
    "app.services.textes_successifs", "app.services.user_identity",
    "app.tasks", "app.tasks.process_law",
    "app.utils", "app.utils.chunk_refiner", "app.utils.file_utils", "app.utils.ocr_utils",
    "app.utils.text_chunker", "app.utils.titre_officiel",
})


@pytest.mark.parametrize(
    "nom", [nom for nom in _modules_de_app() if nom not in DEJA_CHARGES_PAR_LE_CONFTEST]
)
def test_chaque_module_de_app_s_importe(nom):
    importlib.import_module(nom)
