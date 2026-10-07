"""
Le choix du service de generation, fait une seule fois.

Le chat, la comparaison et l'explication d'article choisissaient chacun leur
fournisseur. Deux des trois copies utilisaient `settings` sans l'importer : la
comparaison et l'explication repondaient 500 des qu'aucune doublure n'etait
injectee — c'est-a-dire en production, jamais dans les tests. Ces tests
construisent donc les trois services SANS doublure.

Aucun appel reseau : construire un service ne genere rien.

Usage:
    pytest tests/test_services/test_llm.py -v
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services.comparison_service import ComparisonService
from app.services.explanation_service import ExplanationService
from app.services.gemini_service import (
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiService,
    get_gemini_service,
)
from app.services.llm import (
    ERREURS_LLM,
    ERREURS_QUOTA,
    ERREURS_SATURATION,
    get_llm_service,
)
from app.services.mistral_service import (
    MistralOverloadedError,
    MistralQuotaError,
    MistralService,
    get_mistral_service,
)
from app.services.rag_service import RAGService


@pytest.fixture
def fournisseur(monkeypatch):
    """Pose LLM_PROVIDER et une cle factice ; vide les singletons avant et apres."""

    def _poser(nom):
        monkeypatch.setattr(settings, "LLM_PROVIDER", nom)
        monkeypatch.setattr(settings, "MISTRAL_API_KEY", "test-key-not-a-real-credential")
        get_mistral_service.cache_clear()
        get_gemini_service.cache_clear()

    yield _poser
    get_mistral_service.cache_clear()
    get_gemini_service.cache_clear()


@pytest.mark.parametrize(
    "nom, attendu", [("mistral", MistralService), ("gemini", GeminiService)]
)
def test_le_fournisseur_suit_llm_provider(fournisseur, nom, attendu):
    fournisseur(nom)

    assert isinstance(get_llm_service(), attendu)


@pytest.mark.parametrize("nom", ["mistral", "gemini"])
@pytest.mark.parametrize("service", [ComparisonService, ExplanationService, RAGService])
def test_les_trois_services_se_construisent_sans_doublure(fournisseur, nom, service):
    """Le cas de production : aucun `llm=` injecte."""
    fournisseur(nom)

    instance = service(MagicMock())

    assert instance.llm is get_llm_service()


@pytest.mark.parametrize(
    "erreur",
    [GeminiQuotaError, MistralQuotaError, GeminiOverloadedError, MistralOverloadedError],
)
def test_quota_et_saturation_sont_aussi_des_erreurs_de_generation(erreur):
    """
    L'ordre des `except` des appelants en depend : quota et saturation d'abord,
    l'erreur generique ensuite. Si l'une cessait d'en deriver, elle remonterait
    en 500 au lieu du 429 ou du 503 attendu.
    """
    assert issubclass(erreur, ERREURS_LLM)
    assert issubclass(erreur, ERREURS_QUOTA + ERREURS_SATURATION)


@pytest.mark.parametrize("fin, attendu", [
    ("STOP", ""),
    ("UNKNOWN", ""),
    ("MAX_TOKENS", "longueur maximale atteinte"),
    ("INTERROMPUE", "connexion avec le modèle a été coupée"),
    # Une fin « error » du fournisseur n'est pas une affaire de longueur :
    # « posez une question plus precise » serait le mauvais remede.
    ("ERROR", "connexion avec le modèle a été coupée"),
])
def test_mention_selon_la_fin(fin, attendu):
    from app.services.llm import mention_de_fin

    mention = mention_de_fin(fin)

    assert (attendu in mention) if attendu else mention == ""


def test_mention_dans_la_langue_de_la_reponse():
    from app.services.llm import mention_de_fin

    assert "maximum length reached" in mention_de_fin("MAX_TOKENS", "en")
    assert "Ask again" in mention_de_fin("ERROR", "en")
    # Langue inconnue : le francais, jamais une mention vide.
    assert "longueur maximale" in mention_de_fin("MAX_TOKENS", "de")
