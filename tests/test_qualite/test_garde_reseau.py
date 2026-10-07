"""
La garde reseau de la suite : aucun appel reel a Groq ni a Mistral.

Avant elle, une suite complete envoyait de vrais appels a Groq (classement
des lois dans le pipeline, routes du classifieur) avec la cle du .env. Le
quota gratuit — 200 000 jetons par jour — ne la supportait pas : les derniers
tests tombaient en 429, et le quota du chat etait epuise pour la journee.

Ces tests verifient la garde elle-meme. Ils vident la liste des appels
refuses avant de rendre la main : sinon le demontage de la garde ferait
echouer le test, ce qui est precisement son role.

Usage:
    pytest tests/test_qualite/test_garde_reseau.py -v
"""

import os

import pytest

from app.core.config import settings
from app.services import groq_service, mistral_service
from app.services.groq_service import GroqService
from app.services.mistral_service import MistralService
from tests.conftest import CLE_FACTICE

# JURIX_E2E=1 ouvre tout, vraies cles comprises : il n'y a alors aucune garde
# a verifier, et ces tests echouaient sur un transport absent.
pytestmark = pytest.mark.skipif(
    os.environ.get("JURIX_E2E") == "1", reason="JURIX_E2E : aucune garde posee"
)


def _garde(module):
    """Le gardien derriere le transport pose par conftest."""
    return module.transport_http.handler.__self__


@pytest.mark.parametrize("cle", ["GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"])
def test_les_cles_sont_factices(cle):
    assert getattr(settings, cle) == CLE_FACTICE


async def test_un_appel_groq_reel_est_refuse():
    garde = _garde(groq_service)

    # Le classement d'intention avale toute Exception : l'echec de la garde,
    # lui, n'en est pas une, et traverse.
    with pytest.raises(pytest.fail.Exception, match="groq_live"):
        await GroqService(api_key="x").completer_json(systeme="s", message="bonjour")

    assert garde.appels == ["POST https://api.groq.com/openai/v1/chat/completions"]
    garde.appels.clear()


def test_un_appel_groq_synchrone_est_refuse():
    garde = _garde(groq_service)

    with pytest.raises(pytest.fail.Exception, match="groq_live"):
        GroqService(api_key="x").completer_json_sync(systeme="s", message="Code minier")

    assert len(garde.appels) == 1
    garde.appels.clear()


async def test_un_appel_mistral_reel_est_refuse():
    garde = _garde(mistral_service)

    with pytest.raises(pytest.fail.Exception, match="mistral_live"):
        await MistralService(api_key="x").generate("bonjour")

    assert garde.appels == ["POST https://api.mistral.ai/v1/chat/completions"]
    garde.appels.clear()


@pytest.mark.groq_live
def test_un_test_groq_live_leve_la_garde_de_groq_seulement():
    """Ignore sans `-m groq_live` ; avec, la garde de Mistral reste en place."""
    assert groq_service.transport_http is None
    assert mistral_service.transport_http is not None
    assert settings.GROQ_API_KEY != CLE_FACTICE
