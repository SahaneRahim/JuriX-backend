"""
GroqService : limiteur, 429, quota du jour, pannes, JSON strict.

Tout passe par un httpx.MockTransport et une horloge factice : aucun appel
reseau, aucune vraie attente. Ces tests tournent sans base.

Usage:
    pytest tests/test_services/test_groq_service.py -v
"""

import json

import httpx
import pytest

from app.core.limiteur import Limiteur
from app.services import groq_service
from app.services.groq_service import (
    GroqIndisponibleError,
    GroqLimiteError,
    GroqQuotaError,
    GroqReponseInvalideError,
    GroqService,
    GroqServiceError,
    ReponseGroq,
)

QWEN = "qwen/qwen3.8-27b"
GPT_OSS = "openai/gpt-oss-120b"
SCHEMA = {
    "type": "object",
    "properties": {"intention": {"type": "string", "enum": ["juridique", "smalltalk"]}},
    "required": ["intention"],
    "additionalProperties": False,
}


class Horloge:
    def __init__(self):
        self.t = 0.0
        self.sommeils = []

    def __call__(self):
        return self.t

    def dormir(self, secondes):
        self.sommeils.append(round(secondes, 3))
        self.t += secondes

    async def dormir_async(self, secondes):
        self.dormir(secondes)


class FauxGroq:
    """Repond dans l'ordre les reponses programmees, et garde les requetes recues."""

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.requetes = []

    def __call__(self, requete: httpx.Request) -> httpx.Response:
        self.requetes.append(requete)
        reponse = self.reponses.pop(0) if len(self.reponses) > 1 else self.reponses[0]
        if isinstance(reponse, Exception):
            raise reponse
        return reponse

    def corps(self, i=-1):
        return json.loads(self.requetes[i].content)


def ok(contenu, fin="stop", jetons=120, entree=100, entetes=None):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(contenu)}, "finish_reason": fin}],
            "usage": {"total_tokens": jetons, "prompt_tokens": entree},
        },
        headers=entetes or {},
    )


def refus(code, message="", entetes=None):
    return httpx.Response(code, json={"error": {"message": message}}, headers=entetes or {})


@pytest.fixture
def horloge():
    return Horloge()


@pytest.fixture
def groq(monkeypatch, horloge):
    """Un service sur faux transport, avec des limiteurs sur horloge factice."""
    limiteurs = {}

    def limiteur_pour(modele):
        if modele not in limiteurs:
            limiteurs[modele] = Limiteur(
                modele, requetes_par_minute=30, rafale=5, jetons_par_minute=8000,
                jetons_entree_par_minute=7000, requetes_par_jour=1000, jetons_par_jour=200_000,
                horloge=horloge, dormir=horloge.dormir, dormir_async=horloge.dormir_async,
            )
        return limiteurs[modele]

    def _poser(*reponses):
        faux = FauxGroq(*reponses)
        monkeypatch.setattr(groq_service, "transport_http", httpx.MockTransport(faux))
        service = GroqService(api_key="cle-test", model_name=QWEN, limiteur_pour=limiteur_pour)
        service.limiteurs = limiteurs
        return service, faux

    return _poser


async def _completer(service, **reglages):
    return await service.completer_json(
        systeme="Classe.", message="bonjour", schema=SCHEMA, **reglages
    )


class TestRequete:
    async def test_json_strict_et_reponse(self, groq):
        service, faux = groq(ok({"intention": "smalltalk"}, jetons=130))

        reponse = await _completer(service, max_jetons=50)

        assert reponse == ReponseGroq({"intention": "smalltalk"}, 130, QWEN)
        corps = faux.corps()
        assert corps["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "reponse", "strict": True, "schema": SCHEMA},
        }
        assert corps["max_completion_tokens"] == 50
        assert corps["temperature"] == 0
        assert faux.requetes[0].headers["authorization"] == "Bearer cle-test"

    async def test_qwen_sans_reflexion(self, groq):
        service, faux = groq(ok({"intention": "juridique"}))

        await _completer(service)

        assert faux.corps()["reasoning_effort"] == "none"
        assert "include_reasoning" not in faux.corps()

    async def test_gpt_oss_reflexion_reduite_et_masquee(self, groq):
        service, faux = groq(ok({"intention": "juridique"}))

        await _completer(service, modele=GPT_OSS)

        assert faux.corps()["model"] == GPT_OSS
        assert faux.corps()["reasoning_effort"] == "low"
        assert faux.corps()["include_reasoning"] is False

    async def test_sans_schema_json_libre(self, groq):
        service, faux = groq(ok({"x": 1}))

        await service.completer_json(systeme="s", message="m")

        assert faux.corps()["response_format"] == {"type": "json_object"}

    def test_chemin_synchrone(self, groq):
        service, faux = groq(ok({"intention": "juridique"}))

        reponse = service.completer_json_sync(systeme="s", message="m", schema=SCHEMA)

        assert reponse.donnees == {"intention": "juridique"}
        assert len(faux.requetes) == 1


class Test429:
    async def test_retry_after_respecte_puis_rejoue(self, groq, horloge):
        service, faux = groq(
            refus(429, "Rate limit reached for requests per minute (RPM)", {"retry-after": "7"}),
            ok({"intention": "juridique"}),
        )

        reponse = await _completer(service)

        assert reponse.donnees == {"intention": "juridique"}
        assert len(faux.requetes) == 2
        assert horloge.sommeils == [7.0]

    def test_rejoue_aussi_en_synchrone(self, groq, horloge):
        service, faux = groq(refus(429, "RPM", {"retry-after": "3"}), ok({"intention": "x"}))

        service.completer_json_sync(systeme="s", message="m")

        assert len(faux.requetes) == 2
        assert horloge.sommeils == [3.0]

    async def test_quota_du_jour_ferme_le_modele(self, groq):
        service, faux = groq(refus(
            429,
            "Rate limit reached for model `qwen/qwen3.8-27b` on tokens per day (TPD): "
            "Limit 200000, Used 199900, Requested 300. Please try again in 7m12s.",
            {"retry-after": "432"},
        ))

        with pytest.raises(GroqQuotaError) as erreur:
            await _completer(service)
        assert erreur.value.retry_after == pytest.approx(432)

        # Coupe-circuit : l'appel suivant echoue SANS requete.
        with pytest.raises(GroqQuotaError):
            await _completer(service)
        assert len(faux.requetes) == 1

    async def test_attente_trop_longue_vaut_quota(self, groq):
        service, faux = groq(refus(429, "RPM", {"retry-after": "600"}))

        with pytest.raises(GroqQuotaError):
            await _completer(service)

    async def test_429_persistants(self, groq):
        service, faux = groq(refus(429, "RPM", {"retry-after": "1"}))

        with pytest.raises(GroqLimiteError) as erreur:
            await _completer(service)

        assert not isinstance(erreur.value, GroqQuotaError)
        assert len(faux.requetes) == groq_service.ESSAIS

    async def test_le_quota_ne_retarde_pas_les_autres_modeles(self, groq):
        """Chez Groq, les quotas sont PAR MODELE."""
        service, faux = groq(
            refus(429, "tokens per day (TPD)", {"retry-after": "3000"}),
            ok({"intention": "juridique"}),
        )
        with pytest.raises(GroqQuotaError):
            await _completer(service)

        reponse = await _completer(service, modele=GPT_OSS)

        assert reponse.modele == GPT_OSS


class TestLimiteur:
    async def test_attente_au_dela_du_maximum_sans_requete(self, groq):
        service, faux = groq(ok({"intention": "juridique"}))
        service.limiteur(QWEN).repousser(5, "429")

        with pytest.raises(GroqLimiteError) as erreur:
            await _completer(service, attente_max=0.5)

        assert erreur.value.retry_after == pytest.approx(5)
        assert faux.requetes == []

    async def test_entetes_requetes_du_jour_epuisees(self, groq):
        service, faux = groq(ok(
            {"intention": "juridique"},
            entetes={"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m30s"},
        ))
        await _completer(service)

        with pytest.raises(GroqLimiteError) as erreur:
            await _completer(service, attente_max=60)

        assert erreur.value.retry_after == pytest.approx(150)
        assert len(faux.requetes) == 1

    async def test_usage_reel_corrige_la_reservation(self, groq):
        """L'estimation (entree + max_jetons) est remplacee par l'usage reel."""
        service, faux = groq(ok({"intention": "juridique"}, jetons=200))
        lim = service.limiteur(QWEN)

        await _completer(service, max_jetons=7000)

        assert lim._jetons_minute.total(lim.maintenant()) == 200


class TestPannes:
    async def test_5xx_puis_succes(self, groq, horloge):
        service, faux = groq(refus(503, "over capacity"), ok({"intention": "juridique"}))

        reponse = await _completer(service)

        assert reponse.donnees == {"intention": "juridique"}
        assert horloge.sommeils == [groq_service.DELAI_5XX_S]

    async def test_5xx_persistants(self, groq):
        service, faux = groq(refus(503, "over capacity"))

        with pytest.raises(GroqIndisponibleError):
            await _completer(service)

        assert len(faux.requetes) == groq_service.ESSAIS

    async def test_reseau(self, groq):
        service, faux = groq(httpx.ConnectError("refused"))

        with pytest.raises(GroqIndisponibleError):
            await _completer(service)

    async def test_400_non_rejoue(self, groq):
        service, faux = groq(refus(400, "json_schema not supported"))

        with pytest.raises(GroqServiceError) as erreur:
            await _completer(service)

        assert "json_schema not supported" in str(erreur.value)
        assert len(faux.requetes) == 1

    async def test_reponse_coupee(self, groq):
        service, faux = groq(ok({"intention": "juri"}, fin="length"))

        with pytest.raises(GroqReponseInvalideError):
            await _completer(service)

    async def test_json_invalide(self, groq):
        service, faux = groq(httpx.Response(
            200, json={"choices": [{"message": {"content": "pas du json"}, "finish_reason": "stop"}]}
        ))

        with pytest.raises(GroqReponseInvalideError):
            await _completer(service)


class TestSante:
    async def test_sans_generation_et_en_cache(self, groq):
        service, faux = groq(httpx.Response(200, json={"id": QWEN}))

        premier = await service.health_check()
        second = await service.health_check()

        assert premier == second == {"status": "healthy", "model": QWEN}
        assert len(faux.requetes) == 1
        assert faux.requetes[0].method == "GET"
        assert faux.requetes[0].url.path.endswith(f"/models/{QWEN}")

    async def test_cle_refusee(self, groq):
        service, faux = groq(refus(401, "Invalid API Key"))

        sante = await service.health_check()

        assert sante["status"] == "unhealthy"
        assert "Invalid API Key" in sante["error"]
