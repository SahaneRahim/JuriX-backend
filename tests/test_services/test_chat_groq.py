"""
Cascade de redaction du chat : Groq d'abord, puis le fournisseur habituel.

Tout passe par un httpx.MockTransport et une horloge factice : aucun appel
reseau. Ces tests tournent sans base.
"""

import json

import httpx
import pytest

from app.core.config import settings
from app.core.limiteur import AttenteTropLongue, Limiteur
from app.services import groq_service
from app.services.chat_groq import (
    ChatEnCascade,
    RedactionGroq,
    service_du_chat,
    tient_dans_la_minute,
)

QWEN = "qwen/qwen3.8-27b"
GPT_OSS = "openai/gpt-oss-120b"


class Horloge:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def dormir(self, s):
        self.t += s

    async def dormir_async(self, s):
        self.t += s


class FauxGroq:
    """Une reponse programmee par modele ; garde les requetes recues."""

    def __init__(self, reponses):
        self.reponses = reponses
        self.requetes = []

    def __call__(self, requete):
        corps = json.loads(requete.content)
        self.requetes.append(corps)
        reponse = self.reponses[corps["model"]]
        if isinstance(reponse, Exception):
            raise reponse
        return reponse


def ok(texte, fin="stop"):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": texte}, "finish_reason": fin}],
        "usage": {"total_tokens": 900, "prompt_tokens": 700},
    })


def flux(*morceaux, fin="stop"):
    lignes = [
        "data: " + json.dumps({"choices": [{"delta": {"content": m}, "finish_reason": None}]})
        for m in morceaux
    ]
    lignes.append("data: " + json.dumps({
        "choices": [{"delta": {}, "finish_reason": fin}],
        "x_groq": {"usage": {"total_tokens": 900, "prompt_tokens": 700}},
    }))
    lignes.append("data: [DONE]")
    return httpx.Response(200, content=("\n\n".join(lignes) + "\n\n").encode(),
                          headers={"content-type": "text/event-stream"})


def refus(code, message="refus", entetes=None):
    return httpx.Response(code, json={"error": {"message": message}}, headers=entetes or {})


class FauxRepli:
    """Le service de LLM_PROVIDER : on verifie seulement qu'il est appele."""

    def __init__(self):
        self.appels = []

    async def generate(self, **kwargs):
        self.appels.append(("generate", kwargs))
        return {"response": "réponse de repli", "fin": "STOP"}

    async def generate_stream(self, **kwargs):
        self.appels.append(("generate_stream", kwargs))
        if kwargs.get("fin"):
            kwargs["fin"]("STOP")
        yield "repli"

    async def health_check(self):
        return {"status": "healthy"}


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_GROQ_MAX_JETONS", 2000)
    monkeypatch.setattr(settings, "CHAT_GROQ_ATTENTE_MAX_S", 3.0)
    monkeypatch.setattr(settings, "GROQ_JETONS_PAR_MINUTE", 8000)
    monkeypatch.setattr(settings, "GROQ_JETONS_ENTREE_PAR_MINUTE", 7000)
    horloge = Horloge()
    limiteurs = {}

    def limiteur_pour(modele):
        if modele not in limiteurs:
            limiteurs[modele] = Limiteur(
                modele, requetes_par_minute=30, rafale=5, jetons_par_minute=8000,
                jetons_entree_par_minute=7000, requetes_par_jour=1000, jetons_par_jour=200_000,
                horloge=horloge, dormir=horloge.dormir, dormir_async=horloge.dormir_async,
            )
        return limiteurs[modele]

    def _poser(reponses, modeles=(QWEN, GPT_OSS)):
        faux = FauxGroq(reponses)
        monkeypatch.setattr(groq_service, "transport_http", httpx.MockTransport(faux))
        repli = FauxRepli()
        service = ChatEnCascade(
            repli, list(modeles),
            redaction_pour=lambda m: RedactionGroq(m, api_key="cle-test", limiteur_pour=limiteur_pour),
        )
        service.limiteurs = limiteurs
        return service, faux, repli

    return _poser


async def _lire(service, prompt="Question ?", system="Tu es JuriX."):
    fins = []
    morceaux = [m async for m in service.generate_stream(
        prompt=prompt, system=system, fin=fins.append, max_tokens=8192,
    )]
    return "".join(morceaux), fins


class TestGenerate:
    async def test_le_premier_modele_repond(self, cascade):
        service, faux, repli = cascade({QWEN: ok("Oui, article 309.")})

        reponse = await service.generate(prompt="Q", system="S", max_tokens=8192)

        assert reponse["response"] == "Oui, article 309."
        assert reponse["modele"] == QWEN
        assert not repli.appels
        corps = faux.requetes[0]
        assert corps["max_completion_tokens"] == 2000
        assert corps["reasoning_format"] == "hidden"

    async def test_un_429_passe_au_modele_suivant(self, cascade):
        service, faux, repli = cascade({
            QWEN: refus(429, "Rate limit", {"retry-after": "30"}),
            GPT_OSS: ok("Réponse gpt-oss."),
        })

        reponse = await service.generate(prompt="Q", system="S")

        assert reponse["modele"] == GPT_OSS
        assert faux.requetes[1]["include_reasoning"] is False
        # Le 429 est retenu : le prochain appel ne retente pas qwen tout de suite.
        with pytest.raises(AttenteTropLongue):
            service.limiteurs[QWEN].reserver(10, 3.0)

    async def test_tout_groq_refuse_donne_le_repli(self, cascade):
        service, _, repli = cascade({QWEN: refus(413, "Request too large"), GPT_OSS: refus(503)})

        reponse = await service.generate(prompt="Q", system="S", max_tokens=8192)

        assert reponse["response"] == "réponse de repli"
        assert repli.appels[0][1]["max_tokens"] == 8192

    async def test_un_prompt_trop_long_n_est_pas_envoye(self, cascade):
        service, faux, repli = cascade({QWEN: ok("jamais"), GPT_OSS: ok("jamais")})

        await service.generate(prompt="x" * 30_000, system="S")

        assert faux.requetes == []
        assert repli.appels

    async def test_une_demande_json_va_au_repli(self, cascade):
        service, faux, repli = cascade({QWEN: ok("{}")})

        await service.generate(prompt="Q", system="S", response_mime_type="application/json")

        assert faux.requetes == []
        assert repli.appels

    async def test_une_reponse_coupee_le_dit(self, cascade):
        service, _, _ = cascade({QWEN: ok("Début", fin="length")})

        reponse = await service.generate(prompt="Q", system="S")

        assert reponse["tronquee"] is True
        assert reponse["fin"] == "MAX_TOKENS"


class TestFlux:
    async def test_flux_du_premier_modele(self, cascade):
        service, _, repli = cascade({QWEN: flux("Oui", ", article 309.")})

        texte, fins = await _lire(service)

        assert texte == "Oui, article 309."
        assert fins == ["STOP"]
        assert not repli.appels

    async def test_un_refus_avant_le_premier_morceau_passe_au_suivant(self, cascade):
        service, _, repli = cascade({QWEN: refus(429, "tokens per day (TPD)"), GPT_OSS: flux("gpt-oss")})

        texte, fins = await _lire(service)

        assert texte == "gpt-oss"
        assert fins == ["STOP"]

    async def test_une_panne_reseau_avant_le_premier_morceau_passe_au_suivant(self, cascade):
        service, _, repli = cascade({QWEN: httpx.ConnectError("coupe"), GPT_OSS: refus(500)})

        texte, fins = await _lire(service)

        assert texte == "repli"
        assert repli.appels[0][0] == "generate_stream"
        assert fins == ["STOP"]

    async def test_flux_coupe_par_la_longueur(self, cascade):
        service, _, _ = cascade({QWEN: flux("Début", fin="length")})

        texte, fins = await _lire(service)

        assert texte == "Début"
        assert fins == ["MAX_TOKENS"]


class TestBranchement:
    def test_un_double_de_test_n_est_pas_enveloppe(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "cle")
        double = FauxRepli()

        assert service_du_chat(double) is double

    def test_sans_modele_groq_le_service_reste_le_meme(self, monkeypatch):
        from app.services.mistral_service import MistralService

        monkeypatch.setattr(settings, "GROQ_API_KEY", "cle")
        monkeypatch.setattr(settings, "CHAT_GROQ_MODELES", "")
        mistral = MistralService(api_key="cle-test")

        assert service_du_chat(mistral) is mistral

    def test_mistral_est_enveloppe_quand_groq_est_configure(self, monkeypatch):
        from app.services.mistral_service import MistralService

        monkeypatch.setattr(settings, "GROQ_API_KEY", "cle")
        monkeypatch.setattr(settings, "CHAT_GROQ_MODELES", f"{QWEN},{GPT_OSS}")
        mistral = MistralService(api_key="cle-test")

        service = service_du_chat(mistral)

        assert isinstance(service, ChatEnCascade)
        assert service.modeles == [QWEN, GPT_OSS]
        assert service.repli is mistral
        # Ce qui n'est pas la redaction reste au service de repli.
        assert service.limiteur.__self__ is mistral

    def test_tient_dans_la_minute(self, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_JETONS_PAR_MINUTE", 8000)
        monkeypatch.setattr(settings, "GROQ_JETONS_ENTREE_PAR_MINUTE", 7000)

        assert tient_dans_la_minute(5000, 2000)
        assert not tient_dans_la_minute(6500, 2000)
        assert not tient_dans_la_minute(7100, 0)
