"""
MistralService : reponses completees, 429 rejoues, fins normalisees.

Le chat s'arretait parfois au milieu d'une phrase, sans mention. Ces tests
figent ce qui l'empeche : une reponse coupee (budget de jetons, ou flux
rompu apres les premiers mots) est completee une fois, l'echo du prefixe
retire ; un 429 de debit est rejoue puis bascule sur le modele de secours,
au lieu de se faire passer pour un quota epuise.

Tout passe par un httpx.MockTransport et une horloge factice : aucun appel
reseau, aucune vraie attente. Deux tests `mistral_live` verifient la suite
contre la vraie API (deux appels chacun).

Usage:
    pytest tests/test_services/test_mistral_service.py -v
    pytest tests/test_services/test_mistral_service.py -m mistral_live
"""

import asyncio
import json

import httpx
import pytest

from app.core.config import settings
from app.core.limiteur import Limiteur
from app.services import mistral_service
from app.services.mistral_service import (
    FIN_INTERROMPUE,
    MistralOverloadedError,
    MistralQuotaError,
    MistralService,
    MistralServiceError,
    _SansEcho,
    en_boucle,
    schema_strict,
)

PRINCIPAL = "ministral-14b-latest"
SECOURS = "ministral-8b-latest"


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


class FauxMistral:
    """
    Rend, dans l'ordre, les reponses programmees : une httpx.Response, une
    fabrique qui en construit une (les flux ne se relisent pas), ou une
    exception a lever. La derniere sert indefiniment.
    """

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.requetes = []

    def __call__(self, requete):
        self.requetes.append(requete)
        reponse = self.reponses.pop(0) if len(self.reponses) > 1 else self.reponses[0]
        if isinstance(reponse, Exception):
            raise reponse
        return reponse() if callable(reponse) else reponse

    def corps(self, i=-1):
        return json.loads(self.requetes[i].content)

    def modeles(self):
        return [json.loads(r.content)["model"] for r in self.requetes if r.method == "POST"]


def ok(texte, fin="stop", modele=PRINCIPAL):
    return httpx.Response(200, json={
        "model": modele,
        "choices": [{"message": {"content": texte}, "finish_reason": fin}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    })


def refus(code, message="", entetes=None):
    return httpx.Response(code, json={"message": message}, headers=entetes or {})


def flux(*morceaux, fin="stop", rupture=None):
    """Fabrique d'une reponse SSE ; `rupture` est levee apres les morceaux."""

    def fabriquer():
        async def corps():
            for morceau in morceaux:
                evenement = {"choices": [{"delta": {"content": morceau}, "finish_reason": None}]}
                yield f"data: {json.dumps(evenement)}\n\n".encode()
            if rupture is not None:
                raise rupture
            dernier = {
                "choices": [{"delta": {}, "finish_reason": fin}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            }
            yield f"data: {json.dumps(dernier)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=corps())

    return fabriquer


@pytest.fixture
def horloge():
    return Horloge()


@pytest.fixture
def mistral(monkeypatch, horloge):
    """Un service sur faux transport, avec des limiteurs sur horloge factice."""
    monkeypatch.setattr(settings, "MISTRAL_MODEL_SECOURS", SECOURS)
    monkeypatch.setattr(settings, "MISTRAL_ATTENTE_MAX_S", 20.0)
    monkeypatch.setattr(settings, "MISTRAL_JETONS_SUITE", 2048)
    limiteurs = {}

    def limiteur_pour(modele):
        if modele not in limiteurs:
            limiteurs[modele] = Limiteur(
                modele, requetes_par_minute=30, concurrence=4, horloge=horloge,
                dormir=horloge.dormir, dormir_async=horloge.dormir_async,
            )
        return limiteurs[modele]

    def _poser(*reponses):
        faux = FauxMistral(*reponses)
        monkeypatch.setattr(mistral_service, "transport_http", httpx.MockTransport(faux))
        service = MistralService(
            api_key="cle-test", model_name=PRINCIPAL, timeout=60.0, limiteur_pour=limiteur_pour
        )
        return service, faux

    return _poser


async def _tout_lire(service, **reglages):
    fins = []
    morceaux = [m async for m in service.generate_stream("question", fin=fins.append, **reglages)]
    return morceaux, fins


SCHEMA = {
    "type": "object",
    "properties": {
        "lignes": {
            "type": "array",
            "minItems": 7,
            "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {"critere": {"type": "string"}, "a": {"type": "string"}},
                "required": ["critere"],
            },
        },
    },
}


class TestCorps:
    def test_schema_strict_a_chaque_niveau(self):
        strict = schema_strict(SCHEMA)
        lignes = strict["properties"]["lignes"]

        assert strict["additionalProperties"] is False
        assert strict["required"] == ["lignes"]
        assert "minItems" not in lignes and "maxItems" not in lignes
        assert lignes["items"]["additionalProperties"] is False
        assert lignes["items"]["required"] == ["critere", "a"]
        # L'original n'est pas modifie.
        assert "minItems" in SCHEMA["properties"]["lignes"]

    async def test_json_schema_strict_envoye(self, mistral):
        service, faux = mistral(ok('{"lignes": []}'))

        await service.generate("p", response_mime_type="application/json", response_schema=SCHEMA)

        assert faux.corps()["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "reponse", "schema": schema_strict(SCHEMA), "strict": True},
        }

    async def test_json_libre(self, mistral):
        service, faux = mistral(ok("{}"))

        await service.generate("p", response_mime_type="application/json")

        assert faux.corps()["response_format"] == {"type": "json_object"}

    def test_delai_suit_le_budget(self, mistral):
        service, _ = mistral(ok("x"))

        assert service.delai_hors_flux(8192) == pytest.approx(224.8)
        assert service.delai_hors_flux(1000) == 60.0


class TestFinsHorsFlux:
    async def test_fin_normale(self, mistral):
        service, faux = mistral(ok("Réponse complète."))

        resultat = await service.generate("p")

        assert resultat == {"response": "Réponse complète.", "fin": "STOP"}

    @pytest.mark.parametrize("brute", ["length", "model_length"])
    async def test_coupee_puis_completee_echo_retire(self, mistral, brute):
        service, faux = mistral(
            ok("Le permis de recherche est", fin=brute),
            ok("Le permis de recherche est valable trois ans."),
        )

        resultat = await service.generate("p", max_tokens=8192)

        assert resultat == {"response": "Le permis de recherche est valable trois ans.", "fin": "STOP"}
        suite = faux.corps(1)
        assert suite["messages"][-1] == {
            "role": "assistant", "content": "Le permis de recherche est", "prefix": True,
        }
        assert suite["max_tokens"] == 2048

    async def test_suite_sans_echo(self, mistral):
        service, _ = mistral(ok("Le permis est", fin="length"), ok(" valable trois ans."))

        resultat = await service.generate("p")

        assert resultat["response"] == "Le permis est valable trois ans."

    async def test_encore_coupee_apres_la_suite(self, mistral):
        service, _ = mistral(ok("Début", fin="length"), ok(" et suite", fin="length"))

        resultat = await service.generate("p")

        assert resultat == {"response": "Début et suite", "fin": "MAX_TOKENS", "tronquee": True}

    async def test_suite_en_echec(self, mistral):
        service, _ = mistral(ok("Début", fin="length"), refus(400, "bad request"))

        resultat = await service.generate("p")

        assert resultat == {"response": "Début", "fin": "MAX_TOKENS", "tronquee": True}

    async def test_json_coupe_jamais_complete(self, mistral):
        """Un JSON recolle par une suite n'est pas fiable : l'appelant decide."""
        service, faux = mistral(ok('{"lignes": [', fin="length"))

        resultat = await service.generate("p", response_schema=SCHEMA)

        assert resultat["tronquee"] is True
        assert len(faux.requetes) == 1

    async def test_texte_en_boucle_jamais_complete(self, mistral):
        boucle = "Article 5. " + "La loi dispose que le permis est accordé. " * 20
        service, faux = mistral(ok(boucle, fin="length"))

        resultat = await service.generate("p")

        assert resultat["tronquee"] is True
        assert len(faux.requetes) == 1


class TestSaturation:
    async def test_retry_after_puis_succes(self, mistral, horloge):
        service, faux = mistral(refus(429, "Rate limit exceeded", {"retry-after": "3"}), ok("Oui."))

        resultat = await service.generate("p")

        assert resultat["response"] == "Oui."
        assert horloge.sommeils == [3.0]
        assert faux.modeles() == [PRINCIPAL, PRINCIPAL]

    async def test_429_persistant_bascule_sur_le_secours(self, mistral):
        trop = refus(429, "Rate limit exceeded", {"retry-after": "6"})
        service, faux = mistral(trop, trop, trop, trop, ok("Oui.", modele=SECOURS))

        resultat = await service.generate("p")

        assert resultat["response"] == "Oui."
        assert faux.modeles() == [PRINCIPAL] * 4 + [SECOURS]

    async def test_les_deux_satures_service_sature(self, mistral):
        """« Réessayez dans un instant », plus « revenez demain »."""
        service, faux = mistral(refus(429, "Rate limit exceeded", {"retry-after": "6"}))

        with pytest.raises(MistralOverloadedError):
            await service.generate("p")

        # Le secours part a 18 s : un second essai depasserait les 20 s de
        # l'appel, que le principal a presque toutes consommees.
        assert faux.modeles() == [PRINCIPAL] * 4 + [SECOURS]

    async def test_budget_d_attente_de_20_s(self, mistral):
        service, faux = mistral(refus(429, "Rate limit exceeded", {"retry-after": "15"}))

        with pytest.raises(MistralOverloadedError):
            await service.generate("p")

        # 15 s d'attente puis un second essai ; 15 de plus depasseraient les
        # 20 s de l'appel, secours compris.
        assert faux.modeles() == [PRINCIPAL, PRINCIPAL, SECOURS]

    async def test_quota_mensuel(self, mistral):
        service, faux = mistral(refus(429, "Monthly usage limit exceeded"))

        with pytest.raises(MistralQuotaError):
            await service.generate("p")

        assert len(faux.requetes) == 1

    async def test_5xx_puis_succes(self, mistral):
        service, faux = mistral(refus(503, "unavailable"), ok("Oui."))

        assert (await service.generate("p"))["response"] == "Oui."
        assert len(faux.requetes) == 2

    async def test_requete_invalide_sans_nouvel_essai(self, mistral):
        service, faux = mistral(refus(400, "Invalid schema"))

        with pytest.raises(MistralServiceError) as erreur:
            await service.generate("p")

        assert not isinstance(erreur.value, MistralOverloadedError)
        assert "Invalid schema" in str(erreur.value)
        assert len(faux.requetes) == 1

    async def test_delai_depasse_non_rejoue(self, mistral):
        service, faux = mistral(httpx.ReadTimeout("lent"))

        with pytest.raises(MistralOverloadedError):
            await service.generate("p")

        assert len(faux.requetes) == 1

    async def test_attente_du_limiteur_trop_longue_va_au_secours(self, mistral):
        service, faux = mistral(ok("Oui.", modele=SECOURS))
        service.limiteur(PRINCIPAL).repousser(30)

        assert (await service.generate("p"))["response"] == "Oui."
        assert faux.modeles() == [SECOURS]

    async def test_un_long_retry_after_est_retenu_meme_en_abandonnant(self, mistral):
        """
        30 s depassent le budget : l'appel passe au secours. Le delai doit
        rester inscrit, sinon l'appel suivant retourne frapper le modele
        sanctionne.
        """
        service, faux = mistral(
            refus(429, "Rate limit", {"retry-after": "30"}),
            ok("Oui.", modele=SECOURS),
            ok("Encore.", modele=SECOURS),
        )

        await service.generate("p")
        await service.generate("p")

        assert faux.modeles() == [PRINCIPAL, SECOURS, SECOURS]

    async def test_toutes_les_places_prises_va_au_secours(self, mistral, monkeypatch):
        """L'attente d'une place etait sans limite : ni secours, ni 503."""
        monkeypatch.setattr(settings, "MISTRAL_ATTENTE_MAX_S", 0.05)
        service, faux = mistral(ok("Oui.", modele=SECOURS))
        principal = service.limiteur(PRINCIPAL)
        principal.concurrence = 1
        liberer = asyncio.Event()

        async def occuper():
            async with principal.creneau():
                await liberer.wait()

        occupant = asyncio.create_task(occuper())
        await asyncio.sleep(0)

        resultat = await service.generate("p")

        assert resultat["response"] == "Oui."
        assert faux.modeles() == [SECOURS]
        liberer.set()
        await occupant

    async def test_sans_secours(self, mistral, monkeypatch):
        monkeypatch.setattr(settings, "MISTRAL_MODEL_SECOURS", "")
        service, faux = mistral(ok("Oui."))
        service.limiteur(PRINCIPAL).repousser(30)

        with pytest.raises(MistralOverloadedError):
            await service.generate("p")

        assert faux.requetes == []


class TestFlux:
    async def test_morceaux_et_fin(self, mistral):
        service, faux = mistral(flux("Bon", "jour."))

        morceaux, fins = await _tout_lire(service)

        assert morceaux == ["Bon", "jour."]
        assert fins == ["STOP"]
        assert faux.corps(0)["stream"] is True

    async def test_usage_rapporte(self, mistral):
        service, _ = mistral(flux("Oui."))
        usages = []

        [m async for m in service.generate_stream("q", usage=usages.append)]

        assert usages == [{"prompt_tokens": 12, "completion_tokens": 7, "modele": PRINCIPAL}]

    async def test_coupe_puis_suite_echo_retire(self, mistral):
        service, faux = mistral(
            flux("La loi ", "dispose", fin="length"),
            flux("La loi dis", "pose que le permis", " est accordé.", fin="stop"),
        )

        morceaux, fins = await _tout_lire(service)

        assert "".join(morceaux) == "La loi dispose que le permis est accordé."
        assert fins == ["STOP"]
        suite = faux.corps(1)
        assert suite["messages"][-1] == {"role": "assistant", "content": "La loi dispose", "prefix": True}
        assert suite["max_tokens"] == 2048
        assert suite["stream"] is True

    async def test_suite_sans_echo(self, mistral):
        service, _ = mistral(flux("La loi dispose", fin="length"), flux(" que le permis est accordé."))

        morceaux, fins = await _tout_lire(service)

        assert "".join(morceaux) == "La loi dispose que le permis est accordé."

    async def test_flux_rompu_puis_suite(self, mistral):
        service, _ = mistral(
            flux("La loi dispose", rupture=httpx.ReadTimeout("plus rien depuis 60 s")),
            flux(" que le permis est accordé."),
        )

        morceaux, fins = await _tout_lire(service)

        assert "".join(morceaux) == "La loi dispose que le permis est accordé."
        assert fins == ["STOP"]

    async def test_flux_rompu_et_suite_impossible(self, mistral):
        service, _ = mistral(
            flux("La loi dispose", rupture=httpx.RemoteProtocolError("coupe")),
            refus(400, "bad request"),
        )

        morceaux, fins = await _tout_lire(service)

        assert "".join(morceaux) == "La loi dispose"
        assert fins == [FIN_INTERROMPUE]

    async def test_coupe_et_suite_encore_coupee(self, mistral):
        service, _ = mistral(flux("Début", fin="length"), flux(" et suite", fin="length"))

        morceaux, fins = await _tout_lire(service)

        assert "".join(morceaux) == "Début et suite"
        assert fins == ["MAX_TOKENS"]

    async def test_rupture_avant_tout_texte(self, mistral):
        service, _ = mistral(flux(rupture=httpx.ReadTimeout("muet")))

        with pytest.raises(MistralServiceError):
            await _tout_lire(service)

    async def test_429_avant_le_premier_octet(self, mistral, horloge):
        # 5 s, et non 2 : l'espacement du limiteur (2 s) masquerait un
        # Retry-After ignore.
        service, faux = mistral(refus(429, "Rate limit", {"retry-after": "5"}), flux("Oui."))

        morceaux, fins = await _tout_lire(service)

        assert morceaux == ["Oui."]
        assert horloge.sommeils == [5.0]

    async def test_secours_en_flux(self, mistral):
        trop = refus(429, "Rate limit", {"retry-after": "6"})
        service, faux = mistral(trop, trop, trop, trop, flux("Oui."))

        morceaux, fins = await _tout_lire(service)

        assert morceaux == ["Oui."]
        assert faux.modeles() == [PRINCIPAL] * 4 + [SECOURS]


class TestSante:
    async def test_sans_generation_et_en_cache(self, mistral):
        service, faux = mistral(httpx.Response(200, json={"data": [
            {"id": "ministral-14b-2512", "aliases": [PRINCIPAL]},
        ]}))

        premiere = await service.health_check()
        seconde = await service.health_check()

        assert premiere == seconde == {"status": "healthy", "model": PRINCIPAL}
        assert len(faux.requetes) == 1
        assert faux.requetes[0].method == "GET"
        assert faux.requetes[0].url.path == "/v1/models"

    async def test_modele_absent(self, mistral):
        service, _ = mistral(httpx.Response(200, json={"data": [{"id": "autre"}]}))

        assert (await service.health_check())["status"] == "degraded"

    async def test_cle_refusee(self, mistral):
        service, _ = mistral(refus(401, "Unauthorized"))

        sante = await service.health_check()

        assert sante["status"] == "unhealthy"
        assert "Unauthorized" in sante["error"]


class TestUtilitaires:
    def test_en_boucle(self):
        assert not en_boucle("court " * 10)
        assert not en_boucle(" ".join(f"phrase numéro {i} différente." for i in range(60)))
        assert en_boucle("Intro. " + "Le permis est accordé pour trois ans. " * 15)

    @pytest.mark.parametrize("morceaux, attendu", [
        (["La loi dispose", " que..."], " que..."),
        (["La loi", " dispose que..."], " que..."),
        ([" que..."], " que..."),
        (["La", " suite diverge"], "La suite diverge"),
        (["La loi dispose"], ""),
    ])
    def test_sans_echo(self, morceaux, attendu):
        filtre = _SansEcho("La loi dispose")

        assert "".join(filtre.filtrer(m) for m in morceaux) == attendu


# ==================== VRAIE API (sur demande) ====================

QUESTION_LONGUE = (
    "Cite les dix régions du Cameroun, une par ligne, chacune avec son chef-lieu."
)


@pytest.mark.mistral_live
async def test_vrai_mistral_suite_hors_flux():
    """Deux appels : la reponse coupee a 25 jetons, puis sa suite."""
    resultat = await MistralService().generate(QUESTION_LONGUE, max_tokens=25, temperature=0)

    assert resultat["fin"] == "STOP"
    assert "tronquee" not in resultat
    assert len(resultat["response"]) > 150


@pytest.mark.mistral_live
async def test_vrai_mistral_suite_en_flux():
    """
    Deux appels. Verifie aussi l'ECHO du prefixe en flux : s'il n'etait pas
    retire, le debut de la reponse apparaitrait deux fois.
    """
    fins = []
    morceaux = [
        m async for m in MistralService().generate_stream(
            QUESTION_LONGUE, max_tokens=25, temperature=0, fin=fins.append
        )
    ]
    texte = "".join(morceaux)

    assert fins == ["STOP"]
    assert len(texte) > 150
    assert texte.count(texte[:30]) == 1
