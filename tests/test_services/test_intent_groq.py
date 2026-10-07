"""
Le classement d'intention par Groq (INTENT_CLASSIFIER=groq, le defaut).

Chaque message du chat coute une requete sur le quota du jour (1 000
requetes, 200 000 jetons). Ces tests figent ce qui le menage — consignes
courtes, cache des verdicts — et ce qui protege l'utilisateur quand Groq ne
peut pas repondre : « juridique » tout de suite, jamais une erreur ni une
attente.

Groq est double par un httpx.MockTransport, sur horloge factice : aucun
appel reseau. Le classement local (embeddings) a ete retire.

Usage:
    pytest tests/test_services/test_intent_groq.py -v
"""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.core.limiteur import Limiteur
from app.services import groq_service, intent_classifier
from app.services.groq_service import GroqService
from app.services.intent_classifier import INTENT_JURIDIQUE, classify_intent
from app.services.prompts import CONSIGNES_INTENTION


class _Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class Horloge:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def dormir(self, secondes):
        self.t += secondes

    async def dormir_async(self, secondes):
        self.t += secondes


def verdict(intention, confiance=0.9):
    return httpx.Response(200, json={
        "choices": [{
            "message": {"content": json.dumps({"intention": intention, "confiance": confiance})},
            "finish_reason": "stop",
        }],
        "usage": {"total_tokens": 330, "prompt_tokens": 310},
    })


@pytest.fixture
def groq(monkeypatch):
    """
    Mode groq, cle factice, cache vide, et un faux Groq qui rend les reponses
    programmees. Rend la liste des requetes recues.
    """
    monkeypatch.setattr(settings, "INTENT_CLASSIFIER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "cle-test")
    monkeypatch.setattr(settings, "GROQ_ATTENTE_MAX_429_S", 120.0)
    intent_classifier.cache_des_verdicts.vider()
    horloge = Horloge()
    limiteur = Limiteur(
        "test", requetes_par_minute=30, rafale=5, horloge=horloge,
        dormir=horloge.dormir, dormir_async=horloge.dormir_async,
    )
    service = GroqService(api_key="cle-test", limiteur_pour=lambda modele: limiteur)
    monkeypatch.setattr(groq_service, "get_groq_service", lambda: service)

    requetes = []
    reponses = []

    def repondre(requete):
        requetes.append(json.loads(requete.content))
        reponse = reponses.pop(0) if len(reponses) > 1 else reponses[0]
        if isinstance(reponse, Exception):
            raise reponse
        return reponse

    monkeypatch.setattr(groq_service, "transport_http", httpx.MockTransport(repondre))

    def _programmer(*a_rendre):
        reponses.extend(a_rendre)
        return requetes

    _programmer.limiteur = limiteur
    yield _programmer
    intent_classifier.cache_des_verdicts.vider()


class TestVerdict:
    async def test_consignes_courtes_et_schema_strict(self, groq):
        requetes = groq(verdict("meta", 0.95))

        resultat = await classify_intent("que sais-tu faire ?")

        assert (resultat.intent, resultat.confidence, resultat.rule) == ("meta", 0.95, "groq")
        corps = requetes[0]
        assert corps["messages"][0]["content"] == CONSIGNES_INTENTION
        assert corps["messages"][1]["content"] == "Message à classer : que sais-tu faire ?"
        schema = corps["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False
        # 40 ne suffisaient pas : jetons invisibles avant le verdict, sortie vide.
        assert corps["max_completion_tokens"] == 256

    def test_les_consignes_restent_courtes(self):
        """~870 jetons avant : le chat plafonnait vers 215 questions par jour."""
        assert len(CONSIGNES_INTENTION) < 1500

    async def test_historique_limite_a_deux_echanges_tronques(self, groq):
        requetes = groq(verdict("juridique"))
        historique = [
            _Message("user", "premiere question oubliee"),
            _Message("assistant", "x" * 500),
            _Message("user", "quelles sont les conditions du permis minier ?"),
        ]

        await classify_intent("et pour un étranger ?", history=historique)

        message = requetes[0]["messages"][1]["content"]
        assert "oubliee" not in message
        assert "assistant : " + "x" * 200 + "\n" in message
        assert "x" * 201 not in message
        assert message.endswith("Message à classer : et pour un étranger ?")


class TestCache:
    async def test_deux_messages_identiques_une_seule_requete(self, groq):
        requetes = groq(verdict("smalltalk", 0.97))

        premier = await classify_intent("comment vas-tu ?")
        second = await classify_intent("Comment  vas-tu ?")

        assert len(requetes) == 1
        assert premier.intent == second.intent == "smalltalk"
        assert second.rule == "groq-cache"

    async def test_un_autre_historique_redemande(self, groq):
        requetes = groq(verdict("juridique"))

        await classify_intent("et pour un mineur ?")
        await classify_intent("et pour un mineur ?", history=[_Message("user", "le divorce ?")])

        assert len(requetes) == 2

    async def test_un_defaut_n_est_jamais_mis_en_cache(self, groq):
        groq(httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))

        resultat = await classify_intent("écris-moi un poème")

        assert resultat.rule == "defaut-groq-erreur"
        assert len(intent_classifier.cache_des_verdicts) == 0


class TestRepliSurJuridique:
    async def test_quota_epuise_puis_aucune_requete(self, groq):
        requetes = groq(httpx.Response(
            429,
            json={"error": {"message": "Rate limit reached on tokens per day (TPD)"}},
            headers={"retry-after": "1800"},
        ))

        premier = await classify_intent("écris-moi un poème")
        second = await classify_intent("raconte une blague")

        assert (premier.intent, premier.rule) == (INTENT_JURIDIQUE, "defaut-groq-quota")
        assert (second.intent, second.rule) == (INTENT_JURIDIQUE, "defaut-groq-quota")
        assert len(requetes) == 1

    async def test_attente_trop_longue_sans_requete(self, groq):
        requetes = groq(verdict("smalltalk"))
        groq.limiteur.repousser(2.0, "429")

        resultat = await classify_intent("comment vas-tu ?")

        assert (resultat.intent, resultat.rule) == (INTENT_JURIDIQUE, "defaut-groq-limite")
        assert requetes == []

    async def test_delai_depasse(self, groq, monkeypatch):
        class _Lent:
            async def completer_json(self, **reglages):
                await asyncio.sleep(1)

        monkeypatch.setattr(groq_service, "get_groq_service", lambda: _Lent())

        resultat = await classify_intent("comment vas-tu ?", timeout=0.01)

        assert (resultat.intent, resultat.rule) == (INTENT_JURIDIQUE, "defaut-groq-timeout")

    async def test_panne(self, groq):
        groq(httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))

        resultat = await classify_intent("comment vas-tu ?")

        assert (resultat.intent, resultat.rule) == (INTENT_JURIDIQUE, "defaut-groq-erreur")

    async def test_sans_cle(self, groq, monkeypatch):
        monkeypatch.setattr(settings, "GROQ_API_KEY", "")

        resultat = await classify_intent("comment vas-tu ?")

        assert (resultat.intent, resultat.rule) == (INTENT_JURIDIQUE, "defaut-groq-cle")


class TestCourtCircuits:
    async def test_aucune_requete(self, groq):
        requetes = groq(verdict("hors_sujet"))

        article = await classify_intent("que dit l'article 33 du code minier ?")
        salut = await classify_intent("Bonjour !")

        assert article.rule == "court-circuit-article"
        assert salut.rule == "court-circuit-salutation"
        assert requetes == []


class TestReglage:
    def test_gemini_devient_llm(self):
        assert Settings(INTENT_CLASSIFIER="gemini").INTENT_CLASSIFIER == "llm"

    def test_local_refuse_avec_la_raison(self):
        with pytest.raises(ValidationError) as erreur:
            Settings(INTENT_CLASSIFIER="local")

        assert "classement local" in str(erreur.value)

    def test_groq_par_defaut(self):
        assert Settings.model_fields["INTENT_CLASSIFIER"].default == "groq"
