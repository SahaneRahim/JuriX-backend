"""
Tests du routage d'intention.

Le modele est TOUJOURS double : on teste ce que le code fait de la reponse du
modele, jamais le jugement du modele lui-meme. La qualite du classement reel se
verifie a la main avec une vraie cle (voir la procedure dans le plan), pas ici.

Author: JuriX Team
"""

import asyncio

import pytest

from app.services.gemini_service import GeminiOverloadedError, GeminiQuotaError
from app.services.intent_classifier import (
    INTENT_JURIDIQUE,
    _parse_intent,
    classify_intent,
)


class _LLM:
    """Doublure : rend ce qu'on lui dit, leve ce qu'on lui dit, tarde si on veut."""

    def __init__(self, payload=None, raises=None, delay=0.0):
        self.payload = payload
        self.raises = raises
        self.delay = delay
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        return {"response": self.payload}


def _verdict(intention, confiance=0.95):
    return '{"intention": "%s", "confiance": %s}' % (intention, confiance)


class _Message:
    """Message d'historique minimal, tel que `format_conversation_history` le lit."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


# ==================== LES QUATRE CATEGORIES ====================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question", ["comment vas tu ?", "tu as passe une bonne journee ?", "ca roule pour toi ?"]
)
async def test_bavardage_est_smalltalk(question):
    """
    Des messages de conversation courante que le court-circuit lexical ne
    reconnait PAS : ils passent donc bien par le modele.
    """
    llm = _LLM(_verdict("smalltalk"))

    resultat = await classify_intent(question, llm=llm)

    assert resultat.intent == "smalltalk"
    assert resultat.rule == "llm"


@pytest.mark.asyncio
async def test_question_sur_le_produit_est_meta():
    resultat = await classify_intent("qui es-tu ?", llm=_LLM(_verdict("meta")))

    assert resultat.intent == "meta"


@pytest.mark.asyncio
async def test_question_etrangere_est_hors_sujet():
    resultat = await classify_intent(
        "ecris-moi un poeme sur la pluie", llm=_LLM(_verdict("hors_sujet"))
    )

    assert resultat.intent == "hors_sujet"


@pytest.mark.asyncio
async def test_question_de_droit_est_juridique():
    resultat = await classify_intent(
        "quelles sont les conditions du permis de recherche ?",
        llm=_LLM(_verdict("juridique")),
    )

    assert resultat.intent == INTENT_JURIDIQUE
    assert resultat.rule == "llm"


# ==================== CE QUI PART AU MODELE ====================

@pytest.mark.asyncio
async def test_l_historique_est_transmis_au_modele():
    """
    On teste la TRANSMISSION, pas le verdict : le modele est double, son
    jugement ne prouverait rien. Ce qui doit etre garanti ici, c'est que « et
    l'article 12 ? » arrive au classificateur avec de quoi le comprendre.
    """
    llm = _LLM(_verdict("juridique"))
    historique = [
        _Message("user", "quelles sont les conditions du permis minier ?"),
        _Message("assistant", "Les conditions figurent au Code Minier."),
    ]

    await classify_intent("et pour une SARL ?", llm=llm, history=historique)

    assert "permis minier" in llm.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_sortie_structuree_demandee():
    llm = _LLM(_verdict("smalltalk"))

    await classify_intent("comment vas tu ?", llm=llm)

    assert llm.calls[0]["response_mime_type"] == "application/json"
    assert "intention" in llm.calls[0]["response_schema"]["properties"]


@pytest.mark.asyncio
async def test_temperature_nulle():
    llm = _LLM(_verdict("smalltalk"))

    await classify_intent("comment vas tu ?", llm=llm)

    assert llm.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_systeme_neutre_impose():
    """
    Sans `system`, `GeminiService.generate` retombe sur SYSTEM_INSTRUCTION, qui
    ordonne de citer des articles et de terminer par « Sources: ». Sur une
    tache de classement, elle produirait de la prose au lieu du JSON demande.
    """
    llm = _LLM(_verdict("smalltalk"))

    await classify_intent("comment vas tu ?", llm=llm)

    assert "classificateur" in llm.calls[0]["system"].lower()


def test_delai_superieur_a_la_latence_mesuree():
    """
    Verrouille la valeur contre un resserrage bien intentionne.

    Mesure sur gemini-3-flash-preview, six classifications reelles : mediane
    3,3 s, maximum 6,5 s. La valeur posee au depart, 3,0 s, faisait expirer
    cinq appels sur six — et une expiration retombe sur « juridique » sans
    erreur, sans journal, sans test rouge : le routeur cessait de router en
    silence. Toute valeur sous le maximum observe reintroduit ce mode de
    panne.
    """
    from app.core.config import settings

    assert settings.INTENT_TIMEOUT_S >= 7.0


@pytest.mark.asyncio
async def test_budget_de_jetons_suffisant_pour_un_modele_a_raisonnement():
    """
    Verrouille le piege le plus couteux du lot. Un budget de quelques jetons
    parait logique pour une sortie d'un seul mot, mais la reflexion du modele
    est facturee AVANT la premiere ligne : la reponse revient vide, le
    classificateur retombe sur « juridique », et le routeur ne route jamais —
    sans qu'aucun test ne le voie, puisqu'ils doublent tous le modele.
    """
    llm = _LLM(_verdict("smalltalk"))

    await classify_intent("comment vas tu ?", llm=llm)

    assert llm.calls[0]["max_tokens"] >= 512


# ==================== COURT-CIRCUIT ====================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "que dit l'article 33 du Code Minier ?",
        "explique moi l article 12",
        "Article L 5 du code du travail",
    ],
)
async def test_numero_d_article_court_circuite_l_appel(question):
    llm = _LLM(_verdict("smalltalk"))  # le modele dirait le contraire : ignore

    resultat = await classify_intent(question, llm=llm)

    assert resultat.intent == INTENT_JURIDIQUE
    assert resultat.rule == "court-circuit-article"
    assert llm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question", ["bonjour", "Bonjour !", "MERCI BEAUCOUP.", "bonne journée", "  salut  "]
)
async def test_salutation_pure_court_circuite_l_appel(question):
    """Le message le plus frequent d'une conversation ne coute plus aucun appel."""
    llm = _LLM(_verdict("juridique"))  # le modele dirait le contraire : ignore

    resultat = await classify_intent(question, llm=llm)

    assert resultat.intent == "smalltalk"
    assert resultat.rule == "court-circuit-salutation"
    assert llm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "bonjour, puis-je licencier un salarie malade ?",
        "merci de me dire l age legal du travail",
        "salut, ca va ? j ai une question sur mon bail",
    ],
)
async def test_la_politesse_en_ouverture_ne_court_circuite_pas(question):
    """
    Le rapprochement porte sur le message ENTIER. Un filtre par sous-chaine
    classerait ces trois-la en salutation, et c'est le pire echec possible :
    une vraie question de droit repondue sans source.
    """
    llm = _LLM(_verdict("juridique"))

    resultat = await classify_intent(question, llm=llm)

    assert resultat.rule == "llm"
    assert llm.calls != []


@pytest.mark.asyncio
async def test_pas_de_court_circuit_symetrique_sur_la_politesse():
    """
    « Bonjour, puis-je licencier un salarie malade ? » est une question de
    droit. Un court-circuit lexical sur « bonjour » la classerait en
    salutation : l'asymetrie du court-circuit est voulue.
    """
    llm = _LLM(_verdict("juridique"))

    resultat = await classify_intent(
        "bonjour, puis-je licencier un salarie malade ?", llm=llm
    )

    assert resultat.intent == INTENT_JURIDIQUE
    assert llm.calls != []


# ==================== LE REPLI ====================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    ["pas du json", "", None, '{"intention": "cuisine"}', '["une", "liste"]'],
)
async def test_verdict_inexploitable_replie_sur_juridique(payload):
    resultat = await classify_intent("comment vas tu ?", llm=_LLM(payload))

    assert resultat.intent == INTENT_JURIDIQUE
    assert resultat.rule == "defaut-json"


@pytest.mark.asyncio
async def test_expiration_replie_sur_juridique():
    resultat = await classify_intent(
        "comment vas tu ?", llm=_LLM(_verdict("smalltalk"), delay=0.5), timeout=0.01
    )

    assert resultat.intent == INTENT_JURIDIQUE
    assert resultat.rule == "defaut-timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "erreur",
    [
        GeminiQuotaError("quota epuise"),
        GeminiOverloadedError("503"),
        RuntimeError("boom"),
    ],
)
async def test_aucune_exception_ne_sort(erreur):
    """
    Une defaillance du routeur ne doit JAMAIS faire echouer la question de
    l'utilisateur : le repli d'un composant optionnel, c'est son absence.
    """
    resultat = await classify_intent("comment vas tu ?", llm=_LLM(raises=erreur))

    assert resultat.intent == INTENT_JURIDIQUE
    assert resultat.rule == "defaut-erreur"


@pytest.mark.asyncio
async def test_reponse_nue_sans_enveloppe_replie_sur_juridique():
    """Une doublure qui rend une chaine au lieu d'un dict ne doit pas lever."""

    class _Nu:
        async def generate(self, **kwargs):
            return "juste du texte"

    resultat = await classify_intent("comment vas tu ?", llm=_Nu())

    assert resultat.intent == INTENT_JURIDIQUE


@pytest.mark.asyncio
async def test_desactive_par_reglage(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "INTENT_ROUTING_ENABLED", False)
    llm = _LLM(_verdict("smalltalk"))

    resultat = await classify_intent("comment vas tu ?", llm=llm)

    assert resultat.intent == INTENT_JURIDIQUE
    assert resultat.rule == "desactive"
    assert llm.calls == []


# ==================== LE LECTEUR DE VERDICT, ISOLE ====================

def test_parse_intent_borne_la_confiance():
    assert _parse_intent('{"intention": "meta", "confiance": 5}')[1] == 1.0
    assert _parse_intent('{"intention": "meta", "confiance": -2}')[1] == 0.0


def test_parse_intent_defaut_de_confiance():
    assert _parse_intent('{"intention": "meta"}') == ("meta", 0.5)


def test_parse_intent_confiance_illisible():
    assert _parse_intent('{"intention": "meta", "confiance": "beaucoup"}') == ("meta", 0.5)
