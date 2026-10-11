"""
Tests des prompts.

Le test qui compte est `test_la_grammaire_des_prompts_est_lisible_par_la_regex`:
il relie ce qu'on DEMANDE au modele a ce que le programme sait RELIRE. Aucune
relecture humaine ne garantit ce lien dans la duree, et le jour ou il casse,
les sources disparaissent sous des reponses pourtant justes — sans la moindre
erreur, sans le moindre journal.

Author: JuriX Team
"""

import re

import pytest

from app.core.personas import PERSONAS, PROFIL_GENERAL, PROFILS
from app.services.prompts import (
    AUDIENCE_CONVERSATIONNELLE,
    CONVERSATIONAL_PROMPTS,
    NO_RESULTS_MESSAGE,
    SYSTEM_PROMPTS,
    construire_prompt_de_classification,
    get_conversational_prompt,
    get_system_prompt,
    message_sans_resultat,
)
from app.services.rag_service import RAGService

TOUS_LES_PERSONAS = [
    (langue, persona)
    for langue, personas in SYSTEM_PROMPTS.items()
    for persona in personas
]

TOUTES_LES_INTENTIONS = [
    (langue, intention)
    for langue, intentions in CONVERSATIONAL_PROMPTS.items()
    for intention in intentions
]


# ==================== LE FORMAT N'EST PLUS IMPOSE ====================

@pytest.mark.parametrize("langue,persona", TOUS_LES_PERSONAS)
@pytest.mark.parametrize(
    "rubrique",
    [
        "Exemple de structure",
        "Example structure",
        "Réponse directe",
        "Direct Answer",
        "En termes simples",
        "Source légale",
        "Legal Source",
    ],
)
def test_aucun_persona_n_impose_de_rubrique(langue, persona, rubrique):
    """
    C'est la rubrique obligatoire qui produisait « **Source legale**
    L'information demandee n'est pas presente dans les documents fournis »
    sous une salutation. La structure est desormais une possibilite.
    """
    assert rubrique not in SYSTEM_PROMPTS[langue][persona]


@pytest.mark.parametrize("langue,persona", TOUS_LES_PERSONAS)
def test_chaque_persona_adapte_sa_longueur(langue, persona):
    prompt = SYSTEM_PROMPTS[langue][persona]
    attendu = "deux ou trois phrases" if langue == "fr" else "two or three sentences"

    assert attendu in prompt


# ==================== LA GRAMMAIRE DE CITATION, ELLE, RESTE ====================

@pytest.mark.parametrize("langue,persona", TOUS_LES_PERSONAS)
def test_la_grammaire_des_prompts_est_lisible_par_la_regex(langue, persona):
    """
    Applique la VRAIE regex d'extraction a la phrase-exemple de chaque prompt.

    Si un prompt cesse de demander la forme que `_extract_citations` sait
    relire, le bloc Sources disparait de toutes les reponses de ce persona.
    Rien ne leve, rien ne se journalise : seule cette assertion le voit.
    """
    prompt = SYSTEM_PROMPTS[langue][persona]

    assert re.search(RAGService.CITATION_REGEX, prompt), (
        f"la phrase-exemple de {langue}/{persona} n'est pas reconnue par "
        "CITATION_REGEX : les sources ne s'afficheront plus"
    )


@pytest.mark.parametrize("langue,persona", TOUS_LES_PERSONAS)
def test_chaque_persona_interdit_le_pluriel(langue, persona):
    """
    « les articles 161 et 162 du Code OHADA » n'est pas reconnu par la regex.
    L'interdiction du pluriel dans le prompt est ce qui evite la perte de
    sources — voir test_pluriel_non_reconnu dans test_rag_service.py.
    """
    prompt = SYSTEM_PROMPTS[langue][persona]
    attendu = "au singulier" if langue == "fr" else "singular"

    assert attendu in prompt


@pytest.mark.parametrize("langue,persona", TOUS_LES_PERSONAS)
def test_chaque_persona_interdit_l_invention(langue, persona):
    prompt = SYSTEM_PROMPTS[langue][persona]
    attendu = "N'invente jamais" if langue == "fr" else "Never invent"

    assert attendu in prompt


def test_instruction_de_langue_ajoutee(): 
    assert "FRANÇAIS" in get_system_prompt("citoyen", "fr")
    assert "ENGLISH" in get_system_prompt("citoyen", "en")


def test_persona_inconnu_replie_sur_le_profil_general(caplog):
    """
    Il retombait EN SILENCE sur citoyen : « etudiant » sans accent recevait le
    prompt citoyen, sans un mot. Un public inconnu recoit desormais le profil
    general, qui ne presume rien du lecteur, et le repli se voit au journal.
    """
    assert get_system_prompt("astronaute", "fr") == get_system_prompt(PROFIL_GENERAL, "fr")
    assert "astronaute" in caplog.text


def test_les_alias_designent_le_meme_prompt():
    assert get_system_prompt("etudiant", "fr") == get_system_prompt("étudiant", "fr")
    assert get_system_prompt(None, "fr") == get_system_prompt(PROFIL_GENERAL, "fr")


# ==================== DES PROFILS VRAIMENT DISTINCTS ====================

@pytest.mark.parametrize("langue", ["fr", "en"])
def test_un_prompt_par_profil_et_rien_d_autre(langue):
    assert set(SYSTEM_PROMPTS[langue]) == set(PROFILS)


@pytest.mark.parametrize("langue,persona,marqueur", [
    ("fr", "citoyen", "conseille de consulter un avocat"),
    ("en", "citoyen", "advise consulting a lawyer"),
    ("fr", "avocat", "ni jurisprudence ni doctrine"),
    ("en", "avocat", "neither case law nor legal scholarship"),
    ("fr", "entrepreneur", "obligatoire de ce qui est facultatif"),
    ("en", "entrepreneur", "mandatory from what is optional"),
    ("fr", "étudiant", "définis-la en une phrase"),
    ("en", "étudiant", "define it in one sentence"),
    ("fr", PROFIL_GENERAL, "Tu ne sais pas qui t'interroge"),
    ("en", PROFIL_GENERAL, "You do not know who is asking"),
])
def test_chaque_profil_porte_sa_consigne(langue, persona, marqueur):
    """Les profils ne differaient que de quatre ou cinq lignes de role."""
    assert marqueur in SYSTEM_PROMPTS[langue][persona]


@pytest.mark.parametrize("langue,interdit", [
    ("fr", "Ne recommande pas de consulter un avocat"),
    ("en", "Do not recommend consulting a lawyer"),
])
def test_l_avocat_n_est_pas_renvoye_vers_un_avocat(langue, interdit):
    assert interdit in SYSTEM_PROMPTS[langue]["avocat"]


@pytest.mark.parametrize("langue,phrase,texte", [
    ("fr", "« l'article 33 du Code Minier, alinéa 2 »", "Code Minier"),
    ("en", '"Article 33 of the Mining Code, paragraph 2"', "Mining Code"),
])
def test_l_alinea_de_l_avocat_reste_lisible_par_la_regex(langue, phrase, texte):
    """
    « l'article 33 alinéa 2 du Code Minier » ne serait PAS relu : la source
    disparaitrait. L'alinea vient donc apres la citation complete.
    """
    assert phrase in SYSTEM_PROMPTS[langue]["avocat"]
    correspondance = re.search(RAGService.CITATION_REGEX, phrase)
    assert correspondance and correspondance.group(3).strip() == texte


@pytest.mark.parametrize("langue,persona", TOUS_LES_PERSONAS)
def test_aucun_profil_ne_pousse_a_inventer(langue, persona):
    """Le corpus ne contient ni jurisprudence ni contexte historique."""
    prompt = SYSTEM_PROMPTS[langue][persona]

    for incitation in ("jurisprudence pertinente", "relevant case law",
                       "contexte historique", "historical context"):
        assert incitation not in prompt


# ==================== PROMPTS CONVERSATIONNELS ====================

@pytest.mark.parametrize("langue,intention", TOUTES_LES_INTENTIONS)
def test_prompt_conversationnel_disponible(langue, intention):
    assert get_conversational_prompt(intention, langue).strip()


@pytest.mark.parametrize("langue,intention", TOUTES_LES_INTENTIONS)
def test_prompt_conversationnel_ne_parle_pas_de_documents(langue, intention):
    """Aucun document n'a ete consulte : en evoquer un serait un mensonge."""
    prompt = CONVERSATIONAL_PROMPTS[langue][intention]

    assert "Documents juridiques pertinents" not in prompt


@pytest.mark.parametrize("langue,intention", TOUTES_LES_INTENTIONS)
def test_prompt_conversationnel_interdit_les_citations(langue, intention):
    prompt = CONVERSATIONAL_PROMPTS[langue][intention]
    attendu = "Ne cite aucun article" if langue == "fr" else "Cite no article"

    assert attendu in prompt or (
        "aucune citation d'article" in prompt.lower()
        or "no article citation" in prompt.lower()
    )


def test_prompt_smalltalk_interdit_l_aveu_d_absence():
    """
    Le pire de la reponse d'origine : « l'information demandee n'est pas
    presente dans les documents fournis » en reponse a « comment vas tu ? ».
    Rien n'avait ete cherche, et rien ne devait l'etre.
    """
    assert "N'annonce SURTOUT PAS" in CONVERSATIONAL_PROMPTS["fr"]["smalltalk"]


def test_prompt_meta_porte_les_faits_du_produit():
    """
    Sans faits dans le prompt, le modele invente ses capacites (« je peux
    rediger votre bail »), ce qui est un mensonge commercial.
    """
    prompt = CONVERSATIONAL_PROMPTS["fr"]["meta"]

    assert "OHADA" in prompt
    assert "ne remplace pas un avocat" in prompt


def test_intention_inconnue_replie_sans_lever():
    """Un routeur qui rend une categorie inattendue doit produire une reponse tiede, pas une 500."""
    assert get_conversational_prompt("n'importe quoi", "fr") == get_conversational_prompt(
        "smalltalk", "fr"
    )


def test_langue_inconnue_replie_sur_le_francais():
    assert get_conversational_prompt("smalltalk", "de") == get_conversational_prompt(
        "smalltalk", "fr"
    )


@pytest.mark.parametrize("langue,intention", TOUTES_LES_INTENTIONS)
@pytest.mark.parametrize("persona", PERSONAS)
def test_la_conversation_s_adapte_au_profil(langue, intention, persona):
    """Une salutation ne recoit pas la meme reponse d'un avocat et d'un citoyen."""
    prompt = get_conversational_prompt(intention, langue, persona)

    assert AUDIENCE_CONVERSATIONNELLE[langue][persona] in prompt
    # Les interdits du prompt d'origine survivent a la ligne d'audience.
    assert CONVERSATIONAL_PROMPTS[langue][intention] in prompt
    assert "Documents juridiques pertinents" not in prompt


@pytest.mark.parametrize("langue,intention", TOUTES_LES_INTENTIONS)
def test_le_profil_general_n_ajoute_aucune_audience(langue, intention):
    assert get_conversational_prompt(intention, langue, PROFIL_GENERAL) == get_conversational_prompt(
        intention, langue
    )
    assert get_conversational_prompt(intention, langue, "astronaute") == get_conversational_prompt(
        intention, langue
    )


# ==================== AUCUN RESULTAT ====================

@pytest.mark.parametrize("langue", ["fr", "en"])
def test_le_message_sans_resultat_du_citoyen_ne_change_pas(langue):
    assert message_sans_resultat(langue, "citoyen") == NO_RESULTS_MESSAGE[langue]
    assert message_sans_resultat(langue, PROFIL_GENERAL) == NO_RESULTS_MESSAGE[langue]
    assert NO_RESULTS_MESSAGE[langue].startswith(("Je n'ai pas trouvé", "I couldn't find"))


@pytest.mark.parametrize("langue,consulter", [("fr", "Consultez un avocat"), ("en", "Consult a lawyer")])
def test_on_ne_conseille_pas_un_avocat_a_un_avocat(langue, consulter):
    assert consulter not in message_sans_resultat(langue, "avocat")
    assert consulter in message_sans_resultat(langue, "citoyen")


@pytest.mark.parametrize("persona", PROFILS)
def test_meme_entete_pour_tous(persona):
    premiere = NO_RESULTS_MESSAGE["fr"].splitlines()[0]

    assert message_sans_resultat("fr", persona).splitlines()[0] == premiere


# ==================== PROMPT DE CLASSIFICATION ====================

def test_prompt_de_classification_assemble_sans_format():
    """
    Le corps porte les accolades des exemples JSON : un `.format()` leverait
    KeyError. L'assemblage est une CONCATENATION, et ce test le prouve en
    verifiant que les accolades survivent.
    """
    prompt = construire_prompt_de_classification("comment vas tu ?", "Pas d'historique")

    assert '{"intention": "smalltalk", "confiance": 0.98}' in prompt
    assert "comment vas tu ?" in prompt
    assert "Pas d'historique" in prompt


def test_prompt_de_classification_porte_la_regle_du_doute():
    """
    La regle d'or est ce qui protege du seul echec que l'utilisateur ne peut
    pas reperer : une vraie question de droit traitee comme une conversation.
    """
    prompt = construire_prompt_de_classification("peu importe", "")

    assert "RÈGLE D'OR" in prompt
    assert 'réponds "juridique"' in prompt


def test_prompt_de_classification_couvre_la_politesse_d_ouverture():
    prompt = construire_prompt_de_classification("peu importe", "")

    assert "puis-je divorcer sans avocat" in prompt
