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

from app.services.prompts import (
    CONVERSATIONAL_PROMPTS,
    SYSTEM_PROMPTS,
    construire_prompt_de_classification,
    get_conversational_prompt,
    get_system_prompt,
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


def test_persona_inconnu_replie_sur_citoyen():
    assert get_system_prompt("astronaute", "fr") == get_system_prompt("citoyen", "fr")


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
