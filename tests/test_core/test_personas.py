"""
Tests de app/core/personas.py : la liste unique des profils et ses alias.

Avant ce module, 'etudiant' (sans accent) etait refuse en 422 par le chat, en
400 par /personas/{p}, et retombait EN SILENCE sur le prompt citoyen dans
get_system_prompt. Ces tests figent une seule regle pour tous.

Usage:
    pytest tests/test_core/test_personas.py -v
"""

import unicodedata

import pytest

from app.core.personas import (
    PERSONAS,
    PROFIL_GENERAL,
    PROFILS,
    normaliser_persona,
    normaliser_profil,
)
from app.services.persona_service import PersonaService

ETUDIANT_NFD = unicodedata.normalize("NFD", "étudiant")


def test_les_quatre_profils_dans_l_ordre_d_affichage():
    assert PERSONAS == ("citoyen", "avocat", "entrepreneur", "étudiant")
    assert PROFILS == PERSONAS + ("general",)


def test_chaque_profil_est_en_nfc():
    """La contrainte de la migration 015 compare des octets : NFD y serait refuse."""
    for profil in PROFILS:
        assert unicodedata.is_normalized("NFC", profil)


@pytest.mark.parametrize("valeur", [None, "", "   ", "general", "Général", "GENERAL"])
def test_sans_reponse_donne_le_profil_general(valeur):
    assert normaliser_profil(valeur) == PROFIL_GENERAL


@pytest.mark.parametrize("valeur, attendu", [
    ("étudiant", "étudiant"),
    ("etudiant", "étudiant"),
    ("Étudiant", "étudiant"),
    ("ÉTUDIANT", "étudiant"),
    (ETUDIANT_NFD, "étudiant"),
    (" avocat ", "avocat"),
    ("Entrepreneur", "entrepreneur"),
    ("citoyen", "citoyen"),
])
def test_les_alias_rendent_la_forme_canonique(valeur, attendu):
    resultat = normaliser_profil(valeur)

    assert resultat == attendu
    assert unicodedata.is_normalized("NFC", resultat)


@pytest.mark.parametrize("valeur", ["student", "astronaute", 5, ["citoyen"]])
def test_une_valeur_inconnue_est_refusee(valeur):
    with pytest.raises(ValueError):
        normaliser_profil(valeur)


@pytest.mark.parametrize("valeur", [None, "", "general", "Général", "inconnu"])
def test_un_compte_ne_garde_jamais_un_refus(valeur):
    """normaliser_persona sert au compte : un refus y reste NULL, jamais 'general'."""
    with pytest.raises(ValueError):
        normaliser_persona(valeur)


def test_un_compte_accepte_les_alias():
    assert normaliser_persona("Etudiant") == "étudiant"


def test_le_service_des_personas_suit_la_liste_unique():
    assert PersonaService.VALID_PERSONAS == list(PERSONAS)
    assert set(PersonaService.PERSONA_INFO) == set(PERSONAS)
