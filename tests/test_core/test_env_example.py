"""
.env.example et Settings s'accordent.

`Settings` pose `extra = "ignore"` : une cle mal orthographiee dans le .env
est ignoree EN SILENCE, et le reglage garde sa valeur par defaut. Un exemple
qui documente une cle inexistante — ou qui en oublie une — est donc la source
la plus probable d'un reglage qui « ne marche pas ». Ces tests tournent sans
base ni reseau.

Usage:
    pytest tests/test_core/test_env_example.py -v
"""

import re
from pathlib import Path

from app.core.config import Settings

EXEMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# Lues ailleurs que par Settings : legitimes dans l'exemple.
EXTERNES = {
    "PGSSLMODE": "lue par libpq et asyncpg, pas par l'application",
    "ADMIN_EMAIL": "lue par scripts/create_admin.py",
    "ADMIN_PASSWORD": "lue par scripts/create_admin.py",
}

_LIGNE = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]+)=(.*)$")


def _cles():
    """(numero de ligne, commentee ?, cle, valeur) de chaque affectation."""
    for numero, ligne in enumerate(EXEMPLE.read_text(encoding="utf-8").splitlines(), start=1):
        correspondance = _LIGNE.match(ligne.strip())
        if correspondance:
            commentee, cle, valeur = correspondance.groups()
            yield numero, bool(commentee), cle, valeur


def test_chaque_cle_est_un_reglage():
    inconnues = [
        (numero, cle) for numero, _, cle, _ in _cles()
        if cle not in Settings.model_fields and cle not in EXTERNES
    ]

    assert inconnues == []


def test_aucun_commentaire_en_fin_de_ligne():
    """docker --env-file garde « groq   # groq | local » tout entier comme valeur."""
    fautives = [(numero, cle) for numero, _, cle, valeur in _cles() if re.search(r"\s#", valeur)]

    assert fautives == []


def test_les_valeurs_actives_se_chargent():
    """Chaque valeur active de l'exemple est acceptee par Settings."""
    valeurs = {
        cle: valeur for _, commentee, cle, valeur in _cles()
        if not commentee and cle in Settings.model_fields
    }

    reglages = Settings(**valeurs)

    assert reglages.INTENT_CLASSIFIER == "groq"
    assert reglages.LLM_PROVIDER == "mistral"


def test_les_reglages_des_fournisseurs_sont_documentes():
    """Chaque reglage Mistral et Groq apparait dans l'exemple, actif ou commente."""
    documentees = {cle for _, _, cle, _ in _cles()}
    reglages = [
        cle for cle in Settings.model_fields
        if cle.startswith(("MISTRAL_", "GROQ_")) or cle in ("INTENT_CLASSIFIER", "LLM_PROVIDER")
    ]

    assert [cle for cle in reglages if cle not in documentees] == []
