"""
Le statut d'un texte en une ligne : resume des liens PROUVES de
app/services/liens_entre_textes.py, lu sans y rien changer.
"""

from datetime import date

import pytest

from app.services.liens_entre_textes import Lien
from app.services.statut_en_bref import resumer, statut_en_bref

pytestmark = pytest.mark.unit


def _lien(nature, par_reference, par_date, articles=(), texte="T-1", par_titre=None):
    return Lien(
        texte=texte,
        texte_nom="le texte touche",
        nature=nature,
        par_titre=par_titre or f"le texte {par_reference}",
        par_date=par_date,
        par_reference=par_reference,
        articles=articles,
    )


def test_compte_les_textes_distincts_par_nature():
    """
    Un meme texte peut porter plusieurs liens (des articles modifies, un
    autre complete) : il compte une fois au total, une fois par nature.
    """
    resume = resumer([
        _lien("modifie", "R-2008", date(2008, 4, 14), articles=("14", "15")),
        _lien("complete", "R-2008", date(2008, 4, 14), articles=("67",)),
        _lien("modifie", "R-2026", date(2026, 4, 14), articles=("6",)),
    ])["T-1"]
    assert resume.textes == 2
    assert resume.natures == {"complete": 1, "modifie": 2}
    assert resume.date_dernier_texte == date(2026, 4, 14)
    assert resume.abroge is False


def test_abroge_seulement_si_le_texte_entier_l_est():
    """Abroger trois articles ne rend pas le texte inapplicable ; le remplacer en entier, si."""
    partiel = resumer([_lien("abroge", "R-1", date(2023, 12, 19), articles=("1", "2", "3"))])
    assert partiel["T-1"].abroge is False
    entier = resumer([
        _lien("modifie", "R-1", date(2017, 7, 12), articles=("5",)),
        _lien("remplace", "R-2", date(2025, 7, 18)),
    ])
    assert entier["T-1"].abroge is True
    assert entier["T-1"].textes == 2


def test_un_texte_absent_du_corpus_compte_par_son_intitule():
    """Huit liens nomment un texte qui manque au corpus (par_reference None)."""
    resume = resumer([
        _lien("abroge_partiellement", None, date(1990, 1, 1), par_titre="la Loi A"),
        _lien("abroge_partiellement", None, date(1991, 1, 1), par_titre="la Loi B"),
    ])["T-1"]
    assert resume.natures == {"abroge_partiellement": 2}


def test_donnees_reelles_du_code_civil():
    """La ligne de la maquette P7 : « Abrogé en partie par 11 textes et complété par 1 »."""
    resume = statut_en_bref("CODE-CIVIL")
    assert resume is not None
    assert resume.natures == {"abroge_partiellement": 11, "complete": 1}
    assert resume.textes == 12
    assert resume.abroge is False


def test_aucun_lien_connu_rend_none():
    """None ne veut pas dire « en vigueur » : seulement qu'aucun lien n'est releve."""
    assert statut_en_bref("PRC-292") is None
    assert statut_en_bref(None) is None
    assert statut_en_bref("") is None
