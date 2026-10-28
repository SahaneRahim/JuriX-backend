"""
Le statut d'un texte en une ligne : resume des liens PROUVES de
app/services/liens_entre_textes.py, lu sans y rien changer.
"""

from datetime import date

import pytest

from app.services.liens_entre_textes import LIENS, Lien
from app.services.statut_en_bref import resumer, statut_en_bref

pytestmark = pytest.mark.unit


def _lien(nature, par_reference, par_date, articles=(), texte="T-1", par_titre=None, **autres):
    return Lien(
        texte=texte,
        texte_nom="le texte touche",
        nature=nature,
        par_titre=par_titre or f"le texte {par_reference}",
        par_date=par_date,
        par_reference=par_reference,
        articles=articles,
        **autres,
    )


# Les cles de `natures` et leur libelle (contrat de l'API, §1.3) : une cle, un
# libelle, sans regle cachee a deduire de `abroge`.
LIBELLES = {
    "abroge": "abrogé",
    "remplace": "remplacé",
    "abroge_partiellement": "abrogé en partie",
    "remplace_partiellement": "remplacé en partie",
    "modifie": "modifié",
    "complete": "complété",
}


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
    assert partiel["T-1"].natures == {"abroge_partiellement": 1}
    entier = resumer([
        _lien("modifie", "R-1", date(2017, 7, 12), articles=("5",)),
        _lien("remplace", "R-2", date(2025, 7, 18)),
    ])
    assert entier["T-1"].abroge is True
    # La modification de 2017 est depassee par le remplacement de 2025 : la
    # page du texte la range dans l'historique, la ligne ne la compte plus.
    assert entier["T-1"].textes == 1
    assert entier["T-1"].natures == {"remplace": 1}


def test_un_lien_anterieur_a_une_abrogation_partielle_d_autres_articles_compte():
    """Seul ce que l'abrogation couvre est depasse : l'article 5 modifie reste actuel."""
    resume = resumer([
        _lien("modifie", "R-1", date(2017, 7, 12), articles=("5",)),
        _lien("abroge", "R-2", date(2025, 7, 18), articles=("6", "7")),
    ])["T-1"]
    assert resume.textes == 2
    assert resume.natures == {"abroge_partiellement": 1, "modifie": 1}


def test_une_abrogation_ou_un_remplacement_d_articles_se_lit_en_partie():
    """
    Revue du 2026-10-28 : la loi de finances 2020, dont seul l'article 22 est
    abroge, portait natures {"abroge": 1} et la ligne P7 aurait affiche
    « Abrogé par 1 texte ». Chaque cle a desormais un seul libelle.
    """
    resume = resumer([
        _lien("abroge", "R-1", date(2023, 12, 19), articles=("22",)),
        _lien("remplace", "R-2", date(2021, 12, 16), articles=("1",)),
    ])["T-1"]

    assert resume.natures == {"abroge_partiellement": 1, "remplace_partiellement": 1}
    assert resume.abroge is False


def test_une_etendue_inconnue_n_abroge_pas_le_texte_entier():
    """
    Meme regle que la page : un lien sans liste d'articles dont le releve dit
    que les articles « ne sont pas connus » ne vise pas tout le texte.
    """
    resume = resumer([
        _lien("remplace", "R-1", date(2012, 12, 21),
              portee="les articles modifiés ne sont pas connus"),
    ])["T-1"]

    assert resume.abroge is False
    assert resume.natures == {"remplace_partiellement": 1}


def test_une_abrogation_limitee_a_une_section_laisse_les_autres_sections():
    """
    Le Code civil reprend les memes numeros dans plusieurs titres : abroger
    l'article 53 du titre 5 ne rend pas historique ce qui touche l'article 53
    du titre 2. Dans le meme titre, si.
    """
    modifie = _lien("modifie", "R-1", date(2017, 7, 12), articles=("53",), section="TITRE 2")
    ailleurs = _lien("abroge", "R-2", date(2025, 7, 18), articles=("53",), section="TITRE 5")
    au_meme_endroit = _lien("abroge", "R-2", date(2025, 7, 18), articles=("53",), section="TITRE 2")

    resume = resumer([modifie, ailleurs])["T-1"]
    assert resume.textes == 2
    assert resume.natures == {"abroge_partiellement": 1, "modifie": 1}

    assert resumer([modifie, au_meme_endroit])["T-1"].natures == {"abroge_partiellement": 1}


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


@pytest.mark.parametrize(
    "reference, natures",
    [
        # Loi de finances 2020 : seul l'article 22 est abroge par celle de 2024.
        ("PRC-8034", {"abroge_partiellement": 1}),
        # Loi de finances 2023 : son article 22, de meme.
        ("PRC-9449", {"abroge_partiellement": 1}),
        # Loi 2020/009 : son article 1 seul est remplace par la loi 2021/022.
        ("PRC-8413", {"remplace_partiellement": 1}),
        # Remplacees en entier par l'ordonnance 2025/002.
        ("LOI-2013-004", {"remplace": 1}),
        ("PRC-5697", {"remplace": 1}),
    ],
)
def test_donnees_reelles_abrogees_ou_remplacees(reference, natures):
    resume = statut_en_bref(reference)

    assert resume.natures == natures
    assert resume.abroge is ("remplace" in natures)


@pytest.mark.parametrize("reference", sorted({lien.texte for lien in LIENS}))
def test_chaque_nature_a_son_libelle_et_abroge_se_lit_dans_natures(reference):
    """La regle du front tient dans une table : aucune deduction de `abroge` a faire."""
    resume = statut_en_bref(reference)

    assert set(resume.natures) <= set(LIBELLES)
    assert resume.abroge == ("abroge" in resume.natures or "remplace" in resume.natures)


def test_aucun_lien_connu_rend_none():
    """None ne veut pas dire « en vigueur » : seulement qu'aucun lien n'est releve."""
    assert statut_en_bref("PRC-292") is None
    assert statut_en_bref(None) is None
    assert statut_en_bref("") is None
