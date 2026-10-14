"""
Tests des liens entre textes successifs.

Le chat citait l'ancienne version d'un article reecrit depuis : la recherche
remontait l'article du texte d'origine, et rien dans le contexte ne disait
qu'une loi plus recente l'avait remplace. Chaque extrait porte desormais sa
date et une ligne de statut qui dit ce qui en decoule.

Une ligne de statut fausse egare le modele plus surement qu'une ligne
absente : une bonne part de ces tests verifie qu'un morceau NON touche reste
muet (annexes d'une loi de finances, article en vigueur colle a un article
modifie, meme numero dans un autre titre du Code civil).
"""

import re
from datetime import date

import pytest

from app.schemas.search import ChunkResult
from app.services import textes_successifs
from app.services.prompts import SYSTEM_PROMPTS, build_context_string, format_chunk_block
from app.services.textes_successifs import (
    CONSEQUENCES,
    LIENS,
    NATURES,
    Lien,
    lignes_de_statut,
)
from app.utils.chunk_refiner import est_pseudo_numero

MODIFIE = Lien(
    texte="LOI-2000-001",
    texte_nom="la Loi N°2000/001 portant code de démonstration",
    nature="modifie",
    par_titre="la Loi N°2020/005 du 3 mars 2020",
    par_date=date(2020, 3, 3),
    par_reference="LOI-2020-005",
    articles=("1", "12", "30 à 35"),
)

REMPLACE_HORS_CORPUS = Lien(
    texte="LOI-2001-002",
    texte_nom="la Loi N°2001/002 fixant les règles de démonstration",
    nature="remplace",
    par_titre="la Loi N°2021/009 du 1er juin 2021",
    par_date=date(2021, 6, 1),
    portee="toutes ses dispositions",
)

LIENS_DE_TEST = (MODIFIE, REMPLACE_HORS_CORPUS)


def _statut(reference, numero, contenu=None, section=None, liens=LIENS_DE_TEST):
    return lignes_de_statut(reference, numero, contenu, section, liens)


# ==================== LE TEXTE TOUCHE ====================

@pytest.mark.parametrize("numero", ["12", "1", "1er", "12.2", "30", "33", "35", "33.4"])
def test_un_article_reecrit_porte_le_texte_qui_le_reecrit(numero):
    """« 12.2 » est le deuxieme morceau de l'article 12 : il est reecrit aussi."""
    assert _statut("LOI-2000-001", numero) == [
        "Statut : article modifié par la Loi N°2020/005 du 3 mars 2020 (LOI-2020-005). "
        "Pour ce qui a changé, ce texte fait foi."
    ]


@pytest.mark.parametrize("numero", ["2", "29", "36", "120", "13.1"])
def test_un_article_que_le_texte_recent_ne_touche_pas_reste_muet(numero):
    """Une ligne de statut sur un article intact ferait ecarter une regle en vigueur."""
    assert _statut("LOI-2000-001", numero) == []


@pytest.mark.parametrize("numero", ["PREAMBULE", "LEGAL_BASIS", "SECTION_3"])
def test_un_pseudo_numero_n_est_pas_un_article_reecrit(numero):
    assert _statut("LOI-2000-001", numero) == []


@pytest.mark.parametrize("numero", ["4", "PREAMBULE"])
def test_un_texte_remplace_en_entier_l_est_dans_chaque_extrait(numero):
    assert _statut("LOI-2001-002", numero) == [
        "Statut : texte remplacé par la Loi N°2021/009 du 1er juin 2021 (texte absent des "
        "documents de JuriX) : toutes ses dispositions. Le texte ne s'applique plus."
    ]


@pytest.mark.parametrize("nature,vise,consequence", [
    ("modifie", "article", "Pour ce qui a changé, ce texte fait foi."),
    ("complete", "article", "L'article reste en vigueur."),
    ("abroge", "article", "L'article ne s'applique plus."),
    ("abroge_partiellement", "article", "Hors de ce que vise ce texte, l'article s'applique encore."),
    ("modifie", "texte", "L'extrait peut encore s'appliquer, sous réserve de cette modification."),
    ("remplace", "texte", "Le texte ne s'applique plus."),
])
def test_la_ligne_dit_ce_qui_en_decoule(nature, vise, consequence):
    """
    Un article complete reste en vigueur ; un article modifie l'est pour ce
    qui n'a pas change. Dire « n'est plus la regle » pour les deux faisait
    ecarter des alineas en vigueur.
    """
    lien = Lien(
        texte="T", texte_nom="le texte T", nature=nature, par_titre="la Loi N°2024/001",
        par_date=date(2024, 1, 2), par_reference="R",
        articles=("5",) if vise == "article" else (),
    )

    ligne, = _statut("T", "5", liens=[lien])

    assert ligne.endswith(consequence)
    assert f"Statut : {vise} {NATURES[nature].participe} par" in ligne


def test_chaque_nature_a_sa_consequence_pour_un_article_et_pour_un_texte():
    assert set(CONSEQUENCES) == {(n, v) for n in NATURES for v in ("article", "texte")}


# ==================== LE DECOUPAGE EN BASE ====================

SUITE_DE_L_ARTICLE_6 = Lien(
    texte="PRC-6283", texte_nom="la Constitution", nature="modifie",
    par_titre="la Loi N°2026/002 du 14 avril 2026", par_date=date(2026, 4, 14),
    par_reference="PRC-10679", articles=("6",), morceaux=(("SECTION_2", "6", "par intérim"),),
)


def test_un_article_range_sous_un_autre_numero_garde_son_statut():
    """
    En base, la fin de l'article 6 de la Constitution (l'interim du President
    du Senat) est un morceau « SECTION_2 » : sans statut, le modele y lisait
    l'ancien regime de la vacance comme s'il etait en vigueur.
    """
    contenu = "b) le Président de la République par intérim - le Président du Sénat …"

    assert _statut("PRC-6283", "SECTION_2", contenu, liens=[SUITE_DE_L_ARTICLE_6]) == [
        "Statut : cet extrait contient tout ou partie de l'article 6, modifié par la Loi "
        "N°2026/002 du 14 avril 2026 (PRC-10679). Pour ce qui a changé, ce texte fait foi. "
        "Le reste de l'extrait n'est pas concerné."
    ]


def test_un_morceau_renumerote_ne_recoit_pas_le_statut_d_un_autre():
    """
    SECTION_n est un compteur : apres un nouveau decoupage, « SECTION_2 » peut
    designer un autre passage. Sans le repere, la ligne se poserait dessus.
    """
    autre = "Le Gouvernement est responsable devant l'Assemblée nationale."

    assert _statut("PRC-6283", "SECTION_2", autre, liens=[SUITE_DE_L_ARTICLE_6]) == []
    assert _statut("PRC-6283", "SECTION_3", "par intérim", liens=[SUITE_DE_L_ARTICLE_6]) == []


def test_un_article_en_vigueur_qui_porte_la_fin_d_un_autre_n_est_pas_dit_modifie():
    """
    L'ancien article 8 de la loi sur la Cour supreme est a la fin du morceau
    « 7 » : l'article 7, lui, est en vigueur.
    """
    lien = Lien(
        texte="LOI-2006-016", texte_nom="la Loi N°2006/016", nature="modifie",
        par_titre="la Loi N° 2017/014 du 12 juillet 2017", par_date=date(2017, 7, 12),
        par_reference="PRC-5698", articles=("8", "11"), morceaux=(("7", "8", "ARTICLE 8"),),
    )

    ligne, = _statut("LOI-2006-016", "7", "… ARTICLE 8 : La Chambre Judiciaire comprend", liens=[lien])

    assert ligne.startswith("Statut : cet extrait contient tout ou partie de l'article 8, modifié")
    assert ligne.endswith("Le reste de l'extrait n'est pas concerné.")


CODE_CIVIL = Lien(
    texte="CODE-CIVIL", texte_nom="le Code civil", nature="abroge",
    par_titre="la Loi n° 2024/016 du 23 décembre 2024", par_date=date(2024, 12, 23),
    par_reference="PRC-10257", articles=("34 à 101",),
    section="TITRE 2 Des actes de l'état civil.",
    sauf=(("39", "nom patronymique"),),
)


def test_un_numero_d_un_autre_titre_de_la_compilation_reste_muet():
    """
    La compilation du Code civil reprend les memes numeros dans plusieurs
    titres : « 53 » est aussi un article de l'Acte uniforme sur les suretes.
    """
    assert _statut("CODE-CIVIL", "53", section="TITRE 2 Des actes de l'état civil.", liens=[CODE_CIVIL])
    assert _statut("CODE-CIVIL", "53", section="TITRE 17 Du nantissement.", liens=[CODE_CIVIL]) == []
    assert _statut("CODE-CIVIL", "53", section=None, liens=[CODE_CIVIL]) == []


def test_un_autre_texte_range_sous_le_meme_numero_est_excepte():
    titre = "TITRE 2  des actes de l'état civil."  # espaces et casse ne comptent pas

    assert _statut("CODE-CIVIL", "39", "Les témoins …", titre, liens=[CODE_CIVIL])
    assert _statut("CODE-CIVIL", "39", "Le nom patronymique de l'enfant …", titre, liens=[CODE_CIVIL]) == []


# ==================== LIGNE « NIVEAU LOI » ====================

def test_une_ligne_qui_porte_tout_le_texte_nomme_les_textes_qui_la_touchent():
    """
    La recherche par titre renvoie une ligne sans numero, dont le contenu est
    la loi entiere. Revue du 2026-10-14 : une ligne par lien y faisait, pour
    le Code civil, 29 lignes et 18 000 caracteres hors budget. Une seule
    ligne nomme desormais les textes, au plus quatre.
    """
    assert _statut("LOI-2000-001", None) == [
        "Statut : des articles de ce texte ont été modifiés ou abrogés depuis : "
        "la Loi N°2020/005 du 3 mars 2020 (LOI-2020-005)."
    ]
    liens = [
        Lien(texte="CODE-CIVIL", texte_nom="le Code civil", nature="abroge",
             par_titre=f"la Loi N°{annee}/001 du 2 janvier {annee}", par_date=date(annee, 1, 2),
             par_reference=f"R-{annee}", articles=(str(annee - 1990),))
        for annee in range(2000, 2007)
    ]

    ligne, = _statut("CODE-CIVIL", None, liens=liens)

    assert ligne.count("la Loi N°") == 4 and ligne.endswith("; et 3 autres.")
    assert "2006" in ligne and "2000" not in ligne  # les plus recents d'abord


def test_un_texte_remplace_fait_taire_ce_qui_l_avait_modifie_avant():
    """
    Revue du 2026-10-14 : sous un article de la loi 2013/004, « modifié par la
    loi de 2017, ce texte fait foi » contredisait « remplacé par l'ordonnance
    de 2025, le texte ne s'applique plus ».
    """
    modifie_2017 = Lien(
        texte="T", texte_nom="la Loi T", nature="modifie", par_titre="la Loi N°2017/015",
        par_date=date(2017, 7, 12), par_reference="R-2017", articles=("19",),
    )
    remplace_2025 = Lien(
        texte="T", texte_nom="la Loi T", nature="remplace", par_titre="l'Ordonnance N°2025/002",
        par_date=date(2025, 7, 18), par_reference="R-2025",
    )

    for numero in ("19", None):
        lignes = _statut("T", numero, liens=[modifie_2017, remplace_2025])
        assert len(lignes) == 1 and lignes[0].startswith("Statut : texte remplacé par l'Ordonnance")


def test_un_morceau_designe_tel_quel_ne_designe_pas_ses_voisins():
    """« 67.3 » porte l'article 74 du Code du travail ; 67.1 et 67.2, non."""
    lien = Lien(
        texte="T", texte_nom="le Code T", nature="abroge_partiellement", par_titre="la Loi N°92/007",
        par_date=date(1992, 8, 14), par_reference="R", articles=("2271",), par_articles=("67.3",),
    )

    assert _statut("R", "67.3", liens=[lien])
    assert _statut("R", "67.1", liens=[lien]) == [] and _statut("R", "67", liens=[lien]) == []


def test_la_ligne_du_texte_recent_nomme_le_titre_de_la_compilation():
    """« articles 3 à 27 » d'un titre du Code civil suit la numerotation d'un Acte uniforme."""
    lien = Lien(
        texte="CODE-CIVIL", texte_nom="le Code civil applicable au Cameroun", nature="abroge_partiellement",
        par_titre="l'Acte uniforme du 15 décembre 2010", par_date=date(2010, 12, 15),
        par_reference="AUS", articles=("3 à 27",), section="TITRE I Sûretés personnelles",
    )

    assert _statut("AUS", "1", liens=[lien]) == [
        "Ce texte abroge en partie le Code civil applicable au Cameroun (TITRE I Sûretés personnelles), "
        "articles 3 à 27"
    ]


# ==================== LE TEXTE QUI TOUCHE ====================

def test_le_texte_recent_dit_ce_qu_il_change():
    """
    La recherche peut ne remonter que le nouvel article : sans cette ligne, le
    modele ignore qu'il remplace une disposition plus ancienne.
    """
    assert _statut("LOI-2020-005", "1") == [
        "Ce texte modifie la Loi N°2000/001 portant code de démonstration, articles 1, 12, 30 à 35"
    ]


@pytest.mark.parametrize("numero", ["PREAMBULE", "LEGAL_BASIS", "SIGNATURE"])
def test_ni_les_visas_ni_la_signature_ne_portent_ce_que_le_texte_change(numero):
    assert _statut("LOI-2020-005", numero) == []


LOI_DE_FINANCES = Lien(
    texte="PRC-9875",
    texte_nom="la Loi N°2023/019 portant loi de finances pour l'exercice 2024",
    nature="modifie",
    par_titre="la Loi n° 2024/013 du 23 décembre 2024",
    par_date=date(2024, 12, 23),
    par_reference="PRC-10254",
    articles=("24",),
    par_articles=("24",),
)


@pytest.mark.parametrize("numero", ["24", "24.3"])
def test_une_loi_de_finances_annonce_le_changement_sous_l_article_qui_le_porte(numero):
    assert _statut("PRC-10254", numero, liens=[LOI_DE_FINANCES]) == [
        "Ce texte modifie la Loi N°2023/019 portant loi de finances pour l'exercice 2024, article 24"
    ]


@pytest.mark.parametrize("numero", ["1", "25", "27.1", "PREAMBULE"])
def test_une_loi_de_finances_ne_l_annonce_pas_sous_ses_autres_articles(numero):
    """Une loi de finances touche des dizaines de textes : la ligne n'a sens que sous l'article qui modifie."""
    assert _statut("PRC-10254", numero, liens=[LOI_DE_FINANCES]) == []


@pytest.mark.parametrize("numero", ["35", "35.6", "21", "16"])
def test_les_annexes_et_le_code_des_impots_recopie_ne_sont_pas_dits_modifies(numero):
    """
    Revue du 2026-10-14 : avec « 35 » dans la liste, les dix-sept morceaux
    d'annexes « 35.2 » a « 35.20 » de la loi de finances 2024 (amendes
    routieres, aviation…) etaient dits modifies par la loi de 2025, qui n'y
    touche pas. Le lien reel n'en garde que l'article 24.
    """
    assert lignes_de_statut("PRC-9875", numero) == []


@pytest.mark.parametrize("reference", ["LOI-1999-999", "", None])
def test_un_texte_sans_lien_n_a_aucun_statut(reference):
    assert _statut(reference, "1") == []


# ==================== DANS LE CONTEXTE DU MODELE ====================

def _chunk(**kwargs) -> ChunkResult:
    valeurs = dict(
        article_id=1, law_id=1, number="12", content="Contenu.",
        reference="LOI-2000-001", law_title="Loi N°2000/001 portant code de démonstration",
        relevance_score=0.8,
    )
    valeurs.update(kwargs)
    return ChunkResult(**valeurs)


def test_l_extrait_porte_la_date_de_son_texte():
    bloc = format_chunk_block(_chunk(publication_date=date(2016, 7, 12)), 1)

    assert "Date du texte : 12/07/2016" in bloc


def test_l_extrait_sans_date_le_dit():
    """Sans date, le modele ne doit pas en supposer une pour departager."""
    assert "Date du texte : non indiquée" in format_chunk_block(_chunk(), 1)


def test_un_premier_janvier_de_remplissage_ne_donne_que_l_annee():
    """« Décret N°2024/193 juin 10 » est date du 1er janvier en base : seule l'annee est sure."""
    remplissage = _chunk(publication_date=date(2024, 1, 1), law_title="Décret N°2024/193 juin 10")
    vrai = _chunk(publication_date=date(2024, 1, 1), law_title="Décret N°2024/001 du 1er janvier 2024")

    assert "Date du texte : 2024 (jour non indiqué)" in format_chunk_block(remplissage, 1)
    assert "Date du texte : 01/01/2024" in format_chunk_block(vrai, 1)


def test_l_extrait_porte_son_statut(monkeypatch):
    monkeypatch.setattr(textes_successifs, "LIENS", LIENS_DE_TEST)

    lignes = format_chunk_block(_chunk(), 3).splitlines()

    assert lignes[0] == "[3] LOI-2000-001 — Loi N°2000/001 portant code de démonstration"
    assert lignes[-1].startswith("Statut : article modifié par la Loi N°2020/005")


def test_une_ligne_deja_donnee_n_est_repetee_qu_en_tete(monkeypatch):
    """Huit extraits d'un meme traite portaient huit fois le meme paragraphe."""
    lien = Lien(
        texte="LOI-2000-001", texte_nom="la Loi N°2000/001", nature="modifie",
        par_titre="la Loi N°2020/005 du 3 mars 2020", par_date=date(2020, 3, 3),
        par_reference="LOI-2020-005", articles=("12", "13"), portee="délais allongés",
    )
    monkeypatch.setattr(textes_successifs, "LIENS", (lien,))

    contexte = build_context_string([_chunk(number="12"), _chunk(article_id=2, number="13")])

    assert contexte.count("délais allongés") == 1
    assert (
        "Statut : article modifié par la Loi N°2020/005 du 3 mars 2020 (LOI-2020-005) "
        "(détails sous l'extrait [1])"
    ) in contexte


@pytest.mark.parametrize("langue", ["fr", "en"])
def test_le_prompt_demande_de_suivre_les_lignes_que_le_contexte_emet(langue):
    """Le prompt nomme « Statut » et « Ce texte … » : les deux libelles doivent exister."""
    lignes = _statut("LOI-2000-001", "12") + _statut("LOI-2020-005", "1")

    assert lignes[0].startswith("Statut :")
    assert lignes[1].startswith("Ce texte ")
    for prompt in SYSTEM_PROMPTS[langue].values():
        assert "« Statut »" in prompt or '"Statut"' in prompt
        assert "« Ce texte … »" in prompt or '"Ce texte …"' in prompt


# ==================== LES LIENS SAISIS ====================

_UN_ARTICLE = re.compile(r"\d+(?:\s*(?:er|bis|ter|quater|-\d+))?|\d+\s*à\s*\d+")
_UN_MORCEAU = re.compile(r"(?:[A-Z]\s?)?\d+(?:\.\d+)?(?:\s*(?:er|bis|ter|quater|-\d+))?|\d+\s*à\s*\d+")


@pytest.mark.parametrize("lien", LIENS, ids=lambda lien: f"{lien.texte}<{lien.par_reference or lien.par_date}")
def test_chaque_lien_saisi_est_bien_forme(lien):
    assert lien.nature in NATURES
    assert lien.texte and lien.texte != lien.par_reference
    # Les deux noms entrent dans une phrase : « modifié par la Loi … ».
    assert lien.texte_nom[:1].islower() and lien.par_titre[:1].islower()
    assert str(lien.par_date.year) in lien.par_titre
    assert not lien.portee.endswith("."), "le point final est ajoute a l'affichage"
    for article in lien.articles:
        assert _UN_ARTICLE.fullmatch(article), f"{lien.texte} : article illisible « {article} »"
    # Le texte recent peut designer un morceau tel quel : pseudo-numero,
    # morceau « 67.3 », article du Code general des impots « C 131 » recopie
    # dans une loi de finances. Le texte touche passe par `morceaux`, qui
    # exige un repere.
    for morceau in lien.par_articles:
        assert _UN_MORCEAU.fullmatch(morceau) or est_pseudo_numero(morceau), morceau
    for morceau, article, repere in lien.morceaux:
        assert textes_successifs._designe(lien.articles, article), f"{lien.texte} : morceau « {morceau} »"
        assert repere.strip(), f"{lien.texte} : morceau « {morceau} » sans repere"
    for morceau, repere in lien.sauf:
        assert repere.strip(), f"{lien.texte} : exception « {morceau} » sans repere"


def test_aucun_lien_saisi_en_double():
    cles = [(lien.texte, lien.par_reference or lien.par_titre, lien.articles, lien.section) for lien in LIENS]

    assert len(cles) == len(set(cles))
