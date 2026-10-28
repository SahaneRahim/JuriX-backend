"""
Le statut juridique d'un texte, tel que le sert GET /laws/{id}.

La page de lecture affichait « EN VIGUEUR » sous chaque article sans aucune
donnee. Elle dit desormais ce que le releve PROUVE, et seulement cela : ces
tests verifient qu'un article non touche reste muet, qu'une reecriture
depassee n'est pas presentee comme le droit en vigueur, et qu'un texte deja
consolide n'est pas dit « version d'origine ».

Fonctions pures : aucune base.

Usage:
    pytest tests/test_services/test_statut_juridique.py -v
"""

from datetime import date

import pytest

from app.services.liens_entre_textes import RELEVE_LE, LienHistorique
from app.services.statut_juridique import (
    InfoTexte,
    MorceauDuTexte,
    cle_article,
    construire_statut,
    construire_statut_d_un_article,
)
from app.services.textes_successifs import NATURES, Lien, lignes_de_statut

ANCIEN = "T-ANCIEN"
RECENT = "T-RECENT"

INFOS = {
    ANCIEN: InfoTexte(10, ANCIEN, "loi", "Loi N°2000/001 du 3 mars 2000 portant code de démonstration",
                      date(2000, 3, 3), "published"),
    RECENT: InfoTexte(20, RECENT, "loi", "Loi N°2020/005 du 3 mars 2020 modifiant la loi N°2000/001",
                      date(2020, 3, 3), "published"),
}


def _lien(**valeurs) -> Lien:
    base = dict(
        texte=ANCIEN, texte_nom="la Loi N°2000/001 portant code de démonstration",
        nature="modifie", par_titre="la Loi N°2020/005 du 3 mars 2020", par_date=date(2020, 3, 3),
        par_reference=RECENT, articles=("1er", "12", "30 à 32"), par_articles=("1", "2"),
    )
    base.update(valeurs)
    return Lien(**base)


def _statut(liens, historiques=(), morceaux=(), reference=ANCIEN, infos=INFOS):
    return construire_statut(reference, infos, morceaux, liens, historiques)


# ==================== UN TEXTE SANS LIEN ====================

def test_un_texte_sans_lien_n_affirme_rien():
    """« Aucune modification relevée », date du releve a l'appui : jamais « en vigueur »."""
    statut = _statut([_lien()], reference="T-AUTRE")

    assert statut.etat == "aucune_modification_connue"
    assert statut.releve_le == RELEVE_LE
    assert (statut.modifie_par, statut.modifie, statut.par_article, statut.par_morceau) == ([], [], {}, {})
    assert statut.abroge is False and statut.version_en_base is None


# ==================== LE TEXTE TOUCHE ====================

def test_un_texte_modifie_nomme_le_texte_qui_le_modifie():
    statut = _statut([_lien()])

    changement, = statut.modifie_par
    assert statut.etat == "modifie" and statut.types == ["modifie"]
    assert changement.sens == "modifie_par" and changement.type == "modifie"
    assert changement.articles_cibles == ["1er", "12", "30 à 32"]
    assert changement.articles_source == ["1", "2"]
    assert changement.etendue == "articles"
    assert changement.texte.id == 20 and changement.texte.reference == RECENT
    assert changement.texte.numero == "2020/005" and changement.texte.nature == "loi"
    assert changement.texte.date_signature == date(2020, 3, 3)
    assert changement.texte.consultable is True


def test_chaque_article_vise_mene_au_changement():
    """Plage developpee, « 1er » ecrit « 1 » : les cles du front (normaliserNumeroArticle)."""
    statut = _statut([_lien()])

    assert statut.par_article == {cle: ["c1"] for cle in ("1", "12", "30", "31", "32")}


def test_seuls_les_morceaux_vises_sont_marques():
    morceaux = [
        MorceauDuTexte(1, "PREAMBULE", None),
        MorceauDuTexte(2, "1", None),
        MorceauDuTexte(3, "12.2", None),
        MorceauDuTexte(4, "13", None),
        MorceauDuTexte(5, "31", None),
        MorceauDuTexte(6, "33", None),
    ]

    statut = _statut([_lien()], morceaux=morceaux)

    assert statut.par_morceau == {"2": ["c1"], "3": ["c1"], "5": ["c1"]}


def test_la_page_et_le_chat_marquent_les_memes_morceaux():
    """Meme regle (textes_successifs.morceau_touche) : sections, morceaux ranges ailleurs, exceptions."""
    lien = _lien(
        articles=("6", "35"),
        section="TITRE 2 Des actes",
        morceaux=(("SECTION_2", "6", "par intérim"),),
        sauf=(("35", "librement choisis"),),
    )
    morceaux = [
        MorceauDuTexte(1, "6", "TITRE 2 Des actes de l'état civil.", "Le Président est élu."),
        MorceauDuTexte(2, "SECTION_2", "TITRE 2 Des actes", "b) le Président par intérim ne peut"),
        MorceauDuTexte(3, "35", "TITRE 2 Des actes", "Les prénoms sont librement choisis."),
        MorceauDuTexte(4, "35", "TITRE 2 Des actes", "Les déclarations seront faites."),
        MorceauDuTexte(5, "6", "TITRE 5 Du mariage.", "On ne peut contracter un second mariage."),
    ]

    statut = _statut([lien], morceaux=morceaux)

    marques = {int(cle) for cle in statut.par_morceau}
    assert marques == {1, 2, 4}
    for morceau in morceaux:
        chat = lignes_de_statut(ANCIEN, morceau.numero, morceau.contenu, morceau.section, [lien])
        assert bool(chat) == (morceau.id in marques), morceau


def test_une_compilation_dit_la_section_de_chaque_changement():
    """Le front ne pose le changement que sous l'article de la bonne section."""
    statut = _statut([_lien(articles=("53",), section="TITRE 2 Des actes")])

    assert statut.par_article == {"53": ["c1"]}
    assert statut.modifie_par[0].section == "TITRE 2 Des actes"


@pytest.mark.parametrize("portee,etendue", [
    ("les articles modifiés ne sont pas connus", "inconnue"),
    ("seuls les articles 2, 11 à 20 sont modifiés", "texte"),
    ("", "texte"),
])
def test_un_lien_sans_liste_porte_sur_le_texte_ou_sur_des_articles_inconnus(portee, etendue):
    """Ni l'un ni l'autre ne se pose sous chaque article : la page l'affiche en tete."""
    statut = _statut([_lien(articles=(), par_reference=None, portee=portee)], morceaux=[
        MorceauDuTexte(1, "1", None), MorceauDuTexte(2, "2", None),
    ])

    assert statut.modifie_par[0].etendue == etendue
    assert statut.par_article == {} and statut.par_morceau == {}
    assert statut.etat == "modifie"


@pytest.mark.parametrize("portee,nature", [
    ("réécrit en entier ; le Vice-Président achève le mandat", "reecrit_en_entier"),
    ("réécrits en entier ; …", "reecrit_en_entier"),
    ("seul l'alinéa 4 est réécrit", "modifie"),
    ("", "modifie"),
])
def test_reecrit_en_entier_seulement_quand_le_releve_le_dit(portee, nature):
    changement, = _statut([_lien(portee=portee)]).modifie_par

    assert changement.nature_modification == nature
    assert changement.libelle == ("réécrit en entier" if nature == "reecrit_en_entier" else "modifié")


@pytest.mark.parametrize("nature", sorted(NATURES))
def test_chaque_type_a_son_libelle(nature):
    changement, = _statut([_lien(nature=nature)]).modifie_par

    assert changement.libelle == NATURES[nature].participe


# ==================== TEXTE ABROGE OU REMPLACE, HISTORIQUE ====================

def test_un_texte_remplace_le_dit_et_ce_qui_l_avait_modifie_devient_historique():
    """
    La loi 2013/004, modifiee en 2017, puis remplacee en 2025 par une
    ordonnance : la reecriture de 2017 ne dit plus le droit en vigueur.
    """
    modifie = _lien()
    remplace = _lien(
        nature="remplace", par_titre="l'Ordonnance N°2025/002 du 18 juillet 2025",
        par_date=date(2025, 7, 18), par_reference="T-ORDONNANCE", articles=(), par_articles=("51",),
    )

    statut = _statut([modifie, remplace])

    assert statut.etat == "remplace" and statut.abroge is True
    assert statut.abroge_par.titre == "Ordonnance N°2025/002 du 18 juillet 2025"
    assert statut.abroge_par.id is None and statut.abroge_par.consultable is False
    ancien, nouveau = statut.modifie_par
    assert ancien.depasse is True and ancien.depasse_par.numero == "2025/002"
    assert nouveau.depasse is False and statut.types == ["remplace"]


def test_le_texte_qui_abroge_en_dernier_se_choisit_par_la_date_du_lien():
    """Sa date d'acte peut etre nulle (1er janvier de remplissage) : l'ordre ne doit pas en dependre."""
    infos = dict(INFOS)
    infos["T-2019"] = InfoTexte(30, "T-2019", "loi", "Loi N°2019/001 sans date lisible", date(2019, 1, 1),
                                "published")
    infos["T-2010"] = InfoTexte(40, "T-2010", "loi", "Loi N°2010/001 du 2 mars 2010", date(2010, 3, 2),
                                "published")
    liens = [
        _lien(nature="abroge", articles=(), par_reference="T-2010", par_titre="la Loi N°2010/001",
              par_date=date(2010, 3, 2)),
        _lien(nature="remplace", articles=(), par_reference="T-2019", par_titre="la Loi N°2019/001",
              par_date=date(2019, 1, 1)),
    ]

    statut = _statut(liens, infos=infos)

    assert statut.etat == "remplace" and statut.abroge_par.id == 30
    assert statut.abroge_par.date_signature is None and statut.abroge_par.annee == 2019


def test_une_abrogation_d_autres_articles_ne_depasse_rien():
    modifie = _lien(articles=("12",))
    abroge = _lien(nature="abroge", par_date=date(2024, 1, 2), par_titre="la Loi N°2024/001 du 2 janvier 2024",
                   par_reference="T-2024", articles=("13",))

    ancien, _ = _statut([modifie, abroge]).modifie_par

    assert ancien.depasse is False


def test_un_lien_historique_est_depasse_et_nomme_ce_qui_l_a_depasse():
    """Article 6 de la Constitution : reecrit en 2008, reecrit de nouveau en 2026."""
    actuel = _lien(articles=("6",), par_date=date(2026, 4, 14), par_titre="la Loi N°2026/002 du 14 avril 2026",
                   par_reference="T-2026")
    historique = LienHistorique(
        lien=_lien(articles=("6",), par_date=date(2008, 4, 14), par_titre="la Loi N°2008/001 du 14 avril 2008",
                   par_reference="T-2008"),
        depasse_par_titre="la Loi N°2026/002 du 14 avril 2026",
        depasse_par_reference="T-2026",
        depasse_le=date(2026, 4, 14),
    )

    statut = _statut([actuel], [historique])

    de_2008, de_2026 = statut.modifie_par
    assert de_2008.texte.numero == "2008/001" and de_2008.depasse is True
    assert de_2008.depasse_par.numero == "2026/002"
    assert de_2026.depasse is False
    assert statut.par_article == {"6": ["c1", "c2"]}
    assert statut.etat == "modifie"


def test_un_texte_consolide_n_est_pas_dit_version_d_origine():
    """Ordonnance 74/1 en base : elle integre deja l'ordonnance 77/1."""
    integre = LienHistorique(
        lien=_lien(articles=("4", "5"), par_reference=None, par_titre="l'Ordonnance N°77/1 du 10 janvier 1977",
                   par_date=date(1977, 1, 10), par_articles=()),
        integre_au_texte=True,
    )

    statut = _statut([], [integre])

    assert statut.etat == "aucune_modification_connue"
    assert statut.version_en_base.consolidee is True
    texte, = statut.version_en_base.textes_integres
    assert texte.titre == "Ordonnance N°77/1 du 10 janvier 1977" and texte.nature == "ordonnance"
    assert statut.modifie_par[0].integre_au_texte is True


def test_l_historique_se_lit_du_plus_ancien_au_plus_recent():
    liens = [
        _lien(par_date=date(2024, 1, 1), par_titre="la Loi N°2024/001", par_reference="T-2024"),
        _lien(par_date=date(2001, 1, 1), par_titre="la Loi N°2001/001", par_reference="T-2001"),
        _lien(par_date=date(2012, 1, 1), par_titre="la Loi N°2012/001", par_reference="T-2012"),
    ]

    statut = _statut(liens)

    assert [c.texte.numero for c in statut.modifie_par] == ["2001/001", "2012/001", "2024/001"]
    assert [c.id for c in statut.modifie_par] == ["c1", "c2", "c3"]


# ==================== LE TEXTE QUI MODIFIE ====================

def test_le_texte_recent_dit_ce_qu_il_change_et_sous_quels_articles():
    morceaux = [
        MorceauDuTexte(1, "LEGAL_BASIS", None),
        MorceauDuTexte(2, "1", None),
        MorceauDuTexte(3, "2", None),
        MorceauDuTexte(4, "3", None),
        MorceauDuTexte(5, "SIGNATURE", None),
    ]

    statut = _statut([_lien()], morceaux=morceaux, reference=RECENT)

    changement, = statut.modifie
    assert statut.modifie_par == [] and statut.etat == "aucune_modification_connue"
    assert changement.sens == "modifie" and changement.texte.id == 10
    assert changement.articles_cibles == ["1er", "12", "30 à 32"]
    assert statut.par_article == {"1": ["c1"], "2": ["c1"]}
    assert statut.par_morceau == {"2": ["c1"], "3": ["c1"]}


def test_un_pseudo_numero_du_texte_recent_n_est_pas_une_cle_d_article():
    """« SECTION_1 » porte l'article 142 de la loi 2026/003 : marque par morceau, pas par numero."""
    morceaux = [MorceauDuTexte(1, "SECTION_1", None)]

    statut = _statut([_lien(par_articles=("SECTION_1", "67.3"))], morceaux=morceaux, reference=RECENT)

    assert statut.par_article == {"67": ["c1"]}
    assert statut.par_morceau == {"1": ["c1"]}


def test_un_texte_archive_n_a_pas_d_adresse():
    infos = dict(INFOS)
    infos[RECENT] = INFOS[RECENT]._replace(status="archived")

    changement, = _statut([_lien()], infos=infos).modifie_par

    assert changement.texte.id is None and changement.texte.consultable is False
    assert changement.texte.reference == RECENT


def test_un_texte_absent_de_la_base_se_decrit_par_son_intitule():
    changement, = _statut([_lien()], infos={}).modifie_par

    assert changement.texte.id is None and changement.texte.reference is None
    assert changement.texte.titre == "Loi N°2020/005 du 3 mars 2020"
    assert changement.texte.nature == "loi" and changement.texte.numero == "2020/005"
    assert changement.texte.date_signature == date(2020, 3, 3)


# ==================== L'APERCU D'UN ARTICLE ====================

def test_l_apercu_d_un_article_porte_ses_changements_et_ceux_du_texte_entier():
    liens = [
        _lien(articles=("12",)),
        _lien(articles=("13",), par_reference="T-AUTRE", par_titre="la Loi N°2021/001", par_date=date(2021, 1, 1)),
        _lien(articles=(), par_reference=None, par_titre="la Loi n° 2012/015 du 21 décembre 2012",
              par_date=date(2012, 12, 21), portee="les articles modifiés ne sont pas connus"),
    ]

    statut = construire_statut_d_un_article(ANCIEN, INFOS, [MorceauDuTexte(7, "12", None, "texte")],
                                            liens, ())

    assert [c.texte.numero for c in statut.changements] == ["2012/015", "2020/005"]
    assert statut.etat_du_texte == "modifie"


# ==================== CLES DES ARTICLES ====================

@pytest.mark.parametrize("designation,cle", [
    ("1er", "1"), ("12.2", "12"), ("L 94 bis", "L 94 BIS"), ("C 131", "C 131"),
    ("VINGT-TROISIÈME", "VINGT-TROISIÈME"), ("SECTION_2", None), ("PREAMBULE", None), ("", None),
])
def test_la_cle_d_article_suit_la_normalisation_du_front(designation, cle):
    assert cle_article(designation) == cle


# ==================== LES DONNEES DU RELEVE ====================

def test_la_constitution_montre_toute_la_chaine_de_l_article_6():
    """Le releve ne garde que 2026 ; l'historique ajoute 2008, depasse."""
    statut = construire_statut("PRC-6283", {})

    changements = {c.id: c for c in statut.modifie_par}
    chaine = [changements[i] for i in statut.par_article["6"]]
    assert [(c.texte.numero, c.depasse) for c in chaine] == [("2008/001", True), ("2026/002", False)]
    assert chaine[0].depasse_par.numero == "2026/002"
    assert chaine[1].nature_modification == "reecrit_en_entier"
    assert statut.etat == "modifie" and statut.version_en_base is None


def test_la_loi_sur_les_incitations_est_remplacee():
    statut = construire_statut("LOI-2013-004", {})

    assert statut.etat == "remplace" and statut.abroge is True
    assert statut.abroge_par.numero == "2025/002"
    assert all(c.depasse for c in statut.modifie_par if c.type != "remplace")


def test_le_regime_foncier_en_base_est_consolide():
    statut = construire_statut("ORD-1974-001", {})

    integres = [t.numero for t in statut.version_en_base.textes_integres]
    assert integres == ["77/1", "80-21"]
    assert statut.etat == "modifie", "la loi 83-19 n'est pas integree au texte en base"
