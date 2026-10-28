"""
Le statut prouve d'une source citee : les liens de liens_entre_textes,
structures pour l'interface.

La regle qui compte le plus : l'interface ne dit jamais autre chose que ce
que le modele a lu. Le premier test rejoue donc chaque lien du corpus et
confronte la selection de statut_des_sources aux lignes de statut que
textes_successifs place sous les extraits.

Author: JuriX Team
"""

from datetime import date

import pytest

from app.services.liens_entre_textes import LIENS, Lien
from app.services.statut_des_sources import (
    etat_prouve,
    liens_qui_touchent,
    references_a_resoudre,
    statut_prouve,
)
from app.services.textes_successifs import lignes_de_statut


def _cas_du_corpus():
    """(reference, numero, contenu, section) : chaque article vise, chaque morceau, le texte."""
    cas = set()
    for lien in LIENS:
        cas.add((lien.texte, None, None, None))
        for article in lien.articles:
            premier = article.split("à")[0].strip()
            cas.add((lien.texte, premier, None, lien.section))
        for morceau, _article, repere in lien.morceaux:
            cas.add((lien.texte, morceau, repere, lien.section))
        for morceau, repere in lien.sauf:
            cas.add((lien.texte, morceau, repere, lien.section))
        # Un article que le lien ne vise pas.
        cas.add((lien.texte, "9999", None, lien.section))
    return sorted(cas, key=lambda c: tuple(str(x) for x in c))


@pytest.mark.parametrize("reference, numero, contenu, section", _cas_du_corpus())
def test_meme_selection_que_les_lignes_du_modele(reference, numero, contenu, section):
    liens = liens_qui_touchent(reference, numero, contenu, section)
    lignes = [
        ligne for ligne in lignes_de_statut(reference, numero, contenu, section)
        if ligne.startswith("Statut :")
    ]

    if numero is None:
        entiers = [lien for lien in liens if not lien.articles]
        resume = [ligne for ligne in lignes if ligne.startswith("Statut : des articles de ce texte")]
        assert len(lignes) - len(resume) == len(entiers)
        assert bool(resume) == any(lien.articles for lien in liens)
        assert all(lien.par_titre in ligne for lien, ligne in zip(entiers, lignes))
    else:
        assert len(lignes) == len(liens)
        assert all(lien.par_titre in ligne for lien, ligne in zip(liens, lignes))


def _lien(nature, par_date, articles=(), par_reference="PRC-2"):
    return Lien(
        texte="PRC-1", texte_nom="la Loi N°1", nature=nature,
        par_titre=f"la Loi N°{par_date.year}/001 du 1er mars {par_date.year}",
        par_date=par_date, par_reference=par_reference, articles=tuple(articles),
    )


class TestEtat:

    def test_l_abrogation_l_emporte_sur_la_modification(self):
        liens = [_lien("modifie", date(2010, 1, 1)), _lien("abroge", date(2020, 1, 1))]
        assert etat_prouve(liens, "5") == "abroge"

    def test_le_texte_entier_ne_compte_que_les_liens_qui_le_visent_tout_entier(self):
        """« Abrogé » sur un texte dont un seul article l'est serait faux."""
        assert etat_prouve([_lien("abroge", date(2020, 1, 1), articles=("5",))], None) is None
        assert etat_prouve([_lien("modifie", date(2020, 1, 1))], None) == "modifie"

    def test_sans_lien_rien_n_est_affirme(self):
        assert etat_prouve([], "5") is None


class TestStatutProuve:

    def test_un_article_reecrit_porte_son_lien_et_sa_date(self):
        """
        Constitution, article 14 : reecrit par la loi 2008/001 (PRC-6284),
        relevee le 2026-10-14 dans liens_entre_textes.
        """
        statut = statut_prouve("PRC-6283", "14", ids={"PRC-6284": 6284})

        assert statut.etat == "modifie"
        [lien] = [
            lien for lien in statut.liens
            if lien.par_reference == "PRC-6284" and lien.nature == "modifie"
        ]
        assert lien.par_law_id == 6284
        assert lien.par_date == date(2008, 4, 14)
        assert lien.libelle == "modifié"
        assert "14" in lien.articles
        # Les memes lignes que le modele, pour audit.
        assert statut.lignes == lignes_de_statut("PRC-6283", "14")

    def test_un_texte_absent_du_corpus_reste_sans_identifiant(self):
        statut = statut_prouve("PRC-6283", "14", ids={})
        assert all(lien.par_law_id is None for lien in statut.liens)

    def test_une_source_sans_lien_connu_a_un_statut_vide(self):
        statut = statut_prouve("REF-INCONNUE", "1")
        assert statut.etat is None and statut.liens == [] and statut.lignes == []

    def test_les_references_a_resoudre(self):
        liens = [_lien("modifie", date(2010, 1, 1)), _lien("abroge", date(2020, 1, 1), par_reference=None)]
        assert references_a_resoudre(liens) == ["PRC-2"]

    def test_les_liens_vont_du_plus_recent_au_plus_ancien(self):
        statut = statut_prouve("PRC-6283", None)
        dates = [lien.par_date for lien in statut.liens]
        assert dates == sorted(dates, reverse=True)
