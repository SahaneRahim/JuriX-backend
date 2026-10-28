"""
Recherche affinee, sans base : filtres sur l'ensemble candidat, effectifs
disjonctifs, tri, page, cle de cache et extraits surlignes.

La propriete centrale : parcourir toutes les pages donne exactement les
resultats comptes, et cocher une valeur de facette donne exactement son
effectif. C'est ce que le filtrage dans le navigateur, sur la seule page
affichee, ne pouvait pas garantir.
"""

from datetime import date

import pytest

from app.schemas.search import SearchFilters, SearchRefine, SearchResult
from app.services.recherche_affinee import (
    affiner,
    cle_de_l_ensemble,
    effectifs,
    racines_de_la_requete,
    replier_nature,
    retenu,
    surligner,
    trier,
)

pytestmark = pytest.mark.unit

NATURES = ("décret", "loi", "arrêté", "acte uniforme")


def _resultat(i, nature="décret", categorie=13, jour=date(2020, 1, 15), nom=None):
    return SearchResult(
        law_id=i,
        reference=f"PRC-{i}",
        title=f"Texte {i}",
        type=nature,
        status="published",
        category_id=categorie,
        category_name=nom or (f"Domaine {categorie}" if categorie else None),
        publication_date=jour,
        relevance_score=1.0,
    )


@pytest.fixture
def ensemble():
    """47 resultats, dans l'ordre de pertinence (law_id croissant)."""
    resultats = []
    for i in range(47):
        resultats.append(_resultat(
            i,
            nature=NATURES[i % 4],
            categorie=(13, 5, 8)[i % 3],
            jour=None if i % 10 == 9 else date(2000 + i % 25, 1 + i % 12, 1 + i % 27),
        ))
    return resultats


def test_parcourir_toutes_les_pages_donne_exactement_le_total(ensemble):
    refine = SearchRefine(types=["decret", "loi"], year_from=2005)
    vus = []
    offset = 0
    while True:
        page, total, _ = affiner(ensemble, refine, "date_desc", offset, 20)
        if not page:
            break
        vus.extend(r.law_id for r in page)
        offset += 20
    attendus = [r.law_id for r in ensemble if retenu(r, refine)]
    assert len(vus) == total == len(attendus)
    assert sorted(vus) == sorted(attendus)
    assert len(set(vus)) == len(vus)


def test_les_effectifs_sont_disjonctifs(ensemble):
    """
    L'effectif d'une nature tient compte des autres filtres, pas du sien :
    cocher « loi » rend exactement l'effectif affiche pour « loi », et les
    autres onglets gardent le leur.
    """
    refine = SearchRefine(types=["décret"], category_ids=[13])
    _, total, facettes = affiner(ensemble, refine, None, 0, 20)
    par_nature = {f.value: f.count for f in facettes.types}
    # Sans filtre de nature, toutes les natures du domaine 13 sont comptees.
    assert set(par_nature) == set(NATURES)
    assert par_nature["décret"] == total
    for nature, effectif in par_nature.items():
        _, total_nature, _ = affiner(
            ensemble, SearchRefine(types=[nature], category_ids=[13]), None, 0, 20
        )
        assert total_nature == effectif
    # Les domaines sont comptes parmi les decrets seulement.
    par_domaine = {f.id: f.count for f in facettes.categories}
    for categorie, effectif in par_domaine.items():
        _, total_domaine, _ = affiner(
            ensemble, SearchRefine(types=["décret"], category_ids=[categorie]), None, 0, 20
        )
        assert total_domaine == effectif


def test_effectifs_exhaustifs_et_vide_avec_effectifs(ensemble):
    """
    Un filtre qui ne garde rien rend une page vide, mais les effectifs disent
    ou sont les resultats : l'etat « Aucun décret… les 7 textes trouvés sont… ».
    """
    page, total, facettes = affiner(ensemble, SearchRefine(types=["ordonnance"]), None, 0, 20)
    assert page == [] and total == 0
    assert sum(f.count for f in facettes.types) == len(ensemble)
    assert sum(f.count for f in facettes.years) == 0
    sans_filtre = effectifs(ensemble, None)
    assert sum(f.count for f in sans_filtre.categories) == len(ensemble)
    # L'annee inconnue est une valeur de la facette, en dernier.
    assert sans_filtre.years[-1].value is None
    assert sans_filtre.years[-1].count == sum(1 for r in ensemble if r.publication_date is None)


def test_tri_par_date_stable_et_sans_date_en_dernier():
    a = _resultat(1, jour=date(2020, 5, 1))
    b = _resultat(2, jour=None)
    c = _resultat(3, jour=date(2021, 1, 1))
    d = _resultat(4, jour=date(2020, 5, 1))
    assert [r.law_id for r in trier([a, b, c, d], "date_desc")] == [3, 1, 4, 2]
    assert [r.law_id for r in trier([a, b, c, d], "date_asc")] == [1, 4, 3, 2]
    assert [r.law_id for r in trier([a, b, c, d], "pertinence")] == [1, 2, 3, 4]


def test_filtre_de_periode():
    textes = [_resultat(1, jour=date(2019, 12, 31)), _resultat(2, jour=date(2020, 6, 1)),
              _resultat(3, jour=None)]
    garder = SearchRefine(date_from=date(2020, 1, 1))
    assert [r.law_id for r in textes if retenu(r, garder)] == [2]
    assert [r.law_id for r in textes if retenu(r, SearchRefine(year_to=2019))] == [1]
    # Sans filtre de periode, un texte sans date reste.
    assert [r.law_id for r in textes if retenu(r, SearchRefine(category_ids=[13]))] == [1, 2, 3]


def test_les_natures_se_comparent_sans_accents_ni_espaces():
    assert replier_nature("Acte uniforme") == replier_nature("acte_uniforme") == "acte uniforme"
    assert replier_nature("décret") == replier_nature("DECRET")
    assert retenu(_resultat(1, nature="arrêté"), SearchRefine(types=["arrete"]))


def test_refine_refuse_des_bornes_inversees():
    with pytest.raises(ValueError):
        SearchRefine(year_from=2020, year_to=2019)
    with pytest.raises(ValueError):
        SearchRefine(date_from=date(2020, 2, 1), date_to=date(2020, 1, 1))


def test_cle_de_l_ensemble_ignore_page_tri_et_refine():
    """
    Changer d'onglet ou de page relit le meme ensemble ; changer la requete,
    le mode, le texte vectorise ou les filtres SQL en calcule un autre.
    """
    base = cle_de_l_ensemble("v6", "bail", "hybrid", None, SearchFilters(status="published"), 50)
    assert base == cle_de_l_ensemble("v6", "bail", "hybrid", None, SearchFilters(status="published"), 50)
    for autre in (
        cle_de_l_ensemble("v6", "baux", "hybrid", None, SearchFilters(status="published"), 50),
        cle_de_l_ensemble("v6", "bail", "text", None, SearchFilters(status="published"), 50),
        cle_de_l_ensemble("v6", "bail", "hybrid", "le bail", SearchFilters(status="published"), 50),
        cle_de_l_ensemble("v6", "bail", "hybrid", None, SearchFilters(category_ids=[8]), 50),
        cle_de_l_ensemble("v6", "bail", "hybrid", None, SearchFilters(status="published"), 20),
        cle_de_l_ensemble("v7", "bail", "hybrid", None, SearchFilters(status="published"), 50),
    ):
        assert autre != base
    # En mode texte, le texte vectorise ne sert pas : il n'entre pas dans la cle.
    assert cle_de_l_ensemble("v6", "bail", "text", "x", None, 50) == cle_de_l_ensemble(
        "v6", "bail", "text", None, None, 50
    )


def test_racines_de_la_requete():
    assert racines_de_la_requete("indemnité de licenciement") == ["indemn", "licencie"]
    # Mots vides, mots exclus et nombres seuls ne se surlignent pas.
    assert racines_de_la_requete("le bail -commercial 2018") == ["bail"]
    assert racines_de_la_requete("de la") == []


def test_surligner_marque_les_variantes_et_centre_la_fenetre():
    contenu = (
        "Préambule sans rapport. " * 30
        + "Le travailleur licencié a droit à une indemnité de licenciement distincte du préavis. "
        + "Suite sans rapport. " * 30
    )
    extrait = surligner(contenu, "indemnité de licenciement")
    assert "<mark>licencié</mark>" in extrait
    assert "<mark>indemnité</mark>" in extrait
    assert "<mark>licenciement</mark>" in extrait
    assert extrait.startswith("… ") and extrait.endswith(" …")
    assert len(extrait) < 400
    # Pas de faux ami : « comme » ne passe pas pour « commercial ».
    assert "<mark>" not in (surligner("Il agit comme un mandataire.", "bail commercial") or "")


def test_surligner_rend_none_sans_correspondance():
    assert surligner("Texte sans aucun des mots.", "indemnité") is None
    assert surligner("", "indemnité") is None
    assert surligner("Une indemnité.", "de la") is None
