"""
Les liens d'historique (liens_entre_textes.LIENS_HISTORIQUES) : bien formes,
et coherents avec les liens de l'etat actuel.

Un lien d'historique ne sert qu'a la page d'un texte (GET /laws/{id}) : il dit
qu'un article a ete reecrit par un texte que d'autres ont depuis depasse, ou que
le document en base integre deja un changement. Mal pose, il ferait dire a la
page qu'un texte a touche des articles qu'il n'a pas touches. Leur ancrage en
base se controle avec :
    python scripts/maintenance/verifier_liens_textes.py --historiques

Usage:
    pytest tests/test_services/test_liens_historiques.py -v
"""

import re

import pytest

from app.services import textes_successifs
from app.services.liens_entre_textes import LIENS, LIENS_HISTORIQUES, RELEVE_LE
from app.services.textes_successifs import NATURES
from app.utils.chunk_refiner import est_pseudo_numero

_UN_ARTICLE = re.compile(r"\d+(?:\s*(?:er|bis|ter|quater|-\d+))?|\d+\s*à\s*\d+")


@pytest.mark.parametrize("historique", LIENS_HISTORIQUES, ids=lambda h: f"{h.lien.texte}<{h.lien.par_titre[:30]}")
def test_chaque_lien_historique_est_bien_forme(historique):
    """Memes exigences que les liens de LIENS, plus : depasse OU integre, jamais les deux."""
    lien = historique.lien
    assert lien.nature in NATURES
    assert lien.texte_nom[:1].islower() and lien.par_titre[:1].islower()
    assert str(lien.par_date.year) in lien.par_titre
    assert not lien.portee.endswith(".")
    for article in lien.articles:
        assert _UN_ARTICLE.fullmatch(article), article
    for morceau in lien.par_articles:
        assert re.fullmatch(r"\d+", morceau) or est_pseudo_numero(morceau), morceau
    for morceau, article, repere in lien.morceaux:
        assert textes_successifs._designe(lien.articles, article) and repere.strip()

    assert bool(historique.depasse_par_titre) != historique.integre_au_texte
    if historique.depasse_par_titre:
        assert historique.depasse_le > lien.par_date
        assert str(historique.depasse_le.year) in historique.depasse_par_titre


@pytest.mark.parametrize(
    "historique",
    [h for h in LIENS_HISTORIQUES if h.depasse_par_titre],
    ids=lambda h: f"{h.lien.texte}<{h.lien.par_titre[:30]}",
)
def test_un_lien_depasse_l_est_par_un_lien_saisi_sur_les_memes_articles(historique):
    """Le texte qui depasse a lui-meme un lien, dans LIENS ou dans l'historique, sur ces articles."""
    lien = historique.lien
    tous = list(LIENS) + [h.lien for h in LIENS_HISTORIQUES]
    suivants = [
        autre for autre in tous
        if autre.texte == lien.texte
        and (autre.par_reference or autre.par_titre)
        == (historique.depasse_par_reference or historique.depasse_par_titre)
    ]
    assert suivants, "aucun lien du texte qui depasse"
    for article in lien.articles:
        assert any(textes_successifs._designe(autre.articles, article) for autre in suivants), article


def test_aucun_lien_historique_ne_double_un_lien_saisi():
    cles = {(lien.texte, lien.par_reference or lien.par_titre, lien.articles) for lien in LIENS}

    for historique in LIENS_HISTORIQUES:
        lien = historique.lien
        assert (lien.texte, lien.par_reference or lien.par_titre, lien.articles) not in cles


def test_l_historique_est_anterieur_au_releve():
    """Un lien d'historique est un fait passe : rien de posterieur au releve."""
    for historique in LIENS_HISTORIQUES:
        assert historique.lien.par_date <= RELEVE_LE
        assert historique.depasse_le is None or historique.depasse_le <= RELEVE_LE
