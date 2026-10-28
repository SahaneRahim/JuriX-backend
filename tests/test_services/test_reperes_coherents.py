"""
Un meme texte, decrit de la meme facon partout ou le front l'affiche.

Trois branches ont ajoute, chacune de son cote, les reperes d'un texte : la
page d'un texte et les textes nommes par un statut (app/schemas/law.py,
app/services/statut_juridique.py), les lignes de liste et de recherche
(app/schemas/reperes.py), le chat (app/services/rag_service.py). Fusionnees,
elles se contredisaient sur trois points, que ces tests fixent :

- la DATE : deux Actes uniformes OHADA dont l'intitule ne date que la
  publication au Journal officiel etaient « signés le 15 novembre 2023 » en
  liste et sans date de signature sur leur page ; le chat montrait la date
  brute en base, 1er janvier de remplissage compris ;
- le NUMERO d'acte : deux lectures de l'intitule, dont une seule repare l'OCR
  (« N°202l/345 ») ;
- le STATUT en bref d'une ligne de liste comptait des liens que la page du
  texte range dans l'historique (loi 2013/004).

Fonctions pures et doublures : ni base, ni reseau.
"""

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from app.schemas.law import LawDetailResponse
from app.schemas.law_list import LawListLightItem
from app.services.liens_entre_textes import LIENS
from app.services.prompts import _date_du_texte
from app.services.rag_service import _date_affichee
from app.services.reperes_textes import dates_de_l_acte
from app.services.statut_en_bref import statut_en_bref
from app.services.statut_juridique import InfoTexte, construire_statut, texte_lie
from app.utils.titre_officiel import numero_du_titre

pytestmark = pytest.mark.unit

_FIXTURE_TITRES = Path(__file__).resolve().parents[1] / "fixtures" / "titres_objets.json"

AUPSRVE = (
    "Acte uniforme portant organisation des procédures simplifiées de recouvrement et des voies "
    "d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023"
)

# (date en base, intitule) : les cas ou une regle de date peut diverger.
CAS_DE_DATES = [
    (date(2023, 11, 15), AUPSRVE),
    (date(2024, 1, 1), "Décret N°2024/193 juin 10 portant avancement d'échelon"),
    (date(2015, 1, 1), "Décret N°2015/405 fixant les modalités de rémunération"),
    (date(2020, 1, 1), "Décret N°2020/001 du 1er janvier 2020 portant convocation"),
    (date(2020, 1, 1), "Décret N°2020/001 du 01 JANV. 2020 portant convocation"),
    (date(2021, 12, 28), "Décret N°2021/745 du 28 décembre 2021 fixant les modalités"),
    (None, "Code civil applicable au Cameroun"),
]


def _ligne(jour, titre):
    return LawListLightItem(
        id=1, reference="PRC-1", title=titre, type="décret", status="published",
        publication_date=jour,
    )


def _page(jour, titre):
    return LawDetailResponse(
        id=1, reference="PRC-1", title=titre, type="décret", status="published",
        content="Contenu du texte.", publication_date=jour,
        created_at=datetime(2026, 10, 3, 12, 0),
    )


# ==================== DATES ====================


@pytest.mark.parametrize("jour,titre", CAS_DE_DATES)
def test_la_liste_et_la_page_datent_le_texte_de_la_meme_facon(jour, titre):
    ligne = _ligne(jour, titre).model_dump(mode="json")
    page = _page(jour, titre).model_dump(mode="json")

    for champ in ("date_signature", "annee", "date_publication_jo"):
        assert ligne[champ] == page[champ], champ


def test_une_date_de_journal_officiel_n_est_pas_signee_en_liste():
    ligne = _ligne(date(2023, 11, 15), AUPSRVE)

    assert ligne.date_signature is None
    assert ligne.annee == 2023
    assert ligne.date_publication_jo == date(2023, 11, 15)


@pytest.mark.parametrize("jour,titre", [c for c in CAS_DE_DATES if c[0] is not None])
def test_le_chat_date_comme_la_page(jour, titre):
    """`date_du_texte` des citations, sources et textes trouves du chat."""
    assert _date_affichee(jour, titre) == dates_de_l_acte(jour, titre).date_signature


@pytest.mark.parametrize("jour,titre", [c for c in CAS_DE_DATES if c[0] is not None])
def test_le_modele_lit_l_annee_seule_quand_la_page_n_a_qu_elle(jour, titre):
    """
    Le contexte du modele et la page appliquent la meme regle du 1er janvier
    (« 01 JANV. 2020 » est une vraie date pour les deux). La date de Journal
    officiel seule reste lue telle quelle par le modele : il la voit dans
    l'intitule.
    """
    dates = dates_de_l_acte(jour, titre)
    if dates.date_publication_jo is not None:
        return
    extrait = type("Extrait", (), {"publication_date": jour, "law_title": titre})()
    annee_seule = "jour non indiqué" in _date_du_texte(extrait)

    assert annee_seule == (dates.date_signature is None)


# ==================== NUMERO D'ACTE ====================


def _titres_du_corpus():
    return [cas["titre"] for cas in json.loads(_FIXTURE_TITRES.read_text())["cas"]]


def test_un_texte_lie_porte_le_numero_de_sa_ligne_de_liste():
    """233 intitules reels relus a la main, plus les fautes d'OCR reparees en liste."""
    titres = _titres_du_corpus() + [
        "Décret N°202l/345 du 17 juin 2021 portant révocation d'un Magistrat",
        "Arrêté N°0095/CAB/PR/du 16 février 2026 portant nomination de trente-huit (38) Inspecteurs",
        "Arrêté N1078/CAB/PR du 31 décembre 2022 portant avancement de grade au choix",
    ]
    assert len(titres) > 200
    for titre in titres:
        infos = {"PRC-1": InfoTexte(1, "PRC-1", "décret", titre, date(2021, 6, 17), "published")}
        lie = texte_lie("PRC-1", "le texte", None, infos)
        assert lie.numero == numero_du_titre(titre) == _ligne(None, titre).numero, titre


def test_un_texte_absent_du_corpus_garde_son_numero():
    """Intitules du releve, ecrits pour une phrase : l'article retire, le numero lu."""
    assert texte_lie(None, "la Loi n° 2012/015 du 21 décembre 2012", date(2012, 12, 21), {}).numero == (
        "2012/015"
    )
    # La nature n'ouvre pas l'intitule : seule la seconde lecture trouve le numero.
    constitution = "la Constitution (Loi N°96/06 du 18 janvier 1996 portant révision)"
    assert texte_lie(None, constitution, date(1996, 1, 18), {}).numero == "96/06"


# ==================== STATUT EN BREF ====================


def _actuels(statut):
    return [c for c in statut.modifie_par if not c.depasse and not c.integre_au_texte]


@pytest.mark.parametrize("reference", sorted({lien.texte for lien in LIENS}))
def test_la_ligne_de_liste_resume_ce_que_la_page_dit_actuel(reference):
    """Pour chaque texte du releve : meme abrogation, memes textes, memes natures."""
    bref = statut_en_bref(reference)
    statut = construire_statut(reference, {})
    actuels = _actuels(statut)
    par_nature = {}
    for changement in actuels:
        qui = changement.texte.reference or changement.texte.titre
        par_nature.setdefault(changement.type, set()).add(qui)

    assert bref is not None
    assert bref.abroge == statut.abroge
    assert bref.textes == len({c.texte.reference or c.texte.titre for c in actuels})
    assert bref.natures == {nature: len(qui) for nature, qui in sorted(par_nature.items())}


def test_la_loi_2013_004_est_remplacee_sans_ses_reecritures_de_2017():
    """
    L'ordonnance 2025/002 remplace la loi 2013/004 en entier : les articles que
    la loi 2017/015 avait reecrits sont de l'historique, sur la page comme en
    liste.
    """
    bref = statut_en_bref("LOI-2013-004")

    assert bref.abroge is True
    assert bref.natures == {"remplace": 1}
    assert bref.textes == 1
    assert bref.date_dernier_texte == date(2025, 7, 18)
