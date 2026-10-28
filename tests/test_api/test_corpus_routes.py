"""
GET /api/v1/corpus : les chiffres du corpus, pour l'accueil du chat (« Je
réponds à partir de 2 275 textes officiels camerounais et OHADA… ») et la page
« Aucun résultat » de la recherche.

Usage:
    pytest tests/test_api/test_corpus_routes.py -v
"""

import re
from datetime import date, datetime

import pytest

from app.core import mesure_sql
from app.models.law import Article, Law
from app.services.chiffres_du_corpus import invalider_cache_corpus
from app.services.liens_entre_textes import RELEVE_LE
from app.services.statut_juridique import invalider_cache_statut


@pytest.fixture(autouse=True)
def _caches_vides():
    """Les caches sont globaux au processus : chaque test part de caches vides."""
    invalider_cache_corpus()
    invalider_cache_statut()
    yield
    invalider_cache_corpus()
    invalider_cache_statut()


@pytest.fixture
async def compteur(pg_engine):
    """Le compteur de Server-Timing, pose sur le moteur de test comme en production."""
    mesure_sql.brancher(pg_engine.sync_engine)
    try:
        yield
    finally:
        mesure_sql.debrancher(pg_engine.sync_engine)


def _instructions(reponse) -> int:
    trouve = re.search(r'db;desc="n=(\d+)', reponse.headers["server-timing"])
    assert trouve, reponse.headers["server-timing"]
    return int(trouve.group(1))


async def _loi(db, reference, titre, morceaux=(), **valeurs):
    loi = Law(
        reference=reference, title=titre, type=valeurs.pop("type", "loi"),
        content="Texte intégral du document.", language="fr",
        status=valeurs.pop("status", "published"), **valeurs,
    )
    db.add(loi)
    await db.flush()
    for ordre, (numero, contenu, *kind) in enumerate(morceaux):
        db.add(Article(law_id=loi.id, number=numero, content=contenu,
                       kind=kind[0] if kind else "article", order=ordre))
    await db.commit()
    return loi


async def test_les_chiffres_du_corpus(client, db_session):
    await _loi(db_session, "PRC-1", "Décret N°2026/282 du 31 août 2026 portant nomination", [
        ("LEGAL_BASIS", "Vu la Constitution", "legal_basis"),
        ("1", "Sont nommés :"), ("2.1", "- M. A"), ("2.2", "- M. B"), ("3", "Publié."),
    ], type="décret", publication_date=date(2026, 8, 31))
    await _loi(db_session, "OHADA-AUS-2010", "Acte uniforme révisé portant organisation des sûretés",
               [("1", "Champ d'application.")], type="acte uniforme", publication_date=date(2010, 12, 15))
    await _loi(db_session, "LOI-1992-007", "Loi N°92/007 du 14 août 1992 portant Code du travail",
               [("1", "La présente loi régit les relations de travail.")], publication_date=date(1992, 8, 14))
    await _loi(db_session, "PRC-2", "Décret refusé", [("1", "Doublon.")], type="décret", status="refused")

    reponse = await client.get("/api/v1/corpus")

    assert reponse.status_code == 200
    assert reponse.headers["cache-control"] == "public, max-age=300, s-maxage=600, stale-while-revalidate=3600"
    corps = reponse.json()
    assert corps["textes"] == 3, "seuls les textes publies"
    # 1, 2.1 et 3 du decret (2.2 est la suite de 2.1, les visas ne sont pas un article), puis 1 et 1.
    assert corps["articles"] == 5
    assert corps["textes_par_nature"] == [
        {"nature": "acte uniforme", "textes": 1}, {"nature": "décret", "textes": 1}, {"nature": "loi", "textes": 1},
    ]
    recent = corps["texte_le_plus_recent"]
    assert recent["numero"] == "2026/282" and recent["date_signature"] == "2026-08-31"
    assert recent["consultable"] is True
    assert corps["dernier_ajout_le"] == date.today().isoformat()
    assert corps["liens_releves_le"] == RELEVE_LE.isoformat()
    assert [(s["code"], s["textes"]) for s in corps["sources"]] == [("prc", 1), ("ohada", 1), ("autres", 1)]
    assert corps["sources"][0]["libelle"]["fr"] == "Présidence de la République du Cameroun (prc.cm)"
    assert corps["libelle_sources"] == {
        "fr": "textes officiels camerounais et OHADA", "en": "official texts of Cameroon and OHADA",
    }
    datetime.fromisoformat(corps["calcule_le"])


async def test_un_corpus_vide_repond_sans_erreur(client):
    corps = (await client.get("/api/v1/corpus")).json()

    assert (corps["textes"], corps["articles"], corps["texte_le_plus_recent"]) == (0, 0, None)
    assert corps["sources"] == [] and corps["dernier_ajout_le"] is None
    assert corps["libelle_sources"]["fr"] == "textes officiels camerounais"


async def test_les_chiffres_du_corpus_sont_gardes_en_memoire(client, db_session, compteur):
    await _loi(db_session, "PRC-1", "Décret N°2026/282 du 31 août 2026 portant", [("1", "Texte.")], type="décret")

    froid = await client.get("/api/v1/corpus")
    await _loi(db_session, "PRC-2", "Décret N°2026/283 du 31 août 2026 portant", [("1", "Texte.")], type="décret")
    chaud = await client.get("/api/v1/corpus")
    invalider_cache_corpus()
    recalcule = await client.get("/api/v1/corpus")

    # Textes et dernier ajout, articles, natures par source, texte le plus recent.
    assert _instructions(froid) == 4
    assert _instructions(chaud) == 0 and chaud.json() == froid.json()
    assert froid.json()["textes"] == 1 and recalcule.json()["textes"] == 2
    assert (await client.get("/api/v1/corpus/")).status_code == 200
