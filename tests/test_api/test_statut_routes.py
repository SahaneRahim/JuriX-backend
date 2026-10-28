"""
Statut juridique et dates d'un texte (GET /laws/{id}).

Tout est ADDITIF : le front deploye lit GET /laws/{id} tel qu'avant. Le premier
test le verifie champ par champ.

Les textes de ces tests portent les cotes reelles du releve des liens
(PRC-6283 pour la Constitution) : le statut se calcule sur les vrais liens,
avec les identifiants de la base de test.

Usage:
    pytest tests/test_api/test_statut_routes.py -v
"""

import re
from datetime import date, datetime

import pytest
from sqlalchemy import text

from app.core import mesure_sql
from app.models.law import Article, Law
from app.services.liens_entre_textes import RELEVE_LE
from app.services.statut_juridique import invalider_cache_statut


@pytest.fixture(autouse=True)
def _caches_vides():
    """Le cache des textes lies est global au processus : chaque test part d'un cache vide."""
    invalider_cache_statut()
    yield
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
        content=valeurs.pop("content", "Texte intégral du document."), language="fr",
        status=valeurs.pop("status", "published"), **valeurs,
    )
    db.add(loi)
    await db.flush()
    for ordre, morceau in enumerate(morceaux):
        numero, contenu = morceau[0], morceau[1]
        section = morceau[2] if len(morceau) > 2 else None
        kind = morceau[3] if len(morceau) > 3 else "article"
        db.add(Article(law_id=loi.id, number=numero, content=contenu, section=section,
                       kind=kind, order=ordre))
    await db.commit()
    return loi


@pytest.fixture
async def constitution(db_session):
    """La Constitution, la loi 2008/001 et la loi 2026/002, avec les morceaux que lit le releve."""
    revision_2008 = await _loi(
        db_session, "PRC-6284",
        "Loi N°2008/001 du 14 avril 2008 modifiant et complétant certaines dispositions de la Constitution",
        [("DISPOSITIF", "«ARTICLE 6. (2) (nouveau) : Le Président de la République est rééligible."),
         ("53", "(nouveau) (1) La Haute Cour de Justice est compétente."),
         ("SIGNATURE", "Yaoundé, le 14 avril 2008")],
        publication_date=date(2008, 4, 14),
    )
    revision_2026 = await _loi(
        db_session, "PRC-10679",
        "Loi N°2026/002 du 14 avril 2026 modifiant et complétant certaines dispositions de la Constitution",
        [("1", "Les dispositions des articles 5, 6, 7, 10, 53 et 66 sont modifiées."),
         ("6", "(nouveau) (5) le Vice-Président achève le mandat.")],
        publication_date=date(2026, 4, 14),
    )
    loi = await _loi(
        db_session, "PRC-6283",
        "Loi N°96/06 du 18 janvier 1996 portant révision de la Constitution du 02 juin 1972.",
        [("PREAMBULE", "Le peuple camerounais…"),
         ("5", "(1) Le Président de la République est le Chef de l'État.", "TITRE PREMIER"),
         ("6", "(1) Le Président de la République est élu au suffrage universel direct.", "TITRE PREMIER"),
         ("SECTION_2", "b) le Président de la République par intérim ne peut modifier la Constitution.",
          "CHAPITRE 1"),
         ("8", "(1) Le Président de la République représente l'État.", "CHAPITRE 1"),
         ("53", "(1) La Haute Cour de Justice est compétente.", "TITRE VIII")],
        publication_date=date(1996, 1, 18),
    )
    return {"loi": loi, "2008": revision_2008, "2026": revision_2026}


# ==================== GET /laws/{id} : COMPATIBILITE ====================

ANCIENS_CHAMPS = {
    "id", "reference", "title", "type", "content", "language", "category_id", "status",
    "publication_date", "detected_language", "language_confidence", "suggested_categories",
    "category_confidence", "file_id", "original_filename", "article_count", "created_at",
    "updated_at", "category", "articles",
}
NOUVEAUX_CHAMPS = {"statut_juridique", "date_signature", "annee", "date_publication_jo", "ajoute_le"}


async def test_le_detail_garde_ses_champs_et_n_en_ajoute_que_de_nouveaux(client, db_session):
    loi = await _loi(db_session, "PRC-1740",
                     "Décret N°2021/745 du 28 décembre 2021 fixant les modalités d'exercice",
                     [("1", "Le présent décret fixe les modalités.")],
                     type="décret", publication_date=date(2021, 12, 28))

    reponse = await client.get(f"/api/v1/laws/{loi.id}")

    assert reponse.status_code == 200
    corps = reponse.json()
    assert set(corps) == ANCIENS_CHAMPS | NOUVEAUX_CHAMPS
    assert corps["publication_date"] == "2021-12-28" and corps["content"] == "Texte intégral du document."
    assert corps["article_count"] == 1 and corps["articles"][0]["number"] == "1"


async def test_les_deux_dates_sont_nommees_sans_ambiguite(client, db_session):
    """« Signé le 28 décembre 2021 » et « Ajouté à JuriX le … » : jamais la date de mise à jour de la ligne."""
    loi = await _loi(db_session, "PRC-1740", "Décret N°2021/745 du 28 décembre 2021 fixant",
                     type="décret", publication_date=date(2021, 12, 28),
                     created_at=datetime(2026, 10, 3, 14, 52), updated_at=datetime(2026, 10, 14, 3, 0))

    corps = (await client.get(f"/api/v1/laws/{loi.id}")).json()

    assert corps["date_signature"] == "2021-12-28" and corps["annee"] == 2021
    assert corps["ajoute_le"] == "2026-10-03"
    assert corps["date_publication_jo"] is None
    # Les anciens champs, inchanges, pour le front deploye.
    assert corps["publication_date"] == "2021-12-28" and corps["updated_at"].startswith("2026-10-14")


async def test_un_premier_janvier_de_remplissage_ne_donne_que_l_annee(client, db_session):
    loi = await _loi(db_session, "PRC-9999", "Décret N°2024/193 juin 10", type="décret",
                     publication_date=date(2024, 1, 1))

    corps = (await client.get(f"/api/v1/laws/{loi.id}")).json()

    assert corps["date_signature"] is None and corps["annee"] == 2024
    assert corps["publication_date"] == "2024-01-01"


async def test_une_date_de_journal_officiel_n_est_pas_dite_de_signature(client, db_session):
    loi = await _loi(
        db_session, "OHADA-AUPSRVE-2023",
        "Acte uniforme portant organisation des procédures simplifiées de recouvrement et des voies "
        "d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023",
        type="acte uniforme", publication_date=date(2023, 11, 15),
    )

    corps = (await client.get(f"/api/v1/laws/{loi.id}")).json()

    assert corps["date_signature"] is None
    assert corps["date_publication_jo"] == "2023-11-15" and corps["annee"] == 2023


# ==================== GET /laws/{id} : STATUT ====================

async def test_un_texte_sans_lien_dit_seulement_qu_aucune_modification_n_est_relevee(client, db_session, compteur):
    loi = await _loi(db_session, "PRC-1740", "Décret N°2021/745 du 28 décembre 2021 fixant",
                     [("1", "Le présent décret fixe les modalités.")], type="décret")

    reponse = await client.get(f"/api/v1/laws/{loi.id}")

    statut = reponse.json()["statut_juridique"]
    assert statut["etat"] == "aucune_modification_connue"
    assert statut["releve_le"] == RELEVE_LE.isoformat()
    assert statut["modifie_par"] == [] and statut["par_article"] == {}
    # La loi et ses articles, comme avant : aucune requete de plus pour un texte sans lien.
    assert _instructions(reponse) == 2


async def test_la_constitution_montre_ses_textes_modificatifs_et_l_historique(client, constitution):
    loi = constitution["loi"]

    statut = (await client.get(f"/api/v1/laws/{loi.id}")).json()["statut_juridique"]

    assert statut["etat"] == "modifie" and statut["abroge"] is False
    assert statut["types"] == ["modifie", "complete"]
    changements = {c["id"]: c for c in statut["modifie_par"]}
    ids = {c["texte"]["numero"]: c["texte"]["id"] for c in changements.values()}
    assert ids == {"2008/001": constitution["2008"].id, "2026/002": constitution["2026"].id}

    chaine = [changements[i] for i in statut["par_article"]["6"]]
    assert [(c["texte"]["numero"], c["depasse"]) for c in chaine] == [("2008/001", True), ("2026/002", False)]
    assert chaine[0]["depasse_par"]["id"] == constitution["2026"].id
    assert chaine[1]["nature_modification"] == "reecrit_en_entier"


async def test_la_fin_de_l_article_6_rangee_ailleurs_porte_son_statut(client, constitution, db_session):
    """Le morceau SECTION_2 (repere « par intérim ») porte la fin de l'article 6 : il est marque."""
    loi = constitution["loi"]
    numeros = dict(
        (await db_session.execute(text("SELECT number, id FROM articles WHERE law_id = :id"), {"id": loi.id})).all()
    )

    statut = (await client.get(f"/api/v1/laws/{loi.id}")).json()["statut_juridique"]

    par_morceau = statut["par_morceau"]
    assert par_morceau[str(numeros["SECTION_2"])] == par_morceau[str(numeros["6"])]
    assert str(numeros["8"]) not in par_morceau and str(numeros["PREAMBULE"]) not in par_morceau


async def test_le_texte_modificatif_dit_ce_qu_il_modifie(client, constitution):
    revision = constitution["2008"]

    statut = (await client.get(f"/api/v1/laws/{revision.id}")).json()["statut_juridique"]

    cibles = {c["texte"]["id"] for c in statut["modifie"]}
    assert cibles == {constitution["loi"].id}
    assert statut["etat"] == "modifie", "la loi 2026/002 a reecrit les articles 6 et 53 de la loi de 2008"


async def test_un_statut_en_echec_ne_prive_pas_du_texte(client, constitution, monkeypatch):
    async def _panne(*args, **kwargs):
        raise RuntimeError("panne simulee")

    monkeypatch.setattr("app.api.routes.laws.statut_du_texte", _panne)

    reponse = await client.get(f"/api/v1/laws/{constitution['loi'].id}")

    assert reponse.status_code == 200
    assert reponse.json()["statut_juridique"] is None
    assert reponse.json()["title"].startswith("Loi N°96/06")


async def test_les_textes_lies_sont_lus_une_fois(client, constitution, compteur):
    """Une requete pour les textes lies, gardee en memoire ; puis plus rien."""
    loi = constitution["loi"]

    froid = await client.get(f"/api/v1/laws/{loi.id}")
    chaud = await client.get(f"/api/v1/laws/{loi.id}")

    assert froid.json() == chaud.json()
    # Loi, articles, contenu des morceaux a repere (SECTION_2), textes lies.
    assert _instructions(froid) == 4
    assert _instructions(chaud) == 3
