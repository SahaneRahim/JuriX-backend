"""
POST /api/v1/search/ : recherche affinee (tri, filtres sur l'ensemble,
effectifs) et reperes d'une ligne dense, sur une vraie base PostgreSQL.

Le corpus de test : 26 textes qui portent tous le mot « indemnité » dans un
article, de natures, domaines et dates varies, et un texte trouve par son seul
titre. La recherche plein texte les trouve tous, ce qui permet de verifier
sur l'ensemble entier ce que la page et les effectifs en disent.
"""

from datetime import date

import pytest
from sqlalchemy import text as sa_text

from app.models.law import Article, Law
from app.services.legal_domain_classifier import CIVIL, FONCTION_PUBLIQUE, TRAVAIL
from app.services.search_service import SearchService, ecritures_en_attente

NATURES = ("décret", "loi", "arrêté")


@pytest.fixture
async def corpus(db_session, category_ids):
    domaines = (category_ids[FONCTION_PUBLIQUE], category_ids[TRAVAIL], category_ids[CIVIL])
    lois = []
    for i in range(26):
        nature = NATURES[i % 3]
        jour = None if i == 25 else date(2001 + i % 20, 1 + i % 12, 10)
        lois.append(Law(
            id=100 + i,
            reference=f"PRC-{100 + i}",
            title=(
                f"{nature.capitalize()} N°{2001 + i % 20}/{100 + i} du 10 mars {2001 + i % 20} "
                f"portant organisation du service {i}"
            ),
            type=nature,
            content=f"Texte {i}.",
            language="fr",
            status="published",
            category_id=domaines[i % 3],
            publication_date=jour,
        ))
    # Le texte trouve par son titre seul : sans article en base (un article
    # porte le titre de sa loi en poids A, et serait trouve avec lui), son
    # texte entier porte le mot cherche, loin du debut.
    lois.append(Law(
        id=200,
        reference="CODE-CIVIL",
        title="Loi N°92/007 du 14 août 1992 portant Code du travail et indemnité de rupture",
        type="loi",
        content="REPUBLIQUE DU CAMEROUN. " * 20 + "Le salarié a droit à une indemnité de rupture.",
        language="fr",
        status="published",
        category_id=domaines[1],
        publication_date=date(1992, 8, 14),
    ))
    for loi in lois:
        db_session.add(loi)
    await db_session.flush()
    for loi in lois[:-1]:
        db_session.add(Article(
            law_id=loi.id, number="1", order=1,
            content=f"Le fonctionnaire licencié perçoit une indemnité de licenciement ({loi.id}).",
        ))
    await db_session.commit()
    return lois


async def _chercher(client, **corps):
    reponse = await client.post("/api/v1/search/", json={"query": "indemnité", "mode": "text", **corps})
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


@pytest.mark.asyncio
async def test_le_parcours_classique_ne_change_pas_de_forme(client, corpus):
    """Sans parametre nouveau : pas d'effectifs, pas de tri, comme avant."""
    corps = await _chercher(client, limit=20)
    assert corps["sort"] is None
    assert corps["facets"] is None
    assert corps["candidates"] is None
    assert 0 < len(corps["results"]) <= 20
    # Les reperes d'une ligne dense s'ajoutent, sans rien retirer.
    for cle in ("objet", "numero", "date_signature", "annee", "statut", "highlights", "content"):
        assert cle in corps["results"][0]


@pytest.mark.asyncio
async def test_parcourir_les_pages_filtrees_donne_le_total(client, corpus):
    """
    Filtre puis tri puis page : toutes les pages reunies donnent exactement
    `total` decrets, chacun une fois, par date decroissante.
    """
    vus, offset = [], 0
    while True:
        corps = await _chercher(client, limit=4, offset=offset, sort="date_desc",
                                refine={"types": ["decret"]}, facets=True)
        if not corps["results"]:
            break
        vus.extend(corps["results"])
        offset += 4
    assert len(vus) == corps["total"] == 9
    assert len({r["law_id"] for r in vus}) == 9
    assert all(r["type"] == "décret" for r in vus)
    dates = [r["publication_date"] for r in vus]
    assert dates == sorted(dates, reverse=True)
    assert corps["sort"] == "date_desc"
    assert corps["candidates"] == 27


@pytest.mark.asyncio
async def test_effectifs_disjonctifs_et_coherents(client, corpus, category_ids):
    """Cocher une nature rend exactement l'effectif affiche pour elle."""
    corps = await _chercher(client, facets=True, refine={"category_ids": [category_ids[TRAVAIL]]})
    par_nature = {f["value"]: f["count"] for f in corps["facets"]["types"]}
    for nature, effectif in par_nature.items():
        filtre = await _chercher(client, refine={"types": [nature],
                                                 "category_ids": [category_ids[TRAVAIL]]})
        assert filtre["total"] == effectif
    # Les domaines ignorent leur propre filtre : les trois restent comptes.
    par_domaine = {f["id"]: f["count"] for f in corps["facets"]["categories"]}
    assert set(par_domaine) == {category_ids[FONCTION_PUBLIQUE], category_ids[TRAVAIL], category_ids[CIVIL]}
    assert sum(par_domaine.values()) == corps["candidates"]
    assert corps["total"] == par_domaine[category_ids[TRAVAIL]]


@pytest.mark.asyncio
async def test_page_vide_avec_effectifs(client, corpus):
    """Aucune ordonnance : page vide, mais les effectifs disent ou chercher."""
    corps = await _chercher(client, facets=True, refine={"types": ["ordonnance"]})
    assert corps["results"] == [] and corps["total"] == 0
    assert sum(f["count"] for f in corps["facets"]["types"]) == corps["candidates"] == 27


@pytest.mark.asyncio
async def test_filtre_de_periode_et_texte_sans_date(client, corpus):
    corps = await _chercher(client, refine={"year_from": 2015, "year_to": 2018}, facets=True, limit=50)
    assert corps["total"] > 0
    assert all(2015 <= int(r["publication_date"][:4]) <= 2018 for r in corps["results"])
    annees = {f["value"]: f["count"] for f in corps["facets"]["years"]}
    # La facette des annees ignore le filtre de periode, et compte le texte sans date.
    assert annees[None] == 1
    assert sum(annees.values()) == corps["candidates"]


@pytest.mark.asyncio
async def test_l_ensemble_est_calcule_une_fois_puis_relu(client, corpus, db_session, monkeypatch):
    """Changer de page, de tri ou de filtre relit l'ensemble en cache."""
    appels = []
    origine = SearchService._execute_search_by_mode

    async def compter(self, request):
        appels.append(request.limit)
        return await origine(self, request)

    monkeypatch.setattr(SearchService, "_execute_search_by_mode", compter)
    premier = await _chercher(client, facets=True)
    await ecritures_en_attente()
    deuxieme = await _chercher(client, offset=10, sort="date_asc", refine={"types": ["loi"]})
    assert appels == [SearchService.FENETRE_ENSEMBLE]
    assert premier["candidates"] == deuxieme["candidates"]
    lignes = (await db_session.execute(sa_text("SELECT count(*) FROM query_cache"))).scalar()
    assert lignes == 1


@pytest.mark.asyncio
async def test_reperes_d_une_ligne_dense(client, corpus):
    corps = await _chercher(client, facets=True, limit=50)
    par_id = {r["law_id"]: r for r in corps["results"]}
    ligne = par_id[103]
    assert ligne["objet"] == "Organisation du service 3"
    assert ligne["numero"] == "2004/103"
    assert ligne["date_signature"] == "2004-04-10"
    assert ligne["annee"] == 2004
    assert ligne["statut"] is None
    # Statut prouve : celui du Code civil (liens_entre_textes).
    code = par_id[200]
    assert code["statut"]["natures"] == {"abroge_partiellement": 11, "complete": 1}
    assert code["statut"]["abroge"] is False


@pytest.mark.asyncio
async def test_extrait_surligne_pour_une_correspondance_de_titre(client, corpus):
    """
    Trouve par son titre, le texte n'a pas d'extrait marque par ts_headline :
    la recherche affinee en refait un autour du mot cherche.
    """
    classique = {r["law_id"]: r for r in (await _chercher(client, limit=50))["results"]}
    affinee = {r["law_id"]: r for r in (await _chercher(client, facets=True, limit=50))["results"]}
    assert "<mark>" not in classique[200]["highlights"].get("content", "")
    assert "<mark>indemnité</mark>" in affinee[200]["highlights"]["content"]


@pytest.mark.asyncio
async def test_parametres_invalides(client, corpus):
    reponse = await client.post("/api/v1/search/", json={"query": "x", "sort": "au_hasard"})
    assert reponse.status_code == 422
    reponse = await client.post(
        "/api/v1/search/", json={"query": "x", "refine": {"year_from": 2020, "year_to": 2010}}
    )
    assert reponse.status_code == 422
