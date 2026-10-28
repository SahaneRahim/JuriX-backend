"""
Ce que l'acceleration d'octobre 2026 a change dans les reponses HTTP, verifie
route par route : cache des categories, listes de lois allegees, reponse de
recherche sans chunks, prealable CORS garde deux heures.

Chaque test lit aussi, quand il le faut, le nombre d'instructions SQL de la
requete dans l'en-tete Server-Timing (app/core/mesure_sql.py) : c'est la
mesure meme qui a guide ces changements (allers-retours, octets tires de la
base), et la seule qu'un retour en arriere ferait bouger sans casser une
reponse.
"""

import re
from datetime import date

import pytest
from sqlalchemy import text

from app.core import mesure_sql
from app.models.law import Article, Category, Law
from app.services.legal_domain_classifier import CIVIL, PENAL

ORIGINE_DU_FRONT = "https://juri-x-frontend.vercel.app"


@pytest.fixture
async def compteur(pg_engine):
    """Le compteur de Server-Timing, pose sur le moteur de test comme en production."""
    mesure_sql.brancher(pg_engine.sync_engine)
    try:
        yield
    finally:
        mesure_sql.debrancher(pg_engine.sync_engine)


def _instructions(reponse) -> int:
    """Le n= de la metrique db de Server-Timing."""
    trouve = re.search(r'db;desc="n=(\d+)', reponse.headers["server-timing"])
    assert trouve, reponse.headers["server-timing"]
    return int(trouve.group(1))


@pytest.fixture(autouse=True)
def _cache_des_categories_vide():
    """Le cache est global au processus : chaque test part d'un cache vide."""
    from app.api.routes.categories import invalider_cache_categories

    invalider_cache_categories()
    yield
    invalider_cache_categories()


@pytest.fixture
async def lois(db_session, category_ids):
    """Deux lois au texte long, l'une avec trois articles."""
    texte = "Le bail commercial obeit aux regles suivantes. " * 120  # 5 640 caracteres
    civile = Law(
        reference="LOI-2020-001", title="Loi sur le bail commercial", type="loi",
        content=texte, language="fr", status="published",
        category_id=category_ids[CIVIL], publication_date=date(2020, 1, 1),
    )
    penale = Law(
        reference="LOI-2020-002", title="Code penal", type="loi",
        content="Les infractions sont classees. " * 50, language="fr",
        status="published", category_id=category_ids[PENAL],
        publication_date=date(2020, 2, 1),
    )
    db_session.add_all([civile, penale])
    await db_session.flush()
    db_session.add_all([
        Article(law_id=civile.id, number=str(i), content=f"Article {i} du bail.", order=i)
        for i in range(1, 4)
    ])
    await db_session.commit()
    return {"civile": civile, "penale": penale, "texte": texte}


# ==================== CATEGORIES ====================


class TestCacheDesCategories:
    """
    /categories/{id} chargeait la categorie, TOUTES ses lois puis tous leurs
    articles pour rendre 206 octets (4 instructions, 7,4 s en production).
    """

    @pytest.mark.asyncio
    async def test_la_liste_se_lit_une_fois_puis_depuis_la_memoire(self, client, compteur):
        premiere = await client.get("/api/v1/categories")
        seconde = await client.get("/api/v1/categories")

        assert premiere.status_code == seconde.status_code == 200
        assert premiere.json() == seconde.json()
        # Une instruction (le GROUP BY des comptes), puis aucune.
        assert (_instructions(premiere), _instructions(seconde)) == (1, 0)

    @pytest.mark.asyncio
    async def test_le_detail_compte_les_lois_sans_les_charger(self, client, compteur, lois):
        categorie = lois["civile"].category_id

        reponse = await client.get(f"/api/v1/categories/{categorie}")

        assert reponse.status_code == 200
        assert reponse.json()["law_count"] == 1
        assert _instructions(reponse) == 1
        # Puis depuis la memoire.
        assert _instructions(await client.get(f"/api/v1/categories/{categorie}")) == 0

    @pytest.mark.asyncio
    async def test_les_reponses_reussies_portent_cache_control(self, client, lois):
        from app.api.routes.categories import CACHE_CONTROL_CATEGORIES

        liste = await client.get("/api/v1/categories")
        detail = await client.get(f"/api/v1/categories/{lois['civile'].category_id}")
        absente = await client.get("/api/v1/categories/999999")

        assert liste.headers["cache-control"] == CACHE_CONTROL_CATEGORIES
        assert detail.headers["cache-control"] == CACHE_CONTROL_CATEGORIES
        assert "public" not in absente.headers.get("cache-control", "")
        assert "s-maxage=" in CACHE_CONTROL_CATEGORIES

    @pytest.mark.asyncio
    async def test_une_loi_ajoutee_par_l_orm_invalide_le_cache(
        self, client, db_session, category_ids
    ):
        categorie = category_ids[PENAL]
        avant = await client.get(f"/api/v1/categories/{categorie}")
        assert avant.json()["law_count"] == 0

        db_session.add(Law(
            reference="LOI-2021-009", title="Loi nouvelle", type="loi",
            content="Texte.", language="fr", status="published", category_id=categorie,
        ))
        await db_session.commit()

        apres = await client.get(f"/api/v1/categories/{categorie}")
        assert apres.json()["law_count"] == 1

    @pytest.mark.asyncio
    async def test_une_loi_qui_change_de_categorie_invalide_le_cache(
        self, client, db_session, lois, category_ids
    ):
        await client.get("/api/v1/categories")
        lois["penale"].category_id = category_ids[CIVIL]
        await db_session.commit()

        comptes = {c["id"]: c["law_count"] for c in (await client.get("/api/v1/categories")).json()}
        assert comptes[category_ids[CIVIL]] == 2
        assert comptes[category_ids[PENAL]] == 0

    @pytest.mark.asyncio
    async def test_une_categorie_ecrite_hors_orm_est_lue_en_base(self, client, db_session):
        """Un script de maintenance (SQL brut) : le cache l'ignore, la base tranche."""
        await client.get("/api/v1/categories")
        nouvelle = (await db_session.execute(text(
            "INSERT INTO categories (name, description, display_order) "
            "VALUES ('Ecrite a la main', 'x', 99) RETURNING id"
        ))).scalar_one()
        await db_session.commit()

        reponse = await client.get(f"/api/v1/categories/{nouvelle}")

        assert reponse.status_code == 200
        assert reponse.json()["name"] == "Ecrite a la main"
        # Le cache, en retard, a ete oublie : la liste la montre aussi.
        noms = {c["name"] for c in (await client.get("/api/v1/categories")).json()}
        assert "Ecrite a la main" in noms

    @pytest.mark.asyncio
    async def test_une_lecture_doublee_par_une_ecriture_ne_reinstalle_pas_l_ancien_etat(
        self, db_session, monkeypatch
    ):
        """
        Lue avant une ecriture et finie apres, la liste ne doit pas remettre
        l'etat d'avant en cache pour dix minutes.
        """
        from app.api.routes import categories as module

        lire = module.CategoryService.list_categories

        async def _lecture_doublee(self, *a, **k):
            resultat = await lire(self, *a, **k)
            module.invalider_cache_categories()
            return resultat

        monkeypatch.setattr(module.CategoryService, "list_categories", _lecture_doublee)
        await module._categories_avec_comptes(db_session)

        assert module._cache_categories is None

    @pytest.mark.asyncio
    async def test_category_laws_ne_se_charge_jamais_implicitement(self, db_session, lois):
        """lazy='raise' : un oubli leve, au lieu de charger toutes les lois en silence."""
        from sqlalchemy import select
        from sqlalchemy.exc import InvalidRequestError

        categorie = (
            await db_session.execute(
                select(Category).where(Category.id == lois["civile"].category_id)
            )
        ).scalar_one()

        with pytest.raises(InvalidRequestError):
            categorie.laws


# ==================== LISTE DES LOIS ====================


class TestListeDesLois:
    """
    La liste chargeait tous les articles des lois de la page pour en compter le
    nombre, et renvoie le texte entier par defaut (l'edition d'administration
    en depend) : le front demande content_max_chars.
    """

    @pytest.mark.asyncio
    async def test_sans_parametre_le_texte_reste_entier(self, client, lois):
        """L'ecran d'administration renvoie ce texte tel quel par PUT : le tronquer l'ecraserait."""
        lignes = (await client.get("/api/v1/laws/?limit=10")).json()

        civile = next(ligne for ligne in lignes if ligne["reference"] == "LOI-2020-001")
        assert civile["content"] == lois["texte"]
        assert civile["article_count"] == 3

    @pytest.mark.asyncio
    async def test_content_max_chars_coupe_dans_la_base(self, client, lois):
        lignes = (await client.get("/api/v1/laws/?limit=10&content_max_chars=300")).json()

        civile = next(ligne for ligne in lignes if ligne["reference"] == "LOI-2020-001")
        assert civile["content"] == lois["texte"][:300]
        assert civile["article_count"] == 3

    @pytest.mark.asyncio
    async def test_content_max_chars_zero_ne_rend_pas_le_texte(self, client, compteur, lois):
        reponse = await client.get("/api/v1/laws/?limit=10&content_max_chars=0")

        assert reponse.status_code == 200
        assert all(ligne["content"] is None for ligne in reponse.json())
        assert {ligne["article_count"] for ligne in reponse.json()} == {3, 0}
        # Lois, categories et comptes d'articles : une seule instruction.
        assert _instructions(reponse) == 1
        assert len(reponse.content) < len(lois["texte"])

    @pytest.mark.asyncio
    async def test_content_max_chars_est_borne(self, client):
        assert (await client.get("/api/v1/laws/?content_max_chars=-1")).status_code == 422
        assert (await client.get("/api/v1/laws/?content_max_chars=20001")).status_code == 422

    @pytest.mark.asyncio
    async def test_la_liste_repond_sans_barre_oblique(self, client, lois):
        """GET /laws redirigeait vers /laws/ (vers http:// derriere le proxy) : 3,45 s."""
        reponse = await client.get("/api/v1/laws?limit=5&content_max_chars=0")

        assert reponse.status_code == 200
        assert len(reponse.json()) == 2

    @pytest.mark.asyncio
    async def test_le_detail_ne_charge_ni_le_texte_ni_les_vecteurs_des_articles(
        self, client, compteur, lois
    ):
        reponse = await client.get(f"/api/v1/laws/{lois['civile'].id}")

        assert reponse.status_code == 200
        corps = reponse.json()
        assert corps["content"] == lois["texte"]
        assert [a["number"] for a in corps["articles"]] == ["1", "2", "3"]
        # La loi (categorie jointe), puis le sommaire de ses articles.
        assert _instructions(reponse) == 2


# ==================== RECHERCHE ====================


class TestReponseDeRecherche:
    """
    POST /search renvoyait les chunks et le texte integral de chaque resultat :
    0,06 a 8,6 Mo, pour un front qui en lisait 200 caracteres.
    """

    @pytest.mark.asyncio
    async def test_ni_chunks_ni_texte_integral_mais_les_extraits(self, client, lois):
        from app.schemas.search import APERCU_CONTENU_CARACTERES

        # Le corps exact du front (src/routes/search/+page.svelte).
        reponse = await client.post("/api/v1/search/", json={
            "query": "bail commercial", "mode": "hybrid", "limit": 20, "offset": 0,
            "filters": {"status": "published"},
        })

        assert reponse.status_code == 200
        corps = reponse.json()
        assert "chunks" not in corps
        assert corps["results"], "la loi sur le bail doit etre trouvee"
        for resultat in corps["results"]:
            assert len(resultat["content"] or "") <= APERCU_CONTENU_CARACTERES
        assert any(r["highlights"] for r in corps["results"])
        assert len(reponse.content) < len(lois["texte"])

    @pytest.mark.asyncio
    async def test_le_service_rend_toujours_la_reponse_complete(self, db_session, lois):
        """Le chat et la comparaison lisent les chunks dans le processus."""
        from app.schemas.search import SearchRequest
        from app.services.search_service import SearchService

        service = SearchService(db_session, use_cache=False)
        reponse = await service.search(SearchRequest(query="bail commercial", mode="text"))

        assert reponse.chunks
        assert max(len(c.content) for c in reponse.chunks) > 300


# ==================== CORS ====================


class TestPrealableCors:

    @pytest.mark.asyncio
    async def test_le_prealable_d_une_recherche_vaut_deux_heures(self):
        """Starlette en annonce 600 s : un telephone refaisait le prealable toutes les dix minutes."""
        from httpx import ASGITransport, AsyncClient

        from app.main import CORS_MAX_AGE_S, app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            reponse = await client.options("/api/v1/search/", headers={
                "Origin": ORIGINE_DU_FRONT,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            })

        assert reponse.status_code == 200
        assert reponse.headers["access-control-max-age"] == str(CORS_MAX_AGE_S) == "7200"
