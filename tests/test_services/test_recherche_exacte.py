"""
Le volet semantique EXACT du chat, et le plancher de hnsw.ef_search de /search.

Incident du 29/10/2026 : pour une question d'accueil du chat, l'index HNSW de
production rendait des tableaux de lois de finances (0 des 10 vrais voisins a
ef_search 160), et le modele a repondu dessus. Le chat compare desormais la
question a TOUS les articles (SearchService(semantique_exacte=True)), dans un
budget de temps au-dela duquel il passe par l'index a ef_search 1000 ; /search
garde l'index, a ef_search 400 au moins.

Ces tests disent : que l'enonce exact ne PEUT pas passer par l'index (plan
reel), qu'il rend les vrais plus proches voisins (calcul numpy), que le repli
se declenche sur le delai du serveur et le dit, que ses reglages ne fuient pas
dans la transaction de l'appelant, et que le cache garde au chat ce qu'il
relit (VoletsDeLaRecherche).
"""

import asyncio
import logging
import subprocess
import sys
import time

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law import Article, Law
from app.schemas.search import ChunkResult, SearchFilters, SearchRequest
from app.services import search_service as module
from app.services.embedding_service import EmbeddingService
from app.services.search_service import (
    CHEMIN_EXACT,
    CHEMIN_HNSW,
    CHEMIN_REPLI_HNSW,
    SearchService,
    VoletsDeLaRecherche,
    _delai_depasse,
    ecritures_en_attente,
    est_un_tableau_de_programmes,
    question_budgetaire,
)

DIM = EmbeddingService.EMBEDDING_DIM
INDEX = "idx_articles_embedding_hnsw_halfvec1024"


def _vecteur(graine: int) -> np.ndarray:
    v = np.random.default_rng(graine).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


QUESTION = _vecteur(999)


class _Fournisseur:
    """Le fournisseur d'embeddings, double : la question a toujours le meme vecteur."""

    TASK_QUERY = EmbeddingService.TASK_QUERY
    TASK_DOCUMENT = EmbeddingService.TASK_DOCUMENT

    async def generate_embedding_async(self, texte, task_type=None):
        return QUESTION


def _service(session, exacte=True, cache=False):
    service = SearchService(session, use_cache=cache, semantique_exacte=exacte)
    service.embedding_service = _Fournisseur()
    return service


async def _volet(service, limit=10, filtres=None):
    tache = asyncio.ensure_future(_Fournisseur().generate_embedding_async("q"))
    return await service._chunks_semantiques(tache, filtres, limit, 0)


@pytest.fixture
async def deux_cents_articles(db_session):
    """200 articles vectorises sur deux lois publiees, statistiques a jour."""
    db_session.add_all([
        Law(id=930, reference="LOI-EXACT-A", title="Loi de test exacte A", content="Contenu.",
            type="loi", language="fr", status="published"),
        Law(id=931, reference="LOI-EXACT-B", title="Loi de test exacte B", content="Contenu.",
            type="loi", language="fr", status="published"),
    ])
    await db_session.flush()
    db_session.add_all([
        Article(id=93000 + i, law_id=930 + i % 2, number=str(i + 1),
                content=f"Article {i + 1}.", order=i + 1, embedding=_vecteur(i).tolist())
        for i in range(200)
    ])
    await db_session.commit()
    await db_session.execute(text("ANALYZE articles"))
    await db_session.execute(text("ANALYZE laws"))


def _plus_proches(n):
    """Les n vrais plus proches voisins de QUESTION, calcules hors de la base."""
    similarites = sorted(
        ((float(np.dot(QUESTION, _vecteur(i))), 93000 + i) for i in range(200)), reverse=True
    )
    return [identifiant for _, identifiant in similarites[:n]]


# ==================== SANS BASE ====================


class TestPlancherDeEfSearch:

    @pytest.mark.parametrize("candidats, attendu", [
        (40, "400"), (100, "400"), (160, "400"), (256, "400"), (500, "500"), (5000, "1000"),
    ])
    def test_400_au_moins_1000_au_plus(self, candidats, attendu):
        """160, la valeur de l'incident, ne se pose plus."""
        assert SearchService._ef_search(candidats) == attendu

    def test_l_enonce_de_l_index_porte_le_plancher(self):
        service = SearchService.__new__(SearchService)
        enonce = service._build_semantic_statement([0.001] * DIM, None, 8, 0)

        parametres = enonce.compile(dialect=postgresql.dialect()).params

        assert parametres["hnsw_ef_search"] == "400"

    def test_le_repli_porte_ef_search_1000(self):
        service = SearchService.__new__(SearchService)
        enonce = service._build_semantic_statement([0.001] * DIM, None, 8, 0, ef_search=1000)

        assert enonce.compile(dialect=postgresql.dialect()).params["hnsw_ef_search"] == "1000"


class TestEnonceExact:

    @staticmethod
    def _sql(filtres=None):
        service = SearchService.__new__(SearchService)
        enonce = service._build_semantic_statement(
            [0.001] * DIM, filtres, 32, 0, exacte=True
        )
        return str(enonce.compile(dialect=postgresql.dialect())).replace("\n", " ")

    def test_son_tri_ne_peut_pas_etre_servi_par_l_index(self):
        """
        L'index HNSW ne sert que `colonne <=> constante` nue : le tri porte
        sur `(... <=> ...) + 0`, et aucun ORDER BY ne porte sur l'operateur nu.
        """
        sql = self._sql()

        assert "ORDER BY (articles.embedding <=> " in sql
        assert ") + 0, articles.id" in sql
        assert "ORDER BY articles.embedding <=>" not in sql
        # Aucun reglage de l'index : il n'y a pas de parcours a regler.
        assert "hnsw.ef_search" not in sql

    def test_le_texte_n_est_lu_que_pour_les_plus_proches(self):
        """Etage 1 etroit (identifiant, distance), etage 2 : le texte des seuls retenus."""
        sql = self._sql()
        interieur = sql.split(") AS proches")[0].split("FROM (", 1)[1]

        assert "articles.content" not in interieur
        assert "articles.content" in sql.split(") AS proches")[0].split("FROM (", 1)[0]

    def test_les_filtres_entrent_dans_l_etage_1(self):
        sql = self._sql(SearchFilters(status="published", language="fr"))
        interieur = sql.split(") AS proches")[0]

        assert "laws.status = " in interieur
        assert "laws.language = " in interieur


def test_delai_depasse_reconnait_l_annulation_du_serveur():
    class Origine(Exception):
        sqlstate = "57014"

    class Enveloppe(Exception):
        pass

    erreur = Enveloppe("canceling statement due to statement timeout")
    erreur.orig = Origine()

    assert _delai_depasse(erreur)
    assert not _delai_depasse(RuntimeError("autre chose"))


# ==================== SUR LA BASE ====================


class TestRechercheExacte:

    @pytest.mark.asyncio
    async def test_elle_rend_les_vrais_plus_proches_voisins(self, db_session, deux_cents_articles):
        service = _service(db_session)

        chunks = await _volet(service, limit=10)

        assert service.chemin_semantique == CHEMIN_EXACT
        # Les similarites des dix rendus sont les dix meilleures, dans l'ordre
        # (a la precision du demi-flottant pres : deux voisins presque a
        # egalite peuvent s'echanger, pas un voisin manquer).
        attendues = sorted((float(np.dot(QUESTION, _vecteur(i))) for i in range(200)), reverse=True)
        assert [c.relevance_score for c in chunks] == pytest.approx(attendues[:10], abs=2e-3)
        assert set(c.article_id for c in chunks[:8]) <= set(_plus_proches(10))

    @pytest.mark.asyncio
    async def test_le_plan_ne_passe_jamais_par_l_index(self, db_session, deux_cents_articles):
        """
        Le balayage sequentiel penalise, l'index est TENTANT : il reste hors du
        plan, parce que le tri `+ 0` ne lui est pas rattachable et que les
        parcours d'index sont coupes. Le tri, lui, est bien dans le plan.
        """
        service = SearchService.__new__(SearchService)
        enonce = service._build_semantic_statement(
            QUESTION.tolist(), SearchFilters(status="published"), 32, 0, exacte=True
        )
        sql = str(enonce.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        ))
        await db_session.execute(text("SET LOCAL enable_seqscan = off"))
        await db_session.execute(text(module._REGLAGES_EXACTS_SQL), {"budget_ms": "1500"})
        conn = await db_session.connection()

        plan = "\n".join(r[0] for r in await conn.exec_driver_sql("EXPLAIN " + sql))

        assert INDEX not in plan, plan
        cles_de_tri = [ligne for ligne in plan.splitlines() if "Sort Key:" in ligne]
        assert any("+ '0'::double precision" in ligne for ligne in cles_de_tri), plan

    @pytest.mark.asyncio
    async def test_l_index_sert_toujours_search(self, db_session, deux_cents_articles, caplog):
        service = _service(db_session, exacte=False)

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            chunks = await _volet(service)

        assert len(chunks) == 10
        assert service.chemin_semantique == CHEMIN_HNSW
        assert any("index HNSW, 100 candidats, ef_search 400" in r.getMessage()
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_le_journal_dit_le_chemin_exact(self, db_session, deux_cents_articles, caplog):
        with caplog.at_level(logging.INFO, logger=module.logger.name):
            await _volet(_service(db_session))

        lignes = [r.getMessage() for r in caplog.records]
        assert any(m.startswith("🧠 Semantic search (recherche exacte, sans index) : 10 chunks")
                   for m in lignes), lignes


def _enonce_lent(self, distance, filters, limit, offset):
    """Un enonce exact qui depasse tout budget : une seconde de sommeil d'abord."""
    lent = select(Article.id).where(text("pg_sleep(1) IS NOT NULL")).limit(1).subquery()
    return (
        select(
            Article.id.label("article_id"), Article.number,
            Article.title.label("article_title"), Article.section, Article.page_number,
            Article.content, Article.law_id, Law.reference, Law.title.label("law_title"),
            Law.type, Law.language, Law.status, Law.category_id, Law.publication_date,
            Law.title.label("category_name"), distance.label("distance"),
        )
        .join(Law, Law.id == Article.law_id)
        .where(Article.id.in_(select(lent.c.id)))
    )


class TestRepliSurLIndex:

    @pytest.mark.asyncio
    async def test_au_dela_du_budget_l_index_repond_et_le_journal_alerte(
        self, db_session, deux_cents_articles, monkeypatch, caplog
    ):
        monkeypatch.setattr(SearchService, "_enonce_exact", _enonce_lent)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 100)
        service = _service(db_session)

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            chunks = await _volet(service, limit=10)

        assert service.chemin_semantique == CHEMIN_REPLI_HNSW
        assert len(chunks) == 10
        alertes = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert alertes == [
            "⏱️ Recherche semantique exacte au-dela de 100 ms, annulee : repli sur "
            "l'index HNSW (ef_search 1000)"
        ]
        assert any("REPLI sur l'index HNSW, ef_search 1000" in r.getMessage()
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sur_la_session_de_l_appelant_rien_ne_fuit(
        self, db_session, deux_cents_articles, pg_engine, monkeypatch
    ):
        """
        Sans moteur pour une autre connexion, le volet passe par la session de
        l'appelant, sous SAVEPOINT : apres le delai depasse ET apres un succes,
        enable_indexscan et statement_timeout y sont ceux d'avant, et la
        transaction sert encore.
        """
        async with pg_engine.connect() as connexion:
            async with AsyncSession(bind=connexion) as session:
                avant = (await session.execute(text(
                    "SELECT current_setting('enable_indexscan'), current_setting('statement_timeout')"
                ))).one()
                service = _service(session)
                assert service._moteur_isole() is None

                await _volet(service)
                assert service.chemin_semantique == CHEMIN_EXACT
                monkeypatch.setattr(SearchService, "_enonce_exact", _enonce_lent)
                monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 100)
                chunks = await _volet(service)

                apres = (await session.execute(text(
                    "SELECT current_setting('enable_indexscan'), current_setting('statement_timeout')"
                ))).one()
                assert service.chemin_semantique == CHEMIN_REPLI_HNSW
                assert len(chunks) == 10
                assert tuple(apres) == tuple(avant)
                assert (await session.execute(text("SELECT 1"))).scalar() == 1

    @pytest.mark.asyncio
    async def test_une_autre_erreur_n_est_pas_un_repli(self, db_session, deux_cents_articles,
                                                      monkeypatch):
        """Une requete fausse reste une panne du volet, comptee comme telle."""
        def _faux(self, distance, filters, limit, offset):
            return select(text("colonne_inconnue")).select_from(Article)

        monkeypatch.setattr(SearchService, "_enonce_exact", _faux)

        with pytest.raises(module.VectorSearchError):
            await _volet(_service(db_session))


class TestBudgetAFroid:
    """
    Production, 04/11/2026, premier controle apres le deploiement de la
    recherche exacte : a froid, les deux premieres recherches exactes ont
    depasse 3 s, le repli sur l'index a suivi, et la barriere a ecarte tout le
    contexte d'une question suggeree. Toute recherche exacte lancee avant une
    reussite, apres BASE_FROIDE_APRES_S sans aucune (ou depuis le demarrage du
    processus), a donc le budget long ; une reussite rend la base chaude, et le
    budget court revient, celui du repli en surcharge.
    """

    def test_sans_recherche_reussie_le_budget_est_celui_du_froid(self):
        module.oublier_les_recherches_exactes()

        assert _service(object())._budget_de_la_recherche_exacte_ms() == 30000

    def test_apres_une_reussite_recente_le_budget_est_court(self):
        module.noter_une_recherche_exacte_reussie()

        assert _service(object())._budget_de_la_recherche_exacte_ms() == 3000

    def test_au_dela_de_quatre_minutes_sans_recherche_la_base_est_presumee_froide(self):
        service = _service(object())
        module.noter_une_recherche_exacte_reussie(time.monotonic() - 239)
        assert service._budget_de_la_recherche_exacte_ms() == 3000

        module.noter_une_recherche_exacte_reussie(time.monotonic() - 241)
        assert service._budget_de_la_recherche_exacte_ms() == 30000

    def test_un_budget_a_froid_plus_court_ne_raccourcit_rien(self, monkeypatch):
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_A_FROID_MS", 10)
        module.oublier_les_recherches_exactes()

        assert _service(object())._budget_de_la_recherche_exacte_ms() == 3000

    def test_un_processus_qui_demarre_presume_la_base_froide(self):
        """
        Le cas du 04/11 : la premiere recherche apres un deploiement. Un
        interpreteur neuf, sans la fixture qui marque la base chaude.
        """
        sortie = subprocess.run(
            [sys.executable, "-c",
             "from app.services import search_service as m; "
             "print(m._derniere_recherche_exacte_reussie)"],
            capture_output=True, text=True, timeout=120, check=True,
        )

        assert sortie.stdout.strip().splitlines()[-1] == "None"

    @pytest.mark.asyncio
    async def test_a_froid_la_requete_lente_aboutit_puis_la_base_est_chaude(
        self, db_session, deux_cents_articles, monkeypatch, caplog
    ):
        """
        Le meme enonce lent (une seconde) : a froid, il aboutit par le chemin
        exact, et le journal donne sa duree ; la base est alors chaude, et la
        recherche suivante a le budget court, donc le repli.
        """
        monkeypatch.setattr(SearchService, "_enonce_exact", _enonce_lent)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 100)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_A_FROID_MS", 10000)
        module.oublier_les_recherches_exactes()
        service = _service(db_session)

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            await _volet(service)
        assert (service.chemin_semantique, service.budget_exact_ms) == (CHEMIN_EXACT, 10000)
        assert any(r.getMessage().startswith("🧊 Recherche semantique exacte, base presumee froide : ")
                   and r.getMessage().endswith(" ms (budget 10000 ms)") for r in caplog.records)

        await _volet(service)
        assert (service.chemin_semantique, service.budget_exact_ms) == (CHEMIN_REPLI_HNSW, 100)

    @pytest.mark.asyncio
    async def test_un_repli_ne_rend_pas_la_base_chaude_et_l_alerte_dit_le_budget_a_froid(
        self, db_session, deux_cents_articles, monkeypatch, caplog
    ):
        """
        Seule une reussite dit la base chaude : apres un echec a froid, la
        suivante a encore le budget long. L'alerte du repli, seule trace en
        production, dit le budget de la tentative annulee, ici celui du froid.
        """
        monkeypatch.setattr(SearchService, "_enonce_exact", _enonce_lent)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 100)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_A_FROID_MS", 200)
        module.oublier_les_recherches_exactes()
        service = _service(db_session)

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            await _volet(service)

        assert (service.chemin_semantique, service.budget_exact_ms) == (CHEMIN_REPLI_HNSW, 200)
        assert module._derniere_recherche_exacte_reussie is None
        assert service._budget_de_la_recherche_exacte_ms() == 200
        alertes = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert alertes == [
            "⏱️ Recherche semantique exacte au-dela de 200 ms, annulee : repli sur "
            "l'index HNSW (ef_search 1000)"
        ]

    @pytest.mark.asyncio
    async def test_deux_recherches_simultanees_a_froid_ont_toutes_deux_le_budget_long(
        self, db_session, deux_cents_articles, monkeypatch
    ):
        """Une rafale a froid : chaque recherche lancee avant la premiere reussite a le budget long."""
        monkeypatch.setattr(SearchService, "_enonce_exact", _enonce_lent)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 100)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_A_FROID_MS", 10000)
        module.oublier_les_recherches_exactes()
        services = [_service(db_session), _service(db_session)]

        await asyncio.gather(*(_volet(s) for s in services))

        assert [(s.chemin_semantique, s.budget_exact_ms) for s in services] == [(CHEMIN_EXACT, 10000)] * 2


# ==================== CE QUE LE CHAT RELIT ====================


class TestVoletsDeLaRecherche:

    @staticmethod
    def _hybride(texte, semantique, exacte=True):
        service = SearchService(object(), use_cache=False, semantique_exacte=exacte)
        service.text_chunks = _rendre(texte)
        service._lancer_vectorisation = lambda q: asyncio.ensure_future(asyncio.sleep(0))

        async def _semantique(tache, filters, limit, offset):
            await module._solder(tache)
            service.chemin_semantique = CHEMIN_EXACT if exacte else CHEMIN_HNSW
            return [c.model_copy() for c in semantique]

        service._chunks_semantiques = _semantique
        return service

    @pytest.mark.asyncio
    async def test_similarites_d_avant_la_fusion_et_premiers_articles_du_plein_texte(self):
        texte = [_extrait(None, 1, None, "title_fts", 1.0)] + [
            _extrait(10 + i, 1, str(i), "fts", 1.0 - i / 10) for i in range(6)
        ]
        semantique = [_extrait(50, 2, "7", "semantic", 0.61), _extrait(10, 1, "0", "semantic", 0.44)]
        service = self._hybride(texte, semantique)

        await service.hybrid_chunks("mots cles", None, 8, 0)

        volets = service.volets
        assert volets.chemin == CHEMIN_EXACT
        assert volets.similarites == {50: 0.61, 10: 0.44}
        assert volets.similarite_max == 0.61
        # Des articles seulement, dans l'ordre et avec les scores du plein texte.
        assert [(c.article_id, c.relevance_score) for c in volets.plein_texte] == [
            (10, 1.0), (11, 0.9), (12, 0.8), (13, 0.7),
        ]

    @pytest.mark.asyncio
    async def test_un_volet_en_panne_le_dit(self):
        service = self._hybride([_extrait(10, 1, "1", "fts", 1.0)], [])

        async def _panne(tache, filters, limit, offset):
            await module._solder(tache)
            raise module.VectorSearchError("injoignable")

        service._chunks_semantiques = _panne
        await service.hybrid_chunks("mots cles", None, 8, 0)

        assert service.volets.chemin == module.CHEMIN_ECHEC
        assert service.volets.similarite_max is None


def _rendre(valeur):
    async def _f(*args, **kwargs):
        return [c.model_copy() for c in valeur]
    return _f


# ==================== TABLEAUX DE PROGRAMMES DES LOIS DE FINANCES ====================

LOI_DE_FINANCES = (
    "Loi N°2022/020 du 27 décembre 2022 portant loi de finances de la République du Cameroun "
    "pour l'exercice 2023"
)
# Les formes relevees en base le 29/10/2026 (voir search_service).
TABLEAU_AE_CP = (
    "Les montants des autorisations d'engagement et des crédits de paiement du budget général "
    "ouverts sur les programmes concourant à la réalisation des objectifs assortis d'indicateurs "
    "sont fixés comme suit :\n| N° | CODE | Programme LIBELLE | OBJECTF | INDICATEUR | AE | CP |\n"
    "|---|---|---|---|---|---|---|\n| 130 | 542 | AMELIORATION DE LA PROTECTION DU TRAVAIL | "
    "Promouvoir le travail décent | Proportion | 2 020 161 | 1 870 161 |"
)
TABLEAU_ABIME_PAR_L_OCR = (
    "Les ds its es ns s s ddt budget général ouverts sur les programmes concourant à la "
    "réalisation des objectifs assortis d'indicateurs sont fixés comme suit :\n| 129 | 528 | x |"
)
TABLEAU_VOTE_AE = (
    "| N° | CODE | PROGRAMME LIBELLE | OBJECTF | INDICATEUR | VOTE AE | MODIFIE AE | VOTE CP |\n"
    "|---|---|---|---|---|---|---|---|"
)
PERFORMANCE = (
    "DE LA PERFORMANCE DES PROGRAMMES: BASE INDICATEURS Les écarts ci-après ont été relevés dans "
    "la mise en œuvre des programmes sur la base des indicateurs"
)
AMENDES = (
    "| N° | LIBELLE | AMENDES |\n|---|---|---|\n| 28 | L'emploi d'un membre du personnel "
    "technique non qualifié | 15.000.000 frs |"
)
TAUX = "Le taux de la taxe sur la valeur ajoutée est fixé à 17,5 %.\n| Produit | Taux |\n|---|---|"


def _morceau(contenu, titre=LOI_DE_FINANCES, article_id=1, score=0.5, source="semantic"):
    return ChunkResult(
        article_id=article_id, law_id=2232, number="76.1", content=contenu, reference="PRC-9449",
        law_title=titre, relevance_score=score, source=source,
    )


class TestTableauxDeProgrammes:
    """
    Les tableaux de programmes des lois de finances (credits AE/CP par
    programme, performance des programmes) ne sont pas du droit : la
    recherche du chat les retire, sauf pour une question sur le budget.
    Les taux, les amendes et les articles des lois de finances restent.
    """

    @pytest.mark.parametrize("contenu", [TABLEAU_AE_CP, TABLEAU_ABIME_PAR_L_OCR, TABLEAU_VOTE_AE])
    def test_un_tableau_de_programmes(self, contenu):
        assert est_un_tableau_de_programmes(_morceau(contenu))

    @pytest.mark.parametrize("contenu", [
        AMENDES, TAUX, "Le Président de la République est habilité.",
        # Les tableaux de performance des lois de reglement, sans AE ni CP :
        # le jeu d'evaluation y cherche des reponses (« bilan services de
        # renseignement 2022 »).
        PERFORMANCE,
    ])
    def test_une_regle_ou_un_autre_tableau_reste(self, contenu):
        assert not est_un_tableau_de_programmes(_morceau(contenu))

    def test_hors_d_une_loi_de_finances_rien_n_est_retire(self):
        """Une circulaire de preparation du budget parle d'AE et de CP : elle reste."""
        circulaire = "Circulaire N°001/CAB/PRC du 20 juin 2017 relative à la préparation du budget"
        assert not est_un_tableau_de_programmes(_morceau(TABLEAU_AE_CP, titre=circulaire))
        # « loi des finances » et les ordonnances qui les modifient en sont.
        assert est_un_tableau_de_programmes(_morceau(TABLEAU_AE_CP, titre=(
            "Loi N°2025/012 du 17 décembre 2025 portant loi des Finances de la République")))
        assert est_un_tableau_de_programmes(_morceau(TABLEAU_AE_CP, titre=(
            "Ordonnance N°2024/001 modifiant certaines dispositions de la loi n°2023/019 portant "
            "loi de finances")))

    @pytest.mark.parametrize("question, budgetaire", [
        ("Quelles indemnités en cas de licenciement abusif ?", False),
        ("Le SMIG au Cameroun c'est combien ?", False),
        ("What is the minimum wage in Cameroon?", False),
        ("Quel est le taux de TVA ?", False),
        ("Peut-on obtenir un crédit bancaire ?", False),
        ("Quel est le budget du ministère de la santé ?", True),
        ("Que dit la loi de finances 2024 sur les crédits de paiement ?", True),
        ("Comment sont contrôlées les finances publiques ?", True),
        ("What does the 2024 finance law allocate to health?", True),
        ("Combien l'État camerounais dépense-t-il en 2025 pour aider les PME ?", True),
        ("Combien le gouvernement a-t-il budgétisé en 2019 pour entrer au capital d'entreprises ?",
         True),
        ("Quel est le budget de l'État pour 2025 ?", True),
        ("Quelle dotation budgétaire pour la protection du travail ?", True),
        ("Quelles autorisations d'engagement pour le ministère du travail ?", True),
        ("What is the state budget for 2025?", True),
        ("How much did the government budget in 2019 to take stakes in companies?", True),
        ("What payment appropriations does the finance act grant?", True),
        ("Quel est le montant des allocations familiales ?", False),
        # Une enveloppe, ici, est un pot-de-vin ; une rallonge, un delai.
        ("Le policier m'a demandé une enveloppe", False),
        ("Puis-je avoir une rallonge pour payer mon loyer ?", False),
    ])
    def test_une_question_sur_le_budget(self, question, budgetaire):
        assert question_budgetaire(question) is budgetaire

    @pytest.mark.parametrize("question", [
        # Les faux positifs de la revue du 29/10/2026 (tour 3) : des questions
        # de droit, qui gardaient les tableaux AE/CP.
        "Quel est le montant de l'amende prévue pour le vol ?",
        "Le montant prévu de la pension de veuve",
        "Quel est le montant de la caution prévu par le bail ?",
        "Mon budget est serré, puis-je payer mon loyer en retard ?",
        "Les crédits ouverts par ma banque",
        "Comment calculer la dotation aux amortissements selon l'OHADA ?",
        "How much compensation is allocated for unfair dismissal?",
        "Quel est le montant prévu du SMIG au Cameroun ?",
        "Quel est le montant prévu pour l'indemnité de licenciement ?",
        "Qu'est-ce que l'État a prévu pour les travailleurs licenciés ?",
        "What amount does the state allocate to the unemployed?",
        # Un budget a soi, avec une annee ; les depens d'un proces ;
        # l'appropriation d'une chose.
        "Mon budget 2025 est serré, que faire ?",
        "L'État est-il condamné aux dépens ?",
        "Is the appropriation of a lost object theft?",
        # Un budget PRIVE avec une annee (revue du tour 4, sonde_budget_annee) :
        # la regle « budget + une annee » les prenait pour le budget de l'Etat,
        # et quatre lisaient un tableau AE/CP. « 2000 euros » passait pour une
        # annee.
        "Quel budget prévoir pour créer une SARL en 2025 ?",
        "J'ai un budget de 2000 euros pour acheter un terrain, quelles démarches ?",
        "Quel budget pour un mariage en 2024 ?",
        "Comment gérer le budget familial en 2024 ?",
        "Le budget de l'association pour 2024 doit-il être voté par l'assemblée générale ?",
        "Quel budget pour un avocat en 2025 ?",
        "What budget do I need to start a business in 2025?",
        "Quel budget pour construire une maison en 2025 ?",
        "budget prévisionnel 2025 d'une SARL",
        # Le vocabulaire du budget manque : la question perd ses tableaux.
        "Quelle somme a été prévue en 2019 pour venir en aide aux sinistrés ?",
        "ratification rallonge 2019",
        "What extra funding did the State grant in early 2020 to close out 2019?",
        # « budget » et une annee ne suffisent plus (regle retiree au tour 5) :
        # quatre questions du banc par langue y perdent leurs tableaux.
        "budget 2025 Cameroun PME",
        "budget sinistrés catastrophes naturelles 2019",
        "objectifs budget 2014 lutte corruption",
        "2025 budget Cameroon SMEs",
    ])
    def test_un_montant_prevu_n_est_pas_le_budget(self, question):
        assert question_budgetaire(question) is False

    @staticmethod
    def _hybride(texte, semantique, exclus):
        service = TestVoletsDeLaRecherche._hybride(texte, semantique)
        service.tableaux_de_programmes_exclus = exclus
        return service

    @pytest.mark.asyncio
    async def test_retires_des_deux_volets_avant_la_fusion(self, caplog):
        tableaux = [_morceau(TABLEAU_AE_CP, article_id=70 + i, score=0.5) for i in range(3)]
        code = _extrait(10, 1, "35.1", "semantic", 0.41)
        service = self._hybride(
            [_morceau(TABLEAU_AE_CP, article_id=80, source="fts"), _extrait(11, 1, "37", "fts", 1.0)],
            tableaux + [code], exclus=True,
        )

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            fusion = await service.hybrid_chunks("salaire minimum", None, 8, 0)

        assert {c.article_id for c in fusion} == {10, 11}
        assert service.volets.similarites == {10: 0.41}
        assert service.volets.tableaux_exclus == 4
        assert [c.article_id for c in service.volets.plein_texte] == [11]
        assert any("4 tableau(x) de programmes" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_gardes_quand_la_recherche_ne_les_exclut_pas(self):
        """/search, et le chat sur une question budgetaire : tout le corpus."""
        tableaux = [_morceau(TABLEAU_AE_CP, article_id=70, score=0.5)]
        service = self._hybride([_extrait(11, 1, "37", "fts", 1.0)], tableaux, exclus=False)

        fusion = await service.hybrid_chunks("budget santé", None, 8, 0)

        assert 70 in {c.article_id for c in fusion}
        assert service.volets.tableaux_exclus == 0

    @pytest.mark.asyncio
    async def test_un_volet_vide_de_ses_seuls_tableaux_n_est_pas_une_panne(self, caplog):
        service = self._hybride(
            [_extrait(11, 1, "37", "fts", 1.0)], [_morceau(TABLEAU_AE_CP, article_id=70)], exclus=True,
        )

        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            await service.hybrid_chunks("salaire minimum", None, 8, 0)

        assert not [r for r in caplog.records if "Volet semantique vide" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_un_volet_vide_avant_le_retrait_reste_une_panne(self, caplog):
        """
        Revue du 29/10/2026 (tour 3) : un tableau retire du PLEIN TEXTE faisait
        taire l'alerte, meme quand l'index n'avait rien rendu du tout.
        """
        service = self._hybride(
            [_morceau(TABLEAU_AE_CP, article_id=80, source="fts"), _extrait(11, 1, "37", "fts", 1.0)],
            [], exclus=True,
        )

        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            await service.hybrid_chunks("salaire minimum", None, 8, 0)

        assert service.volets.tableaux_exclus == 1
        assert [r for r in caplog.records if "Volet semantique vide" in r.getMessage()]

    def test_la_cle_du_cache_depend_du_drapeau(self):
        """Une reponse sans tableaux ne sert jamais une question budgetaire, ni l'inverse."""
        service = SearchService(object(), use_cache=True, semantique_exacte=True)
        avec = service._mode_en_cache("hybrid")
        service.tableaux_de_programmes_exclus = True

        assert (avec, service._mode_en_cache("hybrid")) == ("hybrid+exacte", "hybrid+exacte+sans_tableaux")
        assert SearchService(object(), use_cache=True)._mode_en_cache("hybrid") == "hybrid"


def _extrait(article_id, law_id, numero, source, score):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content="Contenu.",
        reference=f"R-{law_id}", law_title=f"Loi {law_id}", relevance_score=score,
        source=source,
    )


class TestCacheDuChat:

    @pytest.mark.asyncio
    async def test_la_reponse_du_cache_rend_ses_volets(self, db_session, deux_cents_articles):
        demande = SearchRequest(query="article", semantic_query="question entiere",
                                mode="hybrid", limit=8)
        premier = _service(db_session, cache=True)
        await premier.search(demande)
        await ecritures_en_attente(10)

        second = _service(db_session, cache=True)
        second.text_chunks = _interdit
        second._chunks_semantiques = _interdit
        await second.search(demande)

        assert second.volets is not None
        assert second.volets.chemin == CHEMIN_EXACT
        assert second.volets.similarites == premier.volets.similarites

    @pytest.mark.asyncio
    async def test_une_entree_sans_volets_est_recalculee(self, db_session, deux_cents_articles):
        """Une reponse ecrite par l'index (ou par l'ancien code) ne sert pas le chat."""
        demande = SearchRequest(query="article", semantic_query="question entiere",
                                mode="hybrid", limit=8)
        cle = module._make_cache_key(demande.query, demande.filters, demande.limit,
                                     demande.offset, "hybrid+exacte", demande.semantic_query)
        index = _service(db_session, exacte=False, cache=True)
        reponse = await index.search(demande)
        await ecritures_en_attente(10)
        await module.store_in_pg_cache(db_session, cle, module._charge_utile_du_cache(reponse))

        chat = _service(db_session, cache=True)
        await chat.search(demande)

        assert chat.volets is not None and chat.volets.chemin == CHEMIN_EXACT

    @pytest.mark.asyncio
    async def test_l_index_et_le_chat_n_ont_pas_la_meme_cle(self, db_session, deux_cents_articles):
        demande = SearchRequest(query="article", semantic_query="question entiere",
                                mode="hybrid", limit=8)
        await _service(db_session, exacte=False, cache=True).search(demande)
        await ecritures_en_attente(10)

        chat = _service(db_session, cache=True)
        await chat.search(demande)
        await ecritures_en_attente(10)

        assert chat.chemin_semantique == CHEMIN_EXACT, "le chat n'a pas relu l'entree de l'index"
        cles = (await db_session.execute(text("SELECT count(*) FROM query_cache"))).scalar()
        assert cles == 2

    @pytest.mark.asyncio
    async def test_un_repli_sur_l_index_n_est_pas_mis_en_cache(
        self, db_session, deux_cents_articles, monkeypatch
    ):
        monkeypatch.setattr(SearchService, "_enonce_exact", _enonce_lent)
        monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 100)
        demande = SearchRequest(query="article", semantic_query="question entiere",
                                mode="hybrid", limit=8)
        chat = _service(db_session, cache=True)

        await chat.search(demande)
        await ecritures_en_attente(10)

        assert chat.volets.chemin == CHEMIN_REPLI_HNSW
        assert (await db_session.execute(text("SELECT count(*) FROM query_cache"))).scalar() == 0

    def test_les_volets_survivent_au_json(self):
        volets = VoletsDeLaRecherche(
            chemin=CHEMIN_EXACT, similarites={12: 0.5, 7: 0.25},
            plein_texte=[_extrait(3, 1, "1", "fts", 0.9)],
        )

        relu = VoletsDeLaRecherche.model_validate(volets.model_dump(mode="json"))

        assert relu == volets


async def _interdit(*args, **kwargs):
    raise AssertionError("la reponse devait venir du cache")
