"""
Le chemin de la recherche apres l'acceleration d'octobre 2026, point par point.

Chacun de ces changements gagnait des secondes sans changer une reponse : un
retour en arriere ne casserait donc aucun autre test. Ceux-ci verifient ce qui
les rend surs — le contexte du chat inchange par la borne des lignes niveau
loi, des reglages HNSW qui atteignent vraiment le parcours, une passe
trigramme qui ne laisse pas son delai a l'appelant — et ce qui les rend utiles
— le vecteur demande avant la fin du plein texte, le re-ranking hors de la
boucle d'evenements.
"""

import asyncio
import threading

import numpy as np
import pytest
from sqlalchemy import text

from app.models.law import Article, Law
from app.schemas.search import ChunkResult, SearchFilters, SearchRequest
from app.services.embedding_service import EmbeddingService
from app.services.postgres_search_service import (
    LAW_CHUNK_MAX_CHARS,
    search_laws_pg,
    search_titles_trgm_pg,
)
from app.services.prompts import CONTEXT_MAX_CHARS_PER_CHUNK, build_context_string
from app.services.search_service import SearchService

DIM = EmbeddingService.EMBEDDING_DIM


def _vecteur(graine: int) -> np.ndarray:
    v = np.random.default_rng(graine).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ==================== 1. LIGNES NIVEAU LOI BORNEES ====================


@pytest.fixture
async def loi_longue(db_session):
    """Une loi de 30 000 caracteres, en paragraphes, dont le titre porte « bail »."""
    paragraphes = [
        f"Article {i}. Le preneur d'un bail commercial doit user des lieux en bon pere "
        f"de famille, conformement a la destination prevue au contrat numero {i}."
        for i in range(1, 400)
    ]
    texte = "\n\n".join(paragraphes)[:30_000]
    db_session.add(Law(
        reference="AUDCG-BAIL", title="Acte uniforme : du bail commercial", type="loi",
        content=texte, language="fr", status="published",
    ))
    await db_session.commit()
    return texte


class TestLignesNiveauLoi:
    """
    Les lignes niveau loi portaient le texte integral : jusqu'a 1,76 M de
    caracteres par ligne, 1 a 8,6 Mo par recherche, re-classes en Python.
    """

    @pytest.mark.asyncio
    async def test_la_branche_titre_rend_un_extrait_borne(self, db_session, loi_longue):
        (ligne,) = await search_laws_pg(db_session, "bail commercial", None, 10, 0)

        assert ligne.article_id is None
        assert ligne.content == loi_longue[:LAW_CHUNK_MAX_CHARS]

    @pytest.mark.asyncio
    async def test_la_passe_floue_sur_les_titres_aussi(self, db_session, loi_longue):
        lignes = await search_titles_trgm_pg(db_session, "bial comercial", None, 10, 0.3)

        assert lignes, "la faute de frappe doit retrouver le titre"
        assert all(len(c.content) <= LAW_CHUNK_MAX_CHARS for c in lignes)

    @pytest.mark.parametrize("longueur", [5_999, 6_000, 6_001, 6_002, 9_000, 30_000])
    def test_le_contexte_du_chat_ne_change_pas_d_un_octet(self, longueur):
        """
        LAW_CHUNK_MAX_CHARS = CONTEXT_MAX_CHARS_PER_CHUNK + 1 : avec ce caractere
        de plus, la coupe du chat tombe au meme endroit, avec la meme marque,
        que sur le texte entier.
        """
        phrases = " ".join(
            f"Phrase {i} du texte, assez longue pour compter." for i in range(2_000)
        )
        texte = ("Premier alinea.\n\n" + phrases)[:longueur]

        def contexte(contenu):
            return build_context_string([ChunkResult(
                article_id=None, law_id=1, content=contenu, excerpt="", reference="R",
                law_title="Loi", type="loi", language="fr", status="published",
                relevance_score=1.0, source="title_fts",
            )])

        assert LAW_CHUNK_MAX_CHARS == CONTEXT_MAX_CHARS_PER_CHUNK + 1
        assert contexte(texte[:LAW_CHUNK_MAX_CHARS]) == contexte(texte)


# ==================== 3. REGLAGES HNSW DANS L'ENONCE ====================


@pytest.fixture
async def deux_cents_articles(db_session):
    """200 articles vectorises sur deux langues, statistiques a jour."""
    db_session.add_all([
        Law(id=910, reference="LOI-EF-FR", title="Loi de test ef", content="Contenu.",
            type="loi", language="fr", status="published"),
        Law(id=911, reference="LOI-EF-EN", title="Ef test law", content="Content.",
            type="loi", language="en", status="published"),
    ])
    await db_session.flush()
    db_session.add_all([
        Article(law_id=910 + i % 2, number=str(i + 1), content=f"Article {i + 1}.",
                order=i + 1, embedding=_vecteur(i).tolist())
        for i in range(200)
    ])
    await db_session.commit()
    await db_session.execute(text("ANALYZE articles"))


class TestReglagesHnswDansLEnonce:
    """
    hnsw.ef_search (et hnsw.iterative_scan sous filtre) sont poses PAR
    l'enonce vectoriel lui-meme, et non plus par deux SAVEPOINT et leurs SET
    LOCAL : seize allers-retours de moins. Encore faut-il que le reglage
    atteigne le parcours de l'index. ef_search vaut 40 par defaut, et un
    parcours HNSW non iteratif ne rend jamais plus de lignes : 50 lignes
    demandees n'arrivent que si le reglage a pris.
    """

    @staticmethod
    async def _forcer_l_index(db_session):
        # Sur 200 lignes, le planificateur prefererait un balayage puis un
        # tri : ces penalites ne laissent que l'index (voir test_vector_schema).
        await db_session.execute(text("SET LOCAL enable_seqscan = off"))
        await db_session.execute(text("SET LOCAL enable_sort = off"))

    @pytest.mark.asyncio
    async def test_le_temoin_l_index_seul_s_arrete_a_40(self, db_session, deux_cents_articles):
        """Sans reglage, ce montage plafonne bien a 40 : le test suivant prouve quelque chose."""
        await self._forcer_l_index(db_session)
        lignes = (await db_session.execute(
            text(
                "SELECT id FROM articles WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(CAST(:v AS text) AS halfvec) LIMIT 400"
            ),
            {"v": str(_vecteur(999).tolist())},
        )).all()

        assert len(lignes) == 40

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filtres", [None, SearchFilters(language="fr")], ids=["sans-filtre", "filtre"])
    async def test_l_enonce_porte_son_ef_search_jusqu_au_parcours(
        self, db_session, deux_cents_articles, filtres
    ):
        await self._forcer_l_index(db_session)
        service = SearchService(db_session, use_cache=False)

        lignes = await service._lire_vecteurs(db_session, _vecteur(999), filtres, 50, 0)

        assert len(lignes) == 50
        distances = [ligne.distance for ligne in lignes]
        assert distances == sorted(distances)


# ==================== 4. PASSE TRIGRAMME SUR LES ARTICLES ====================


@pytest.fixture
async def articles_courts(db_session):
    db_session.add(Law(id=920, reference="LOI-TRGM", title="Code du travail", type="loi",
                       content="Contenu.", language="fr", status="published"))
    await db_session.flush()
    db_session.add_all([
        Article(law_id=920, number="1", content="Le licenciement abusif.", order=1),
        Article(law_id=920, number="2", content="Le preavis de licenciement.", order=2),
    ])
    await db_session.commit()


class TestPasseTrigramme:
    """
    La passe trigramme sur le texte des articles coutait 5 a 13 s par faute
    de frappe (15,9 s pour « licensiement »). Coupee en hybride, bornee en
    mode texte.
    """

    @staticmethod
    def _espionner(monkeypatch):
        from app.services import postgres_search_service as module

        appels = []
        original = module._trigram_articles_query

        async def _espion(*a, **k):
            appels.append(a[1])
            return await original(*a, **k)

        monkeypatch.setattr(module, "_trigram_articles_query", _espion)
        return appels

    @pytest.mark.asyncio
    async def test_l_hybride_ne_la_lance_pas(self, db_session, articles_courts, monkeypatch):
        appels = self._espionner(monkeypatch)
        service = SearchService(db_session, use_cache=False)

        await service.search(SearchRequest(query="licensiement", mode="hybrid"))

        assert appels == []

    @pytest.mark.asyncio
    async def test_le_mode_texte_la_garde_sans_laisser_son_delai(
        self, db_session, articles_courts, monkeypatch
    ):
        """
        Le delai d'execution est pose dans un SAVEPOINT puis annule avec lui :
        laisse en place, il couperait les requetes suivantes de l'appelant
        (le chat enregistre ses messages dans la meme transaction).
        """
        appels = self._espionner(monkeypatch)
        avant = (await db_session.execute(text("SHOW statement_timeout"))).scalar()
        service = SearchService(db_session, use_cache=False)

        await service.search(SearchRequest(query="licensiement", mode="text"))

        assert appels == ["licensiement"]
        assert db_session.in_transaction()
        assert (await db_session.execute(text("SHOW statement_timeout"))).scalar() == avant


# ==================== 7. RE-RANKING HORS DE LA BOUCLE ====================


class TestReRankingHorsDeLaBoucle:

    @pytest.mark.asyncio
    async def test_le_re_ranking_tourne_sur_ses_propres_fils(
        self, db_session, articles_courts, monkeypatch
    ):
        """
        Du Python pur, 0,2 a 3 s mesures : dans la boucle, il gelait l'unique
        worker et toutes les autres requetes. Ni dans la boucle, ni sur
        l'executeur par defaut, que l'ingestion occupe.
        """
        from app.services import search_service as module

        fils = []
        original = module.rerank_chunks

        def _espion(question, chunks):
            fils.append(threading.current_thread().name)
            return original(question, chunks)

        monkeypatch.setattr(module, "rerank_chunks", _espion)
        monkeypatch.setattr(module.settings, "RERANK_ENABLED", True)
        service = SearchService(db_session, use_cache=False)

        reponse = await service.search(SearchRequest(query="licenciement", mode="text"))

        assert reponse.total >= 1
        assert len(fils) == 1 and fils[0].startswith("rerank")


# ==================== 8. LE VECTEUR PART AVANT LE PLEIN TEXTE ====================


class TestVecteurEnParallele:

    @pytest.mark.asyncio
    async def test_le_vecteur_est_demande_pendant_le_plein_texte(
        self, db_session, articles_courts
    ):
        """
        La question n'etait envoyee au fournisseur qu'apres le plein texte :
        ses 0,4 a 0,7 s s'ajoutaient au reste. Le vecteur est desormais
        demande avant que le plein texte ne finisse — et rendu seulement
        apres, ici, pour le prouver sans dependre d'une horloge.
        """
        evenements = []
        plein_texte_fini = asyncio.Event()

        class _Fournisseur:
            TASK_QUERY = EmbeddingService.TASK_QUERY

            async def generate_embedding_async(self, texte, task_type=None):
                evenements.append("vecteur demande")
                await plein_texte_fini.wait()
                return _vecteur(1)

        service = SearchService(db_session, use_cache=False)
        service.embedding_service = _Fournisseur()
        original = service.text_chunks

        async def _texte(*a, **k):
            resultat = await original(*a, **k)
            evenements.append("plein texte fini")
            plein_texte_fini.set()
            return resultat

        service.text_chunks = _texte

        await service.search(SearchRequest(query="licenciement", mode="hybrid"))

        assert evenements == ["vecteur demande", "plein texte fini"]
        assert service.semantique_degradee is False
