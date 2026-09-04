"""
Tests de la recherche au niveau chunk.

Chaque test de ce fichier correspond a un comportement qui etait casse et qui
passait inapercu :

- les articles etaient ecrases a un par loi (DISTINCT ON) ;
- l'ordre transmis a la fusion etait celui des identifiants, pas de la
  pertinence (ORDER BY law_id) ;
- l'identite de l'article (numero, section, page) n'arrivait jamais jusqu'au
  resultat ;
- l'extrait etait un prefixe brut de 400 caracteres, jamais un passage
  pertinent ;
- les filtres n'etaient pas transmis a la branche article, donc sans effet ;
- la fusion RRF etait indexee par loi, ce qui perdait un article sur deux.
"""

from datetime import date

import numpy as np
import pytest

from app.models.law import Article, Law
from app.schemas.search import SearchFilters, SearchRequest
from app.services.embedding_service import EmbeddingService
from app.services.postgres_search_service import search_articles_pg, search_laws_pg
from app.services.search_service import SearchService


def _unit_vector(seed: int) -> list:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=EmbeddingService.EMBEDDING_DIM)
    return (vec / np.linalg.norm(vec)).tolist()


@pytest.fixture
async def corpus(db_session):
    """
    Un code publie a trois articles, et un projet en brouillon.

    Les trois articles du meme code sont indispensables : c'est ce que le
    DISTINCT ON reduisait a un seul.
    """
    published = Law(
        id=100,
        reference="LOI-2024-100",
        title="Code des sociétés commerciales",
        content="Code régissant les sociétés commerciales et la responsabilité des dirigeants.",
        type="loi",
        language="fr",
        status="published",
        category_id=1,
        publication_date=date(2024, 5, 1),
    )
    draft = Law(
        id=101,
        reference="PROJET-2025-001",
        title="Projet de réforme de la responsabilité des dirigeants",
        content="Projet portant réforme de la responsabilité des dirigeants sociaux.",
        type="loi",
        language="fr",
        status="draft",
        category_id=1,
        publication_date=date(2025, 1, 10),
    )
    db_session.add_all([published, draft])
    await db_session.flush()

    articles = [
        Article(
            id=1001, law_id=100, number="1",
            title="Responsabilité des dirigeants",
            section="TITRE I — DE LA RESPONSABILITÉ",
            page_number=12,
            content=(
                "Les dirigeants sociaux engagent leur responsabilité envers la société "
                "pour les fautes commises dans l'exercice de leurs fonctions. "
                "La responsabilité des dirigeants est personnelle."
            ),
            order=1,
            embedding=_unit_vector(11),
        ),
        Article(
            id=1002, law_id=100, number="2",
            title="Responsabilité solidaire",
            section="TITRE I — DE LA RESPONSABILITÉ",
            page_number=13,
            content=(
                "La responsabilité des dirigeants peut être solidaire lorsque plusieurs "
                "dirigeants ont concouru au même fait dommageable."
            ),
            order=2,
            embedding=_unit_vector(12),
        ),
        Article(
            id=1003, law_id=100, number="3",
            title="Prescription",
            section="TITRE II — DE LA PRESCRIPTION",
            page_number=20,
            content="L'action en responsabilité se prescrit par trois ans.",
            order=3,
            embedding=_unit_vector(13),
        ),
        Article(
            id=1004, law_id=101, number="1",
            title="Objet",
            content="Le présent projet réforme la responsabilité des dirigeants sociaux.",
            order=1,
            embedding=_unit_vector(14),
        ),
    ]
    db_session.add_all(articles)
    await db_session.commit()

    # search_vector est pose par trigger a l'INSERT ; on le rafraichit pour les
    # lois, dont le contenu a ete ecrit directement.
    from sqlalchemy import text as sql_text

    await db_session.execute(sql_text("""
        UPDATE laws SET search_vector =
            to_tsvector('french', coalesce(title,'') || ' ' || coalesce(content,''))
            || to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
    """))
    await db_session.execute(sql_text("""
        UPDATE articles SET search_vector =
            to_tsvector('french', coalesce(content,''))
            || to_tsvector('english', coalesce(content,''))
            || to_tsvector('simple', coalesce(number,''))
    """))
    await db_session.commit()

    return {"published": published, "draft": draft, "articles": articles}


@pytest.fixture
async def service(db_session, mock_embedding_service_chunks):
    svc = SearchService(db_session, use_cache=False)
    svc.embedding_service = mock_embedding_service_chunks
    return svc


@pytest.fixture
def mock_embedding_service_chunks():
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock.TASK_QUERY = "RETRIEVAL_QUERY"
    mock.TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    mock.generate_embedding_async = AsyncMock(
        return_value=np.array(_unit_vector(11), dtype=np.float32)
    )
    return mock


class TestChunkLevelFTS:

    @pytest.mark.asyncio
    async def test_returns_several_chunks_from_same_law(self, db_session, corpus):
        """Acte de deces du DISTINCT ON (law_id)."""
        chunks = await search_articles_pg(db_session, "responsabilité dirigeants", None, 15, 0)

        same_law = [c for c in chunks if c.law_id == 100]
        assert len(same_law) >= 2
        assert len({c.article_id for c in same_law}) == len(same_law)

    @pytest.mark.asyncio
    async def test_chunks_are_in_relevance_order(self, db_session, corpus):
        """Acte de deces du ORDER BY law_id."""
        chunks = await search_articles_pg(db_session, "responsabilité dirigeants", None, 15, 0)

        scores = [c.relevance_score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_chunks_carry_article_identity(self, db_session, corpus):
        """Numero, section et page arrivent jusqu'au resultat."""
        chunks = await search_articles_pg(db_session, "responsabilité", None, 15, 0)

        target = next(c for c in chunks if c.article_id == 1001)
        assert target.number == "1"
        assert target.article_title == "Responsabilité des dirigeants"
        assert target.section.startswith("TITRE I")
        assert target.page_number == 12
        assert "responsabilité" in target.content.lower()
        assert target.fusion_key == ("a", 1001)

    @pytest.mark.asyncio
    async def test_excerpt_is_highlighted(self, db_session, corpus):
        """ts_headline, et non un prefixe aveugle de 400 caracteres."""
        chunks = await search_articles_pg(db_session, "solidaire", None, 15, 0)

        matching = [c for c in chunks if c.article_id == 1002]
        assert matching, "l'article 2 devrait correspondre a 'solidaire'"
        assert "<mark>" in matching[0].excerpt.lower()

    @pytest.mark.asyncio
    async def test_excerpt_is_centred_on_the_match(self, db_session, corpus):
        """
        L'extrait suit le terme cherche.

        Le mecanisme precedent renvoyait content[:400], donc toujours le debut
        de l'article, quel que soit l'endroit ou se trouvait la correspondance.
        """
        chunks = await search_articles_pg(db_session, "solidaire", None, 15, 0)

        matching = [c for c in chunks if c.article_id == 1002]
        assert "solidaire" in matching[0].excerpt.lower()

    @pytest.mark.asyncio
    async def test_filters_apply_on_article_branch(self, db_session, corpus):
        """
        Le filtre de statut mord enfin sur la branche article.

        Le projet en brouillon correspond a la requete ; il ne doit pas
        remonter. Auparavant search_articles_pg n'acceptait aucun filtre.
        """
        chunks = await search_articles_pg(
            db_session, "responsabilité dirigeants",
            SearchFilters(status="published"), 15, 0,
        )

        assert chunks
        assert all(c.status == "published" for c in chunks)
        assert all(c.law_id != 101 for c in chunks)

    @pytest.mark.asyncio
    async def test_filter_year_range_on_article_branch(self, db_session, corpus):
        """year_from / year_to n'etaient appliques nulle part."""
        chunks = await search_articles_pg(
            db_session, "responsabilité dirigeants",
            SearchFilters(year_from=2025), 15, 0,
        )

        assert all(c.law_id == 101 for c in chunks)

    @pytest.mark.asyncio
    async def test_law_fallback_rows_have_no_article_id(self, db_session, corpus):
        chunks = await search_laws_pg(db_session, "sociétés commerciales", None, 5, 0)

        assert chunks
        assert all(c.article_id is None for c in chunks)
        assert all(c.fusion_key[0] == "l" for c in chunks)


class TestChunkLevelSemantic:

    @pytest.mark.asyncio
    async def test_semantic_returns_per_article_rows(self, service, corpus):
        """Plus de func.max() + GROUP BY : chaque article a sa ligne."""
        chunks = await service.semantic_chunks("responsabilité", None, 10, 0)

        same_law = [c for c in chunks if c.law_id == 100]
        assert len(same_law) >= 2
        assert all(c.article_id is not None for c in same_law)
        assert all(c.content for c in same_law)

    @pytest.mark.asyncio
    async def test_semantic_scores_stay_in_range(self, service, corpus):
        """
        Avec de vrais vecteurs la distance cosinus peut depasser 1, donc
        1 - distance devenir negatif — or relevance_score est declare ge=0.0.
        """
        chunks = await service.semantic_chunks("responsabilité", None, 10, 0)

        assert all(0.0 <= c.relevance_score <= 1.0 for c in chunks)


class TestLawLevelDerivation:

    @pytest.mark.asyncio
    async def test_law_results_derived_from_chunks(self, service, corpus):
        """
        Les resultats niveau document portent enfin leurs articles.

        matched_articles etait vide sur le chemin semantique et content nul.
        """
        response = await service.search(SearchRequest(
            query="responsabilité dirigeants", mode="text", limit=5,
        ))

        assert response.results
        assert response.chunks
        first = response.results[0]
        assert first.matched_articles
        assert first.matched_articles[0].article_id is not None
        assert first.content

    @pytest.mark.asyncio
    async def test_law_results_are_deduplicated(self, service, corpus):
        """Une loi n'apparait qu'une fois cote document, malgre N chunks."""
        response = await service.search(SearchRequest(
            query="responsabilité dirigeants", mode="text", limit=5,
        ))

        law_ids = [r.law_id for r in response.results]
        assert len(law_ids) == len(set(law_ids))


class TestTwoStageSemantic:
    """
    L'index est parcouru en fp16 (halfvec), le classement final se fait en fp32.

    Le type `vector` n'est indexable que jusqu'a 2000 dimensions, `halfvec`
    jusqu'a 4000 : c'est ce qui permet de garder 3072 dimensions en stockage
    exact tout en ayant un index. Le fp16 ne doit servir qu'a SELECTIONNER des
    candidats, jamais a les classer.
    """

    @pytest.fixture
    async def near_ties(self, db_session):
        """
        Deux articles dont l'ecart de distance est inferieur a la resolution
        du fp16 : c'est la seule configuration ou les deux ordres divergent.
        """
        dim = EmbeddingService.EMBEDDING_DIM

        def _half(vec):
            return vec.astype(np.float16).astype(np.float64)

        # Recherche deterministe d'un couple que fp16 et fp32 classent
        # DIFFEREMMENT. Deux vecteurs pris au hasard s'accordent presque
        # toujours : sans cette recherche, le test se contenterait de sauter et
        # ne prouverait jamais rien.
        vec_a = vec_b = base = None
        for seed in range(200):
            rng = np.random.default_rng(seed)
            candidate = rng.normal(size=dim)
            candidate /= np.linalg.norm(candidate)

            def _perturb(scale: float) -> np.ndarray:
                noise = rng.normal(size=dim)
                noise -= noise.dot(candidate) * candidate   # orthogonal
                noise /= np.linalg.norm(noise)
                vec = candidate + scale * noise
                return vec / np.linalg.norm(vec)

            a, b = _perturb(0.0200), _perturb(0.020002)
            exact = (np.dot(candidate, a), np.dot(candidate, b))
            half = (np.dot(_half(candidate), _half(a)),
                    np.dot(_half(candidate), _half(b)))
            if (exact[0] > exact[1]) != (half[0] > half[1]):
                base, vec_a, vec_b = candidate, a, b
                break

        if base is None:
            pytest.skip("aucun couple divergent fp16/fp32 trouve en 200 tirages")

        law = Law(
            id=300, reference="LOI-TIES", title="Loi ex aequo",
            content="Contenu.", type="loi", language="fr", status="published",
        )
        db_session.add(law)
        await db_session.flush()
        db_session.add_all([
            Article(id=3001, law_id=300, number="1", content="Article A.",
                    order=1, embedding=vec_a.tolist()),
            Article(id=3002, law_id=300, number="2", content="Article B.",
                    order=2, embedding=vec_b.tolist()),
        ])
        await db_session.commit()
        return {"query": base, "a": vec_a, "b": vec_b}

    @pytest.mark.asyncio
    async def test_exact_rerank_wins_over_halfvec_order(
        self, db_session, near_ties, mock_embedding_service_chunks
    ):
        query = near_ties["query"]
        mock_embedding_service_chunks.generate_embedding_async.return_value = (
            query.astype(np.float32)
        )
        service = SearchService(db_session, use_cache=False)
        service.embedding_service = mock_embedding_service_chunks

        # La fixture garantit que fp16 et fp32 divergent sur ce couple.
        similarity = [
            float(np.dot(query, near_ties["a"])),
            float(np.dot(query, near_ties["b"])),
        ]
        exact_order = [3001, 3002] if similarity[0] >= similarity[1] else [3002, 3001]

        chunks = await service.semantic_chunks("question", None, 10, 0)

        assert [c.article_id for c in chunks[:2]] == exact_order

    @pytest.mark.asyncio
    async def test_relevance_score_comes_from_the_exact_distance(
        self, db_session, near_ties, mock_embedding_service_chunks
    ):
        """
        Le score rendu doit valoir 1 - distance EXACTE, pas 1 - distance fp16.

        Sinon un appelant qui re-trie sur le score reordonne silencieusement le
        resultat par rapport a l'ordre rendu.
        """
        query = near_ties["query"]
        mock_embedding_service_chunks.generate_embedding_async.return_value = (
            query.astype(np.float32)
        )
        service = SearchService(db_session, use_cache=False)
        service.embedding_service = mock_embedding_service_chunks

        chunks = await service.semantic_chunks("question", None, 10, 0)

        by_id = {c.article_id: c for c in chunks}
        for article_id, key in ((3001, "a"), (3002, "b")):
            # relevance_score = 1 - distance, et distance = 1 - similarite
            # cosinus : le score EST la similarite cosinus exacte.
            expected = float(np.dot(query, near_ties[key]))
            assert by_id[article_id].relevance_score == pytest.approx(expected, abs=1e-5)

    @pytest.mark.asyncio
    async def test_candidate_budget_is_bounded(self, db_session):
        service = SearchService(db_session, use_cache=False)

        assert service._ann_candidates(1) == service.ANN_MIN_CANDIDATES
        assert service._ann_candidates(8) == service.ANN_MIN_CANDIDATES
        assert service._ann_candidates(50) == 400
        assert service._ann_candidates(10_000) == service.ANN_MAX_CANDIDATES


class TestRerankWiring:
    """Le re-ranking doit s'appliquer AVANT la troncature et AVANT le cache."""

    @pytest.mark.asyncio
    async def test_rerank_applied_before_cache_write(self, db_session, corpus, monkeypatch):
        """
        Reclasser apres l'ecriture du cache laisserait cinq minutes de reponses
        pre-reranking en circulation apres chaque deploiement.
        """
        from app.services import search_service as module

        written = {}

        async def _capture(db, key, payload, ttl_seconds=None):
            written["payload"] = payload

        monkeypatch.setattr(module, "store_in_pg_cache", _capture)

        service = SearchService(db_session, use_cache=True)
        service.embedding_service = None

        response = await service.search(SearchRequest(
            query="responsabilité dirigeants", mode="text", limit=5,
        ))

        assert written, "le cache doit avoir ete ecrit"
        cached_ids = [c["article_id"] for c in written["payload"]["chunks"]]
        assert cached_ids == [c.article_id for c in response.chunks]
        assert all(c.rerank_score is not None for c in response.chunks)

    @pytest.mark.asyncio
    async def test_rerank_can_be_disabled(self, db_session, corpus, monkeypatch):
        from app.services import search_service as module

        monkeypatch.setattr(module.settings, "RERANK_ENABLED", False)

        service = SearchService(db_session, use_cache=False)
        service.embedding_service = None

        response = await service.search(SearchRequest(
            query="responsabilité dirigeants", mode="text", limit=5,
        ))

        assert all(c.rerank_score is None for c in response.chunks)


class TestSuggestions:
    """
    /search/suggest renvoyait une liste vide pour a peu pres toute saisie.

    Deux causes cumulees : similarity() compare la requete au titre ENTIER (un
    mot de dix caracteres contre un titre de quatre-vingt-dix plafonne vers 0,13,
    tres sous le seuil de 0,3), et la branche prefixe ne pouvait rien rattraper
    puisqu'aucun titre ne commence par le mot cherche. S'y ajoutait la sensibilite
    aux accents, alors que personne ne tape "société" avec son accent.
    """

    @pytest.mark.asyncio
    async def test_finds_a_word_inside_a_long_title(self, client, corpus):
        response = await client.get("/api/v1/search/suggest?q=commerciales&limit=5")

        assert response.status_code == 200
        suggestions = response.json()["suggestions"]
        assert suggestions, "un mot present au milieu du titre doit suggerer"
        assert any("commerciales" in s["title"].lower() for s in suggestions)

    @pytest.mark.asyncio
    async def test_is_accent_insensitive(self, client, corpus):
        """"societes" sans accent doit trouver "sociétés"."""
        accented = await client.get("/api/v1/search/suggest?q=sociétés&limit=5")
        plain = await client.get("/api/v1/search/suggest?q=societes&limit=5")

        assert accented.status_code == plain.status_code == 200
        assert plain.json()["suggestions"], "la saisie sans accents doit suggerer"
        assert (
            {s["id"] for s in plain.json()["suggestions"]}
            == {s["id"] for s in accented.json()["suggestions"]}
        )

    @pytest.mark.asyncio
    async def test_short_query_returns_nothing(self, client, corpus):
        response = await client.get("/api/v1/search/suggest?q=a&limit=5")

        assert response.status_code == 200
        assert response.json()["suggestions"] == []
