"""
Le titre passe devant le corps du texte, et les fautes de frappe passent.

Ces tests figent trois comportements qui n'existaient pas :

1. Un document dont le TITRE porte le mot cherche arrive avant un document qui
   ne le porte que dans son corps. Mesure avant correctif, requete
   « nomination » : le document dont le titre ne contient PAS le mot sortait
   PREMIER (ts_rank_cd 0,4000) devant cinq documents dont le titre le porte
   (0,2000), parce qu'il le repete dans son corps.
2. Une faute de frappe trouve quand meme, sans jamais devancer une
   correspondance exacte.
3. Une seule loi ne peut plus prendre tout le budget de chunks. Mesure avant
   correctif : « avancement de grade » rendait 60 chunks issus de la MEME loi,
   donc UN SEUL document affiche.

Le principe de ces tests : ils assertent un ORDRE et des IDENTIFIANTS. Le test
qu'ils remplacent (`test_text_search_typo_tolerance`) n'assertait que
`isinstance(results, list)`, ne demandait meme pas de donnees, et passait sur une
base vide.

Usage:
    pytest tests/test_services/test_title_priority.py -v
"""

from datetime import date

import pytest

from app.models.law import Article, Law
from app.schemas.search import SearchRequest
from app.services.postgres_search_service import (
    MAX_CHUNKS_PER_LAW,
    search_articles_pg,
)
from app.services.search_service import SearchService
from app.services.search_vectors import REINDEX_ARTICLES_SQL, REINDEX_LAWS_SQL


# Loi 200-204 : le mot « nomination » est dans le TITRE.
# Loi 210-211 : il n'est QUE dans le corps, et repete, ce qui suffisait a les
# faire passer devant.
TITLE_LAWS = (200, 201, 202, 203, 204)
BODY_LAWS = (210, 211)


@pytest.fixture
async def ranking_corpus(db_session, category_ids):
    """Cinq lois « nomination » par le titre, deux par le corps seulement."""
    from sqlalchemy import text as sql_text

    laws = []
    for offset, law_id in enumerate(TITLE_LAWS):
        laws.append(Law(
            id=law_id, reference=f"DEC-2024-{law_id}",
            title=f"Décret N°2024/{law_id} portant nomination d'un Directeur",
            content="Le présent décret sera enregistré et publié au Journal Officiel.",
            type="décret", language="fr", status="published",
            category_id=category_ids["Fonction Publique"],
            publication_date=date(2024, 1, 1 + offset),
        ))
    for offset, law_id in enumerate(BODY_LAWS):
        laws.append(Law(
            id=law_id, reference=f"DEC-2024-{law_id}",
            title=f"Décret N°2024/{law_id} portant approbation des statuts",
            # « nomination » repete : c'est ce qui faisait gagner ces lois.
            content=("La nomination des dirigeants, la nomination du président et "
                     "la nomination des commissaires obéissent aux statuts."),
            type="décret", language="fr", status="published",
            category_id=category_ids["Droit des Affaires et OHADA"],
            publication_date=date(2024, 2, 1 + offset),
        ))
    db_session.add_all(laws)
    await db_session.flush()

    # 25 articles par loi : sans plafond par loi, UNE SEULE loi remplirait les
    # 60 chunks du budget et le front n'afficherait qu'un document. C'est le
    # comportement mesure sur le corpus reel avant correctif.
    articles = []
    for law in laws:
        for number in range(1, 26):
            articles.append(Article(
                law_id=law.id, number=str(number),
                content=(f"Article {number} du texte. " + (law.content or "")) * 2,
                order=number, kind="article", embed=True,
            ))
    db_session.add_all(articles)
    await db_session.commit()

    await db_session.execute(sql_text(REINDEX_LAWS_SQL))
    await db_session.execute(sql_text(REINDEX_ARTICLES_SQL))
    await db_session.commit()
    return laws


@pytest.fixture
def service(db_session):
    return SearchService(db_session, use_cache=False)


class TestTitleBeforeBody:
    @pytest.mark.asyncio
    async def test_a_title_match_leads_a_body_only_match(self, service, ranking_corpus):
        results = await service.text_search("nomination", limit=20)

        assert results, "aucun resultat"
        assert results[0].law_id in TITLE_LAWS, (
            f"le premier resultat est {results[0].law_id}, qui ne porte pas le mot "
            "dans son titre"
        )

    @pytest.mark.asyncio
    async def test_no_body_match_appears_before_a_title_match(self, service, ranking_corpus):
        scopes = [r.match_scope for r in await service.text_search("nomination", limit=20)]

        assert scopes, "aucun resultat"
        assert scopes == sorted(scopes, key=lambda s: s != "title"), (
            f"ordre des portees obtenu : {scopes}"
        )

    @pytest.mark.asyncio
    async def test_nothing_is_lost(self, service, ranking_corpus):
        """« Titre d'abord » ne veut pas dire « titre seulement »."""
        found = {r.law_id for r in await service.text_search("nomination", limit=20)}

        assert set(TITLE_LAWS) <= found
        assert set(BODY_LAWS) <= found, "les correspondances de corps ont disparu"

    @pytest.mark.asyncio
    async def test_match_scope_is_exact_not_guessed(self, service, ranking_corpus):
        by_law = {r.law_id: r.match_scope for r in await service.text_search("nomination", limit=20)}

        for law_id in TITLE_LAWS:
            assert by_law[law_id] == "title"
        for law_id in BODY_LAWS:
            assert by_law[law_id] == "body"

    @pytest.mark.asyncio
    async def test_a_word_absent_from_every_title_still_returns_results(
        self, service, ranking_corpus
    ):
        """La branche titre s'ajoute a la recherche de corps, elle ne la remplace pas."""
        results = await service.text_search("commissaires", limit=20)

        assert {r.law_id for r in results} >= set(BODY_LAWS)
        assert all(r.match_scope == "body" for r in results)

    @pytest.mark.asyncio
    async def test_the_stored_vector_carries_the_A_weight(self, db_session, ranking_corpus):
        """Sans `setweight`, tout ce qui precede est indecidable."""
        from sqlalchemy import text as sql_text

        vector = (await db_session.execute(sql_text(
            "SELECT search_vector::text FROM laws WHERE id = :i"), {"i": TITLE_LAWS[0]}
        )).scalar_one()

        assert "A" in vector, "aucun lexeme de poids A : la ponderation est absente"


class TestTypoTolerance:
    @pytest.mark.asyncio
    async def test_a_typo_finds_exactly_the_right_laws(self, service, ranking_corpus):
        """« nominaton » ne rend RIEN en plein texte ; le trigramme rattrape."""
        results = await service.text_search("nominaton", limit=20)

        assert {r.law_id for r in results} == set(TITLE_LAWS)
        assert all(r.match_scope == "title" for r in results)

    @pytest.mark.asyncio
    async def test_a_typo_below_the_default_threshold_is_still_caught(
        self, service, ranking_corpus
    ):
        """
        word_similarity('nominasion', <titre>) = 0,571 : SOUS le defaut de
        session de l'operateur %> (0,6), au-dessus du seuil configure (0,5).
        Ce test echoue si le `SET LOCAL` disparait — la faute moyenne serait
        alors silencieusement perdue.
        """
        results = await service.text_search("nominasion", limit=20)

        assert {r.law_id for r in results} == set(TITLE_LAWS)

    @pytest.mark.asyncio
    async def test_noise_is_not_a_typo(self, service, ranking_corpus):
        """« fonciere » plafonne a 0,333 sur ce corpus : du bruit, pas une faute."""
        assert await service.text_search("fonciere", limit=20) == []

    @pytest.mark.asyncio
    async def test_an_exact_match_is_never_pushed_down_by_a_fuzzy_one(
        self, service, ranking_corpus
    ):
        results = await service.text_search("nomination", limit=20)

        assert results[0].law_id in TITLE_LAWS
        assert results[0].relevance_score >= max(r.relevance_score for r in results)


class TestPerLawCap:
    @pytest.mark.asyncio
    async def test_one_law_cannot_take_the_whole_budget(self, db_session, ranking_corpus):
        """
        Sans plafond, une loi dont le titre correspond marque le meme score sur
        TOUS ses articles et rafle le budget. Mesure sur le corpus reel :
        « avancement de grade » rendait 60 chunks, tous de la meme loi.
        """
        from collections import Counter

        chunks = await search_articles_pg(db_session, "nomination", None, 60, 0)
        per_law = Counter(c.law_id for c in chunks)

        assert per_law, "aucun chunk"
        assert max(per_law.values()) <= MAX_CHUNKS_PER_LAW
        assert len(per_law) >= 4, f"un seul document remonte : {dict(per_law)}"


class TestTotalCount:
    @pytest.mark.asyncio
    async def test_total_counts_documents_not_the_page(self, service, ranking_corpus):
        """
        `total` valait `len(results)` APRES la decoupe de page, donc toujours
        <= limit : le front calcule `Math.ceil(total / 20)` et n'affichait
        jamais plus d'une page.
        """
        response = await service.search(
            SearchRequest(query="nomination", mode="text", limit=3)
        )

        assert len(response.results) == 3
        assert response.total >= 7, f"total={response.total}, attendu au moins 7"
