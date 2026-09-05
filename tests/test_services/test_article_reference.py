"""
« article 35 du code minier » ouvre le bon document sur le bon article.

Trois choses devaient exister ensemble : reconnaitre le numero, reconnaitre le
document, et verifier que l'article s'y trouve. Avant, la premiere seule etait
faite — `_parse_article_reference` extrayait bien `doc_hint`, mais personne ne
s'en servait, et `direct_navigation` etait conditionne a `total == 1`, ce qui ne
se produit presque jamais.

Usage:
    pytest tests/test_services/test_article_reference.py -v
"""

from datetime import date

import pytest

from app.models.law import Article, Law
from app.schemas.search import SearchRequest
from app.services.article_reference import (
    ArticleReference,
    normalize_number,
    parse_reference,
)
from app.services.search_service import SearchService
from app.services.search_vectors import REINDEX_ARTICLES_SQL, REINDEX_LAWS_SQL

CODE_MINIER = 300
CODE_TRAVAIL = 301


@pytest.fixture
async def codes(db_session, category_ids):
    from sqlalchemy import text as sql_text

    laws = [
        Law(id=CODE_MINIER, reference="LOI-2023-014",
            title="Loi N°2023/014 du 19 décembre 2023 portant Code Minier",
            content="Code régissant les activités minières au Cameroun.",
            type="loi", language="fr", status="published",
            category_id=category_ids["Droit de l'Environnement et des Ressources Naturelles"],
            publication_date=date(2023, 12, 19)),
        Law(id=CODE_TRAVAIL, reference="LOI-1992-007",
            title="Loi N°92/007 du 14 août 1992 portant Code du Travail",
            content="Code régissant les relations de travail.",
            type="loi", language="fr", status="published",
            category_id=category_ids["Droit du Travail et Sécurité Sociale"],
            publication_date=date(1992, 8, 14)),
    ]
    db_session.add_all(laws)
    await db_session.flush()

    articles = []
    for law_id, numbers in ((CODE_MINIER, ("1", "35", "36")), (CODE_TRAVAIL, ("1", "2"))):
        for number in numbers:
            articles.append(Article(
                law_id=law_id, number=number,
                content=(f"Dispositions de l'article {number}. "
                         "La superficie couverte par le permis de recherche est limitée."),
                order=int(number), kind="article", embed=True,
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


class TestParsing:
    """Le module unique remplace huit motifs « article N » divergents."""

    @pytest.mark.parametrize("query,number,hint", [
        ("article 35 du code minier", "35", "code minier"),
        ("Article 35 de la Constitution", "35", "Constitution"),
        ("art. 35 du code du travail", "35", "code du travail"),
        # « art.35 » sans espace : le motif du RAG exigeait `art\.?\s+`.
        ("art.35", "35", ""),
        ("article 35", "35", ""),
        ("article premier de la constitution", "1", "constitution"),
        ("article 1er du code minier", "1", "code minier"),
        ("article L 94 du code des impots", "L 94", "code des impots"),
    ])
    def test_parses(self, query, number, hint):
        reference = parse_reference(query)
        assert reference == ArticleReference(number, hint), f"pour {query!r}"

    @pytest.mark.parametrize("query", [
        "code minier", "permis de recherche", "", "les articles du code",
    ])
    def test_not_a_reference(self, query):
        assert parse_reference(query) is None

    @pytest.mark.parametrize("raw,expected", [
        ("Article 1er", "1"), ("PREMIER", "1"), ("1ÈRE", "1"),
        ("deuxieme", "2"), ("35", "35"), ("Art. 35", "35"), ("", ""),
    ])
    def test_normalisation(self, raw, expected):
        """`articles.number` stocke « 1 » : rendre « PREMIER » ne matcherait rien."""
        assert normalize_number(raw) == expected


class TestDirectNavigation:
    @pytest.mark.asyncio
    async def test_opens_the_right_document_on_the_right_article(self, service, codes):
        response = await service.search(
            SearchRequest(query="article 35 du code minier", mode="text", limit=10)
        )

        assert response.target_law_id == CODE_MINIER
        assert response.target_article == "35"
        assert response.direct_navigation is True

    @pytest.mark.asyncio
    async def test_the_document_hint_tolerates_a_typo(self, service, codes):
        """word_similarity('code miner', 'Loi ... portant Code Minier') = 0,727."""
        response = await service.search(
            SearchRequest(query="article 35 du code miner", mode="text", limit=10)
        )

        assert response.target_law_id == CODE_MINIER
        assert response.direct_navigation is True

    @pytest.mark.asyncio
    async def test_the_hint_selects_between_two_codes(self, service, codes):
        response = await service.search(
            SearchRequest(query="article 1 du code du travail", mode="text", limit=10)
        )

        assert response.target_law_id == CODE_TRAVAIL

    @pytest.mark.asyncio
    async def test_an_absent_article_does_not_navigate(self, service, codes):
        """
        Ouvrir un document sur un article qui n'y est pas serait pire que ne pas
        l'ouvrir : l'utilisateur croirait avoir lu la reponse.
        """
        response = await service.search(
            SearchRequest(query="article 999 du code minier", mode="text", limit=10)
        )

        assert response.direct_navigation is False
        assert response.target_article == "999"
        assert response.results, "les resultats de recherche doivent rester"

    @pytest.mark.asyncio
    async def test_article_premier_resolves_to_the_stored_number(self, service, codes):
        response = await service.search(
            SearchRequest(query="article premier du code minier", mode="text", limit=10)
        )

        assert response.target_article == "1"
        assert response.target_law_id == CODE_MINIER

    @pytest.mark.asyncio
    async def test_a_plain_query_does_not_navigate(self, service, codes):
        response = await service.search(
            SearchRequest(query="permis de recherche", mode="text", limit=10)
        )

        assert response.direct_navigation is False
        assert response.target_article is None


class TestRemovedEndpoint:
    @pytest.mark.asyncio
    async def test_the_dead_article_endpoint_is_gone(self, client, db_session):
        """
        Repondait 500 : il appelait `_parse_article_reference` ET
        `_estimate_page_number`, aucun des deux defini dans ce module. Aucun
        test ne le couvrait — c'est ce qui lui a permis de survivre.
        """
        response = await client.get("/api/v1/search/article?q=article%205")

        assert response.status_code == 404
