"""
Le blog public ne montre que des articles aux citations vérifiées.

Correctif du 04/11/2026 : les articles écrits par le premier générateur (et les
trois du 26/10) portent des citations jamais vérifiées et un texte où la
vérification du 29/10 a relevé des erreurs. Ils restent en base, mais ni la
liste publique ni leur adresse ne les servent.
"""

import pytest

from app.models.blog import BlogPost

pytestmark = pytest.mark.asyncio


def _article(slug: str, citations: list, publie: bool = True) -> BlogPost:
    return BlogPost(
        slug=slug,
        title=f"Titre {slug}",
        category="Travail",
        context="Contexte.",
        summary="Résumé.",
        key_points=["Point."],
        citations=citations,
        citizen_impact="Impact.",
        suggested_question="Question ?",
        tags=[],
        is_published=publie,
    )


CITATION_NON_VERIFIEE = {"law_id": 1, "article_number": "39", "reference": "Code du travail", "excerpt": "…"}
CITATION_VERIFIEE = {**CITATION_NON_VERIFIEE, "verifie": True}


async def test_un_article_aux_citations_non_verifiees_n_est_pas_servi(client, db_session):
    db_session.add(_article("ancien", [CITATION_NON_VERIFIEE]))
    await db_session.commit()

    liste = await client.get("/api/v1/blog")
    assert liste.status_code == 200
    assert liste.json() == []
    assert (await client.get("/api/v1/blog/ancien")).status_code == 404


async def test_un_article_sans_citation_n_est_pas_servi(client, db_session):
    db_session.add(_article("sans-source", []))
    await db_session.commit()

    assert (await client.get("/api/v1/blog")).json() == []
    assert (await client.get("/api/v1/blog/sans-source")).status_code == 404


async def test_un_brouillon_n_est_jamais_servi_meme_verifie(client, db_session):
    db_session.add(_article("brouillon", [CITATION_VERIFIEE], publie=False))
    await db_session.commit()

    assert (await client.get("/api/v1/blog")).json() == []
    assert (await client.get("/api/v1/blog/brouillon")).status_code == 404


async def test_un_article_publie_aux_citations_verifiees_est_servi(client, db_session):
    db_session.add(_article("verifie", [CITATION_VERIFIEE]))
    db_session.add(_article("melange", [CITATION_VERIFIEE, CITATION_NON_VERIFIEE]))
    await db_session.commit()

    liste = (await client.get("/api/v1/blog")).json()
    assert [article["slug"] for article in liste] == ["verifie"]
    assert (await client.get("/api/v1/blog/verifie")).status_code == 200
    assert (await client.get("/api/v1/blog/melange")).status_code == 404
