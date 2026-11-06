"""
Le blog public ne montre que des articles aux citations vérifiées.

Correctif du 04/11/2026 : les articles écrits par le premier générateur (et les
trois du 26/10) portent des citations jamais vérifiées et un texte où la
vérification du 29/10 a relevé des erreurs. Ils restent en base, mais ni la
liste publique ni leur adresse ne les servent.

Règle renforcée par le blog refait : des citations marquées vérifiées ne
suffisent plus. Un article n'est servi que publié ET publiable, c'est-à-dire
avec ses sources gardées, un contrôle IA à jour sans phrase contredite et la
relecture humaine de chaque partie (app/api/routes/blog.py). Le cas positif,
l'article passé par tout le circuit, est couvert par test_blog_sources_routes.py
(TestLesArticlesDAvant) et test_blog_routes.py.
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


async def test_des_citations_marquees_verifiees_ne_suffisent_plus(client, db_session):
    """
    Publié, toutes ses citations marquées vérifiées, mais sans sources gardées,
    sans contrôle IA ni relecture : ce n'est pas un article publiable, il n'est
    pas servi. Le correctif du 04/11 le servait ; le blog refait, non.
    """
    db_session.add(_article("verifie", [CITATION_VERIFIEE]))
    db_session.add(_article("melange", [CITATION_VERIFIEE, CITATION_NON_VERIFIEE]))
    await db_session.commit()

    assert (await client.get("/api/v1/blog")).json() == []
    assert (await client.get("/api/v1/blog/verifie")).status_code == 404
    assert (await client.get("/api/v1/blog/melange")).status_code == 404
