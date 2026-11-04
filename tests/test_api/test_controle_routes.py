"""
GET /api/v1/controle/chat : la route publique du controle du chat
(app/api/routes/controle.py, app/services/controle_du_chat.py).

Ce que cron-job.org en attend : 200 quand les questions garanties tiennent,
503 quand quelque chose de NOUVEAU echoue (dit dans `raisons`), un rapport
lisible, et une reponse dans ses 30 secondes ; 202 (un succes pour lui) tant
qu'aucun rapport n'existe, pour qu'un redemarrage ne fasse pas sonner
l'alerte. Les faiblesses connues ne changent pas le code : elles sont dans le
rapport, et une faiblesse qui tient y est dite « faiblesse résolue ». Ce
qu'un inconnu ne doit pas en obtenir : plus d'un calcul par demi-heure, ou
un secret.
"""

import asyncio
from functools import partial

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.routes import controle as route
from app.core.database import AsyncSessionLocal
from app.main import app
from app.services import search_service
from app.services.controle_du_chat import (
    ControleEnCache,
    controler_le_chat,
    sonder_le_fournisseur,
)
from app.services.questions_de_reference import FAIBLESSES_CONNUES, QUESTIONS_DE_REFERENCE
from app.services.rag_service import RAGService
from tests.test_services.test_controle_du_chat import _corpus, _question, _VecteurDeLaQuestion

URL = "/api/v1/controle/chat"


def _rapport(statut="ok", **champs):
    return {"statut": statut, "questions": 44, "en_echec": 0 if statut == "ok" else 1, **champs}


@pytest.fixture
def poser():
    """`poser(calculer)` branche un controle neuf sur la route, et le rend."""

    def _poser(calculer, **reglages):
        controle = ControleEnCache(calculer, **reglages)
        app.dependency_overrides[route.controle_en_cache] = lambda: controle
        return controle

    yield _poser
    app.dependency_overrides.pop(route.controle_en_cache, None)


@pytest.fixture
async def http():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_tout_tient_200_et_le_rapport(poser, http):
    async def calculer():
        return _rapport()

    poser(calculer)
    reponse = await http.get(URL)

    assert reponse.status_code == 200
    assert reponse.json()["statut"] == "ok" and reponse.json()["age_s"] == 0
    # Jamais une copie gardee en route : chaque appel lit l'etat du serveur.
    assert reponse.headers["cache-control"] == "no-store"


async def test_un_echec_503_avec_les_questions_en_echec(poser, http):
    echec = {"question": "Quelles indemnités en cas de licenciement abusif ?",
             "raisons": ["textes attendus absents du contexte lu par le modele : LOI-1992-007"]}

    async def calculer():
        return _rapport("echec", echecs=[echec])

    poser(calculer)
    reponse = await http.get(URL)

    assert reponse.status_code == 503
    assert reponse.json()["echecs"] == [echec]


async def test_vingt_appels_simultanes_un_seul_calcul(poser, http):
    appels = []

    async def calculer():
        appels.append(1)
        await asyncio.sleep(0.05)
        return _rapport()

    poser(calculer)
    reponses = await asyncio.gather(*(http.get(URL) for _ in range(20)))
    reponses += [await http.get(URL) for _ in range(5)]

    assert {r.status_code for r in reponses} == {200}
    assert len(appels) == 1


async def test_une_erreur_ne_sort_que_par_sa_classe(poser, http):
    async def calculer():
        raise ConnectionError("postgresql://jurix:motdepasse@ep-secret.neon.tech/jurix")

    poser(calculer)
    reponse = await http.get(URL)

    assert reponse.status_code == 503
    assert reponse.json()["raison"] == "controle en erreur (ConnectionError)"
    assert "motdepasse" not in reponse.text and "neon" not in reponse.text


async def test_un_calcul_trop_long_rend_la_main_avant_cron_job_org(poser, http):
    async def calculer():
        await asyncio.sleep(0.3)
        return _rapport()

    poser(calculer, attente_max_s=0.02)
    reponse = await http.get(URL)

    assert reponse.status_code == 202 and reponse.json()["statut"] == "en_cours"
    await asyncio.sleep(0.4)
    assert (await http.get(URL)).status_code == 200


async def test_un_calcul_trop_long_rend_l_echec_precedent_marque_perime(poser, http):
    """L'appel de la minute 0 lance le calcul ; il rend le rapport d'avant, avec son code."""
    horloge = [1000.0]
    appels = []

    async def calculer():
        appels.append(1)
        if len(appels) > 1:
            await asyncio.sleep(0.3)
            return _rapport()
        return _rapport("echec", echecs=[{"question": "Q", "raisons": ["manque"]}])

    poser(calculer, attente_max_s=0.02, horloge=lambda: horloge[0])
    assert (await http.get(URL)).status_code == 503
    horloge[0] += 60 * 60

    reponse = await http.get(URL)

    assert reponse.status_code == 503
    assert reponse.json()["perime"] is True and reponse.json()["echecs"][0]["question"] == "Q"
    await asyncio.sleep(0.4)
    # L'appel de la minute 5 : le nouveau rapport, sans second calcul.
    horloge[0] += 5 * 60
    assert (await http.get(URL)).status_code == 200
    assert len(appels) == 2


async def test_la_route_ne_fait_que_lire(http):
    assert (await http.post(URL)).status_code == 405


def test_le_controle_de_la_route_porte_toutes_les_questions_de_reference():
    calculer = route._controle._calculer

    assert calculer.func is controler_le_chat
    assert calculer.args == (AsyncSessionLocal,)
    # En production, le fournisseur d'embeddings est sonde a chaque calcul.
    assert calculer.keywords == {"sonde": sonder_le_fournisseur}
    assert controler_le_chat.__defaults__[0] is QUESTIONS_DE_REFERENCE
    assert route._controle.duree_de_vie_s == 30 * 60
    # cron-job.org coupe a 30 s ; le calcul, lui, est borne.
    assert route._controle.attente_max_s < 30
    assert route._controle.delai_max_s <= 180


def test_la_docstring_de_la_route_nomme_chaque_faiblesse_connue():
    """
    Le proprietaire branche cron-job.org d'apres cette docstring : elle dit
    quelles questions ne feront jamais rendre 503. Une faiblesse ajoutee ou
    retiree de la liste sans elle la rendrait fausse.
    """
    doc = " ".join(route.controle_du_chat.__doc__.split())
    nommees = [q for q in QUESTIONS_DE_REFERENCE if f"« {q.question} »" in doc]

    assert {q.question for q in nommees} == set(FAIBLESSES_CONNUES)
    assert "faiblesse résolue" in doc


# ==================== DE BOUT EN BOUT, SUR LA BASE DE TEST ====================


@pytest.fixture
def vecteur_de_la_question(monkeypatch):
    monkeypatch.setattr(search_service, "_embedding_service_instance", _VecteurDeLaQuestion())


@pytest.mark.parametrize("textes, code", [
    ((("LOI-1992-007",),), 200),
    ((("CODE-CIVIL",),), 503),
], ids=["les-textes-sont-lus", "un-texte-manque"])
async def test_de_bout_en_bout_sur_la_base(
    textes, code, db_session, category_ids, pg_engine, vecteur_de_la_question, poser, http
):
    await _corpus(db_session, category_ids, panne=False)
    fabrique = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
    poser(partial(controler_le_chat, fabrique, [_question(textes=textes, rappel=True)]))

    reponse = await http.get(URL)

    assert reponse.status_code == code
    rapport = reponse.json()
    assert rapport["questions"] == rapport["verifiees"] == 1
    assert rapport["rappel_hnsw"][0]["rappel_a_10"] == {"200": 1.0, "400": 1.0}
    if code == 503:
        assert rapport["raisons"] == ["1 question(s) garantie(s) en echec sur 1"]


@pytest.mark.parametrize("lecture_directe, code", [(True, 200), (False, 503)],
                         ids=["lu-en-tete", "lecture-directe-coupee"])
async def test_une_question_qui_nomme_un_article_de_bout_en_bout(
    lecture_directe, code, db_session, category_ids, pg_engine, vecteur_de_la_question,
    poser, http, monkeypatch,
):
    """
    « Que dit l'article 39 du Code du travail ? » sur le petit corpus, ou
    l'article 39 est aussi le plus proche voisin de la question. La lecture
    directe coupee, la recherche le met encore en tete : la route doit
    quand meme rendre 503, et dire pourquoi.
    """
    await _corpus(db_session, category_ids, panne=False)
    if not lecture_directe:
        async def rien(self, question):
            return []

        monkeypatch.setattr(RAGService, "_article_nomme", rien)
    article_39 = ("LOI-1992-007", "39")
    question = _question(question="Que dit l'article 39 du Code du travail ?",
                         articles_attendus=(article_39,), article_nomme=article_39)
    fabrique = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
    poser(partial(controler_le_chat, fabrique, [question]))

    reponse = await http.get(URL)

    assert reponse.status_code == code, reponse.json()
    rapport = reponse.json()
    if code == 200:
        assert rapport["questions_verifiees"][0]["article_nomme_en_tete"] is True
    else:
        echec = rapport["echecs"][0]
        assert echec["article_nomme"] == "LOI-1992-007 art. 39"
        assert echec["article_nomme_en_tete"] is False
        # Lu en tete, mais trouve par la recherche : la panne se voit quand meme.
        assert echec["rang_de_l_article"] == 1, echec["lus"]
        assert any(r.startswith("l'article nomme par la question (LOI-1992-007 art. 39) est lu "
                                "en tete par la recherche") for r in echec["raisons"]), echec["raisons"]


@pytest.mark.parametrize("textes, etat", [
    ((("CODE-CIVIL",),), "faiblesse connue"),
    ((("LOI-1992-007",),), "faiblesse résolue"),
], ids=["faiblesse-en-echec", "faiblesse-resolue"])
async def test_une_faiblesse_connue_ne_fait_jamais_rendre_503_de_bout_en_bout(
    textes, etat, db_session, category_ids, pg_engine, vecteur_de_la_question, poser, http
):
    """
    Revue du tour 5 : une question que la recuperation fait echouer a raison
    rendait la route rouge a chaque calcul, et l'alerte ne voyait plus rien
    de neuf. Une faiblesse connue est rapportee a part, avec ce que le modele
    lit, qu'elle echoue ou qu'elle tienne : 200 dans les deux cas.
    """
    await _corpus(db_session, category_ids, panne=False)
    garantie = _question(rappel=True)
    faiblesse = _question(question="Quelles sont les causes de divorce ?", textes=textes,
                          faiblesse="le Code civil jamais lu")
    fabrique = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
    poser(partial(controler_le_chat, fabrique, [garantie, faiblesse]))

    reponse = await http.get(URL)

    assert reponse.status_code == 200, reponse.json()
    rapport = reponse.json()
    assert rapport["statut"] == "ok" and rapport["raisons"] == [] and rapport["echecs"] == []
    assert (rapport["garanties"], rapport["en_echec"], rapport["faiblesses_connues"]) == (1, 0, 1)
    detail = rapport["faiblesses"][0]
    assert detail["question"] == faiblesse.question and detail["etat"] == etat
    assert detail["diagnostic_connu"] == "le Code civil jamais lu"
    if etat == "faiblesse connue":
        assert rapport["faiblesses_en_echec"] == 1 and rapport["faiblesses_resolues"] == []
        assert detail["raisons"][0].startswith("textes attendus absents du contexte lu")
        assert detail["lus_avant_la_reponse"] == detail["lus"] and detail["lus"]
    else:
        assert rapport["faiblesses_en_echec"] == 0
        assert rapport["faiblesses_resolues"] == [faiblesse.question]


@pytest.mark.parametrize("mots, code", [
    ("donne lieu à des dommages-intérêts", 200),
    ("ne peut être inférieur à trois (3) mois de salaire", 503),
], ids=["la-regle-est-lue", "l-article-sans-la-regle"])
async def test_l_article_qui_repond_doit_porter_la_regle_de_bout_en_bout(
    mots, code, db_session, category_ids, pg_engine, vecteur_de_la_question, poser, http
):
    """
    Revue du tour 4 : un article lu ne repond que s'il porte la regle
    demandee. Sur le petit corpus, l'article 39 est lu en tete ; s'il ne porte
    pas les mots de la regle, la route rend 503 et dit qu'il est lu sans elle.
    """
    await _corpus(db_session, category_ids, panne=False)
    article_39 = ("LOI-1992-007", "39")
    question = _question(articles_attendus=(article_39,), regles=(article_39 + (mots,),))
    fabrique = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
    poser(partial(controler_le_chat, fabrique, [question]))

    reponse = await http.get(URL)

    assert reponse.status_code == code, reponse.json()
    if code == 503:
        echec = reponse.json()["echecs"][0]
        assert echec["rang_de_l_article"] is None
        assert echec["raisons"] == [
            "aucun article qui repond n'est lu par le modele : LOI-1992-007 art. 39 "
            "(lu sans la regle demandee : LOI-1992-007 art. 39)"
        ]
