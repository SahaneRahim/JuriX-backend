"""
Les questions de reference sur la vraie base : le filet de l'incident du
29/10/2026. A LANCER AVANT CHAQUE ENVOI DU CODE, avec JURIX_EXIGER_CORPUS=1 :

    JURIX_EXIGER_CORPUS=1 .venv/bin/python -m pytest -q -rxX \
        tests/test_integration/test_questions_de_reference.py

(-rxX : le resume liste chaque faiblesse connue de la recuperation, avec ce
que le modele lit avant la reponse ; sans lui, elles ne seraient qu'un compte
de « xfailed », et un envoi les laisserait passer sans qu'on les lise.)

Pour chaque question de reference (app/services/questions_de_reference.py :
les suggestions du front, des questions de fond, des questions qui nomment un
article), la recuperation du chat, SANS generation
(app/services/controle_du_chat.py) :
- un article qui repond, ET QUI PORTE LA REGLE DEMANDEE (ses mots, relus
  dans la base), doit etre parmi les 4 premiers extraits que le modele
  lirait, sans aucun extrait d'un autre texte avant lui, hors ceux que la
  question tolere ; s'il renvoie a un autre article pour sa peine, celui-ci
  aussi ;
- une question qui nomme un article doit le faire lire en premier, par la
  lecture directe de l'article nomme ;
- un texte attendu doit etre parmi les dix premiers voisins du volet
  semantique ;
- pour cinq questions, l'index HNSW doit retrouver au moins 8 des 10 vrais
  plus proches voisins, a ef_search 200 et 400.
C'est le controle de production, GET /api/v1/controle/chat, rejoue sur la
base locale : la route doit rendre 200 aujourd'hui. Et des pannes rejouees
doivent la faire rendre 503 : celle de l'incident (index perdu), Voyage en
panne, la perte du morceau qui repond (35.1, 318, 33 : deux volets, ou le
plein texte seul), les poids de fusion, la lecture directe de l'article nomme
coupee, un article nomme ecarte par _sans_doute, RRF_K a 500, la recherche
exacte coupee. Quatre suggestions anglaises reformulees le 29/10/2026 sont au
front : tests/test_services/test_controle_du_chat.py verifie l'accord.

LES FAIBLESSES CONNUES (questions_de_reference.FAIBLESSES_CONNUES). Depuis le
30/10/2026, l'article attendu est celui qui porte la regle (la peine du vol a
l'art. 318, et non l'art. 320 qui la double) : la recuperation du jour fait
echouer quelques questions, A RAISON. La route les rapporte a part, sans 503
(app/services/controle_du_chat.py, docstring). Ici, chacune est marquee
xfail(strict=True), avec ce que le modele lit avant la reponse : elle doit
echouer, et par la reponse seule. Une autre raison (volet en panne, contexte
vide) reste un echec ordinaire ; une question qui tient de nouveau fait
echouer son test (XPASS strict) : la retirer de la liste, pour que sa
prochaine panne fasse 503 en production (la route, elle, dit seulement
« faiblesse résolue »). Ne jamais y ajouter une question pour faire passer
un envoi : c'est a la recuperation de la corriger.

DEUX FACONS :
- `pytest tests/test_integration/test_questions_de_reference.py` (et la
  suite ordinaire), SANS RESEAU : les vecteurs des questions sont ceux que
  Voyage a rendus, gardes dans
  tests/fixtures/questions_de_reference/voyage-4-large.npz. Le code de la
  recuperation, la base et son index sont ceux du jour ; seul l'appel a
  Voyage est rejoue. La suite interdit tout appel reel (tests/conftest.py) :
  c'est ce qui permet a ce filet de tourner a chaque `pytest`.
- `pytest -m voyage_live tests/test_integration/test_questions_de_reference.py` :
  les vecteurs du vrai Voyage, et la preuve que le fichier les suit encore ;
  ignore sans VOYAGE_API_KEY. Avec JURIX_REGENERER_VECTEURS=1, le fichier est
  recrit : a faire quand une question de reference change.

LA BASE : JURIX_CORPUS_DATABASE_URL, par defaut jurix_dev du conteneur
jurix-pg (memes articles et memes vecteurs que la production), et non la base
de test, vide. En LECTURE SEULE : chaque transaction l'est
(default_transaction_read_only), une ecriture y echouerait. Base injoignable,
sans le corpus vectorise par voyage-4-large, ou sans la table que lit la
recherche exacte (une base d'avant la migration 018_articles_vecteurs) : tests
ignores, avec la raison ; ou EN ECHEC avec JURIX_EXIGER_CORPUS=1, pour qu'un
filet absent ne passe pas pour un filet vert.
"""

import dataclasses
import os
from datetime import date
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest
import pytest_asyncio
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import literal, literal_column, make_url, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ClauseElement, Executable

from app.models.law import Article, Law
from app.schemas.search import ChunkResult
from app.services import controle_du_chat, search_service, statut_des_sources, textes_successifs
from app.services.controle_du_chat import (
    CODE_ECHEC,
    CODE_OK,
    DELAI_DU_CALCUL_S,
    EF_SEARCH_DU_RAPPEL,
    FAIBLESSE_CONNUE,
    FAIBLESSE_RESOLUE,
    RANG_MAX_DE_LA_REPONSE,
    SEUIL_RAPPEL,
    ControleEnCache,
    _enonce_des_voisins,
    _enonce_des_vrais_voisins,
    compter_les_vecteurs,
    controler_le_chat,
    porte_l_article,
    porte_la_regle,
    sonder_le_fournisseur,
    verifier_une_question,
)
from app.services.embedding_service import EmbeddingService, empreinte_voyage
from app.services.liens_entre_textes import Lien
from app.services.questions_de_reference import FAIBLESSES_CONNUES, QUESTIONS_DE_REFERENCE
from app.services.rag_service import RAGService
from app.services.search_service import SearchService

pytestmark = pytest.mark.integration

URL_DU_CORPUS = os.environ.get(
    "JURIX_CORPUS_DATABASE_URL",
    "postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_dev",
)
# Pour les messages : sans le mot de passe.
BASE_DU_CORPUS = make_url(URL_DU_CORPUS).render_as_string(hide_password=True)
MODELE = "voyage-4-large"
EMPREINTE = empreinte_voyage(MODELE)
VECTEURS = (
    Path(__file__).resolve().parents[1] / "fixtures" / "questions_de_reference" / f"{MODELE}.npz"
)
# Un vecteur de Voyage varie a peine d'un appel a l'autre ; au-dela, le
# fichier ne decrit plus ce que la production calcule. Mesure du 29/10/2026,
# vecteurs enregistres contre vivants : cosinus minimum 0,9993 sur les 44
# questions dans une passe, et sous 0,999 pour une question dans une autre ;
# identiques (1,0) entre appels rapproches. 0,999 echouait donc au hasard. Un
# autre modele, ou une autre dimension, donne des vecteurs sans rapport :
# 0,995 laisse sept fois l'ecart observe et les arrete encore.
COSINUS_MIN = 0.995
QUESTION_DE_L_INCIDENT = "Quelles indemnités en cas de licenciement abusif ?"
SA_JUMELLE_ANGLAISE = "What damages does the Labour Code provide for unfair dismissal?"

# Les faiblesses connues de la recuperation, avec leur diagnostic, sont dans
# app/services/questions_de_reference.py (FAIBLESSES_CONNUES) : une seule liste,
# que la route et ce test lisent. Les raisons qui disent que la reponse n'est
# pas lue, ou trop tard, sont celles d'une faiblesse ; toute autre raison est
# une panne.
RAISONS_DE_LA_REPONSE = (
    "un autre texte est lu avant l'extrait qui repond",
    "extrait qui repond au rang",
    "aucun article qui repond n'est lu",
    "l'extrait qui repond (",
)


class EchecDeLaRecuperation(Exception):
    """Une faiblesse connue (FAIBLESSES_CONNUES) echoue, et par la reponse seule."""


def _par_la_reponse_seule(raisons) -> bool:
    return bool(raisons) and all(r.startswith(RAISONS_DE_LA_REPONSE) for r in raisons)

# None tant que la base n'a pas ete sondee ; ensuite "" si elle convient, ou
# la raison de l'ignorer.
_etat_du_corpus = None


def _ids(item) -> str:
    return f"{item.langue}-{item.question[:48]}"


async def _sonder(moteur) -> str:
    """"" si la base porte le corpus vectorise par le modele des questions."""
    try:
        async with moteur.connect() as connexion:
            vectorises = (await connexion.execute(
                text("SELECT count(*) FROM articles WHERE embedding_model = :empreinte"),
                {"empreinte": EMPREINTE},
            )).scalar()
    except Exception as erreur:
        return f"base du corpus injoignable ({type(erreur).__name__}) : {BASE_DU_CORPUS}"
    if not vectorises:
        return f"aucun article vectorise par {EMPREINTE} dans {BASE_DU_CORPUS}"
    try:
        async with moteur.connect() as connexion:
            await connexion.execute(text("SELECT 1 FROM articles_vecteurs LIMIT 1"))
    except Exception as erreur:
        return (
            f"pas de table articles_vecteurs dans {BASE_DU_CORPUS} ({type(erreur).__name__}) : "
            "appliquer la migration 018_articles_vecteurs"
        )
    return ""


@pytest_asyncio.fixture
async def corpus():
    """Une fabrique de sessions EN LECTURE SEULE sur la base du corpus."""
    global _etat_du_corpus
    moteur = create_async_engine(
        URL_DU_CORPUS,
        connect_args={"server_settings": {"default_transaction_read_only": "on"}},
        pool_size=4,
        max_overflow=2,
    )
    try:
        if _etat_du_corpus is None:
            _etat_du_corpus = await _sonder(moteur)
        if _etat_du_corpus:
            # Avant un envoi, un filet qui ne tourne pas doit se voir.
            if os.environ.get("JURIX_EXIGER_CORPUS") == "1":
                pytest.fail(f"JURIX_EXIGER_CORPUS=1 : {_etat_du_corpus}")
            pytest.skip(_etat_du_corpus)
        yield async_sessionmaker(moteur, class_=AsyncSession, expire_on_commit=False)
    finally:
        await moteur.dispose()


class _VecteursEnregistres:
    """
    Le service d'embeddings du chat, rejoue : le vecteur que Voyage a rendu
    pour chaque question de reference. Une autre question leve : elle n'a
    pas de vecteur, et la passer en silence ferait tourner le chat en plein
    texte seul.
    """

    TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    TASK_QUERY = "RETRIEVAL_QUERY"
    EMBEDDING_DIM = EmbeddingService.EMBEDDING_DIM

    class _Fournisseur:
        label = "vecteurs enregistres de Voyage"
        fingerprint = EMPREINTE

    provider = _Fournisseur()

    def __init__(self, vecteurs: Dict[str, np.ndarray]):
        self._vecteurs = vecteurs

    async def generate_embedding_async(self, text, task_type=TASK_QUERY):
        assert task_type == self.TASK_QUERY, task_type
        if text not in self._vecteurs:
            raise AssertionError(f"question sans vecteur enregistre : {text!r}")
        return self._vecteurs[text]


def _lire_les_vecteurs() -> Dict[str, np.ndarray]:
    with np.load(VECTEURS, allow_pickle=False) as fichier:
        assert str(fichier["empreinte"]) == EMPREINTE
        return {
            str(question): np.asarray(vecteur, dtype=np.float32)
            for question, vecteur in zip(fichier["questions"], fichier["vecteurs"])
        }


@pytest.fixture
def vecteurs_enregistres(monkeypatch):
    if not VECTEURS.exists():
        pytest.skip(
            f"{VECTEURS} absent : pytest -m voyage_live {Path(__file__).name} "
            "avec JURIX_REGENERER_VECTEURS=1 le cree"
        )
    vecteurs = _lire_les_vecteurs()
    manquantes = [q.question for q in QUESTIONS_DE_REFERENCE if q.question not in vecteurs]
    assert not manquantes, (
        "questions de reference sans vecteur enregistre (pytest -m voyage_live avec "
        f"JURIX_REGENERER_VECTEURS=1) : {manquantes}"
    )
    monkeypatch.setattr(
        search_service, "_embedding_service_instance", _VecteursEnregistres(vecteurs)
    )


# ==================== SANS RESEAU ====================


async def test_le_corpus_porte_chaque_texte_et_chaque_article_attendus(corpus):
    """Une reference ou un numero mal recopie ferait echouer une question pour rien."""
    references = sorted({ref for q in QUESTIONS_DE_REFERENCE for ref in q.attendues()})
    async with corpus() as db:
        trouvees = dict((await db.execute(text(
            "SELECT l.reference, count(a.embedding) FROM laws l "
            "JOIN articles a ON a.law_id = l.id "
            "WHERE l.reference = ANY(:refs) AND l.status = 'published' GROUP BY l.reference"
        ), {"refs": references})).all())
        morceaux = [
            SimpleNamespace(reference=ref, number=numero, content=contenu)
            for ref, numero, contenu in (await db.execute(text(
                "SELECT l.reference, a.number, a.content FROM laws l "
                "JOIN articles a ON a.law_id = l.id "
                "WHERE l.reference = ANY(:refs) AND l.status = 'published'"
            ), {"refs": references})).all()
        ]
        await db.rollback()
    absentes = [ref for ref in references if not trouvees.get(ref)]
    assert not absentes, f"textes attendus absents, non publies ou sans vecteurs : {absentes}"

    articles = sorted({a for q in QUESTIONS_DE_REFERENCE for a in q.articles_attendus + q.renvoi})
    introuvables = [
        f"{ref} art. {numero}" for ref, numero in articles
        if not any(porte_l_article(m, ref, numero) for m in morceaux)
    ]
    assert not introuvables, f"articles attendus qu'aucun morceau ne porte : {introuvables}"

    # Les mots de chaque regle, dans un morceau qui porte son article : une
    # reingestion qui deplace la regle (l'en-tete dans un morceau, l'alinea
    # dans le suivant) ou en change l'OCR doit se voir ici, et non comme une
    # question qui echoue sans que rien n'ait change pour le modele.
    sans_leur_regle = [
        f"{ref} art. {numero} « {mots} » ({q.question})"
        for q in QUESTIONS_DE_REFERENCE for ref, numero, mots in q.regles
        if not any(porte_l_article(m, ref, numero) and porte_la_regle(m, mots) for m in morceaux)
    ]
    assert not sans_leur_regle, f"regles qu'aucun morceau de leur article ne porte : {sans_leur_regle}"

    # Le meme numero range plusieurs articles : « 134 » porte aussi les art.
    # 134-1 et 134-2 du Code penal, « 311 » l'art. 311-1 de l'AUSCGIE. La
    # regle n'en designe qu'un, celui qui repond.
    for reference, numero in (("PRC-4722", "134"), ("OHADA-AUSCGIE-2014", "311")):
        mots = next(q.regle_de(reference, numero) for q in QUESTIONS_DE_REFERENCE
                    if q.regle_de(reference, numero))
        portent = [m for m in morceaux if porte_l_article(m, reference, numero)]
        assert len(portent) > 1, (reference, numero, len(portent))
        assert len([m for m in portent if porte_la_regle(m, mots)]) == 1, (reference, numero)
    # Les art. 320 et 321 du Code penal ne font que doubler les peines de
    # l'art. 318 : sa peine n'est que dans son morceau.
    peine_du_vol = "emprisonnement de cinq (5) à dix (10) ans"
    assert [m.number for m in morceaux if m.reference == "PRC-4722"
            and porte_la_regle(m, peine_du_vol)] == ["318"]

    # Le decoupage que supposent les articles attendus (porte_l_article) :
    # 35.2 commence par l'alinea 6 de l'article 39 et ne porte rien de l'article
    # 35 ; 65.2 est l'alinea 2 de l'article 65, l'acces non autorise.
    morceau = {(m.reference, m.number): m for m in morceaux}
    suite_35 = morceau[("LOI-1992-007", "35.2")]
    assert suite_35.content.lstrip().startswith("(6) Le salaire"), suite_35.content[:80]
    assert not any(porte_l_article(suite_35, "LOI-1992-007", n) for n in ("35", "36", "37", "39"))
    assert "tout accès non autorisé" in morceau[("LOI-2010-012", "65.2")].content

    # Un morceau tolere qu'une reingestion renumerote ne serait plus reconnu :
    # sa question echouerait sans que rien n'ait change pour le modele.
    toleres = sorted({t for q in QUESTIONS_DE_REFERENCE for t in q.toleres_avant})
    async with corpus() as db:
        presents = {tuple(ligne) for ligne in (await db.execute(text(
            "SELECT l.reference, a.number FROM laws l JOIN articles a ON a.law_id = l.id "
            "WHERE l.status = 'published' AND l.reference || '|' || a.number = ANY(:cles)"
        ), {"cles": [f"{ref}|{numero}" for ref, numero in toleres]})).all()}
        await db.rollback()
    assert not [t for t in toleres if t not in presents], (toleres, presents)


def _questions_et_faiblesses_connues():
    for item in QUESTIONS_DE_REFERENCE:
        marques = () if item.garantie else (pytest.mark.xfail(
            strict=True, raises=EchecDeLaRecuperation,
            reason=f"faiblesse connue de la recuperation, a corriger chez elle : {item.faiblesse}",
        ),)
        yield pytest.param(item, id=_ids(item), marks=marques)


@pytest.mark.parametrize("item", list(_questions_et_faiblesses_connues()))
async def test_le_modele_lirait_un_article_qui_repond(item, corpus, vecteurs_enregistres):
    verdict = await verifier_une_question(corpus, item)

    message = (
        f"« {item.question} » ({item.langue}) : {' ; '.join(verdict.raisons)}\n"
        f"  attendus : {[list(g) for g in item.textes]}, {item.articles_attendus} "
        f"({item.articles})\n"
        f"  lus par le modele : {verdict.lus}\n"
        f"  rang du texte : {verdict.rang}, de l'article : {verdict.rang_article}, "
        f"intrus avant : {verdict.intrus}, rang semantique : {verdict.rang_semantique}, "
        f"rappel : {verdict.rappels}"
    )
    if not item.garantie and _par_la_reponse_seule(verdict.raisons):
        raise EchecDeLaRecuperation(message)
    assert verdict.ok, message
    if item.rappel:
        assert verdict.rappels is not None
        assert set(verdict.rappels) == set(EF_SEARCH_DU_RAPPEL)
        assert all(r >= SEUIL_RAPPEL for r in verdict.rappels.values())


async def _la_route(corpus, questions=QUESTIONS_DE_REFERENCE, sonde=None):
    """
    (code HTTP, rapport) de la route telle qu'en production : ControleEnCache
    autour de controler_le_chat (CONCURRENCE questions a la fois, borne
    DELAI_DU_CALCUL_S), sur la vraie base. L'attente est levee : la duree,
    qui depend de la charge de la machine, n'est pas jugee ici.
    """
    async def sonde_ok():
        return None

    controle = ControleEnCache(
        partial(
            controler_le_chat, corpus, questions, sonde=sonde or sonde_ok,
            comptes=compter_les_vecteurs,
        ),
        attente_max_s=DELAI_DU_CALCUL_S + 60,
    )
    return await controle.lire()


async def test_aujourd_hui_la_route_rend_200_et_rapporte_les_faiblesses(corpus, vecteurs_enregistres):
    """
    Le calcul de la route tel quel sur la vraie base : 200, chaque question
    verifiee dans la borne, aucune en erreur, pas de connexion qui manque, la
    recherche exacte du chat qui tient, et les faiblesses connues rapportees
    a part, avec ce que le modele lit avant la reponse. Revue du tour 5 : la
    route rendait 503 a chaque calcul, et l'alerte ne voyait plus rien de
    neuf. Les verdicts eux-memes sont ceux des tests par question, ci-dessus.
    La duree n'est pas jugee ici : elle depend de la charge de la machine
    (trois suites en parallele la doublent) ; la borne, elle, est jugee par
    les questions « non verifiee ».
    """
    code, rapport = await _la_route(corpus)

    raisons = [raison for echec in rapport["echecs"] + rapport["faiblesses"]
               for raison in echec["raisons"]]
    pannes = [r for r in raisons if r.startswith(
        ("non verifiee", "recuperation en echec", "mesure de l'index en echec",
         "volet semantique indisponible")
    )]
    assert rapport["questions"] == rapport["verifiees"] == len(QUESTIONS_DE_REFERENCE)
    assert not pannes, (pannes, f"{rapport['duree_ms']} ms, borne {DELAI_DU_CALCUL_S} s")
    assert (code, rapport["statut"], rapport["raisons"]) == (CODE_OK, "ok", []), (
        rapport["raisons"], [e["question"] for e in rapport["echecs"]])
    table = rapport["table_des_vecteurs"]
    assert table["articles_vectorises"] == table["lignes"] > 0, table
    garanties = [q for q in QUESTIONS_DE_REFERENCE if q.garantie]
    assert (rapport["garanties"], rapport["en_echec"]) == (len(garanties), 0)
    rappels = [r["rappel_a_10"] for r in rapport["rappel_hnsw"]]
    assert len(rappels) == 5
    assert all(r is not None and min(r.values()) >= SEUIL_RAPPEL for r in rappels), rappels

    # Chaque faiblesse connue, a part : son etat, son rang, ce qui est lu
    # avant la reponse. L'etat lui-meme est juge question par question
    # (xfail strict, ci-dessus) : ici, qu'il soit dit.
    faiblesses = {f["question"]: f for f in rapport["faiblesses"]}
    assert set(faiblesses) == set(FAIBLESSES_CONNUES)
    assert rapport["faiblesses_connues"] == len(FAIBLESSES_CONNUES)
    for question, faiblesse in faiblesses.items():
        assert faiblesse["etat"] in (FAIBLESSE_CONNUE, FAIBLESSE_RESOLUE), faiblesse
        assert faiblesse["diagnostic_connu"] == FAIBLESSES_CONNUES[question]
        rang = faiblesse["rang_de_l_article"]
        assert faiblesse["lus_avant_la_reponse"] == (
            faiblesse["lus"] if rang is None else faiblesse["lus"][:rang - 1]), faiblesse
    assert rapport["faiblesses_resolues"] == [
        q for q, f in faiblesses.items() if f["etat"] == FAIBLESSE_RESOLUE]
    assert rapport["faiblesses_en_echec"] == len(FAIBLESSES_CONNUES) - len(
        rapport["faiblesses_resolues"])


async def test_la_recherche_exacte_n_emprunte_pas_l_index(corpus):
    """
    Le rappel ne vaut que si l'une des deux listes est EXACTE : la verite ne
    doit pas toucher l'index HNSW, et la liste de l'index doit l'emprunter
    (sinon on comparerait deux recherches exactes).
    """
    vecteur = [1.0] + [0.0] * (EmbeddingService.EMBEDDING_DIM - 1)

    async def plan(enonce, reglage: str = "SELECT 1") -> str:
        async with corpus() as db:
            await db.execute(text(reglage))
            lignes = (await db.execute(_Expliquer(enonce))).scalars().all()
            await db.rollback()
        return "\n".join(lignes)

    verite = await plan(_enonce_des_vrais_voisins(vecteur, 10))
    assert "hnsw" not in verite and "articles_vecteurs" in verite, verite
    assert "hnsw" in (await plan(
        _enonce_des_voisins(vecteur, 160, 10), "SELECT set_config('hnsw.ef_search', '160', true)"
    ))


async def _tableaux_les_plus_proches(corpus, vecteur, n: int):
    """
    Les `n` morceaux de lois de finances les plus proches de `vecteur`, par
    recherche exacte : ce que rendait l'index de production le 29/10/2026.
    """
    distance = Article.embedding.cosine_distance(
        literal([float(x) for x in vecteur], HALFVEC(EmbeddingService.EMBEDDING_DIM))
    )
    enonce = (
        select(Article, Law, distance.label("distance"))
        .join(Law, Law.id == Article.law_id)
        .where(Article.embedding.isnot(None), Law.status == "published",
               Law.title.ilike("%portant loi de finances%"))
        .order_by(distance + literal_column("0"))
        .limit(n)
    )
    async with corpus() as db:
        # Lus AVANT le rollback, qui expire les objets de la session.
        tableaux = [
            ChunkResult(
                article_id=article.id, law_id=loi.id, number=article.number,
                content=article.content, excerpt=article.content[:400],
                reference=loi.reference, law_title=loi.title, type=loi.type or "loi",
                language=loi.language, status=loi.status, category_id=loi.category_id,
                publication_date=loi.publication_date,
                relevance_score=max(0.0, 1.0 - float(ecart)), source="semantic",
            )
            for article, loi, ecart in (await db.execute(enonce)).all()
        ]
        await db.rollback()
    return tableaux


async def test_le_controle_voit_la_panne_de_l_incident(corpus, vecteurs_enregistres, monkeypatch):
    """
    La panne du 29/10/2026 rejouee sur la vraie base, pour la question qui
    l'a revelee : le volet semantique du chat ne rend que des tableaux de lois
    de finances, et l'index HNSW aucun vrai voisin. La question doit echouer,
    quoi que la recuperation du jour fasse ensuite du plein texte : ce test
    prouve que le filet tient, pas seulement qu'il est vert.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE if q.question == QUESTION_DE_L_INCIDENT)
    service = search_service._embedding_service_instance
    vecteur = await service.generate_embedding_async(item.question, task_type=service.TASK_QUERY)
    tableaux = await _tableaux_les_plus_proches(corpus, vecteur, 32)
    assert {c.reference for c in tableaux}.isdisjoint(item.attendues())

    async def volet_des_tableaux(self, tache_vecteur, filters, limit, offset):
        await search_service._solder(tache_vecteur)
        return [c.model_copy(deep=True) for c in tableaux[offset:offset + limit]]

    async def index_des_tableaux(db, vecteur, k=10, ef_search=400):
        return [c.article_id for c in tableaux[:k]]

    monkeypatch.setattr(search_service.SearchService, "_chunks_semantiques", volet_des_tableaux)
    monkeypatch.setattr(controle_du_chat, "voisins_par_l_index", index_des_tableaux)
    verdict = await verifier_une_question(corpus, item)

    assert not verdict.ok
    assert "aucun texte attendu parmi les 10 premiers voisins du volet semantique" in (
        verdict.raisons
    )
    assert verdict.rappels == {ef: 0.0 for ef in EF_SEARCH_DU_RAPPEL}


@pytest.mark.parametrize("question", [QUESTION_DE_L_INCIDENT, SA_JUMELLE_ANGLAISE])
async def test_le_controle_voit_la_perte_du_morceau_qui_repond(
    question, corpus, vecteurs_enregistres, monkeypatch
):
    """
    La revue du 29/10/2026 : sans le morceau 35.1 du Code du travail (en-tetes
    des articles 36 a 39), retire des deux volets, la question restait verte
    au niveau du TEXTE (35.2 et 33 sont du meme code), et le modele lisait en
    tete l'indemnite du fonctionnaire licencie. Une reingestion peut deplacer
    ou abimer ce seul morceau : le controle doit le voir.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE if q.question == question)
    perdus = await _sans_le_morceau_35_1(corpus)
    semantique = SearchService._chunks_semantiques
    plein_texte = SearchService.text_chunks

    async def sans_le_morceau_semantique(self, *args, **kwargs):
        return [c for c in await semantique(self, *args, **kwargs) if c.article_id not in perdus]

    async def sans_le_morceau_plein_texte(self, *args, **kwargs):
        return [c for c in await plein_texte(self, *args, **kwargs) if c.article_id not in perdus]

    monkeypatch.setattr(SearchService, "_chunks_semantiques", sans_le_morceau_semantique)
    monkeypatch.setattr(SearchService, "text_chunks", sans_le_morceau_plein_texte)
    verdict = await verifier_une_question(corpus, item)

    assert not verdict.ok
    assert verdict.rang_article is None, verdict.lus
    assert "aucun article qui repond n'est lu par le modele : LOI-1992-007 art. 39" in (
        verdict.raisons
    )


async def _sans_le_morceau_35_1(corpus):
    """L'identifiant du morceau 35.1 du Code du travail (en-tetes des art. 36 a 39)."""
    return await _le_morceau(corpus, "LOI-1992-007", "35.1")


async def _le_morceau(corpus, reference: str, numero: str):
    """L'identifiant du seul morceau `numero` du texte `reference`."""
    async with corpus() as db:
        perdus = set((await db.execute(text(
            "SELECT a.id FROM articles a JOIN laws l ON l.id = a.law_id "
            "WHERE l.reference = :reference AND a.number = :numero"
        ), {"reference": reference, "numero": numero})).scalars().all())
        await db.rollback()
    assert len(perdus) == 1, (reference, numero, perdus)
    return perdus


@pytest.mark.parametrize("question, reference, numero", [
    ("Quelle est la peine prévue pour l’escroquerie ?", "PRC-4722", "318"),
    ("What is the notice period for dismissing an employee?", "LOI-1992-007", "33"),
], ids=["escroquerie-sans-318", "preavis-en-sans-33"])
async def test_le_controle_voit_la_perte_du_morceau_qui_porte_la_regle(
    question, reference, numero, corpus, vecteurs_enregistres, monkeypatch
):
    """
    Revue du tour 4 : le morceau 318 du Code penal retire des deux volets,
    « Quelle est la peine prévue pour l’escroquerie ? » restait verte par
    l'art. 321 (« Les peines de l'article 318 ci-dessus sont doublées »),
    sans aucune peine lisible. De meme, le morceau 33 (art. 34) perdu, le
    preavis en anglais restait vert par 35.1, qui ne dit aucune duree. Les
    deux questions tiennent aujourd'hui : leur echec ne vient que de la perte.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE if q.question == question)
    assert (await verifier_une_question(corpus, item)).ok
    perdus = await _le_morceau(corpus, reference, numero)
    semantique = SearchService._chunks_semantiques
    plein_texte = SearchService.text_chunks

    async def sans_le_morceau_semantique(self, *args, **kwargs):
        return [c for c in await semantique(self, *args, **kwargs) if c.article_id not in perdus]

    async def sans_le_morceau_plein_texte(self, *args, **kwargs):
        return [c for c in await plein_texte(self, *args, **kwargs) if c.article_id not in perdus]

    monkeypatch.setattr(SearchService, "_chunks_semantiques", sans_le_morceau_semantique)
    monkeypatch.setattr(SearchService, "text_chunks", sans_le_morceau_plein_texte)
    verdict = await verifier_une_question(corpus, item)

    assert not verdict.ok and verdict.rang_article is None, verdict.lus
    assert any(r.startswith(("aucun article qui repond n'est lu par le modele",
                             "textes attendus absents")) for r in verdict.raisons), verdict.raisons


@pytest.mark.parametrize("panne", ["plein-texte-sans-35-1", "poids-de-fusion-0-1-0-9"])
async def test_le_controle_voit_un_autre_regime_avant_l_article_qui_repond(
    panne, corpus, vecteurs_enregistres, monkeypatch
):
    """
    Revue du tour 3, deux pannes ou le volet semantique reste sain :
    - le plein texte seul perd le morceau 35.1 ;
    - les poids de fusion passent a 0,1 / 0,9 (reglage d'exploitation, sans
      redeploiement).
    Le modele lit alors d'abord l'indemnite du FONCTIONNAIRE licencie (Statut
    general de la fonction publique, art. 120), puis le Code du travail.
    L'ancienne regle (deux autres textes toleres) restait verte, la route
    aussi.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE if q.question == QUESTION_DE_L_INCIDENT)
    if panne == "plein-texte-sans-35-1":
        perdus = await _sans_le_morceau_35_1(corpus)
        plein_texte = SearchService.text_chunks

        async def sans_le_morceau(self, *args, **kwargs):
            return [c for c in await plein_texte(self, *args, **kwargs)
                    if c.article_id not in perdus]

        monkeypatch.setattr(SearchService, "text_chunks", sans_le_morceau)
    else:
        monkeypatch.setattr(SearchService, "TEXT_WEIGHT", 0.1)
        monkeypatch.setattr(SearchService, "SEMANTIC_WEIGHT", 0.9)

    verdict = await verifier_une_question(corpus, item)

    assert not verdict.ok
    assert verdict.rang_article is not None and verdict.intrus >= 1, verdict.lus
    assert "DEC-1994-199 art. 120" in verdict.autres_textes_avant, verdict.lus
    assert any(r.startswith("un autre texte est lu avant l'extrait qui repond")
               for r in verdict.raisons), verdict.raisons


async def test_sans_35_1_le_preavis_n_est_pas_repondu_par_35_2(
    corpus, vecteurs_enregistres, monkeypatch
):
    """
    Revue du tour 3 : « 35.2 » comptait pour l'article 35 a son seul numero,
    alors qu'il porte la fin de l'article 39 et l'article 40. Le plein texte
    prive de 35.1, « Quel est le délai de préavis en cas de licenciement ? »
    restait vert par lui (mesure du 29/10/2026 : 35.2 en tete, puis l'AUPCAP
    et le preavis d'un agent commercial, avant le morceau 33 qui porte
    l'article 34). 35.2 est lu, mais il ne repond pas.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE
                if q.question == "Quel est le délai de préavis en cas de licenciement ?")
    perdus = await _sans_le_morceau_35_1(corpus)
    plein_texte = SearchService.text_chunks

    async def sans_le_morceau(self, *args, **kwargs):
        return [c for c in await plein_texte(self, *args, **kwargs) if c.article_id not in perdus]

    monkeypatch.setattr(SearchService, "text_chunks", sans_le_morceau)
    verdict = await verifier_une_question(corpus, item)

    assert "LOI-1992-007 art. 35.2" in verdict.lus, verdict.lus
    repond = verdict.lus[verdict.rang_article - 1] if verdict.rang_article else None
    assert repond != "LOI-1992-007 art. 35.2", verdict.lus


async def test_le_controle_voit_la_lecture_directe_de_l_article_nomme_coupee(
    corpus, vecteurs_enregistres, monkeypatch
):
    """
    Revue du tour 3 : la lecture directe coupee (_article_nomme rend []), une
    seule des cinq questions qui nomment un article echouait ; l'article 76,
    celui de l'incident, restait vert au rang 5. Les cinq doivent echouer.
    """
    async def rien(self, question):
        return []

    monkeypatch.setattr(RAGService, "_article_nomme", rien)
    nommes = [q for q in QUESTIONS_DE_REFERENCE if q.article_nomme is not None]
    verdicts = [await verifier_une_question(corpus, item) for item in nommes]

    assert len(nommes) == 5
    for item, verdict in zip(nommes, verdicts):
        assert not verdict.ok and verdict.article_nomme_en_tete is False, (item.question,
                                                                          verdict.lus)
        assert any(r.startswith("l'article nomme par la question") for r in verdict.raisons)


async def test_le_controle_voit_un_article_nomme_ecarte_par_sans_doute(
    corpus, vecteurs_enregistres, monkeypatch
):
    """
    Revue du tour 3 : un texte plus recent du corpus qui « modifie » l'art.
    318 du Code penal (un lien de plus, simule). _sans_doute ecarte alors
    l'article, a raison, et le modele lit en tete l'article 318 du Code de
    procedure penale : le meme numero d'un autre texte, la forme de
    l'incident. La route rendait 200.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE
                if q.question == "What does Article 318 of the Penal Code say?")
    lien = Lien(
        texte="PRC-4722", texte_nom="la Loi N°2016/007 portant Code pénal", nature="modifie",
        par_titre="une loi simulée modifiant le Code pénal", par_date=date(2027, 1, 1),
        articles=("318",),
    )
    monkeypatch.setattr(statut_des_sources, "LIENS", statut_des_sources.LIENS + (lien,))
    monkeypatch.setattr(textes_successifs, "LIENS", textes_successifs.LIENS + (lien,))

    verdict = await verifier_une_question(corpus, item)

    assert not verdict.ok and verdict.article_nomme_en_tete is False, verdict.lus
    assert not verdict.lus[0].startswith("PRC-4722 art. 318"), verdict.lus
    assert any(r.startswith("l'article nomme par la question (PRC-4722 art. 318)")
               for r in verdict.raisons), verdict.raisons


async def test_le_controle_voit_l_article_qui_repond_noye_dans_son_propre_texte(
    corpus, vecteurs_enregistres, monkeypatch
):
    """
    RRF_K a 500, un reglage d'exploitation qui aplatit la fusion : pour « Quelle
    est la durée maximale de la garde à vue ? », le modele lit la detention
    provisoire (art. 218) et la garde du MINEUR (art. 702) avant les articles
    119 et 120. Tout vient du Code de procedure penale : aucun autre texte,
    seul le rang de l'article qui repond le voit (mesure du 29/10/2026 : rang
    6). L'ancienne regle restait verte.
    """
    item = next(q for q in QUESTIONS_DE_REFERENCE
                if q.question == "Quelle est la durée maximale de la garde à vue ?")
    monkeypatch.setattr(SearchService, "RRF_K", 500)

    verdict = await verifier_une_question(corpus, item)

    assert not verdict.ok, verdict.lus
    assert verdict.rang_article > RANG_MAX_DE_LA_REPONSE and verdict.intrus == 0, verdict.lus
    assert verdict.raisons == [
        f"extrait qui repond au rang {verdict.rang_article}, au-dela des "
        f"{RANG_MAX_DE_LA_REPONSE} premiers extraits lus"
    ]


async def test_le_controle_voit_la_recherche_exacte_coupee(
    corpus, vecteurs_enregistres, monkeypatch
):
    """
    Une base trop lente : chaque recherche exacte du chat depasse son budget,
    et le chat lit l'index HNSW (repli, ef_search 1000). Les questions
    peuvent tenir, l'index de ce jour etant sain, mais la protection contre
    l'incident est coupee : le controle doit echouer, et dire pourquoi.
    """
    monkeypatch.setattr(SearchService, "BUDGET_RECHERCHE_EXACTE_MS", 1)
    questions = [q for q in QUESTIONS_DE_REFERENCE if not q.rappel][:4]

    code, rapport = await _la_route(corpus, questions)

    assert code == CODE_ECHEC and rapport["statut"] == "echec"
    assert rapport["chemins_semantiques"].get("repli_hnsw", 0) > len(questions) / 2, (
        rapport["chemins_semantiques"]
    )
    assert any(r.startswith("recherche exacte du chat en repli sur l'index HNSW")
               for r in rapport["raisons"]), rapport["raisons"]


# ==================== LA ROUTE SOUS LES PANNES REJOUEES ====================

# Les pannes rejouees par les revues des tours 3 a 5, et les questions
# GARANTIES que chacune fait echouer. Mesure du 30/10/2026 : la route sur les
# 51 questions, sous chaque panne (data/fiabilite/tmp/controle6/ et controle7/,
# route_sous_les_pannes.py). Seules ces questions sont rejouees ici, avec une
# faiblesse connue : la route doit rendre 503, et la faiblesse rester a part.
# Revue du tour 5 : ces pannes rendaient le meme 503 qu'un jour sans panne.
_RAPPEL = tuple(q.question for q in QUESTIONS_DE_REFERENCE if q.rappel)
_NOMMEES = tuple(q.question for q in QUESTIONS_DE_REFERENCE if q.article_nomme is not None)
_ACCES_EN_FRANCAIS = "Quelles peines pour l’accès non autorisé à un système d’information ?"
PANNES_REJOUEES = {
    # L'incident du 29/10/2026 : seules les questions `rappel` mesurent l'index.
    "index-perdu": _RAPPEL,
    "plein-texte-sans-35-1": (QUESTION_DE_L_INCIDENT,),
    "deux-volets-sans-35-1": (QUESTION_DE_L_INCIDENT, SA_JUMELLE_ANGLAISE),
    "poids-0-1-0-9": (QUESTION_DE_L_INCIDENT, "What is the maximum length of police custody?"),
    "poids-0-9-0-1": ("Quels sont mes droits en garde à vue ?", _ACCES_EN_FRANCAIS),
    "rrf-k-500": ("Quelle est la durée maximale de la garde à vue ?", _ACCES_EN_FRANCAIS),
    "lecture-directe-coupee": _NOMMEES,
    "art-318-ecarte": ("What does Article 318 of the Penal Code say?",),
    "deux-volets-sans-318": ("Quelle est la peine prévue pour l’escroquerie ?",
                             "Que risque l’auteur d’une escroquerie sur internet ?"),
    "deux-volets-sans-33": ("What is the notice period for dismissing an employee?",),
}
UNE_FAIBLESSE = "Quelle peine encourt l’auteur d’un vol ?"
_MORCEAUX_PERDUS = {
    "35-1": ("LOI-1992-007", "35.1"), "318": ("PRC-4722", "318"), "33": ("LOI-1992-007", "33"),
}


async def _rejouer(panne: str, corpus, monkeypatch) -> None:
    """Pose la panne `panne` le temps du test, comme les revues l'ont rejouee."""
    if panne == "index-perdu":
        async def index_perdu(db, vecteur, k=10, ef_search=400):
            # L'index rend des articles, mais aucun des vrais voisins.
            return [-1 - i for i in range(k)]

        monkeypatch.setattr(controle_du_chat, "voisins_par_l_index", index_perdu)
    elif panne.startswith("poids-"):
        texte, semantique = (0.1, 0.9) if panne == "poids-0-1-0-9" else (0.9, 0.1)
        monkeypatch.setattr(SearchService, "TEXT_WEIGHT", texte)
        monkeypatch.setattr(SearchService, "SEMANTIC_WEIGHT", semantique)
    elif panne == "rrf-k-500":
        monkeypatch.setattr(SearchService, "RRF_K", 500)
    elif panne == "lecture-directe-coupee":
        async def rien(self, question):
            return []

        monkeypatch.setattr(RAGService, "_article_nomme", rien)
    elif panne == "art-318-ecarte":
        lien = Lien(
            texte="PRC-4722", texte_nom="la Loi N°2016/007 portant Code pénal",
            nature="modifie", par_titre="une loi simulée modifiant le Code pénal",
            par_date=date(2027, 1, 1), articles=("318",),
        )
        monkeypatch.setattr(statut_des_sources, "LIENS", statut_des_sources.LIENS + (lien,))
        monkeypatch.setattr(textes_successifs, "LIENS", textes_successifs.LIENS + (lien,))
    else:
        volets, morceau = panne.split("-sans-")
        perdus = await _le_morceau(corpus, *_MORCEAUX_PERDUS[morceau])
        plein_texte = SearchService.text_chunks
        semantique = SearchService._chunks_semantiques

        async def plein_texte_sans(self, *args, **kwargs):
            return [c for c in await plein_texte(self, *args, **kwargs)
                    if c.article_id not in perdus]

        async def semantique_sans(self, *args, **kwargs):
            return [c for c in await semantique(self, *args, **kwargs)
                    if c.article_id not in perdus]

        monkeypatch.setattr(SearchService, "text_chunks", plein_texte_sans)
        if volets == "deux-volets":
            monkeypatch.setattr(SearchService, "_chunks_semantiques", semantique_sans)


@pytest.mark.parametrize("panne", list(PANNES_REJOUEES))
async def test_la_route_rend_503_sous_chaque_panne_rejouee(
    panne, corpus, vecteurs_enregistres, monkeypatch
):
    questions = [q for q in QUESTIONS_DE_REFERENCE
                 if q.question in PANNES_REJOUEES[panne] or q.question == UNE_FAIBLESSE]
    assert len(questions) == len(PANNES_REJOUEES[panne]) + 1
    # Des questions garanties : une faiblesse ne ferait pas rendre 503.
    assert all(q.garantie for q in questions if q.question != UNE_FAIBLESSE)
    await _rejouer(panne, corpus, monkeypatch)

    code, rapport = await _la_route(corpus, questions)

    attendues = set(PANNES_REJOUEES[panne])
    en_echec = {e["question"] for e in rapport["echecs"]}
    assert (code, rapport["statut"]) == (CODE_ECHEC, "echec"), rapport["raisons"]
    # Au moins une question garantie, et seulement celles que la panne touche.
    # Mesure du 30/10/2026 : toutes ; une recuperation meilleure peut en
    # sauver une, la route doit rester rouge par les autres.
    assert en_echec and en_echec <= attendues, [(e["question"], e["raisons"])
                                                for e in rapport["echecs"]]
    assert rapport["raisons"] == [
        f"{len(en_echec)} question(s) garantie(s) en echec sur {len(attendues)}"]
    # La faiblesse est lue avec elles, et reste a part.
    assert [f["question"] for f in rapport["faiblesses"]] == [UNE_FAIBLESSE]
    if panne in ("index-perdu", "lecture-directe-coupee"):
        # Par construction, chacune de ces questions voit la panne.
        assert en_echec == attendues
    if panne == "index-perdu":
        for echec in rapport["echecs"]:
            assert echec["rappel_a_10"] == {str(ef): 0.0 for ef in EF_SEARCH_DU_RAPPEL}, echec


class _VoyageInjoignable:
    """Le fournisseur d'embeddings en panne : chaque appel leve, comme Voyage injoignable."""

    label = "Voyage injoignable (panne rejouee)"
    fingerprint = EMPREINTE
    native_dim = 2048
    retryable = False

    def embed(self, texts, task_type, dim):
        raise ConnectionError("Voyage injoignable")


async def test_voyage_en_panne_la_route_rend_503(corpus, vecteurs_enregistres, monkeypatch):
    """
    La vraie sonde du fournisseur, comme en production : les vecteurs des
    questions sont en memoire des le deuxieme calcul, elle seule voit Voyage
    tomber. Revue du tour 5 : ce 503 etait celui de tous les jours.
    """
    service = search_service._embedding_service_instance
    monkeypatch.setattr(service, "provider", _VoyageInjoignable())

    code, rapport = await _la_route(corpus, sonde=partial(sonder_le_fournisseur, pause_s=0))

    assert code == CODE_ECHEC and rapport["verifiees"] == 0
    assert rapport["raisons"] == [
        "fournisseur d'embeddings en echec, 2 essais (EmbeddingServiceError) : questions non "
        "verifiees"]


async def test_une_faiblesse_qui_tient_est_dite_resolue_sans_503(corpus, vecteurs_enregistres):
    """
    Une question qui tient aujourd'hui, marquee faiblesse : la route rend 200
    et dit « faiblesse résolue ». A cote d'elle, une question garantie : des
    faiblesses seules ne garantiraient rien.
    """
    escroquerie, legitime_defense = (
        next(q for q in QUESTIONS_DE_REFERENCE if q.question == question) for question in (
            "Quelle est la peine prévue pour l’escroquerie ?",
            "Quand peut-on invoquer la légitime défense ?"))
    resolue = dataclasses.replace(escroquerie, faiblesse="l'art. 318 jamais lu (simule)")

    code, rapport = await _la_route(corpus, [resolue, legitime_defense])

    assert (code, rapport["statut"], rapport["raisons"]) == (CODE_OK, "ok", [])
    assert rapport["faiblesses_resolues"] == [resolue.question]
    assert rapport["faiblesses"][0]["etat"] == FAIBLESSE_RESOLUE
    assert (rapport["garanties"], rapport["en_echec"], rapport["faiblesses_en_echec"]) == (1, 0, 0)


class _Expliquer(Executable, ClauseElement):
    """EXPLAIN d'un enonce SQLAlchemy, parametres lies compris (recette de la doc)."""

    inherit_cache = False

    def __init__(self, enonce):
        self.enonce = enonce


@compiles(_Expliquer)
def _compiler_l_explication(element, compiler, **kw):
    return "EXPLAIN " + compiler.process(element.enonce, **kw)


# ==================== LE VRAI VOYAGE ====================


@pytest.fixture
def voyage(monkeypatch):
    """
    Le vrai service d'embeddings (cles du .env, garde levee par le marqueur).

    Delai d'une question porte a 20 s : ce test juge les vecteurs et ce que
    le chat en fait, pas la latence de Voyage, que surveille la sonde du
    controle de production. Le 29/10/2026 vers 17 h, Voyage repondait en 6,5
    a 7 s (mesure) : au delai de 3 s, le chat lisait le plein texte seul, et
    ce test echouait sans rien dire des vecteurs.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "VOYAGE_TIMEOUT_QUESTION_S", 20.0)
    service = EmbeddingService(use_cache=True)
    monkeypatch.setattr(search_service, "_embedding_service_instance", service)
    yield service
    # Une connexion TLS laissee ouverte leverait un ResourceWarning.
    service.provider.fermer()


@pytest.mark.voyage_live
async def test_avec_voyage_le_modele_lirait_un_article_qui_repond(corpus, voyage):
    """
    Chaque question par le vrai Voyage (une requete chacune, vecteurs ensuite
    gardes en memoire), puis le fichier des vecteurs compare a eux, ou recrit
    avec JURIX_REGENERER_VECTEURS=1.
    """
    echecs = []
    vivants = {}
    for item in QUESTIONS_DE_REFERENCE:
        verdict = await verifier_une_question(corpus, item)
        # Une faiblesse connue (FAIBLESSES_CONNUES) n'echoue que par la
        # reponse ; toute autre raison est une panne.
        connu = not item.garantie and _par_la_reponse_seule(verdict.raisons)
        if not verdict.ok and not connu:
            echecs.append(
                f"« {item.question} » : {' ; '.join(verdict.raisons)} ; lus {verdict.lus}"
            )
        vivants[item.question] = await voyage.generate_embedding_async(
            item.question, task_type=voyage.TASK_QUERY
        )

    if os.environ.get("JURIX_REGENERER_VECTEURS") == "1":
        VECTEURS.parent.mkdir(parents=True, exist_ok=True)
        questions = [q.question for q in QUESTIONS_DE_REFERENCE]
        # float16 : la base compare en halfvec, la question y est convertie.
        np.savez_compressed(
            VECTEURS,
            questions=np.array(questions),
            vecteurs=np.stack([vivants[q] for q in questions]).astype(np.float16),
            empreinte=np.array(EMPREINTE),
        )
    elif VECTEURS.exists():
        enregistres = _lire_les_vecteurs()
        ecarts = [
            q for q, v in vivants.items()
            if q not in enregistres
            or float(v @ enregistres[q] / (np.linalg.norm(v) * np.linalg.norm(enregistres[q])))
            < COSINUS_MIN
        ]
        assert not ecarts, (
            f"vecteurs enregistres perimes (JURIX_REGENERER_VECTEURS=1 les recrit) : {ecarts}"
        )

    assert not echecs, "\n".join(echecs)
