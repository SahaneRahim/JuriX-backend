"""
Le contexte du chat apres la recherche : article nomme, textes confirmes quand
le volet semantique est tres faible, et la BARRIERE avant le modele
(RAGService._retenir_le_pertinent).

Incident du 29/10/2026, « Quelles indemnités en cas de licenciement abusif ? »
(une question d'accueil du chat) : l'index HNSW de production n'a rendu que des
tableaux de lois de finances (similarite ~0,34), le plein texte trouvait le
Code du travail, et le modele a recu huit tableaux. Il a repondu que les
documents ne disaient rien, puis cite de memoire un « article 76 du Code du
travail ».

Ces tests disent :
- la langue de la QUESTION, et l'article qu'elle nomme avec son texte : un
  texte de la TABLE des noms, ou un numero propre, jamais un intitule qui ne
  fait que contenir les mots du nom ; rien dans le doute (numero ambigu,
  article qu'un texte plus recent modifie) ;
- ce que devient le contexte, volets doubles : inchange quand le volet
  semantique passe le seuil (bas), article nomme AJOUTE sans rien evincer ;
  reduit aux textes confirmes sous ce seuil ; vide — donc sans appel au
  modele — quand ni la similarite ni le plein texte ne trouvent rien ;
- une RELANCE (« Peux-tu détailler ? », « Et l'article 38 ? ») est cherchee
  avec le SUJET de la conversation (la derniere question de fond, pas le
  dernier message), et un article nomme sans texte, ou avec un renvoi au
  texte de la conversation (« l'article 40 du même code »), est lu dans le
  texte du tour precedent, s'il est sans ambiguite (un renvoi qui a ses
  mots, « de cette loi sur la nationalité », seulement si l'article les
  porte ; un renvoi qui porte une nature, « de ce décret », seulement si le
  texte est de cette nature, sinon rien, et le mot de nature reste dans les
  mots-cles) ; une relance ouverte qui apporte ses mots (« Et la
  nationalité ? ») est cherchee seule ET jointe, les deux listes alternees,
  et refusee seule elle ne garde de la recherche jointe que ce qui porte ses
  mots (de meme la seconde chance d'une question refusee seule) ; la
  relance sans mot propre qui la suit relit la fin du sujet, seule, alternee
  avec le sujet entier ; une question qui a son sujet est cherchee seule ;
- servi par l'index (repli), le volet semantique n'est pas cru seul : seuls
  les textes confirmes sont lus (le rejeu de l'incident ne lit plus de lois
  de finances) ;
- ce que voient ask et ask_stream d'un refus : le message de la barriere, ni
  modele, ni repli sur la Constitution ;
- aucune source fabriquee : `sources` et `Message.sources` ne portent que les
  articles cites et rattaches ; ce que le modele a lu reste dans « Textes
  consultés » ;
- et, sur la base de test, par la vraie recherche (exacte) : l'incident, la
  question hors sujet refusee, l'article lu par son en-tete ou par le numero
  de sa loi, les doutes, les tableaux de programmes retires, le cache.
"""

import json
import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schemas.rag import RAGRequest
from app.schemas.search import ChunkResult
from app.services import rag_service as module
from app.services.embedding_service import EmbeddingService
from app.services.intent_classifier import IntentResult
from app.services.prompts import message_sans_texte_pertinent
from app.services.rag_service import (
    SOURCE_ARTICLE_NOMME,
    TEXTES_NOMMABLES,
    RAGService,
    de_la_nature,
    est_une_relance,
    forme_du_numero,
    langue_de_la_question,
    mots_propres,
    nature_du_renvoi,
    numero_de_l_intitule,
    numero_du_nom,
    numero_sans_texte,
    peut_changer_de_sujet,
    porte_un_mot_propre,
    reference_d_article,
    renvoi_au_texte,
    sujet_de_la_conversation,
    texte_de_la_table,
    textes_de_la_table_dans,
)
from app.services.search_service import (
    CHEMIN_ECHEC,
    CHEMIN_EXACT,
    CHEMIN_REPLI_HNSW,
    VoletsDeLaRecherche,
    ecritures_en_attente,
)

QUESTION = "Quelles indemnités en cas de licenciement abusif ?"
MOTS_CLES = "indemnités cas licenciement abusif"
SEUIL_FR = RAGService.SEUILS_DE_SIMILARITE["fr"]
SEUIL_EN = RAGService.SEUILS_DE_SIMILARITE["en"]
# Sous les deux seuils : un volet semantique TRES faible.
TRES_FAIBLE = round(min(SEUIL_FR, SEUIL_EN) - 0.05, 3)

TITRE_CODE_DU_TRAVAIL = "Loi N°92/007 du 14 août 1992 portant Code du travail"
TITRE_LOI_DE_FINANCES = (
    "Loi N°2022/020 du 27 décembre 2022 portant loi de finances de la République "
    "du Cameroun pour l'exercice 2023"
)


def _extrait(article_id, law_id, numero, reference, titre, source="semantic",
             contenu="Contenu.", pertinence=0.5):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        excerpt=contenu[:400], reference=reference, law_title=titre,
        relevance_score=pertinence, source=source, publication_date=date(2022, 12, 27),
    )


def _tableaux(n=8):
    return [
        _extrait(76000 + i, 2232, f"76.{i}", "PRC-9449", TITRE_LOI_DE_FINANCES)
        for i in range(1, n + 1)
    ]


def _code_du_travail(source="fts"):
    return [
        _extrait(97420, 2274, "35.1", "LOI-1992-007", TITRE_CODE_DU_TRAVAIL, source,
                 "Article 37.- (1) En cas de rupture du contrat du fait de l'employeur, "
                 "le travailleur a droit à une indemnité de licenciement.", 1.0),
        _extrait(97421, 2274, "35.2", "LOI-1992-007", TITRE_CODE_DU_TRAVAIL, source,
                 "(7) Ces dommages-intérêts ne se confondent ni avec l'indemnité de préavis, "
                 "ni avec l'indemnité de licenciement.", 0.8),
    ]


def _volets(similarites, plein_texte=(), chemin=CHEMIN_EXACT):
    return VoletsDeLaRecherche(
        chemin=chemin, similarites=dict(similarites), plein_texte=list(plein_texte)
    )


@pytest.fixture
def chat():
    """Un RAGService sans base ni modele ; recherche, lecture stricte et article nomme doubles."""
    service = RAGService(AsyncMock())
    service.llm = None
    # Un SAVEPOINT sans base : un gestionnaire de contexte qui ne fait rien.
    service.db.begin_nested = MagicMock(return_value=MagicMock())
    service.search_service = MagicMock()
    service._articles_stricts = AsyncMock(return_value=set())
    service._article_nomme = AsyncMock(return_value=[])
    return service


def _recherche(chat, extraits, volets):
    chat.search_service.search = AsyncMock(return_value=MagicMock(
        chunks=list(extraits), search_time_ms=3,
    ))
    chat.search_service.volets = volets


# ==================== LA LANGUE DE LA QUESTION ====================


class TestLangueDeLaQuestion:

    @pytest.mark.parametrize("question", [
        QUESTION,
        "Que risque l’auteur d’une escroquerie sur internet ?",
        "Qu’est-ce qu’un groupement d’intérêt économique ?",
        "art. 37 du Code du travail",
        "MINDDEVEL contrôle finances communes",
    ])
    def test_francaise(self, question):
        assert langue_de_la_question(question) == "fr"

    @pytest.mark.parametrize("question", [
        "What compensation is due for unfair dismissal?",
        "What are the employer's obligations when hiring under the Labour Code?",
        "How is the President of the Republic elected?",
    ])
    def test_anglaise(self, question):
        assert langue_de_la_question(question) == "en"

    @pytest.mark.parametrize("question", [
        "ratification rallonge 2019",
        "salary composition mayor deputy mayors",
        # Des noms propres accentues ne font pas une question francaise.
        "new police commissioner 11th Yaoundé 2021",
        "Who wrote Les Misérables?",
    ])
    def test_incertaine_jamais_francaise_a_tort(self, question):
        """Une question lue francaise a tort prendrait le seuil le plus haut."""
        assert langue_de_la_question(question) in (None, "en")


# ==================== L'ARTICLE NOMME : QUEL TEXTE ? ====================


class TestReferenceDArticle:

    @pytest.mark.parametrize("question, attendu", [
        ("Que dit l'article 39 du Code du travail ?", ("39", ["Code du travail"])),
        ("Que dit l'art. 37 du Code du travail ?", ("37", ["Code du travail"])),
        ("art. 37 du Code du travail", ("37", ["Code du travail"])),
        ("Que dit l'article 39 du Code du travail sur le licenciement ?",
         ("39", ["Code du travail sur le licenciement", "Code du travail"])),
        ("l'article 74 du Code pénal et l'article 75",
         ("74", ["Code pénal et l'article 75", "Code pénal"])),
        ("Que prévoit l'article premier de la Constitution ?", ("1", ["Constitution"])),
        ("What does Article 39 of the Labour Code say?",
         ("39", ["the Labour Code say", "the Labour Code"])),
        ("Que dit l'article 35 de la loi n° 92/007 ?", ("35", ["loi n° 92/007"])),
    ])
    def test_un_article_et_son_texte(self, question, attendu):
        assert reference_d_article(question) == attendu

    @pytest.mark.parametrize("question", [QUESTION, "Que dit l'article 12 ?"])
    def test_rien_a_lire(self, question):
        assert reference_d_article(question) is None

    @pytest.mark.parametrize("question, numero", [
        ("Que dit l'article 38 de ce code ?", "38"),
        ("Et l'article 38 de cette loi ?", "38"),
        ("Et l'article 40 de ce texte ?", "40"),
        ("Que prévoit l'article 40 du même code ?", "40"),
        ("Que dit l'article 38 dudit code ?", "38"),
        ("Que dit l'article 38 de ladite loi ?", "38"),
        ("Que dit l'article 38 de la loi ?", "38"),
        ("Que dit l'article 38 du code ?", "38"),
        ("Que dit l'article 12 de la présente loi ?", "12"),
        ("Que dit l'article 38 de ce code sur le préavis ?", "38"),
        ("What does Article 40 of this law say?", "40"),
        ("And Article 40 of the same code?", "40"),
        ("What does Article 40 of the law say?", "40"),
        # Le nom s'arrete au premier verbe (tour 5) : sans cela, « the code
        # provide on notice » passait pour un nom, et la question etait
        # cherchee seule.
        ("What does Article 40 of the code provide on notice?", "40"),
        ("What does Article 40 of the law say about notice?", "40"),
        ("Est-ce que l'article 40 de ce code dit que le préavis est dû ?", "40"),
        ("Article 5 of the said Act", "5"),
    ])
    def test_un_renvoi_au_texte_de_la_conversation_n_est_pas_un_texte(self, question, numero):
        """
        Revue du tour 4 : « ce code » passait pour un nom, que la table ne
        resout pas ; la question etait cherchee seule, et le modele lisait
        l'article 38 de huit textes sans rapport. C'est l'article d'un texte
        deja nomme : aucun texte ici, le numero seul, comme « Et l'article 38 ? ».
        """
        assert reference_d_article(question) is None
        assert renvoi_au_texte(question)
        assert numero_sans_texte(question) == numero

    @pytest.mark.parametrize("question", [
        # Un article defini suivi d'une suite nomme un texte : la table, ou rien.
        "Que dit l'article 2 de la loi sur le terrorisme ?",
        "What does Article 2 of the law on terrorism say?",
        "Que dit l'article 2 de la loi de 2024 ?",
        "Que dit l'article 30 de la loi de finances ?",
        "Que dit l'article 39 du Code du travail ?",
        "Que dit l'article 1 de la loi n° 2013/004 ?",
        "Que dit l'article 6 de la Constitution ?",
        # Un verbe clot le nom, qui reste un nom.
        "What does Article 39 of the Labour Code say about notice?",
        "What does Article 2 of the law on terrorism provide?",
        # Sans texte du tout : ni nom, ni renvoi.
        "Et l'article 38 ?",
        QUESTION,
    ])
    def test_un_nom_de_texte_n_est_pas_un_renvoi(self, question):
        assert not renvoi_au_texte(question)


TITRE_DECRET = (
    "Décret N°2024/056 du 21 février 2024 portant revalorisation du montant des allocations "
    "familiales servies aux travailleurs"
)


class TestNatureDUnRenvoi:
    """
    Revue du tour 5 : « l'article 2 de ce décret » heritait du Code du
    travail, quelle que soit sa nature. Un renvoi qui porte une nature n'herite
    que d'un texte de cette nature, lue dans son intitule.
    """

    @pytest.mark.parametrize("question, nature", [
        ("Que dit l'article 2 de ce décret ?", "decret"),
        ("Et l'article 1er dudit décret ?", "decret"),
        ("Que dit l'article 3 du décret ?", "decret"),
        ("What does Article 2 of that decree say?", "decret"),
        ("Et l'article 38 de cette loi ?", "loi"),
        ("Que dit l'article 38 de la loi ?", "loi"),
        ("What does Article 40 of this law say?", "loi"),
        ("What does Article 38 of the law say about notice?", "loi"),
        ("Que dit l'article 38 de ce code ?", "code"),
        ("Que prévoit l'article 40 du même code ?", "code"),
        ("Que dit l'article 38 dudit code ?", "code"),
        ("And Article 40 of the same code?", "code"),
        ("Et l'article 5 de cette ordonnance ?", "ordonnance"),
        ("Et l'article 5 desdites ordonnances ?", "ordonnance"),
        # Sans nature : tout texte.
        ("Et l'article 40 de ce texte ?", None),
        ("Article 5 of the said Act", None),
        # Sans renvoi.
        ("Et l'article 38 ?", None),
        ("Que dit l'article 39 du Code du travail ?", None),
        (QUESTION, None),
    ])
    def test_la_nature_du_renvoi(self, question, nature):
        assert nature_du_renvoi(question) == nature

    @pytest.mark.parametrize("intitule, natures", [
        # Une loi ET un code : « ce code » et « cette loi » le designent.
        (TITRE_CODE_DU_TRAVAIL, {"loi", "code"}),
        (TITRE_DECRET, {"decret"}),
        ("Décret N°2018/366 du 20 juin 2018 portant Code des Marchés Publics", {"decret", "code"}),
        # Le code que cite son objet n'en fait pas un code.
        ("Loi N°2026/003 du 14 avril 2026 modifiant et complétant certaines dispositions de la "
         "loi N°2012/001 du 19 avril 2012 portant code électoral", {"loi"}),
        ("Code civil applicable au Cameroun, annoté de la jurisprudence de la Cour suprême",
         {"code"}),
        ("Acte uniforme relatif au droit de l'arbitrage du 23 novembre 2017", set()),
        ("Ordonnance N°2025/001 du 15 mai 2025 portant incitations à l'investissement",
         {"ordonnance"}),
        (None, set()),
    ])
    def test_la_nature_d_un_intitule(self, intitule, natures):
        assert {n for n in ("loi", "code", "decret", "ordonnance") if de_la_nature(intitule, n)} \
            == natures


class TestTableDesNoms:
    """
    Un nom d'usage designe un texte de la table, mot pour mot ; tout autre
    nom, rien. Revue du 29/10/2026 : « l'intitule contient tous les mots »
    menait 13 noms courants sur 55 a un autre texte.
    """

    @pytest.mark.parametrize("nom, reference", [
        ("Code du travail", "LOI-1992-007"),
        ("code de travail", "LOI-1992-007"),
        ("the Labour Code", "LOI-1992-007"),
        ("Code du travail camerounais", "LOI-1992-007"),
        ("Code du travail de 1992", "LOI-1992-007"),
        ("Code pénal", "PRC-4722"),
        ("le nouveau Code pénal", "PRC-4722"),
        ("Penal Code", "PRC-4722"),
        ("Code de procédure pénale", "LOI-2005-007"),
        ("CPP", "LOI-2005-007"),
        ("Code of Criminal Procedure", "LOI-2005-007"),
        ("Code civil", "CODE-CIVIL"),
        ("Constitution", "PRC-6283"),
        ("Constitution de la République du Cameroun", "PRC-6283"),
        ("Constitution de 1972", "PRC-6283"),
        ("Code électoral", "LOI-2012-001"),
        ("code minier", "PRC-9866"),
        ("Acte uniforme relatif au droit des sociétés", "OHADA-AUSCGIE-2014"),
        ("Acte uniforme OHADA portant sur le droit commercial général", "OHADA-AUDCG-2010"),
        ("Acte uniforme relatif au droit commercial général", "OHADA-AUDCG-2010"),
        ("Acte uniforme relatif au droit des sociétés coopératives", "OHADA-AUSCOOP-2010"),
        ("traité OHADA", "OHADA-TRAITE-1993"),
    ])
    def test_un_nom_de_la_table(self, nom, reference):
        assert texte_de_la_table(nom).reference == reference

    @pytest.mark.parametrize("nom", [
        # Les noms de la revue qui menaient a un autre texte.
        "loi sur le terrorisme", "loi sur la cybercriminalité", "loi sur la décentralisation",
        "loi sur les associations", "loi foncière", "loi de finances", "décret de 2018",
        "loi de 2024",
        # Absents du corpus, ou d'un autre pays, ou d'une autre annee.
        "Code général des impôts", "Code du travail français", "code minier de 2016",
        "Constitution française",
        # Generiques.
        "Acte uniforme", "la loi", "traité", "Code", "décret",
    ])
    def test_un_autre_nom_ne_designe_rien(self, nom):
        assert texte_de_la_table(nom) is None

    def test_aucun_nom_en_double(self):
        """Deux textes sous un meme nom rendraient la table ambigue (verifie a l'import)."""
        assert len({t.reference for t in TEXTES_NOMMABLES}) == len(TEXTES_NOMMABLES)
        assert all(t.noms and t.intitule for t in TEXTES_NOMMABLES)


class TestNumeroPropre:

    @pytest.mark.parametrize("numero, forme", [
        ("92/007", (1992, 7)), ("1992-07", (1992, 7)), ("2016/007", (2016, 7)),
        ("96/06", (1996, 6)), ("67-LF-7", (1967, "lf", 7)), ("0342/CAB/PR", (342, "cab", "pr")),
    ])
    def test_forme_comparable(self, numero, forme):
        assert forme_du_numero(numero) == forme

    @pytest.mark.parametrize("nom, attendu", [
        ("loi n° 92/007", ("loi", (1992, 7))),
        ("Loi N°2013/004", ("loi", (2013, 4))),
        ("la loi 2016-007", ("loi", (2016, 7))),
        ("décret n° 2018/366", ("decret", (2018, 366))),
        ("Law No. 2005/007", ("loi", (2005, 7))),
        ("Code du travail", None),
        ("loi de 2024", None),
    ])
    def test_le_numero_d_un_nom(self, nom, attendu):
        assert numero_du_nom(nom) == attendu

    def test_le_numero_propre_et_non_celui_que_l_intitule_cite(self):
        """« Loi N° 2017/015 … modifiant la loi N° 2013/004 » est la loi 2017/015."""
        assert numero_de_l_intitule(
            "Loi N° 2017/015 du 12 juillet 2017 modifiant et complétant certaines dispositions "
            "de la loi N° 2013/004 du 18 avril 2013"
        ) == ("loi", (2017, 15))
        assert numero_de_l_intitule(TITRE_CODE_DU_TRAVAIL) == ("loi", (1992, 7))
        assert numero_de_l_intitule("Code civil applicable au Cameroun") is None


# ==================== LE CONTEXTE, VOLETS DOUBLES ====================


class TestVoletSain:

    @pytest.mark.asyncio
    async def test_le_contexte_de_la_recherche_passe_tel_quel(self, chat, caplog):
        hybride = _code_du_travail("semantic") + _tableaux(6)
        _recherche(chat, hybride, _volets({97420: 0.54, 97421: 0.51}, _tableaux(2)))

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert contexte == hybride
        etat = chat._recuperation
        assert (etat.mode, etat.refus, etat.ajouts, etat.langue) == ("sain", False, [], "fr")
        assert etat.similarite == 0.54
        # Aucune lecture de plus sur une question qui marche.
        chat._articles_stricts.assert_not_called()
        assert any(
            f"🧭 Contexte du chat : volet semantique exact, similarite 0.540 >= {SEUIL_FR} (fr) : "
            "8 extraits de la recherche" == r.getMessage() for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_l_article_nomme_s_ajoute_sans_rien_evincer(self, chat):
        """
        Revue du 29/10/2026 : l'article nomme passait en tete et chassait le
        huitieme extrait (le bon article d'une loi, quand le nom menait a une
        autre). Il s'ajoute : les huit extraits restent, dans leur ordre.
        """
        nomme = _code_du_travail()[0].model_copy(update={"source": SOURCE_ARTICLE_NOMME})
        chat._article_nomme = AsyncMock(return_value=[nomme])
        hybride = _tableaux(8)
        _recherche(chat, hybride, _volets({76001: 0.45}))

        contexte = await chat._retrieve_chunks("Que dit l'article 37 du Code du travail ?", "fr")

        assert contexte == [nomme] + hybride
        assert chat._recuperation.ajouts == ["LOI-1992-007 art. 35.1"]

    @pytest.mark.asyncio
    async def test_une_question_incertaine_prend_le_seuil_anglais(self, chat):
        """Des mots-cles nus : le seuil le plus bas, jamais un refus a tort."""
        _recherche(chat, _tableaux(4), _volets({76001: SEUIL_EN}))

        contexte = await chat._retrieve_chunks("ratification rallonge 2019", "fr")

        assert len(contexte) == 4
        assert (chat._recuperation.langue, chat._recuperation.seuil) == (None, SEUIL_EN)

    @pytest.mark.asyncio
    async def test_la_similarite_jugee_est_celle_du_contexte(self, chat):
        """Un voisin proche que la fusion a laisse hors du contexte ne compte pas."""
        _recherche(chat, _tableaux(3), _volets({76001: TRES_FAIBLE, 99999: 0.80}))

        assert await chat._retrieve_chunks(QUESTION, "fr") == []
        assert chat._recuperation.similarite == TRES_FAIBLE

    @pytest.mark.asyncio
    async def test_la_question_budgetaire_garde_les_tableaux(self, chat):
        """Le drapeau de la recherche suit la question : tableaux retires, sauf sur le budget."""
        _recherche(chat, _tableaux(2), _volets({76001: 0.6}))

        await chat._retrieve_chunks(QUESTION, "fr")
        assert chat.search_service.tableaux_de_programmes_exclus is True
        assert chat._recuperation.budgetaire is False

        await chat._retrieve_chunks("Quel est le budget du ministère de la santé en 2023 ?", "fr")
        assert chat.search_service.tableaux_de_programmes_exclus is False
        assert chat._recuperation.budgetaire is True


class TestVoletTresFaible:

    @pytest.mark.asyncio
    async def test_la_forme_de_l_incident(self, chat, caplog):
        """
        Des extraits tres loin de la question, le Code du travail au plein
        texte, qui porte tous les mots-cles : lui seul est lu.
        """
        chat._articles_stricts = AsyncMock(return_value={97420})
        _recherche(chat, _tableaux(), _volets(
            {c.article_id: TRES_FAIBLE for c in _tableaux()}, _code_du_travail()
        ))

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert [(c.reference, c.number) for c in contexte] == [("LOI-1992-007", "35.1")]
        etat = chat._recuperation
        assert (etat.mode, etat.refus, etat.ajouts) == ("faible", False, ["LOI-1992-007 art. 35.1"])
        mots, extraits = chat._articles_stricts.call_args[0]
        assert mots == MOTS_CLES
        assert {c.article_id for c in extraits} >= {97420, 97421, 76001}
        assert any(
            r.getMessage().startswith(
                f"📌 Volet semantique tres faible (exact, similarite {TRES_FAIBLE:.3f} < "
                f"{SEUIL_FR}, fr) : 1 texte(s) confirme(s) sur 8 : LOI-1992-007 art. 35.1"
            ) for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_une_question_anglaise_ne_recoit_pas_le_plein_texte(self, chat):
        """
        Le plein texte d'une question anglaise ne retrouve pas l'article d'un
        corpus francais : ses premiers articles ne sont pas essayes. Les
        extraits deja confirmes du contexte restent.
        """
        chat._articles_stricts = AsyncMock(return_value={76002})
        _recherche(chat, _tableaux(4), _volets({76001: 0.05}, _code_du_travail()))

        contexte = await chat._retrieve_chunks(
            "What compensation is due for unfair dismissal?", "fr"
        )

        assert [c.article_id for c in contexte] == [76002]
        _, extraits = chat._articles_stricts.call_args[0]
        assert 97420 not in {c.article_id for c in extraits}

    @pytest.mark.asyncio
    async def test_une_question_incertaine_essaie_le_plein_texte(self, chat):
        """Des mots-cles nus, souvent francais : les premiers du plein texte sont essayes."""
        chat._articles_stricts = AsyncMock(return_value={97420})
        _recherche(chat, _tableaux(4), _volets({76001: 0.05}, _code_du_travail()))

        contexte = await chat._retrieve_chunks("licenciement abusif indemnite", "fr")

        assert chat._recuperation.langue is None
        assert [c.article_id for c in contexte] == [97420]
        _, extraits = chat._articles_stricts.call_args[0]
        assert {97420, 97421} <= {c.article_id for c in extraits}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", ["Et dans mon cas ?", "Et la prime ?"])
    async def test_un_seul_mot_cle_ne_confirme_rien(self, chat, question):
        """
        Revue du 29/10/2026 : « Et dans mon cas ? » se reduisait a « cas »,
        present dans 5 204 articles, et le modele lisait un decret d'adhesion
        aux conventions sur l'apatridie. La question brute ne confirme rien.
        """
        _recherche(chat, _tableaux(3), _volets({76001: TRES_FAIBLE}, _code_du_travail()))

        assert await chat._retrieve_chunks(question, "fr") == []
        mots, _ = chat._articles_stricts.call_args[0]
        assert mots == ""

    @pytest.mark.asyncio
    async def test_un_article_nomme_garde_son_texte_seul(self, chat):
        """« article code travail » est partout : seuls les extraits du texte nomme restent."""
        nomme = _code_du_travail()[0].model_copy(update={"source": SOURCE_ARTICLE_NOMME})
        chat._article_nomme = AsyncMock(return_value=[nomme])
        autre_loi = _extrait(5000, 77, "3", "LOI-1967-LF-7", "Code des prestations familiales")
        chat._articles_stricts = AsyncMock(return_value={97421, 5000})
        _recherche(chat, [autre_loi] + _tableaux(3),
                   _volets({76001: TRES_FAIBLE}, _code_du_travail()))

        contexte = await chat._retrieve_chunks("Que dit l'article 39 du Code du travail ?", "fr")

        assert [c.number for c in contexte] == ["35.1", "35.2"]
        assert contexte[0].source == SOURCE_ARTICLE_NOMME

    @pytest.mark.asyncio
    async def test_sans_article_nomme_les_textes_qui_modifient_restent(self, chat):
        """
        Revue du 29/10/2026 : en mode faible, l'article nomme d'un texte revise
        ne laissait entrer que ce texte, et la loi qui le modifie sortait. Un
        article modifie n'est plus nomme (_sans_doute) : la loi modificative
        confirmee par le plein texte reste.
        """
        constitution = _extrait(6, 2143, "6", "PRC-6283", "Loi N°96/06 portant révision de la "
                                "Constitution")
        revision = _extrait(7, 1761, "SECTION_2", "PRC-10679", "Loi N°2026/002 modifiant la "
                            "Constitution")
        chat._articles_stricts = AsyncMock(return_value={6, 7})
        _recherche(chat, [constitution, revision], _volets({6: TRES_FAIBLE, 7: TRES_FAIBLE}))

        contexte = await chat._retrieve_chunks("Que dit l'article 6 de la Constitution ?", "fr")

        assert [c.reference for c in contexte] == ["PRC-6283", "PRC-10679"]

    @pytest.mark.asyncio
    async def test_la_decision_sous_le_seuil_se_rejoue_hors_du_chemin(self, chat):
        """
        La mesure (scripts/eval/evaluer_contexte_chat.py) rejoue sur une
        question saine ce que la barriere deciderait sous le seuil : meme
        methode, sur le contexte juge que garde l'etat.
        """
        chat._articles_stricts = AsyncMock(return_value={97420})
        volets = _volets({76001: 0.6}, _code_du_travail())
        _recherche(chat, _tableaux(3), volets)

        contexte = await chat._retrieve_chunks(QUESTION, "fr")
        etat = chat._recuperation
        assert etat.mode == "sain" and contexte == etat.juge and etat.nommes == []
        chat._articles_stricts.assert_not_called()

        confirmes = await chat._textes_confirmes(MOTS_CLES, etat.juge, etat.nommes, volets)

        assert [c.article_id for c in confirmes] == [97420]

    def test_le_seuil_suit_la_langue_de_la_question(self):
        service = RAGService.__new__(RAGService)
        seuils = RAGService.SEUILS_DE_SIMILARITE

        assert service.seuil_de_similarite("fr") == seuils["fr"]
        # Anglaise ou incertaine : le seuil le plus bas, jamais un refus a tort.
        assert service.seuil_de_similarite("en") == service.seuil_de_similarite(None) == seuils["en"]
        assert seuils["en"] < seuils["fr"]

    @pytest.mark.asyncio
    async def test_en_panne_le_plein_texte_confirme_suffit(self, chat):
        """Volet semantique en echec : aucune similarite, le plein texte strict decide."""
        chat._articles_stricts = AsyncMock(return_value={97420, 97421})
        _recherche(chat, _code_du_travail(), _volets({}, _code_du_travail(), CHEMIN_ECHEC))

        contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert [c.number for c in contexte] == ["35.1", "35.2"]
        assert chat._recuperation.similarite is None


class TestRepliSurLIndex:
    """
    Repli sur l'index (la recherche exacte a depasse son budget) : l'index
    n'est pas cru seul, quelle que soit la similarite de ses voisins. Seuls
    les textes confirmes (plein texte strict, article nomme) sont lus ; sinon
    rien, avec le message « réessayez ».
    """

    @pytest.mark.asyncio
    async def test_des_voisins_tres_loin_ne_sont_pas_lus(self, chat):
        chat._articles_stricts = AsyncMock(return_value={97420})
        _recherche(chat, _tableaux(), _volets(
            {c.article_id: TRES_FAIBLE for c in _tableaux()}, _code_du_travail(),
            chemin=CHEMIN_REPLI_HNSW,
        ))

        contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert [c.number for c in contexte] == ["35.1"]
        assert (chat._recuperation.chemin, chat._recuperation.mode) == (CHEMIN_REPLI_HNSW, "faible")

    @pytest.mark.asyncio
    async def test_au_dessus_du_seuil_l_index_n_est_pas_cru_seul(self, chat, caplog):
        """
        Le rejeu de l'incident (revue du 29/10/2026, tour 3) : un index faux
        rend des extraits de lois de finances (baremes d'amendes, marches
        publics) a 0,337, au-dessus du seuil de 0,30. Sur le repli, seul le
        Code du travail, que le plein texte confirme, est lu.
        """
        leurre = [
            _extrait(98750 + i, 2300, f"35.{i}", "PRC-9875", TITRE_LOI_DE_FINANCES,
                     contenu="| N° | Libellé | Montant à payer |\n|---|---|---|\n| 1 | Faute | 50 000 |")
            for i in range(1, 9)
        ]
        chat._articles_stricts = AsyncMock(return_value={97420})
        _recherche(chat, leurre, _volets(
            {c.article_id: 0.337 for c in leurre}, _code_du_travail(), chemin=CHEMIN_REPLI_HNSW,
        ))

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert [(c.reference, c.number) for c in contexte] == [("LOI-1992-007", "35.1")]
        etat = chat._recuperation
        assert (etat.mode, etat.refus, etat.similarite) == ("faible", False, 0.337)
        assert etat.similarite >= SEUIL_FR
        assert any(r.getMessage().startswith(
            "📌 Volet semantique servi par l'index (repli_hnsw, similarite 0.337, fr, non "
            "verifiable) : 1 texte(s) confirme(s) sur 8 : LOI-1992-007 art. 35.1"
        ) for r in caplog.records)

    @pytest.mark.asyncio
    async def test_l_anglaise_de_l_incident_est_refusee(self, chat):
        """« What compensation is due for unfair dismissal? » : 0,238 > 0,22, rien de confirme."""
        _recherche(chat, _tableaux(4), _volets(
            {76001: 0.238}, _code_du_travail(), chemin=CHEMIN_REPLI_HNSW,
        ))

        assert await chat._retrieve_chunks("What compensation is due for unfair dismissal?", "en") == []
        assert chat._recuperation.refus is True

    @pytest.mark.asyncio
    async def test_l_article_nomme_reste_sur_le_repli(self, chat):
        nomme = _code_du_travail()[0].model_copy(update={"source": SOURCE_ARTICLE_NOMME})
        chat._article_nomme = AsyncMock(return_value=[nomme])
        _recherche(chat, _tableaux(3), _volets({76001: 0.9}, chemin=CHEMIN_REPLI_HNSW))

        contexte = await chat._retrieve_chunks("Que dit l'article 37 du Code du travail ?", "fr")

        assert contexte == [nomme]


class TestBarriere:

    @pytest.mark.asyncio
    async def test_rien_de_confirme_rien_pour_le_modele(self, chat, caplog):
        _recherche(chat, _tableaux(), _volets({c.article_id: TRES_FAIBLE for c in _tableaux()}))

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            contexte = await chat._retrieve_chunks("Quelle est la recette du ndolé ?", "fr")

        assert contexte == []
        assert chat._recuperation.refus is True
        alertes = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert alertes == [
            "🚧 Barriere du chat : aucun texte de la recherche pertinent, contexte ecarte "
            f"(volet semantique exact, similarite {TRES_FAIBLE:.3f} < {SEUIL_FR}, fr ; aucun "
            "article avec tous les mots-cles « recette ndolé » ; aucun article nomme)"
        ]

    @pytest.mark.asyncio
    async def test_sans_volets_rien_a_juger(self, chat):
        """Une recherche doublee sans volets (les tests anciens) : le contexte passe."""
        chat.search_service.search = AsyncMock(return_value=MagicMock(
            chunks=_tableaux(3), search_time_ms=1,
        ))

        contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert len(contexte) == 3
        assert chat._recuperation.mode == "sans volets"

    @pytest.mark.asyncio
    async def test_une_lecture_stricte_en_echec_ne_confirme_rien(self, caplog):
        service = RAGService(AsyncMock())
        service.db.begin_nested = MagicMock(side_effect=RuntimeError("connexion perdue"))

        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            stricts = await service._articles_stricts(MOTS_CLES, _code_du_travail())

        assert stricts == set()
        assert any("Plein texte strict indisponible" in r.getMessage() for r in caplog.records)


def test_l_article_nomme_garde_la_tete_du_prompt():
    """
    _pin_requested_article remontait le premier extrait de meme numero, d'un
    autre texte : l'article 37 d'une loi de finances devant le Code du travail.
    """
    service = RAGService.__new__(RAGService)
    nomme = _code_du_travail()[0].model_copy(update={"source": SOURCE_ARTICLE_NOMME})
    autre = _extrait(1, 2232, "37", "PRC-9449", TITRE_LOI_DE_FINANCES)

    assert service._pin_requested_article(
        "Que dit l'article 37 du Code du travail ?", [nomme, autre]
    ) == [nomme, autre]


def test_une_question_qui_nomme_un_texte_ne_remonte_pas_l_article_d_un_autre():
    """
    « l'article 6 de la Constitution », article reecrit depuis, donc non
    ajoute : le premier extrait numerote 6 etait l'article 6 de la loi sur le
    Conseil constitutionnel, remonte en tete comme la reponse. L'ordre de la
    recherche reste. Un numero seul (« Que dit l'article 42 ? ») est toujours
    remonte (test_rag_context.TestArticlePinning).
    """
    service = RAGService.__new__(RAGService)
    constitution = _extrait(1, 2143, "26", "PRC-6283", "Loi N°96/06 portant révision de la Constitution")
    conseil = _extrait(2, 2257, "6", "LOI-2004-004", "Loi N°2004/004 portant organisation et "
                       "fonctionnement du Conseil constitutionnel")

    assert service._pin_requested_article(
        "Que dit l'article 6 de la Constitution ?", [constitution, conseil]
    ) == [constitution, conseil]
    assert service._pin_requested_article("Que dit l'article 6 ?", [constitution, conseil]) == [
        conseil, constitution,
    ]


# ==================== LE DOUTE SUR L'ARTICLE NOMME ====================


def _ligne(numero, ordre, lecture=0, reference="PRC-4722", contenu="Texte.", section=None):
    return SimpleNamespace(
        article_id=ordre, law_id=1, number=numero, article_title=None, section=section,
        page_number=None, content=contenu, ordre=ordre, reference=reference,
        law_title="Loi", type="loi", language="fr", status="published", category_id=None,
        publication_date=None, lecture=lecture,
    )


class TestSansDoute:

    def test_deux_lignes_qui_se_suivent_sont_un_seul_article(self):
        """« 74 » et « 74-1 » du Code penal sont ranges sous le meme numero, a la suite."""
        service = RAGService.__new__(RAGService)

        articles = service._sans_doute("74", "PRC-4722", [_ligne("74", 92), _ligne("74", 93)])

        assert [c.article_id for c in articles] == [92, 93]
        assert all(c.source == SOURCE_ARTICLE_NOMME for c in articles)

    def test_un_numero_porte_par_deux_articles_eloignes_ne_designe_rien(self):
        """La compilation du Code civil reprend « 3 » dans trois titres."""
        service = RAGService.__new__(RAGService)

        assert service._sans_doute("3", "CODE-CIVIL", [
            _ligne("3", 2, reference="CODE-CIVIL"), _ligne("3", 37, reference="CODE-CIVIL"),
        ]) == []

    def test_un_morceau_en_double_ne_designe_rien(self):
        service = RAGService.__new__(RAGService)

        assert service._sans_doute("328", "CIMA-CODE-2019", [
            _ligne("328.1", 10, lecture=1), _ligne("328.1", 11, lecture=1),
        ]) == []

    def test_un_article_qu_un_texte_plus_recent_modifie_n_est_pas_ajoute(self, caplog):
        """
        La Constitution de 1996 dit encore « renouvelable une fois » a son
        article 6, reecrit depuis (liens_entre_textes) : ni en tete du
        contexte, ni comme la reponse.
        """
        service = RAGService.__new__(RAGService)

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            assert service._sans_doute("6", "PRC-6283", [_ligne("6", 6, reference="PRC-6283")]) == []

        assert any("modifie par PRC-10679, rien d'ajoute" in r.getMessage() for r in caplog.records)

    def test_un_article_seulement_complete_reste(self):
        """L'article 67 de la Constitution n'est que complete (un alinea ajoute en 2008)."""
        service = RAGService.__new__(RAGService)

        assert len(service._sans_doute("67", "PRC-6283", [_ligne("67", 67, reference="PRC-6283")])) == 1

    def test_au_plus_quatre_morceaux(self):
        service = RAGService.__new__(RAGService)

        lignes = [_ligne(f"55.{i}", 50 + i, lecture=1) for i in range(1, 7)]

        assert len(service._sans_doute("55", "PRC-4722", lignes)) == module.MORCEAUX_DE_L_ARTICLE_NOMME


# ==================== LES RELANCES ====================


def _historique(*questions, reponse="Le Code du travail prévoit...", sources=None):
    questions = questions or (QUESTION,)
    messages = []
    for question in questions:
        messages += [SimpleNamespace(role="user", content=question),
                     SimpleNamespace(role="assistant", content=reponse, sources=sources)]
    return messages


def _recherches(chat, similarite=0.5, extraits=None):
    """La recherche doublee, qui garde chaque requete (semantique, mots-cles)."""
    appels = []
    hybride = list(extraits or _code_du_travail("semantic"))

    async def chercher(requete):
        appels.append((requete.semantic_query, requete.query))
        chat.search_service.volets = _volets({c.article_id: similarite for c in hybride})
        return MagicMock(chunks=list(hybride), search_time_ms=1)

    chat.search_service.search = chercher
    return appels


def _recherches_successives(chat, *reponses):
    """
    La recherche doublee, une reponse par appel et dans l'ordre : (extraits,
    similarite de chacun). Garde chaque requete (semantique, mots-cles).
    """
    appels = []
    restantes = list(reponses)

    async def chercher(requete):
        appels.append((requete.semantic_query, requete.query))
        extraits, similarite = restantes.pop(0)
        chat.search_service.volets = _volets({c.article_id: similarite for c in extraits})
        return MagicMock(chunks=list(extraits), search_time_ms=1)

    chat.search_service.search = chercher
    return appels


TITRE_NATIONALITE = "Loi N°1968-LF-3 du 11 juin 1968 portant Code de la nationalité camerounaise"


def _nationalite():
    return [
        _extrait(68000 + i, 1968, str(i), "LOI-1968-LF-3", TITRE_NATIONALITE,
                 contenu="La nationalité camerounaise s'acquiert par le mariage.")
        for i in (7, 8)
    ]


# Un extrait sur la dot (une note de jurisprudence du Code civil), et un autre
# sans rapport : ce que trouve « la dot » seule, sous le seuil.
DOT = _extrait(13822, 1804, "1382.2", "CODE-CIVIL", "Code civil applicable au Cameroun",
               contenu="Rupture des fiançailles : refus de restituer la dot, remboursement.")
SANS_RAPPORT = _extrait(5003, 77, "3", "PRC-77", "Décret sans rapport", contenu="Sans rapport.")

# La conversation de la revue du tour 5 (sonde_renvoi_nature) : la question
# nomme le Code du travail, la reponse parle d'un decret.
ALLOCATIONS = "Que dit le Code du travail sur les allocations familiales ?"
REPONSE_DECRET = (
    "Le Code du travail renvoie au régime de prévoyance sociale. Le décret n° 2024/056 du "
    "21 février 2024 a revalorisé le montant des allocations familiales servies aux travailleurs."
)


class TestEstUneRelance:
    """
    Une relance continue la conversation sans nommer son sujet ; une question
    qui a son sujet n'en est pas une. Sur les jeux de mesure (tour 5) : 0
    question de fond sur 535 prise pour une relance, 79 relances sur 80
    reconnues (relances et chaines), dont les 10 articles nommes avec un
    renvoi au texte ; 15 changements de sujet sur 16 cherches seuls et
    joints, le seizieme (quatre mots propres) cherche seul
    (scripts/eval/evaluer_contexte_chat.py).
    """

    @pytest.mark.parametrize("question", [
        "Peux-tu détailler ?", "Explique plus simplement s'il te plaît",
        "Donne-moi un exemple concret", "Et l'article 38 ?", "Quel article exactement ?",
        "Peux-tu citer l'article ?", "Combien ça coûte ?", "Et la prime ?", "Et le montant ?",
        "Et pour un CDD ?", "Et si l'employeur refuse de payer ?", "Que dit l'article 12 ?",
        "Peut-on faire appel de cette décision ?", "Est-ce que c'est toujours en vigueur ?",
        "Quelles sont les sanctions prévues ?", "Que dois-je faire maintenant ?",
        "Can you explain more?", "What should I do now?", "Give me an example",
        "What about the notice?", "And Article 40?", "Is it still in force?",
        "Can this decision be appealed?", "Quel est le délai pour saisir le tribunal ?",
        # Un article d'un texte deja nomme (revue du tour 4) : comme « Et
        # l'article 38 ? ».
        "Que dit l'article 38 de ce code ?", "Et l'article 38 de cette loi ?",
        "Que prévoit l'article 40 du même code ?", "Que dit l'article 38 dudit code ?",
        "Que dit l'article 38 de la loi ?", "What does Article 40 of this law say?",
        "And Article 40 of the same code?",
    ])
    def test_une_relance(self, question):
        assert est_une_relance(question)

    def test_un_renvoi_au_texte_n_a_pas_de_mot_propre(self):
        """« même », « code », « say » ne font pas un sujet : celui de la conversation."""
        assert mots_propres("Que prévoit l'article 40 du même code ?") == []
        assert mots_propres("What does Article 40 of this law say?") == []
        assert mots_propres("Que dit l'article 38 de ce code sur le préavis ?") == ["preavis"]

    @pytest.mark.parametrize("question", [
        # Des mots-cles nus ont leur sujet.
        "divorce", "TVA", "vol", "garde à vue", "inheritance",
        "Combien de temps dure le préavis ?",
        "Comment saisir le tribunal de commerce ?",
        QUESTION, "Le SMIG au Cameroun c'est combien ?",
        # Ouverte par « Et », mais avec son sujet.
        "Et quelles sont les conditions pour acquérir la nationalité camerounaise par mariage ?",
        # « ce montant » renvoie a la question elle-meme.
        "Combien l'État avait-il prévu pour les CAS en 2023, et comment ce montant se "
        "répartit-il ?",
        # Un article avec son texte, un texte par son numero.
        "Et l'article 38 du Code pénal ?", "Que dit la loi n° 2013/004 ?",
        "",
    ])
    def test_une_question_qui_a_son_sujet(self, question):
        assert not est_une_relance(question)

    def test_les_mots_propres(self):
        assert mots_propres("Et si l'employeur refuse de payer ?") == ["employeur", "refuse"]
        assert mots_propres("Peux-tu détailler ?") == []
        assert mots_propres("Quelles indemnités en cas de licenciement abusif ?") == [
            "indemnites", "licenciement", "abusif",
        ]

    def test_le_numero_sans_texte(self):
        assert numero_sans_texte("Et l'article 38 ?") == "38"
        assert numero_sans_texte("Et l'article premier ?") == "1"
        assert numero_sans_texte("Que dit l'article 39 du Code du travail ?") is None
        assert numero_sans_texte("Peux-tu détailler ?") is None
        assert numero_sans_texte("Que prévoit l'article 40 du même code ?") == "40"


class TestPeutChangerDeSujet:
    """
    Revue du tour 4 : une relance ouverte par « Et », « Mais », « What about »,
    « But what if », qui apporte ses mots, peut changer de sujet. Jointe
    seulement au sujet precedent, « Et la nationalité ? » lisait 0 extrait du
    Code de la nationalite sur 8. Elle est cherchee seule ET jointe.
    """

    @pytest.mark.parametrize("question", [
        # Les changements de sujet de la revue (sonde_nouveau_sujet).
        "Et si mon mari me bat ?", "Et si mon mari me frappe, que faire ?",
        "Et si on me vole mon téléphone ?", "Et en cas de viol ?",
        "Mais que dit la loi sur le harcèlement sexuel ?", "Mais si la police m'arrête sans mandat ?",
        "Et pour le divorce ?", "Et l'héritage ?", "Et la nationalité ?",
        "Et pour un terrain sans titre foncier ?", "What about domestic violence?",
        "But what if my landlord evicts me?", "What about inheritance?",
        # De vraies suites aussi : les mots seuls ne tranchent pas, les deux
        # recherches sont lues.
        "Et pour un CDD ?", "Et le préavis ?", "Et si l'employeur refuse de payer ?",
        "And what about the notice period?",
    ])
    def test_ouverte_avec_ses_mots(self, question):
        assert est_une_relance(question) and peut_changer_de_sujet(question)

    @pytest.mark.parametrize("question", [
        # Sans mot propre : le sujet de la conversation.
        "Et l'article 38 ?", "Combien ça coûte ?", "Et les frais ?", "Peux-tu détailler ?",
        "Et le montant ?",
        # Un renvoi a ce qui precede, ou au texte de la conversation.
        "Peut-on faire appel de cette décision ?", "Et cette décision ?",
        "Que prévoit l'article 40 du même code ?", "Et l'article 38 de ce code sur le préavis ?",
        # Un article sans son texte : celui de la conversation.
        "Et l'article suivant du code ?", "Et l'article 38 sur la faute lourde ?",
        "Et l'art. 38 pour un CDD ?",
        # Pas des relances : cherchees seules, comme avant.
        "la dot", "Quelles sont les conditions pour obtenir la nationalité camerounaise ?",
        "Et l'article 38 du Code pénal ?", "And if the police arrest me without a warrant?",
    ])
    def test_jointe_seulement_ou_seule(self, question):
        assert not peut_changer_de_sujet(question)

    def test_un_extrait_porte_un_mot_propre(self):
        """Mot pour mot, au pluriel pres : « dot » n'est pas « dotation »."""
        dot = _extrait(1, 1804, "1382.2", "CODE-CIVIL", "Code civil",
                       contenu="Rupture des fiançailles : refus de restituer la dot.")
        dotation = dot.model_copy(update={"article_id": 2, "content": "Les dotations de l'État."})
        indemnites = dot.model_copy(update={"article_id": 3, "content": "Les indemnités dues."})

        assert porte_un_mot_propre(dot, {"dot"})
        assert not porte_un_mot_propre(dotation, {"dot"})
        assert porte_un_mot_propre(indemnites, {"indemnite"})
        # L'intitule de la loi compte aussi.
        assert porte_un_mot_propre(dotation, {"civil"})


class TestSujetDeLaConversation:

    @pytest.mark.parametrize("questions, fin", [
        ((QUESTION, "Et la nationalité ?"), ("Et la nationalité ?",)),
        ((QUESTION, "Et la nationalité ?", "Peut-on faire appel de cette décision ?"),
         ("Et la nationalité ?", "Peut-on faire appel de cette décision ?")),
        ((QUESTION, "Et la nationalité ?", "Et le divorce ?"), ("Et le divorce ?",)),
        ((QUESTION,), None),
        ((QUESTION, "Peut-on faire appel de cette décision ?"), None),
        ((QUESTION, "Peux-tu détailler ?"), None),
    ])
    def test_depuis_un_changement_possible(self, questions, fin):
        sujet = sujet_de_la_conversation(_historique(*questions))
        attendu = None if fin is None else module.SujetDeLaConversation(fin)
        assert sujet.depuis_un_changement_possible() == attendu

    def test_la_derniere_question_de_fond(self):
        sujet = sujet_de_la_conversation(_historique(QUESTION, "Peux-tu détailler ?"))
        assert sujet.questions == (QUESTION,)

    def test_les_relances_qui_ont_leurs_mots_restent(self):
        sujet = sujet_de_la_conversation(_historique(
            "Que dit le Code pénal sur le vol ?", QUESTION, "Et pour un CDD ?", "Peux-tu détailler ?",
        ))
        assert sujet.texte == f"{QUESTION} Et pour un CDD ?"

    def test_sans_question_de_fond_rien(self):
        assert sujet_de_la_conversation(_historique("Peux-tu détailler ?", "Merci")) is None
        assert sujet_de_la_conversation([]) is None


class TestTextesNommesDans:

    @pytest.mark.parametrize("texte, attendus", [
        ("L'article 39 du Code du travail prévoit des dommages-intérêts.", {"LOI-1992-007"}),
        ("Selon la Constitution et le Code pénal", {"PRC-6283", "PRC-4722"}),
        ("Under the Labour Code, an unfair dismissal...", {"LOI-1992-007"}),
        # « constitution » au sens commun.
        ("La constitution d'une SARL et la constitution de partie civile", set()),
        ("Le Code du travail français", {"LOI-1992-007"}),
        ("Rien ici.", set()),
    ])
    def test_les_noms_de_la_table(self, texte, attendus):
        assert textes_de_la_table_dans(texte) == attendus


class TestRelance:

    @pytest.mark.asyncio
    async def test_une_relance_est_cherchee_avec_son_sujet_des_la_premiere_recherche(self, chat):
        """
        Revue du 29/10/2026 (tour 3) : « Et l'article 38 ? » passait seule la
        barriere (0,486) et le modele lisait l'article 38 de huit textes sans
        rapport. Elle est cherchee jointe au sujet, sans attendre un refus.
        """
        appels = _recherches(chat, similarite=0.52)

        contexte = await chat._retrieve_chunks("Combien ça coûte ?", "fr", history=_historique())

        assert contexte == _code_du_travail("semantic")
        assert appels[0][0] == f"{QUESTION} Combien ça coûte ?"
        # « Combien ça coûte ? » n'a aucun mot propre : les mots-cles du sujet.
        assert appels[0][1] == MOTS_CLES
        etat = chat._recuperation
        assert (etat.relance, etat.sujet, etat.question_jointe) == (
            True, QUESTION, f"{QUESTION} Combien ça coûte ?",
        )
        assert len(appels) == 1

    @pytest.mark.asyncio
    async def test_les_mots_propres_de_la_relance_passent_en_tete_du_plein_texte(self, chat):
        appels = _recherches(chat)

        await chat._retrieve_chunks("Et pour un CDD ?", "fr", history=_historique())

        # Cherchee seule d'abord (elle peut changer de sujet), puis jointe.
        assert appels[1][1] == f"cdd {MOTS_CLES}"
        assert chat._recuperation.mots_cles == appels[1][1]

    @pytest.mark.asyncio
    async def test_un_changement_de_sujet_possible_est_cherche_seul_et_joint(self, chat):
        """
        Revue du tour 4 : « Et la nationalité ? », jointe seulement au sujet
        du licenciement, lisait 0 extrait du Code de la nationalite sur 8.
        Cherchee seule ET jointe : les deux listes alternees, la sienne au
        premier rang.
        """
        seule, jointe = _nationalite(), _code_du_travail("semantic")
        appels = _recherches_successives(chat, (seule, 0.445), (jointe, 0.395))

        contexte = await chat._retrieve_chunks("Et la nationalité ?", "fr", history=_historique())

        assert contexte == [seule[0], jointe[0], seule[1], jointe[1]]
        assert [a[0] for a in appels] == ["Et la nationalité ?", f"{QUESTION} Et la nationalité ?"]
        etat = chat._recuperation
        assert etat.seule_et_jointe and etat.relance and not etat.refus and etat.mode == "sain"
        assert etat.question_jointe == f"{QUESTION} Et la nationalité ?"
        assert (etat.seule.similarite, etat.jointe.similarite, etat.similarite) == (
            0.445, 0.395, 0.445,
        )
        assert [c.article_id for c in etat.hybride] == [c.article_id for c in seule + jointe]

    @pytest.mark.asyncio
    async def test_la_fusion_est_sans_doublon(self, chat):
        """Un extrait trouve par les deux recherches n'est lu qu'a sa premiere place."""
        code = _code_du_travail("semantic")
        _recherches_successives(chat, ([code[0]] + _nationalite(), 0.44), (code, 0.5))

        contexte = await chat._retrieve_chunks("Et la nationalité ?", "fr", history=_historique())

        assert [c.article_id for c in contexte] == [97420, 68007, 97421, 68008]

    @pytest.mark.asyncio
    async def test_une_vraie_suite_lit_le_texte_de_la_conversation_au_rang_2(self, chat):
        """
        Revue du tour 5 (sonde_vraies_suites) : ses extraits d'abord, « Et le
        préavis ? » lisait en tete le preavis d'autres regimes (cooperatives,
        universites, marches publics...) et le Code du travail au rang 9. Les
        deux recherches alternees : le Code du travail au rang 2, chacune dans
        son ordre, la plus longue finissant seule.
        """
        autres = [
            _extrait(5000 + i, 500 + i, str(i), f"PRC-{500 + i}", "Décret sans rapport",
                     contenu="Un préavis de cent vingt jours.")
            for i in range(1, 9)
        ]
        code = _code_du_travail("semantic")
        _recherches_successives(chat, (autres, 0.383), (code, 0.569))

        contexte = await chat._retrieve_chunks("Et le préavis ?", "fr", history=_historique())

        assert contexte == [autres[0], code[0], autres[1], code[1]] + autres[2:]
        assert chat._recuperation.seule_et_jointe and not chat._recuperation.refus

    @pytest.mark.asyncio
    async def test_l_article_herite_reste_en_tete_sans_decaler_la_jointe(self, chat):
        """
        L'article nomme ouvre deja la liste jointe (_retenir_le_pertinent) : en
        tete du contexte, puis les deux listes alternees, le 1er extrait joint
        qui suit l'article au 3e rang (et non au 4e, decale par l'article).
        """
        nomme = TestArticleDUneRelance._nomme()
        seule = _nationalite()
        jointe = [nomme] + _code_du_travail("semantic")[1:] + [DOT]
        etats = [
            module.EtatDeLaRecuperation(mode="sain", similarite=0.44, hybride=list(seule)),
            module.EtatDeLaRecuperation(
                mode="sain", similarite=0.5, hybride=list(jointe[1:]), nommes=[nomme],
                relance=True,
            ),
        ]
        rendus = [list(seule), list(jointe)]

        async def une_recherche(*args, **kwargs):
            chat._recuperation = etats.pop(0)
            return rendus.pop(0)

        chat._une_recherche = une_recherche
        sujet = sujet_de_la_conversation(_historique())

        contexte = await chat._chercher_seule_et_jointe("Et la nationalité ?", _historique(), sujet)

        assert contexte == [nomme, seule[0], jointe[1], seule[1], DOT]
        assert chat._recuperation.nommes == [nomme] and chat._recuperation.juge == contexte

    @pytest.mark.asyncio
    async def test_refusee_seule_la_jointe_ne_garde_que_ce_qui_porte_ses_mots(self, chat):
        """
        Refusee seule, « Et la dot ? » n'a rien trouve sur ses mots : de la
        recherche jointe, seul l'extrait qui porte « dot » reste ; le Code du
        travail, l'ancien sujet, est ecarte.
        """
        _recherches_successives(
            chat, ([SANS_RAPPORT], TRES_FAIBLE), (_code_du_travail("semantic") + [DOT], 0.49)
        )

        contexte = await chat._retrieve_chunks("Et la dot ?", "fr", history=_historique())

        assert contexte == [DOT]
        etat = chat._recuperation
        assert etat.seule.refus and not etat.jointe.refus
        assert not etat.refus and not etat.jointe_ecartee

    @pytest.mark.asyncio
    async def test_refusee_seule_rien_qui_porte_ses_mots_refus(self, chat):
        _recherches_successives(
            chat, ([SANS_RAPPORT], TRES_FAIBLE), (_code_du_travail("semantic"), 0.49)
        )

        contexte = await chat._retrieve_chunks("Et la dot ?", "fr", history=_historique())

        assert contexte == []
        etat = chat._recuperation
        assert etat.refus and etat.jointe_ecartee and etat.question_jointe is not None

    @pytest.mark.asyncio
    async def test_un_voisin_de_la_question_seule_corrobore(self, chat):
        """Un extrait que la question seule avait trouve, sous le seuil, et que la jointe retient."""
        code = _code_du_travail("semantic")
        _recherches_successives(chat, ([code[0]], TRES_FAIBLE), (code, 0.49))

        contexte = await chat._retrieve_chunks("Et la dot ?", "fr", history=_historique())

        assert contexte == [code[0]]

    @pytest.mark.asyncio
    async def test_seule_et_refusee_jointe_ses_extraits_seuls(self, chat):
        _recherches_successives(chat, (_nationalite(), 0.445), (_tableaux(2), TRES_FAIBLE))

        contexte = await chat._retrieve_chunks("Et la nationalité ?", "fr", history=_historique())

        assert contexte == _nationalite()
        assert chat._recuperation.jointe.refus and not chat._recuperation.refus

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "Combien ça coûte ?", "Peut-on faire appel de cette décision ?",
        "Et l'article suivant du code ?",
    ])
    async def test_sans_mot_propre_ou_qui_renvoie_jointe_seulement(self, chat, question):
        """Elles parlent de la conversation : une recherche, jointe, comme au tour 4."""
        appels = _recherches(chat)

        await chat._retrieve_chunks(question, "fr", history=_historique())

        assert [a[0] for a in appels] == [f"{QUESTION} {question}"]
        assert chat._recuperation.relance and not chat._recuperation.seule_et_jointe

    @pytest.mark.asyncio
    async def test_la_seconde_relance_prend_le_sujet_et_non_la_premiere(self, chat):
        """
        Revue du tour 3 : « Donne-moi un exemple concret » apres « Peux-tu
        détailler ? » etait jointe a cette relance (0,186) et refusee.
        """
        appels = _recherches(chat)

        await chat._retrieve_chunks(
            "Donne-moi un exemple concret", "fr", history=_historique(QUESTION, "Peux-tu détailler ?")
        )

        assert appels[0][0] == f"{QUESTION} Donne-moi un exemple concret"

    @pytest.mark.asyncio
    async def test_la_suite_d_un_changement_de_sujet_relit_la_fin_du_sujet(self, chat):
        """
        Sonde du tour 5 : apres « Et la nationalité ? », « Peux-tu détailler ? »
        jointe au sujet entier ne lisait aucun extrait du Code de la
        nationalite. La fin du sujet est cherchee de nouveau, seule, et la
        relance jointe au sujet entier ; les deux alternees, la fin au premier
        rang.
        """
        nationalite, code = _nationalite(), _code_du_travail("semantic")
        appels = _recherches_successives(chat, (nationalite, 0.445), (code, 0.42))

        contexte = await chat._retrieve_chunks(
            "Peux-tu détailler ?", "fr", history=_historique(QUESTION, "Et la nationalité ?")
        )

        assert contexte == [nationalite[0], code[0], nationalite[1], code[1]]
        assert [a[0] for a in appels] == [
            "Et la nationalité ?", f"{QUESTION} Et la nationalité ? Peux-tu détailler ?",
        ]
        etat = chat._recuperation
        assert (etat.seule_et_jointe, etat.relance, etat.refus) == (True, True, False)
        assert etat.sujet_recent == "Et la nationalité ?"
        assert etat.question_jointe == f"{QUESTION} Et la nationalité ? Peux-tu détailler ?"

    @pytest.mark.asyncio
    async def test_la_fin_du_sujet_refusee_reste_le_sujet_entier(self, chat):
        """Sans mot propre, rien a corroborer : la recherche jointe reste, seule."""
        code = _code_du_travail("semantic")
        _recherches_successives(chat, ([SANS_RAPPORT], TRES_FAIBLE), (code, 0.42))

        contexte = await chat._retrieve_chunks(
            "Peux-tu détailler ?", "fr", history=_historique(QUESTION, "Et la nationalité ?")
        )

        assert contexte == code
        etat = chat._recuperation
        assert etat.seule.refus and not etat.refus and not etat.jointe_ecartee

    @pytest.mark.asyncio
    @pytest.mark.parametrize("precedentes, question, cherchees", [
        # Aucune relance qui puisse changer de sujet : une recherche, jointe.
        ((QUESTION, "Peux-tu détailler ?"), "Donne-moi un exemple concret",
         [f"{QUESTION} Donne-moi un exemple concret"]),
        ((QUESTION, "Peut-on faire appel de cette décision ?"), "Peux-tu détailler ?",
         [f"{QUESTION} Peut-on faire appel de cette décision ? Peux-tu détailler ?"]),
        # Un article nomme sans texte : celui de la conversation, une recherche.
        ((QUESTION, "Et la nationalité ?"), "Et l'article 38 ?",
         [f"{QUESTION} Et la nationalité ? Et l'article 38 ?"]),
        # Ses propres mots : elle peut elle-meme changer de sujet, cherchee
        # seule puis jointe.
        ((QUESTION, "Et la nationalité ?"), "Et pour un enfant né à l'étranger ?",
         ["Et pour un enfant né à l'étranger ?",
          f"{QUESTION} Et la nationalité ? Et pour un enfant né à l'étranger ?"]),
    ])
    async def test_ce_qui_n_est_pas_la_suite_d_un_changement(
        self, chat, precedentes, question, cherchees
    ):
        appels = _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={})
        chat._lire_l_article_nomme = AsyncMock(return_value=[])

        await chat._retrieve_chunks(question, "fr", history=_historique(*precedentes))

        assert [a[0] for a in appels] == cherchees
        assert chat._recuperation.sujet_recent is None

    @pytest.mark.asyncio
    async def test_une_question_qui_a_son_sujet_est_cherchee_seule(self, chat):
        appels = _recherches(chat)
        question = "Quelle est la durée du congé de maternité ?"

        await chat._retrieve_chunks(question, "fr", history=_historique())

        assert appels == [(question, chat.extract_keywords(question))]
        assert chat._recuperation.relance is False and chat._recuperation.question_jointe is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("relance, budgetaire", [
        ("Et pour 2024 ?", True),
        # Un sujet a elle : une question de droit, sans tableaux.
        ("Et le SMIG ?", False),
    ])
    async def test_une_relance_n_herite_du_budget_que_sans_sujet_a_elle(
        self, chat, relance, budgetaire
    ):
        _recherches(chat)

        await chat._retrieve_chunks(relance, "fr", history=_historique(
            "Quel est le budget du ministère de la santé dans la loi de finances 2023 ?"
        ))

        assert chat._recuperation.relance is True
        assert chat._recuperation.budgetaire is budgetaire
        assert chat.search_service.tableaux_de_programmes_exclus is not budgetaire

    @pytest.mark.asyncio
    async def test_sans_historique_une_relance_est_cherchee_seule(self, chat):
        appels = _recherches(chat)

        await chat._retrieve_chunks("Peux-tu détailler ?", "fr")

        assert appels[0][0] == "Peux-tu détailler ?"

    @pytest.mark.asyncio
    async def test_le_sujet_sorti_de_l_historique_est_relu(self, chat):
        """
        Troisieme relance de suite : la question de fond est sortie des cinq
        derniers messages. Une lecture des messages de l'utilisateur la
        retrouve.
        """
        appels = _recherches(chat)
        lus = MagicMock()
        lus.scalars.return_value.all.return_value = [
            "Donne-moi un exemple concret", "Peux-tu détailler ?", QUESTION,
        ]
        chat.db.execute = AsyncMock(return_value=lus)
        chat.db.begin_nested = MagicMock(return_value=MagicMock())
        fenetre = _historique("Peux-tu détailler ?", "Donne-moi un exemple concret")[1:]

        await chat._retrieve_chunks(
            "Explique plus simplement", "fr", history=fenetre,
            conversation=SimpleNamespace(id=7),
        )

        assert appels[0][0] == f"{QUESTION} Explique plus simplement"

    @pytest.mark.asyncio
    async def test_refusee_seule_elle_est_cherchee_avec_le_sujet(self, chat):
        """
        Une question que rien ne designait comme une relance, refusee seule :
        cherchee de nouveau jointe au SUJET, pas au dernier message.
        """
        appels = []
        hybride = _code_du_travail("semantic")

        async def chercher(requete):
            appels.append(requete.semantic_query)
            chat.search_service.volets = _volets(
                {c.article_id: (0.5 if len(appels) > 1 else TRES_FAIBLE) for c in hybride}
            )
            return MagicMock(chunks=list(hybride), search_time_ms=1)

        chat.search_service.search = chercher
        question = "Combien de temps dure le préavis ?"
        requete = RAGRequest(question=question, persona="citoyen", language="fr")
        historique = _historique(QUESTION, "Peux-tu détailler ?")

        await chat._retrieve_chunks(question, "fr", history=historique)
        assert chat._recuperation.refus is True and chat._recuperation.relance is False
        contexte = await chat._chercher_la_relance(requete, historique)

        assert contexte == hybride
        assert appels == [question, f"{QUESTION} {question}"]
        assert chat._recuperation.question_jointe == f"{QUESTION} {question}"
        assert chat._recuperation.refus is False

    @pytest.mark.asyncio
    async def test_la_seconde_chance_ne_garde_que_ce_qui_porte_ses_mots(self, chat):
        """
        Revue du tour 4 : « la dot », refusee seule (0,235), passait jointe au
        sujet du licenciement (0,492) avec huit extraits du Code du travail et
        du statut des fonctionnaires. Seul ce qui porte « dot » reste.
        """
        _recherches_successives(
            chat, ([SANS_RAPPORT], TRES_FAIBLE), (_code_du_travail("semantic") + [DOT], 0.49)
        )
        requete = RAGRequest(question="la dot", persona="citoyen", language="fr")

        await chat._retrieve_chunks("la dot", "fr", history=_historique())
        assert chat._recuperation.refus is True and chat._recuperation.relance is False
        contexte = await chat._chercher_la_relance(requete, _historique())

        assert contexte == [DOT]
        assert chat._recuperation.refus is False

    @pytest.mark.asyncio
    async def test_la_seconde_chance_sans_rien_qui_porte_ses_mots_refuse(self, chat):
        """Ni ses mots ni ses voisins : refus, et pas de repli sur la Constitution."""
        _recherches_successives(
            chat, ([SANS_RAPPORT], TRES_FAIBLE), (_code_du_travail("semantic"), 0.49)
        )
        chat._fallback_search = AsyncMock(side_effect=AssertionError("repli lance"))
        requete = RAGRequest(question="c koi la dot", persona="citoyen", language="fr")

        contexte, reponse = await chat._recuperer(requete, None, _historique())

        assert (contexte, reponse) == ([], None)
        etat = chat._recuperation
        assert etat.refus and etat.jointe_ecartee
        assert etat.question_jointe == f"{QUESTION} c koi la dot"

    @pytest.mark.asyncio
    async def test_rien_sans_historique_ni_sans_refus(self, chat):
        requete = RAGRequest(question="Peux-tu détailler ?", persona="citoyen", language="fr")
        chat._retrieve_chunks = AsyncMock(side_effect=AssertionError("recherche relancee"))

        chat._recuperation.refus = True
        assert await chat._chercher_la_relance(requete, []) == []
        chat._recuperation.refus = False
        assert await chat._chercher_la_relance(requete, _historique()) == []

    @pytest.mark.asyncio
    async def test_rien_apres_une_relance_deja_jointe(self, chat):
        requete = RAGRequest(question="Peux-tu détailler ?", persona="citoyen", language="fr")
        chat._retrieve_chunks = AsyncMock(side_effect=AssertionError("recherche relancee"))
        chat._recuperation.refus = chat._recuperation.relance = True

        assert await chat._chercher_la_relance(requete, _historique()) == []


class TestArticleDUneRelance:
    """« Et l'article 38 ? » : l'article 38 du texte du tour precedent, ou rien."""

    @staticmethod
    def _nomme(numero="35.1"):
        return _code_du_travail()[0].model_copy(update={
            "source": SOURCE_ARTICLE_NOMME, "number": numero,
        })

    @pytest.mark.asyncio
    async def test_le_texte_que_l_utilisateur_a_nomme(self, chat):
        _recherches(chat)
        chat._texte_nomme = AsyncMock(return_value=(2274, "LOI-1992-007"))
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme("35.2")])

        contexte = await chat._retrieve_chunks(
            "Et l'article 40 ?", "fr",
            history=_historique("Que dit l'article 39 du Code du travail ?"),
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("40", (2274, "LOI-1992-007"))
        assert contexte[0].source == SOURCE_ARTICLE_NOMME
        etat = chat._recuperation
        assert (etat.texte_herite, etat.article_nomme) == ("LOI-1992-007", "LOI-1992-007 art. 35.2")

    @pytest.mark.asyncio
    async def test_les_sources_de_la_derniere_reponse(self, chat):
        _recherches(chat)
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme()])
        sources = [{"law_id": 2274, "law_reference": "LOI-1992-007", "article_number": "37"}]

        await chat._retrieve_chunks(
            "Et l'article 38 ?", "fr", history=_historique(reponse="Voir l'article 37.", sources=sources)
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("38", (2274, "LOI-1992-007"))

    @pytest.mark.asyncio
    async def test_le_texte_que_nomme_la_derniere_reponse(self, chat):
        _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme()])

        await chat._retrieve_chunks(
            "Et l'article 38 ?", "fr",
            history=_historique(reponse="L'article 39 du Code du travail prévoit des dommages."),
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("38", (2274, "LOI-1992-007"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "Que dit l'article 38 de ce code ?", "Et l'article 38 de cette loi ?",
        "Que prévoit l'article 38 du même code ?", "Que dit l'article 38 dudit code ?",
        "Que dit l'article 38 de la loi ?", "What does Article 38 of this law say?",
        "And Article 38 of the same code?",
    ])
    async def test_un_renvoi_au_texte_herite_comme_et_l_article_38(self, chat, question):
        """
        Revue du tour 4 (sonde_renvoi_article_flux) : ces formes etaient
        cherchees seules, et le modele lisait l'article 38 de six a huit
        textes sans rapport, aucun extrait du Code du travail. L'article 38 du
        texte du tour precedent, en tete, et une recherche jointe au sujet.
        """
        appels = _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        # « Loi … portant Code du travail » : une loi ET un code.
        chat._intitule_du_texte = AsyncMock(return_value=TITRE_CODE_DU_TRAVAIL)
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme()])

        contexte = await chat._retrieve_chunks(
            question, "fr",
            history=_historique(reponse="L'article 39 du Code du travail prévoit des dommages."),
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("38", (2274, "LOI-1992-007"))
        chat._intitule_du_texte.assert_awaited_once_with(2274)
        assert contexte[0].source == SOURCE_ARTICLE_NOMME
        assert [a[0] for a in appels] == [f"{QUESTION} {question}"]
        # Herite : les mots-cles du sujet, sans le mot de nature (tour 5).
        assert appels[0][1] == MOTS_CLES
        etat = chat._recuperation
        assert (etat.relance, etat.seule_et_jointe, etat.texte_herite) == (
            True, False, "LOI-1992-007",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question, mot", [
        ("Que dit l'article 2 de ce décret ?", "décret"),
        ("Et l'article 1er dudit décret ?", "décret"),
        ("Que dit l'article 3 du décret ?", "décret"),
        ("What does Article 2 of that decree say?", "decree"),
        ("Et l'article 5 de cette ordonnance ?", "ordonnance"),
    ])
    async def test_un_renvoi_d_une_autre_nature_n_herite_de_rien(self, chat, question, mot):
        """
        Revue du tour 5 (sonde_renvoi_nature) : la conversation nomme le Code
        du travail, la reponse parle du decret n° 2024/056, et « l'article 2 de
        ce décret » heritait du Code du travail : son article 2 (« Le droit au
        travail est reconnu… ») en tete, pertinence 1,00. Le Code du travail
        n'est pas un decret : rien d'herite, une recherche jointe au sujet,
        comme toute relance, et le mot de nature en tete des mots-cles (« Et
        l'article 1er dudit décret ? » ne lisait plus le decret sans lui).
        """
        appels = _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._intitule_du_texte = AsyncMock(return_value=TITRE_CODE_DU_TRAVAIL)
        chat._lire_l_article_nomme = AsyncMock(side_effect=AssertionError("article herite"))

        contexte = await chat._retrieve_chunks(
            question, "fr", history=_historique(ALLOCATIONS, reponse=REPONSE_DECRET)
        )

        chat._intitule_du_texte.assert_awaited_once_with(2274)
        assert all(c.source != SOURCE_ARTICLE_NOMME for c in contexte)
        etat = chat._recuperation
        assert (etat.relance, etat.seule_et_jointe, etat.texte_herite, etat.article_nomme) == (
            True, False, None, None,
        )
        assert [a[0] for a in appels] == [f"{ALLOCATIONS} {question}"]
        assert appels[0][1] == f"{mot} {chat.extract_keywords(ALLOCATIONS)}"
        assert etat.mots_cles == appels[0][1]

    @pytest.mark.asyncio
    async def test_un_renvoi_de_meme_nature_herite(self, chat):
        """« ce décret », quand le texte de la conversation est un decret : son article, en tete."""
        appels = _recherches(chat)
        de_fond = "Que dit le décret n° 2024/056 sur les allocations familiales ?"
        chat._texte_nomme = AsyncMock(return_value=(9955, "PRC-9955"))
        chat._identifiants_des_textes = AsyncMock(return_value={})
        chat._intitule_du_texte = AsyncMock(return_value=TITRE_DECRET)
        article = _extrait(99552, 9955, "2", "PRC-9955", TITRE_DECRET, SOURCE_ARTICLE_NOMME,
                           "Sont abrogées toutes les dispositions antérieures contraires.")
        chat._lire_l_article_nomme = AsyncMock(return_value=[article])

        contexte = await chat._retrieve_chunks(
            "Que dit l'article 2 de ce décret ?", "fr", history=_historique(de_fond)
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("2", (9955, "PRC-9955"))
        assert contexte[0] == article
        assert chat._recuperation.texte_herite == "PRC-9955"
        assert appels[0][1] == chat.extract_keywords(de_fond)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "Et l'article 40 de ce texte ?", "Article 40 of the said Act",
    ])
    async def test_un_renvoi_sans_nature_herite_sans_lire_l_intitule(self, chat, question):
        """« ce texte », « the said Act » ne disent pas de nature : le texte du tour precedent."""
        appels = _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._intitule_du_texte = AsyncMock(side_effect=AssertionError("intitule lu"))
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme("35.2")])

        await chat._retrieve_chunks(
            question, "fr",
            history=_historique(reponse="L'article 39 du Code du travail prévoit des dommages."),
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("40", (2274, "LOI-1992-007"))
        assert chat._recuperation.texte_herite == "LOI-1992-007"
        assert appels[0][1] == MOTS_CLES

    @pytest.mark.asyncio
    async def test_un_intitule_illisible_rien_d_herite(self, chat):
        """Dans le doute, rien : la nature du texte n'est pas lue, « ce code » n'en herite pas."""
        appels = _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._intitule_du_texte = AsyncMock(return_value=None)
        chat._lire_l_article_nomme = AsyncMock(side_effect=AssertionError("article herite"))

        await chat._retrieve_chunks(
            "Que dit l'article 38 de ce code ?", "fr",
            history=_historique(reponse="L'article 39 du Code du travail prévoit des dommages."),
        )

        assert chat._recuperation.texte_herite is None
        assert appels[0][1] == f"code {MOTS_CLES}"

    @pytest.mark.asyncio
    async def test_l_intitule_en_echec_ne_lit_rien(self, chat, caplog):
        chat.db.execute = AsyncMock(side_effect=RuntimeError("base indisponible"))

        with caplog.at_level(logging.WARNING, logger="app.services.rag_service"):
            assert await chat._intitule_du_texte(2274) is None
        assert "Intitule du texte 2274 non lu" in caplog.text

    @pytest.mark.asyncio
    async def test_le_meme_code_que_la_question_precedente(self, chat):
        """Apres « l'article 39 du Code du travail », « l'article 40 du même code »."""
        _recherches(chat)
        chat._texte_nomme = AsyncMock(return_value=(2274, "LOI-1992-007"))
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._intitule_du_texte = AsyncMock(return_value=TITRE_CODE_DU_TRAVAIL)
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme("35.2")])

        await chat._retrieve_chunks(
            "Que dit l'article 40 du même code ?", "fr",
            history=_historique("Que dit l'article 39 du Code du travail ?"),
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("40", (2274, "LOI-1992-007"))
        assert chat._recuperation.article_nomme == "LOI-1992-007 art. 35.2"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sources, attendu", [
        # La derniere reponse cite le Code de la nationalite : son article 38.
        ([{"law_id": 1968, "law_reference": "LOI-1968-LF-3"}], (1968, "LOI-1968-LF-3")),
        # Elle ne dit aucun texte : rien, et non le texte nomme AVANT.
        (None, None),
    ])
    async def test_apres_un_changement_de_sujet_le_texte_nomme_avant_ne_compte_plus(
        self, chat, sources, attendu
    ):
        """
        « Que dit l'article 39 du Code du travail ? », puis « Et la
        nationalité ? », puis « Et l'article 38 ? » : l'article 38 du Code du
        travail passait en tete (texte nomme par la question de fond). La
        regle du texte nomme ne lit que la fin du sujet.
        """
        _recherches(chat)
        chat._texte_nomme = AsyncMock(return_value=(2274, "LOI-1992-007"))
        chat._identifiants_des_textes = AsyncMock(return_value={})
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme()])
        historique = _historique(
            "Que dit l'article 39 du Code du travail ?", "Et la nationalité ?",
            reponse="Réponse.", sources=sources,
        )

        contexte = await chat._retrieve_chunks("Et l'article 38 ?", "fr", history=historique)

        if attendu is None:
            chat._lire_l_article_nomme.assert_not_awaited()
            assert all(c.source != SOURCE_ARTICLE_NOMME for c in contexte)
            assert chat._recuperation.texte_herite is None
        else:
            chat._lire_l_article_nomme.assert_awaited_once_with("38", attendu)
            assert chat._recuperation.texte_herite == attendu[1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "Que dit l'article 38 de cette loi sur la nationalité ?",
        "What does Article 38 of the law say about notice?",
    ])
    async def test_un_renvoi_qui_a_ses_mots_n_herite_pas_d_un_article_qui_ne_les_porte_pas(
        self, chat, question
    ):
        """
        « cette loi sur la nationalité », dans une conversation sur le Code
        du travail : la suite du renvoi designe un autre texte. L'article 38
        du Code du travail, qui ne parle pas de nationalite, passait en tete
        (et franchissait la barriere a lui seul) : rien n'est herite, la
        question reste jointe au sujet (sonde du tour 5 : l'article 38 du
        Code de la nationalite y est lu au rang 5).
        """
        appels = _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._intitule_du_texte = AsyncMock(return_value=TITRE_CODE_DU_TRAVAIL)
        chat._lire_l_article_nomme = AsyncMock(return_value=[self._nomme()])

        contexte = await chat._retrieve_chunks(
            question, "fr",
            history=_historique(reponse="L'article 39 du Code du travail prévoit des dommages."),
        )

        chat._lire_l_article_nomme.assert_awaited_once_with("38", (2274, "LOI-1992-007"))
        assert all(c.source != SOURCE_ARTICLE_NOMME for c in contexte)
        etat = chat._recuperation
        assert (etat.relance, etat.texte_herite, etat.article_nomme) == (True, None, None)
        assert [a[0] for a in appels] == [f"{QUESTION} {question}"]

    @pytest.mark.asyncio
    async def test_un_renvoi_dont_l_article_porte_les_mots_herite(self, chat):
        """« … de ce code sur le préavis » : le morceau 35.1 parle du préavis."""
        _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(return_value={"LOI-1992-007": 2274})
        chat._intitule_du_texte = AsyncMock(return_value=TITRE_CODE_DU_TRAVAIL)
        preavis = self._nomme().model_copy(update={
            "content": "Article 38.- Pendant la durée du préavis, le travailleur bénéficie d'un jour.",
        })
        chat._lire_l_article_nomme = AsyncMock(return_value=[preavis])

        contexte = await chat._retrieve_chunks(
            "Que dit l'article 38 de ce code sur le préavis ?", "fr",
            history=_historique(reponse="L'article 39 du Code du travail prévoit des dommages."),
        )

        assert contexte[0] == preavis
        assert chat._recuperation.texte_herite == "LOI-1992-007"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reponse, sources", [
        # Deux textes cites : ambigu.
        ("Voir les articles.", [
            {"law_id": 2274, "law_reference": "LOI-1992-007"},
            {"law_id": 2016, "law_reference": "PRC-4722"},
        ]),
        ("Le Code du travail et le Code pénal le prévoient.", None),
        # Aucun texte.
        ("Réponse.", None),
    ])
    async def test_ambigu_ou_absent_rien(self, chat, reponse, sources):
        _recherches(chat)
        chat._identifiants_des_textes = AsyncMock(
            return_value={"LOI-1992-007": 2274, "PRC-4722": 2016}
        )
        chat._lire_l_article_nomme = AsyncMock(side_effect=AssertionError("article lu"))

        contexte = await chat._retrieve_chunks(
            "Et l'article 38 ?", "fr", history=_historique(reponse=reponse, sources=sources)
        )

        assert all(c.source != SOURCE_ARTICLE_NOMME for c in contexte)
        assert chat._recuperation.texte_herite is None

    @pytest.mark.asyncio
    async def test_sur_la_page_d_un_texte_rien_n_est_herite(self, chat):
        """« Et l'article 38 ? » sur la page d'un texte : celui du document ouvert, lu a part."""
        _recherches(chat)
        chat._lire_l_article_nomme = AsyncMock(side_effect=AssertionError("article herite"))

        await chat._retrieve_chunks(
            "Et l'article 38 ?", "fr", history=_historique(
                reponse="Voir l'article 37.",
                sources=[{"law_id": 2274, "law_reference": "LOI-1992-007"}],
            ),
            heriter_l_article=False,
        )

        assert chat._recuperation.relance is True and chat._recuperation.texte_herite is None

    def test_aucune_remontee_par_le_seul_numero_dans_une_relance(self):
        """L'article 38 d'une loi sur la concurrence ne remonte plus en tete."""
        service = RAGService.__new__(RAGService)
        service._recuperation = module.EtatDeLaRecuperation(relance=True)
        code = _extrait(1, 2274, "35.1", "LOI-1992-007", TITRE_CODE_DU_TRAVAIL)
        concurrence = _extrait(2, 1998, "38", "LOI-1998-013", "Loi relative à la concurrence")

        assert service._pin_requested_article("Et l'article 38 ?", [code, concurrence]) == [
            code, concurrence,
        ]

    @pytest.mark.parametrize("question", [
        "Que dit l'article 38 dudit code ?", "Que dit l'article 38 de ce code ?",
        "What does Article 38 of this law say?",
    ])
    def test_un_renvoi_sans_historique_ne_remonte_rien_par_le_numero(self, question):
        """
        Revue du tour 4 : « … dudit code » remontait en tete l'article 38 du
        Code de la nationalite. Sans historique, rien ne dit quel est ce code :
        l'ordre de la recherche reste.
        """
        service = RAGService.__new__(RAGService)
        service._recuperation = module.EtatDeLaRecuperation()
        code = _extrait(1, 2274, "35.1", "LOI-1992-007", TITRE_CODE_DU_TRAVAIL)
        nationalite = _extrait(2, 1968, "38", "LOI-1968-LF-3", TITRE_NATIONALITE)

        assert service._pin_requested_article(question, [code, nationalite]) == [code, nationalite]
        # Sans renvoi, le numero seul remonte toujours son article (« Que dit
        # l'article 38 ? », sans conversation).
        assert service._pin_requested_article("Que dit l'article 38 ?", [code, nationalite]) == [
            nationalite, code,
        ]


# ==================== CE QUE VOIENT ASK ET ASK_STREAM ====================


@pytest.fixture
def base():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value=None)
    ))
    session.add = MagicMock()
    session.begin_nested = MagicMock(return_value=MagicMock())
    return session


@pytest.fixture(autouse=True)
def routage_juridique():
    with patch(
        "app.services.rag_service.classify_intent",
        new=AsyncMock(return_value=IntentResult("juridique", 1.0, "test")),
    ) as double:
        yield double


def _chat_refuse(base):
    """Le chat d'une question hors sujet : rien de confirme, modele et repli espionnes."""
    service = RAGService(base)
    service.llm = MagicMock()
    service.llm.generate = AsyncMock(side_effect=AssertionError("modele appele"))
    service.llm.generate_stream = MagicMock(side_effect=AssertionError("modele appele"))
    service.search_service = MagicMock()
    _recherche(service, _tableaux(), _volets({c.article_id: TRES_FAIBLE for c in _tableaux()}))
    service._articles_stricts = AsyncMock(return_value=set())
    service._article_nomme = AsyncMock(return_value=[])
    service._fallback_search = AsyncMock(side_effect=AssertionError("repli lance"))
    return service


def _requete(question="Que dit l'article 12 sur la recette du ndolé ?", langue="fr"):
    return RAGRequest(question=question, persona="citoyen", language=langue)


class TestRefusDeLaBarriere:

    @pytest.mark.asyncio
    async def test_ask_repond_sans_le_modele(self, base):
        service = _chat_refuse(base)

        reponse = await service.ask(_requete())

        assert reponse.answer == message_sans_texte_pertinent("fr")
        assert reponse.sources == [] and reponse.confidence == 0.0
        service._fallback_search.assert_not_called()
        # Enregistre dans l'historique, comme une recherche vide.
        assert base.commit.await_count >= 1

    @pytest.mark.asyncio
    async def test_ask_stream_repond_sans_le_modele(self, base):
        service = _chat_refuse(base)

        evenements = [json.loads(e) async for e in service.ask_stream(_requete(langue="en"))]

        etapes = [e["progression"]["etape"] for e in evenements if e.get("progression")]
        assert etapes == ["recherche", "textes_trouves"]
        trouves = [e["progression"] for e in evenements if e.get("progression")][1]
        assert trouves["nb_textes"] == 0 and trouves["libelle"] == "No text found"
        fin = evenements[-1]
        assert fin["done"] is True and not fin.get("error")
        assert fin["chunk"] == message_sans_texte_pertinent("en")
        assert fin["sources"] == [] and fin["citations"] == [] and fin["articles_cites"] == []
        service._fallback_search.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chemin", ["ask", "ask_stream"])
    async def test_une_relance_est_cherchee_avec_le_sujet_de_l_historique(self, base, chemin):
        """
        ask comme ask_stream : la seconde relance de suite est cherchee avec la
        question de fond, une fois ; refusee, rien n'est cherche de plus.
        """
        service = _chat_refuse(base)
        service._load_or_create_conversation = AsyncMock(return_value=(
            MagicMock(session_id="s", persona="citoyen"), _historique(QUESTION, "Peux-tu détailler ?"),
        ))
        requetes = []
        recherche = service.search_service.search

        async def espion(requete):
            requetes.append(requete.semantic_query)
            return await recherche(requete)

        service.search_service.search = espion
        relance = "Donne-moi un exemple concret"

        if chemin == "ask":
            reponse = (await service.ask(_requete(relance))).answer
        else:
            reponse = [json.loads(e) async for e in service.ask_stream(_requete(relance))][-1]["chunk"]

        assert requetes == [f"{QUESTION} {relance}"]
        assert reponse == message_sans_texte_pertinent("fr", relance=True)

    @pytest.mark.asyncio
    async def test_une_relance_refusee_meme_jointe_a_son_message(self, base):
        """Refusee jointe a son sujet : la reposer en entier, avec ses mots."""
        service = _chat_refuse(base)
        service._load_or_create_conversation = AsyncMock(return_value=(
            MagicMock(session_id="s", persona="citoyen"),
            _historique("Quelle est la recette du ndolé ?"),
        ))

        reponse = await service.ask(_requete("Peux-tu détailler ?"))

        assert reponse.answer == message_sans_texte_pertinent("fr", relance=True)
        assert service._recuperation.question_jointe == (
            "Quelle est la recette du ndolé ? Peux-tu détailler ?"
        )
        assert "en entier" in reponse.answer

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chemin", ["ask", "ask_stream"])
    async def test_un_changement_de_sujet_refuse_seul_et_joint(self, base, chemin):
        """
        Cherchee seule puis jointe, refusee les deux fois : ni modele, ni
        seconde chance, ni repli ; le message d'une relance.
        """
        service = _chat_refuse(base)
        service._load_or_create_conversation = AsyncMock(return_value=(
            MagicMock(session_id="s", persona="citoyen"), _historique(),
        ))
        requetes = []
        recherche = service.search_service.search

        async def espion(requete):
            requetes.append(requete.semantic_query)
            return await recherche(requete)

        service.search_service.search = espion
        question = "Et la recette du ndolé ?"

        if chemin == "ask":
            reponse = (await service.ask(_requete(question))).answer
        else:
            reponse = [json.loads(e) async for e in service.ask_stream(_requete(question))][-1]["chunk"]

        assert requetes == [question, f"{QUESTION} {question}"]
        assert reponse == message_sans_texte_pertinent("fr", relance=True)
        assert service._recuperation.seule_et_jointe and service._recuperation.refus

    @pytest.mark.asyncio
    async def test_servi_par_l_index_le_refus_dit_de_reessayer(self, base):
        """La recherche exacte n'a pas repondu a temps : reessayer, et non reformuler."""
        service = _chat_refuse(base)
        _recherche(service, _tableaux(), _volets({76001: 0.6}, chemin=CHEMIN_REPLI_HNSW))

        reponse = await service.ask(_requete())

        assert reponse.answer == message_sans_texte_pertinent("fr", volet_en_panne=True)
        assert service._recuperation.refus is True

    def test_le_message_ne_cite_aucun_article(self):
        for langue in ("fr", "en"):
            for panne, relance in ((False, False), (True, False), (False, True)):
                message = message_sans_texte_pertinent(langue, volet_en_panne=panne, relance=relance)
                assert not any(chiffre in message for chiffre in "0123456789")
                assert ("recherche" if langue == "fr" else "search") in message
        assert message_sans_texte_pertinent("de") == message_sans_texte_pertinent("fr")
        # Volet en panne : reessayer, et non reformuler.
        assert "Réessayez" in message_sans_texte_pertinent("fr", volet_en_panne=True)
        assert "Reformulez" in message_sans_texte_pertinent("fr")

    @pytest.mark.asyncio
    async def test_volet_semantique_en_panne_le_message_le_dit(self, base):
        """Le fournisseur d'embeddings n'a pas repondu : reessayer, pas reformuler."""
        service = _chat_refuse(base)
        _recherche(service, _tableaux(), _volets({}, chemin=CHEMIN_ECHEC))

        reponse = await service.ask(_requete())

        assert reponse.answer == message_sans_texte_pertinent("fr", volet_en_panne=True)
        assert service._recuperation.refus is True

    @pytest.mark.asyncio
    async def test_le_document_consulte_n_est_pas_soumis_a_la_barriere(self, base, caplog):
        """
        Sur la page d'un texte (`law_id`), le lecteur a choisi ce texte : la
        barriere ecarte ce que la recherche a trouve, pas lui.
        """
        service = _chat_refuse(base)
        document = _code_du_travail("priority")[0]
        service._get_priority_document_context = AsyncMock(return_value=([document], None))
        service._generate_and_save_response = AsyncMock(return_value="reponse redigee")
        requete = RAGRequest(question="Et la recette du ndolé ?", persona="citoyen",
                             language="fr", law_id=2274)

        with caplog.at_level(logging.INFO, logger=module.logger.name):
            assert await service.ask(requete) == "reponse redigee"

        assert service._generate_and_save_response.call_args[0][3] == [document]
        service._fallback_search.assert_not_called()
        assert any("seul le document consulte reste" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_une_recherche_vide_garde_son_message_et_son_repli(self, base):
        """Rien trouve du tout n'est pas un refus : le repli et l'ancien message restent."""
        service = _chat_refuse(base)
        _recherche(service, [], None)
        service._fallback_search = AsyncMock(return_value=[])

        reponse = await service.ask(_requete())

        service._fallback_search.assert_awaited_once()
        assert reponse.answer.startswith("Je n'ai pas trouvé d'information pertinente")


# ==================== LA SOURCE DE REPLI ====================


def _chat_qui_repond(base, reponse, contexte):
    """Le chat de l'incident : recherche doublee, modele double qui rend `reponse`."""
    service = RAGService(base)
    service.search_service = MagicMock()
    _recherche(service, contexte, _volets({c.article_id: 0.5 for c in contexte}))
    service._articles_stricts = AsyncMock(return_value=set())
    service._article_nomme = AsyncMock(return_value=[])
    service._valider_avant_la_generation = AsyncMock()
    service._identifiants_des_textes = AsyncMock(return_value={})
    service._save_interaction = AsyncMock()

    async def flux(**kwargs):
        yield reponse

    service.llm = MagicMock()
    service.llm.generate_stream = MagicMock(side_effect=lambda **kw: flux(**kw))
    service.llm.generate = AsyncMock(return_value={"response": reponse})
    return service


# Les reponses de la revue du 29/10/2026 : toutes leurs citations sont
# signalees (aucun extrait lu ne porte un article 74 ou 76 du Code du travail),
# aucune n'emploie une formule d'aveu, et le premier extrait est pertinent.
REPONSES_DE_L_INCIDENT = [
    "En cas de licenciement abusif, l'article 76 du Code du travail prévoit des "
    "dommages-intérêts.",
    "Voir l'article 74 du Code du travail. Les dommages relèvent de l'article 76 de la loi.",
    "En droit du travail, l'article 76 couvre les dommages (Code du travail).",
    "Voir l'article 76 in fine du Code du travail.",
]


class TestSourceDeRepli:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reponse", REPONSES_DE_L_INCIDENT)
    async def test_jamais_sous_une_reponse_qui_cite(self, base, reponse):
        """
        Revue du 29/10/2026 : toutes les citations signalees, `citations`
        vide, et le repli sortait le premier extrait (« Sources : Article
        76.37 — Loi N°2022/020 »), dans le dernier evenement ET dans
        l'historique enregistre.
        """
        tableaux = [c.model_copy(update={"relevance_score": 0.9}) for c in _tableaux(2)]
        service = _chat_qui_repond(base, reponse, tableaux)

        evenements = [json.loads(e) async for e in service.ask_stream(_requete(QUESTION))]

        fin = evenements[-1]
        assert fin["done"] is True and not fin.get("error")
        assert fin["sources"] == []
        assert fin["citations_introuvables"], "la citation de l'incident doit rester signalee"
        assert service._save_interaction.await_args[0][3] == []

    @pytest.mark.asyncio
    async def test_jamais_sous_une_citation_rattachee_non_plus(self, base):
        """Une citation rattachee a un texte lu SANS numero d'article : pas de source fabriquee."""
        contexte = [c.model_copy(update={"relevance_score": 0.9}) for c in _code_du_travail()]
        service = _chat_qui_repond(
            base, "La loi n° 92/007 protège le travailleur licencié.", contexte
        )

        reponse = await service.ask(_requete(QUESTION))

        assert reponse.sources == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("aveu", [
        "Les textes fournis ne précisent pas le montant de l'indemnité.",
        "The provided documents do not specify the amount.",
    ])
    async def test_jamais_sous_un_aveu(self, base, aveu):
        contexte = [c.model_copy(update={"relevance_score": 0.9}) for c in _code_du_travail()]
        service = _chat_qui_repond(base, aveu, contexte)

        reponse = await service.ask(_requete(QUESTION))

        assert reponse.sources == []
        assert service._save_interaction.await_args.kwargs["citations"] == []

    @pytest.mark.asyncio
    async def test_une_reponse_sans_citation_n_a_aucune_source(self, base):
        """
        Plus de source de repli : une reponse juste qui ne cite rien n'a pas
        de source, ni dans la reponse ni dans l'historique. Ce que le modele a
        lu reste dans « Textes consultés, non cités », que le front affiche.
        """
        contexte = [c.model_copy(update={"relevance_score": 0.9}) for c in _code_du_travail()]
        service = _chat_qui_repond(
            base, "Le travailleur licencié abusivement a droit à des dommages-intérêts.", contexte
        )

        reponse = await service.ask(_requete(QUESTION))

        assert reponse.sources == [] and reponse.articles_cites == []
        assert service._save_interaction.await_args.kwargs["citations"] == []
        assert [t.law_reference for t in reponse.textes_consultes_non_cites] == ["LOI-1992-007"]

    @pytest.mark.asyncio
    async def test_un_texte_nomme_sans_extrait_lu_n_est_pas_une_source(self, base):
        """
        Revue du 29/10/2026 (tour 3), par ask_stream : « Selon le Code du
        travail, la procédure est gratuite. » sur huit extraits de lois de
        finances. Le repli enregistrait PRC-9875 art. 35.5, un tableau de
        droits d'exportation d'animaux, comme source de la reponse.
        """
        lois_de_finances = [
            _extrait(98750 + i, 2300, f"35.{i}", "PRC-9875", TITRE_LOI_DE_FINANCES, pertinence=1.0)
            for i in (5, 8)
        ]
        service = _chat_qui_repond(
            base, "Selon le Code du travail, la procédure est gratuite.", lois_de_finances
        )

        evenements = [json.loads(e) async for e in service.ask_stream(_requete("Combien ça coûte ?"))]

        fin = evenements[-1]
        assert fin["done"] is True and not fin.get("error")
        assert fin["sources"] == [] and fin["articles_cites"] == []
        assert service._save_interaction.await_args[0][3] == []
        assert [t["law_reference"] for t in fin["textes_consultes_non_cites"]] == ["PRC-9875"]


# ==================== PAR LA VRAIE RECHERCHE ====================


CODE_DU_TRAVAIL_ID, LOI_DE_FINANCES_ID, COLLECTIVITES_ID = 1992, 2023, 2019
CODE_PENAL_ID, CONSTITUTION_ID = 2016, 1996


def _unitaire(v):
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).tolist()


def _aleatoire(graine):
    return _unitaire(np.random.default_rng(graine).normal(size=EmbeddingService.EMBEDDING_DIM))


VECTEUR_DE_LA_QUESTION = _aleatoire(7)
# Un tableau budgetaire long, sans un mot de la question.
TABLEAU = "Chapitre 45 — Programme 512 — Dotations : 1 250 000 | 980 000 | 2 230 000. " * 72
# Un vrai tableau de programmes (credits AE/CP), comme ceux de l'incident.
TABLEAU_DE_PROGRAMMES = (
    "Les montants des autorisations d'engagement et des crédits de paiement du budget général "
    "ouverts sur les programmes concourant à la réalisation des objectifs assortis d'indicateurs "
    "sont fixés comme suit :\n| N° | CODE | Programme LIBELLE | OBJECTF | INDICATEUR | AE | CP |\n"
    "|---|---|---|---|---|---|---|\n"
    "| 130 | 542 | AMELIORATION DE LA PROTECTION DU TRAVAIL | Promouvoir le travail décent | "
    "Proportion des travailleurs | 2 020 161 | 1 870 161 |\n"
)


class _VecteurDeLaQuestion:
    """Le fournisseur d'embeddings, double : toute question a le meme vecteur."""

    TASK_QUERY = EmbeddingService.TASK_QUERY
    TASK_DOCUMENT = EmbeddingService.TASK_DOCUMENT

    async def generate_embedding_async(self, texte, task_type=None):
        return np.array(VECTEUR_DE_LA_QUESTION, dtype=np.float32)


@pytest.fixture
async def incident(db_session, category_ids):
    """
    Le Code du travail, que seul le plein texte trouve (ses articles sont loin
    de la question pour le volet semantique), et une loi de finances dont les
    40 tableaux sont ses plus proches voisins — a 0,15 environ : un volet
    semantique TRES FAIBLE. Les articles 35 a 39 du Code sont ranges sous
    « 35.1 », comme dans le corpus. Le Code penal porte « 74 » deux fois a la
    suite (74 et 74-1, comme en base) ; la Constitution, son article 6 reecrit
    depuis.
    """
    from sqlalchemy import text as sql_text

    from app.models.law import Article, Law
    from app.services.legal_domain_classifier import FINANCES, TRAVAIL
    from app.services.search_vectors import REINDEX_ARTICLES_SQL, REINDEX_LAWS_SQL

    articles_du_code = {
        "35.1": "(1) Pendant la durée du préavis, les obligations demeurent, conformément "
                "à l'article 36 ci-dessous. Article 37.- (1) "
                "En cas de licenciement abusif ou de rupture du contrat du fait de "
                "l'employeur, le travailleur a droit à une indemnité de licenciement. "
                "Article 39.- (1) Toute rupture abusive du contrat peut donner lieu à des "
                "dommages-intérêts.",
        "25": "(1) Le contrat de travail peut être conclu pour une durée déterminée ou "
              "indéterminée.",
    }
    db_session.add_all([
        Law(id=CODE_DU_TRAVAIL_ID, reference="LOI-1992-007", title=TITRE_CODE_DU_TRAVAIL,
            content="TITRE PREMIER — DISPOSITIONS GÉNÉRALES.", type="loi", language="fr",
            status="published", category_id=category_ids[TRAVAIL],
            publication_date=date(1992, 8, 14)),
        Law(id=LOI_DE_FINANCES_ID, reference="PRC-9449", title=TITRE_LOI_DE_FINANCES,
            content=TABLEAU, type="loi", language="fr", status="published",
            category_id=category_ids[FINANCES], publication_date=date(2022, 12, 27)),
        Law(id=COLLECTIVITES_ID, reference="PRC-8036",
            title="Loi 2019/024 du 24 décembre 2019 portant code général des collectivités "
                  "territoriales décentralisées",
            content="Dispositions générales.", type="loi", language="fr", status="published",
            publication_date=date(2019, 12, 24)),
        Law(id=CODE_PENAL_ID, reference="PRC-4722",
            title="Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal",
            content="Livre premier.", type="loi", language="fr", status="published",
            publication_date=date(2016, 7, 12)),
        Law(id=CONSTITUTION_ID, reference="PRC-6283",
            title="Loi N°96/06 du 18 janvier 1996 portant révision de la Constitution du 02 juin "
                  "1972.",
            content="Préambule.", type="loi", language="fr", status="published",
            publication_date=date(1996, 1, 18)),
    ])
    await db_session.flush()

    question = np.asarray(VECTEUR_DE_LA_QUESTION)
    bruit = np.random.default_rng(11)
    db_session.add_all([
        Article(id=19920000 + i, law_id=CODE_DU_TRAVAIL_ID, number=numero, content=contenu,
                order=i, embedding=_aleatoire(100 + i))
        for i, (numero, contenu) in enumerate(articles_du_code.items())
    ] + [
        Article(id=20230000 + i, law_id=LOI_DE_FINANCES_ID, number=f"76.{i}", content=TABLEAU,
                order=i, embedding=_unitaire(question + 0.2 * bruit.normal(size=question.size)))
        for i in range(1, 41)
    ] + [
        Article(id=20190005, law_id=COLLECTIVITES_ID, number="5",
                content="Les collectivités territoriales sont des personnes morales.",
                order=5, embedding=_aleatoire(55)),
        Article(id=20160074, law_id=CODE_PENAL_ID, number="74", order=92,
                content="Peine et responsabilité (1) Aucune peine ne peut être prononcée qu'à "
                        "l'encontre d'une personne pénalement responsable.",
                embedding=_aleatoire(74)),
        Article(id=20160075, law_id=CODE_PENAL_ID, number="74", order=93,
                content="1.- Personnes morales pénalement responsables.",
                embedding=_aleatoire(75)),
        Article(id=19960006, law_id=CONSTITUTION_ID, number="6", order=6,
                content="(2) Le Président de la République est élu pour un mandat de sept (7) "
                        "ans renouvelable une fois.",
                embedding=_aleatoire(66)),
    ])
    await db_session.commit()
    await db_session.execute(sql_text(REINDEX_LAWS_SQL))
    await db_session.execute(sql_text(REINDEX_ARTICLES_SQL))
    await db_session.commit()


@pytest.fixture
async def tableaux_de_programmes(db_session, incident):
    """Les tableaux de la loi de finances deviennent de vrais tableaux de programmes."""
    from sqlalchemy import text as sql_text

    await db_session.execute(
        sql_text("UPDATE articles SET content = :c WHERE law_id = :l"),
        {"c": TABLEAU_DE_PROGRAMMES, "l": LOI_DE_FINANCES_ID},
    )
    await db_session.commit()


def _chat_reel(db_session):
    service = RAGService(db_session)
    # Aucun appel au modele : seule la recuperation est en jeu.
    service.llm = None
    service.search_service.use_cache = False
    service.search_service.embedding_service = _VecteurDeLaQuestion()
    return service


class TestParLaVraieRecherche:

    @pytest.mark.asyncio
    async def test_l_incident_le_code_du_travail_remplace_les_tableaux(self, incident, db_session):
        chat = _chat_reel(db_session)

        contexte = await chat._retrieve_chunks(QUESTION, "fr")
        chat._build_prompt(QUESTION, contexte, [], "citoyen")

        etat = chat._recuperation
        assert etat.chemin == CHEMIN_EXACT and etat.mode == "faible"
        assert etat.similarite < SEUIL_FR
        # La recherche rendait les tableaux ; le modele lit le Code du travail.
        assert {c.law_id for c in etat.hybride} >= {LOI_DE_FINANCES_ID}
        assert [(c.law_id, c.number) for c in contexte] == [(CODE_DU_TRAVAIL_ID, "35.1")]
        assert [c.number for c in chat._extraits_du_prompt] == ["35.1"]

    @pytest.mark.asyncio
    async def test_les_tableaux_de_programmes_ne_sont_pas_lus(
        self, tableaux_de_programmes, db_session
    ):
        """
        De vrais tableaux AE/CP, plus proches de la question que tout le
        reste : retires des deux volets avant la fusion. Le modele lit le Code
        du travail ; aucun tableau, meme sous le seuil.
        """
        chat = _chat_reel(db_session)

        contexte = await chat._retrieve_chunks(QUESTION, "fr")

        assert LOI_DE_FINANCES_ID not in {c.law_id for c in chat._recuperation.hybride}
        assert chat._recuperation.tableaux_exclus > 0
        assert [c.law_id for c in contexte] == [CODE_DU_TRAVAIL_ID]

    @pytest.mark.asyncio
    async def test_une_question_sur_le_budget_les_garde(self, tableaux_de_programmes, db_session):
        chat = _chat_reel(db_session)

        await chat._retrieve_chunks(
            "Quels crédits de paiement la loi de finances ouvre-t-elle pour le travail ?", "fr"
        )

        assert chat._recuperation.budgetaire is True
        assert LOI_DE_FINANCES_ID in {c.law_id for c in chat._recuperation.hybride}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "Quel est le montant prévu pour l'indemnité de licenciement ?",
        "Quelle somme est prévue en cas de licenciement abusif ?",
        "How much compensation is allocated for unfair dismissal?",
    ])
    async def test_un_montant_prevu_n_est_pas_une_question_budgetaire(
        self, tableaux_de_programmes, db_session, question
    ):
        """
        Revue du 29/10/2026 (tour 3) : « montant prévu », « allocated »
        faisaient garder les tableaux AE/CP (7 sur 8 pour la premiere, dans le
        leurre). Ils sont retires.
        """
        chat = _chat_reel(db_session)

        await chat._retrieve_chunks(question, "fr")

        assert chat._recuperation.budgetaire is False
        assert LOI_DE_FINANCES_ID not in {c.law_id for c in chat._recuperation.hybride}
        assert chat._recuperation.tableaux_exclus > 0

    @pytest.mark.asyncio
    async def test_la_relance_lit_l_article_dans_le_texte_du_tour_precedent(
        self, incident, db_session
    ):
        """
        « Et l'article 37 ? » apres une question sur l'article 39 du Code du
        travail : l'article 37 de CE code, lu par son en-tete dans « 35.1 ».
        """
        chat = _chat_reel(db_session)
        historique = _historique("Que dit l'article 39 du Code du travail ?",
                                 reponse="L'article 39 prévoit des dommages-intérêts.")

        contexte = await chat._retrieve_chunks("Et l'article 37 ?", "fr", history=historique)

        assert (contexte[0].law_id, contexte[0].number, contexte[0].source) == (
            CODE_DU_TRAVAIL_ID, "35.1", SOURCE_ARTICLE_NOMME,
        )
        etat = chat._recuperation
        assert (etat.relance, etat.texte_herite) == (True, "LOI-1992-007")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "Que dit l'article 37 de ce code ?", "Que dit l'article 37 du même code ?",
        "Que dit l'article 37 dudit code ?", "What does Article 37 of this law say?",
    ])
    async def test_un_renvoi_au_texte_lit_l_article_du_tour_precedent(
        self, incident, db_session, question
    ):
        """Revue du tour 4 : « l'article 37 de ce code » est celui du Code du travail, lu en tete."""
        chat = _chat_reel(db_session)
        historique = _historique("Que dit l'article 39 du Code du travail ?",
                                 reponse="L'article 39 prévoit des dommages-intérêts.")

        contexte = await chat._retrieve_chunks(question, "fr", history=historique)

        assert (contexte[0].law_id, contexte[0].number, contexte[0].source) == (
            CODE_DU_TRAVAIL_ID, "35.1", SOURCE_ARTICLE_NOMME,
        )
        assert (chat._recuperation.relance, chat._recuperation.texte_herite) == (
            True, "LOI-1992-007",
        )

    @pytest.mark.asyncio
    async def test_un_renvoi_d_une_autre_nature_n_herite_pas(self, incident, db_session):
        """
        Revue du tour 5 : « l'article 37 de ce décret » apres une question sur
        le Code du travail. Son intitule, lu en base, est celui d'une loi et
        d'un code : rien d'herite, et « décret » reste dans les mots-cles.
        """
        chat = _chat_reel(db_session)
        historique = _historique("Que dit l'article 39 du Code du travail ?",
                                 reponse="L'article 39 prévoit des dommages-intérêts.")

        assert await chat._intitule_du_texte(CODE_DU_TRAVAIL_ID) == TITRE_CODE_DU_TRAVAIL
        contexte = await chat._retrieve_chunks(
            "Que dit l'article 37 de ce décret ?", "fr", history=historique
        )

        assert all(c.source != SOURCE_ARTICLE_NOMME for c in contexte)
        etat = chat._recuperation
        assert (etat.relance, etat.texte_herite, etat.article_nomme) == (True, None, None)
        assert etat.mots_cles.split()[0] == "décret"

    @pytest.mark.asyncio
    async def test_sur_le_repli_seuls_les_textes_confirmes(self, incident, db_session):
        """
        La recherche exacte depasse son budget : le volet passe par l'index.
        Le chat n'en croit que ce que le plein texte confirme.
        """
        chat = _chat_reel(db_session)
        chat.search_service.BUDGET_RECHERCHE_EXACTE_MS = 1

        contexte = await chat._retrieve_chunks(QUESTION, "fr")

        etat = chat._recuperation
        assert (etat.chemin, etat.mode) == (CHEMIN_REPLI_HNSW, "faible")
        assert [(c.law_id, c.number) for c in contexte] == [(CODE_DU_TRAVAIL_ID, "35.1")]

    @pytest.mark.asyncio
    async def test_une_question_hors_sujet_est_refusee(self, incident, db_session):
        chat = _chat_reel(db_session)

        assert await chat._retrieve_chunks("Quelle est la recette du ndolé ?", "fr") == []
        assert chat._recuperation.refus is True

    @pytest.mark.asyncio
    async def test_l_article_nomme_est_lu_par_son_en_tete(self, incident, db_session):
        """
        « l'article 37 du Code du travail » : aucun article ne porte le numero 37,
        le morceau « 35.1 » porte son en-tete. Lu sans les deux volets, et il
        suffit a la barriere.
        """
        chat = _chat_reel(db_session)

        contexte = await chat._retrieve_chunks("Que dit l'article 37 du Code du travail ?", "fr")

        assert (contexte[0].law_id, contexte[0].number, contexte[0].source) == (
            CODE_DU_TRAVAIL_ID, "35.1", SOURCE_ARTICLE_NOMME,
        )
        assert chat._recuperation.article_nomme == "LOI-1992-007 art. 35.1"
        assert chat._recuperation.refus is False

    @pytest.mark.asyncio
    async def test_l_article_nomme_par_le_numero_de_sa_loi(self, incident, db_session):
        chat = _chat_reel(db_session)

        nommes = await chat._article_nomme("Que dit l'article 25 de la loi n° 92/007 ?")

        assert [(c.law_id, c.number) for c in nommes] == [(CODE_DU_TRAVAIL_ID, "25")]

    @pytest.mark.asyncio
    async def test_le_numero_exact_passe_avant_l_en_tete(self, incident, db_session):
        chat = _chat_reel(db_session)

        nommes = await chat._article_nomme("What does Article 25 of the Labour Code say?")

        assert [(c.law_id, c.number) for c in nommes] == [(CODE_DU_TRAVAIL_ID, "25")]

    @pytest.mark.asyncio
    async def test_74_et_74_1_du_code_penal(self, incident, db_session):
        chat = _chat_reel(db_session)

        nommes = await chat._article_nomme("Que dit l'article 74 du Code pénal ?")

        assert [c.article_id for c in nommes] == [20160074, 20160075]

    @pytest.mark.asyncio
    async def test_un_article_reecrit_depuis_n_est_pas_ajoute(self, incident, db_session):
        """L'article 6 de la Constitution de 1996, reecrit en 2008 puis en 2026."""
        chat = _chat_reel(db_session)

        assert await chat._article_nomme("Que dit l'article 6 de la Constitution ?") == []

    @pytest.mark.asyncio
    async def test_un_renvoi_n_est_pas_un_en_tete(self, incident, db_session):
        """« l'article 37 » dans un autre article n'en fait pas l'article 37."""
        chat = _chat_reel(db_session)

        assert await chat._article_nomme("Que dit l'article 36 du Code du travail ?") == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        # resolve_law_by_hint rattache « Code général des impôts », absent, au
        # code general des collectivites : son article 5 n'est pas celui-la.
        "Que dit l'article 5 du Code général des impôts ?",
        # La table ne connait pas ce nom : rien.
        "Que dit l'article 5 de la loi sur les collectivités ?",
        # Aucun texte publie ne porte ce numero.
        "Que dit l'article 5 de la loi n° 2019/999 ?",
    ])
    async def test_un_texte_que_rien_ne_designe_n_est_pas_remplace(
        self, incident, db_session, question
    ):
        chat = _chat_reel(db_session)

        assert await chat._article_nomme(question) == []

    @pytest.mark.asyncio
    async def test_servi_par_le_cache_le_contexte_est_le_meme(self, incident, db_session):
        """Une question d'accueil reposee : volets relus du cache, meme contexte, meme barriere."""
        premier = _chat_reel(db_session)
        premier.search_service.use_cache = True
        direct = await premier._retrieve_chunks(QUESTION, "fr")
        await ecritures_en_attente(10)

        chat = _chat_reel(db_session)
        chat.search_service.use_cache = True
        chat.search_service.text_chunks = AsyncMock(side_effect=AssertionError("recalcul"))
        depuis_le_cache = await chat._retrieve_chunks(QUESTION, "fr")

        assert [c.fusion_key for c in depuis_le_cache] == [c.fusion_key for c in direct]
        assert chat._recuperation.mode == "faible"
