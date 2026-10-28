"""
Le chat rend ses citations cliquables, signale les introuvables, et separe les
articles cites des textes seulement consultes.

Ce que le front recoit, et sur quoi il s'appuie :
- dans le dernier evenement (et dans la reponse de /ask) : `citations`,
  `citations_introuvables`, `articles_cites`, `textes_consultes_non_cites`,
  avec des indices EXACTS dans le texte qu'il a affiche.

Aucun appel reel : le modele, la recherche et la base sont doubles.

Author: JuriX Team
"""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.rag import RAGRequest
from app.schemas.search import ChunkResult
from app.services.intent_classifier import IntentResult
from app.services.rag_service import RAGService

CODE_DU_TRAVAIL = "Loi n° 92/007 du 14 août 1992 portant Code du travail"


def _extrait(article_id, law_id, numero, titre, reference, contenu="Contenu de l'article.",
             pertinence=0.8):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        reference=reference, law_title=titre, publication_date=date(1992, 8, 14),
        relevance_score=pertinence,
    )


@pytest.fixture
def extraits():
    return [
        _extrait(371, 3, "37", CODE_DU_TRAVAIL, "PRC-3",
                 "(1) En cas de rupture du contrat, le travailleur a droit à une indemnité."),
        _extrait(401, 3, "40", CODE_DU_TRAVAIL, "PRC-3", "Les délégués du personnel sont réunis."),
        _extrait(81, 8, "8", "Décret N°2018/9387/CAB/PM fixant les modalités", "PRC-8"),
    ]


@pytest.fixture
def base():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value=None)
    ))
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    # Un point de sauvegarde, comme AsyncSession.begin_nested : appel
    # synchrone, rendu utilisable par `async with`.
    session.begin_nested = MagicMock(return_value=MagicMock())
    return session


@pytest.fixture(autouse=True)
def routage_juridique():
    with patch(
        "app.services.rag_service.classify_intent",
        new=AsyncMock(return_value=IntentResult("juridique", 1.0, "test")),
    ) as double:
        yield double


def _service(base, extraits, morceaux=(), fin="STOP"):
    service = RAGService(base)
    service.llm = MagicMock()
    service.search_service = MagicMock()
    service.search_service.search = AsyncMock(
        return_value=MagicMock(results=[], chunks=list(extraits), search_time_ms=1)
    )

    def _flux(**kwargs):
        async def _generateur():
            for morceau in morceaux:
                yield morceau
            if kwargs.get("fin"):
                kwargs["fin"](fin)
        return _generateur()

    service.llm.generate_stream = MagicMock(side_effect=_flux)
    return service


def _requete(**champs):
    return RAGRequest(**{
        "question": "Quelles indemnités en cas de licenciement ?", "persona": "citoyen",
        "language": "fr", **champs,
    })


async def _evenements(service, requete):
    return [json.loads(e) async for e in service.ask_stream(requete)]


def _slice_js(texte, debut, fin):
    """`texte.slice(debut, fin)` en JavaScript : des unites UTF-16."""
    return texte.encode("utf-16-le")[2 * debut:2 * fin].decode("utf-16-le")


# ==================== CITATIONS DU DERNIER EVENEMENT ====================


class TestCitationsDuFlux:

    @pytest.mark.asyncio
    async def test_les_indices_designent_la_citation_dans_le_texte_affiche(self, base, extraits):
        """
        Le texte affiche est la concatenation des `chunk` recus, mention de fin
        comprise ; une citation a cheval sur deux morceaux reste entiere.
        """
        service = _service(
            base, extraits,
            ["📌 Vous avez droit à une indemnité (l'article 37 du Code ",
             "du travail) ; l'employeur réunit les délégués (l'article 40 du Code du travail)."],
            fin="MAX_TOKENS",
        )

        evenements = await _evenements(service, _requete())

        texte = "".join(e["chunk"] for e in evenements)
        fin = evenements[-1]
        assert [c["texte_cite"] for c in fin["citations"]] == [
            "article 37 du Code du travail", "article 40 du Code du travail",
        ]
        for citation in fin["citations"]:
            assert _slice_js(texte, citation["debut"], citation["fin"]) == citation["texte_cite"]
        assert fin["citations"][0]["article"] == "37"
        assert fin["citations"][0]["article_id"] == 371
        assert fin["citations"][0]["titre"] == CODE_DU_TRAVAIL
        assert fin["citations"][0]["date_du_texte"] == "1992-08-14"
        # Aucun lien prouve : un statut vide, ENTIER (etat null, pas absent).
        assert fin["citations"][0]["statut"] == {"etat": None, "liens": [], "lignes": []}

    @pytest.mark.asyncio
    async def test_une_citation_introuvable_est_signalee(self, base, extraits):
        service = _service(base, extraits, [
            "L'article 37 du Code du travail fixe l'indemnité. L'article 312 du Code pénal "
            "punit, et le barème est fixé par l'arrêté n° 016/MTPS/SG/CJ du 26 mai 1993.",
        ])

        evenements = await _evenements(service, _requete())

        texte = "".join(e["chunk"] for e in evenements)
        introuvables = evenements[-1]["citations_introuvables"]
        assert [(c["texte_cite"], c["raison"]) for c in introuvables] == [
            ("article 312 du Code pénal", "texte_non_consulte"),
            ("arrêté n° 016/MTPS/SG/CJ du 26 mai 1993", "texte_non_consulte"),
        ]
        for citation in introuvables:
            assert _slice_js(texte, citation["debut"], citation["fin"]) == citation["texte_cite"]
        # `sources` garde son contenu d'avant : la seule citation rattachee.
        assert [s["article_number"] for s in evenements[-1]["sources"]] == ["37"]

    @pytest.mark.asyncio
    async def test_articles_cites_et_textes_consultes_non_cites(self, base, extraits):
        service = _service(base, extraits, [
            "L'article 40 du Code du travail impose une réunion ; l'article 37 du Code du "
            "travail fixe l'indemnité, et l'article 40 du Code du travail le rappelle.",
        ])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert [(c["law_id"], c["article_number"]) for c in fin["articles_cites"]] == [
            (3, "40"), (3, "37"),
        ]
        assert fin["articles_cites"][0]["law_title"] == CODE_DU_TRAVAIL
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [8]
        assert fin["textes_consultes_non_cites"][0]["articles"] == ["8"]
        assert len(fin["citations"]) == 3

    @pytest.mark.asyncio
    async def test_la_source_de_repli_n_est_pas_une_citation(self, base, extraits):
        """
        `sources` garde la source de repli pour le front deploye ; les champs
        nouveaux la rangent parmi les textes consultes, non cites.
        """
        service = _service(base, extraits, ["Le licenciement ouvre droit à une indemnité."])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert fin["sources"], "la source de repli manquait"
        assert fin["articles_cites"] == [] and fin["citations"] == []
        assert {t["law_id"] for t in fin["textes_consultes_non_cites"]} == {3, 8}

    @pytest.mark.asyncio
    async def test_un_extrait_ecarte_du_contexte_n_a_pas_ete_consulte(self, base, monkeypatch):
        """
        Le budget du contexte ecarte les derniers extraits : le modele ne les a
        pas lus. Une citation qui les vise n'est portee par rien qu'il ait lu.
        """
        from app.services import prompts

        monkeypatch.setattr(prompts, "CONTEXT_MAX_CHARS", 1_000)
        long = "Texte. " * 200
        extraits = [
            _extrait(371, 3, "37", CODE_DU_TRAVAIL, "PRC-3", long),
            _extrait(81, 8, "8", "Décret N°2018/9387/CAB/PM fixant les modalités", "PRC-8", long),
        ]
        service = _service(base, extraits, ["Voir l'article 8 du Décret N°2018/9387/CAB/PM."])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert [c["texte_cite"] for c in fin["citations_introuvables"]] == [
            "article 8 du Décret N°2018/9387/CAB/PM",
        ]
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [3]

    @pytest.mark.asyncio
    async def test_le_statut_prouve_accompagne_la_citation(self, base):
        """
        Constitution, article 14 : reecrit par la loi 2008/001 (PRC-6284). Le
        lien vers ce texte porte son identifiant, lu en une requete.
        """
        constitution = _extrait(141, 6283, "14", "Constitution du 18 janvier 1996", "PRC-6283")
        service = _service(base, [constitution], [
            "Selon l'article 14 de la Constitution, modifié par la Loi N°2008/001 du 14 avril 2008.",
        ])
        base.execute = AsyncMock(side_effect=lambda *a, **k: MagicMock(
            scalar_one_or_none=MagicMock(return_value=None),
            all=MagicMock(return_value=[("PRC-6284", 6284)]),
        ))

        evenements = await _evenements(service, _requete())

        texte = "".join(e["chunk"] for e in evenements)
        article, loi = evenements[-1]["citations"]
        assert article["statut"]["etat"] == "modifie"
        modification = next(
            lien for lien in article["statut"]["liens"] if lien["par_reference"] == "PRC-6284"
        )
        assert modification["par_law_id"] == 6284
        assert modification["par_date"] == "2008-04-14"
        # La loi nommee par la ligne de statut : liee, pas signalee.
        assert loi["law_id"] == 6284 and loi["article"] is None
        assert loi["titre"] == "Loi N°2008/001 du 14 avril 2008"
        assert _slice_js(texte, loi["debut"], loi["fin"]) == "Loi N°2008/001 du 14 avril 2008"
        assert evenements[-1]["citations_introuvables"] == []


# ==================== IDENTIFIANTS, SUR LA VRAIE BASE ====================


class TestIdentifiantsSurLaVraieBase:
    """
    Le lien vers le texte qui modifie porte l'identifiant lu dans `laws`. Et
    un echec de cette lecture, une vraie erreur SQL, ne fait pas sortir de
    l'historique une reponse deja affichee : l'echec reste dans son point de
    sauvegarde. Un rollback complet expirait la conversation, et son
    enregistrement echouait ensuite (MissingGreenlet).
    """

    REPONSE = "Selon l'article 14 de la Constitution, modifié par la Loi N°2008/001 du 14 avril 2008."

    def _service_sur(self, db_session):
        constitution = _extrait(141, 6283, "14", "Constitution du 18 janvier 1996", "PRC-6283")
        return _service(db_session, [constitution], [self.REPONSE])

    async def _messages(self, db_session, session_id):
        from sqlalchemy import select

        from app.models.conversation import Conversation, Message

        return (await db_session.execute(
            select(Message).join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.session_id == session_id).order_by(Message.id)
        )).scalars().all()

    @pytest.mark.asyncio
    async def test_l_identifiant_vient_de_la_table_des_textes(self, db_session):
        from app.models.law import Law

        loi = Law(reference="PRC-6284", title="Loi N°2008/001 du 14 avril 2008", type="loi",
                  content="Article 1er. La Constitution est modifiée.")
        db_session.add(loi)
        await db_session.commit()

        evenements = await _evenements(self._service_sur(db_session), _requete())

        article, nommee = evenements[-1]["citations"]
        [lien] = [lien for lien in article["statut"]["liens"] if lien["par_reference"] == "PRC-6284"]
        assert lien["par_law_id"] == loi.id
        assert nommee["law_id"] == loi.id and nommee["article"] is None

    @pytest.mark.asyncio
    async def test_un_echec_de_cette_lecture_ne_perd_pas_l_echange(self, db_session, monkeypatch):
        from sqlalchemy import text

        from app.services import rag_service as module

        monkeypatch.setattr(
            module, "_requete_des_identifiants",
            lambda references: text("SELECT reference, id FROM table_qui_n_existe_pas"),
        )

        evenements = await _evenements(self._service_sur(db_session), _requete())

        fin = evenements[-1]
        assert fin["done"] is True and "error" not in fin
        # Sans identifiant : le lien reste sans adresse, la loi nommee n'est ni
        # liee ni signalee.
        [article] = fin["citations"]
        assert all(lien["par_law_id"] is None for lien in article["statut"]["liens"])
        assert fin["citations_introuvables"] == []
        messages = await self._messages(db_session, fin["session_id"])
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].content == self.REPONSE


# ==================== REPONSE SANS FLUX ====================


class TestReponseSansFlux:

    @pytest.mark.asyncio
    async def test_ask_rend_les_memes_champs(self, base, extraits):
        service = _service(base, extraits)
        service.llm.generate = AsyncMock(return_value={
            "response": "L'article 37 du Code du travail fixe l'indemnité ; voir l'article 312 du Code pénal.",
        })

        reponse = await service.ask(_requete())

        assert [c.texte_cite for c in reponse.citations] == ["article 37 du Code du travail"]
        assert reponse.answer[reponse.citations[0].debut:reponse.citations[0].fin] == (
            "article 37 du Code du travail"
        )
        assert [c.raison for c in reponse.citations_introuvables] == ["texte_non_consulte"]
        assert [s.article_number for s in reponse.articles_cites] == ["37"]
        assert [t.law_id for t in reponse.textes_consultes_non_cites] == [8]
        assert [s.article_number for s in reponse.sources] == ["37"]

    @pytest.mark.asyncio
    async def test_la_mention_de_fin_ne_decale_rien(self, base, extraits):
        """La mention d'une reponse tronquee suit le texte : les indices restent justes."""
        service = _service(base, extraits)
        service.llm.generate = AsyncMock(return_value={
            "response": "Voir l'article 40 du Code du travail", "tronquee": True,
        })

        reponse = await service.ask(_requete())

        [citation] = reponse.citations
        assert reponse.answer[citation.debut:citation.fin] == "article 40 du Code du travail"
        assert reponse.answer.endswith(")*")
