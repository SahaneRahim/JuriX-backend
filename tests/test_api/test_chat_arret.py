"""
Bouton « Arrêter » du chat : la deconnexion du lecteur arrete la generation.

Le front coupe sa requete (AbortController). Ce qui doit suivre, cote serveur :
- le flux du modele est FERME aussitot, et aucun autre appel n'est fait ;
- ce que le lecteur a lu est enregistre, avec une mention qui dit que la
  reponse est incomplete : la question suivante garde son contexte, et une
  conversation neuve ne reste pas vide dans la liste du compte.

Les deux voies par lesquelles un serveur ASGI signale la deconnexion sont
jouees : le message `http.disconnect` (uvicorn, ASGI 2.3 : Starlette annule le
flux) et l'envoi qui echoue (ASGI 2.4). Le modele est une doublure qui ecrirait
sans fin : s'il n'etait pas arrete, le test ne finirait pas.

Author: JuriX Team
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.api.routes import rag as route
from app.main import app
from app.models.conversation import Conversation, Message
from app.schemas.rag import RAGRequest
from app.schemas.search import ChunkResult
from app.services.intent_classifier import IntentResult
from app.services.llm import FIN_ARRETEE, mention_de_fin
from app.services.rag_service import RAGService

MENTION_ARRETEE = mention_de_fin(FIN_ARRETEE, "fr")


class _ModeleSansFin:
    """Doublure du modele : ecrit sans fin, compte ce qu'il produit, dit quand on le ferme."""

    def __init__(self):
        self.appels = 0
        self.morceaux = 0
        self.ferme = asyncio.Event()

    def generate_stream(self, **kwargs):
        self.appels += 1

        async def flux():
            try:
                for i in range(1_000_000):
                    await asyncio.sleep(0.002)
                    self.morceaux += 1
                    yield f"mot{i} "
            finally:
                self.ferme.set()

        return flux()

    async def generate(self, **kwargs):
        raise AssertionError("aucun appel hors flux n'est attendu")


def _service(base) -> RAGService:
    with patch("app.services.rag_service.get_llm_service", return_value=None):
        service = RAGService(base)
    service.llm = _ModeleSansFin()
    service.search_service = MagicMock()
    service.search_service.search = AsyncMock(return_value=MagicMock(
        results=[], search_time_ms=1,
        chunks=[ChunkResult(
            article_id=None, law_id=1, number="37", content="Le travailleur a droit à une indemnité.",
            reference="PRC-1", law_title="Code du travail", relevance_score=0.9,
        )],
    ))
    return service


def _base_doublee():
    base = AsyncMock()
    base.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    base.add = MagicMock()
    return base


@pytest.fixture(autouse=True)
def routage_juridique():
    with patch(
        "app.services.rag_service.classify_intent",
        new=AsyncMock(return_value=IntentResult("juridique", 1.0, "test")),
    ):
        yield


@contextlib.contextmanager
def _service_de_la_route(service):
    app.dependency_overrides[route.get_rag_service] = lambda: service
    try:
        yield
    finally:
        app.dependency_overrides.pop(route.get_rag_service, None)


async def _lecteur_qui_part(couper_apres: int, spec: str = "2.3"):
    """
    POST /api/v1/rag/ask/stream joue en ASGI brut ; le lecteur part apres
    `couper_apres` morceaux de texte. Rend les evenements recus.
    """
    parti = asyncio.Event()
    recus = []
    corps_envoye = False

    async def receive():
        nonlocal corps_envoye
        if not corps_envoye:
            corps_envoye = True
            corps = json.dumps({"question": "Quelle indemnité de licenciement ?", "language": "fr"})
            return {"type": "http.request", "body": corps.encode(), "more_body": False}
        await parti.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if parti.is_set() and spec >= "2.4":
            raise OSError("le lecteur est parti")
        if message["type"] != "http.response.body":
            return
        for bloc in message.get("body", b"").decode().split("\n\n"):
            if bloc.startswith("data: "):
                recus.append(json.loads(bloc[len("data: "):]))
        if sum(1 for e in recus if e.get("chunk")) >= couper_apres:
            parti.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/api/v1/rag/ask/stream",
        "raw_path": b"/api/v1/rag/ask/stream", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
        "client": ("127.0.0.1", 50000), "server": ("test", 80),
    }
    # Un serveur ASGI 2.4 voit l'envoi echouer : l'erreur remonte jusqu'a lui.
    with contextlib.suppress(OSError, Exception):
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return recus


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.asyncio
async def test_la_deconnexion_arrete_le_modele_sur_le_champ(spec):
    base = _base_doublee()
    service = _service(base)

    with _service_de_la_route(service):
        recus = await _lecteur_qui_part(couper_apres=3, spec=spec)

    modele = service.llm
    assert [e["progression"]["etape"] for e in recus if "progression" in e] == [
        "recherche", "textes_trouves", "redaction",
    ]
    # Ferme AVANT la fin de la requete : ni ramasse-miettes, ni delai.
    assert modele.ferme.is_set(), "le flux du modele est reste ouvert"
    assert modele.appels == 1
    produits = modele.morceaux
    assert produits <= 3 + 2, f"{produits} morceaux produits pour 3 lus"
    await asyncio.sleep(0.05)
    assert modele.morceaux == produits, "le modele a continue d'ecrire"


@pytest.mark.asyncio
async def test_la_reponse_arretee_est_gardee_avec_sa_mention():
    base = _base_doublee()
    service = _service(base)

    with _service_de_la_route(service):
        recus = await _lecteur_qui_part(couper_apres=3)

    lu = "".join(e["chunk"] for e in recus)
    gardees = [
        c.args[0] for c in base.add.call_args_list
        if isinstance(c.args[0], Message) and c.args[0].role == "assistant"
    ]
    assert len(gardees) == 1
    contenu = gardees[0].content
    assert contenu.endswith(MENTION_ARRETEE)
    # Ce que le lecteur a lu, au plus un morceau de plus en vol.
    assert contenu.startswith(lu)
    # Le flux n'a jamais eu de dernier evenement : l'identifiant de la
    # conversation est venu avec la progression.
    assert all(not e.get("done") for e in recus)
    assert {e["session_id"] for e in recus if "progression" in e} == {
        base.add.call_args_list[0].args[0].session_id
    }


@pytest.mark.asyncio
async def test_le_service_ferme_le_modele_quand_on_ferme_son_flux():
    """Sans la route : fermer le flux du service ferme celui du modele, et garde le lu."""
    base = _base_doublee()
    service = _service(base)
    requete = RAGRequest(question="Quelle indemnité de licenciement ?", language="fr")

    flux = service.ask_stream(requete)
    textes = []
    async for evenement in flux:
        if json.loads(evenement)["chunk"]:
            textes.append(json.loads(evenement)["chunk"])
            if len(textes) == 2:
                break
    await flux.aclose()

    assert service.llm.ferme.is_set()
    assert service.llm.appels == 1
    [gardee] = [
        c.args[0] for c in base.add.call_args_list
        if isinstance(c.args[0], Message) and c.args[0].role == "assistant"
    ]
    assert gardee.content == "".join(textes) + MENTION_ARRETEE


@pytest.mark.asyncio
async def test_rien_n_est_garde_si_rien_n_a_ete_ecrit():
    """Arretee avant le premier mot : pas de reponse vide dans l'historique."""
    base = _base_doublee()
    service = _service(base)
    requete = RAGRequest(question="Quelle indemnité de licenciement ?", language="fr")

    flux = service.ask_stream(requete)
    async for evenement in flux:
        if json.loads(evenement).get("progression", {}).get("etape") == "redaction":
            break
    await flux.aclose()

    assert not any(
        isinstance(c.args[0], Message) for c in base.add.call_args_list
    )


@pytest.mark.asyncio
async def test_l_enregistrement_resiste_a_l_annulation_sur_la_vraie_base(db_session):
    """
    La deconnexion ANNULE la tache (ASGI 2.3, la voie d'uvicorn), et Starlette
    renouvelle l'annulation a chaque attente : l'echange n'est enregistre que
    s'il est a l'abri. Verifie sur PostgreSQL, ou chaque ecriture attend.
    """
    service = _service(db_session)

    with _service_de_la_route(service):
        recus = await _lecteur_qui_part(couper_apres=3)

    session_id = next(e["session_id"] for e in recus if "progression" in e)
    conversation = (await db_session.execute(
        select(Conversation).where(Conversation.session_id == session_id)
    )).scalar_one()
    messages = (await db_session.execute(
        select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id)
    )).scalars().all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content.endswith(MENTION_ARRETEE)
    assert conversation.title == "Quelle indemnité de licenciement ?"
