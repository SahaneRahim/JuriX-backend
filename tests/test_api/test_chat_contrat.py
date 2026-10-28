"""
Le contrat du chat tel que le front le recoit, par HTTP : evenements SSE de
POST /api/v1/rag/ask/stream et JSON de POST /api/v1/rag/ask.

Les tests du service verifient les valeurs ; ceux-ci epinglent la FORME au
bout de la chaine (route, response_model de FastAPI, en_json du flux) : les
cles que le front nouveau lira, et l'absence de tout changement pour le front
deploye, qui ne lit que `chunk`, `done`, `error`, `sources`, `session_id` et
`persona`. Tout est double : base, recherche, modele ; aucun appel reseau.

Author: JuriX Team
"""

import contextlib
import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.routes import rag as route
from app.main import app
from app.schemas.search import ChunkResult
from app.services.intent_classifier import IntentResult
from app.services.rag_service import RAGService

CODE_DU_TRAVAIL = "Loi n° 92/007 du 14 août 1992 portant Code du travail"
REPONSE = (
    "📌 L'article 37 du Code du travail prévoit une indemnité ; "
    "l'article 312 du Code pénal ne s'applique pas."
)

CLES_CITATION = {
    "debut", "fin", "texte_cite", "law_id", "article", "article_id", "reference", "titre",
    "date_du_texte", "statut",
}
CLES_STATUT = {"etat", "liens", "lignes"}
CLES_INTROUVABLE = {"debut", "fin", "texte_cite", "raison"}
CLES_SOURCE_CITEE = {
    "law_id", "law_reference", "law_title", "article_number", "article_id", "excerpt",
    "relevance_score", "date_du_texte", "statut",
}
CLES_TEXTE_CONSULTE = {
    "law_id", "law_reference", "law_title", "date_du_texte", "articles", "statut",
}


def _service():
    base = AsyncMock()
    base.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    base.add = MagicMock()
    base.begin_nested = MagicMock(return_value=MagicMock())
    with patch("app.services.rag_service.get_llm_service", return_value=None):
        service = RAGService(base)
    service.search_service = MagicMock()
    service.search_service.search = AsyncMock(return_value=MagicMock(
        results=[], search_time_ms=1,
        chunks=[
            ChunkResult(article_id=371, law_id=3, number="37", content="Le travailleur a droit…",
                        reference="PRC-3", law_title=CODE_DU_TRAVAIL,
                        publication_date=date(1992, 8, 14), relevance_score=0.9),
            ChunkResult(article_id=81, law_id=8, number="8", content="Les modalités…",
                        reference="PRC-8", law_title="Décret N°2018/9387/CAB/PM",
                        relevance_score=0.6),
        ],
    ))
    service.llm = MagicMock()

    def flux(**kwargs):
        async def generateur():
            for morceau in (REPONSE[:30], REPONSE[30:]):
                yield morceau
        return generateur()

    service.llm.generate_stream = MagicMock(side_effect=flux)
    service.llm.generate = AsyncMock(return_value={"response": REPONSE})
    return service


@pytest.fixture
def client():
    with patch(
        "app.services.rag_service.classify_intent",
        new=AsyncMock(return_value=IntentResult("juridique", 1.0, "test")),
    ):
        app.dependency_overrides[route.get_rag_service] = _service
        try:
            yield TestClient(app)
        finally:
            app.dependency_overrides.pop(route.get_rag_service, None)


def _evenements(reponse):
    """Les evenements SSE : une ligne `data:` chacun, JSON d'un seul tenant."""
    blocs = [bloc for bloc in reponse.text.split("\n\n") if bloc.strip()]
    assert all(bloc.startswith("data: ") and "\n" not in bloc for bloc in blocs)
    return [json.loads(bloc[len("data: "):]) for bloc in blocs]


def _js_slice(texte, debut, fin):
    return texte.encode("utf-16-le")[2 * debut:2 * fin].decode("utf-16-le")


def _controler_les_champs(donnees, texte):
    [citation] = donnees["citations"]
    assert set(citation) == CLES_CITATION
    assert set(citation["statut"]) == CLES_STATUT
    assert citation["statut"]["etat"] is None  # rien de prouve : null, pas absent
    assert _js_slice(texte, citation["debut"], citation["fin"]) == citation["texte_cite"]
    assert (citation["law_id"], citation["article"], citation["article_id"]) == (3, "37", 371)
    assert citation["date_du_texte"] == "1992-08-14"

    [introuvable] = donnees["citations_introuvables"]
    assert set(introuvable) == CLES_INTROUVABLE
    assert introuvable["raison"] == "texte_non_consulte"
    assert _js_slice(texte, introuvable["debut"], introuvable["fin"]) == "article 312 du Code pénal"

    [cite] = donnees["articles_cites"]
    assert set(cite) == CLES_SOURCE_CITEE
    [consulte] = donnees["textes_consultes_non_cites"]
    assert set(consulte) == CLES_TEXTE_CONSULTE
    assert consulte["law_id"] == 8 and consulte["articles"] == ["8"]


def test_le_flux_par_http(client):
    reponse = client.post(
        "/api/v1/rag/ask/stream", json={"question": "Quelle indemnité de licenciement ?"}
    )

    assert reponse.status_code == 200
    evenements = _evenements(reponse)
    progressions = [e for e in evenements if "progression" in e]
    assert [p["progression"]["etape"] for p in progressions] == [
        "recherche", "textes_trouves", "redaction",
    ]
    for p in progressions:
        # Le front deploye n'y voit ni texte, ni fin, ni erreur.
        assert set(p) == {"chunk", "done", "session_id", "persona", "progression"}
        assert p["chunk"] == "" and p["done"] is False
    morceaux = [e for e in evenements if e.get("chunk")]
    assert all(set(e) == {"chunk", "done"} for e in morceaux)  # inchanges

    fin = evenements[-1]
    assert fin["done"] is True and "error" not in fin
    assert {"sources", "confidence", "session_id", "persona", "intent"} <= set(fin)
    assert [s["article_number"] for s in fin["sources"]] == ["37"]
    _controler_les_champs(fin, "".join(e["chunk"] for e in evenements))


def test_la_reponse_de_ask_par_http(client):
    reponse = client.post("/api/v1/rag/ask", json={"question": "Quelle indemnité de licenciement ?"})

    assert reponse.status_code == 200
    donnees = reponse.json()
    assert [s["article_number"] for s in donnees["sources"]] == ["37"]
    _controler_les_champs(donnees, donnees["answer"])


def test_une_reponse_ecrite_par_le_serveur_a_les_champs_vides(client):
    """Aucun resultat : les quatre champs sont la, vides, pour une forme constante."""
    with contextlib.ExitStack() as pile:
        pile.enter_context(patch.object(
            RAGService, "_retrieve_and_merge_context", AsyncMock(return_value=([], None))
        ))
        pile.enter_context(patch.object(RAGService, "_fallback_search", AsyncMock(return_value=[])))
        evenements = _evenements(client.post(
            "/api/v1/rag/ask/stream", json={"question": "Quelle indemnité de licenciement ?"}
        ))

    fin = evenements[-1]
    assert fin["done"] is True
    assert (fin["citations"], fin["citations_introuvables"], fin["articles_cites"],
            fin["textes_consultes_non_cites"]) == ([], [], [], [])
