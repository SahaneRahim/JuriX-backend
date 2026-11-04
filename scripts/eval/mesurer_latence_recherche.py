"""
Latence de la recherche du chat : la recherche semantique EXACTE seule (la
requete SQL), comparee a l'index HNSW, puis la recuperation entiere du chat
(RAGService._retrieve_chunks : plein texte, volet exact, regles du chat),
seule et sous 4 ou 8 questions simultanees.

Pourquoi. La recherche exacte compare la question a tous les articles : son
cout croit avec la charge, et au-dela de SearchService.BUDGET_RECHERCHE_EXACTE_MS
(BUDGET_RECHERCHE_EXACTE_A_FROID_MS pour la premiere recherche d'une base presumee
froide) le chat repasse par l'index (repli). Revue du 29/10/2026 : « 111 ms seule,
259 ms de mediane a 4 questions simultanees, 428 ms a 8 ». Ce script le
remesure sur la base de .env, sans appel au fournisseur pour une question
deja vectorisee (--vecteurs, le fichier de evaluer_contexte_chat.py).

Lecture seule : ni cache, ni journal de recherche, aucun appel au modele.

Usage (depuis la racine du depot) :
    python -m scripts.eval.mesurer_latence_recherche \\
        --vecteurs data/fiabilite/tmp/tour3/vecteurs.json --questions 40 \\
        --sortie data/eval_runs/latence_recherche.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

import numpy as np

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.database import engine as moteur
from app.schemas.search import SearchFilters
from app.services import search_service as module_recherche
from app.services.rag_service import RAGService
from app.services.search_service import SearchService

JEU = RACINE / "tests" / "fixtures" / "eval" / "retrieval_eval_v1.json"


class _VecteursEnMemoire:
    """Le fournisseur d'embeddings, servi par les vecteurs du fichier."""

    def __init__(self, vecteurs: Dict[str, Any]) -> None:
        self._vecteurs = vecteurs
        self.TASK_QUERY = "query"

    async def generate_embedding_async(self, texte: str, task_type: Any = None) -> Any:
        return self._vecteurs[texte]


def _stats(valeurs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not valeurs:
        return {"n": 0, "mediane": None, "p90": None, "max": None}
    ordonnees = sorted(valeurs)
    return {
        "n": len(ordonnees),
        "mediane": round(statistics.median(ordonnees), 1),
        "p90": round(ordonnees[min(len(ordonnees) - 1, int(0.9 * len(ordonnees)))], 1),
        "max": round(ordonnees[-1], 1),
    }


async def _requete_seule(vecteur: Any, exacte: bool) -> float:
    """Duree (ms) de la seule requete du volet semantique, sur une session neuve."""
    async with AsyncSessionLocal() as session:
        service = SearchService(session, use_cache=False, semantique_exacte=exacte)
        debut = time.perf_counter()
        if exacte:
            await service._lire_vecteurs_exacts(
                session, np.asarray(vecteur), SearchFilters(status="published"), 32, 0, False
            )
        else:
            async with session.begin_nested():
                await service._lire_vecteurs(
                    session, np.asarray(vecteur), SearchFilters(status="published"), 32, 0
                )
        duree = (time.perf_counter() - debut) * 1000
        await session.rollback()
    return duree


async def _recuperation(question: str, fournisseur: _VecteursEnMemoire) -> Dict[str, Any]:
    """La recuperation entiere du chat pour une question : durees et chemin."""
    async with AsyncSessionLocal() as session:
        rag = RAGService(session)
        rag.llm = None
        rag.search_service.use_cache = False
        rag.search_service.embedding_service = fournisseur
        debut = time.perf_counter()
        await rag._retrieve_chunks(question, "fr")
        duree = (time.perf_counter() - debut) * 1000
        sortie = {
            "duree_ms": duree,
            "sql_semantique_ms": rag.search_service.duree_sql_semantique_ms,
            "chemin": rag._recuperation.chemin,
        }
        await session.rollback()
    return sortie


async def mesurer(questions: List[str], vecteurs: Dict[str, Any]) -> Dict[str, Any]:
    fournisseur = _VecteursEnMemoire(vecteurs)
    sortie: Dict[str, Any] = {"questions": len(questions)}

    # Chauffe : le premier balayage lit la table depuis le disque.
    for question in questions[:3]:
        await _requete_seule(vecteurs[question], exacte=True)

    for nom, exacte in (("requete_exacte_seule", True), ("requete_index_seule", False)):
        durees = [await _requete_seule(vecteurs[q], exacte) for q in questions]
        sortie[nom] = _stats(durees)
        print(f"{nom:28} {sortie[nom]}", flush=True)

    for simultanees in (1, 4, 8):
        resultats: List[Dict[str, Any]] = []
        for debut in range(0, len(questions), simultanees):
            lot = questions[debut:debut + simultanees]
            resultats += await asyncio.gather(*(_recuperation(q, fournisseur) for q in lot))
        nom = f"recuperation_x{simultanees}"
        sortie[nom] = {
            "recuperation_ms": _stats([r["duree_ms"] for r in resultats]),
            "sql_semantique_ms": _stats([r["sql_semantique_ms"] for r in resultats
                                         if r["sql_semantique_ms"] is not None]),
            "chemins": {c: sum(1 for r in resultats if r["chemin"] == c)
                        for c in sorted({str(r["chemin"]) for r in resultats})},
        }
        print(f"{nom:28} {json.dumps(sortie[nom], ensure_ascii=False)}", flush=True)
    return sortie


def main(argv: Optional[Sequence[str]] = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parseur.add_argument("--vecteurs", type=Path, required=True,
                         help="fichier JSON {question: vecteur} (evaluer_contexte_chat --vecteurs)")
    parseur.add_argument("--questions", type=int, default=40, help="questions du jeu mesurees")
    parseur.add_argument("--sortie", type=Path, help="ecrit le detail en JSON")
    args = parseur.parse_args(argv)

    settings.RERANK_LLM_ENABLED = False
    moteur.echo = False

    async def _rien_journaliser(*args: Any, **kwargs: Any) -> None:
        return None

    module_recherche.record_search_event = _rien_journaliser
    module_recherche._init_global_singletons()

    vecteurs = {
        q: np.asarray(v, dtype=np.float32)
        for q, v in json.loads(args.vecteurs.read_text(encoding="utf-8")).items()
    }
    items = json.loads(JEU.read_text(encoding="utf-8"))["items"]
    questions = [it["question"] for it in items if it["question"] in vecteurs][:args.questions]
    sortie = asyncio.run(mesurer(questions, vecteurs))
    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nDetail : {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
