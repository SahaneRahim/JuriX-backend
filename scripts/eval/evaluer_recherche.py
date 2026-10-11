"""
Mesure le classement de la PAGE DE RECHERCHE, document par document.

La page /search envoie une recherche en mode « text », filtree sur
status=published, 20 resultats par page. Ce script rejoue exactement cette
requete, sans cache et sans journaliser d'evenement de recherche (lecture
seule), et mesure le rang de la loi attendue :

- les 102 questions relues de tests/fixtures/eval/retrieval_eval_v1.json
  (loi attendue : expected_law_id) : hit@1, hit@3, hit@10, MRR@20 ;
- des requetes TEMOINS, chacune avec ce qu'on attend d'elle : la requete qui
  a revele le defaut (« peine de mort au cameroun »), et celles qui fondent la
  passe floue sur les titres (fautes de frappe : « nominaton », « code miner »).

Usage:
    python -m scripts.eval.evaluer_recherche
    python -m scripts.eval.evaluer_recherche --modes text hybrid --sortie data/eval_runs/recherche.json
"""

import argparse
import asyncio
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from app.core.database import AsyncSessionLocal
from app.schemas.search import SearchFilters, SearchRequest
from app.services.search_service import SearchService

JEU = RACINE / "tests" / "fixtures" / "eval" / "retrieval_eval_v1.json"
TAILLE_PAGE = 20  # TAILLE_PAGE_RECHERCHE du front (src/lib/pagination.ts)

CODE_PENAL, JUSTICE_MILITAIRE = 2236, 2085
CODES_MINIERS = {2210, 2228}


def _sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texte.lower())
                   if unicodedata.category(c) != "Mn")


def _titres_contiennent(mot: str, n: int) -> Callable[[Dict[str, Any]], bool]:
    """Les n premiers titres contiennent tous `mot` (sans accents)."""
    def verifier(reponse: Dict[str, Any]) -> bool:
        titres = [_sans_accents(r["title"]) for r in reponse["results"][:n]]
        return len(titres) == n and all(mot in t for t in titres)
    return verifier


def _rang_parmi(ids: set, n: int) -> Callable[[Dict[str, Any]], bool]:
    """Une des lois `ids` figure dans les n premiers resultats."""
    def verifier(reponse: Dict[str, Any]) -> bool:
        return any(r["law_id"] in ids for r in reponse["results"][:n])
    return verifier


def _navigation_directe(ids: set) -> Callable[[Dict[str, Any]], bool]:
    def verifier(reponse: Dict[str, Any]) -> bool:
        return bool(reponse["direct_navigation"]) and reponse["target_law_id"] in ids
    return verifier


# (requete, attente lisible, verification)
TEMOINS: List[tuple] = [
    ("peine de mort au cameroun", "Code penal ou justice militaire dans le top 3",
     _rang_parmi({CODE_PENAL, JUSTICE_MILITAIRE}, 3)),
    ("peine de mort", "Code penal ou justice militaire dans le top 3",
     _rang_parmi({CODE_PENAL, JUSTICE_MILITAIRE}, 3)),
    ("justice militaire", "Code de justice militaire premier",
     _rang_parmi({JUSTICE_MILITAIRE}, 1)),
    ("code pénal", "Code penal (loi 2236 ou decret 1769) dans le top 2",
     _rang_parmi({CODE_PENAL, 1769}, 2)),
    ("code minier", "un code minier dans le top 2", _rang_parmi(CODES_MINIERS, 2)),
    ("code miner", "faute de frappe : un code minier dans le top 3", _rang_parmi(CODES_MINIERS, 3)),
    ("cod minier", "faute de frappe : un code minier dans le top 3", _rang_parmi(CODES_MINIERS, 3)),
    ("nominaton", "faute de frappe : les 5 premiers titres parlent de nomination",
     _titres_contiennent("nomination", 5)),
    ("avancment", "faute de frappe : les 3 premiers titres parlent d'avancement",
     _titres_contiennent("avancement", 3)),
    ("article 35 du code minier", "navigation directe vers un code minier",
     _navigation_directe(CODES_MINIERS)),
]


async def rejouer(service: SearchService, requete: str, mode: str) -> Dict[str, Any]:
    """La requete de la page, sans cache ni journal d'evenement."""
    demande = SearchRequest(
        query=requete, mode=mode, limit=TAILLE_PAGE, offset=0,
        filters=SearchFilters(status="published"),
    )
    chunks = await service._execute_search_by_mode(demande)
    reponse = await service._build_search_response(demande, chunks, 0)
    return {
        "results": [{"law_id": r.law_id, "title": r.title, "score": round(r.relevance_score, 3),
                     "portee": r.match_scope} for r in reponse.results],
        "total": reponse.total,
        "direct_navigation": reponse.direct_navigation,
        "target_law_id": reponse.target_law_id,
    }


def _rang(reponse: Dict[str, Any], law_id: int) -> Optional[int]:
    for position, r in enumerate(reponse["results"], start=1):
        if r["law_id"] == law_id:
            return position
    return None


def synthese(rangs: Sequence[Optional[int]]) -> Dict[str, Any]:
    n = len(rangs) or 1
    return {
        "n": len(rangs),
        "hit@1": round(sum(1 for r in rangs if r and r <= 1) / n, 3),
        "hit@3": round(sum(1 for r in rangs if r and r <= 3) / n, 3),
        "hit@10": round(sum(1 for r in rangs if r and r <= 10) / n, 3),
        "mrr@20": round(sum(1 / r for r in rangs if r) / n, 3),
        "aucun_resultat": None,
    }


async def evaluer(modes: Sequence[str]) -> Dict[str, Any]:
    items = json.loads(JEU.read_text(encoding="utf-8"))["items"]
    sortie: Dict[str, Any] = {"modes": {}}
    async with AsyncSessionLocal() as session:
        service = SearchService(session, use_cache=False)
        for mode in modes:
            par_item, rangs, vides = [], {"keyword": [], "paraphrase": [], "tous": []}, 0
            for item in items:
                reponse = await rejouer(service, item["question"], mode)
                rang = _rang(reponse, item["expected_law_id"])
                vides += not reponse["results"]
                rangs["tous"].append(rang)
                rangs.setdefault(item.get("difficulty", "?"), []).append(rang)
                par_item.append({"question": item["question"], "attendu": item["expected_law_id"],
                                 "rang": rang, "total": reponse["total"]})
            temoins = []
            for requete, attente, verifier in TEMOINS:
                reponse = await rejouer(service, requete, mode)
                temoins.append({"requete": requete, "attente": attente, "ok": verifier(reponse),
                                "top5": [f"{r['law_id']} {r['portee']} {r['score']} {r['title'][:70]}"
                                         for r in reponse["results"][:5]]})
            resume = {cle: synthese(valeurs) for cle, valeurs in rangs.items() if valeurs}
            resume["tous"]["aucun_resultat"] = vides
            sortie["modes"][mode] = {"synthese": resume, "temoins": temoins, "items": par_item}
    return sortie


def afficher(sortie: Dict[str, Any]) -> None:
    for mode, donnees in sortie["modes"].items():
        print(f"\n=== mode {mode}")
        for cle, s in donnees["synthese"].items():
            print(f"  {cle:10} n={s['n']:3}  hit@1 {s['hit@1']:.3f}  hit@3 {s['hit@3']:.3f}  "
                  f"hit@10 {s['hit@10']:.3f}  mrr@20 {s['mrr@20']:.3f}"
                  + (f"  sans resultat {s['aucun_resultat']}" if s["aucun_resultat"] is not None else ""))
        ok = sum(t["ok"] for t in donnees["temoins"])
        print(f"  temoins : {ok}/{len(donnees['temoins'])}")
        for t in donnees["temoins"]:
            print(f"    [{'ok' if t['ok'] else 'KO'}] {t['requete']!r} : {t['attente']}")
            if not t["ok"]:
                for ligne in t["top5"]:
                    print(f"         {ligne}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parseur.add_argument("--modes", nargs="+", default=["text"], choices=["text", "semantic", "hybrid"])
    parseur.add_argument("--sortie", type=Path, help="Ecrit le detail en JSON")
    args = parseur.parse_args(argv)
    sortie = asyncio.run(evaluer(args.modes))
    afficher(sortie)
    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nDetail : {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
