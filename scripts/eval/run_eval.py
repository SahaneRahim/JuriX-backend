"""
Mesure la recuperation sur le jeu d'evaluation.

Trois questions distinctes, trois chemins :

1. LA DIMENSION (--dims). Compare 3072, 1536 et 768 en FORCE BRUTE, en numpy,
   sur les memes chunks : la question est la qualite de l'EMBEDDING, pas celle
   de l'index. Les vecteurs des dimensions inferieures sont obtenus en
   tronquant puis renormalisant les vecteurs 3072 — gemini-embedding-001 est
   entraine en Matryoshka, ce que scripts/eval/validate_slicing.py verifie
   contre l'API. Cout : zero appel supplementaire pour le corpus.

2. LE MODE ET LE RE-RANKING (--modes, --rerank). Passe par le vrai
   SearchService, donc par l'index et le cache : c'est le systeme reel qui est
   mesure.

3. LES POIDS (--sweep). Balaie RRF_K et TEXT_WEIGHT sur le lot dev.

Aucun resultat ne se lit sans son intervalle de confiance. Avec 60 questions,
l'erreur type est d'environ 6 points : un ecart de 2 points n'est pas un
resultat, et la conclusion « aucune difference mesurable, prends le moins
cher » est une conclusion valide.

Usage:
    python -m scripts.eval.run_eval --dims 3072 1536 768
    python -m scripts.eval.run_eval --modes text semantic hybrid --rerank none stage1
    python -m scripts.eval.run_eval --sweep rrf
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import text

sys.path.insert(0, ".")

from app.core.database import AsyncSessionLocal, SyncSessionLocal  # noqa: E402
from app.schemas.search import SearchRequest  # noqa: E402
from app.services.embedding_service import EmbeddingService, get_embedding_service  # noqa: E402
from app.services.reranker import rerank_chunks  # noqa: E402
from app.services.search_service import SearchService  # noqa: E402
from scripts.eval.metrics import summarise  # noqa: E402

logger = logging.getLogger("run_eval")

DEFAULT_SET = Path("tests/fixtures/eval/retrieval_eval_v1.json")
RUNS_DIR = Path("data/eval_runs")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--dims", type=int, nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=["hybrid"],
                        choices=["text", "semantic", "hybrid"])
    parser.add_argument("--rerank", nargs="*", default=["none", "stage1"],
                        choices=["none", "stage1"])
    parser.add_argument("--sweep", choices=["rrf"], default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--limit", type=int, default=20,
                        help="chunks demandes par question")
    parser.add_argument("--label", default="run")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--skip-snapshot-check", action="store_true")
    return parser.parse_args(argv)


def _load_eval_set(path: Path, split: str, allow_unreviewed: bool,
                   skip_snapshot: bool) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Jeu d'evaluation absent : {path}\n"
            "  python -m scripts.eval.generate_eval_set --sample 120"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [i for i in payload["items"] if split == "all" or i.get("split") == split]

    unreviewed = [i for i in items if not i.get("reviewed")]
    if unreviewed and not allow_unreviewed:
        raise SystemExit(
            f"{len(unreviewed)} item(s) non relus sur {len(items)}.\n"
            "  Relisez tests/fixtures/eval/*.review.md et passez reviewed=true,\n"
            "  ou --allow-unreviewed en connaissance de cause."
        )
    if unreviewed:
        logger.warning("%s item(s) non relus inclus dans la mesure", len(unreviewed))

    if not skip_snapshot:
        with SyncSessionLocal() as session:
            head = session.execute(text("SELECT version_num FROM alembic_version")).scalar()
            articles = session.execute(text("SELECT count(*) FROM articles")).scalar()
        expected = payload.get("corpus_snapshot", {})
        # Les identifiants d'article du jeu ne designent quelque chose que
        # contre UN etat de corpus. Sans ce controle, les chiffres deviennent
        # faux en silence apres une reingestion.
        if expected.get("alembic_head") != head or expected.get("articles") != articles:
            raise SystemExit(
                "Le corpus a change depuis la generation du jeu :\n"
                f"  attendu : head={expected.get('alembic_head')} "
                f"articles={expected.get('articles')}\n"
                f"  actuel  : head={head} articles={articles}\n"
                "  Regenerez le jeu, reattachez les identifiants, "
                "ou --skip-snapshot-check."
            )

    payload["items"] = items
    return payload


# ==================== 1. DIMENSION, EN FORCE BRUTE ====================


def _load_corpus_vectors() -> tuple:
    """Tous les vecteurs d'articles en memoire, une seule fois."""
    with SyncSessionLocal() as session:
        rows = session.execute(text(
            "SELECT id, embedding::text FROM articles WHERE embedding IS NOT NULL ORDER BY id"
        )).fetchall()

    if not rows:
        raise SystemExit(
            "Aucun article vectorise. Lancez d'abord :\n"
            "  python scripts/regenerate_embeddings.py --all"
        )

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    matrix = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
    return ids, matrix


def _slice_and_renormalise(matrix: np.ndarray, dim: int) -> np.ndarray:
    """
    Troncature Matryoshka + renormalisation.

    La renormalisation n'est pas cosmetique : le prefixe d'un vecteur unitaire
    n'est pas unitaire, et toute la comparaison cosinus la suppose.
    """
    sliced = matrix[:, :dim]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return sliced / norms


async def _embed_questions(questions: List[str]) -> np.ndarray:
    service = get_embedding_service()
    vectors = []
    for question in questions:
        vectors.append(await service.generate_embedding_async(
            question, task_type=EmbeddingService.TASK_QUERY
        ))
    return np.array(vectors, dtype=np.float32)


def _brute_force_rankings(article_ids: np.ndarray, corpus: np.ndarray,
                          queries: np.ndarray, top_k: int) -> List[List[int]]:
    scores = queries @ corpus.T                      # vecteurs unitaires : produit = cosinus
    order = np.argsort(-scores, axis=1)[:, :top_k]
    return [[int(article_ids[i]) for i in row] for row in order]


# ==================== 2. LE SYSTEME REEL ====================


async def _system_rankings(items: List[Dict[str, Any]], mode: str, rerank: str,
                           limit: int) -> List[List[int]]:
    rankings = []
    async with AsyncSessionLocal() as session:
        service = SearchService(session, use_cache=False)
        for item in items:
            response = await service.search(SearchRequest(
                query=item["question"], mode=mode, limit=limit,
            ))
            chunks = response.chunks
            if rerank == "stage1":
                chunks = rerank_chunks(item["question"], chunks)
            rankings.append([c.article_id for c in chunks if c.article_id is not None])
    return rankings


# ==================== RAPPORT ====================


def _print_table(rows: List[Dict[str, Any]], ks: Sequence[int]) -> None:
    header = ["configuration"] + [f"R@{k}" for k in ks] + ["MRR@10", "IC 95% (MRR)", "n"]
    print("\n" + " | ".join(header))
    print("-" * (18 + 12 * len(header)))
    for row in rows:
        cells = [row["config"]]
        cells += [f"{row['metrics'][f'recall@{k}']:.3f}" for k in ks]
        low, high = row["metrics"]["mrr@10_ci"]
        cells += [f"{row['metrics']['mrr@10']:.3f}", f"[{low:.3f}, {high:.3f}]",
                  str(row["metrics"]["n"])]
        print(" | ".join(cells))
    print(
        "\nAvec un echantillon de cette taille, un ecart inferieur a la largeur\n"
        "de l'intervalle de confiance n'est PAS un resultat.\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    payload = _load_eval_set(args.eval_set, args.split, args.allow_unreviewed,
                             args.skip_snapshot_check)
    items = payload["items"]
    if not items:
        raise SystemExit(f"Aucun item dans le lot '{args.split}'")

    gold = [item["expected_article_id"] for item in items]
    questions = [item["question"] for item in items]
    logger.info("%s questions (lot %s)", len(items), args.split)

    rows: List[Dict[str, Any]] = []

    if args.dims:
        article_ids, corpus = _load_corpus_vectors()
        logger.info("%s vecteurs charges (%s dimensions)", len(article_ids), corpus.shape[1])
        query_vectors = asyncio.run(_embed_questions(questions))

        for dim in args.dims:
            if dim > corpus.shape[1]:
                logger.warning("Dimension %s > %s stockees, ignoree", dim, corpus.shape[1])
                continue
            rankings = _brute_force_rankings(
                article_ids,
                _slice_and_renormalise(corpus, dim),
                _slice_and_renormalise(query_vectors, dim),
                max(args.k),
            )
            rows.append({
                "config": f"brute-force/{dim}d",
                "metrics": summarise(rankings, gold, ks=args.k),
            })

    if args.sweep == "rrf":
        from app.services import search_service as module
        original_k, original_w = module.SearchService.RRF_K, module.SearchService.TEXT_WEIGHT
        try:
            for rrf_k in (10, 20, 40, 60, 80, 120):
                for weight in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                    module.SearchService.RRF_K = rrf_k
                    module.SearchService.TEXT_WEIGHT = weight
                    module.SearchService.SEMANTIC_WEIGHT = 1.0 - weight
                    rankings = asyncio.run(
                        _system_rankings(items, "hybrid", "stage1", args.limit)
                    )
                    rows.append({
                        "config": f"hybrid/k={rrf_k}/text={weight:.1f}",
                        "metrics": summarise(rankings, gold, ks=args.k),
                    })
        finally:
            module.SearchService.RRF_K = original_k
            module.SearchService.TEXT_WEIGHT = original_w
            module.SearchService.SEMANTIC_WEIGHT = 1.0 - original_w
    elif not args.dims:
        for mode in args.modes:
            for rerank in args.rerank:
                rankings = asyncio.run(_system_rankings(items, mode, rerank, args.limit))
                rows.append({
                    "config": f"{mode}/rerank={rerank}",
                    "metrics": summarise(rankings, gold, ks=args.k),
                })

    rows.sort(key=lambda r: r["metrics"]["mrr@10"], reverse=True)
    _print_table(rows, args.k)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.label}.json"
    out.write_text(json.dumps({
        "label": args.label,
        "split": args.split,
        "eval_set": str(args.eval_set),
        "corpus_snapshot": payload.get("corpus_snapshot"),
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Resultats ecrits dans %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
