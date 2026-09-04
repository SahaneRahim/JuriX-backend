"""
Reference d'extraction, phase 0 : mesurer AVANT de corriger quoi que ce soit.

POURQUOI CE SCRIPT EXISTE. Le pipeline d'extraction va etre modifie (pagination,
normalisation avant decoupage, largeur des colonnes). Sans une reference figee
prise avant les corrections, aucun gain ne sera demontrable ensuite : on aura
seulement l'intuition que "c'est mieux". Ce script fige les trois chiffres qui
comptent, dans data/eval_runs/, et ne touche a rien d'autre.

STRICTEMENT EN LECTURE. Aucune ecriture en base, aucun appel d'API, aucun credit
consomme. Les seules ecritures sont le fichier JSON de sortie.

CE QU'IL MESURE, et pourquoi chaque mesure est la :

  1. Le degrade silencieux. Quand aucun motif d'article n'est reconnu,
     extract_articles ne leve pas : il bascule sur _extract_paragraph_chunks et
     fabrique des pseudo-articles PARA_1, PARA_2... La loi est publiee, son
     compte d'articles est non nul, et rien ne signale l'echec. Le taux de
     PARA_ est donc le vrai indicateur de sante, pas articles_count > 0.

  2. Le gain latent de normalize_for_chunking(). La fonction existe dans
     app/utils/chunk_refiner.py, elle est testee, et elle n'est appelee nulle
     part en production. On rejoue la detection de motif sur les caches OCR
     deja payes, avec et sans elle, pour chiffrer ce que son cablage rapporte.

  3. La couverture de pages. _pages_from renvoie une seule page quand la reponse
     de LlamaParse est une chaine, alors que le markdown contient les vraies
     coupures. On compare donc les pages declarees par le cache, les pages
     reellement presentes dans le texte, et le nombre de pages du PDF source.

Usage :
    DATABASE_URL=postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_dev \\
        python -m scripts.eval.extraction.baseline
    python -m scripts.eval.extraction.baseline --label phase0_avant_correctifs

Author: JuriX Team
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, ".")

from sqlalchemy import text  # noqa: E402

from app.core.database import SyncSessionLocal  # noqa: E402
from app.utils.chunk_refiner import normalize_for_chunking  # noqa: E402
from app.utils.text_chunker import _detect_article_pattern  # noqa: E402

logger = logging.getLogger("eval.extraction.baseline")

CACHE_DIR = Path("data/ocr_cache")
UPLOADS_DIR = Path("data/uploads")
RUNS_DIR = Path("data/eval_runs")

# Separateur de page emis par LlamaParse dans le markdown. _pages_from l'ignore
# aujourd'hui, d'ou l'ecart entre pages declarees et pages reelles.
PAGE_SEPARATOR = "\n---\n"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="phase0_baseline",
                        help="nom du fichier ecrit dans data/eval_runs/")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--skip-db", action="store_true",
                        help="ne mesurer que les caches (aucune connexion base)")
    return parser.parse_args(argv)


# ==================== 1. CACHES OCR DEJA PAYES ====================


def _count_article_headers(body: str) -> int:
    """Nombre d'en-tetes d'article que le chunker reconnaitrait dans ce texte."""
    pattern = _detect_article_pattern(body)
    return len(re.findall(pattern, body)) if pattern else 0


@contextmanager
def _quiet_chunker():
    """
    Fait taire le chunker le temps de la mesure.

    _detect_article_pattern journalise le motif retenu a chaque appel. On
    l'appelle deux fois par document : sur 61 caches, cela noie le rapport sous
    122 lignes sans interet ici.
    """
    chunker_log = logging.getLogger("app.utils.text_chunker")
    previous = chunker_log.level
    chunker_log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        chunker_log.setLevel(previous)


def measure_cache(cache_dir: Path) -> Dict[str, Any]:
    """
    Rejoue la detection de motif sur les caches, avec et sans normalisation.

    Le gain mesure ici est disponible immediatement et gratuitement : il ne
    demande aucun changement de moteur OCR, seulement de brancher une fonction
    qui existe deja.
    """
    files = sorted(cache_dir.glob("*.json"))
    if not files:
        logger.warning("Aucun cache dans %s", cache_dir)

    declared_pages = real_pages = 0
    headers_before = headers_after = 0
    degraded_before = degraded_after = 0
    with_table = 0
    per_doc: List[Dict[str, Any]] = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache illisible %s : %s", path.name, exc)
            continue

        pages = payload.get("pages") or []
        body = "\n\n".join(pages)
        # Les vraies coupures sont dans le markdown, pas dans la liste `pages`.
        n_real = body.count(PAGE_SEPARATOR) + 1 if body else 0

        with _quiet_chunker():
            before = _count_article_headers(body)
            after = _count_article_headers(normalize_for_chunking(body))

        declared_pages += len(pages)
        real_pages += n_real
        headers_before += before
        headers_after += after
        degraded_before += before == 0
        degraded_after += after == 0
        with_table += "<table" in body

        per_doc.append({
            "cache": path.stem,
            "declared_pages": len(pages),
            "real_pages": n_real,
            "chars": len(body),
            "headers_before": before,
            "headers_after": after,
        })

    n = len(per_doc)
    return {
        "documents": n,
        "declared_pages": declared_pages,
        "real_pages": real_pages,
        "pages_lost_ratio": round(1 - declared_pages / real_pages, 4) if real_pages else None,
        "article_headers_before_normalize": headers_before,
        "article_headers_after_normalize": headers_after,
        "headers_gain_factor": round(headers_after / headers_before, 2) if headers_before else None,
        "degraded_docs_before": degraded_before,
        "degraded_docs_after": degraded_after,
        "docs_with_table": with_table,
        "chars_total": sum(d["chars"] for d in per_doc),
        "per_doc": per_doc,
    }


# ==================== 2. ETAT DE LA BASE ====================


# Une seule requete par grandeur, en lecture. `number LIKE 'PARA\_%'` echappe le
# tiret bas, qui est un joker en SQL : sans l'echappement, 'PARAX' compterait.
_DB_QUERIES: Dict[str, str] = {
    "articles_total": "SELECT count(*) FROM articles",
    "articles_para": r"SELECT count(*) FROM articles WHERE number LIKE 'PARA\_%'",
    "articles_full_text": "SELECT count(*) FROM articles WHERE number = 'FULL_TEXT'",
    "articles_legal_basis": "SELECT count(*) FROM articles WHERE number = 'LEGAL_BASIS'",
    "articles_page_1": "SELECT count(*) FROM articles WHERE page_number = 1",
    "articles_page_null": "SELECT count(*) FROM articles WHERE page_number IS NULL",
    "articles_no_embedding": "SELECT count(*) FROM articles WHERE embedding IS NULL",
    "laws_total": "SELECT count(*) FROM laws",
    "laws_published": "SELECT count(*) FROM laws WHERE status = 'published'",
    "laws_refused": "SELECT count(*) FROM laws WHERE status = 'refused'",
    "laws_degraded": (
        r"SELECT count(DISTINCT law_id) FROM articles "
        r"WHERE number LIKE 'PARA\_%' OR number = 'FULL_TEXT'"
    ),
    "laws_published_without_article": (
        "SELECT count(*) FROM laws l WHERE l.status = 'published' "
        "AND NOT EXISTS (SELECT 1 FROM articles a WHERE a.law_id = l.id)"
    ),
    "alembic_head": "SELECT version_num FROM alembic_version",
}


def _plain(value: Any) -> Any:
    """
    Rend un scalaire SQLAlchemy serialisable en JSON.

    round() cote Postgres renvoie un NUMERIC, que psycopg2 mappe sur Decimal,
    que json.dumps refuse. Convertir ici plutot qu'au moment d'ecrire evite de
    disperser le probleme sur chaque champ ajoute plus tard.
    """
    return float(value) if isinstance(value, Decimal) else value


def measure_db() -> Dict[str, Any]:
    """Compte les symptomes observables en base. Lecture seule."""
    with SyncSessionLocal() as session:
        out: Dict[str, Any] = {
            key: _plain(session.execute(text(sql)).scalar())
            for key, sql in _DB_QUERIES.items()
        }

        # Part des caracteres de laws.content retrouves dans les articles : ce
        # qui manque a ete supprime entre l'extraction et le decoupage.
        out["char_retention_pct"] = _plain(session.execute(text(
            "SELECT round(100.0 * sum(a_len) / NULLIF(sum(l_len), 0)) FROM ("
            "  SELECT length(l.content) AS l_len,"
            "         COALESCE((SELECT sum(length(content)) FROM articles"
            "                   WHERE law_id = l.id), 0) AS a_len"
            "  FROM laws l) t"
        )).scalar())

        rows = session.execute(text(
            "SELECT l.id, l.file_id, count(a.id) AS articles,"
            "       count(DISTINCT a.page_number) AS pages_extracted "
            "FROM laws l LEFT JOIN articles a ON a.law_id = l.id "
            "GROUP BY l.id, l.file_id ORDER BY l.id"
        )).all()

    total = out["articles_total"] or 0
    out["articles_para_pct"] = round(100.0 * (out["articles_para"] or 0) / total) if total else None
    out["page_coverage"] = _measure_page_coverage(rows)
    return out


def _measure_page_coverage(rows: Sequence[Any]) -> Dict[str, Any]:
    """
    Compare les pages extraites au nombre de pages du PDF source.

    C'est le garde-fou le moins cher du projet et il n'existe pas : un PDF de
    400 pages tronque a une seule passe aujourd'hui le seul controle en place
    (plus de 50 caracteres) et est publie sans le moindre signal.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf absent : couverture de pages non mesuree")
        return {"available": False}

    per_law: List[Dict[str, Any]] = []
    for law_id, file_id, articles, pages_extracted in rows:
        if not file_id:
            continue
        pdf = UPLOADS_DIR / f"{file_id}.pdf"
        if not pdf.exists():
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdf_pages = len(PdfReader(str(pdf)).pages)
        except Exception as exc:  # PDF corrompu, chiffre, tronque
            logger.warning("PDF illisible pour la loi %s : %s", law_id, exc)
            continue

        per_law.append({
            "law_id": law_id,
            "pdf_pages": pdf_pages,
            "pages_extracted": pages_extracted,
            "articles": articles,
            "coverage": round(pages_extracted / pdf_pages, 3) if pdf_pages else None,
        })

    measured = [r for r in per_law if r["coverage"] is not None]
    below = [r for r in measured if r["coverage"] < 0.95]
    return {
        "available": True,
        "laws_measured": len(measured),
        "pdf_pages_total": sum(r["pdf_pages"] for r in measured),
        "pages_extracted_total": sum(r["pages_extracted"] for r in measured),
        "mean_coverage": round(
            sum(r["coverage"] for r in measured) / len(measured), 3
        ) if measured else None,
        "laws_below_95pct": len(below),
        "per_law": per_law,
    }


# ==================== 3. RAPPORT ====================


def _print_report(cache: Dict[str, Any], db: Optional[Dict[str, Any]]) -> None:
    print()
    print("=" * 68)
    print("  REFERENCE D'EXTRACTION - PHASE 0 (avant tout correctif)")
    print("=" * 68)

    print(f"\nCaches OCR deja payes ({cache['documents']} documents)")
    print(f"  pages declarees / reelles     {cache['declared_pages']} / {cache['real_pages']}")
    if cache["pages_lost_ratio"] is not None:
        print(f"  structure de pages perdue     {cache['pages_lost_ratio']:.1%}")
    print(f"  en-tetes avant normalisation  {cache['article_headers_before_normalize']}")
    print(f"  en-tetes apres normalisation  {cache['article_headers_after_normalize']}"
          f"  (x{cache['headers_gain_factor']})")
    print(f"  documents sans motif detecte  {cache['degraded_docs_before']}"
          f" -> {cache['degraded_docs_after']}  sur {cache['documents']}")
    print(f"  documents avec tableau        {cache['docs_with_table']}")

    if db is None:
        print("\n(base non mesuree : --skip-db)")
        return

    print("\nBase de donnees")
    print(f"  lois                          {db['laws_total']}"
          f"  (publiees {db['laws_published']}, refusees {db['laws_refused']})")
    print(f"  articles                      {db['articles_total']}")
    print(f"  dont pseudo-articles PARA_    {db['articles_para']}"
          f"  ({db['articles_para_pct']}%)  sur {db['laws_degraded']} lois")
    print(f"  dont FULL_TEXT / LEGAL_BASIS  {db['articles_full_text']} / {db['articles_legal_basis']}")
    print(f"  page_number = 1               {db['articles_page_1']} / {db['articles_total']}")
    print(f"  caracteres conserves          {db['char_retention_pct']}%")
    print(f"  lois publiees sans article    {db['laws_published_without_article']}   (doit valoir 0)")

    cov = db["page_coverage"]
    if cov.get("available"):
        print(f"\nCouverture de pages ({cov['laws_measured']} lois mesurees)")
        print(f"  pages PDF / pages extraites   {cov['pdf_pages_total']} / {cov['pages_extracted_total']}")
        print(f"  couverture moyenne            {cov['mean_coverage']}")
        print(f"  lois sous 95 %                {cov['laws_below_95pct']} / {cov['laws_measured']}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    cache = measure_cache(args.cache_dir)
    db = None if args.skip_db else measure_db()
    _print_report(cache, db)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.label}.json"
    out.write_text(json.dumps({
        "label": args.label,
        "phase": 0,
        "note": "Reference figee AVANT les correctifs de pipeline. Ne pas regenerer apres.",
        # Meme convention que scripts/eval/run_eval.py : les chiffres ne veulent
        # rien dire hors de l'etat de corpus qui les a produits.
        "corpus_snapshot": {
            "alembic_head": db["alembic_head"] if db else None,
            "articles": db["articles_total"] if db else None,
        },
        "cache": cache,
        "database": db,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Reference ecrite dans %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
