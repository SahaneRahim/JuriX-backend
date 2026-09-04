"""
Ingere un echantillon du corpus prc.cm dans la base.

Chaine complete par document : copie dans le repertoire d'upload, creation de la
ligne `laws`, extraction LlamaParse, decoupage en articles, embeddings 3072,
tsvector. C'est le meme chemin que l'upload par l'API, appele hors HTTP.

COUT REEL : LlamaParse est facture a la page. Sur un echantillon de 30 PDF du
corpus, la mediane est de 2 pages et la moyenne de 3,5. En cost_effective
(3 credits/page, 1000 credits = 1,25 $), 100 documents coutent environ 1,31 $.
--dry-run affiche l'echantillon et l'estimation sans depenser un credit.

L'echantillonnage est GRAINE et l'ingestion IDEMPOTENTE (une reference deja
presente est ignoree) : deux executions avec la meme graine donnent le meme
corpus, ce dont depend toute comparaison ulterieure.

Usage:
    python scripts/ingest_corpus.py --count 100 --seed 7 --dry-run
    python scripts/ingest_corpus.py --count 100 --seed 7
"""

import argparse
import json
import logging
import random
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, ".")

from app.core.database import SyncSessionLocal  # noqa: E402
from app.models.law import Law  # noqa: E402
from app.services.file_upload_service import get_upload_service  # noqa: E402
from app.tasks.process_law import process_law_sync  # noqa: E402

logger = logging.getLogger("ingest_corpus")

INDEX = Path("/home/rahim/jurix/documents_index.json")

# Taille maximale acceptee par FileUploadService.
MAX_SIZE_MB = 50

_MONTHS = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}

_TYPES = (
    ("constitution", "constitution"),
    ("ordonnance", "ordonnance"),
    ("arrete", "arrete"), ("arrêté", "arrete"),
    ("decision", "decision"), ("décision", "decision"),
    ("decret", "decret"), ("décret", "decret"),
    ("loi", "loi"),
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--normative-share", type=float, default=0.7,
                        help="part de textes normatifs (le reste au hasard)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args(argv)


def _infer_type(title: str) -> str:
    lowered = title.lower()
    for needle, value in _TYPES:
        if needle in lowered:
            return value
    return "loi"


def _infer_date(title: str):
    """Date de publication depuis le titre ("du 8 mai 2013")."""
    from datetime import date

    match = re.search(r"du\s+(\d{1,2})\s+([a-zéèûôA-Z]+)\s+(\d{4})", title, re.IGNORECASE)
    if not match:
        year = re.search(r"\b(19|20)\d{2}\b", title)
        return date(int(year.group(0)), 1, 1) if year else None
    day, month_name, year = match.groups()
    month = _MONTHS.get(month_name.lower())
    if not month:
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


# Actes de pure administration : nominations, integrations, ratifications. Ils
# representent 52,6 % du corpus et leurs articles sont quasi identiques d'un
# texte a l'autre ("Est nomme...", "sera enregistre, publie au Journal
# Officiel"). Un corpus de mesure qui en serait fait ne mesurerait qu'une
# correspondance de noms propres, sur laquelle la recherche plein texte gagne
# d'avance et le vecteur n'apporte rien.
_ADMINISTRATIVE = re.compile(
    r"portant\s+(nomination|integration|intégration|recrutement|reclassement|"
    r"admission|titularisation)"
    r"|ratifiant|portant\s+ratification"
    r"|portant\s+(adhesion|adhésion)",
    re.IGNORECASE,
)


def _is_normative(entry: Dict[str, Any]) -> bool:
    return not _ADMINISTRATIVE.search(entry.get("title", ""))


def _sample(entries: List[Dict[str, Any]], count: int, seed: int,
            normative_share: float) -> List[Dict[str, Any]]:
    """
    Echantillonnage stratifie : une part de textes normatifs, le reste au hasard.

    Un tirage uniforme donnerait un corpus a moitie fait de decrets de
    nomination — realiste, mais inutilisable pour comparer des methodes de
    recherche.
    """
    usable = [
        e for e in entries
        if e.get("path") and Path(e["path"]).is_file()
        and e.get("size_bytes", 0) < MAX_SIZE_MB * 1024 * 1024
    ]
    rng = random.Random(seed)

    normative = [e for e in usable if _is_normative(e)]
    others = [e for e in usable if not _is_normative(e)]

    wanted_normative = min(int(count * normative_share), len(normative))
    picked = rng.sample(normative, wanted_normative)
    remaining = min(count - len(picked), len(others))
    picked += rng.sample(others, remaining)
    rng.shuffle(picked)
    return picked


def _estimate_cost(entries: List[Dict[str, Any]]) -> str:
    """Estimation grossiere : 3,5 pages en moyenne, 3 credits par page."""
    credits = len(entries) * 3.5 * 3
    return f"~{credits:.0f} credits LlamaParse ≈ {credits / 1000 * 1.25:.2f} USD"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if not args.index.is_file():
        logger.error("Index introuvable : %s", args.index)
        return 1

    entries = json.loads(args.index.read_text(encoding="utf-8"))
    sample = _sample(entries, args.count, args.seed, args.normative_share)
    normative = sum(1 for e in sample if _is_normative(e))
    logger.info("%s documents echantillonnes (graine %s, %s normatifs / %s administratifs) — %s",
                len(sample), args.seed, normative, len(sample) - normative,
                _estimate_cost(sample))

    if args.dry_run:
        for entry in sample[:10]:
            logger.info("  PRC-%s | %s", entry["doc_id"], entry["title"][:90])
        logger.info("--dry-run : aucune copie, aucun appel a LlamaParse, aucune ecriture")
        return 0

    storage = get_upload_service().storage_path
    storage.mkdir(parents=True, exist_ok=True)

    ingested = skipped = failed = 0
    started = time.time()

    for position, entry in enumerate(sample, start=1):
        reference = f"PRC-{entry['doc_id']}"

        with SyncSessionLocal() as session:
            existing = session.query(Law).filter(Law.reference == reference).first()
            if existing:
                skipped += 1
                continue

            file_id = uuid.uuid4().hex          # respecte ^[A-Za-z0-9_-]{8,64}$
            destination = storage / f"{file_id}.pdf"
            try:
                shutil.copy2(entry["path"], destination)
            except Exception as exc:
                logger.error("Copie impossible pour %s : %s", reference, exc)
                failed += 1
                continue

            law = Law(
                reference=reference,
                title=entry["title"][:500],
                type=_infer_type(entry["title"]),
                # Rempli par le pipeline ; la colonne est NOT NULL.
                content="",
                language="fr",
                status="processing",
                file_id=file_id,
                original_filename=entry["filename"][:500],
                publication_date=_infer_date(entry["title"]),
            )
            session.add(law)
            session.commit()
            law_id = law.id

        try:
            result = process_law_sync(law_id, file_id)
            status = result.get("status", "?")
            logger.info(
                "[%s/%s] %s — %s (%s articles, %s embeddings, %.1fs)",
                position, len(sample), reference, status,
                result.get("articles_count", 0), result.get("embeddings_generated", 0),
                result.get("duration", 0.0),
            )
            ingested += 1
        except Exception as exc:
            logger.error("[%s/%s] %s ECHEC : %s", position, len(sample), reference, exc)
            failed += 1
            if not args.continue_on_error:
                return 1

    elapsed = time.time() - started
    logger.info("Termine en %.0f min : %s traites, %s deja presents, %s en echec",
                elapsed / 60, ingested, skipped, failed)

    with SyncSessionLocal() as session:
        from sqlalchemy import text as sql_text

        row = session.execute(sql_text(
            "SELECT (SELECT count(*) FROM laws) AS lois,"
            " (SELECT count(*) FROM articles) AS articles,"
            " (SELECT count(*) FROM articles WHERE embedding IS NOT NULL) AS vectorises,"
            " (SELECT count(*) FROM laws WHERE status = 'refused') AS refuses"
        )).one()
        logger.info("Base : %s lois, %s articles, %s vectorises, %s refuses",
                    row.lois, row.articles, row.vectorises, row.refuses)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
