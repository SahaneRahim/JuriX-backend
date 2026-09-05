#!/usr/bin/env python3
"""
Re-decoupe les lois deja en base avec le decoupeur corrige.

POURQUOI. Les motifs de marqueur d'article attendaient `Article <numero>` en
debut de ligne. LlamaParse rend du markdown : le corpus ecrit `**ARTICLE 1er.**-`
et parfois `**Article1er.-**`, avec des `**` intercales et sans espace. Mesure
avant correction : 8 lois sur 27 avaient au moins un numero d'article reconnu,
les 19 autres n'ayant que des pseudo-numeros `PARA_1, PARA_2...`. Aucune requete
« article 35 du code minier » ne pouvait aboutir sur ces documents.

Ce script re-decoupe DEPUIS `laws.content`, pas depuis le PDF : le markdown est
deja en base, donc ni LlamaParse ni l'OCR ne sont rappeles. Il fait aussi passer
les chunks par `chunk_refiner`, ce qui renseigne `kind`, `embed` et `embed_text`
— nuls sur la totalite du corpus, qui est anterieur a son cablage.

Les embeddings NE SONT PAS regeneres ici. Enchainer ensuite :
    python scripts/regenerate_embeddings.py --all --force

Usage:
    python scripts/maintenance/rechunk_laws.py --dry-run
    python scripts/maintenance/rechunk_laws.py --apply
    python scripts/maintenance/rechunk_laws.py --apply --law-id 16

Codes de sortie:
    0  termine
    1  au moins une loi a echoue
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.database import SyncSessionLocal  # noqa: E402
from app.models.law import Article, Law  # noqa: E402
from app.utils.chunk_refiner import normalize_for_chunking  # noqa: E402
from app.utils.text_chunker import extract_articles  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rechunk")

# Un vrai numero d'article commence par un chiffre. PARA_n, LEGAL_BASIS,
# PREAMBULE et les ordinaux en toutes lettres sont des pseudo-numeros : ils
# designent du texte conserve mais non citable comme « article N ».
_REAL_NUMBER = re.compile(r"^\d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-decoupe les lois en base avec le decoupeur corrige.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="n'ecrit rien, compare les numeros avant/apres (defaut)")
    mode.add_argument("--apply", action="store_true",
                      help="remplace les articles en base, une transaction par loi")
    parser.add_argument("--law-id", type=int, action="append", dest="law_ids",
                        help="ne traiter que cette loi (repetable)")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _real_numbers(numbers: List[str]) -> int:
    return sum(1 for n in numbers if n and _REAL_NUMBER.match(n))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    with SyncSessionLocal() as session:
        query = session.query(Law).order_by(Law.id)
        if args.law_ids:
            query = query.filter(Law.id.in_(args.law_ids))
        if args.limit:
            query = query.limit(args.limit)
        laws = query.all()

        rows = []
        for law in laws:
            before = [
                n for (n,) in session.query(Article.number)
                .filter(Article.law_id == law.id).all()
            ]
            try:
                extracted = extract_articles(
                    normalize_for_chunking(law.content or ""),
                    strict=False, min_article_length=1,
                )
            except Exception as exc:
                logger.warning("  #%-4s decoupage impossible : %s", law.id, exc)
                extracted = []
            after = [str(a.get("number", "")) for a in extracted]
            rows.append({
                "law": law,
                "before_total": len(before), "before_real": _real_numbers(before),
                "after_total": len(after), "after_real": _real_numbers(after),
            })

    # ==================== RAPPORT ====================
    logger.info("%-6s %-9s %-9s %s", "loi", "avant", "apres", "titre")
    logger.info("%s", "-" * 78)
    gained = 0
    for row in rows:
        law = row["law"]
        marker = " " if row["after_real"] <= row["before_real"] else "+"
        if marker == "+":
            gained += 1
        logger.info(
            "%-6s %3d/%-5d %3d/%-5d %s %s",
            law.id, row["before_real"], row["before_total"],
            row["after_real"], row["after_total"], marker, (law.title or "")[:44],
        )

    total_before = sum(r["before_real"] for r in rows)
    total_after = sum(r["after_real"] for r in rows)
    laws_before = sum(1 for r in rows if r["before_real"] > 0)
    laws_after = sum(1 for r in rows if r["after_real"] > 0)
    logger.info("%s", "-" * 78)
    logger.info("Numeros d'article reels : %d -> %d", total_before, total_after)
    logger.info("Lois avec au moins un numero : %d -> %d (sur %d)",
                laws_before, laws_after, len(rows))
    logger.info("%d loi(s) y gagnent", gained)

    if not args.apply:
        logger.info("\n(dry-run : rien n'a ete ecrit. Relancez avec --apply.)")
        return 0

    # ==================== ECRITURE ====================
    # Une transaction PAR LOI : une loi dont le decoupage echoue ne doit pas
    # emporter les autres, et surtout ne doit pas rester sans aucun article
    # apres la suppression des anciens.
    from app.tasks.process_law import _split_and_save_articles

    failures = 0
    for row in rows:
        law = row["law"]
        try:
            count = _split_and_save_articles(law.id, law.content or "")
            if count == 0:
                logger.warning("  #%-4s AUCUN chunk produit — loi laissee vide", law.id)
                failures += 1
        except Exception as exc:
            logger.error("  #%-4s echec : %s", law.id, exc)
            failures += 1

    with SyncSessionLocal() as session:
        remaining = session.query(Article).filter(Article.kind.is_(None)).count()
        real_laws = session.execute(
            __import__("sqlalchemy").text(
                "SELECT count(DISTINCT law_id) FROM articles WHERE number ~ '^[0-9]'"
            )
        ).scalar_one()

    logger.info("\n%s", "=" * 78)
    logger.info("Lois avec des numeros d'article reels : %s", real_laws)
    logger.info("Chunks sans `kind` restants : %s (doit valoir 0)", remaining)
    if failures:
        logger.error("%d loi(s) en echec", failures)
    logger.info("Etape suivante : python scripts/regenerate_embeddings.py --all --force")
    logger.info("%s", "=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
