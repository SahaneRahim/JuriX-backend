#!/usr/bin/env python3
"""
Reclasse les lois existantes dans les 14 domaines canoniques via Groq / Qwen.

Utilise le modèle Groq (qwen/qwen3.8-27b) de manière asynchrone concurrente
pour reclassifier avec précision l'ensemble du corpus sans heuristiques regex.

Usage:
    python scripts/maintenance/reclassify_domains.py --dry-run --limit 20 --explain
    python scripts/maintenance/reclassify_domains.py --dry-run
    python scripts/maintenance/reclassify_domains.py --apply --force
    python scripts/maintenance/reclassify_domains.py --apply --force --concurrency 6
"""

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.database import SyncSessionLocal
from app.models.law import Category, Law
from app.services.category_resolver import load_domain_map
from app.services.legal_domain_classifier import (
    CANONICAL_DOMAINS,
    DomainResult,
    get_legal_domain_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reclassify")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reclasse les lois dans les domaines canoniques via Groq / Qwen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="N'écrit rien, affiche la distribution avant/après (défaut)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Écrit en base (lois sans catégorie, ou toutes avec --force)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Avec --apply : réaffecte aussi les lois ayant déjà une catégorie",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Affiche le domaine et la justification pour chaque document",
    )
    parser.add_argument(
        "--law-id",
        type=int,
        action="append",
        dest="law_ids",
        help="Ne traiter que cette loi (répétable)",
    )
    parser.add_argument(
        "--domain",
        action="append",
        dest="domains",
        help="Filtrer les résultats sur ce domaine cible",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Borne le nombre de lois lues"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Délai en secondes entre requêtes Groq (défaut: 1.5s, évite les 429)",
    )
    return parser


async def main_async(args) -> int:
    apply_changes = args.apply
    classifier = get_legal_domain_classifier()

    with SyncSessionLocal() as session:
        domain_map = load_domain_map(session)

        missing = [d for d in CANONICAL_DOMAINS if d.lower() not in domain_map]
        if missing:
            logger.error("❌ Domaines absents de la table categories : %s", ", ".join(missing))
            logger.error("   Lancez d'abord : alembic upgrade head")
            return 2

        id_to_name = {
            row[1]: row[0]
            for row in session.execute(
                __import__("sqlalchemy").select(Category.name, Category.id)
            ).all()
        }

        query = session.query(Law).order_by(Law.id)
        if args.law_ids:
            query = query.filter(Law.id.in_(args.law_ids))
        if args.limit:
            query = query.limit(args.limit)
        laws = query.all()

        total_laws = len(laws)
        logger.info(f"📋 {total_laws} documents à classer via Groq / Qwen (délai: {args.delay}s)...")

        t0 = time.time()
        results_by_id = {}

        for i, law in enumerate(laws, 1):
            try:
                res = await classifier.classify_async(
                    law.title or "",
                    (law.content or "")[:1500],
                    law.type or "",
                )
            except Exception as e:
                logger.warning(f"❌ Erreur sur #{law.id} : {e}")
                res = DomainResult(
                    domain="Droit Administratif",
                    confidence=0.10,
                    rule=f"error:{e}",
                    source="default",
                )

            results_by_id[law.id] = res

            if args.explain:
                marker = "→" if domain_map.get(res.domain.lower()) != law.category_id else " ="
                logger.info(
                    "  #%-5s %s %-45s [%.2f] %s",
                    law.id,
                    marker,
                    res.domain[:45],
                    res.confidence,
                    (law.title or "")[:70],
                )
            elif i % 10 == 0 or i == total_laws:
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                logger.info(f"  Progression: {i}/{total_laws} ({i/total_laws*100:.1f}%) - {speed:.1f} doc/s")

            if args.delay > 0 and i < total_laws:
                await asyncio.sleep(args.delay)

        elapsed_total = time.time() - t0
        logger.info(f"⚡ Classification terminée en {elapsed_total:.2f}s ({total_laws/elapsed_total:.1f} doc/s)\n")

        before = Counter()
        after = Counter()
        rules = Counter()
        moved = []
        protected = []

        for law in laws:
            current_name = id_to_name.get(law.category_id, "(nulle)")
            before[current_name] += 1

            result = results_by_id.get(law.id)
            if not result:
                continue

            if args.domains and result.domain not in args.domains:
                after[current_name] += 1
                continue

            after[result.domain] += 1
            rules[result.rule] += 1

            target_id = domain_map.get(result.domain.lower())
            if not target_id:
                logger.warning(f"⚠️ Domaine non résolu : {result.domain}")
                continue

            suggested = [
                domain_map[name.lower()]
                for name in [result.domain, *(d for d, _ in result.runners_up)]
                if name.lower() in domain_map
            ]

            changes = target_id != law.category_id
            if changes:
                if law.category_id is not None and not args.force:
                    protected.append((law.id, current_name, result.domain))
                else:
                    moved.append((law.id, current_name, result.domain, law.title or ""))

            if args.explain:
                marker = "→" if changes else " ="
                logger.info(
                    "  #%-5s %s %-45s [%.2f] %s",
                    law.id,
                    marker,
                    result.domain[:45],
                    result.confidence,
                    (law.title or "")[:70],
                )

            if apply_changes:
                law.category_confidence = result.confidence
                if suggested:
                    law.suggested_categories = suggested
                if law.category_id is None or args.force:
                    law.category_id = target_id

        if apply_changes:
            session.commit()
            logger.info("💾 Modifications enregistrées avec succès en base de données.")

    # ==================== RAPPORT ====================
    logger.info("\n" + "=" * 72)
    logger.info("Lois lues : %d", len(laws))

    logger.info("\nDistribution AVANT :")
    for name, count in before.most_common():
        logger.info("  %-55s %5d", name, count)

    logger.info("\nDistribution APRÈS :")
    for name, count in after.most_common():
        logger.info("  %-55s %5d", name, count)

    if moved:
        logger.info("\nChangements prévus/effectués : %d", len(moved))
        for lid, prev, new_dom, title in moved[:15]:
            logger.info("  #%-5s %s  ->  %s  (%s)", lid, prev, new_dom, title[:50])
        if len(moved) > 15:
            logger.info("  ... et %d autres documents", len(moved) - 15)

    if protected:
        logger.info(
            "\nDocuments protégés (gardent leur catégorie actuelle, utiliser --force pour écraser) : %d",
            len(protected),
        )

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
