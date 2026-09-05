#!/usr/bin/env python3
"""
Reclasse les lois existantes dans les 14 domaines juridiques canoniques.

Aucun retraitement, aucun appel reseau : le script lit `title`, `type` et
`content`, et n'ecrit que `category_id`, `category_confidence` et
`suggested_categories`. Il est donc relançable autant de fois que voulu —
l'idempotence est structurelle, le classifieur etant une fonction pure de
(titre, contenu, type) et le script n'ecrivant aucun des trois.

Usage:
    python scripts/maintenance/reclassify_domains.py --dry-run
    python scripts/maintenance/reclassify_domains.py --dry-run --explain
    python scripts/maintenance/reclassify_domains.py --apply
    python scripts/maintenance/reclassify_domains.py --apply --force
    python scripts/maintenance/reclassify_domains.py --dry-run --domain "Fonction Publique"

Codes de sortie:
    0  termine
    1  erreur inattendue
    2  la table `categories` ne contient pas les 14 domaines canoniques
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.database import SyncSessionLocal  # noqa: E402
from app.models.law import Law  # noqa: E402
from app.services.category_resolver import load_domain_map  # noqa: E402
from app.services.legal_domain_classifier import (  # noqa: E402
    CANONICAL_DOMAINS,
    get_legal_domain_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reclassify")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reclasse les lois dans les domaines juridiques canoniques.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="n'ecrit rien, affiche la distribution avant/apres (defaut)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="ecrit en base, mais SEULEMENT les lois dont category_id est nulle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="avec --apply : ECRASE AUSSI les categories choisies par un administrateur",
    )
    parser.add_argument("--explain", action="store_true",
                        help="affiche la regle qui a decide, document par document")
    parser.add_argument("--law-id", type=int, action="append", dest="law_ids",
                        help="ne traiter que cette loi (repetable)")
    parser.add_argument("--domain", action="append", dest="domains",
                        help="n'afficher que les lois classees dans ce domaine (repetable)")
    parser.add_argument("--limit", type=int, default=None, help="borne le nombre de lois lues")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_changes = args.apply
    classifier = get_legal_domain_classifier()

    with SyncSessionLocal() as session:
        domain_map = load_domain_map(session)

        # Garde-fou : on refuse d'ecrire tant que la table n'est pas alignee sur
        # le code. Sans cette verification, --apply mettrait a NULL les lois des
        # domaines manquants.
        missing = [d for d in CANONICAL_DOMAINS if d.lower() not in domain_map]
        if missing:
            logger.error("❌ Domaines absents de la table categories : %s", ", ".join(missing))
            logger.error("   Lancez d'abord : alembic upgrade head")
            return 2

        id_to_name = {identifier: name for name, identifier in
                      session.execute(__import__("sqlalchemy").select(
                          __import__("app.models.law", fromlist=["Category"]).Category.name,
                          __import__("app.models.law", fromlist=["Category"]).Category.id,
                      )).all()}

        query = session.query(Law)
        if args.law_ids:
            query = query.filter(Law.id.in_(args.law_ids))
        if args.limit:
            query = query.limit(args.limit)
        laws = query.order_by(Law.id).all()

        before = Counter()
        after = Counter()
        rules = Counter()
        moved = []
        protected = []
        low_confidence = []

        for law in laws:
            current_name = id_to_name.get(law.category_id, "(nulle)")
            before[current_name] += 1

            result = classifier.classify(law.title or "", law.content or "", law.type)
            if args.domains and result.domain not in args.domains:
                after[current_name] += 1
                continue

            after[result.domain] += 1
            rules[result.rule] += 1
            if result.confidence < 0.30:
                low_confidence.append((law.id, law.title or "", result))

            target_id = domain_map[result.domain.lower()]
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
                    "  #%-5s %s %-40s [%s] %.2f  %s",
                    law.id, marker, result.domain[:40], result.rule,
                    result.confidence, (law.title or "")[:60],
                )

            if apply_changes:
                law.category_confidence = result.confidence
                if suggested:
                    law.suggested_categories = suggested
                if law.category_id is None or args.force:
                    law.category_id = target_id

        if apply_changes:
            session.commit()

    # ==================== RAPPORT ====================
    logger.info("\n%s", "=" * 72)
    logger.info("Lois lues : %d", len(laws))

    logger.info("\nAvant :")
    for name, count in before.most_common():
        logger.info("  %-52s %5d", name, count)

    logger.info("\nApres :")
    for name, count in after.most_common():
        logger.info("  %-52s %5d", name, count)

    logger.info("\nRegles declenchees :")
    for rule, count in rules.most_common(15):
        logger.info("  %-32s %5d", rule, count)

    if low_confidence:
        logger.info("\n⚠️  %d loi(s) a faible confiance (aucune regle de titre) :", len(low_confidence))
        for law_id, title, result in low_confidence[:15]:
            logger.info("  #%-5s %-46s %s", law_id, title[:46], result.rule)

    if protected:
        logger.info(
            "\n🔒 %d loi(s) conservent la categorie choisie par un administrateur "
            "(--force pour les ecraser) :", len(protected)
        )
        for law_id, current, proposed in protected[:15]:
            logger.info("  #%-5s %s (propose : %s)", law_id, current, proposed)

    verb = "deplacees" if apply_changes else "a deplacer"
    logger.info("\n%d loi(s) %s :", len(moved), verb)
    for law_id, current, proposed, title in moved[:30]:
        logger.info("  #%-5s %-24s -> %-40s %s", law_id, current[:24], proposed[:40], title[:40])
    if len(moved) > 30:
        logger.info("  ... et %d autres", len(moved) - 30)

    if not apply_changes:
        logger.info("\n(dry-run : rien n'a ete ecrit. Relancez avec --apply.)")
    logger.info("%s", "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
