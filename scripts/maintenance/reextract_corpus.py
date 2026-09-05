#!/usr/bin/env python3
"""
Re-extrait les documents deja en base avec le moteur Gemini.

POURQUOI. L'extraction precedente ne signalait pas les coupures de page : les
710 articles portent tous `page_number = 1`, et sur les 27 lois, 271 pages de
PDF ne donnent que 27 pages extraites — couverture moyenne 0,60. La consequence
visible est qu'un lien `?article=35` ouvre le PDF a la premiere page.

REPRENABLE PAR CONSTRUCTION. Le cache est indexe par sha256 du fichier : un
document deja extrait n'est jamais repaye, donc relancer le script reprend ou le
quota l'avait arrete. Sur 429, il s'arrete proprement, dit combien de documents
restent, et sort en code 2 — ce qui distingue « quota atteint » de « ca a
plante ».

Le palier gratuit plafonne a 20 appels de generation par jour et par modele.
Un document d'une page coute un appel ; les 72 PDF locaux (481 pages) en
coutent 83.

Usage:
    python scripts/maintenance/reextract_corpus.py --dry-run
    python scripts/maintenance/reextract_corpus.py --apply --max-calls 18
    python scripts/maintenance/reextract_corpus.py --apply --law-id 16

Enchainement complet :
    reextract_corpus.py --apply        # laws.content recoit les vrais <<PAGE:n>>
    rechunk_laws.py --apply            # articles.page_number devient exact
    regenerate_embeddings.py --all --force

Codes de sortie:
    0  termine
    1  au moins un document en echec
    2  quota epuise, relancer plus tard
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.database import SyncSessionLocal
from app.models.law import Law
from app.services.pdf_extraction_service import (
    PdfExtractionError,
    PdfExtractionQuotaError,
    get_pdf_extractor,
)
from app.utils.file_utils import resolve_upload_path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reextract")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-extrait les documents en base avec Gemini.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="n'appelle rien, chiffre le cout en appels (defaut)")
    mode.add_argument("--apply", action="store_true",
                      help="re-extrait et ecrit laws.content")
    parser.add_argument("--law-id", type=int, action="append", dest="law_ids",
                        help="ne traiter que cette loi (repetable)")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="plafond dur d'appels pour cette session — le garde-fou du quota")
    parser.add_argument("--only-multipage", action="store_true",
                        help="ignorer les documents d'une seule page")
    return parser


def _documents(session, law_ids: Optional[List[int]]):
    """Rend [(loi, chemin_du_pdf)] pour les lois dont le fichier existe."""
    query = session.query(Law).order_by(Law.id)
    if law_ids:
        query = query.filter(Law.id.in_(law_ids))

    sortie = []
    for law in query.all():
        if not law.file_id:
            logger.warning("  #%-5s sans file_id, ignoree", law.id)
            continue
        try:
            sortie.append((law, resolve_upload_path(law.file_id)))
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("  #%-5s fichier introuvable (%s)", law.id, exc)
    return sortie


async def _traiter(documents, extracteur, appliquer: bool, max_calls: Optional[int]) -> dict:
    bilan = {"traites": 0, "echecs": 0, "appels": 0, "pages_refusees": 0, "quota": False}

    for law, chemin in documents:
        cout = extracteur.compter_appels(chemin)
        if max_calls is not None and bilan["appels"] + cout > max_calls:
            logger.info(
                "  #%-5s %s appel(s) — plafond de session atteint, arret ici",
                law.id, cout,
            )
            break

        try:
            texte = await extracteur.extract_text(chemin)
        except PdfExtractionQuotaError as exc:
            logger.error("\n⛔ %s", exc)
            bilan["quota"] = True
            break
        except PdfExtractionError as exc:
            logger.error("  #%-5s ECHEC : %s", law.id, exc)
            bilan["echecs"] += 1
            continue

        bilan["appels"] += cout
        bilan["traites"] += 1
        refusees = list(extracteur.pages_refusees)
        bilan["pages_refusees"] += len(refusees)

        marqueurs = texte.count("<<PAGE:")
        detail = f" — pages refusees : {refusees}" if refusees else ""
        logger.info(
            "  #%-5s %5d car, %3d page(s), %s appel(s)%s  %s",
            law.id, len(texte), marqueurs, cout, detail, (law.title or "")[:38],
        )

        if appliquer:
            # Une transaction PAR document : un echec en cours de route ne doit
            # pas laisser la moitie du corpus dans un etat intermediaire.
            with SyncSessionLocal() as session:
                ligne = session.get(Law, law.id)
                if ligne:
                    ligne.content = texte
                    ligne.processing_error = (
                        f"Pages non transcrites (refus du modele) : "
                        f"{', '.join(str(n) for n in refusees)}"
                        if refusees else None
                    )
                    session.commit()

    return bilan


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    extracteur = get_pdf_extractor()

    if not extracteur.is_available():
        logger.error("❌ GEMINI_API_KEY absente : rien a faire.")
        return 1

    with SyncSessionLocal() as session:
        documents = _documents(session, args.law_ids)
        # Detacher les objets : la session se ferme, et le traitement ouvre la
        # sienne par document.
        documents = [
            (session.merge(law), chemin) for law, chemin in documents
        ]
        session.expunge_all()

    if args.only_multipage:
        documents = [
            (law, chemin) for law, chemin in documents
            if extracteur._compter_pages(chemin) > 1
        ]

    total_appels = sum(extracteur.compter_appels(chemin) for _, chemin in documents)
    logger.info("Documents            : %d", len(documents))
    logger.info("Appels necessaires   : %d  (cache non compris)", total_appels)
    logger.info("Soit, a 20 par jour  : %.1f jour(s)", total_appels / 20)
    if args.max_calls:
        logger.info("Plafond de session   : %d appel(s)", args.max_calls)
    logger.info("%s", "-" * 72)

    if not args.apply:
        logger.info("(dry-run : aucun appel emis. Relancez avec --apply.)")
        return 0

    bilan = asyncio.run(_traiter(documents, extracteur, True, args.max_calls))

    logger.info("%s", "=" * 72)
    logger.info("Documents traites    : %d", bilan["traites"])
    logger.info("Appels consommes     : %d", bilan["appels"])
    if bilan["pages_refusees"]:
        logger.info("Pages refusees       : %d  (voir laws.processing_error)",
                    bilan["pages_refusees"])
    if bilan["echecs"]:
        logger.error("Echecs               : %d", bilan["echecs"])

    restants = len(documents) - bilan["traites"] - bilan["echecs"]
    if bilan["quota"]:
        logger.info("\n⛔ Quota epuise. %d document(s) restent.", restants)
        logger.info("   Relancez demain : le cache reprend ou il s'est arrete.")
        return 2
    if restants:
        logger.info("\n%d document(s) restent (plafond de session).", restants)

    logger.info("\nEtape suivante : python scripts/maintenance/rechunk_laws.py --apply")
    logger.info("%s", "=" * 72)
    return 1 if bilan["echecs"] else 0


if __name__ == "__main__":
    sys.exit(main())
