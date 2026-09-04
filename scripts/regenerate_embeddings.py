"""
Regenere les embeddings des articles.

A lancer apres la migration f5a6b7c8d9e0 : elle remet la colonne
`articles.embedding` a NULL en la repassant en vector(3072), donc tous les
vecteurs existants doivent etre recalcules. Tant que le backfill n'est
pas termine, la recherche semantique ne renvoie rien et l'hybride degrade en
recherche plein texte.

Reprise : par defaut le script ne traite que les articles dont l'embedding est
NULL et progresse par curseur sur l'id. Une interruption ne coute donc qu'un
lot, et une re-execution reprend ou elle s'est arretee.

Usage:
    python scripts/regenerate_embeddings.py --all
    python scripts/regenerate_embeddings.py --law-id 3 12 --batch-size 8
    python scripts/regenerate_embeddings.py --all --dry-run
    python scripts/regenerate_embeddings.py --reindex
"""

import argparse
import logging
import re
import sys
import time
from typing import List, Optional, Sequence

from sqlalchemy import text

sys.path.insert(0, ".")

from app.core.database import SyncSessionLocal, sync_engine  # noqa: E402
from app.services.embedding_service import (  # noqa: E402
    EmbeddingService,
    EmbeddingServiceError,
)

logger = logging.getLogger("regenerate_embeddings")

# Messages d'erreur qui signalent un depassement de quota et non une panne.
QUOTA_PATTERN = re.compile(r"429|RESOURCE_EXHAUSTED|quota", re.IGNORECASE)

MAX_QUOTA_WAIT_SECONDS = 900


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true", help="Traite tout le corpus")
    scope.add_argument("--law-id", type=int, nargs="+", help="Limite a ces lois")
    scope.add_argument(
        "--reindex",
        action="store_true",
        help="Reconstruit l'index HNSW (a faire une fois le backfill termine)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Nombre max d'articles")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcule aussi les articles qui ont deja un embedding",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Pause entre les lots")
    parser.add_argument("--max-quota-waits", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _fetch_batch(session, law_ids, force: bool, cursor: int, size: int) -> List[dict]:
    """
    Lot suivant, par curseur sur l'id.

    Curseur et non OFFSET : les lignes traitees sortent du filtre
    `embedding IS NULL` au fur et a mesure, ce qui decale un OFFSET et fait
    sauter des articles.
    """
    clauses = ["a.id > :cursor"]
    params = {"cursor": cursor, "size": size}

    if not force:
        clauses.append("a.embedding IS NULL")
    if law_ids:
        clauses.append("a.law_id = ANY(:law_ids)")
        params["law_ids"] = list(law_ids)

    sql = text(f"""
        SELECT a.id, a.law_id, a.number, a.content
        FROM articles a
        WHERE {' AND '.join(clauses)}
        ORDER BY a.id
        LIMIT :size
    """)

    rows = session.execute(sql, params).fetchall()
    return [
        {"id": r.id, "law_id": r.law_id, "number": r.number, "content": r.content or ""}
        for r in rows
    ]


def _count_remaining(session, law_ids, force: bool) -> int:
    clauses = ["TRUE"]
    params = {}
    if not force:
        clauses.append("embedding IS NULL")
    if law_ids:
        clauses.append("law_id = ANY(:law_ids)")
        params["law_ids"] = list(law_ids)
    sql = text(f"SELECT count(*) FROM articles WHERE {' AND '.join(clauses)}")
    return int(session.execute(sql, params).scalar() or 0)


def _write_batch(session, ids: List[int], embeddings) -> None:
    """
    Ecrit les vecteurs.

    CAST(:embedding AS vector) sur une CHAINE "[x,y,...]" : une liste Python
    passee a travers text() est adaptee en ARRAY par le pilote, et un ARRAY ne
    se caste pas proprement en vector.
    """
    sql = text("UPDATE articles SET embedding = CAST(:embedding AS vector) WHERE id = :id")
    for article_id, embedding in zip(ids, embeddings):
        # %.6g et non %.7f : a 3072 composantes, le format fixe produit
        # ~31 Ko de texte de requete par UPDATE, pour une precision au-dela de
        # ce que fp32 represente.
        literal = "[" + ",".join(f"{v:.6g}" for v in embedding.tolist()) + "]"
        session.execute(sql, {"embedding": literal, "id": article_id})


def _embed_with_quota_retry(service, texts, batch_size, max_waits):
    """Genere un lot, en attendant si le quota est atteint."""
    for attempt in range(max_waits + 1):
        try:
            return service.generate_batch_embeddings(
                texts=texts, batch_size=batch_size, normalize=True
            )
        except EmbeddingServiceError as exc:
            if not QUOTA_PATTERN.search(str(exc)) or attempt == max_waits:
                raise
            wait = min(60 * (2 ** attempt), MAX_QUOTA_WAIT_SECONDS)
            logger.warning("Quota atteint, reprise du MEME lot dans %ss", wait)
            time.sleep(wait)
    raise EmbeddingServiceError("Quota toujours atteint apres attentes repetees")


def reindex(session) -> None:
    """
    Reconstruit l'index HNSW.

    La migration cree l'index sur une table dont la colonne vient d'etre videe,
    donc sur zero ligne. Apres un backfill, une reconstruction en masse donne un
    graphe de meilleure qualite que les insertions incrementales. CONCURRENTLY
    exige l'autocommit, d'ou la connexion dediee.

    maintenance_work_mem n'est pas decoratif ici : le graphe HNSW de 20 000
    vecteurs halfvec(3072) pese environ 130 Mo, contre 64 Mo de defaut serveur.
    En dessous, pgvector bascule sur une construction disque bien plus lente.
    A relever au-dela de ~40 000 articles.
    """
    logger.info("Reconstruction de l'index HNSW (peut etre long)...")
    with sync_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("SET maintenance_work_mem = '512MB'"))
        conn.execute(text("REINDEX INDEX CONCURRENTLY idx_articles_embedding_hnsw_halfvec"))
    logger.info("Index reconstruit")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.reindex:
        with SyncSessionLocal() as session:
            reindex(session)
        return 0

    law_ids = args.law_id
    max_len = EmbeddingService.MAX_TEXT_LENGTH

    processed = 0
    failed_ids: List[int] = []
    cursor = 0

    with SyncSessionLocal() as session:
        remaining = _count_remaining(session, law_ids, args.force)
        logger.info(
            "%s article(s) a traiter (dimension %s)", remaining, EmbeddingService.EMBEDDING_DIM
        )
        if args.dry_run:
            logger.info("--dry-run : aucun appel a l'API, aucune ecriture")
            return 0
        if remaining == 0:
            return 0

        # Construit APRES le dry-run : le constructeur exige GEMINI_API_KEY, et
        # un compte a blanc n'a aucune raison d'en demander une.
        service = EmbeddingService(use_cache=True)

        while True:
            if args.limit is not None and processed >= args.limit:
                break

            size = args.batch_size
            if args.limit is not None:
                size = min(size, args.limit - processed)

            batch = _fetch_batch(session, law_ids, args.force, cursor, size)
            if not batch:
                break
            cursor = batch[-1]["id"]

            texts = []
            for row in batch:
                content = row["content"]
                if len(content) > max_len:
                    logger.warning(
                        "Article %s (loi %s) tronque : %s > %s caracteres",
                        row["number"], row["law_id"], len(content), max_len,
                    )
                    content = content[:max_len]
                texts.append(content or " ")

            try:
                embeddings = _embed_with_quota_retry(
                    service, texts, args.batch_size, args.max_quota_waits
                )
                _write_batch(session, [r["id"] for r in batch], embeddings)
                # Commit par lot : une interruption brutale ne perd qu'un lot.
                session.commit()
                processed += len(batch)
                logger.info("%s/%s traite(s)", processed, remaining)
            except Exception as exc:
                session.rollback()
                ids = [r["id"] for r in batch]
                failed_ids.extend(ids)
                logger.error("Lot %s ignore : %s", ids, exc)

            if args.sleep:
                time.sleep(args.sleep)

    logger.info("Termine : %s article(s) traite(s)", processed)
    if failed_ids:
        logger.error("%s article(s) en echec : %s", len(failed_ids), failed_ids)
        return 1

    if processed:
        logger.info(
            "Pensez a reconstruire l'index : "
            "python scripts/regenerate_embeddings.py --reindex"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
