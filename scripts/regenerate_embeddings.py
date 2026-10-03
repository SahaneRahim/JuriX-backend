"""
Regenere les embeddings des articles.

A lancer apres la migration c4d5e6f7a8b9 : elle remet la colonne
`articles.embedding` a NULL en la passant en vector(768), donc tous les
vecteurs existants doivent etre recalcules. Et apres tout changement de
fournisseur — EMBEDDING_PROVIDER, modele, revision, gabarits : les vecteurs de
l'ancien ne se comparent pas a ceux du nouveau. Tant que le backfill n'est pas
termine, la recherche semantique ne renvoie rien, ou compare deux espaces, et
l'hybride degrade en recherche plein texte.

Selection : par defaut le script traite les articles dont le vecteur manque OU
vient d'un autre fournisseur que celui configure. La colonne
`articles.embedding_model` porte l'empreinte de celui qui l'a produit ; un
vecteur sans empreinte est d'origine inconnue, donc refait. --force recalcule
tout.

Reprise : le script progresse par curseur sur l'id. Une interruption ne coute
donc qu'un lot, et une re-execution reprend ou elle s'est arretee.

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

from app.core.database import SyncSessionLocal, sync_engine
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    QuotaExhaustedError,
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
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Pause entre les lots, en secondes. Defaut : celle du fournisseur "
        "(0,5 s pour l'API Gemini, aucune en local)",
    )
    parser.add_argument("--max-quota-waits", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


# Ce qui reste a faire : pas de vecteur, ou un vecteur d'un autre fournisseur.
# IS DISTINCT FROM et non `<>` : une empreinte NULL — vecteur d'origine
# inconnue — doit etre refaite, or `NULL <> 'x'` vaut NULL, donc faux dans un
# WHERE. Le vecteur serait passe pour bon.
def _a_refaire(prefixe: str = "") -> str:
    return (
        f"({prefixe}embedding IS NULL "
        f"OR {prefixe}embedding_model IS DISTINCT FROM :empreinte)"
    )


def _fetch_batch(
    session, law_ids, force: bool, empreinte: str, cursor: int, size: int
) -> List[dict]:
    """
    Lot suivant, par curseur sur l'id.

    Curseur et non OFFSET : les lignes traitees sortent du filtre au fur et a
    mesure, ce qui decale un OFFSET et fait sauter des articles.
    """
    # `a.embed IS TRUE` : le pipeline d'ingestion ne vectorise que les chunks
    # que le raffineur a juges vectorisables (process_law._generate_article_embeddings).
    # Sans ce filtre, ce script depensait du quota sur les visas, les formules
    # d'execution et les fragments, que l'ingestion exclut deliberement.
    clauses = ["a.id > :cursor", "a.embed IS TRUE"]
    params = {"cursor": cursor, "size": size}

    if not force:
        clauses.append(_a_refaire("a."))
        params["empreinte"] = empreinte
    if law_ids:
        clauses.append("a.law_id = ANY(:law_ids)")
        params["law_ids"] = list(law_ids)

    sql = text(f"""
        -- coalesce(embed_text, content) : c'est EXACTEMENT ce qu'envoie
        -- process_law._generate_article_embeddings. Vectoriser `content` ici
        -- placerait le meme article a deux endroits differents de l'espace
        -- d'embedding selon le chemin de code qui l'a ecrit.
        SELECT a.id, a.law_id, a.number, coalesce(a.embed_text, a.content) AS content
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


def _count_remaining(session, law_ids, force: bool, empreinte: str) -> int:
    # Meme filtre que _fetch_batch, sinon la progression affichee ment : elle
    # comptait des chunks que le lot suivant n'irait jamais chercher.
    clauses = ["embed IS TRUE"]
    params = {}
    if not force:
        clauses.append(_a_refaire())
        params["empreinte"] = empreinte
    if law_ids:
        clauses.append("law_id = ANY(:law_ids)")
        params["law_ids"] = list(law_ids)
    sql = text(f"SELECT count(*) FROM articles WHERE {' AND '.join(clauses)}")
    return int(session.execute(sql, params).scalar() or 0)


def _write_batch(session, ids: List[int], embeddings, empreinte: str) -> None:
    """
    Ecrit les vecteurs, et leur provenance dans la meme instruction.

    CAST(:embedding AS vector) sur une CHAINE "[x,y,...]" : une liste Python
    passee a travers text() est adaptee en ARRAY par le pilote, et un ARRAY ne
    se caste pas proprement en vector.
    """
    sql = text(
        "UPDATE articles SET embedding = CAST(:embedding AS vector), "
        "embedding_model = :empreinte WHERE id = :id"
    )
    for article_id, embedding in zip(ids, embeddings):
        # %.6g et non %.7f : six chiffres significatifs quelle que soit la
        # grandeur, quand le format fixe tronque les petites composantes
        # (1.2e-7 s'ecrit 0.0000001). La longueur est la meme : ~8 Ko de texte
        # par UPDATE a 768 composantes, mesure.
        literal = "[" + ",".join(f"{v:.6g}" for v in embedding.tolist()) + "]"
        session.execute(
            sql, {"embedding": literal, "empreinte": empreinte, "id": article_id}
        )


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
    vecteurs vector(768) pese 78 Mo (mesure), deja plus que les 64 Mo de
    defaut serveur. En dessous, pgvector bascule sur une construction disque
    bien plus lente. 512 Mo couvrent ~130 000 articles.

    Construction SANS processus paralleles. En parallele, pgvector place le
    graphe en memoire partagee dynamique, donc dans /dev/shm, qu'un conteneur
    Docker limite a 64 Mo par defaut. Constate sur le corpus (20 394
    vecteurs) : « could not resize shared memory segment ... No space left on
    device ». En serie, le graphe reste dans la memoire du processus.

    Un REINDEX CONCURRENTLY interrompu laisse une copie INVALIDE de l'index
    (suffixe _ccnew), que chaque ecriture continue pourtant de maintenir : elle
    est supprimee avant de propager l'erreur.
    """
    logger.info("Reconstruction de l'index HNSW (peut etre long)...")
    with sync_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("SET maintenance_work_mem = '512MB'"))
        conn.execute(text("SET max_parallel_maintenance_workers = 0"))
        try:
            conn.execute(text("REINDEX INDEX CONCURRENTLY idx_articles_embedding_hnsw_vector"))
        except Exception:
            conn.execute(text(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_articles_embedding_hnsw_vector_ccnew"
            ))
            raise
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
        # Construit AVANT le compte, dry-run compris : ce qui reste a faire
        # depend de l'empreinte du fournisseur configure. La construction ne
        # coute rien — aucun appel reseau, le modele local n'est charge qu'au
        # premier encodage — mais exige une configuration valide :
        # GEMINI_API_KEY pour gemini, les fichiers du modele pour gemma.
        # Sans cache : voir process_law._generate_article_embeddings.
        service = EmbeddingService(use_cache=False)
        provider = service.provider
        empreinte = provider.fingerprint
        pause = args.sleep if args.sleep is not None else provider.inter_batch_delay_s

        remaining = _count_remaining(session, law_ids, args.force, empreinte)
        logger.info(
            "%s article(s) a traiter par %s (dimension %s)",
            remaining, provider.label, EmbeddingService.EMBEDDING_DIM,
        )
        if args.dry_run:
            logger.info("--dry-run : aucun encodage, aucune ecriture")
            return 0
        if remaining == 0:
            return 0

        while True:
            if args.limit is not None and processed >= args.limit:
                break

            size = args.batch_size
            if args.limit is not None:
                size = min(size, args.limit - processed)

            batch = _fetch_batch(session, law_ids, args.force, empreinte, cursor, size)
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
                _write_batch(session, [r["id"] for r in batch], embeddings, empreinte)
                # Commit par lot : une interruption brutale ne perd qu'un lot.
                session.commit()
                processed += len(batch)
                logger.info("%s/%s traite(s)", processed, remaining)
            except QuotaExhaustedError as exc:
                # Le quota journalier ne se recharge pas : continuer ferait
                # echouer tous les lots suivants, un par un, pour rien. On
                # s'arrete en disant exactement ou on en est.
                session.rollback()
                remaining = _count_remaining(session, law_ids, args.force, empreinte)
                logger.error(
                    "Quota journalier epuise apres %s article(s) traite(s). "
                    "Il en reste %s. Le quota se reinitialise a minuit, heure du "
                    "Pacifique ; relancer la meme commande reprendra ou elle "
                    "s'est arretee. Detail : %s",
                    processed, remaining, exc,
                )
                return 2
            except Exception as exc:
                session.rollback()
                ids = [r["id"] for r in batch]
                failed_ids.extend(ids)
                logger.error("Lot %s ignore : %s", ids, exc)

            if pause:
                time.sleep(pause)

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
