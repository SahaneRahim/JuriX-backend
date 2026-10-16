"""
Regenere les embeddings des articles.

A lancer apres la migration a8b9c0d1e2f3 : elle remet la colonne
`articles.embedding` a NULL en la passant en halfvec(1024), donc tous les
vecteurs existants doivent etre recalcules — ou charges depuis les lots deja
calcules par la mesure (--depuis-lots). Et apres tout changement de
fournisseur — EMBEDDING_PROVIDER, modele : les vecteurs de l'ancien ne se
comparent pas a ceux du nouveau. Tant que le backfill n'est pas termine, la
recherche semantique ne renvoie rien, ou compare deux espaces, et l'hybride
degrade en recherche plein texte.

Selection : par defaut le script traite les articles dont le vecteur manque OU
vient d'un autre fournisseur que celui configure. La colonne
`articles.embedding_model` porte l'empreinte de celui qui l'a produit ; un
vecteur sans empreinte est d'origine inconnue, donc refait. --force recalcule
tout.

Reprise : le script progresse par curseur sur l'id. Une interruption ne coute
donc qu'un lot, et une re-execution reprend ou elle s'est arretee.

Chargement des lots (--depuis-lots) : les vecteurs voyage-4-large du corpus
entier ont deja ete calcules par la mesure (data/mesure_voyage), sur un
instantane des textes. Les charger evite environ 10 M de jetons d'API. Un
vecteur n'est charge que si le texte de l'article en base est ENCORE celui de
l'instantane (meme empreinte SHA-256) : un article modifie depuis garde son
vecteur a NULL, et --all le calculera par l'API.

Usage:
    python scripts/regenerate_embeddings.py --all
    python scripts/regenerate_embeddings.py --law-id 3 12 --batch-size 8
    python scripts/regenerate_embeddings.py --all --dry-run
    python scripts/regenerate_embeddings.py --depuis-lots data/mesure_voyage/lots/voyage-4-large \
        --instantane data/mesure_voyage/morceaux.jsonl [--dry-run]
    python scripts/regenerate_embeddings.py --reindex
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import text

sys.path.insert(0, ".")

from app.core.config import settings
from app.core.database import SyncSessionLocal, sync_engine
from app.services.embedding_service import (
    EmbeddingDefinitiveError,
    EmbeddingService,
    EmbeddingServiceError,
    QuotaExhaustedError,
    empreinte_voyage,
)

logger = logging.getLogger("regenerate_embeddings")

# Messages d'erreur qui signalent un depassement de quota et non une panne.
# 429 en mot entier : un message de Voyage porte des index et des comptes de
# jetons (« index 4290 »), qui ne doivent pas faire attendre un lot.
QUOTA_PATTERN = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|quota", re.IGNORECASE)

MAX_QUOTA_WAIT_SECONDS = 900

# Index HNSW de la colonne, cree par la migration a8b9c0d1e2f3.
INDEX_EMBEDDING = "idx_articles_embedding_hnsw_halfvec1024"

# Ecart de norme tolere pour un vecteur charge : Voyage rend des vecteurs
# unitaires (mesure : 1,000000 sur les 32 303 lots).
TOLERANCE_NORME = 1e-3

# Lignes par UPDATE groupe au chargement (environ 10 Ko de texte par vecteur).
LIGNES_PAR_ECRITURE = 500


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
    scope.add_argument(
        "--depuis-lots",
        type=Path,
        metavar="DOSSIER",
        help="Charge les vecteurs deja calcules (.npz : ids, v) au lieu d'appeler "
        "l'API. Le nom du dossier doit etre celui du modele configure.",
    )
    parser.add_argument(
        "--instantane",
        type=Path,
        metavar="MORCEAUX_JSONL",
        help="Avec --depuis-lots : les textes vectorises (id, texte), pour ne "
        "charger que les articles dont le texte n'a pas change",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Articles par lot (et par commit). Defaut : les lots du fournisseur "
        "(900 textes pour Voyage, decoupes ensuite par jetons), sinon 16",
    )
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
        "(0,5 s pour l'API Gemini, aucune pour Voyage)",
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

    CAST(:embedding AS halfvec) sur une CHAINE "[x,y,...]" : une liste Python
    passee a travers text() est adaptee en ARRAY par le pilote, et un ARRAY ne
    se caste pas proprement en halfvec. La colonne garde le controle de la
    dimension par son typmod : un vecteur de 768 est refuse (« expected 1024
    dimensions, not 768 »).
    """
    sql = text(
        "UPDATE articles SET embedding = CAST(:embedding AS halfvec), "
        "embedding_model = :empreinte WHERE id = :id"
    )
    for article_id, embedding in zip(ids, embeddings):
        session.execute(
            sql, {"embedding": _litteral(embedding), "empreinte": empreinte, "id": article_id}
        )


def _litteral(embedding) -> str:
    # %.6g et non %.7f : six chiffres significatifs quelle que soit la
    # grandeur, quand le format fixe tronque les petites composantes
    # (1.2e-7 s'ecrit 0.0000001). Plus que n'en garde le halfvec (fp16,
    # environ 3 chiffres) : l'arrondi est fait par le serveur, une seule fois.
    return "[" + ",".join(f"{v:.6g}" for v in np.asarray(embedding).tolist()) + "]"


def _empreinte_texte(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def _lire_lots(dossier: Path, dim: int, rapport: Counter) -> Dict[int, np.ndarray]:
    """
    Vecteurs des lots .npz (cles `ids` et `v`), par id d'article.

    Un vecteur de mauvaise dimension, non fini ou non unitaire est ecarte et
    compte ; un id present dans deux lots aussi, faute de savoir lequel croire.
    """
    vecteurs: Dict[int, np.ndarray] = {}
    doublons = set()
    fichiers = sorted(dossier.glob("*.npz"))
    if not fichiers:
        raise SystemExit(f"Aucun lot .npz dans {dossier}")
    for fichier in fichiers:
        with np.load(fichier) as z:
            ids, v = z["ids"], z["v"]
        if v.ndim != 2 or v.shape[1] != dim or len(ids) != len(v):
            raise SystemExit(
                f"{fichier.name} : {len(ids)} ids et des vecteurs {v.shape}, attendu "
                f"(n, {dim}). Lot d'une autre dimension ou d'un autre modele ?"
            )
        for article_id, vecteur in zip(ids.tolist(), v):
            if article_id in vecteurs:
                doublons.add(article_id)
            vecteurs[article_id] = vecteur
    for article_id in doublons:
        del vecteurs[article_id]
        rapport["id_en_double_dans_les_lots"] += 1
    for article_id in list(vecteurs):
        vecteur = vecteurs[article_id]
        if not np.all(np.isfinite(vecteur)) or abs(float(np.linalg.norm(vecteur)) - 1.0) > TOLERANCE_NORME:
            del vecteurs[article_id]
            rapport["vecteur_invalide"] += 1
    return vecteurs


def charger_les_lots(
    session,
    dossier: Path,
    instantane: Path,
    empreinte: str,
    dim: int,
    force: bool = False,
    dry_run: bool = False,
) -> Counter:
    """
    Charge les vecteurs deja calcules, apres verification article par article.

    Un vecteur n'est ecrit que si, en base, l'article existe, est vectorisable
    (embed) et porte EXACTEMENT le texte de l'instantane — coalesce(embed_text,
    content), ce que vectorisent process_law et --all. Sans --force, un article
    deja a jour pour cette empreinte n'est pas reecrit.

    Rend le decompte de ce qui a ete charge, et de ce qui a ete ecarte, par
    motif.
    """
    rapport: Counter = Counter()
    textes: Dict[int, str] = {}
    with open(instantane, encoding="utf-8") as f:
        for ligne in f:
            morceau = json.loads(ligne)
            textes[int(morceau["id"])] = _empreinte_texte(morceau["texte"])

    vecteurs = _lire_lots(dossier, dim, rapport)
    for article_id in [i for i in vecteurs if i not in textes]:
        del vecteurs[article_id]
        rapport["absent_de_l_instantane"] += 1

    ids = sorted(vecteurs)
    rangs = session.execute(
        text(
            "SELECT id, embed, coalesce(embed_text, content) AS texte, "
            f"{_a_refaire()} AS a_refaire "
            "FROM articles WHERE id = ANY(:ids)"
        ),
        {"ids": ids, "empreinte": empreinte},
    ).fetchall()
    en_base = {r.id: r for r in rangs}

    a_ecrire: List[int] = []
    manquants = 0
    for article_id in ids:
        rang = en_base.get(article_id)
        if rang is None:
            rapport["absent_de_la_base"] += 1
        elif not rang.embed:
            rapport["non_vectorisable"] += 1
        elif _empreinte_texte(rang.texte or "") != textes[article_id]:
            rapport["texte_modifie_depuis_l_instantane"] += 1
        elif not force and not rang.a_refaire:
            rapport["deja_a_jour"] += 1
        else:
            a_ecrire.append(article_id)
            manquants += bool(rang.a_refaire)

    if dry_run:
        rapport["a_charger"] = len(a_ecrire)
        # Ceux qui manquaient vraiment : avec --force, a_charger compte aussi
        # des vecteurs deja a jour.
        rapport["a_charger_qui_manquaient"] = manquants
        return rapport

    sql = text(
        "UPDATE articles AS a SET embedding = CAST(v.e AS halfvec), embedding_model = :empreinte "
        "FROM (SELECT unnest(CAST(:ids AS integer[])) AS id, "
        "unnest(CAST(:es AS text[])) AS e) AS v "
        "WHERE a.id = v.id"
    )
    for debut in range(0, len(a_ecrire), LIGNES_PAR_ECRITURE):
        lot = a_ecrire[debut:debut + LIGNES_PAR_ECRITURE]
        session.execute(sql, {
            "ids": lot,
            "es": [_litteral(vecteurs[i]) for i in lot],
            "empreinte": empreinte,
        })
        # Commit par lot : une interruption ne perd qu'un lot, et une reprise
        # saute ce qui est deja a jour.
        session.commit()
        rapport["charges"] += len(lot)
        logger.info("%s/%s vecteur(s) charge(s)", rapport["charges"], len(a_ecrire))
    return rapport


def _embed_with_quota_retry(service, texts, batch_size, max_waits):
    """Genere un lot, en attendant si le quota est atteint."""
    for attempt in range(max_waits + 1):
        try:
            return service.generate_batch_embeddings(
                texts=texts, batch_size=batch_size, normalize=True
            )
        except EmbeddingDefinitiveError:
            # Cle ou requete refusees : attendre ne changerait rien.
            raise
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

    maintenance_work_mem n'est pas decoratif ici : le graphe HNSW de 32 303
    vecteurs halfvec(1024) pese 84 Mo (mesure, pgvector 0.8.6), deja plus que
    les 64 Mo de defaut serveur. En dessous, pgvector bascule sur une
    construction disque bien plus lente. 512 Mo couvrent ~190 000 articles.
    Duree mesuree : environ une minute.

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
            conn.execute(text(f"REINDEX INDEX CONCURRENTLY {INDEX_EMBEDDING}"))
        except Exception:
            conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_EMBEDDING}_ccnew"))
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

    if args.depuis_lots is not None:
        return _main_depuis_lots(args)
    if args.instantane is not None:
        logger.error("--instantane ne sert qu'avec --depuis-lots")
        return 2

    law_ids = args.law_id

    processed = 0
    failed_ids: List[int] = []
    cursor = 0
    service = None

    try:
        with SyncSessionLocal() as session:
            # Avant tout appel paye : sur une base pas encore migree, chaque
            # lot serait vectorise par l'API, puis refuse par la colonne.
            if not _colonne_prete(session):
                return 2
            # Construit AVANT le compte, dry-run compris : ce qui reste a faire
            # depend de l'empreinte du fournisseur configure. La construction
            # ne coute rien — aucun appel reseau — mais exige une
            # configuration valide : la cle du fournisseur.
            # Sans cache : voir process_law._generate_article_embeddings.
            service = EmbeddingService(use_cache=False)
            provider = service.provider
            empreinte = provider.fingerprint
            # Limite du fournisseur, lue sur l'instance.
            max_len = service.MAX_TEXT_LENGTH
            pause = args.sleep if args.sleep is not None else provider.inter_batch_delay_s
            # Des lots de 16 articles, le reglage d'un modele local, enverraient
            # environ 2 000 requetes pour le corpus ; Voyage en accepte 900
            # textes, que le fournisseur redecoupe selon les jetons.
            taille_des_lots = args.batch_size or getattr(provider, "TEXTES_PAR_REQUETE", None) or 16

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

                size = taille_des_lots
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
                        service, texts, taille_des_lots, args.max_quota_waits
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
                except EmbeddingDefinitiveError as exc:
                    session.rollback()
                    if exc.vise_la_configuration:
                        # Cle refusee, droits, modele inconnu : chaque lot
                        # suivant echouerait pareil. S'arreter, plutot que
                        # d'envoyer des milliers de requetes vouees au meme
                        # refus.
                        logger.error(
                            "Refus du fournisseur apres %s article(s) traite(s), arret : %s",
                            processed, exc,
                        )
                        return 2
                    ids = [r["id"] for r in batch]
                    failed_ids.extend(ids)
                    logger.error("Lot %s refuse, ignore : %s", ids, exc)
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
    finally:
        # Dans tous les cas, arret sur quota ou sur refus compris : des lots
        # sont peut-etre deja commites.
        if processed:
            _purger_le_cache_de_recherche()
        fermer = getattr(service, "fermer", None)
        if fermer is not None:
            fermer()


def _colonne_prete(session) -> bool:
    """La colonne a-t-elle le type que le code ecrit ? Sinon, dit pourquoi."""
    attendu = f"halfvec({EmbeddingService.EMBEDDING_DIM})"
    reel = session.execute(text(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'articles'::regclass AND attname = 'embedding'"
    )).scalar()
    if reel != attendu:
        logger.error(
            "articles.embedding est en %s, le code ecrit du %s : appliquer d'abord "
            "les migrations (alembic upgrade head)",
            reel, attendu,
        )
        return False
    return True


def _purger_le_cache_de_recherche() -> None:
    """
    Les reponses mises en cache pendant que les vecteurs manquaient sont du
    plein texte seul : elles seraient resservies jusqu'a cinq minutes encore.

    Un echec ici n'est qu'un avertissement : les vecteurs sont deja commites,
    et le cache expire de lui-meme en cinq minutes.
    """
    try:
        with SyncSessionLocal() as session:
            session.execute(text("DELETE FROM query_cache"))
            session.commit()
        logger.info("Cache des recherches vide : les reponses suivantes utilisent les nouveaux vecteurs")
    except Exception as exc:
        logger.warning("Cache des recherches non vide (il expire en cinq minutes) : %s", exc)


def _main_depuis_lots(args) -> int:
    """
    Charge les lots du dossier, sous l'empreinte du fournisseur configure.

    L'empreinte est CALCULEE par empreinte_voyage(), la fonction qu'utilise
    VoyageProvider, jamais recopiee : un seul caractere d'ecart, et les
    vecteurs charges passeraient pour etrangers (GET /search/health en
    « degraded », et --all les recalculerait tous par l'API).
    """
    if args.instantane is None:
        logger.error("--depuis-lots exige --instantane (le morceaux.jsonl des textes vectorises)")
        return 2
    if settings.EMBEDDING_PROVIDER != "voyage":
        logger.error(
            "Les lots sont des vecteurs Voyage, or EMBEDDING_PROVIDER=%s : les "
            "questions seraient vectorisees dans un autre espace",
            settings.EMBEDDING_PROVIDER,
        )
        return 2
    modele = settings.VOYAGE_EMBEDDING_MODEL
    if args.depuis_lots.name != modele:
        logger.error(
            "Le dossier %s ne porte pas le nom du modele configure (%s) : des "
            "vecteurs d'un autre modele seraient charges sous son empreinte",
            args.depuis_lots, modele,
        )
        return 2

    empreinte = empreinte_voyage(modele)
    with SyncSessionLocal() as session:
        if not _colonne_prete(session):
            return 2
        rapport = charger_les_lots(
            session, args.depuis_lots, args.instantane, empreinte,
            EmbeddingService.EMBEDDING_DIM, force=args.force, dry_run=args.dry_run,
        )
        restants = _count_remaining(session, None, False, empreinte)
        if args.dry_run:
            restants -= rapport["a_charger_qui_manquaient"]
    if rapport["charges"]:
        _purger_le_cache_de_recherche()

    for motif, nombre in sorted(rapport.items()):
        logger.info("%-36s %s", motif, nombre)
    logger.info(
        "%s article(s) vectorisable(s) resteront sans vecteur %s : a calculer par --all",
        restants, empreinte,
    )
    if args.dry_run:
        logger.info("--dry-run : aucune ecriture")
    elif rapport["charges"]:
        logger.info(
            "Pensez a reconstruire l'index : "
            "python scripts/regenerate_embeddings.py --reindex"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
