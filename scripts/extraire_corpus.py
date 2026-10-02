"""
Passe 1 de l'ingestion : extraction de TOUT le corpus par Docling, avec reprise.

Ne touche pas a la base. Chaque document est converti par lots de pages, chaque
lot ecrit en cache des qu'il est fait (app/services/docling_extraction.py) : une
interruption ne coute que le lot en cours, et une relance reprend exactement ou
l'extraction s'etait arretee. Rien n'est jamais converti deux fois.

ISOLATION. Le travail se fait dans un PROCESSUS ENFANT, recycle :
- apres --docs-par-processus documents, ou des que sa memoire depasse
  --rss-max-mo. Docling garde des caches et de la memoire native qui
  grossissent au fil des documents : mesure, 1,1 Go -> 4,6 Go en une centaine
  de pages ;
- s'il MEURT (memoire epuisee, plantage natif), le superviseur lit le lot qu'il
  avait en cours (`en_cours.json`) et lui compte une tentative perdue. Au bout
  de TENTATIVES_MAX, ce lot est repris page a page, puis chaque page fautive
  est abandonnee a son tour : une page toxique ne tue pas l'ingestion entiere,
  n'emporte pas les pages saines de son lot, et elle est signalee a
  l'indexation.

ORDRE. Les petits documents d'abord : la moitie du corpus tient en une page,
et l'avancement se voit des la premiere heure. Les plus gros — jusqu'a 169
pages, et 11 fichiers de plus de 50 Mo — passent a la fin, sans aucune limite
de taille.

ARRET PROPRE. `touch data/ingestion/STOP` (ou SIGTERM, ou Ctrl-C) : le lot en
cours se termine, puis tout s'arrete. Relancer la meme commande reprend.

A lancer DETACHE, pour survivre a la fermeture du terminal et de la session :

    setsid nohup .venv/bin/python scripts/extraire_corpus.py \\
        > data/ingestion/extraction.out 2>&1 &

Suivi : data/ingestion/extraction.log, et data/ingestion/progression.json ou
`python scripts/extraire_corpus.py --etat`.

Usage:
    python scripts/extraire_corpus.py                   # tout le corpus
    python scripts/extraire_corpus.py --doc-id 291 9866
    python scripts/extraire_corpus.py --limite 20       # pilote
    python scripts/extraire_corpus.py --etat            # avancement seul
"""

import argparse
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Les reglages (.env) et les caches (./data/...) se resolvent depuis le
# repertoire courant : on s'y place, ou que la commande ait ete lancee.
RACINE = Path(__file__).resolve().parents[1]
os.chdir(RACINE)
sys.path.insert(0, str(RACINE))

from app.services.docling_extraction import (
    DoclingPdfExtractor,
    compter_pages,
)

logger = logging.getLogger("extraire_corpus")

INDEX_PAR_DEFAUT = Path(
    os.environ.get("JURIX_CORPUS_INDEX", "/home/rahim/jurix/documents_index.json")
)
DOSSIER_SUIVI = RACINE / "data" / "ingestion"


@dataclass
class Document:
    doc_id: str
    chemin: str
    titre: str
    sha256: str
    pages: int


# ==================== CORPUS ====================


def charger_corpus(index: Path, dossier: Path) -> List[Document]:
    """
    Les documents de l'index, avec leur nombre de pages.

    Le nombre de pages est mis en cache (`pages.json`, par sha256) : le
    recompter a chaque relance coute la lecture des 5 Go du corpus. Le sha256
    vient de l'index, recalcule et verifie a l'identique sur le disque.
    """
    entrees = json.loads(index.read_text(encoding="utf-8"))
    cache_pages_chemin = dossier / "pages.json"
    try:
        cache_pages: Dict[str, int] = json.loads(cache_pages_chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache_pages = {}

    documents = []
    for e in entrees:
        chemin = Path(e["path"])
        if not chemin.is_file():
            logger.error("PRC-%s : fichier absent (%s)", e["doc_id"], chemin)
            continue
        sha = e["sha256"]
        if sha not in cache_pages:
            cache_pages[sha] = compter_pages(chemin)
        documents.append(Document(
            doc_id=str(e["doc_id"]), chemin=str(chemin), titre=e.get("title", ""),
            sha256=sha, pages=cache_pages[sha],
        ))

    dossier.mkdir(parents=True, exist_ok=True)
    cache_pages_chemin.write_text(json.dumps(cache_pages), encoding="utf-8")
    return documents


def selectionner(
    documents: List[Document], doc_ids: Optional[Sequence[str]], limite: Optional[int]
) -> List[Document]:
    """Les petits documents d'abord ; filtre et limite eventuels."""
    if doc_ids:
        voulus = {str(d) for d in doc_ids}
        documents = [d for d in documents if d.doc_id in voulus]
    documents = sorted(documents, key=lambda d: (d.pages, int(d.doc_id) if d.doc_id.isdigit() else 0))
    return documents[:limite] if limite else documents


# ==================== PROCESSUS DE TRAVAIL ====================


def _rss_mo() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return 0


def configurer_journal(chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    racine = logging.getLogger()
    racine.setLevel(logging.INFO)
    if not any(getattr(h, "_jurix", False) for h in racine.handlers):
        for handler in (logging.FileHandler(chemin, encoding="utf-8"), logging.StreamHandler()):
            handler._jurix = True
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(process)d] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
            ))
            racine.addHandler(handler)
    # Docling et RapidOCR journalisent chaque initialisation en INFO.
    for bavard in ("docling", "rapidocr", "RapidOCR", "transformers", "httpx"):
        logging.getLogger(bavard).setLevel(logging.WARNING)


# Lots en echec d'affilee, sur des documents differents, au-dela desquels le
# processus s'arrete : une meme panne partout (GPU indisponible, modele casse)
# n'est pas l'echec de ces lots, et les compter les abandonnerait tous.
ECHECS_EN_SERIE_MAX = 5

# Codes de sortie du processus de travail, lus par le superviseur
SORTIE_PANNE_INIT = 4
SORTIE_PANNE_EN_SERIE = 5
SORTIE_AUTRE_CACHE = 6


def _terminer(code: int) -> None:
    """
    Sortie franche du processus de travail.

    os._exit et non return : un lot depasse par le delai laisse derriere lui
    des fils que Docling abandonne sans les arreter. Une sortie normale les
    attendrait, et le processus ne finirait jamais.
    """
    logging.shutdown()
    os._exit(code)


def _processus_de_travail(
    groupe: List[Dict], dossier: str, rss_max_mo: int, cache_attendu: Optional[str] = None
) -> None:
    """Point d'entree du processus enfant : travaille, puis sort franchement."""
    _terminer(travailler(groupe, dossier, rss_max_mo, cache_attendu))


def travailler(
    groupe: List[Dict], dossier: str, rss_max_mo: int, cache_attendu: Optional[str] = None
) -> int:
    """
    Corps du processus enfant : convertit les lots restants d'un groupe.

    Returns:
        le code de sortie du processus (0, SORTIE_PANNE_INIT,
        SORTIE_PANNE_EN_SERIE, SORTIE_AUTRE_CACHE)

    Ignore SIGINT : un Ctrl-C au terminal atteint tout le groupe de processus,
    et tuerait la conversion en cours, comptee alors comme une tentative
    perdue. L'arret passe par le fichier STOP, que le superviseur pose.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    dossier = Path(dossier)
    configurer_journal(dossier / "extraction.log")
    extracteur = DoclingPdfExtractor()
    stop = dossier / "STOP"
    en_cours = dossier / "en_cours.json"

    # L'enfant relit .env et les versions des paquets. Si l'un a change depuis
    # le lancement du superviseur — un `pip install` ou une modification de
    # .env pendant les 15 heures —, l'enfant ecrirait dans un AUTRE cache :
    # le superviseur ne verrait aucun progres et relancerait le meme groupe
    # sans fin, sans jamais compter de mort.
    if cache_attendu and str(extracteur.dossier_cache) != cache_attendu:
        logger.error(
            "Le processus de travail ecrirait dans %s, le superviseur lit %s : "
            ".env ou paquets modifies depuis le lancement. Relancer l'extraction.",
            extracteur.dossier_cache, cache_attendu,
        )
        return SORTIE_AUTRE_CACHE

    # Les modeles AVANT le premier lot, et hors de tout lot en cours : une
    # panne d'initialisation (memoire du GPU, poids introuvables) tue le
    # processus sans qu'aucun lot ne la paie. Le superviseur s'arrete apres
    # quelques morts de ce genre, au lieu d'abandonner tout le corpus.
    try:
        extracteur.precharger()
    except Exception as e:
        logger.error("Initialisation de Docling impossible : %s: %s", type(e).__name__, e)
        return SORTIE_PANNE_INIT

    echecs_en_serie = 0
    for d in groupe:
        chemin = Path(d["chemin"])
        for debut, fin in extracteur.etat(chemin, d["sha256"], d["pages"]).lots_a_faire:
            if stop.exists():
                logger.info("Arret demande : fin du processus de travail")
                return 0
            en_cours.write_text(json.dumps({**d, "debut": debut, "fin": fin}), encoding="utf-8")
            t0 = time.perf_counter()
            lot = extracteur.traiter_lot(chemin, d["sha256"], debut, fin)
            en_cours.unlink(missing_ok=True)
            logger.info(
                "PRC-%s pages %s-%s/%s : %s en %.1f s (tentative %s, rss %s Mo)",
                d["doc_id"], debut, fin, d["pages"], lot.get("statut"),
                time.perf_counter() - t0, lot.get("tentatives"), _rss_mo(),
            )
            echecs_en_serie = echecs_en_serie + 1 if lot.get("statut") != "ok" else 0
            if echecs_en_serie >= ECHECS_EN_SERIE_MAX:
                logger.error(
                    "%s lots en echec d'affilee (dernier : %s) : panne generale presumee",
                    echecs_en_serie, (lot.get("erreurs") or ["?"])[-1],
                )
                return SORTIE_PANNE_EN_SERIE
            if _rss_mo() > rss_max_mo:
                logger.info("Memoire au-dela de %s Mo : recyclage du processus", rss_max_mo)
                return 0
    return 0


def lancer_processus(
    groupe: List[Document], dossier: Path, rss_max_mo: int,
    delai_lot_s: float = 900.0, delai_init_s: float = 1200.0,
    marge_s: float = 600.0, pas_s: float = 30.0, cible: Callable = None,
    cache_attendu: Optional[str] = None,
) -> int:
    """
    Un processus enfant NEUF (spawn : CUDA ne survit pas a un fork), sous garde.

    Le chien de garde tue le processus si un lot depasse son delai d'une large
    marge (Docling bloque, fils qui ne rendent pas la main), ou si aucun lot n'a
    commence apres delai_init_s (initialisation figee). Sans lui, un seul
    blocage arretait toute l'ingestion, en silence, pour la nuit.
    """
    contexte = multiprocessing.get_context("spawn")
    processus = contexte.Process(
        target=cible or _processus_de_travail,
        args=([asdict(d) for d in groupe], str(dossier), rss_max_mo),
        kwargs={"cache_attendu": cache_attendu} if cache_attendu else {},
    )
    en_cours = dossier / "en_cours.json"
    depart = time.time()
    lot_vu = False
    processus.start()
    while True:
        processus.join(timeout=pas_s)
        if processus.exitcode is not None:
            return processus.exitcode
        maintenant = time.time()
        try:
            age = maintenant - en_cours.stat().st_mtime
            lot_vu = True
        except OSError:
            age = None
        if age is not None and age > delai_lot_s + marge_s:
            logger.error("Lot bloque depuis %.0f s : processus tue", age)
        elif not lot_vu and maintenant - depart > delai_init_s:
            logger.error("Aucun lot commence en %.0f s : processus tue", maintenant - depart)
        else:
            continue
        processus.kill()
        processus.join()
        return processus.exitcode if processus.exitcode is not None else -9


# ==================== SUPERVISION ====================


def ecrire_progression(
    documents: List[Document], extracteur: DoclingPdfExtractor, dossier: Path,
    debut_session: float, pages_au_depart: int, morts: int, lances: int,
) -> Dict:
    termines = faits = abandonnes = a_faire = pages_faites = 0
    for d in documents:
        etat = extracteur.etat(Path(d.chemin), d.sha256, d.pages)
        termines += etat.termine
        faits += len(etat.lots_faits)
        abandonnes += len(etat.lots_abandonnes)
        a_faire += len(etat.lots_a_faire)
        pages_faites += etat.nb_pages - etat.pages_restantes

    total_pages = sum(d.pages for d in documents)
    ecoule_h = max((time.time() - debut_session) / 3600, 1e-6)
    rythme = (pages_faites - pages_au_depart) / ecoule_h
    restant_h = (total_pages - pages_faites) / rythme if rythme > 0 else None
    progression = {
        "mise_a_jour": time.strftime("%Y-%m-%d %H:%M:%S"),
        "documents": {"total": len(documents), "termines": termines,
                      "restants": len(documents) - termines},
        "lots": {"faits": faits, "abandonnes": abandonnes, "a_faire": a_faire},
        "pages": {"total": total_pages, "faites": pages_faites},
        "rythme_pages_par_heure": round(rythme, 1),
        "heures_restantes_estimees": round(restant_h, 1) if restant_h else None,
        "processus": {"lances": lances, "morts": morts},
    }
    dossier.mkdir(parents=True, exist_ok=True)
    tmp = dossier / "progression.json.tmp"
    tmp.write_text(json.dumps(progression, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(dossier / "progression.json")
    return progression


def superviser(
    documents: List[Document],
    extracteur: DoclingPdfExtractor,
    dossier: Path,
    docs_par_processus: int = 25,
    rss_max_mo: int = 5000,
    lancer: Optional[Callable[[List[Document], Path, int], int]] = None,
    morts_consecutives_max: int = 3,
) -> int:
    """
    Relance des processus de travail jusqu'a ce que chaque lot soit fait ou
    abandonne.

    Returns:
        0 si tout est extrait, 1 sur arret demande, 2 si les processus meurent
        en boucle sans laisser de lot en cours (panne generale : pilote CUDA,
        modeles introuvables...) ou n'ecrivent plus dans le cache lu ici
        (.env ou paquets modifies en cours de route) — inutile d'insister.
    """
    if lancer is None:
        cache = str(extracteur.dossier_cache)

        def lancer(groupe, dossier_, rss):
            return lancer_processus(
                groupe, dossier_, rss, delai_lot_s=extracteur.timeout_lot_s, cache_attendu=cache,
            )
    stop = dossier / "STOP"
    en_cours = dossier / "en_cours.json"
    debut_session = time.time()
    pages_au_depart = None
    morts = lances = morts_consecutives = 0

    while True:
        restants = [
            d for d in documents
            if not extracteur.etat(Path(d.chemin), d.sha256, d.pages).termine
        ]
        progression = ecrire_progression(
            documents, extracteur, dossier, debut_session,
            pages_au_depart or 0, morts, lances,
        )
        if pages_au_depart is None:
            pages_au_depart = progression["pages"]["faites"]
            debut_session = time.time()
        logger.info(
            "Avancement : %s/%s documents, %s/%s pages, %s lots abandonnes",
            progression["documents"]["termines"], progression["documents"]["total"],
            progression["pages"]["faites"], progression["pages"]["total"],
            progression["lots"]["abandonnes"],
        )
        if not restants:
            logger.info("Extraction terminee")
            return 0
        if stop.exists():
            logger.info("Arret demande : relancer la meme commande pour reprendre")
            return 1

        lances += 1
        code = lancer(restants[:docs_par_processus], dossier, rss_max_mo)
        if code == 0:
            morts_consecutives = 0
            continue
        if code == SORTIE_AUTRE_CACHE:
            return 2

        morts += 1
        try:
            marque = json.loads(en_cours.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marque = None
        en_cours.unlink(missing_ok=True)

        if marque:
            morts_consecutives = 0
            logger.error(
                "Processus de travail mort (code %s) sur PRC-%s pages %s-%s : "
                "tentative comptee", code, marque["doc_id"], marque["debut"], marque["fin"],
            )
            extracteur.noter_echec(
                Path(marque["chemin"]), marque["debut"], marque["fin"],
                f"processus de travail mort (code {code})", sha=marque.get("sha256"),
            )
        else:
            morts_consecutives += 1
            logger.error("Processus de travail mort (code %s) hors conversion", code)
            if morts_consecutives >= morts_consecutives_max:
                logger.error(
                    "%s morts d'affilee sans lot en cours : panne generale, arret",
                    morts_consecutives,
                )
                return 2


# ==================== POINT D'ENTREE ====================


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX_PAR_DEFAUT)
    parser.add_argument("--doc-id", nargs="+", help="Limite a ces documents")
    parser.add_argument("--limite", type=int, help="Nombre maximal de documents (pilote)")
    parser.add_argument("--docs-par-processus", type=int, default=25)
    parser.add_argument("--rss-max-mo", type=int, default=5000,
                        help="Recycle le processus de travail au-dela (Mo)")
    parser.add_argument("--etat", action="store_true", help="Avancement seul, sans convertir")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    configurer_journal(DOSSIER_SUIVI / "extraction.log")

    documents = selectionner(charger_corpus(args.index, DOSSIER_SUIVI), args.doc_id, args.limite)
    extracteur = DoclingPdfExtractor()
    logger.info(
        "%s documents, %s pages ; OCR %s, cache %s",
        len(documents), sum(d.pages for d in documents), extracteur.moteur_ocr,
        extracteur.dossier_cache,
    )

    if args.etat:
        progression = ecrire_progression(documents, extracteur, DOSSIER_SUIVI, time.time(), 0, 0, 0)
        print(json.dumps(progression, ensure_ascii=False, indent=1))
        return 0

    stop = DOSSIER_SUIVI / "STOP"
    stop.unlink(missing_ok=True)

    def _arreter(signum, _frame):
        logger.info("Signal %s : arret apres le lot en cours", signum)
        stop.touch()

    signal.signal(signal.SIGTERM, _arreter)
    signal.signal(signal.SIGINT, _arreter)

    return superviser(
        documents, extracteur, DOSSIER_SUIVI,
        docs_par_processus=args.docs_par_processus, rss_max_mo=args.rss_max_mo,
    )


if __name__ == "__main__":
    sys.exit(main())
