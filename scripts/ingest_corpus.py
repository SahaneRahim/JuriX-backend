"""
Passe 2 de l'ingestion : indexation du corpus en base, avec reprise.

Lit l'extraction faite par la passe 1 (scripts/extraire_corpus.py) et ne
convertit RIEN elle-meme : un document que la passe 1 n'a pas fini est laisse
pour une prochaine execution. Docling n'est donc jamais charge ici, ce qui
laisse la memoire a EmbeddingGemma.

Pour chaque document : ligne `laws`, copie du PDF dans le repertoire d'upload,
decoupage en articles, langue, domaine, embeddings EmbeddingGemma (locaux),
tsvector. C'est le pipeline de l'upload par l'API, appele hors HTTP.

REPRISE. Une loi en echec passe 'refused', la cause dans processing_error. Une
relance reprend les lignes 'refused', 'processing' et 'pending', et ignore les
'published' : rien n'est traite deux fois, rien n'est oublie. Tous les
documents de l'index sont pris, sans limite de taille.

Apres la derniere passe, reconstruire l'index vectoriel en masse :

    python scripts/regenerate_embeddings.py --all       # rattrape les vecteurs manquants
    python scripts/regenerate_embeddings.py --reindex

Usage:
    python scripts/ingest_corpus.py                  # tout ce que la passe 1 a extrait
    python scripts/ingest_corpus.py --doc-id 291 9866
    python scripts/ingest_corpus.py --limite 20      # pilote
    python scripts/ingest_corpus.py --dry-run        # combien, sans rien ecrire
"""

import argparse
import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Les reglages (.env) et les chemins relatifs (./data/...) se resolvent depuis
# le repertoire courant : on s'y place, ou que la commande ait ete lancee.
RACINE = Path(__file__).resolve().parents[1]
os.chdir(RACINE)
sys.path.insert(0, str(RACINE))

from sqlalchemy import text as sql_text

from app.core.database import SyncSessionLocal
from app.models.law import Law
from app.services.docling_extraction import DoclingPdfExtractor
from app.services.file_upload_service import get_upload_service
from app.tasks.process_law import process_law_sync
from scripts.extraire_corpus import (
    DOSSIER_SUIVI,
    INDEX_PAR_DEFAUT,
    Document,
    charger_corpus,
    configurer_journal,
    selectionner,
)

logger = logging.getLogger("ingest_corpus")

# Les valeurs de droite doivent appartenir a la liste close de
# LawResponse.type (app/schemas/law.py), ACCENTS COMPRIS. Une seule ligne au
# type invalide fait echouer la serialisation de TOUTE la reponse de
# GET /api/v1/laws/ : la liste entiere repond 500 a cause d'un document.
_TYPES = (
    ("ordonnance", "ordonnance"),
    ("arrete", "arrêté"),
    ("decision", "décision"),
    ("decret", "décret"),
    ("circulaire", "circulaire"),
    ("instruction", "instruction"),
    ("constitution", "loi"),
    ("loi", "loi"),
)

# Statuts que l'ingestion reprend. Une loi 'archived' ou 'draft' a ete mise
# ainsi par un administrateur : la republier a chaque relance defaisait son
# geste.
_A_REPRENDRE = ("pending", "processing", "refused")

# Le titre nomme l'acte, sa date, puis son objet. La date cherchee est celle de
# l'acte, pas celle d'un acte cite dans l'objet (« modifiant le decret du 12
# mars 2001 ») : on ne lit que ce qui precede le premier verbe d'objet.
_DEBUT_OBJET = re.compile(
    r"\b(portant|modifiant|compl[ée]tant|fixant|ratifiant|habilitant|autorisant"
    r"|abrogeant|accordant|nommant|relatif|relative|instituant|cr[ée]ant|approuvant"
    r"|rendant|d[ée]clarant|convoquant|mettant|prorogeant|r[ée]gissant"
    r"|r[ée]glementant|organisant|d[ée]finissant|instaurant|attribuant)\b",
    re.IGNORECASE,
)

# Mois par leurs quatre premieres lettres, accents retires : couvre les formes
# abregees (« SEP. », « janv. », « fév ») et les fautes courantes (« juilet »).
_MOIS = {
    "janv": 1, "jan": 1, "fevr": 2, "fev": 2, "feb": 2, "mars": 3, "mar": 3,
    "avri": 4, "avr": 4, "mai": 5, "juin": 6, "juil": 7, "aout": 8, "aou": 8,
    "sept": 9, "sep": 9, "octo": 10, "oct": 10, "nove": 11, "nov": 11,
    "dece": 12, "dec": 12,
}


def _replier(texte: str) -> str:
    """Minuscules, accents retires : « Arrête » et « arrete » se comparent."""
    import unicodedata

    decompose = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _mois(nom: str) -> Optional[int]:
    nom = _replier(nom).rstrip(".")
    return _MOIS.get(nom[:4]) or _MOIS.get(nom[:3])


# ==================== METADONNEES DEPUIS LE TITRE ====================


def _infer_type(title: str) -> str:
    """
    Type d'acte, lu sur le PREMIER mot du titre, accents replies.

    La recherche du mot n'importe ou dans le titre se trompait sur 17
    documents du corpus : « Loi n° 2016/007 portant ratification de
    l'ordonnance... » trouvait « ordonnance » avant « loi », et 13 lois
    devenaient des ordonnances. Un acte qu'aucun type ne decrit — accord,
    memorandum, pleins pouvoirs — est « autre », et non « loi ».
    """
    replie = _replier(title).lstrip(" \t*#-_")
    for needle, value in _TYPES:
        if replie.startswith(needle):
            return value
    for needle, value in _TYPES:
        if re.search(rf"\b{needle}\b", replie):
            return value
    return "autre"


def _infer_date(title: str):
    """
    Date de signature depuis le titre : « du 8 mai 2013 », « du 1er juin
    2024 », « du 10.12.2014 », « DU 0 9 SEP. 2013 », « of 26 janvier 2026 »,
    « du 07septembre 1982 », « du 06 septembre 2017portant ».

    Lue d'abord AVANT l'objet de l'acte, ou « du » n'est pas exige
    (« N°2017/591 04 décembre 2017 ») : la premiere date venue etait parfois
    celle d'un acte modifie, jusqu'a 17 ans d'ecart. L'annee du numero
    (« N°2017/591 ») sert de garde — une date qui s'en ecarte de plus d'un an
    est ecartee — et de secours : annee mal tapee (« 1013 », « 20107 ») ou
    absente (« du 26 octobre »). A defaut de tout, le 1er janvier de l'annee.
    """
    from datetime import date

    def _date(annee: Optional[int], mois: Optional[int], jour: int):
        if not annee or not mois or not 1900 <= annee <= 2100:
            return None
        try:
            return date(annee, mois, jour)
        except ValueError:
            return None

    numero = re.search(
        r"n\s*[°ºo]?\s*(?:\d{1,2}[./-])?((?:19|20)\d{2})\s*/", title, re.IGNORECASE
    )
    annee_numero = int(numero.group(1)) if numero else None

    def _retenir(annee: Optional[int], mois: Optional[int], jour: int):
        trouvee = _date(annee, mois, jour)
        if trouvee:
            # Une annee valide mais loin du numero : c'est la date d'un AUTRE
            # acte (« modifiant le decret du 14 aout 2014 »), pas une faute.
            if not annee_numero or abs(trouvee.year - annee_numero) <= 1:
                return trouvee
            return None
        # Annee absente ou impossible (« 1013 », « 20107 ») : celle du numero
        return _date(annee_numero, mois, jour)

    def _chercher(zone: str, avec_du: bool):
        prefixe = (
            r"(?<![A-Za-zÀ-ÿ])(?:du|of|le|en\s+date\s+du)[\s_]+" if avec_du else r"(?<!\d)"
        )
        for m in re.finditer(
            prefixe + r"(\d\s?\d|\d)(?:er|ᵉʳ)?[\s_]*([A-Za-zÀ-ÿ]{3,10})\.?"
            r"(?:[\s_,]+(\d{4,5})(?!\d))?",
            zone, re.IGNORECASE,
        ):
            mois = _mois(m.group(2))
            if mois:
                annee = int(m.group(3)) if m.group(3) else None
                trouvee = _retenir(annee, mois, int(m.group(1).replace(" ", "")))
                if trouvee:
                    return trouvee
        for m in re.finditer(
            prefixe + r"(\d{1,2})[./\- ](\d{1,2})[./\- ]((?:19|20)\d{2})(?!\d)",
            zone, re.IGNORECASE,
        ):
            jour, mois, annee = (int(g) for g in m.groups())
            trouvee = _retenir(annee, mois, jour)
            if trouvee:
                return trouvee
        return None

    objet = _DEBUT_OBJET.search(title)
    tete = title[:objet.start()] if objet else title
    trouvee = _chercher(tete, avec_du=False) or _chercher(title, avec_du=True)
    if trouvee:
        return trouvee
    if annee_numero:
        return date(annee_numero, 1, 1)
    year = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", title)
    return date(int(year.group(1)), 1, 1) if year else None


# ==================== DOUBLONS ====================


def _numero_acte(titre: str) -> Optional[str]:
    trouve = re.search(r"n\s*[°ºo]\s*([\d][\d/ .-]*\d)", titre, re.IGNORECASE)
    return re.sub(r"\s+", "", trouve.group(1)) if trouve else None


def _canoniques(documents: List[Document]) -> Dict[str, Optional[Document]]:
    """
    Pour chaque document, le document qui porte son contenu en base.

    12 PDF du corpus sont publies sous deux doc_id. Indexes deux fois, ils
    doublaient chaque article dans la recherche et le chat — et dans deux cas
    (8467/8480, 9841/9842) un decret paraissait sous le titre d'un autre. Un
    seul est indexe : celui dont le titre commence par un type d'acte, sinon le
    plus petit doc_id.

    Returns:
        doc_id -> None si le document est le canonique, sinon le canonique
    """
    par_sha: Dict[str, List[Document]] = {}
    for d in documents:
        par_sha.setdefault(d.sha256, []).append(d)
    canoniques: Dict[str, Optional[Document]] = {}
    for groupe in par_sha.values():
        groupe = sorted(groupe, key=lambda d: (
            _infer_type(d.titre) == "autre" or not re.match(r"\s*[A-Za-zÀ-ÿ]+\s+N", d.titre),
            int(d.doc_id) if d.doc_id.isdigit() else 0,
        ))
        canoniques[groupe[0].doc_id] = None
        for d in groupe[1:]:
            canoniques[d.doc_id] = groupe[0]
    return canoniques


def _noter_doublon(document: Document, canonique: Document) -> None:
    """La ligne du doublon existe, 'refused', et dit pourquoi : visible, jamais indexee."""
    reference = f"PRC-{document.doc_id}"
    raison = f"Meme PDF que PRC-{canonique.doc_id} : non indexe en double."
    if _numero_acte(document.titre) != _numero_acte(canonique.titre):
        raison += " Numeros d'acte differents : l'un des deux titres est faux, a verifier."
    with SyncSessionLocal() as session:
        loi = session.query(Law).filter(Law.reference == reference).first()
        if loi is None:
            session.add(Law(
                reference=reference, title=document.titre[:500],
                type=_infer_type(document.titre), content="", language="fr",
                status="refused", processing_error=raison,
                original_filename=Path(document.chemin).name[:500],
                publication_date=_infer_date(document.titre),
            ))
            session.commit()


# ==================== INDEXATION ====================


def _copier(source: str, destination: Path) -> None:
    """
    Copie atomique du PDF dans le stockage, verifiee a la reprise.

    copyfile et non copy2 : copy2 conservait la date du fichier source, et le
    nettoyage du stockage traitait les PDF ingeres comme vieux. Copie vers un
    fichier temporaire puis renommage : une copie interrompue (Ctrl-C sur un
    PDF de 124 Mo) laissait un fichier tronque, reutilise ensuite a chaque
    relance — et la loi refusee pour « PDF illisible », indefiniment.
    """
    taille = Path(source).stat().st_size
    if destination.is_file() and destination.stat().st_size == taille:
        return
    temporaire = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporaire)
    os.replace(temporaire, destination)


def _preparer_loi(document: Document, stockage: Path, force: bool) -> Optional[Dict]:
    """
    Ligne `laws` du document, creee ou reprise, et son PDF dans le stockage.

    Returns:
        {"law_id", "file_id"}, ou None si la loi n'est pas a (re)traiter.
    """
    reference = f"PRC-{document.doc_id}"
    with SyncSessionLocal() as session:
        loi = session.query(Law).filter(Law.reference == reference).first()
        if loi and loi.status not in _A_REPRENDRE and not force:
            return None

        if loi is None:
            loi = Law(
                reference=reference,
                title=document.titre[:500],
                # Rempli par le pipeline ; la colonne est NOT NULL.
                content="",
                language="fr",
                status="pending",
                file_id=uuid.uuid4().hex,          # respecte ^[A-Za-z0-9_-]{8,64}$
                original_filename=Path(document.chemin).name[:500],
            )
            session.add(loi)
        # Recalcules a chaque reprise : les regles s'ameliorent, la ligne suit.
        loi.type = _infer_type(document.titre)
        loi.publication_date = _infer_date(document.titre)
        loi.status = "processing"
        session.commit()
        law_id, file_id = loi.id, loi.file_id

    _copier(document.chemin, stockage / f"{file_id}.pdf")
    return {"law_id": law_id, "file_id": file_id}


def _panne_de_base(exc: BaseException) -> bool:
    """La base, et non le document : PostgreSQL redemarre, la connexion coupe."""
    from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

    while exc is not None:
        if isinstance(exc, (OperationalError, InterfaceError)):
            return True
        if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class PanneDeBase(RuntimeError):
    """La base reste injoignable : inutile de compter chaque document en echec."""


def indexer(
    documents: List[Document],
    extracteur: DoclingPdfExtractor,
    stockage: Path,
    force: bool = False,
    traiter=process_law_sync,
    attentes_s: Sequence[float] = (5, 15, 30, 60, 120, 300),
    corpus: Optional[List[Document]] = None,
) -> Dict[str, int]:
    """
    Indexe chaque document dont l'extraction est terminee.

    Une panne de BASE n'est pas l'echec d'un document : le meme document est
    rejoue apres une attente croissante, et la passe s'arrete (PanneDeBase)
    si la base ne revient pas. Sinon une coupure de 30 secondes comptait tous
    les documents restants en echec, en quelques secondes.

    Returns:
        compteurs : publies, deja_publies, en_attente, doublons, echecs
    """
    bilan = {"publies": 0, "deja_publies": 0, "en_attente": 0, "doublons": 0, "echecs": 0}
    debut = time.time()
    # Les doublons se reconnaissent sur le CORPUS ENTIER, pas sur la selection :
    # traite par lots (--doc-id), un document dont le jumeau etait dans un
    # autre lot passait pour unique, et partait en traitement sans fichier.
    canoniques = _canoniques(corpus or documents)

    for position, document in enumerate(documents, start=1):
        if not extracteur.etat(Path(document.chemin), document.sha256, document.pages).termine:
            bilan["en_attente"] += 1
            continue

        for attente in [*attentes_s, None]:
            try:
                canonique = canoniques.get(document.doc_id)
                if canonique is not None:
                    _noter_doublon(document, canonique)
                    bilan["doublons"] += 1
                    break
                loi = _preparer_loi(document, stockage, force)
                if loi is None:
                    bilan["deja_publies"] += 1
                    break
                resultat = traiter(loi["law_id"], loi["file_id"], extraction_cache_seulement=True)
                bilan["publies"] += 1
                logger.info(
                    "[%s/%s] PRC-%s publie : %s articles, %s vecteurs",
                    position, len(documents), document.doc_id,
                    resultat.get("articles_count", 0), resultat.get("embeddings_generated", 0),
                )
                break
            except Exception as exc:
                if _panne_de_base(exc):
                    if attente is None:
                        raise PanneDeBase(f"Base injoignable : {exc}") from exc
                    logger.warning("Base injoignable (%s) : nouvel essai dans %s s", exc, attente)
                    time.sleep(attente)
                    continue
                # process_law_sync a deja passe la loi en 'refused', la cause
                # dans processing_error : la prochaine execution la reprendra.
                bilan["echecs"] += 1
                logger.error(
                    "[%s/%s] PRC-%s ECHEC : %s", position, len(documents), document.doc_id, exc
                )
                break

    logger.info(
        "Indexation en %.0f min : %s publies, %s deja publies, %s en attente "
        "d'extraction, %s doublons, %s echecs", (time.time() - debut) / 60, bilan["publies"],
        bilan["deja_publies"], bilan["en_attente"], bilan["doublons"], bilan["echecs"],
    )
    return bilan


def _bilan_base() -> None:
    with SyncSessionLocal() as session:
        row = session.execute(sql_text(
            "SELECT (SELECT count(*) FROM laws) AS lois,"
            " (SELECT count(*) FROM laws WHERE status = 'published') AS publiees,"
            " (SELECT count(*) FROM laws WHERE status = 'refused') AS refusees,"
            " (SELECT count(*) FROM articles) AS articles,"
            " (SELECT count(*) FROM articles WHERE embedding IS NOT NULL) AS vectorises"
        )).one()
    logger.info(
        "Base : %s lois (%s publiees, %s refusees), %s articles, %s vectorises",
        row.lois, row.publiees, row.refusees, row.articles, row.vectorises,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX_PAR_DEFAUT)
    parser.add_argument("--doc-id", nargs="+", help="Limite a ces documents")
    parser.add_argument("--limite", type=int, help="Nombre maximal de documents (pilote)")
    parser.add_argument("--force", action="store_true", help="Retraite aussi les lois publiees")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    configurer_journal(DOSSIER_SUIVI / "indexation.log")

    corpus = charger_corpus(args.index, DOSSIER_SUIVI)
    documents = selectionner(corpus, args.doc_id, args.limite)
    extracteur = DoclingPdfExtractor()
    prets = [
        d for d in documents
        if extracteur.etat(Path(d.chemin), d.sha256, d.pages).termine
    ]
    logger.info("%s documents, dont %s extraits", len(documents), len(prets))

    if args.dry_run:
        logger.info("--dry-run : aucune ecriture")
        return 0

    stockage = get_upload_service().storage_path
    stockage.mkdir(parents=True, exist_ok=True)
    try:
        bilan = indexer(documents, extracteur, stockage, force=args.force, corpus=corpus)
    except PanneDeBase as exc:
        logger.error("%s — relancer la meme commande une fois la base revenue", exc)
        return 3
    _bilan_base()
    return 0 if bilan["echecs"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
