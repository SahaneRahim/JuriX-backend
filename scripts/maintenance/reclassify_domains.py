#!/usr/bin/env python3
"""
Reclasse les lois dans les 14 domaines canoniques, par lots, en deux temps.

1. `classer` interroge Groq par lots et ECRIT LES VERDICTS dans un journal,
   data/reclassement/verdicts.jsonl. Rien n'est ecrit en base. Une relance
   saute les lois deja classees : une coupure ne coute que le lot en cours.
   Quota epuise : arret propre (code 4), avec l'heure de reprise.
2. `appliquer` lit le journal et ecrit categorie, confiance et suggestions.
   `--dry-run` n'ecrit RIEN en base, mais produit le rapport, changements.csv
   et a_revoir.csv : a relire avant d'appliquer pour de bon.

POURQUOI DEUX TEMPS. Le palier gratuit de Groq (1 000 requetes et 200 000
jetons par jour et par modele) ne permet pas de tout classer d'une traite, ni
de recommencer pour corriger une erreur d'application. Le journal garde ce
qui a coute du quota ; l'application, elle, se rejoue a volonte.

POURQUOI PAS DE DEFAUT. L'ancienne version rangeait en Droit Administratif
(confiance 0,10) toute loi dont le classement echouait — 89 sur 122 pendant
un essai, a cause des 429. Ici, une loi sans verdict exploitable, ou sous le
seuil de confiance, n'est pas appliquee : elle part dans a_revoir.csv.

Codes de sortie : 0 ; 2 configuration (domaines absents de la table, journal
vide) ; 3 panne de Groq, relancer ; 4 quota epuise, relancer a l'heure dite.

Usage:
    # 1. Titres seuls, 40 lois par requete (~56 requetes pour le corpus)
    python scripts/maintenance/reclassify_domains.py classer --extrait 0 --lot 40
    # 2. Les incertains, avec l'article premier, 15 par requete
    python scripts/maintenance/reclassify_domains.py classer --incertains --extrait 500 --lot 15
    # 3. Simuler EXACTEMENT ce que fera l'etape 4, puis relire le rapport,
    #    changements.csv et a_revoir.csv
    python scripts/maintenance/reclassify_domains.py appliquer --dry-run --force
    # 4. Ecrire en base, categories existantes comprises
    python scripts/maintenance/reclassify_domains.py appliquer --force
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from sqlalchemy import text
from sqlalchemy.orm import load_only

from app.core.config import settings
from app.core.database import SyncSessionLocal
from app.models.law import Category, Law
from app.services.category_resolver import load_domain_map
from app.services.legal_domain_classifier import (
    CANONICAL_DOMAINS,
    VERSION_DES_CONSIGNES,
    ClassementIndisponible,
    DocumentAClasser,
    LegalDomainClassifier,
    extrait_pour_classement,
    get_legal_domain_classifier,
)

logger = logging.getLogger("reclassement")

DOSSIER = RACINE / "data" / "reclassement"
JOURNAL = DOSSIER / "verdicts.jsonl"

SORTIE_CONFIGURATION = 2
SORTIE_PANNE = 3
SORTIE_QUOTA = 4

# Une attente imposee plus courte est observee sur place ; au-dela, le script
# s'arrete et dit quand reprendre.
ATTENTE_SUR_PLACE_MAX_S = 600.0
ESSAIS_PAR_LOT = 3
LOIS_PAR_COMMIT = 200


# ==================== JOURNAL ====================


def charger_verdicts(chemin: Path = JOURNAL) -> Dict[int, dict]:
    """Le DERNIER verdict de chaque loi : une passe plus fine remplace la premiere."""
    verdicts: Dict[int, dict] = {}
    if not chemin.exists():
        return verdicts
    with chemin.open(encoding="utf-8") as journal:
        for numero, ligne in enumerate(journal, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                verdict = json.loads(ligne)
            except json.JSONDecodeError:
                # Une ligne tronquee par une coupure : les autres restent bonnes.
                logger.warning("Ligne %d du journal illisible, ignoree", numero)
                continue
            verdicts[int(verdict["law_id"])] = verdict
    return verdicts


def _incertain(verdict: dict, seuil: float) -> bool:
    return verdict.get("statut") != "ok" or float(verdict.get("confiance") or 0) < seuil


def _meme_loi(verdict: dict, loi: Law) -> bool:
    """
    Le verdict designe-t-il bien CETTE loi ? Le journal est indexe par
    l'identifiant, qui change si la base est reconstruite ; la reference, non.
    Une ligne d'avant ce controle (sans reference) est crue sur parole.
    """
    reference = verdict.get("reference")
    return reference is None or reference == loi.reference


# ==================== CLASSER ====================


def lois_a_classer(
    session,
    verdicts: Dict[int, dict],
    *,
    incertains: bool = False,
    seuil: float = 0.6,
    longueur_extrait: int = 0,
    law_ids: Optional[Sequence[int]] = None,
    limite: Optional[int] = None,
) -> List[Law]:
    """
    Les lois publiees qui attendent un verdict.

    Par defaut : celles sans verdict « ok » obtenu avec les consignes
    actuelles. Avec `incertains` : celles dont le dernier verdict est « a
    revoir » ou sous le seuil, a redemander avec un extrait — sauf si ce
    verdict a deja ete obtenu avec un extrait au moins aussi long : apres un
    arret sur quota, la relance ne repaie pas les lois deja retentees.

    Un verdict d'une AUTRE loi (meme identifiant, autre reference : base
    reconstruite, autre base) compte comme absent.
    """
    # Sans le contenu : 2 226 textes integraux en memoire pour lire des titres.
    # Il n'est charge, loi par loi, que pour l'extrait de repli (extrait_de).
    requete = (
        session.query(Law)
        .options(load_only(Law.id, Law.reference, Law.title, Law.type, Law.status))
        .filter(Law.status == "published")
        .order_by(Law.id)
    )
    if law_ids:
        requete = requete.filter(Law.id.in_(law_ids))
    lois = []
    for loi in requete:
        verdict = verdicts.get(loi.id)
        if verdict is not None and not _meme_loi(verdict, loi):
            verdict = None
        if incertains:
            garder = (
                verdict is not None
                and _incertain(verdict, seuil)
                and int(verdict.get("extrait") or 0) < max(1, longueur_extrait)
            )
        else:
            garder = (
                verdict is None
                or verdict.get("statut") != "ok"
                or verdict.get("consignes") != VERSION_DES_CONSIGNES
            )
        if garder:
            lois.append(loi)
            if limite and len(lois) >= limite:
                break
    return lois


def extrait_de(session, loi: Law, longueur: int) -> str:
    """Le premier chunk `article` ; a defaut, le texte prive de ses visas."""
    if longueur <= 0:
        return ""
    premier = session.execute(
        text(
            'SELECT content FROM articles WHERE law_id = :id AND kind = \'article\' '
            'ORDER BY "order" LIMIT 1'
        ),
        {"id": loi.id},
    ).scalar()
    return extrait_pour_classement(premier or loi.content, longueur)


def _terminer_la_derniere_ligne(journal: Path) -> None:
    """
    Un arret brutal peut laisser une ligne coupee, sans saut de ligne : la
    suivante s'y collerait, et les deux seraient perdues a la relecture.
    """
    if not journal.exists() or journal.stat().st_size == 0:
        return
    with journal.open("rb") as fichier:
        fichier.seek(-1, 2)
        if fichier.read(1) == b"\n":
            return
    with journal.open("a", encoding="utf-8") as fichier:
        fichier.write("\n")


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _heure_de_reprise(secondes: Optional[float]) -> str:
    if secondes is None:
        return "inconnue"
    return (datetime.now() + timedelta(seconds=secondes)).strftime("%d/%m %H:%M")


def classer(
    session,
    classifieur: LegalDomainClassifier,
    *,
    journal: Path = JOURNAL,
    taille_lot: int = 40,
    longueur_extrait: int = 0,
    incertains: bool = False,
    seuil: float = 0.6,
    law_ids: Optional[Sequence[int]] = None,
    limite: Optional[int] = None,
    dormir: Callable[[float], None] = time.sleep,
) -> int:
    """Classe par lots et ajoute les verdicts au journal. Rend le code de sortie."""
    lois = lois_a_classer(
        session, charger_verdicts(journal),
        incertains=incertains, seuil=seuil, longueur_extrait=longueur_extrait,
        law_ids=law_ids, limite=limite,
    )
    lots = [lois[i:i + taille_lot] for i in range(0, len(lois), taille_lot)]
    logger.info(
        "%d lois a classer, %d lots de %d au plus (modele %s, extrait %d)",
        len(lois), len(lots), taille_lot, classifieur.modele, longueur_extrait,
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    _terminer_la_derniere_ligne(journal)
    jetons_total = requetes_total = 0
    debut = time.time()

    for numero, lot in enumerate(lots, start=1):
        documents = [
            DocumentAClasser(
                titre=loi.title or "",
                extrait=extrait_de(session, loi, longueur_extrait),
                type_acte=loi.type or None,
            )
            for loi in lot
        ]
        for essai in range(1, ESSAIS_PAR_LOT + 1):
            try:
                resultat = classifieur.classer_lot(
                    documents, attente_max=settings.GROQ_ATTENTE_MAX_429_S
                )
                break
            except ClassementIndisponible as e:
                if e.quota:
                    logger.error(
                        "⛔ Quota Groq epuise apres %d lots (%s). Reprise possible vers %s : "
                        "relancer la meme commande, les lois deja classees seront sautees.",
                        numero - 1, e.raison, _heure_de_reprise(e.retry_after),
                    )
                    return SORTIE_QUOTA
                attente = e.retry_after
                if essai == ESSAIS_PAR_LOT or attente is None or attente > ATTENTE_SUR_PLACE_MAX_S:
                    logger.error("⛔ Groq indisponible (%s) : relancer plus tard", e.raison)
                    return SORTIE_PANNE
                logger.warning("Groq sature (%s) : nouvel essai dans %.0f s", e.raison, attente)
                dormir(attente)

        jetons_total += resultat.jetons
        requetes_total += resultat.requetes
        with journal.open("a", encoding="utf-8") as sortie:
            for loi, verdict in zip(lot, resultat.verdicts):
                ligne = {
                    "law_id": loi.id,
                    "reference": loi.reference,
                    "titre": (loi.title or "")[:200],
                    "statut": "ok" if verdict else "a_revoir",
                    "domaine": verdict.domain if verdict else None,
                    "secondaires": [nom for nom, _ in verdict.runners_up] if verdict else [],
                    "confiance": verdict.confidence if verdict else None,
                    "modele": resultat.modele,
                    "consignes": VERSION_DES_CONSIGNES,
                    "extrait": longueur_extrait,
                    "lot": numero,
                    "jetons_lot": resultat.jetons,
                    "horodatage": _horodatage(),
                }
                sortie.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        a_revoir = sum(1 for v in resultat.verdicts if v is None)
        logger.info(
            "Lot %d/%d : %d lois, %d a revoir, %d jetons, %d requete(s)",
            numero, len(lots), len(lot), a_revoir, resultat.jetons, resultat.requetes,
        )

    logger.info(
        "✅ %d lots en %.0f s : %d requetes, %d jetons", len(lots), time.time() - debut,
        requetes_total, jetons_total,
    )
    return 0


# ==================== APPLIQUER ====================


@dataclass
class Bilan:
    appliquees: int = 0
    changees: int = 0
    protegees: int = 0
    a_revoir: int = 0
    avant: Counter = field(default_factory=Counter)
    apres: Counter = field(default_factory=Counter)


def appliquer(
    session,
    verdicts: Dict[int, dict],
    *,
    dossier: Path = DOSSIER,
    seuil: float = 0.6,
    force: bool = False,
    dry_run: bool = False,
) -> Bilan:
    """
    Ecrit les verdicts « ok » au-dessus du seuil. Sans `force`, une loi qui a
    deja une autre categorie la garde (choix d'un administrateur, peut-etre).

    Ecrit changements.csv et a_revoir.csv dans tous les cas. changements.csv
    liste TOUTE loi dont le verdict differe de la categorie actuelle, avec ce
    qui lui arrive (`suite`) : appliquee, ou gardee faute de --force. Une
    simulation sans --force ne montrait pas les categories qu'un --force
    allait remplacer — precisement celles qu'il faut relire.
    """
    carte = load_domain_map(session)
    noms = {identifiant: nom for nom, identifiant in session.query(Category.name, Category.id)}
    bilan = Bilan()
    changements, a_revoir = [], []
    lois = (
        session.query(Law)
        .options(load_only(
            Law.id, Law.reference, Law.title, Law.category_id,
            Law.category_confidence, Law.suggested_categories, Law.processing_error,
        ))
        .filter(Law.id.in_(list(verdicts)))
        .order_by(Law.id)
        .all()
    )

    ecrites = 0
    for loi in lois:
        verdict = verdicts[loi.id]
        actuelle = noms.get(loi.category_id, "(aucune)")
        bilan.avant[actuelle] += 1
        cible = carte.get((verdict.get("domaine") or "").lower())
        if not _meme_loi(verdict, loi):
            # Le verdict d'une autre loi : base reconstruite depuis le classement.
            verdict = {**verdict, "statut": "autre_loi"}

        if _incertain(verdict, seuil) or cible is None:
            bilan.a_revoir += 1
            bilan.apres[actuelle] += 1
            a_revoir.append([
                loi.id, loi.reference, loi.title, actuelle, verdict.get("domaine") or "",
                verdict.get("confiance") if verdict.get("confiance") is not None else "",
                verdict.get("statut"),
            ])
            continue

        if loi.category_id not in (None, cible) and not force:
            bilan.protegees += 1
            bilan.apres[actuelle] += 1
            changements.append([
                loi.id, loi.reference, loi.title, actuelle, verdict["domaine"],
                verdict["confiance"], "gardee (--force pour appliquer)",
            ])
            continue

        bilan.appliquees += 1
        bilan.apres[verdict["domaine"]] += 1
        if loi.category_id != cible:
            bilan.changees += 1
            changements.append([
                loi.id, loi.reference, loi.title, actuelle, verdict["domaine"],
                verdict["confiance"], "appliquee",
            ])
        if not dry_run:
            secondaires = [carte[nom.lower()] for nom in verdict.get("secondaires") or []
                           if nom.lower() in carte]
            loi.category_id = cible
            loi.category_confidence = float(verdict["confiance"])
            loi.suggested_categories = [cible, *[s for s in secondaires if s != cible]]
            loi.processing_error = _sans_anomalie_de_classement(loi.processing_error)
            ecrites += 1
            # Compte les lois ECRITES : compter les positions sautait le commit
            # chaque fois que la 200e loi etait gardee ou a revoir.
            if ecrites % LOIS_PAR_COMMIT == 0:
                session.commit()

    if not dry_run:
        session.commit()

    dossier.mkdir(parents=True, exist_ok=True)
    _ecrire_csv(
        dossier / "changements.csv",
        ["law_id", "reference", "titre", "avant", "apres", "confiance", "suite"], changements,
    )
    _ecrire_csv(
        dossier / "a_revoir.csv",
        ["law_id", "reference", "titre", "categorie_actuelle", "proposition", "confiance", "statut"],
        a_revoir,
    )
    return bilan


# Prefixe de l'anomalie que le pipeline pose quand le classement est
# indisponible (app/tasks/process_law.py). Elle n'a plus lieu d'etre une fois
# la loi classee ; les autres anomalies (pages illisibles, vecteurs) restent.
_ANOMALIE_DE_CLASSEMENT = "Classement indisponible"


def _sans_anomalie_de_classement(erreur: Optional[str]) -> Optional[str]:
    if not erreur:
        return erreur
    restes = [
        morceau for morceau in erreur.split(" | ")
        if not morceau.startswith(_ANOMALIE_DE_CLASSEMENT)
    ]
    return " | ".join(restes) or None


def _ecrire_csv(chemin: Path, entete: List[str], lignes: List[list]) -> None:
    with chemin.open("w", encoding="utf-8", newline="") as sortie:
        ecrivain = csv.writer(sortie)
        ecrivain.writerow(entete)
        ecrivain.writerows(lignes)


def _rapport(bilan: Bilan, dossier: Path, dry_run: bool) -> None:
    logger.info("%-55s %7s %7s", "Domaine", "avant", "apres")
    for nom in [*CANONICAL_DOMAINS, "(aucune)"]:
        if bilan.avant[nom] or bilan.apres[nom]:
            logger.info("%-55s %7d %7d", nom[:55], bilan.avant[nom], bilan.apres[nom])
    logger.info(
        "%s : %d verdicts appliques dont %d changements de categorie, %d categories "
        "existantes gardees (--force pour les remplacer ; elles figurent dans "
        "changements.csv), %d a revoir",
        "SIMULATION, rien n'est ecrit" if dry_run else "Ecrit en base",
        bilan.appliquees, bilan.changees, bilan.protegees, bilan.a_revoir,
    )
    logger.info("Relire : %s et %s", dossier / "changements.csv", dossier / "a_revoir.csv")
    if bilan.changees and not dry_run:
        logger.info(
            "Les categories changees sont figees dans embed_text : redecouper et "
            "revectoriser les %d lois de changements.csv.", bilan.changees,
        )


# ==================== LIGNE DE COMMANDE ====================


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parseur.add_argument("--journal", type=Path, default=JOURNAL)
    parseur.add_argument("--seuil", type=float, default=0.6,
                         help="Confiance minimale d'un verdict applique (defaut 0.6)")
    actions = parseur.add_subparsers(dest="action", required=True)

    p_classer = actions.add_parser("classer", help="Interroge Groq, ecrit le journal")
    p_classer.add_argument("--lot", type=int, default=40, help="Lois par requete (defaut 40)")
    p_classer.add_argument("--extrait", type=int, default=0,
                           help="Caracteres d'extrait par loi, 0 = titre seul (defaut 0)")
    p_classer.add_argument("--incertains", action="store_true",
                           help="Ne redemander que les verdicts a revoir ou sous le seuil")
    p_classer.add_argument("--law-id", type=int, action="append", dest="law_ids")
    p_classer.add_argument("--limit", type=int, default=None)

    p_appliquer = actions.add_parser("appliquer", help="Ecrit les verdicts en base")
    p_appliquer.add_argument("--dry-run", action="store_true",
                             help="N'ecrit rien en base ; produit le rapport et les CSV")
    p_appliquer.add_argument("--force", action="store_true",
                             help="Remplace aussi les categories deja posees")
    p_appliquer.add_argument("--dossier", type=Path, default=DOSSIER)
    return parseur


def main(argv: Optional[Sequence[str]] = None, classifieur=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = construire_parseur().parse_args(argv)

    with SyncSessionLocal() as session:
        manquants = [d for d in CANONICAL_DOMAINS if d.lower() not in load_domain_map(session)]
        if manquants:
            logger.error("❌ Domaines absents de la table categories : %s", ", ".join(manquants))
            logger.error("   Appliquer d'abord les migrations (alembic upgrade head)")
            return SORTIE_CONFIGURATION

        if args.action == "classer":
            return classer(
                session, classifieur or get_legal_domain_classifier(),
                journal=args.journal, taille_lot=args.lot, longueur_extrait=args.extrait,
                incertains=args.incertains, seuil=args.seuil,
                law_ids=args.law_ids, limite=args.limit,
            )

        verdicts = charger_verdicts(args.journal)
        if not verdicts:
            logger.error("❌ Journal vide (%s) : lancer d'abord `classer`", args.journal)
            return SORTIE_CONFIGURATION
        bilan = appliquer(
            session, verdicts, dossier=args.dossier, seuil=args.seuil,
            force=args.force, dry_run=args.dry_run,
        )
        _rapport(bilan, args.dossier, args.dry_run)
        return 0


if __name__ == "__main__":
    sys.exit(main())
