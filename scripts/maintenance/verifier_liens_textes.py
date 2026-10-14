#!/usr/bin/env python3
"""
Controle en base des liens entre textes successifs. Lecture seule.

POURQUOI. Les liens de app/services/textes_successifs.LIENS designent des
morceaux par leur numero en base, leur section, et parfois un repere de
contenu. Un nouveau decoupage peut renumeroter (« SECTION_2 » est un compteur),
deplacer un article, ou fondre deux articles : le lien se taira alors, ou se
posera sur le mauvais morceau, sans qu'aucun test ne le voie. Ce script
rejoue les lignes de statut sur les vrais morceaux et signale :

- un texte, ou un texte recent, absent de la base (un texte archive n'est
  qu'une note : ses liens servent quand il est remis en ligne) ;
- un article de la liste qu'aucun morceau ne porte (article ajoute par le
  texte recent ? numero mal lu ? article range sous un autre numero ?) ;
- un morceau de `morceaux` absent, ou qui ne contient plus son repere ;
- une exception de `sauf` qui ne s'applique a aucun morceau.

A lancer apres chaque ingestion, re-decoupage ou reclassement d'un texte lie.

Usage :
    python scripts/maintenance/verifier_liens_textes.py
    python scripts/maintenance/verifier_liens_textes.py --detail
    python scripts/maintenance/verifier_liens_textes.py --candidats liens.py

`--candidats` charge un fichier Python qui definit LIENS (des
textes_successifs.Lien) a la place de ceux du module : pour verifier un
lien avant de le saisir.

Codes de sortie :
    0  aucun defaut
    1  au moins un defaut
"""

import argparse
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text

from app.core.database import SyncSessionLocal
from app.services import textes_successifs
from app.services.textes_successifs import Lien, _numero_de_base, _vise, lignes_de_statut


def _morceaux_du_texte(session, reference: str) -> List[dict]:
    lignes = session.execute(
        text(
            """
            SELECT a.number, a.section, a.content, a."order", l.status
            FROM articles a JOIN laws l ON l.id = a.law_id
            WHERE l.reference = :reference
            ORDER BY a."order"
            """
        ),
        {"reference": reference},
    ).mappings()
    return [dict(ligne) for ligne in lignes]


def _statut_du_texte(session, reference: Optional[str]) -> Optional[str]:
    if not reference:
        return None
    return session.execute(
        text("SELECT status FROM laws WHERE reference = :reference"), {"reference": reference}
    ).scalar()


def controler(liens: Sequence[Lien], detail: bool = False) -> int:
    defauts = 0
    with SyncSessionLocal() as session:
        cache: Dict[str, List[dict]] = {}

        def morceaux(reference: str) -> List[dict]:
            if reference not in cache:
                cache[reference] = _morceaux_du_texte(session, reference)
            return cache[reference]

        for lien in liens:
            entete = f"{lien.texte} <- {lien.par_reference or '(absent)'} [{lien.nature}]"
            problemes: List[str] = []
            notes: List[str] = []

            statut = _statut_du_texte(session, lien.texte)
            if statut is None:
                problemes.append(f"texte {lien.texte} absent de la base")
            elif statut != "published":
                # Saisi d'avance, pour le jour ou le texte sera remis en ligne.
                notes.append(f"texte {lien.texte} au statut « {statut} » : aucun extrait n'en sort aujourd'hui")
            if lien.par_reference and _statut_du_texte(session, lien.par_reference) is None:
                problemes.append(f"texte recent {lien.par_reference} absent de la base")

            marques = defaultdict(list)
            for m in morceaux(lien.texte):
                for ligne in lignes_de_statut(lien.texte, m["number"], m["content"], m["section"], [lien]):
                    marques[ligne.split(" par ")[0]].append(m["number"])

            # Chaque article designe doit etre porte par au moins un morceau
            # de la bonne section.
            dans_la_section = [
                m for m in morceaux(lien.texte)
                if textes_successifs._dans_la_section(lien, m["section"])
            ]
            bases = {_numero_de_base(m["number"]) for m in dans_la_section}
            for designation in lien.articles:
                if not any(base and _vise(designation, base) for base in bases):
                    rangees = {article for _, article, _ in lien.morceaux}
                    if designation not in rangees:
                        problemes.append(f"article « {designation} » porte par aucun morceau")

            for numero, article, repere in lien.morceaux:
                trouves = [m for m in morceaux(lien.texte) if str(m["number"]).strip() == numero]
                if not trouves:
                    problemes.append(f"morceau « {numero} » (article {article}) absent")
                elif not any(textes_successifs._contient(m["content"], repere) for m in trouves):
                    problemes.append(f"morceau « {numero} » sans son repere « {repere} »")

            for numero, repere in lien.sauf:
                if not any(
                    _numero_de_base(m["number"]) == _numero_de_base(numero)
                    and textes_successifs._contient(m["content"], repere)
                    for m in dans_la_section
                ):
                    problemes.append(f"exception « {numero} » / « {repere} » sans objet")

            inverses = []
            if lien.par_reference:
                for m in morceaux(lien.par_reference):
                    if lignes_de_statut(lien.par_reference, m["number"], m["content"], m["section"], [lien]):
                        inverses.append(m["number"])

            total = sum(len(v) for v in marques.values())
            print(f"{'DEFAUT' if problemes else 'ok    '} {entete} : {total} morceau(x) marque(s), "
                  f"{len(inverses)} morceau(x) du texte recent annote(s)")
            for probleme in problemes:
                print(f"         - {probleme}")
            for note in notes:
                print(f"         (note) {note}")
            if detail:
                for tete, numeros in marques.items():
                    print(f"         {tete} : {', '.join(map(str, numeros))}")
                if inverses:
                    print(f"         texte recent : {', '.join(map(str, inverses))}")
            defauts += bool(problemes)
    return defauts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Controle en base des liens entre textes successifs.")
    parser.add_argument("--candidats", help="fichier Python definissant LIENS")
    parser.add_argument("--detail", action="store_true", help="liste les morceaux marques")
    args = parser.parse_args(argv)

    liens = runpy.run_path(args.candidats)["LIENS"] if args.candidats else textes_successifs.LIENS
    defauts = controler(liens, args.detail)
    print(f"\n{len(liens)} lien(s), {defauts} en defaut")
    return 1 if defauts else 0


if __name__ == "__main__":
    sys.exit(main())
