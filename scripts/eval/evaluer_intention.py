"""
Mesure le classement d'intention par Groq sur le jeu etiquete.

tests/fixtures/intent_eval.json porte 122 messages etiquetes (juridique,
smalltalk, meta, hors_sujet), dont des questions de suivi (champ
`precedente`). Le script les passe par `classify_intent`, en mode groq, et
compare.

UNE EXIGENCE, ET SA RAISON. Le rappel « juridique » doit atteindre 0,98 : une
question de droit classee en conversation recoit une reponse SANS SOURCE, et
l'utilisateur n'a aucun moyen de s'en apercevoir. L'erreur inverse (un
bonjour envoye a la recherche) ne coute qu'un paragraphe inutile.

COUT. Chaque message est une requete Groq (~330 jetons) sur le quota du chat :
--limite borne la depense (60 messages : ~20 000 jetons, 10 % du quota du
jour). Les messages sont espaces (--pause) : sans cela, le limiteur ferait
attendre plus de 0,5 s, et le classement rendrait « juridique » par defaut —
ce qui mesurerait le limiteur, pas le modele. Les replis (`defaut-groq-*`)
sont comptes a part.

Usage:
    python -m scripts.eval.evaluer_intention --limite 60
    python -m scripts.eval.evaluer_intention --limite 122 --pause 3
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from app.core.config import settings
from app.services.intent_classifier import INTENTS, cache_des_verdicts, classify_intent

JEU = RACINE / "tests" / "fixtures" / "intent_eval.json"
RAPPEL_JURIDIQUE_EXIGE = 0.98


class _Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


def echantillon(messages: List[dict], limite: Optional[int]) -> List[dict]:
    """
    Les `limite` premiers messages, en alternant les intentions : un jeu
    tronque garde ainsi des exemples de chaque categorie.
    """
    par_intention: Dict[str, List[dict]] = defaultdict(list)
    for message in messages:
        par_intention[message["intention"]].append(message)
    choisis: List[dict] = []
    while any(par_intention.values()) and (limite is None or len(choisis) < limite):
        for intention in INTENTS:
            if par_intention[intention] and (limite is None or len(choisis) < limite):
                choisis.append(par_intention[intention].pop(0))
    return choisis


async def evaluer(messages: Sequence[dict], pause: float) -> dict:
    cache_des_verdicts.vider()
    confusion: Counter = Counter()
    replis: Counter = Counter()
    erreurs = []
    for position, cas in enumerate(messages, start=1):
        historique = [_Message("user", cas["precedente"])] if cas.get("precedente") else None
        verdict = await classify_intent(cas["message"], history=historique)
        if verdict.rule.startswith("defaut"):
            replis[verdict.rule] += 1
        confusion[(cas["intention"], verdict.intent)] += 1
        if verdict.intent != cas["intention"]:
            erreurs.append((cas["message"], cas["intention"], verdict.intent, verdict.rule))
        print(f"[{position}/{len(messages)}] {cas['intention']:>10} -> {verdict.intent:<10} "
              f"({verdict.rule}) {cas['message'][:60]}", flush=True)
        if position < len(messages):
            await asyncio.sleep(pause)
    return {"confusion": confusion, "replis": replis, "erreurs": erreurs, "total": len(messages)}


def rapport(resultat: dict) -> float:
    confusion, total = resultat["confusion"], resultat["total"]
    justes = sum(n for (attendu, rendu), n in confusion.items() if attendu == rendu)
    juridiques = sum(n for (attendu, _), n in confusion.items() if attendu == "juridique")
    rappel = (confusion[("juridique", "juridique")] / juridiques) if juridiques else 1.0

    print("\nMatrice (lignes : attendu, colonnes : rendu)")
    print(f"{'':>11}" + "".join(f"{i:>11}" for i in INTENTS))
    for attendu in INTENTS:
        print(f"{attendu:>11}" + "".join(f"{confusion[(attendu, r)]:>11}" for r in INTENTS))
    print(f"\nJustes : {justes}/{total} ({justes / total:.1%})")
    print(f"Rappel juridique : {rappel:.3f} (exige : {RAPPEL_JURIDIQUE_EXIGE})")
    if resultat["replis"]:
        print(f"Replis par defaut (infrastructure, pas le modele) : {dict(resultat['replis'])}")
    for message, attendu, rendu, regle in resultat["erreurs"]:
        print(f"  ERREUR {attendu} -> {rendu} ({regle}) : {message}")
    return rappel


def main(argv: Optional[Sequence[str]] = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--limite", type=int, default=60, help="Messages evalues (defaut 60)")
    parseur.add_argument("--pause", type=float, default=3.0,
                         help="Secondes entre deux messages (defaut 3 : sous 30 req/min et 7 000 jetons/min)")
    args = parseur.parse_args(argv)

    if not settings.GROQ_API_KEY:
        print("GROQ_API_KEY absente du .env : rien a mesurer.")
        return 2
    settings.INTENT_CLASSIFIER = "groq"
    messages = echantillon(json.loads(JEU.read_text(encoding="utf-8"))["messages"], args.limite)
    print(f"{len(messages)} messages, modele {settings.GROQ_MODEL}")

    resultat = asyncio.run(evaluer(messages, args.pause))
    rappel = rapport(resultat)
    return 0 if rappel >= RAPPEL_JURIDIQUE_EXIGE else 1


if __name__ == "__main__":
    sys.exit(main())
