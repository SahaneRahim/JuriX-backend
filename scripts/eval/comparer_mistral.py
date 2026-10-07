"""
Compare ministral-14b et ministral-8b sur le chat et sur la comparaison.

PROTOCOLE, FIXE AVANT LA MESURE :
- 20 questions juridiques, prises dans le jeu d'evaluation RELU (difficulte
  « paraphrase »), et 4 comparaisons de regimes ;
- le MEME contexte pour les deux modeles : la recherche est faite une fois,
  et les extraits sont ecrits avec les reponses. Aucun appel a Groq : pas de
  classement d'intention, la question part droit au chat. Le modele de
  secours est coupe, pour que chaque reponse vienne du modele mesure ;
- mesures : premier morceau, duree, jetons, fin ; articles cites presents
  dans les extraits ; pour les comparaisons, JSON lisible et lignes remplies ;
- les reponses sont ecrites ANONYMISEES (X et Y, tires au sort par question)
  dans un fichier de relecture a l'aveugle des ajouts non sources ;
- budget : 48 appels au plus (24 taches, 2 modeles). Le bras facultatif
  (temperature 0,3 contre 0,7) n'est pas lance : il depasserait ce budget.

REGLE DE DECISION, ECRITE AVANT : on garde le 14b, sauf si le 8b egale la
justesse des citations (a 2 points pres) sans ajouter plus de contenu non
source. Le 8b reste le modele de secours dans tous les cas.

Usage:
    python -m scripts.eval.comparer_mistral --dry-run      # contextes seuls, aucun appel
    python -m scripts.eval.comparer_mistral
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.schemas.comparison import CRITERES_PAR_DEFAUT, mention_absence
from app.services.article_reference import normalize_number
from app.services.comparison_service import ComparisonError, ComparisonService
from app.services.mistral_service import MistralService
from app.services.prompts import get_system_prompt
from app.services.rag_service import ANSWER_MAX_TOKENS, RAGService

JEU = RACINE / "tests" / "fixtures" / "eval" / "retrieval_eval_v1.json"
SORTIES = RACINE / "data" / "eval_runs"
MODELES = ("ministral-14b-latest", "ministral-8b-latest")

COMPARAISONS = [
    ("permis de recherche minière", "permis d'exploitation minière"),
    ("contrat de travail à durée déterminée", "contrat de travail à durée indéterminée"),
    ("garde à vue", "détention provisoire"),
    ("titre foncier", "concession provisoire du domaine national"),
]

# « article 32 », « articles 40.1 et 41 », « art. 1er », « l'article premier ».
_CITATION = re.compile(
    r"\bart(?:icle)?s?\.?\s+((?:\d+(?:[.-]\d+)*(?:\s*(?:er|bis|ter))?|premier)"
    r"(?:\s*(?:,|et|à)\s*\d+(?:[.-]\d+)*)*)",
    re.IGNORECASE,
)
_NUMERO = re.compile(r"\d+(?:[.-]\d+)*(?:\s*(?:er|bis|ter))?|premier", re.IGNORECASE)


def numeros_cites(texte: str) -> List[str]:
    """Les numeros d'article cites dans une reponse, normalises, dans l'ordre."""
    vus: List[str] = []
    for groupe in _CITATION.findall(texte or ""):
        for numero in _NUMERO.findall(groupe):
            cle = normalize_number(numero)
            if cle and cle not in vus:
                vus.append(cle)
    return vus


def questions_relues(limite: int) -> List[dict]:
    if not JEU.exists():
        raise SystemExit(f"Jeu d'evaluation absent : {JEU} (lancer generate_eval_set, puis relire)")
    items = json.loads(JEU.read_text(encoding="utf-8"))["items"]
    retenues = [i for i in items if i.get("reviewed") and i.get("difficulty") == "paraphrase"]
    return retenues[:limite]


async def repondre(modele: str, prompt: str, systeme: str) -> Dict[str, Any]:
    """Une reponse en flux, chronometree."""
    service = MistralService(model_name=modele)
    fins: List[str] = []
    usages: List[dict] = []
    morceaux: List[str] = []
    debut = time.monotonic()
    premier = None
    try:
        async for morceau in service.generate_stream(
            prompt, system=systeme, temperature=0.7, max_tokens=ANSWER_MAX_TOKENS,
            fin=fins.append, usage=usages.append,
        ):
            if premier is None:
                premier = time.monotonic() - debut
            morceaux.append(morceau)
        erreur = None
    except Exception as exc:
        erreur = repr(exc)
    usage = usages[0] if usages else {}
    return {
        "modele": modele,
        "texte": "".join(morceaux),
        "fin": fins[0] if fins else None,
        "premier_morceau_s": round(premier, 2) if premier is not None else None,
        "duree_s": round(time.monotonic() - debut, 2),
        "jetons_sortie": usage.get("completion_tokens"),
        "modele_servi": usage.get("modele"),
        "erreur": erreur,
    }


def citations(texte: str, numeros_du_contexte: set) -> Dict[str, Any]:
    cites = numeros_cites(texte)
    presents = [n for n in cites if n in numeros_du_contexte]
    return {"cites": cites, "presents": presents, "absents": [n for n in cites if n not in presents]}


async def evaluer(args) -> Dict[str, Any]:
    resultats: Dict[str, Any] = {"questions": [], "comparaisons": []}
    systeme = get_system_prompt("citoyen", "fr")
    hasard = random.Random(args.graine)

    async with AsyncSessionLocal() as session:
        rag = RAGService(session)
        for position, item in enumerate(questions_relues(args.questions), start=1):
            chunks = await rag._retrieve_chunks(item["question"], "fr")
            prompt = rag._build_prompt(
                question=item["question"], search_results=chunks, history=[], persona="citoyen"
            )
            contexte = {normalize_number(str(c.number or "")) for c in chunks if c.number}
            entree = {
                "question": item["question"],
                "attendu": f"{item['expected_law_id']}/{item['expected_article_number']}",
                "extraits": [
                    {"loi": c.reference, "numero": c.number, "texte": (c.content or "")[:600]}
                    for c in chunks
                ],
                "reponses": [],
            }
            print(f"[{position}] {item['question'][:80]} ({len(chunks)} extraits)", flush=True)
            if not args.dry_run:
                for modele in MODELES:
                    reponse = await repondre(modele, prompt, systeme)
                    reponse["citations"] = citations(reponse["texte"], contexte)
                    entree["reponses"].append(reponse)
                    print(f"    {modele}: {reponse['duree_s']} s, fin={reponse['fin']}, "
                          f"cites={len(reponse['citations']['cites'])}, "
                          f"absents={reponse['citations']['absents']}", flush=True)
                entree["aveugle"] = hasard.sample(["X", "Y"], 2)
            resultats["questions"].append(entree)

        for sujet_a, sujet_b in COMPARAISONS[: args.comparaisons]:
            # Le service n'est construit que pour sa recherche et son analyse :
            # son modele est remplace a chaque mesure.
            service = ComparisonService(session, llm=object())
            chunks_a, chunks_b = await service._recuperer(sujet_a, sujet_b, None, 8)
            entree = {"sujets": [sujet_a, sujet_b], "extraits": [len(chunks_a), len(chunks_b)],
                      "reponses": []}
            print(f"[comparaison] {sujet_a} / {sujet_b} : {len(chunks_a)} + {len(chunks_b)} extraits",
                  flush=True)
            if not args.dry_run and chunks_a and chunks_b:
                for modele in MODELES:
                    service.llm = MistralService(model_name=modele)
                    debut = time.monotonic()
                    try:
                        brut = await service._generer(
                            sujet_a, sujet_b, chunks_a, chunks_b, CRITERES_PAR_DEFAUT, "fr"
                        )
                        lignes, orphelines = service._assembler(
                            brut, CRITERES_PAR_DEFAUT, service._indexer(chunks_a),
                            service._indexer(chunks_b), "fr",
                        )
                        absence = mention_absence("fr")
                        remplies = sum(1 for ligne in lignes
                                       if ligne.a.value != absence or ligne.b.value != absence)
                        mesure = {"json": True, "lignes_rendues": len(brut.get("lignes") or []),
                                  "lignes_remplies": remplies, "orphelines": orphelines}
                    except ComparisonError as exc:
                        mesure = {"json": False, "erreur": str(exc)}
                    mesure.update({"modele": modele, "duree_s": round(time.monotonic() - debut, 2)})
                    entree["reponses"].append(mesure)
                    print(f"    {modele}: {mesure}", flush=True)
            resultats["comparaisons"].append(entree)
    return resultats


def synthese(resultats: Dict[str, Any]) -> Dict[str, Any]:
    par_modele: Dict[str, Dict[str, Any]] = {}
    for modele in MODELES:
        reponses = [r for q in resultats["questions"] for r in q["reponses"] if r["modele"] == modele]
        cites = sum(len(r["citations"]["cites"]) for r in reponses)
        presents = sum(len(r["citations"]["presents"]) for r in reponses)
        premiers = sorted(r["premier_morceau_s"] for r in reponses if r["premier_morceau_s"] is not None)
        durees = sorted(r["duree_s"] for r in reponses)
        comparaisons = [m for c in resultats["comparaisons"] for m in c["reponses"] if m["modele"] == modele]
        par_modele[modele] = {
            "reponses": len(reponses),
            "erreurs": sum(1 for r in reponses if r["erreur"]),
            "incompletes": sum(1 for r in reponses if r["fin"] not in (None, "STOP")),
            "justesse_citations": round(presents / cites, 3) if cites else None,
            "citations": cites,
            "premier_morceau_median_s": premiers[len(premiers) // 2] if premiers else None,
            "duree_mediane_s": durees[len(durees) // 2] if durees else None,
            "jetons_sortie": sum(r["jetons_sortie"] or 0 for r in reponses),
            "comparaisons_json": sum(1 for m in comparaisons if m.get("json")),
            "comparaisons_7_sur_7": sum(1 for m in comparaisons if m.get("lignes_remplies") == 7),
        }
    return par_modele


def fichier_a_l_aveugle(resultats: Dict[str, Any], chemin: Path) -> None:
    """Les deux reponses de chaque question sous X et Y, sans le nom du modele."""
    lignes = ["# Relecture a l'aveugle : ajouts non sources", "",
              "Pour chaque question, compter dans X et dans Y les affirmations",
              "qu'aucun extrait ne soutient. La correspondance X/Y -> modele est",
              "dans le fichier JSON, champ `aveugle`, a ne lire qu'apres.", ""]
    for numero, question in enumerate(resultats["questions"], start=1):
        if not question["reponses"]:
            continue
        lignes += [f"## {numero}. {question['question']}", "", "### Extraits", ""]
        for extrait in question["extraits"]:
            lignes.append(f"- {extrait['loi']} art. {extrait['numero']} : {extrait['texte']}")
        for etiquette, reponse in zip(question["aveugle"], question["reponses"]):
            lignes += ["", f"### Reponse {etiquette}", "", reponse["texte"] or f"(erreur : {reponse['erreur']})"]
        lignes.append("")
    chemin.write_text("\n".join(lignes), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--questions", type=int, default=20)
    parseur.add_argument("--comparaisons", type=int, default=4)
    parseur.add_argument("--graine", type=int, default=7)
    parseur.add_argument("--dry-run", action="store_true", help="Recherche seule, aucun appel a Mistral")
    args = parseur.parse_args(argv)

    if not args.dry_run and not settings.MISTRAL_API_KEY:
        print("MISTRAL_API_KEY absente du .env.")
        return 2
    # Chaque reponse doit venir du modele mesure : pas de bascule.
    settings.MISTRAL_MODEL_SECOURS = ""

    resultats = asyncio.run(evaluer(args))
    if args.dry_run:
        return 0
    resultats["synthese"] = synthese(resultats)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M")
    SORTIES.mkdir(parents=True, exist_ok=True)
    chemin = SORTIES / f"mistral_14b_8b_{horodatage}.json"
    chemin.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    fichier_a_l_aveugle(resultats, chemin.with_suffix(".aveugle.md"))
    print(json.dumps(resultats["synthese"], ensure_ascii=False, indent=2))
    print(f"Ecrit : {chemin} et {chemin.with_suffix('.aveugle.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
