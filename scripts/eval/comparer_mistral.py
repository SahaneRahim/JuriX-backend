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

VARIANTES. `--modeles` et `--consignes` font varier le modele et le prompt
systeme du chat : « actuelles » (celui de la production) ou « renforcees »
(des regles de fond en tete, pour un petit modele qui s'ecarte des extraits).
Les comparaisons de regimes passent une fois par modele, avec leur prompt.

PROFILS (--personas). Meme protocole, un seul modele : chaque question est
posee sous chaque profil, avec le MEME contexte. --temoin rejoue le premier
profil une seconde fois : l'ecart entre deux tirages du meme prompt, a
temperature 0,7, est le plancher de bruit auquel comparer l'ecart entre
profils. --budget borne le nombre d'appels AVANT le premier.

REGLE DE DECISION DES PROFILS, ECRITE AVANT LA MESURE. Les nouveaux prompts
sont gardes si, sur la mesure APRES :
1. justesse des citations cumulee >= 0,94, et aucun profil sous 0,90 parmi
   ceux qui citent au moins 8 fois (reference : 0,962 sur 26 citations,
   ministral-14b-2512, prompts actuels, 2026-10-07) ;
2. forme_relue >= 0,90 pour chaque profil (les sources affichees sous la
   reponse dependent de CITATION_REGEX) ;
3. 0 erreur, 0 reponse incomplete, duree mediane <= 1,5 fois celle de citoyen ;
4. similarite mediane entre profils inferieure a celle du temoin, et plus
   basse que sur la mesure AVANT ;
5. l'avocat ne renvoie jamais vers un avocat ; aucun profil ne cite de
   jurisprudence ;
6. relecture a l'aveugle de 15 reponses (5 profils x 3 questions) : profil
   devine au moins 9 fois sur 15 (le hasard en donne 3).

SUGGESTIONS (--questions-libres). Avec --dry-run, chaque question passe par
la recherche du chat, sans aucun appel au modele : nombre d'extraits et trois
meilleurs textes, pour juger qu'une suggestion du front a une reponse dans le
corpus. Sans --dry-run, ces questions remplacent le jeu relu.

Usage:
    python -m scripts.eval.comparer_mistral --dry-run      # contextes seuls, aucun appel
    python -m scripts.eval.comparer_mistral
    python -m scripts.eval.comparer_mistral --modeles ministral-3b-2512 \
        --consignes actuelles renforcees --label 3b
    python -m scripts.eval.comparer_mistral --modeles ministral-14b-2512 --questions 8 \
        --personas citoyen avocat entrepreneur etudiant --temoin --comparaisons 0 \
        --label personas_avant
    python -m scripts.eval.comparer_mistral --dry-run --questions-libres \
        "Quel est le role du Conseil constitutionnel ?" "Comment obtenir un titre foncier ?"
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.personas import normaliser_profil
from app.schemas.comparison import CRITERES_PAR_DEFAUT, mention_absence
from app.services.article_reference import normalize_number
from app.services.comparison_service import ComparisonError, ComparisonService
from app.services.mistral_service import MistralService
from app.services.prompts import SYSTEM_PROMPTS, get_system_prompt
from app.services.rag_service import ANSWER_MAX_TOKENS, RAGService

JEU = RACINE / "tests" / "fixtures" / "eval" / "retrieval_eval_v1.json"
SORTIES = RACINE / "data" / "eval_runs"
MODELES = ("ministral-14b-2512", "ministral-8b-2512")

# Regles de fond placees EN TETE du prompt systeme, pour la variante
# « renforcees ». Un petit modele suit mal des consignes noyees dans la forme :
# celles-ci disent d'abord d'ou viennent les faits, et rien d'autre.
REGLES_DE_FOND = """RÈGLES DE FOND — elles passent avant toutes les autres.
1. Ta seule source est le bloc « Documents juridiques pertinents » du message. Ce que tu sais par ailleurs du droit, camerounais ou d'un autre pays, ne compte pas.
2. Chaque montant, délai, date, condition, sanction, autorité ou démarche que tu écris doit se lire dans un extrait. Sinon, ne l'écris pas : ni « en général », ni « habituellement », ni étape « à prévoir ».
3. Repère d'abord le ou les extraits qui répondent, puis appuie chaque affirmation sur l'un d'eux, avec son numéro d'article.
4. Si les extraits ne répondent qu'en partie, dis ce qu'ils disent, puis ce qu'ils ne disent pas. S'ils ne répondent pas, dis-le en une phrase.
5. Ne cite que des numéros d'article qui figurent dans les extraits.

"""


def systeme_pour(consignes: str, persona: str = "citoyen") -> str:
    base = get_system_prompt(persona, "fr")
    return REGLES_DE_FOND + base if consignes == "renforcees" else base


# Suffixe de la variante temoin : le premier profil, rejoue a l'identique.
TEMOIN = "#2"

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


# Les citations que le chat sait relire : ce sont elles qui deviennent les
# sources affichees sous la reponse. « art. 33 » ou « l'article 33 alinéa 2 du
# Code Minier » sont vus par numeros_cites, mais pas relus par le chat.
_CITATION_CHAT = re.compile(RAGService.CITATION_REGEX)
_RENVOI_AVOCAT = re.compile(
    r"consult(?:ez|er|e)\s+(?:un|une|votre)\s+(?:avocat|juriste|professionnel du droit)"
    r"|faire appel à un avocat|consult(?:ing)?\s+a\s+lawyer",
    re.IGNORECASE,
)
_JURISPRUDENCE = re.compile(r"jurisprudence|case[- ]law", re.IGNORECASE)
_DEFINITION = re.compile(
    r"\b(?:désigne|se définit|on entend par|est défini|signifie|means|is defined as|refers to)\b",
    re.IGNORECASE,
)


def forme_relue(texte: str) -> Dict[str, int]:
    """Numeros d'article cites, et parmi eux ceux que le chat sait relire."""
    vus = set(numeros_cites(texte))
    relus = {normalize_number(m.group(2)) for m in _CITATION_CHAT.finditer(texte or "")}
    return {"vus": len(vus), "relus": len(vus & relus)}


def traits(texte: str) -> Dict[str, Any]:
    """Ce qui distingue une reponse de profil : longueur, forme, renvois."""
    lignes = (texte or "").splitlines()
    return {
        "mots": len(re.findall(r"\w+", texte or "")),
        "liste_numerotee": any(re.match(r"\s*\d+[.)]\s", ligne) for ligne in lignes),
        "puces": any(re.match(r"\s*[-*•]\s", ligne) for ligne in lignes),
        "renvoi_avocat": bool(_RENVOI_AVOCAT.search(texte or "")),
        # Une MENTION : « les textes ne contiennent pas de jurisprudence » en est
        # une aussi. A relire avant de conclure.
        "jurisprudence": bool(_JURISPRUDENCE.search(texte or "")),
        "definition": bool(_DEFINITION.search(texte or "")),
    }


def _trigrammes(texte: str) -> set:
    mots = re.findall(r"\w+", (texte or "").lower())
    return {tuple(mots[i:i + 3]) for i in range(len(mots) - 2)}


def similarite(a: str, b: str) -> float:
    """Jaccard sur les trigrammes de mots : 1 pour deux textes identiques."""
    ta, tb = _trigrammes(a), _trigrammes(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def distinction(resultats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Par question, la similarite mediane entre profils differents, et celle du
    temoin (le meme profil rejoue). Des profils vraiment distincts se
    ressemblent MOINS entre eux que deux tirages du meme prompt.
    """
    entre, temoin = [], []
    for question in resultats["questions"]:
        textes = {r["variante"]: r["texte"] for r in question["reponses"]
                  if r.get("variante") and not r.get("erreur")}
        profils = [cle for cle in textes if not cle.endswith(TEMOIN)]
        paires = [similarite(textes[a], textes[b])
                  for i, a in enumerate(profils) for b in profils[i + 1:]]
        if paires:
            entre.append(statistics.median(paires))
        for cle in textes:
            if cle.endswith(TEMOIN) and cle[: -len(TEMOIN)] in textes:
                temoin.append(similarite(textes[cle], textes[cle[: -len(TEMOIN)]]))

    def mediane(valeurs):
        return round(statistics.median(valeurs), 3) if valeurs else None

    return {"entre_profils": mediane(entre), "temoin": mediane(temoin), "questions": len(entre)}


def variantes_de(args) -> List[tuple]:
    """(modele, consignes, profil) ; le temoin rejoue la premiere sous TEMOIN."""
    variantes = [(m, c, p) for m in args.modeles for c in args.consignes for p in args.personas]
    if args.temoin and variantes:
        m, c, p = variantes[0]
        variantes.append((m, c, p + TEMOIN))
    return variantes


def appels_prevus(n_questions: int, n_variantes: int, n_comparaisons: int, n_modeles: int) -> int:
    return n_questions * n_variantes + n_comparaisons * n_modeles


def questions_de(args) -> List[dict]:
    if args.questions_libres:
        return [{"question": q, "expected_law_id": None, "expected_article_number": None}
                for q in args.questions_libres]
    return questions_relues(args.questions)


async def evaluer(args) -> Dict[str, Any]:
    resultats: Dict[str, Any] = {"questions": [], "comparaisons": []}
    variantes = variantes_de(args)
    etiquettes = [chr(ord("X") + i) if i < 3 else f"R{i}" for i in range(len(variantes))]
    hasard = random.Random(args.graine)

    async with AsyncSessionLocal() as session:
        rag = RAGService(session)
        for position, item in enumerate(questions_de(args), start=1):
            chunks = await rag._retrieve_chunks(item["question"], args.langue)
            prompt = rag._build_prompt(
                question=item["question"], search_results=chunks, history=[], persona="citoyen"
            )
            contexte = {normalize_number(str(c.number or "")) for c in chunks if c.number}
            attendu = (f"{item['expected_law_id']}/{item['expected_article_number']}"
                       if item.get("expected_law_id") else None)
            entree = {
                "question": item["question"],
                "attendu": attendu,
                "extraits": [
                    {"loi": c.reference, "numero": c.number, "texte": (c.content or "")[:600]}
                    for c in chunks
                ],
                "reponses": [],
            }
            print(f"[{position}] {item['question'][:80]} ({len(chunks)} extraits)", flush=True)
            if args.dry_run and args.questions_libres:
                # Validation d'une suggestion : y a-t-il une reponse dans le corpus ?
                if not chunks:
                    print("    REJETEE : aucun extrait", flush=True)
                for c in chunks[:3]:
                    print(f"    - {c.reference} art. {c.number} : {(c.law_title or '')[:80]}", flush=True)
            if not args.dry_run:
                for modele, consignes, persona in variantes:
                    reponse = await repondre(
                        modele, prompt, systeme_pour(consignes, persona.removesuffix(TEMOIN))
                    )
                    reponse["consignes"] = consignes
                    reponse["persona"] = persona
                    reponse["variante"] = f"{modele}|{consignes}|{persona}"
                    reponse["citations"] = citations(reponse["texte"], contexte)
                    reponse["forme"] = forme_relue(reponse["texte"])
                    reponse["traits"] = traits(reponse["texte"])
                    entree["reponses"].append(reponse)
                    print(f"    {reponse['variante']}: {reponse['duree_s']} s, fin={reponse['fin']}, "
                          f"cites={len(reponse['citations']['cites'])}, "
                          f"absents={reponse['citations']['absents']}", flush=True)
                entree["aveugle"] = hasard.sample(etiquettes, len(etiquettes))
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
                for modele in args.modeles:
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
    """Par variante (modele, consignes, profil) ; les comparaisons, par modele."""
    par_modele: Dict[str, Dict[str, Any]] = {}
    variantes = []
    for q in resultats["questions"]:
        for r in q["reponses"]:
            cle = r.get("variante") or r["modele"]
            if cle not in variantes:
                variantes.append(cle)
    for variante in variantes:
        modele = variante.split("|")[0]
        reponses = [r for q in resultats["questions"] for r in q["reponses"]
                    if (r.get("variante") or r["modele"]) == variante]
        cites = sum(len(r["citations"]["cites"]) for r in reponses)
        presents = sum(len(r["citations"]["presents"]) for r in reponses)
        premiers = sorted(r["premier_morceau_s"] for r in reponses if r["premier_morceau_s"] is not None)
        durees = sorted(r["duree_s"] for r in reponses)
        comparaisons = [m for c in resultats["comparaisons"] for m in c["reponses"] if m["modele"] == modele]
        formes = [r["forme"] for r in reponses if r.get("forme")]
        vus = sum(f["vus"] for f in formes)
        lus = [r["traits"] for r in reponses if r.get("traits")]

        def part(trait, lus=lus):
            return round(sum(1 for t in lus if t[trait]) / len(lus), 2) if lus else None

        par_modele[variante] = {
            "reponses": len(reponses),
            "erreurs": sum(1 for r in reponses if r["erreur"]),
            "incompletes": sum(1 for r in reponses if r["fin"] not in (None, "STOP")),
            "justesse_citations": round(presents / cites, 3) if cites else None,
            "citations": cites,
            "forme_relue": round(sum(f["relus"] for f in formes) / vus, 3) if vus else None,
            "premier_morceau_median_s": premiers[len(premiers) // 2] if premiers else None,
            "duree_mediane_s": durees[len(durees) // 2] if durees else None,
            "jetons_sortie": sum(r["jetons_sortie"] or 0 for r in reponses),
            "mots_median": statistics.median([t["mots"] for t in lus]) if lus else None,
            "part_liste_numerotee": part("liste_numerotee"),
            "part_puces": part("puces"),
            "part_renvoi_avocat": part("renvoi_avocat"),
            "part_jurisprudence": part("jurisprudence"),
            "part_definition": part("definition"),
            "comparaisons_json": sum(1 for m in comparaisons if m.get("json")),
            "comparaisons_7_sur_7": sum(1 for m in comparaisons if m.get("lignes_remplies") == 7),
            # Une case remplie ne vaut que si ses citations tiennent : le 3b
            # remplissait les 28 cases, avec 37 articles absents des extraits.
            "comparaisons_citations_orphelines": sum(len(m.get("orphelines") or []) for m in comparaisons),
        }
    return par_modele


def fichier_a_l_aveugle(resultats: Dict[str, Any], chemin: Path) -> None:
    """Les reponses de chaque question sous X, Y..., sans le nom du modele."""
    lignes = ["# Relecture a l'aveugle : ajouts non sources", "",
              "Pour chaque question, compter dans chaque reponse les affirmations",
              "qu'aucun extrait ne soutient. La correspondance etiquette -> modele",
              "est dans le fichier JSON, champ `aveugle`, a ne lire qu'apres.", ""]
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


def _profil(valeur: str) -> str:
    try:
        return normaliser_profil(valeur)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--questions", type=int, default=20)
    parseur.add_argument("--comparaisons", type=int, default=4)
    parseur.add_argument("--graine", type=int, default=7)
    parseur.add_argument("--modeles", nargs="+", default=list(MODELES))
    parseur.add_argument("--consignes", nargs="+", default=["actuelles"],
                         choices=["actuelles", "renforcees"])
    parseur.add_argument("--personas", nargs="+", type=_profil, default=["citoyen"],
                         help="Profils compares, alias acceptes (etudiant, general)")
    parseur.add_argument("--temoin", action="store_true",
                         help="Rejoue le premier profil : plancher de bruit")
    parseur.add_argument("--budget", type=int, default=48,
                         help="Nombre maximal d'appels ; au-dela, rien n'est lance")
    parseur.add_argument("--questions-libres", nargs="+", default=None,
                         help="Remplace le jeu relu ; avec --dry-run, valide des suggestions")
    parseur.add_argument("--langue", default="fr", choices=["fr", "en"],
                         help="Langue de la recherche (questions libres)")
    parseur.add_argument("--label", default="14b_8b")
    parseur.add_argument("--dry-run", action="store_true", help="Recherche seule, aucun appel a Mistral")
    return parseur


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parseur().parse_args(argv)

    sans_prompt = [p for p in args.personas if p.removesuffix(TEMOIN) not in SYSTEM_PROMPTS["fr"]]
    if sans_prompt:
        # get_system_prompt se replierait en silence : on mesurerait un autre profil.
        print(f"Profil sans prompt : {sans_prompt}")
        return 2

    n_questions = len(questions_de(args))
    prevus = appels_prevus(n_questions, len(variantes_de(args)),
                           min(args.comparaisons, len(COMPARAISONS)), len(args.modeles))
    print(f"Appels prevus : {prevus} (budget {args.budget})")
    if not args.dry_run and prevus > args.budget:
        print("Budget depasse : rien n'est lance. Reduire --questions, --personas ou --modeles.")
        return 2
    if not args.dry_run and not settings.MISTRAL_API_KEY:
        print("MISTRAL_API_KEY absente du .env.")
        return 2
    # Chaque reponse doit venir du modele mesure : pas de bascule.
    settings.MISTRAL_MODEL_SECOURS = ""

    resultats = asyncio.run(evaluer(args))
    if args.dry_run:
        return 0
    resultats["synthese"] = synthese(resultats)
    resultats["distinction"] = distinction(resultats)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M")
    SORTIES.mkdir(parents=True, exist_ok=True)
    chemin = SORTIES / f"mistral_{args.label}_{horodatage}.json"
    chemin.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    fichier_a_l_aveugle(resultats, chemin.with_suffix(".aveugle.md"))
    print(json.dumps({"synthese": resultats["synthese"], "distinction": resultats["distinction"]},
                     ensure_ascii=False, indent=2))
    print(f"Ecrit : {chemin} et {chemin.with_suffix('.aveugle.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
