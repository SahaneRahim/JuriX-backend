"""
Mesure le CONTEXTE DU CHAT : les extraits que RAGService._retrieve_chunks
donne au modele, et ceux qu'il refuse (la barriere) — y compris quand le volet
semantique de la recherche se trompe ou tombe en panne.

Pourquoi un script de plus. run_eval.py et evaluer_recherche.py mesurent la
RECHERCHE (SearchService) ; le chat ajoute ses regles apres elle
(tableaux de programmes retires, article nomme, textes confirmes quand le
volet semantique est tres faible, barriere), et ce sont elles qu'il faut voir.
Pour chaque question, deux contextes, dans le meme processus et avec les
memes vecteurs : celui de la recherche hybride (AVANT les regles) et celui que
le modele recoit (APRES). Le rang qui compte est celui des extraits que le
modele LIT : le budget du contexte (prompts.build_context_string) ecarte les
derniers quand les premiers sont longs.

Jeux de questions :
- jeu : les 102 questions relues (tests/fixtures/eval/retrieval_eval_v1.json),
  et leurs traductions anglaises (--questions-en) ; article attendu connu ;
- suggerees : les questions que le front propose (src/lib/suggestions.ts du
  front, 16 en francais, 16 en anglais), avec les textes qu'elles visent ;
- hors_sujet : 20 questions sans rapport avec le droit, par langue (le
  classificateur d'intention les ecarte avant, en production : ici il n'est
  pas appele) ;
- nommees : des questions qui nomment un article et son texte, et
  nommees_revue : celles de la revue du 29/10/2026, avec le texte attendu en
  tete ou RIEN (texte hors de la table, revise, remplace, numero ambigu) ;
- reelles : des questions de tous les jours (revue du 29/10/2026, fautes et
  pidgin compris ; anglaises ecrites sur les memes sujets), voir
  scripts/eval/questions_courantes.py ;
- montants et budgetaires : des questions de montants, ou les tableaux de
  programmes sortaient en tete, et de vraies questions sur le budget, avec
  son vocabulaire, ou ils doivent rester ;
- montants_prevus : des questions de droit sur un montant « prévu » ou
  « alloué », un budget ou des credits prives (revue du tour 3, prises pour
  budgetaires) : aucun tableau de programmes ;
- relances : une relance (« Peux-tu détailler ? ») apres une question, par le
  chemin du chat (RAGService._recuperer) : reconnue (est_une_relance), elle
  est cherchee jointe au sujet de la conversation ; sinon seule, puis,
  refusee, jointe (_chercher_la_relance). La loi de la question precedente
  doit etre lue (« loi_lue ») ;
- chaines : des relances de suite, et des articles nommes sans leur texte
  (revue du tour 3) ou avec un renvoi au texte de la conversation (« l'article
  40 du même code », revue du tour 4), avec le texte qui doit etre lu ;
- changements_de_sujet : une question courte ouverte par « Et », « Mais »,
  « What about », qui apporte ses mots (« Et la nationalité ? »), apres une
  question sur le licenciement : le texte du NOUVEAU sujet doit etre lu
  (revue du tour 4) ;
- secondes_chances : une question refusee seule (« la dot ») : refusee, ou un
  contexte qui porte ses mots ; jamais l'ancien sujet seul (« sans_ses_mots »,
  zero attendu) ;
- fond_avec_historique : les questions de tous les jours et les suggestions,
  posees apres une question de fond : leur contexte doit etre celui qu'elles
  ont seules (aucune prise pour une relance) ;
- courtes : des mots-cles seuls (« divorce »).

Etats du volet semantique :
- normal : le systeme tel quel (recherche exacte du chat) ;
- index : le volet semantique par l'index HNSW (ef_search 400 et plus), comme
  /search : la difference avec « normal » est celle de la recherche exacte ;
- repli : la recherche exacte depasse toujours son budget (1 ms), le volet
  semantique est servi par l'index HNSW (ef_search 1000), sain : ce que coute
  la regle « l'index n'est pas cru seul » (RAGService._retenir_le_pertinent) ;
- leurre : pour TOUTE question, le volet semantique rend les tableaux de lois
  de finances les plus proches de la question de l'incident du 29/10/2026,
  avec leur vraie similarite a la question posee : la panne observee en
  production, par le seul chemin ou le chat lit encore l'index (le repli
  d'une recherche exacte trop lente, chemin « repli_hnsw ») ;
- leurre2 : la meme panne, mais l'index rend des articles quelconques, qui
  ne sont pas des tableaux (les plus proches d'une question hors sujet) :
  seule la barriere peut les ecarter ;
- panne : le volet semantique echoue (fournisseur d'embeddings injoignable).

Compteurs (synthese) : rang de l'article attendu (hit@1/3/8, MRR@8), article
et loi lus, refus de la barriere, modes (sain / faible), textes AJOUTES par
les regles du chat et ceux qui ne sont pas attendus (« ajouts_hors_sujet »),
extraits de lois de finances lus (« tableaux_lus »), parmi eux les tableaux
(« tableaux_de_finances_lus », lignes « | ... | ») et les tableaux de
programmes (« programmes_lus », zero attendu hors question budgetaire),
articles nommes ajoutes et justes ou faux, similarites, durees.

CALIBRATION de la barriere (etat normal, et leurres) : pour chaque question,
la meilleure similarite du contexte et ce qui la confirmerait si elle passait
sous le seuil (RAGService._textes_confirmes : article nomme, article qui
porte tous les mots-cles). La grille dit, seuil par seuil et par langue de la
question, combien de questions legitimes (tous les jeux sauf hors sujet),
hors sujet et leurrees seraient refusees ; la marge est l'ecart entre le
seuil retenu et la plus faible similarite d'une question legitime que rien
d'autre ne confirme.

Lecture seule : ni cache, ni journal de recherche. AUCUN appel au modele de
generation : rag.llm est retire et le re-ranking par le modele coupe. Le
fournisseur d'embeddings est appele une fois par question (vecteurs gardes en
memoire du processus).

Usage (depuis la racine du depot, pour que .env soit lu) :
    python -m scripts.eval.evaluer_contexte_chat \\
        --questions-en data/mesure_voyage/questions_en.json \\
        --etats normal index leurre leurre2 panne --sortie data/eval_runs/contexte.json
    python -m scripts.eval.evaluer_contexte_chat --releve   # table des noms, tableaux
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from sqlalchemy import text

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.database import engine as moteur
from app.schemas.search import ChunkResult
from app.services import search_service as module_recherche
from app.services.prompts import build_context_string
from app.services.rag_service import (
    TEXTES_NOMMABLES,
    VOLETS_SERVIS_PAR_L_INDEX,
    RAGService,
    _sans_accents,
    mots_propres,
)
from app.services.search_service import _TITRE_DE_LOI_DE_FINANCES, est_un_tableau_de_programmes
from scripts.eval.questions_courantes import (
    BUDGETAIRES,
    CHANGEMENTS_DE_SUJET,
    MONTANTS,
    MONTANTS_PREVUS,
    NOMMEES_REVUE,
    REELLES,
    RELANCES_AVEC_HISTORIQUE,
    RELANCES_EN_CHAINE,
    REPONSE_PRECEDENTE,
    SECONDES_CHANCES,
)

JEU = RACINE / "tests" / "fixtures" / "eval" / "retrieval_eval_v1.json"

QUESTION_DE_L_INCIDENT = "Quelles indemnités en cas de licenciement abusif ?"

# Les textes que visent les questions suggerees du front (src/lib/suggestions.ts,
# commentaire d'en-tete : Code du travail, Code penal, Code de procedure
# penale, cybersecurite, Actes uniformes OHADA, ordonnance de 2025,
# Constitution et Code electoral). Le Code penal compte sa loi modificative.
_CODE_PENAL = ("PRC-4722", "PRC-8028")
_CONSTITUTION_ET_ELECTIONS = (
    "PRC-6283", "PRC-6284", "PRC-10679", "LOI-2012-001", "PRC-7480", "PRC-10680",
)
SUGGEREES: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
    "fr": [
        (QUESTION_DE_L_INCIDENT, ("LOI-1992-007",)),
        ("Que risque l’auteur d’une escroquerie sur internet ?", _CODE_PENAL + ("LOI-2010-012",)),
        ("Quels sont mes droits en garde à vue ?", ("LOI-2005-007",)),
        ("Quand peut-on invoquer la légitime défense ?", _CODE_PENAL),
        ("Quelle différence entre tentative et complicité en droit pénal ?", _CODE_PENAL),
        ("Comment est élu le Président de la République ?", _CONSTITUTION_ET_ELECTIONS),
        ("Qu’est-ce qui distingue un contrat de travail à durée déterminée d’un contrat à "
         "durée indéterminée ?", ("LOI-1992-007",)),
        ("Qu’est-ce qu’un groupement d’intérêt économique ?", ("OHADA-AUSCGIE-2014",)),
        ("Quels sont les éléments constitutifs de l’abus de confiance ?", _CODE_PENAL),
        ("Quelles peines pour l’accès non autorisé à un système d’information ?",
         ("LOI-2010-012",) + _CODE_PENAL),
        ("Comment former opposition à une injonction de payer en droit OHADA ?",
         ("OHADA-AUPSRVE-2023",)),
        ("Quelle est la durée maximale de la garde à vue ?", ("LOI-2005-007",)),
        ("Quels avantages fiscaux l’ordonnance de 2025 accorde-t-elle à un nouvel "
         "investissement ?", ("PRC-10455", "PRC-10560")),
        ("Quelles pièces justificatives pour immatriculer une société commerciale ?",
         ("OHADA-AUDCG-2010", "OHADA-AUSCGIE-2014")),
        ("Quelles obligations pour l’employeur lors d’une embauche ?", ("LOI-1992-007",)),
        ("Comment obtenir une injonction de payer pour une facture impayée ?",
         ("OHADA-AUPSRVE-2023",)),
    ],
    "en": [
        ("What compensation is due for unfair dismissal?", ("LOI-1992-007",)),
        ("What does the Penal Code say about obtaining money by false pretences on the "
         "internet?", _CODE_PENAL + ("LOI-2010-012",)),
        ("What are my rights in police custody?", ("LOI-2005-007",)),
        ("When can self-defence be invoked?", _CODE_PENAL),
        ("What is the difference between attempt and complicity in criminal law?", _CODE_PENAL),
        ("How is the President of the Republic elected?", _CONSTITUTION_ET_ELECTIONS),
        ("What distinguishes a fixed-term employment contract from an open-ended one?",
         ("LOI-1992-007",)),
        ("What is an economic interest group under OHADA law?", ("OHADA-AUSCGIE-2014",)),
        ("What are the constituent elements of the offence of breach of trust under the "
         "Penal Code?", _CODE_PENAL),
        ("How does the law punish unauthorised access to a computer system?",
         ("LOI-2010-012",) + _CODE_PENAL),
        ("How is an OHADA injunction to pay obtained and opposed?", ("OHADA-AUPSRVE-2023",)),
        ("What is the maximum length of police custody?", ("LOI-2005-007",)),
        ("What tax incentives does the 2025 investment incentives ordinance grant to a new "
         "investment?", ("PRC-10455", "PRC-10560")),
        ("How do I register a company in the trade and personal property credit register?",
         ("OHADA-AUDCG-2010", "OHADA-AUSCGIE-2014")),
        ("What are the employer's obligations when hiring under the Labour Code?",
         ("LOI-1992-007",)),
        ("How can I make a customer pay an unpaid invoice?", ("OHADA-AUPSRVE-2023",)),
    ],
}

HORS_SUJET: Dict[str, List[str]] = {
    "fr": [
        "Quelle est la recette du ndolé ?",
        "Qui a gagné la Coupe d'Afrique des nations en 2017 ?",
        "Quel temps fera-t-il demain à Douala ?",
        "Comment réparer une fuite d'eau sous l'évier ?",
        "Quelle est la capitale de l'Australie ?",
        "Comment apprendre à programmer en Python ?",
        "Quel est le meilleur smartphone de l'année ?",
        "Combien de calories contient une banane ?",
        "Comment soigner un rhume rapidement ?",
        "Quelle est la distance entre la Terre et la Lune ?",
        "Comment faire pousser des tomates sur un balcon ?",
        "Quels films regarder ce week-end ?",
        "Comment changer un pneu de voiture ?",
        "Qui a écrit Les Misérables ?",
        "Quelle est la vitesse de la lumière ?",
        "Comment dresser un chien à rester assis ?",
        "Comment jouer aux échecs ?",
        "Pourquoi le ciel est-il bleu ?",
        "Quelle est la meilleure façon de cuire du riz ?",
        "Comment se maquiller pour un mariage ?",
    ],
    "en": [
        "What is the recipe for ndole?",
        "Who won the 2017 Africa Cup of Nations?",
        "What will the weather be like tomorrow in Douala?",
        "How do I fix a leaking pipe under the sink?",
        "What is the capital of Australia?",
        "How can I learn to program in Python?",
        "What is the best smartphone this year?",
        "How many calories are in a banana?",
        "How do I cure a cold quickly?",
        "What is the distance between the Earth and the Moon?",
        "How do I grow tomatoes on a balcony?",
        "Which movies should I watch this weekend?",
        "How do I change a car tyre?",
        "Who wrote Les Misérables?",
        "What is the speed of light?",
        "How do I train a dog to sit?",
        "How do you play chess?",
        "Why is the sky blue?",
        "What is the best way to cook rice?",
        "How should I do my make-up for a wedding?",
    ],
}

# (question, reference du texte, numero) : l'article nomme doit etre lu.
NOMMEES: Dict[str, List[Tuple[str, str, str]]] = {
    "fr": [
        ("Que dit l'article 39 du Code du travail ?", "LOI-1992-007", "39"),
        ("Que dit l'art. 37 du Code du travail ?", "LOI-1992-007", "37"),
        ("art. 37 du Code du travail", "LOI-1992-007", "37"),
        ("Que prévoit l'article 25 du Code du travail ?", "LOI-1992-007", "25"),
        ("Que dit l'article 74 du Code pénal ?", "PRC-4722", "74"),
        ("Que dit l'article 119 du Code de procédure pénale ?", "LOI-2005-007", "119"),
        ("Que dit l'article 39 du Code du travail sur le licenciement ?", "LOI-1992-007", "39"),
    ],
    "en": [
        ("What does Article 39 of the Labour Code say?", "LOI-1992-007", "39"),
        ("What does article 74 of the Penal Code provide?", "PRC-4722", "74"),
    ],
}

# Relances du jeu d'origine, jugees comme celles de la revue : seules, puis
# jointes a la question precedente (voir questions_courantes).
RELANCES: Dict[str, List[str]] = {
    "fr": [
        "Et pour un CDD ?",
        "Et en cas de faute lourde ?",
        "Quel est le délai pour saisir le tribunal ?",
        "Et si l'employeur refuse de payer ?",
        "Combien de temps dure le préavis ?",
        "Peut-on faire appel de cette décision ?",
        "Et pour une SARL ?",
        "Quelles sont les sanctions prévues ?",
    ],
    "en": [
        "What about a fixed-term contract?",
        "And in case of gross misconduct?",
        "What are the penalties?",
        "Can this decision be appealed?",
    ],
}
COURTES: Dict[str, List[str]] = {
    "fr": ["divorce", "TVA", "vol", "licenciement", "garde à vue", "héritage",
           "bail commercial", "OHADA", "permis de conduire", "pension alimentaire"],
    "en": ["divorce", "VAT", "theft", "dismissal", "police custody", "inheritance"],
}

# Les jeux legitimes, que la barriere ne doit JAMAIS refuser (montants_prevus
# commence par « montants »). Pas les secondes chances : leur refus est
# attendu quand rien ne porte leurs mots.
LEGITIMES = (
    "jeu", "suggerees", "nommees", "reelles", "montants", "budgetaires", "relances", "courtes",
    "chaines", "changements",
)

# La question dont les plus proches voisins font le second leurre : des
# articles qui ne sont pas des tableaux, sans rapport avec les questions.
QUESTION_DU_LEURRE2 = "Combien de calories contient une banane ?"

# Les tableaux de l'incident : les morceaux des lois de finances (« portant
# loi de finances »), par distance EXACTE a la question de l'incident (le
# `+ 0` interdit l'index). C'est la liste qu'un parcours rate rendrait.
_SQL_TABLEAUX = text("""
    SELECT a.id AS article_id, a.number, a.title AS article_title, a.section,
           a.page_number, a.content, a.law_id, l.reference, l.title AS law_title,
           l.type, l.language, l.status, l.category_id, l.publication_date,
           c.name AS category_name,
           (a.embedding <=> CAST(CAST(:vecteur AS text) AS halfvec)) + 0 AS distance
    FROM articles a
    JOIN laws l ON l.id = a.law_id
    LEFT JOIN categories c ON c.id = l.category_id
    WHERE l.title ILIKE '%portant loi de finances%'
      AND l.status = 'published' AND a.embedding IS NOT NULL
    ORDER BY distance
    LIMIT :n
""")

# Les plus proches voisins publies d'une question, sans les lois de finances :
# le second leurre.
_SQL_VOISINS_HORS_FINANCES = text("""
    SELECT a.id AS article_id, a.number, a.title AS article_title, a.section,
           a.page_number, a.content, a.law_id, l.reference, l.title AS law_title,
           l.type, l.language, l.status, l.category_id, l.publication_date,
           c.name AS category_name,
           (a.embedding <=> CAST(CAST(:vecteur AS text) AS halfvec)) + 0 AS distance
    FROM articles a
    JOIN laws l ON l.id = a.law_id
    LEFT JOIN categories c ON c.id = l.category_id
    WHERE l.title NOT ILIKE '%finances%'
      AND l.status = 'published' AND a.embedding IS NOT NULL
    ORDER BY distance
    LIMIT :n
""")

# Leur similarite a UNE question : celle que l'index aurait rendue avec eux.
_SQL_SIMILARITES = text("""
    SELECT a.id, 1 - (a.embedding <=> CAST(CAST(:vecteur AS text) AS halfvec)) AS similarite
    FROM articles a WHERE a.id = ANY(:ids)
""")


def base(numero: Optional[str]) -> str:
    """Numero d'article de base : « 35.1 » (morceau de l'article 35) -> « 35 »."""
    return str(numero or "").strip().lower().split(".")[0]


def rang(chunks: Sequence[ChunkResult], law_id: int, numero: str) -> Optional[int]:
    """Rang (1-indexe) du premier extrait de l'article attendu, None s'il manque."""
    attendu = (int(law_id), base(numero))
    for position, chunk in enumerate(chunks, start=1):
        if (chunk.law_id, base(chunk.number)) == attendu:
            return position
    return None


def lus_par_le_modele(rag: RAGService, question: str,
                      chunks: List[ChunkResult]) -> List[ChunkResult]:
    """
    Les extraits que le modele lit vraiment : ceux que _build_prompt place,
    apres la remontee de l'article demande, dans le budget du contexte.
    """
    retenus: List[ChunkResult] = []
    build_context_string(rag._pin_requested_article(question, chunks), retenus)
    return retenus


def ajouts_hors_sujet(avant: Sequence[ChunkResult], apres: Sequence[ChunkResult],
                      attendus: Sequence[str]) -> List[str]:
    """
    Les textes que les regles du chat ont AJOUTES au contexte de la recherche
    et qui ne sont pas un texte attendu (references). Une liste vide est le
    seul resultat acceptable sur une question qui marche.
    """
    deja = {c.law_id for c in avant}
    return sorted({
        c.reference for c in apres
        if c.law_id not in deja and c.reference not in set(attendus)
    })


def _de_loi_de_finances(extrait: ChunkResult) -> bool:
    """Un extrait d'une loi de finances ou de reglement (« loi des Finances » compris)."""
    return bool(_TITRE_DE_LOI_DE_FINANCES.search(extrait.law_title or ""))


def tableaux_lus(lus: Sequence[ChunkResult]) -> int:
    """Les extraits de lois de finances que le modele lit : la forme de l'incident."""
    return sum(1 for c in lus if _de_loi_de_finances(c))


def tableaux_de_finances_lus(lus: Sequence[ChunkResult]) -> int:
    """
    Parmi eux, les TABLEAUX (au moins deux lignes « | ... | ») : baremes
    d'amendes, droits, tableaux AE/CP. Ceux que le rejeu de l'incident ne doit
    jamais donner au modele.
    """
    return sum(
        1 for c in lus if _de_loi_de_finances(c)
        and sum(1 for ligne in (c.content or "").splitlines() if ligne.lstrip().startswith("|")) >= 2
    )


def programmes_lus(lus: Sequence[ChunkResult]) -> int:
    """Parmi eux, les tableaux de programmes (search_service.est_un_tableau_de_programmes)."""
    return sum(1 for c in lus if est_un_tableau_de_programmes(c))


def _forme(mot: str) -> str:
    """Un mot sans accents, sans la marque du pluriel (« indemnités » et « indemnité »)."""
    return mot[:-1] if len(mot) > 3 and mot[-1] in "sx" else mot


def porte_ses_mots(question: str, lus: Sequence[ChunkResult]) -> bool:
    """
    Un extrait lu porte-t-il l'un des mots propres de la question
    (rag_service.mots_propres), dans son texte ou son intitule ? Ecrit ICI, et
    non repris du chat : la mesure verifie la corroboration du dehors. Faux
    pour une question sans mot propre.
    """
    propres = {_forme(m) for m in mots_propres(question)}
    if not propres:
        return False
    for extrait in lus:
        texte = " ".join(filter(None, (extrait.content, extrait.law_title, extrait.article_title)))
        if propres & {_forme(m) for m in re.findall(r"[a-z0-9]+", _sans_accents(texte))}:
            return True
    return False


def _taux(lignes: Sequence[Dict[str, Any]], predicat) -> float:
    return round(sum(1 for x in lignes if predicat(x)) / (len(lignes) or 1), 3)


def _arrondi(valeur: Optional[float]) -> Optional[float]:
    return None if valeur is None else round(valeur, 3)


def synthese(lignes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Taux, MRR et compteurs sur les lignes d'une passe (une par question)."""
    n = len(lignes) or 1
    similarites = [x["similarite"] for x in lignes if x.get("similarite") is not None]
    premiers = [x["premier_voisin"] for x in lignes if x.get("premier_voisin") is not None]
    resume: Dict[str, Any] = {
        "n": len(lignes),
        "refus": sum(1 for x in lignes if x["refus"]),
        "faibles": sum(1 for x in lignes if x.get("mode") == "faible"),
        "ajouts": sum(len(x.get("ajouts", [])) for x in lignes),
        "ajouts_hors_sujet": sum(len(x.get("ajouts_hors_sujet") or []) for x in lignes),
        "questions_avec_ajout_hors_sujet": sum(1 for x in lignes if x.get("ajouts_hors_sujet")),
        "tableaux_lus": sum(x.get("tableaux_lus", 0) for x in lignes),
        "questions_avec_tableaux_lus": sum(1 for x in lignes if x.get("tableaux_lus")),
        "tableaux_de_finances_lus": sum(x.get("tableaux_de_finances_lus", 0) for x in lignes),
        "questions_avec_tableaux_de_finances_lus": sum(
            1 for x in lignes if x.get("tableaux_de_finances_lus")
        ),
        "programmes_lus": sum(x.get("programmes_lus", 0) for x in lignes),
        "questions_avec_programmes_lus": sum(1 for x in lignes if x.get("programmes_lus")),
        "tableaux_exclus": sum(x.get("tableaux_exclus", 0) for x in lignes),
        "similarite_min": _arrondi(min(similarites)) if similarites else None,
        "similarite_mediane": _arrondi(statistics.median(similarites)) if similarites else None,
        "similarite_max": _arrondi(max(similarites)) if similarites else None,
        "premier_voisin_min": _arrondi(min(premiers)) if premiers else None,
        "duree_mediane_ms": (
            round(statistics.median(x["duree_ms"] for x in lignes)) if lignes else 0
        ),
        "sql_semantique_median_ms": (
            round(statistics.median(x["sql_ms"] for x in lignes if x.get("sql_ms") is not None))
            if any(x.get("sql_ms") is not None for x in lignes) else None
        ),
        "chemins": {
            chemin: sum(1 for x in lignes if x.get("chemin") == chemin)
            for chemin in sorted({str(x.get("chemin")) for x in lignes})
        },
    }
    if lignes and "rang" in lignes[0]:
        for nom, cle in (("avant", "rang_avant"), ("apres", "rang")):
            resume[nom] = {
                "hit@1": _taux(lignes, lambda x, c=cle: x[c] == 1),
                "hit@3": _taux(lignes, lambda x, c=cle: x[c] is not None and x[c] <= 3),
                "hit@8": _taux(lignes, lambda x, c=cle: x[c] is not None),
                "mrr@8": round(sum(1 / x[cle] for x in lignes if x[cle]) / n, 3),
            }
        resume["apres"]["lu"] = _taux(lignes, lambda x: x["rang_lu"] is not None)
    if lignes and "loi_lue" in lignes[0]:
        resume["loi_lue"] = _taux(lignes, lambda x: x["loi_lue"])
    if lignes and "nomme_juste" in lignes[0]:
        resume["nommes_justes"] = sum(1 for x in lignes if x["nomme_juste"])
        resume["nommes_faux"] = sum(1 for x in lignes if not x["nomme_juste"])
        resume["nommes_ajoutes"] = sum(1 for x in lignes if x.get("article_nomme"))
    # Toutes les lignes d'une passe n'ont pas les memes cles : une relance
    # reconnue n'est pas mesuree seule.
    if any("refus_seule" in x for x in lignes):
        resume["refus_seules"] = sum(1 for x in lignes if x.get("refus_seule"))
    if lignes and "relance" in lignes[0]:
        resume["reconnues"] = sum(1 for x in lignes if x["relance"])
        resume["textes_herites"] = sum(1 for x in lignes if x.get("texte_herite"))
    if any("seule_et_jointe" in x for x in lignes):
        # Cherchees seules ET jointes (rag_service.peut_changer_de_sujet), et
        # celles dont la recherche jointe a ete ecartee faute de corroboration.
        resume["seules_et_jointes"] = sum(1 for x in lignes if x.get("seule_et_jointe"))
        resume["jointes_ecartees"] = sum(1 for x in lignes if x.get("jointe_ecartee"))
        # Dont les suites d'un changement de sujet possible, cherchees avec la
        # fin du sujet et avec le sujet entier.
        resume["suites_d_un_changement"] = sum(1 for x in lignes if x.get("sujet_recent"))
    if any("sans_ses_mots" in x for x in lignes):
        # Lues sans qu'aucun extrait porte un mot propre de la question : la
        # forme de « la dot » lue dans le Code du travail.
        resume["sans_ses_mots"] = sum(1 for x in lignes if x.get("sans_ses_mots"))
    if lignes and "meme_contexte" in lignes[0]:
        resume["meme_contexte"] = sum(1 for x in lignes if x["meme_contexte"])
        resume["prises_pour_des_relances"] = sum(1 for x in lignes if x["relance"])
    return resume


# ==================== CALIBRATION DE LA BARRIERE ====================


def _refuse(similarite: Optional[float], confirmee: Optional[bool], seuil: float,
            par_l_index: bool = False) -> bool:
    if par_l_index:
        # Servi par l'index : seuls les textes confirmes passent, a tout seuil.
        return not confirmee
    return (similarite is None or similarite < seuil) and not confirmee


def refuserait(ligne: Dict[str, Any], seuil: float) -> bool:
    """
    La barriere refuserait-elle cette question a ce seuil ? Sous le seuil (ou
    sans similarite : volet en panne), seulement si rien ne la confirme ;
    servie par l'index (repli), seulement si rien ne la confirme, quel que
    soit le seuil. Une relance reconnue (« relance ») n'est jugee que jointe a
    son sujet ; une autre n'est refusee que si elle l'est seule ET jointe
    (RAGService._chercher_la_relance), de meme qu'une relance cherchee seule et
    jointe (« seule_et_jointe », rag_service.peut_changer_de_sujet). La
    corroboration d'une recherche jointe, quand la question est refusee seule,
    n'est pas rejouee ici : la grille peut dire passee une question que le chat
    refuse (« refus » de la ligne).
    """
    index = ligne.get("chemin") in VOLETS_SERVIS_PAR_L_INDEX
    if _jugee_jointe_seulement(ligne):
        return _refuse(ligne.get("similarite_jointe"), ligne.get("confirmee_jointe"), seuil, index)
    if "similarite_jointe" in ligne:
        return _refuse(ligne.get("similarite_seule"), ligne.get("confirmee_seule"), seuil, index) \
            and _refuse(ligne.get("similarite_jointe"), ligne.get("confirmee_jointe"), seuil, index)
    return _refuse(ligne.get("similarite"), ligne.get("confirmee"), seuil, index)


def _jugee_jointe_seulement(ligne: Dict[str, Any]) -> bool:
    """Une relance reconnue, cherchee seulement jointe a son sujet."""
    return bool(ligne.get("relance")) and not ligne.get("seule_et_jointe")


def _similarite_effective(ligne: Dict[str, Any]) -> Optional[float]:
    """La similarite qui decide : la meilleure des deux recherches pour une relance."""
    if _jugee_jointe_seulement(ligne):
        return ligne.get("similarite_jointe")
    if "similarite_jointe" in ligne:
        valeurs = [v for v in (ligne.get("similarite_seule"), ligne.get("similarite_jointe"))
                   if v is not None]
        return max(valeurs) if valeurs else None
    return ligne.get("similarite")


def _confirmee_effective(ligne: Dict[str, Any]) -> bool:
    if _jugee_jointe_seulement(ligne):
        return bool(ligne.get("confirmee_jointe"))
    if "similarite_jointe" in ligne:
        return bool(ligne.get("confirmee_seule") or ligne.get("confirmee_jointe"))
    return bool(ligne.get("confirmee"))


def _cle_du_seuil(ligne: Dict[str, Any]) -> str:
    """Le seuil que prend la question : francais, ou anglais (langue anglaise ou incertaine)."""
    return "fr" if ligne.get("langue") == "fr" else "en"


GRILLES = {
    "fr": [round(0.20 + i / 100, 2) for i in range(27)],
    "en": [round(0.10 + i / 100, 2) for i in range(31)],
}


def calibration(resultats: Dict[str, Dict[str, Any]],
                seuils: Dict[str, float]) -> Dict[str, Any]:
    """
    Par seuil (cle « fr » ou « en », selon la langue de la QUESTION) : les
    questions legitimes (etat normal), hors sujet (etat normal) et leurrees
    (jeux legitimes, etats leurre et leurre2) que la barriere refuserait, et
    la marge du seuil retenu : sa distance a la plus faible similarite d'une
    question legitime, et a celle d'une question legitime que rien d'autre ne
    confirme (la seule qui compte : une question confirmee passe a tout seuil).
    """
    familles: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"fr": {}, "en": {}}

    def ajouter(famille: str, lignes: Sequence[Dict[str, Any]]) -> None:
        for ligne in lignes:
            familles[_cle_du_seuil(ligne)].setdefault(famille, []).append(ligne)

    for nom, passe in resultats.get("normal", {}).items():
        if nom.startswith(LEGITIMES):
            ajouter("legitimes", passe["items"])
        elif nom.startswith("hors_sujet"):
            ajouter("hors_sujet", passe["items"])
    for etat in ("leurre", "leurre2"):
        for nom, passe in resultats.get(etat, {}).items():
            if nom.startswith(LEGITIMES):
                ajouter(etat, passe["items"])

    sortie: Dict[str, Any] = {}
    for cle, groupes in familles.items():
        legitimes = groupes.get("legitimes", [])
        sims = [s for s in map(_similarite_effective, legitimes) if s is not None]
        seules = [
            _similarite_effective(x) for x in legitimes
            if _similarite_effective(x) is not None and not _confirmee_effective(x)
        ]
        seuil = seuils[cle]
        sortie[cle] = {
            "seuil": seuil,
            "effectifs": {famille: len(lignes) for famille, lignes in groupes.items()},
            "legitimes_similarite_min": _arrondi(min(sims)) if sims else None,
            "marge": _arrondi(min(sims) - seuil) if sims else None,
            "legitimes_sans_autre_preuve_similarite_min": _arrondi(min(seules)) if seules else None,
            "marge_sans_autre_preuve": _arrondi(min(seules) - seuil) if seules else None,
            "legitimes_les_plus_faibles": sorted(
                ({"question": x["question"], "similarite": _similarite_effective(x),
                  "confirmee": _confirmee_effective(x)} for x in legitimes
                 if _similarite_effective(x) is not None),
                key=lambda x: x["similarite"],
            )[:8],
            "hors_sujet_similarite_max": _arrondi(max(
                (x["similarite"] for x in groupes.get("hors_sujet", [])
                 if x.get("similarite") is not None), default=None,
            )) if groupes.get("hors_sujet") else None,
            "grille": [
                {
                    "seuil": t,
                    **{f"refus_{famille}": sum(1 for x in lignes if refuserait(x, t))
                       for famille, lignes in groupes.items()},
                }
                for t in GRILLES[cle]
            ],
        }
    return sortie


# ==================== VOLET SEMANTIQUE SIMULE ====================


async def _rien() -> None:
    return None


@contextmanager
def volet_semantique(etat: str, listes: Dict[Any, List[ChunkResult]],
                     cle_courante: Dict[str, Any]) -> Iterator[None]:
    """
    Remplace le volet semantique de SearchService, sans appel au fournisseur
    d'embeddings : « leurre » et « leurre2 » rendent
    `listes[cle_courante["cle"]]`, « panne » leve l'erreur d'un fournisseur
    injoignable. Tout autre etat : le volet reel, inchange.

    Des COPIES a chaque appel : la fusion RRF ecrase relevance_score en place.
    """
    if etat not in ("leurre", "leurre2", "panne"):
        yield
        return

    classe = module_recherche.SearchService
    lancer, chercher = classe._lancer_vectorisation, classe._chunks_semantiques

    def _lancer(self, texte):
        return asyncio.ensure_future(_rien())

    async def _chercher(self, tache_vecteur, filters, limit, offset):
        await module_recherche._solder(tache_vecteur)
        if etat == "panne":
            raise module_recherche.VectorSearchError("fournisseur injoignable (simule)")
        liste = listes.get(cle_courante["cle"], [])
        # Le chat ne lit l'index que par le repli de sa recherche exacte.
        self.chemin_semantique = module_recherche.CHEMIN_REPLI_HNSW
        return [c.model_copy(deep=True) for c in liste[offset:offset + limit]]

    classe._lancer_vectorisation, classe._chunks_semantiques = _lancer, _chercher
    try:
        yield
    finally:
        classe._lancer_vectorisation, classe._chunks_semantiques = lancer, chercher


# ==================== MESURE ====================


# Fichier des vecteurs deja calcules (--vecteurs) : une seconde mesure ne
# rappelle pas le fournisseur pour les memes questions.
_FICHIER_DES_VECTEURS: Optional[Path] = None


async def _vecteurs(questions: Sequence[str]) -> Dict[str, Any]:
    """
    Vectorise chaque question une fois, avant toute passe : les durees des
    passes ne mesurent plus que la base et le classement, et se comparent.
    """
    import numpy as np

    service = module_recherche._embedding_service_instance
    vecteurs: Dict[str, Any] = {}
    fichier = _FICHIER_DES_VECTEURS
    if fichier is not None and fichier.exists():
        vecteurs = {
            q: np.asarray(v, dtype=np.float32)
            for q, v in json.loads(fichier.read_text(encoding="utf-8")).items()
        }
    for question in dict.fromkeys(questions):
        if question in vecteurs:
            continue
        vecteurs[question] = await _vecteur_reessaye(service, question)
    if fichier is not None:
        fichier.parent.mkdir(parents=True, exist_ok=True)
        fichier.write_text(json.dumps(
            {q: [float(x) for x in v] for q, v in vecteurs.items()}
        ), encoding="utf-8")
    return vecteurs


async def _vecteur_reessaye(service: Any, texte: str, task_type: Any = None,
                            essais: int = 8) -> Any:
    """
    Le vecteur de `texte`, le fournisseur reessaye : le reseau a coupe une
    mesure de vingt minutes (ECONNRESET, nuit du 30/10/2026), et une coupure
    de quelques secondes ne doit pas la faire echouer.
    """
    for essai in range(essais):
        try:
            return await service.generate_embedding_async(
                texte, task_type=task_type or service.TASK_QUERY
            )
        except Exception:
            if essai == essais - 1:
                raise
            await asyncio.sleep(min(3 * (essai + 1), 15))


class _VecteursEnMemoire:
    """Le fournisseur d'embeddings, servi par les vecteurs deja calcules."""

    def __init__(self, service: Any, vecteurs: Dict[str, Any]) -> None:
        self._service, self._vecteurs = service, vecteurs
        self.TASK_QUERY = service.TASK_QUERY
        self.TASK_DOCUMENT = getattr(service, "TASK_DOCUMENT", None)

    async def generate_embedding_async(self, texte: str, task_type: Any = None) -> Any:
        if texte not in self._vecteurs:
            self._vecteurs[texte] = await _vecteur_reessaye(self._service, texte, task_type)
        return self._vecteurs[texte]


def _litteral(vecteur: Any) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vecteur) + "]"


def _extraits(lignes: Sequence[Any]) -> List[ChunkResult]:
    return [
        ChunkResult(
            article_id=r.article_id, law_id=r.law_id,
            number=str(r.number) if r.number else None,
            article_title=r.article_title, section=r.section, page_number=r.page_number,
            content=r.content or "", excerpt=(r.content or "")[:400],
            reference=r.reference or "", law_title=r.law_title or "",
            type=r.type or "loi", language=r.language, status=r.status or "published",
            category_id=r.category_id, category_name=r.category_name,
            publication_date=r.publication_date,
            relevance_score=max(0.0, min(1.0, 1.0 - float(r.distance))),
            source="semantic",
        )
        for r in lignes
    ]


async def tableaux_de_l_incident(vecteur: Any, n: int = 32) -> List[ChunkResult]:
    """Les `n` morceaux de lois de finances les plus proches de la question de l'incident."""
    async with AsyncSessionLocal() as session:
        lignes = (await session.execute(_SQL_TABLEAUX, {
            "vecteur": _litteral(vecteur), "n": n,
        })).all()
    return _extraits(lignes)


async def voisins_hors_finances(vecteur: Any, n: int = 32) -> List[ChunkResult]:
    """Les `n` articles publies les plus proches d'un vecteur, hors lois de finances."""
    async with AsyncSessionLocal() as session:
        lignes = (await session.execute(_SQL_VOISINS_HORS_FINANCES, {
            "vecteur": _litteral(vecteur), "n": n,
        })).all()
    return _extraits(lignes)


async def leurre_pour(tableaux: List[ChunkResult], vecteur: Any) -> List[ChunkResult]:
    """Les extraits du leurre, avec leur VRAIE similarite a la question posee, dans cet ordre."""
    async with AsyncSessionLocal() as session:
        similarites = dict((await session.execute(_SQL_SIMILARITES, {
            "vecteur": _litteral(vecteur), "ids": [c.article_id for c in tableaux],
        })).all())
    copies = [
        c.model_copy(update={"relevance_score": max(0.0, min(1.0, float(similarites[c.article_id])))})
        for c in tableaux
    ]
    return sorted(copies, key=lambda c: -c.relevance_score)


async def _confirmee(rag: RAGService, question: str) -> Optional[bool]:
    """
    La question passerait-elle la barriere SOUS le seuil ? Le meme calcul que
    _textes_confirmes, sur le contexte juge que garde l'etat ; None sans
    volets.
    """
    recuperation = rag._recuperation
    volets = rag.search_service.volets
    if volets is None:
        return None
    if recuperation.mode == "faible":
        return not recuperation.refus
    # Les mots-cles que la recuperation a retenus (ceux d'une relance et de
    # son sujet, ou de la question).
    confirmes = await rag._textes_confirmes(
        recuperation.mots_cles, recuperation.juge, recuperation.nommes, volets
    )
    return bool(confirmes)


async def contexte(question: str, etat: str
                   ) -> Tuple[RAGService, List[ChunkResult], float, Optional[bool]]:
    """
    (service, contexte final, duree en ms, confirmee) pour UNE question, par le
    chemin du chat. `confirmee` : la question passerait-elle la barriere sous le
    seuil ; None sans volets.
    """
    async with AsyncSessionLocal() as db:
        rag = _rag(db, etat)
        debut = time.monotonic()
        chunks = await rag._retrieve_chunks(question, "fr")
        duree_ms = (time.monotonic() - debut) * 1000
        confirmee = await _confirmee(rag, question)
        await db.rollback()
    return rag, chunks, duree_ms, confirmee


class _Message:
    """Un message d'historique : ce que RAGService en lit (role, contenu, sources)."""

    def __init__(self, role: str, content: str, sources: Optional[List[Any]] = None) -> None:
        self.role, self.content, self.sources = role, content, sources


def historique_de(precedentes: Sequence[str], reponse: str = "Réponse.") -> List[_Message]:
    """Les questions precedentes de l'utilisateur, chacune suivie d'une reponse."""
    messages: List[_Message] = []
    for question in precedentes:
        messages += [_Message("user", question), _Message("assistant", reponse)]
    return messages


def _sim(valeur: Optional[float]) -> Optional[float]:
    return None if valeur is None else round(valeur, 4)


async def contexte_de_relance(precedentes: Sequence[str], relance: str, etat: str,
                              reponse: str = "Réponse.",
                              attendus: Optional[Sequence[str]] = None,
                              langue: str = "fr") -> Dict[str, Any]:
    """
    Une relance par le CHEMIN DU CHAT (RAGService._recuperer), apres
    `precedentes` : reconnue (est_une_relance), elle est cherchee jointe au
    sujet de la conversation, et AUSSI seule quand elle apporte ses mots
    apres une ouverture (« seule_et_jointe », peut_changer_de_sujet) ; sinon
    seule, puis, refusee, jointe. Pour la grille, une relance non reconnue ou
    cherchee seule et jointe est mesuree des deux facons, quoi qu'ait decide
    le chat. `attendus` : les textes dont un extrait doit etre lu
    (« loi_lue »). « sans_ses_mots » : lue, avec des mots propres, sans
    qu'aucun extrait lu n'en porte un (porte_ses_mots).
    """
    from app.schemas.rag import RAGRequest
    from app.services.rag_service import sujet_de_la_conversation

    historique = historique_de(precedentes, reponse)
    async with AsyncSessionLocal() as db:
        rag = _rag(db, etat)
        debut = time.monotonic()
        chunks, _ = await rag._recuperer(
            RAGRequest(question=relance, language=langue), None, historique
        )
        duree_ms = (time.monotonic() - debut) * 1000
        finale = rag._recuperation
        confirmee = await _confirmee(rag, relance)
        volets_finaux = rag.search_service.volets
        lus = lus_par_le_modele(rag, relance, chunks)
        ligne: Dict[str, Any] = {
            "question": relance, "precedentes": list(precedentes),
            "relance": finale.relance, "question_jointe": finale.question_jointe,
            "texte_herite": finale.texte_herite, "article_nomme": finale.article_nomme,
            "langue": finale.langue, "mode": finale.mode, "chemin": finale.chemin,
            "refus": finale.refus, "budgetaire": finale.budgetaire,
            "similarite": _sim(finale.similarite),
            "premier_voisin": (
                None if volets_finaux is None else _sim(volets_finaux.similarite_max)
            ),
            "tableaux_lus": tableaux_lus(lus), "programmes_lus": programmes_lus(lus),
            "tableaux_de_finances_lus": tableaux_de_finances_lus(lus),
            "tableaux_exclus": finale.tableaux_exclus,
            "ajouts": list(finale.ajouts), "ajouts_hors_sujet": None,
            "duree_ms": round(duree_ms, 1), "sql_ms": rag.search_service.duree_sql_semantique_ms,
            "lus": [f"{c.reference} art. {c.number}" for c in lus],
        }
        if attendus is not None:
            ligne["loi_lue"] = any(c.reference in attendus for c in lus)
        # getattr : la mesure tourne aussi sur le code d'avant (tour 4), sans
        # ces champs.
        seule_et_jointe = bool(getattr(finale, "seule_et_jointe", False))
        sujet_recent = getattr(finale, "sujet_recent", None)
        ligne["seule_et_jointe"] = seule_et_jointe
        ligne["sujet_recent"] = sujet_recent
        ligne["jointe_ecartee"] = bool(getattr(finale, "jointe_ecartee", False))
        ligne["porte_ses_mots"] = porte_ses_mots(relance, lus)
        ligne["sans_ses_mots"] = bool(lus) and bool(mots_propres(relance)) \
            and not ligne["porte_ses_mots"]
        if finale.relance and not seule_et_jointe:
            ligne["similarite_jointe"], ligne["confirmee_jointe"] = ligne["similarite"], confirmee
        else:
            # Non reconnue, ou cherchee seule et jointe : les deux recherches,
            # toujours mesurees. La suite d'un changement de sujet possible
            # (« sujet_recent ») : sa recherche « seule » est celle de la fin
            # du sujet, comme dans le chat.
            await rag._retrieve_chunks(sujet_recent or relance, langue)
            ligne["refus_seule"] = rag._recuperation.refus
            ligne["similarite_seule"] = _sim(rag._recuperation.similarite)
            ligne["confirmee_seule"] = await _confirmee(rag, relance)
            sujet = sujet_de_la_conversation(historique)
            if sujet is not None:
                if seule_et_jointe:
                    # La recherche jointe du chat : sujet + relance, ses mots
                    # propres en tete des mots-cles.
                    await rag._une_recherche(relance, historique, sujet)
                else:
                    await rag._retrieve_chunks(f"{sujet.texte} {relance}", langue)
                ligne["similarite_jointe"] = _sim(rag._recuperation.similarite)
                ligne["confirmee_jointe"] = await _confirmee(rag, relance)
        await db.rollback()
    return ligne


async def contexte_avec_historique(question: str, precedente: str, etat: str
                                   ) -> Tuple[RAGService, List[ChunkResult]]:
    """Une question de fond posee apres `precedente`, par le chemin du chat."""
    from app.schemas.rag import RAGRequest

    async with AsyncSessionLocal() as db:
        rag = _rag(db, etat)
        chunks, _ = await rag._recuperer(
            RAGRequest(question=question, language="fr"), None, historique_de([precedente])
        )
        await db.rollback()
    return rag, chunks


def _rag(db: Any, etat: str) -> RAGService:
    rag = RAGService(db)
    # Aucun appel au modele de generation, quoi que dise .env.
    rag.llm = None
    rag.search_service.use_cache = False
    if _VECTEURS is not None:
        rag.search_service.embedding_service = _VECTEURS
    if etat == "index":
        rag.search_service.semantique_exacte = False
    if etat == "repli":
        # Toute recherche exacte depasse son budget : le volet passe par
        # l'index (ef_search 1000), comme sous une surcharge. Le budget a
        # froid aussi : sinon la premiere recherche, ou toute recherche apres
        # 4 min sans succes, aboutirait par le chemin exact.
        rag.search_service.BUDGET_RECHERCHE_EXACTE_MS = 1
        rag.search_service.BUDGET_RECHERCHE_EXACTE_A_FROID_MS = 1
    return rag


# Les vecteurs des questions, calcules une fois (voir evaluer) ; None : le
# fournisseur, a chaque question.
_VECTEURS: Optional[_VecteursEnMemoire] = None


def _ligne_commune(rag: RAGService, question: str, chunks: List[ChunkResult],
                   duree_ms: float, attendus: Optional[Sequence[str]],
                   confirmee: Optional[bool]) -> Dict[str, Any]:
    etat = rag._recuperation
    volets = rag.search_service.volets
    lus = lus_par_le_modele(rag, question, chunks)
    premier = volets.similarite_max if volets is not None else None
    return {
        "question": question,
        "langue": etat.langue, "mode": etat.mode, "chemin": etat.chemin,
        "similarite": None if etat.similarite is None else round(etat.similarite, 4),
        "premier_voisin": None if premier is None else round(premier, 4),
        "confirmee": confirmee,
        "refus": etat.refus,
        "article_nomme": etat.article_nomme,
        "budgetaire": etat.budgetaire,
        "ajouts": list(etat.ajouts),
        # Sans texte attendu connu (relances, questions courtes), un ajout
        # n'est ni juste ni hors sujet : non compte.
        "ajouts_hors_sujet": (
            None if attendus is None else ajouts_hors_sujet(etat.hybride, chunks, attendus)
        ),
        "tableaux_lus": tableaux_lus(lus),
        "programmes_lus": programmes_lus(lus),
        "tableaux_de_finances_lus": tableaux_de_finances_lus(lus),
        "tableaux_exclus": etat.tableaux_exclus,
        "duree_ms": round(duree_ms, 1),
        "sql_ms": rag.search_service.duree_sql_semantique_ms,
        "lus": [f"{c.reference} art. {c.number}" for c in lus],
        "avant": [f"{c.reference} art. {c.number}" for c in etat.hybride],
        "_lus": lus,
    }


# Les jeux a mesurer (--jeux) ; None : tous.
_JEUX_DEMANDES: Optional[Sequence[str]] = None


def _demande(nom: str) -> bool:
    return _JEUX_DEMANDES is None or any(nom.startswith(j) for j in _JEUX_DEMANDES)


async def evaluer(etats: Sequence[str], questions_en: Optional[Path],
                  limite: Optional[int]) -> Dict[str, Any]:
    global _VECTEURS
    items = json.loads(JEU.read_text(encoding="utf-8"))["items"]
    if limite:
        items = items[:limite]
    references = await _references([int(it["expected_law_id"]) for it in items])
    jeu = {"fr": [it["question"] for it in items]}
    if questions_en:
        jeu["en"] = json.loads(questions_en.read_text(encoding="utf-8"))[:len(items)]

    relances = {
        langue: list(paires) + [
            (RELANCES_AVEC_HISTORIQUE[langue][0][0], r) for r in RELANCES[langue]
        ]
        for langue, paires in RELANCES_AVEC_HISTORIQUE.items()
    }
    from app.services.rag_service import est_une_relance, sujet_de_la_conversation
    toutes = [q for qs in jeu.values() for q in qs]
    toutes += [q for qs in SUGGEREES.values() for q, _ in qs]
    toutes += [q for qs in HORS_SUJET.values() for q in qs]
    toutes += [q for qs in NOMMEES.values() for q, _, _ in qs]
    toutes += [q for qs in NOMMEES_REVUE.values() for q, _ in qs]
    toutes += [q for jeux in (REELLES, MONTANTS, MONTANTS_PREVUS, BUDGETAIRES, COURTES)
               for qs in jeux.values() for q in qs]
    toutes += [x for paires in relances.values() for p, r in paires for x in (p, r, f"{p} {r}")]
    # Ce que le chat vectorise pour une relance : son sujet, puis elle ; et
    # elle seule (changement de sujet possible, seconde chance).
    for jeux in (RELANCES_EN_CHAINE, CHANGEMENTS_DE_SUJET, SECONDES_CHANCES):
        for chaines in jeux.values():
            for precedentes, relance, _ in chaines:
                toutes += list(precedentes) + [relance]
                sujet = sujet_de_la_conversation(historique_de(precedentes))
                if sujet is not None:
                    toutes.append(f"{sujet.texte} {relance}")
                    # La suite d'un changement de sujet possible cherche aussi
                    # la fin du sujet seule (getattr : la mesure tourne aussi
                    # sur le code du tour 4, qui ne la cherche pas).
                    fin = getattr(sujet, "depuis_un_changement_possible", lambda: None)()
                    if fin is not None:
                        toutes.append(fin.texte)
    vecteurs = await _vecteurs(toutes + [QUESTION_DE_L_INCIDENT, QUESTION_DU_LEURRE2])
    _VECTEURS = _VecteursEnMemoire(module_recherche._embedding_service_instance, vecteurs)
    tableaux = (
        await tableaux_de_l_incident(vecteurs[QUESTION_DE_L_INCIDENT])
        if "leurre" in etats else []
    )
    quelconques = (
        await voisins_hors_finances(vecteurs[QUESTION_DU_LEURRE2])
        if "leurre2" in etats else []
    )

    sortie: Dict[str, Any] = {
        "etats": list(etats),
        "seuils": dict(RAGService.SEUILS_DE_SIMILARITE),
        "tableaux_de_l_incident": [f"{c.reference} art. {c.number}" for c in tableaux[:8]],
        "leurre2": [f"{c.reference} art. {c.number}" for c in quelconques[:8]],
        "resultats": {},
    }
    cle_courante: Dict[str, Any] = {"cle": None}
    listes: Dict[Any, List[ChunkResult]] = {}

    async def passe(etat: str, nom: str, questions: List[Tuple[str, Dict[str, Any]]]):
        if not _demande(nom):
            return
        lignes = []
        for question, attendu in questions:
            cle_courante["cle"] = question
            if etat in ("leurre", "leurre2"):
                listes[question] = await leurre_pour(
                    tableaux if etat == "leurre" else quelconques, vecteurs[question]
                )
            with volet_semantique(etat, listes, cle_courante):
                rag, chunks, duree_ms, confirmee = await contexte(question, etat)
            ligne = _ligne_commune(
                rag, question, chunks, duree_ms,
                None if attendu.get("sans_reference") else attendu.get("references", ()),
                confirmee,
            )
            lus = ligne.pop("_lus")
            if "law_id" in attendu:
                ligne["rang_avant"] = rang(rag._recuperation.hybride, attendu["law_id"],
                                           attendu["numero"])
                ligne["rang"] = rang(chunks, attendu["law_id"], attendu["numero"])
                ligne["rang_lu"] = rang(lus, attendu["law_id"], attendu["numero"])
            if attendu.get("references"):
                ligne["loi_lue"] = any(c.reference in attendu["references"] for c in lus)
            if "nomme" in attendu:
                reference, numero = attendu["nomme"]
                ligne["nomme_en_tete"] = bool(chunks) and chunks[0].reference == reference \
                    and chunks[0].source == "article_nomme"
            if "nomme_attendu" in attendu:
                ajoute = {c.reference for c in rag._recuperation.nommes}
                ligne["nomme_attendu"] = attendu["nomme_attendu"]
                ligne["nomme_juste"] = (
                    ajoute == ({attendu["nomme_attendu"]} if attendu["nomme_attendu"] else set())
                )
            lignes.append(ligne)
        resume = synthese(lignes)
        if lignes and "nomme_en_tete" in lignes[0]:
            resume["nomme_en_tete"] = sum(1 for x in lignes if x["nomme_en_tete"])
        sortie["resultats"].setdefault(etat, {})[nom] = {"synthese": resume, "items": lignes}
        print(_ligne(etat, nom, resume), flush=True)

    async def passe_de_relances(etat: str, nom: str, paires: List[Tuple[str, str]], langue: str):
        if not _demande(nom):
            return
        lignes = []
        for precedente, relance in paires:
            cle_courante["cle"] = relance
            lignes.append(await contexte_de_relance(
                [precedente], relance, etat, attendus=("LOI-1992-007",), langue=langue,
            ))
        resume = synthese(lignes)
        sortie["resultats"].setdefault(etat, {})[nom] = {"synthese": resume, "items": lignes}
        print(_ligne(etat, nom, resume), flush=True)

    async def passe_de_chaines(etat: str, nom: str, chaines, langue: str):
        if not _demande(nom):
            return
        lignes = []
        for precedentes, relance, attendus in chaines:
            cle_courante["cle"] = relance
            lignes.append(await contexte_de_relance(
                precedentes, relance, etat, reponse=REPONSE_PRECEDENTE[langue],
                # Sans texte attendu (secondes chances) : pas de « loi_lue ».
                attendus=attendus or None, langue=langue,
            ))
        resume = synthese(lignes)
        sortie["resultats"].setdefault(etat, {})[nom] = {"synthese": resume, "items": lignes}
        print(_ligne(etat, nom, resume), flush=True)

    async def passe_avec_historique(etat: str, nom: str, questions: List[str],
                                    precedente: str, seules: Dict[str, List[str]]):
        """
        Des questions de fond posees apres une autre : aucune ne doit etre
        prise pour une relance, et leur contexte doit etre celui qu'elles ont
        seules (`seules`, de la passe sans historique).
        """
        if not _demande(nom):
            return
        lignes = []
        for question in questions:
            cle_courante["cle"] = question
            rag, chunks = await contexte_avec_historique(question, precedente, etat)
            lus = lus_par_le_modele(rag, question, chunks)
            vus = [f"{c.reference} art. {c.number}" for c in lus]
            lignes.append({
                "question": question, "relance": est_une_relance(question),
                "refus": rag._recuperation.refus, "mode": rag._recuperation.mode,
                "meme_contexte": vus == seules.get(question), "lus": vus,
            })
        resume = {
            "n": len(lignes), "refus": sum(1 for x in lignes if x["refus"]),
            "meme_contexte": sum(1 for x in lignes if x["meme_contexte"]),
            "prises_pour_des_relances": sum(1 for x in lignes if x["relance"]),
        }
        sortie["resultats"].setdefault(etat, {})[nom] = {"synthese": resume, "items": lignes}
        print(f"{etat:7} {nom:20} n={resume['n']:3} refus {resume['refus']:3} meme contexte "
              f"{resume['meme_contexte']} prises pour des relances "
              f"{resume['prises_pour_des_relances']}", flush=True)

    for etat in etats:
        for langue, questions in jeu.items():
            await passe(etat, f"jeu_{langue}", [
                (q, {"law_id": int(it["expected_law_id"]), "numero": it["expected_article_number"],
                     "references": (references.get(int(it["expected_law_id"]), ""),)})
                for q, it in zip(questions, items)
            ])
        for langue, questions in SUGGEREES.items():
            await passe(etat, f"suggerees_{langue}", [
                (q, {"references": refs}) for q, refs in questions
            ])
        for langue, questions in HORS_SUJET.items():
            await passe(etat, f"hors_sujet_{langue}", [(q, {}) for q in questions])
        for langue, questions in NOMMEES.items():
            await passe(etat, f"nommees_{langue}", [
                (q, {"references": (ref,), "nomme": (ref, numero), "nomme_attendu": ref})
                for q, ref, numero in questions
            ])
        for langue, questions in NOMMEES_REVUE.items():
            await passe(etat, f"nommees_revue_{langue}", [
                (q, {"sans_reference": True, "nomme_attendu": ref}) for q, ref in questions
            ])
        for nom, jeux in (("reelles", REELLES), ("montants", MONTANTS),
                          ("montants_prevus", MONTANTS_PREVUS),
                          ("budgetaires", BUDGETAIRES), ("courtes", COURTES)):
            for langue, questions in jeux.items():
                await passe(etat, f"{nom}_{langue}",
                            [(q, {"sans_reference": True}) for q in questions])
        # Les relances ne se mesurent que dans l'etat normal : l'historique
        # ne change rien a une panne.
        if etat == "normal":
            for langue, paires in relances.items():
                await passe_de_relances(etat, f"relances_{langue}", paires, langue)
            for langue, chaines in RELANCES_EN_CHAINE.items():
                await passe_de_chaines(etat, f"chaines_{langue}", chaines, langue)
            for langue, chaines in CHANGEMENTS_DE_SUJET.items():
                await passe_de_chaines(etat, f"changements_de_sujet_{langue}", chaines, langue)
            for langue, chaines in SECONDES_CHANCES.items():
                await passe_de_chaines(etat, f"secondes_chances_{langue}", chaines, langue)
            for langue in ("fr", "en"):
                seules = {
                    x["question"]: x["lus"]
                    for nom in (f"reelles_{langue}", f"suggerees_{langue}")
                    for x in sortie["resultats"].get(etat, {}).get(nom, {}).get("items", [])
                }
                questions = REELLES[langue] + [q for q, _ in SUGGEREES[langue]]
                await passe_avec_historique(
                    etat, f"fond_avec_historique_{langue}", questions,
                    RELANCES_AVEC_HISTORIQUE[langue][0][0], seules,
                )

    sortie["calibration"] = calibration(sortie["resultats"], sortie["seuils"])
    for cle, valeurs in sortie["calibration"].items():
        print(_ligne_de_calibration(cle, valeurs), flush=True)
    return sortie


async def _references(law_ids: Sequence[int]) -> Dict[int, str]:
    async with AsyncSessionLocal() as session:
        lignes = (await session.execute(
            text("SELECT id, reference FROM laws WHERE id = ANY(:ids)"), {"ids": list(set(law_ids))}
        )).all()
    return {int(i): r for i, r in lignes}


def _ligne(etat: str, nom: str, s: Dict[str, Any]) -> str:
    morceaux = [f"{etat:7} {nom:20} n={s['n']:3} refus {s['refus']:3} faibles {s['faibles']:3}"]
    if "apres" in s:
        a, p = s["avant"], s["apres"]
        morceaux.append(
            f"hit@8 {a['hit@8']:.3f}->{p['hit@8']:.3f} hit@3 {a['hit@3']:.3f}->{p['hit@3']:.3f} "
            f"mrr@8 {a['mrr@8']:.3f}->{p['mrr@8']:.3f} lu {p['lu']:.3f}"
        )
    if "loi_lue" in s:
        morceaux.append(f"loi_lue {s['loi_lue']:.3f}")
    if "nomme_en_tete" in s:
        morceaux.append(f"nomme_en_tete {s['nomme_en_tete']}")
    if "nommes_justes" in s:
        morceaux.append(f"nommes justes {s['nommes_justes']} faux {s['nommes_faux']}")
    if "refus_seules" in s:
        morceaux.append(f"refus seules {s['refus_seules']}")
    if "reconnues" in s:
        morceaux.append(f"reconnues {s['reconnues']} textes herites {s['textes_herites']}")
    if "seules_et_jointes" in s:
        morceaux.append(
            f"seules+jointes {s['seules_et_jointes']} jointes ecartees {s['jointes_ecartees']}"
            f" suites d'un changement {s.get('suites_d_un_changement', 0)}"
        )
    if "sans_ses_mots" in s:
        morceaux.append(f"sans ses mots {s['sans_ses_mots']}")
    morceaux.append(
        f"ajouts {s['ajouts']} hors_sujet {s['ajouts_hors_sujet']} "
        f"tableaux_lus {s['tableaux_lus']} ({s['questions_avec_tableaux_lus']} q.) "
        f"dont tableaux {s['tableaux_de_finances_lus']} "
        f"({s['questions_avec_tableaux_de_finances_lus']} q.) "
        f"programmes_lus {s['programmes_lus']} exclus {s['tableaux_exclus']} "
        f"sim min {s['similarite_min']} med {s['similarite_mediane']} max {s['similarite_max']} "
        f"{s['duree_mediane_ms']} ms (SQL sem. {s['sql_semantique_median_ms']} ms) {s['chemins']}"
    )
    return "  ".join(morceaux)


def _ligne_de_calibration(cle: str, c: Dict[str, Any]) -> str:
    retenu = next((g for g in c["grille"] if g["seuil"] == c["seuil"]), None)
    return (
        f"calibration {cle} : seuil {c['seuil']} effectifs {c['effectifs']} "
        f"legitimes sim min {c['legitimes_similarite_min']} (marge {c['marge']}), "
        f"sans autre preuve {c['legitimes_sans_autre_preuve_similarite_min']} "
        f"(marge {c['marge_sans_autre_preuve']}), hors sujet sim max "
        f"{c['hors_sujet_similarite_max']} ; au seuil retenu {retenu}"
    )


# ==================== RELEVE : TABLE DES NOMS, TABLEAUX ====================


async def releve() -> Dict[str, Any]:
    """
    Ce que les regles du chat supposent du corpus, relu en base :
    - chaque texte de TEXTES_NOMMABLES : publie, et son intitule commence bien
      comme la table le dit ;
    - les tableaux de programmes des lois de finances que le chat retire,
      comptes par marque, et ce qui reste des tableaux de lois de finances.
    """
    from app.services import search_service as s

    sortie: Dict[str, Any] = {"table": [], "tableaux": {}}
    async with AsyncSessionLocal() as session:
        for texte in TEXTES_NOMMABLES:
            ligne = (await session.execute(text(
                "SELECT l.title, l.status, (SELECT count(*) FROM articles a WHERE a.law_id = l.id) "
                "AS n FROM laws l WHERE l.reference = :r"
            ), {"r": texte.reference})).first()
            conforme = ligne is not None and ligne.status == "published" and " ".join(
                (ligne.title or "").split()
            ).lower().startswith(" ".join(texte.intitule.split()).lower())
            sortie["table"].append({
                "reference": texte.reference, "conforme": conforme,
                "intitule_en_base": None if ligne is None else ligne.title,
                "statut": None if ligne is None else ligne.status,
                "articles": None if ligne is None else ligne.n,
            })
            print(f"{'OK ' if conforme else 'ECART'} {texte.reference:20} "
                  f"{(ligne.title if ligne else 'ABSENT')[:110]}", flush=True)
        lignes = (await session.execute(text(
            "SELECT a.id, l.reference, a.number, l.title AS law_title, a.content "
            "FROM articles a JOIN laws l ON l.id = a.law_id WHERE l.status = 'published'"
        ))).all()
    marques = {
        "ligne_ae_cp": s._LIGNE_AE_CP, "programmes_concourant": s._PROGRAMMES_ET_OBJECTIFS,
    }
    exclus, textes, par_marque, restes = 0, set(), {}, 0
    for r in lignes:
        extrait = ChunkResult(article_id=r.id, law_id=0, number=r.number, content=r.content or "",
                              reference=r.reference, law_title=r.law_title or "")
        finances = bool(s._TITRE_DE_LOI_DE_FINANCES.search(r.law_title or ""))
        if est_un_tableau_de_programmes(extrait):
            exclus += 1
            textes.add(r.reference)
            for nom, motif in marques.items():
                if motif.search(r.content or ""):
                    par_marque[nom] = par_marque.get(nom, 0) + 1
            if s._AUTORISATIONS_D_ENGAGEMENT.search(r.content or "") and \
                    s._CREDITS_DE_PAIEMENT.search(r.content or ""):
                par_marque["intro_ae_cp"] = par_marque.get("intro_ae_cp", 0) + 1
        elif finances and (r.content or "").count("\n|") >= 2:
            restes += 1
    sortie["tableaux"] = {
        "morceaux_retires": exclus, "textes": len(textes), "par_marque": par_marque,
        "tableaux_de_lois_de_finances_gardes": restes,
    }
    print(json.dumps(sortie["tableaux"], ensure_ascii=False), flush=True)
    return sortie


def _sonder_le_fournisseur(vecteurs: Optional[Path], essais: int = 5) -> bool:
    """
    La sonde du fournisseur d'embeddings (precharger_les_embeddings), reessayee.
    Un echec PASSAGER garde le service : avec un fichier de vecteurs (--vecteurs),
    la mesure part quand meme, les questions connues servies par le fichier et
    les autres reessayant le fournisseur (_vecteur_reessaye). Un reseau
    instable a coupe la mesure du 30/10/2026 ; une mesure sans volet semantique,
    elle, ne part jamais (service ecarte).
    """
    for essai in range(essais):
        if module_recherche.precharger_les_embeddings():
            return True
        if module_recherche._embedding_service_instance is None:
            return False
        if essai < essais - 1:
            time.sleep(3 * (essai + 1))
    return vecteurs is not None and vecteurs.exists()


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _JEUX_DEMANDES, _FICHIER_DES_VECTEURS
    parseur = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parseur.add_argument("--etats", nargs="+", default=["normal", "index", "leurre", "panne"],
                         choices=["normal", "index", "repli", "leurre", "leurre2", "panne"])
    parseur.add_argument("--questions-en", type=Path, default=None,
                         help="traductions anglaises du jeu, dans son ordre (liste JSON)")
    parseur.add_argument("--limite", type=int, default=None, help="n premieres questions du jeu")
    parseur.add_argument("--jeux", nargs="+", default=None,
                         help="ne mesurer que ces jeux (prefixes : jeu, reelles, relances...)")
    parseur.add_argument("--releve", action="store_true",
                         help="relire la table des noms et compter les tableaux, sans mesure")
    parseur.add_argument("--vecteurs", type=Path, default=None,
                         help="fichier JSON des vecteurs des questions, relu et complete")
    parseur.add_argument("--sortie", type=Path, help="ecrit le detail en JSON")
    args = parseur.parse_args(argv)
    _JEUX_DEMANDES = args.jeux
    _FICHIER_DES_VECTEURS = args.vecteurs

    # Le re-ranking par le modele est un appel facture : jamais ici.
    settings.RERANK_LLM_ENABLED = False
    # Le journal SQL (DEBUG, actif par defaut) noierait les resultats.
    moteur.echo = False

    async def _rien_journaliser(*args: Any, **kwargs: Any) -> None:
        return None

    # Lecture seule : aucune ligne dans le journal des recherches.
    module_recherche.record_search_event = _rien_journaliser

    if args.releve:
        sortie = asyncio.run(releve())
    else:
        # Synchrone et bloquant (sonde du fournisseur) : hors de la boucle.
        if not _sonder_le_fournisseur(args.vecteurs):
            raise SystemExit(
                "Fournisseur d'embeddings injoignable : la mesure n'aurait pas de volet semantique"
            )
        sortie = asyncio.run(evaluer(list(dict.fromkeys(args.etats)), args.questions_en, args.limite))
    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nDetail : {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
