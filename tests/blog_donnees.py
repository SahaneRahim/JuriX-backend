"""
Données et doublures communes aux tests du blog.

Les textes reprennent la FORME des vrais articles de jurix_dev (relevés du
29 et du 30/10/2026) : l'intitulé en tête (« Violation de domicile »), les
annotations du Code civil annoté dans le même morceau, l'article 1751 rangé en
morceaux « 1751.1 » et « 1751.2 », deux lignes « 1 » au Code pénal (la table
des matières, puis l'article premier). Et les formes de la revue du tour 1 :
« 35.2 » du Code du travail, qui commence par la FIN de l'article 39 collé dans
« 35.1 » ; « 67.2 », qui répète le début de l'article 67 avant sa suite ; la
première phrase de l'article 8 du Code de procédure pénale coupée par l'OCR ;
les articles 683 et 684 collés dans « 679.2 ». C'est sur ces formes que le
générateur se trompait.

Le modèle et la recherche sont doublés : aucun test du blog n'appelle le vrai
modèle, et la garde réseau de conftest reste en place. La doublure du modèle
répond aussi au CONTRÔLE IA (tour 3) : un verdict par phrase du prompt,
« soutenue » sauf ce que le test impose.
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import AsyncMock

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law import Article, Law
from app.schemas.search import ChunkResult
from app.services.controle_du_billet import PARTIES, SYSTEM_DU_CONTROLE
from app.services.search_service import CHEMIN_EXACT, VoletsDeLaRecherche

SUJET = "Expulsion du locataire sans décision de justice"

TITRE_CODE_PENAL = "Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal"
TITRE_CODE_DU_TRAVAIL = "Loi N°92/007 du 14 août 1992 portant Code du travail"
TITRE_CODE_CIVIL = "Code civil applicable au Cameroun, annoté de la jurisprudence de la Cour suprême"

CP_299 = (
    "Violation de domicile\n\n"
    "(1) Est puni d'un emprisonnement de dix (10) jours à un (01) an et d'une amende de "
    "cinq mille (5 000) à cinquante mille (50 000) francs ou de l'une de ces deux peines "
    "seulement, celui qui s'introduit ou se maintient dans le domicile d'autrui contre son gré."
)
CP_322 = (
    "1.-Filouteries de loyers\n\n"
    "(1) Est puni d'un emprisonnement de six (06) mois à trois (03) ans et d'une amende de "
    "cent mille (100 000) à trois cent mille (300 000) francs ou de l'une de ces deux peines "
    "seulement, le preneur par bail, dûment enregistré, d'un immeuble qui ne paie pas le loyer."
)
CP_1_CONTENU = "CONTENU\n\nLe Code Pénal comprend :\n\n- a) le Livre I, constitué des articles 1 à 101 ;"
CP_1 = "1.- Aucune exemption\n\nLa loi pénale s'impose à tous."
CC_1736 = (
    "Si le bail a été fait sans écrit, l'une des parties ne pourra donner congé à l'autre "
    "qu'en observant les délais fixés par l'usage des lieux.\n\n"
    "Jugements et arrêts : Article 1736 du Code civil, articles 3 et 37. Rejet. CS, Arr. n° 101."
)
CC_1751_1 = (
    "L'acquéreur à pacte de rachat ne peut user de la faculté d'expulser le preneur, jusqu'à "
    "ce que, par l'expiration du délai fixé pour le réméré, il devienne propriétaire "
    "incommutable.\nSECT. II Des règles particulières aux baux à loyer.\n"
    "1. Bail civil - Expulsion du locataire sans décision de justice : voie de fait."
)
# Les morceaux suivants de l'article 1751 sont des annotations, comme en base.
CC_1751_2 = (
    "2. Non paiement des loyers - offres réelles partielles - non expulsion du locataire. "
    "CS Arrêt n° 117/cc du 24 février 2005."
)
# Le morceau « 35.1 » du Code du travail porte, collés, les articles 38 et 39
# (en base : 36 à 39), sur une seule ligne.
CT_35_1 = (
    "(1) Pendant la durée du préavis, l'employeur et le travailleur sont tenus au respect de "
    "toutes les obligations réciproques qui leur incombent. Article 38.- Le contrat de travail "
    "à durée déterminée ne peut cesser avant terme qu'en cas de faute lourde, de force majeure "
    "ou d'accord des parties constaté par écrit. Article 39.- (1) Toute rupture abusive du "
    "contrat peut donner lieu à des dommages-intérêts."
)
# « 35.2 » (jurix_dev, id 97421) : les alinéas (6) et (7) de l'ARTICLE 39, collé
# à la fin de « 35.1 », puis l'article 40.
CT_35_2 = (
    "(6) Le salaire à prendre en considération aux alinéas précédents est le salaire moyen "
    "mensuel brut des douze (12) derniers mois d'activité du travailleur. (7) Ces "
    "dommages-intérêts ne se confondent ni avec l'indemnité pour non-observation du préavis, ni "
    "avec l'indemnité de licenciement. Article 40.- (1) Les dispositions de l'article 34 alinéa "
    "(1) ci-dessus ne s'appliquent pas en cas de licenciement pour motif économique."
)
# « 67.1 » et « 67.2 » (jurix_dev, ids 97434 et 97435) : le découpage répète devant
# « 67.2 » les 200 premiers caractères du chapeau de l'article 67, qui porte
# l'en-tête de l'article 68, puis passe à la suite de l'article 68.
CT_67_1 = (
    "En dehors des prestations prévues à l'article 66, alinéas (1) et (3), le salaire doit être "
    "payé en monnaie ayant cours légal, toute stipulation contraire est nulle. Article 68.- (1) "
    "A l'exception des professions pour lesquelles des usages établis prévoient une périodicité "
    "différente, le salaire doit être payé à intervalles réguliers ne pouvant excéder un mois."
)
CT_67_2 = (
    CT_67_1[:200].strip() + "\n(3) En cas de résiliation ou de rupture de contrat, le salaire et "
    "les indemnités doivent être payés dès la cessation de service."
)
CC_PREAMBULE = "Vu la Constitution ;\nVu le Code civil ;"
PROJET_5 = "Le bailleur peut expulser le locataire sans décision de justice."
# Code de procédure pénale, article 8 (jurix_dev, id 97488) : l'OCR a coupé la
# première phrase en deux lignes. Le tour 1 en retirait la première.
CPP_8 = (
    "(1) Toute personne suspectée d'avoir commis une infraction est présumée\n"
    "innocente jusqu'à ce que sa culpabilité ait été légalement établie au cours d'un procès "
    "où toutes les garanties nécessaires à sa défense lui seront assurées.\n"
    "(2) La présomption d'innocence s'applique au suspect, à l'inculpé, au prévenu et à l'accusé."
)
# « 679.2 » (jurix_dev, id 97890) : la suite d'un article, puis les articles 683 et
# 684 collés. La carte de l'article 683 débordait sur l'article 684.
# « 679.1 » porte, collés, les articles 680 à 682 (comme en base) : la tête de
# « 679.2 » est la suite de l'article 682, que suit l'article 683.
CPP_679_1 = (
    "(1) La réhabilitation ne peut être demandée qu'après un délai de cinq (5) ans. "
    "Article 680 — (1) La réhabilitation de plein droit est acquise au condamné qui n'a fait "
    "l'objet d'aucune condamnation nouvelle. Article 681 — Toute personne réhabilitée qui a fait "
    "l'objet d'une nouvelle condamnation n'est recevable à demander sa réhabilitation qu'à "
    "l'expiration d'un délai de quinze (15) ans. Article 682 — (1) Pour être réhabilité, le "
    "condamné doit justifier du paiement des frais de justice."
)
CPP_679_2 = (
    "(4) Si le condamné prétend que la partie civile a refusé de recevoir les sommes qui lui "
    "sont dues, il doit rapporter la preuve du refus. (5) La prescription quadriennale n'est "
    "pas applicable en cette matière. Article 683 — La Cour d'Appel du lieu de résidence du "
    "condamné est compétente en matière de réhabilitation. Article 684 — (1) Le condamné "
    "adresse la demande de réhabilitation au Procureur de la République du lieu de sa résidence."
)
TITRE_CPP = "Loi N°2005/007 du 27 juillet 2005 portant Code de procédure pénale"


@dataclass
class Textes:
    """Identifiants des textes et articles créés en base."""

    code_penal: int
    code_civil: int
    projet: int
    code_du_travail: int
    cp_299: int
    cp_322: int
    cp_1_contenu: int
    cp_1: int
    cc_1736: int
    cc_1751_1: int
    cc_1751_2: int
    cc_preambule: int
    projet_5: int
    ct_35_1: int
    ct_35_2: int
    ct_67_1: int
    ct_67_2: int
    code_de_procedure_penale: int
    cpp_8_1: int
    cpp_679_1: int
    cpp_679_2: int


async def creer_les_textes(db: AsyncSession) -> Textes:
    """Deux textes publiés et un texte en brouillon, avec leurs articles."""
    code_penal = Law(
        reference="PRC-4722", title=TITRE_CODE_PENAL, type="loi", content="Code pénal.",
        language="fr", status="published",
    )
    code_civil = Law(
        reference="CODE-CIVIL", title=TITRE_CODE_CIVIL, type="code", content="Code civil.",
        language="fr", status="published",
    )
    projet = Law(
        reference="LOI-2099-001", title="Loi N° 2099/001 portant statut des bailleurs",
        type="loi", content="Projet.", language="fr", status="draft",
    )
    code_du_travail = Law(
        reference="LOI-1992-007", title=TITRE_CODE_DU_TRAVAIL, type="loi",
        content="Code du travail.", language="fr", status="published",
    )
    code_de_procedure_penale = Law(
        reference="LOI-2005-007", title=TITRE_CPP, type="loi", content="Code de procédure pénale.",
        language="fr", status="published",
    )
    db.add_all([code_penal, code_civil, projet, code_du_travail, code_de_procedure_penale])
    await db.flush()

    def article(law: Law, numero: str, contenu: str, ordre: int, kind: str = "article") -> Article:
        return Article(law_id=law.id, number=numero, content=contenu, order=ordre, kind=kind)

    lignes = {
        "cp_1_contenu": article(code_penal, "1", CP_1_CONTENU, 1),
        "cp_1": article(code_penal, "1", CP_1, 2),
        "cp_299": article(code_penal, "299", CP_299, 300),
        "cp_322": article(code_penal, "322", CP_322, 323),
        "cc_preambule": article(code_civil, "PREAMBULE", CC_PREAMBULE, 1, kind="preambule"),
        "cc_1736": article(code_civil, "1736", CC_1736, 1736),
        "cc_1751_1": article(code_civil, "1751.1", CC_1751_1, 1751),
        "cc_1751_2": article(code_civil, "1751.2", CC_1751_2, 1752),
        "projet_5": article(projet, "5", PROJET_5, 5),
        "ct_35_1": article(code_du_travail, "35.1", CT_35_1, 25),
        "ct_35_2": article(code_du_travail, "35.2", CT_35_2, 26),
        "ct_67_1": article(code_du_travail, "67.1", CT_67_1, 39),
        "ct_67_2": article(code_du_travail, "67.2", CT_67_2, 40),
        "cpp_8_1": article(code_de_procedure_penale, "8.1", CPP_8, 8),
        "cpp_679_1": article(code_de_procedure_penale, "679.1", CPP_679_1, 408),
        "cpp_679_2": article(code_de_procedure_penale, "679.2", CPP_679_2, 409),
    }
    db.add_all(lignes.values())
    await db.commit()
    return Textes(
        code_penal=code_penal.id, code_civil=code_civil.id, projet=projet.id,
        code_du_travail=code_du_travail.id, code_de_procedure_penale=code_de_procedure_penale.id,
        **{nom: ligne.id for nom, ligne in lignes.items()},
    )


_LOIS = {
    "PRC-4722": TITRE_CODE_PENAL,
    "CODE-CIVIL": TITRE_CODE_CIVIL,
    "LOI-2099-001": "Loi N° 2099/001 portant statut des bailleurs",
    "LOI-1992-007": TITRE_CODE_DU_TRAVAIL,
    "LOI-2005-007": TITRE_CPP,
}


def extrait(
    law_id: int,
    article_id: Optional[int],
    numero: Optional[str],
    contenu: str,
    reference: str = "PRC-4722",
    status: str = "published",
) -> ChunkResult:
    """Un extrait tel que la recherche le rend."""
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        reference=reference, law_title=_LOIS.get(reference, reference), status=status,
        language="fr",
    )


def extraits_du_sujet(t: Textes) -> List[ChunkResult]:
    """Ce que la recherche rend pour SUJET, dans son ordre."""
    return [
        extrait(t.code_penal, t.cp_299, "299", CP_299),
        extrait(t.code_civil, t.cc_1736, "1736", CC_1736, reference="CODE-CIVIL"),
        extrait(t.code_civil, t.cc_1751_1, "1751.1", CC_1751_1, reference="CODE-CIVIL"),
        extrait(t.code_civil, t.cc_1751_2, "1751.2", CC_1751_2, reference="CODE-CIVIL"),
    ]


SIMILARITES_DU_SUJET = {"cp_299": 0.52, "cc_1736": 0.41, "cc_1751_1": 0.38, "cc_1751_2": 0.36}


def similarites(t: Textes, valeurs: Optional[Dict[str, float]] = None) -> Dict[int, float]:
    return {getattr(t, nom): v for nom, v in (valeurs or SIMILARITES_DU_SUJET).items()}


class RechercheDoublee:
    """
    La recherche, doublée : rend les extraits donnés et pose ses volets comme
    SearchService (VoletsDeLaRecherche), après chaque recherche.
    """

    def __init__(
        self,
        extraits: List[ChunkResult],
        similarites: Optional[Dict[int, float]] = None,
        chemin: str = CHEMIN_EXACT,
        erreur: Optional[BaseException] = None,
    ):
        self.extraits = extraits
        self.similarites = similarites or {}
        self.chemin = chemin
        self.erreur = erreur
        self.requetes: List[Any] = []
        self.tableaux_de_programmes_exclus = False
        self.tableaux_exclus_a_la_recherche: Optional[bool] = None
        self.volets: Optional[VoletsDeLaRecherche] = None

    async def search(self, requete):
        self.requetes.append(requete)
        self.tableaux_exclus_a_la_recherche = self.tableaux_de_programmes_exclus
        if self.erreur is not None:
            raise self.erreur
        self.volets = VoletsDeLaRecherche(chemin=self.chemin, similarites=self.similarites)
        return SimpleNamespace(chunks=list(self.extraits))


def article_du_modele(**champs: Any) -> Dict[str, Any]:
    """Un article tel que le modèle le rend, au format du schéma imposé."""
    article = {
        "extraits_suffisants": True,
        "title": "Expulsion sans juge : ce que dit la loi",
        "category": "Immobilier",
        "context": "Un locataire trouve ses affaires sur le trottoir à son retour.",
        "summary": (
            "S'introduire dans le domicile d'autrui contre son gré est puni par "
            "l'article 299 du Code Pénal."
        ),
        "key_points": [
            "Entrer chez quelqu'un contre son gré est une infraction.",
            "Sans bail écrit, le congé suit les délais de l'usage des lieux.",
            "Le bail se termine selon les règles du Code civil.",
        ],
        "citations": [
            {"extrait": 1, "objet": "La violation de domicile est punie."},
            {"extrait": 2, "objet": "Le congé d'un bail verbal suit l'usage des lieux."},
        ],
        "citizen_impact": "Vous pouvez porter plainte si l'on entre chez vous contre votre gré.",
        "suggested_question": "Que faire si mon bailleur change les serrures sans jugement ?",
        "tags": ["bail", "expulsion", "domicile"],
    }
    article.update(champs)
    return article


# Une phrase à juger dans le prompt du contrôle : « P3 : Le salarié doit… ».
_PHRASE_DU_PROMPT = re.compile(r"^(P\d+) : (.*)$", re.MULTILINE)
# Un extrait du prompt du contrôle : son en-tête « [2] … — Article 39 », puis
# son texte, jusqu'à l'extrait suivant ou aux phrases.
_EXTRAIT_DU_PROMPT = re.compile(
    r"^\[(\d+)\] [^\n]*\n\n(.*?)(?=\n\n---\n\n\[\d+\] |\n\nPHRASES À JUGER)",
    re.MULTILINE | re.DOTALL,
)

# Ce qu'un test impose au contrôle : rien (tout « soutenue »), des verdicts
# par identifiant (« P3 ») ou par morceau de phrase, une réponse brute
# ({"response": …}), ou une exception à lever.
Controle = Union[None, Dict[str, Any], BaseException, Callable[[str], Dict[str, Any]]]


def phrases_du_prompt(prompt: str) -> List[tuple]:
    """Les (identifiant, phrase) du prompt du contrôle, dans l'ordre."""
    return _PHRASE_DU_PROMPT.findall(prompt)


def extraits_du_prompt(prompt: str) -> Dict[int, str]:
    """{numéro: texte} des extraits du prompt du contrôle."""
    return {int(n): texte for n, texte in _EXTRAIT_DU_PROMPT.findall(prompt)}


def passage_de(texte: str, mots: int = 8) -> str:
    """Les premiers mots d'un extrait : un passage que le contrôle peut « copier »."""
    return " ".join(texte.split()[:mots])


def jugement(
    identifiant: str, verdict: str, passage: str = "", extraits: Optional[List[int]] = None,
    nature: str = "regle", raison: Optional[str] = None,
) -> Dict[str, Any]:
    """Un verdict au format du schéma du contrôle (tour 4)."""
    return {
        "id": identifiant, "affirmation": f"Ce qu'affirme {identifiant}.", "nature": nature,
        "extraits": extraits if extraits is not None else [], "passage": passage,
        "raison": raison if raison is not None else f"Verdict de la doublure : {verdict}.",
        "verdict": verdict,
    }


def reponse_du_controle(prompt: str, verdicts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    La réponse du contrôle doublé : chaque phrase du prompt « soutenue » par
    l'extrait 1 (son début, copié comme passage), sauf celles que `verdicts`
    désigne, par identifiant ou par un morceau de leur texte.
    """
    passage = passage_de(extraits_du_prompt(prompt).get(1, ""))
    jugees = []
    for identifiant, phrase in phrases_du_prompt(prompt):
        verdict = "soutenue"
        for cle, impose in (verdicts or {}).items():
            if cle == identifiant or (not re.fullmatch(r"P\d+", cle) and cle in phrase):
                verdict = impose
        appuye = verdict in ("soutenue", "contredite")
        jugees.append(jugement(
            identifiant, verdict, passage if appuye else "", [1] if appuye else [],
        ))
    return {"response": json.dumps({"phrases": jugees}), "fin": "STOP"}


def modele_double(
    article: Optional[Dict[str, Any]] = None, controle: Controle = None, **reponse: Any
) -> AsyncMock:
    """
    Le modèle, doublé. À la génération : l'article en JSON, ou ce que le test
    impose (`reponse`, `side_effect`). Au contrôle (prompt système du
    contrôle) : un verdict par phrase (`controle`, voir reponse_du_controle).
    """
    double = AsyncMock()
    double.model_name = "modele-double"
    if "side_effect" in reponse:
        double.generate = AsyncMock(side_effect=reponse.pop("side_effect"))
        return double
    generation = {
        "response": json.dumps(article if article is not None else article_du_modele()),
        "fin": "STOP",
        **reponse,
    }

    async def repondre(**kwargs: Any) -> Dict[str, Any]:
        if kwargs.get("system") != SYSTEM_DU_CONTROLE:
            return generation
        if isinstance(controle, BaseException):
            raise controle
        if callable(controle):
            return controle(kwargs["prompt"])
        if isinstance(controle, dict) and "response" in controle:
            return controle
        return reponse_du_controle(kwargs["prompt"], controle)

    double.generate = AsyncMock(side_effect=repondre)
    return double


def appels_du_controle(double: AsyncMock) -> List[Dict[str, Any]]:
    """Les arguments des appels de contrôle reçus par la doublure."""
    return [
        appel.kwargs for appel in double.generate.await_args_list
        if appel.kwargs.get("system") == SYSTEM_DU_CONTROLE
    ]


def a_confirmer(article: Dict[str, Any]) -> List[str]:
    """
    Les `id` des phrases à confirmer du contrôle que montre `article` (une réponse de
    l'administration) : de nature « regle », jugées non_soutenue ou hors_sources (tour 7).
    """
    return [
        p["id"] for p in (article.get("controle_ia") or {}).get("phrases", [])
        if p.get("nature") == "regle" and p.get("verdict") in ("non_soutenue", "hors_sources")
    ]


def relecture_cochee(
    article: Dict[str, Any], parties: Optional[Dict[str, bool]] = None, *, confirmer: bool = True,
) -> Dict[str, Any]:
    """
    Les cases d'un PUT, cochées contre le contrôle que montre `article` (une
    réponse de l'API), comme les envoie le front depuis l'avenant du tour 5 :
    {"parties": …, "controle": son controle_ia.fait_le}. Toutes cochées par
    défaut. `confirmer` (avenant du tour 7) : et toutes ses phrases à confirmer
    confirmées (`phrases_confirmees`), s'il en a.
    """
    relecture = {
        "parties": dict(parties) if parties is not None else {partie: True for partie in PARTIES},
        "controle": (article.get("controle_ia") or {}).get("fait_le"),
    }
    if confirmer and a_confirmer(article):
        relecture["phrases_confirmees"] = a_confirmer(article)
    return relecture


# ==================== LIGNES RÉELLES DE LA REVUE DU TOUR 2 ====================

_LIGNES_REELLES = Path(__file__).parent / "fixtures" / "blog" / "lignes_reelles.json"


def lignes_reelles() -> Dict[str, Any]:
    """
    Des lignes de jurix_dev recopiées telles quelles (lues en lecture seule) :
    - 56288-56290 : décret 2021/597, dont l'article 1 CITE la nouvelle
      rédaction de l'article 2 d'un autre décret (« «ARTICLE 2.- (nouveau) ») ;
    - 58733-58735 : loi 2020/009, « «ARTICLE 5.- (4) (nouveau) » ;
    - 97434-97436 : Code du travail 67.1 à 67.3, l'en-tête de l'article 73
      abîmé par l'OCR à la fin de 67.2 ;
    - 97293-97295 : loi sur la cybersécurité 52.1 à 52.3, l'en-tête de
      l'article 57 abîmé à la fin de 52.2 ;
    - 97475-97476 : Code du travail 161.1 et 161.2 (même forme, rattachement
      juste, perdu aussi par la règle du doute).
    """
    return json.loads(_LIGNES_REELLES.read_text(encoding="utf-8"))


async def creer_les_lignes_reelles(db: AsyncSession, ids: List[int]) -> Dict[int, int]:
    """
    Crée les textes et les lignes réelles demandés ; rend {id dans jurix_dev:
    id dans la base de test}, et {-law_id de jurix_dev: id du texte de test}.
    """
    donnees = lignes_reelles()
    voulues = [x for x in donnees["lignes"] if x["id"] in ids]
    lois: Dict[int, Law] = {}
    for law_id in sorted({x["law_id"] for x in voulues}):
        lois[law_id] = Law(
            reference=f"REEL-{law_id}", title=donnees["textes"][str(law_id)], type="loi",
            content=".", language="fr", status="published",
        )
    db.add_all(lois.values())
    await db.flush()
    lignes = {
        x["id"]: Article(
            law_id=lois[x["law_id"]].id, number=x["number"], content=x["content"],
            order=x["order"], kind=x["kind"], title=x["title"],
        )
        for x in voulues
    }
    db.add_all(lignes.values())
    await db.commit()
    return {
        **{i: ligne.id for i, ligne in lignes.items()},
        **{-law_id: loi.id for law_id, loi in lois.items()},
    }


# ==================== ARTICLES D'AVANT LA VÉRIFICATION ====================

# Les citations de la ligne 1 de jurix_dev (26/10/2026, publiée) : law_id 1 du
# gabarit de l'ancien prompt, un nom et un « extrait » écrits par le modèle,
# pas de `verifie`. L'article 299 du Code pénal y est donné à « trois mois à
# trois ans » ; le texte en base dit « dix (10) jours à un (01) an ».
CITATIONS_D_AVANT = [
    {
        "title": "Obligations principales du bailleur", "law_id": 1,
        "excerpt": "Le bailleur est obligé, par la nature du contrat, de délivrer au preneur la "
                   "chose louée.",
        "reference": "Code Civil / Droit des obligations", "article_number": "1719",
    },
    {
        "title": "Violation de domicile et entrave à la jouissance", "law_id": 1,
        "excerpt": "Est puni d'un emprisonnement de trois mois à trois ans et d'une amende "
                   "quiconque s'introduit ou se maintient dans le domicile d'autrui.",
        "reference": "Loi portant Code Pénal camerounais", "article_number": "299",
    },
]


async def article_d_avant(
    db: AsyncSession, slug: str = "expulsion-locative-loi-cameroun",
    citations: Optional[List[Dict[str, Any]]] = None, publie: bool = True,
) -> int:
    """
    Un article enregistré AVANT le contrat du tour 1, comme les trois lignes du
    26/10 de jurix_dev : publié, citations jamais vérifiées. Écrit en SQL, comme
    elles l'ont été : aucune route d'aujourd'hui ne peut en créer.
    """
    ligne = (await db.execute(text(
        "INSERT INTO blog_posts (slug, title, category, context, summary, key_points, citations, "
        "citizen_impact, suggested_question, tags, is_published) VALUES (:slug, :title, "
        "'Immobilier & Logement', :context, :summary, CAST(:points AS jsonb), "
        "CAST(:citations AS jsonb), :impact, :question, CAST('[\"bail\"]' AS jsonb), :publie) "
        "RETURNING id"
    ), {
        "slug": slug,
        "title": "Litige Bailleur-Locataire : votre propriétaire a-t-il le droit de couper l'eau ?",
        "context": "Des propriétaires coupent l'eau pour forcer des locataires à partir.",
        "summary": "La justice privée est interdite au bailleur.",
        "points": json.dumps(["Le bailleur doit assurer la jouissance paisible."]),
        "citations": json.dumps(CITATIONS_D_AVANT if citations is None else citations),
        "impact": "Vous pouvez porter plainte pour voie de fait.",
        "question": "Mon bailleur peut-il couper le courant ?",
        "publie": publie,
    })).scalar_one()
    await db.commit()
    return ligne


# ==================== LE GRAND MODÈLE AU RYTHME MESURÉ (REVUE DU TOUR 6) ====================

# La mesure du 04/11 (data/blog/tour5-modeles/large4-low.log du dépôt du backend),
# mistral-large-4, effort « low » : par phrase jugée, les jetons de sortie (réflexion
# comprise) et les secondes ; l'entrée au rapport mesuré de 3,97 caractères par jeton
# (bail : 16 045 caractères de système et de prompt pour 4 046 jetons).
RYTHMES = {"travail": (381, 51.2 / 21), "bail": (466, 78.7 / 23)}
# TOUR 7 : la sortie et la durée d'UN appel, quelle que soit sa taille (la réflexion du
# modèle domine) ; mesurées le 04/11 : « pire », les 23 phrases du bail en un appel
# (large4-low.log), remplacées par l'appel coupé de l'essai réel du 06/11, 12 000 jetons
# en 105 s ; « plafond », ce que le plan prévoit au pire ; « lot-de-12 », un lot de 12 phrases de l'essai réel du tour 6 ;
# « vite », le plus court de ses appels (11 phrases).
APPELS_MESURES = {"pire": (12_000, 105.0), "plafond": (15_500, 140.0), "lot-de-12": (10_462, 76.0), "vite": (5_951, 40.5)}
CARACTERES_PAR_JETON_MESURES = 3.97


class GrandModeleAuRythme:
    """
    mistral-large-4 au niveau HTTP (pour httpx.MockTransport), au coût et au rythme
    MESURÉS le 04/11 : l'entrée au rapport mesuré, la sortie et la durée par phrase
    jugée du prompt (`rythme` « travail » ou « bail »), ou par appel quelle que soit
    sa taille (« pire », « plafond », « lot-de-12 », « vite » : APPELS_MESURES, tour 7) ; le contenu
    en morceaux (réflexion, puis le JSON des verdicts, tout « soutenue »). Le temps est
    celui d'une horloge factice (`horloge.t`), que chaque appel avance de sa durée.

    `restants` : l'en-tête x-ratelimit-remaining-tokens-minute de chaque réponse (le
    04/11 : 0 après le contrôle du travail). `pas_s` : la durée s'écoule par pas, en
    rendant la main entre deux, pour qu'un contrôle lancé pendant un autre parte à son
    heure. `pause_a` : l'instant où l'appel en cours s'arrête (`en_pause`) jusqu'à
    `reprendre` : un test y lance un autre contrôle, l'horloge figée.
    """

    def __init__(self, horloge: Any, rythme: str = "travail", *, restants: Optional[int] = None,
                 pas_s: Optional[float] = None, pause_a: Optional[float] = None):
        self.horloge = horloge
        self.par_appel = APPELS_MESURES.get(rythme)
        if self.par_appel is None:
            self.sortie_par_phrase, self.secondes_par_phrase = RYTHMES[rythme]
        self.restants = restants
        self.pas_s = pas_s
        self.pause_a = pause_a
        self.en_pause = asyncio.Event()
        self.reprendre = asyncio.Event()
        self.requetes: List[Dict[str, Any]] = []

    async def __call__(self, requete: httpx.Request) -> httpx.Response:
        corps = json.loads(requete.content)
        systeme, prompt = corps["messages"][0]["content"], corps["messages"][-1]["content"]
        n = len(phrases_du_prompt(prompt))
        entree = math.ceil((len(systeme) + len(prompt)) / CARACTERES_PAR_JETON_MESURES)
        if self.par_appel is not None:
            sortie, duree = self.par_appel
        else:
            sortie, duree = self.sortie_par_phrase * n, self.secondes_par_phrase * n
        appel = {
            "depart": round(self.horloge.t, 1), "phrases": n, "entree": entree,
            "sortie": sortie, "modele": corps["model"],
            "un_seul_appel": "autres phrases de l'article sont jugées à part" not in prompt,
        }
        self.requetes.append(appel)
        fin = self.horloge.t + duree
        while self.horloge.t < fin - 1e-9:
            if self.pause_a is not None and self.horloge.t >= self.pause_a and not self.en_pause.is_set():
                self.en_pause.set()
                await self.reprendre.wait()
            pas = self.pas_s if self.pas_s else fin - self.horloge.t
            self.horloge.t = min(fin, self.horloge.t + pas)
            if self.pas_s:
                await asyncio.sleep(0)
        appel["fin"] = round(self.horloge.t, 1)
        entetes = {"x-ratelimit-limit-tokens-minute": "20000"}
        if self.restants is not None:
            entetes["x-ratelimit-remaining-tokens-minute"] = str(self.restants)
        contenu = [
            {"type": "thinking", "thinking": [{"type": "text", "text": "Réflexion."}], "closed": True},
            {"type": "text", "text": reponse_du_controle(prompt)["response"]},
        ]
        return httpx.Response(200, headers=entetes, json={
            "id": "double", "object": "chat.completion", "model": corps["model"],
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": contenu, "tool_calls": None},
            }],
            "usage": {
                "prompt_tokens": entree, "completion_tokens": appel["sortie"],
                "total_tokens": entree + appel["sortie"],
            },
        })


# L'article RÉEL du travail, généré le 04/11 par la vraie chaîne (essai du tour 4) :
# 22 phrases, lots 11 + 11, 10 052 caractères d'extraits ; et les sources réelles du
# bail, pour porter ses extraits plus haut (fixtures/blog/article_reel_travail_tour4.json).
_ARTICLE_REEL = Path(__file__).parent / "fixtures" / "blog" / "article_reel_travail_tour4.json"
# Trois phrases de plus au résumé : 25, le plafond, en lots 13 + 12 (revue du tour 6).
TROIS_PHRASES_DE_PLUS = (
    " Le juge apprécie le motif. Le salarié peut saisir l'inspecteur du travail. "
    "La procédure est gratuite."
)


def article_reel(caracteres: Optional[int] = None, *, au_plafond: bool = False) -> SimpleNamespace:
    """
    L'article réel du travail, au format que le contrôle lit. `caracteres` : ses
    sources complétées par d'autres extraits réels (ceux du bail, puis les siens),
    renumérotés à la suite, jusqu'à ce total (le générateur en donne 24 000 au plus).
    `au_plafond` : trois phrases de plus au résumé, PHRASES_AU_PLUS en tout. Ses
    citations ne sont pas dans la base de test : le contrôle n'ajoute aucun extrait.
    """
    donnees = json.loads(_ARTICLE_REEL.read_text(encoding="utf-8"))
    reel = donnees["travail"]
    sources = [dict(s) for s in reel["sources"]]
    reserve = [dict(s) for s in donnees["sources_du_bail"]] + [dict(s) for s in reel["sources"]]
    while caracteres and sum(len(s["extrait"]) for s in sources) < caracteres:
        ajout = dict(reserve[(len(sources) - len(reel["sources"])) % len(reserve)])
        ajout["n"] = len(sources) + 1
        sources.append(ajout)
    return SimpleNamespace(
        title=reel["title"], context=reel["context"],
        summary=reel["summary"] + (TROIS_PHRASES_DE_PLUS if au_plafond else ""),
        key_points=list(reel["key_points"]), citizen_impact=reel["citizen_impact"],
        citations=[dict(c) for c in reel["citations"]],
        suggested_question=reel["suggested_question"], sources=sources,
    )
