"""
Classement d'un texte juridique dans les 14 domaines canoniques, par Groq.

Ce module classifie les documents juridiques camerounais en s'appuyant sur un
LLM (GROQ_MODEL_CLASSEMENT, via Groq) avec sortie JSON structuree stricte.

TROIS CHOIX, ET LEUR RAISON :

- PAR LOTS. Le palier gratuit de Groq compte les requetes (1 000 par jour) et
  les jetons (8 000 par minute, 200 000 par jour). Une loi par requete, le
  corpus de 2 226 lois demandait 11 jours. Un lot de documents numerotes
  partage les consignes et rend un verdict par numero, sous un code court
  plutot que le nom complet du domaine (un jeton de sortie au lieu de 10 a 15).

- SANS REPLI SILENCIEUX. La version precedente rangeait en Droit
  Administratif, confiance 0,20, toute loi dont la reponse etait vide ou
  illisible — en pratique toute loi classee pendant un 429, soit 89 sur 122
  lors d'un essai de reclassement. Rien ne distinguait ensuite ces lois des
  vrais textes administratifs. Desormais, un classement impossible leve
  ClassementIndisponible ; le pipeline laisse la categorie NULL et le dit.

- UN EXTRAIT, PAS LE DEBUT DU TEXTE. Les 1 500 premiers caracteres d'un
  decret sont ses visas (« Vu la Constitution ; Vu la loi n°... ») : ils
  citent d'autres textes et ne disent rien de l'objet. L'extrait envoye part
  de l'article premier.

Author: JuriX Team
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)


# ==================== LES 14 DOMAINES CANONIQUES ====================

CANONICAL_DOMAINS: Tuple[str, ...] = (
    "Droit Constitutionnel",
    "Droit Administratif",
    "Fonction Publique",
    "Droit International",
    "Finances Publiques et Fiscalité",
    "Droit Pénal et Procédure Pénale",
    "Droit Civil et Procédure Civile",
    "Droit des Personnes, de la Famille et État Civil",
    "Droit du Travail et Sécurité Sociale",
    "Droit des Affaires, Banque et OHADA",
    "Droit Foncier et Domanial",
    "Droit de l'Environnement et des Ressources Naturelles",
    "Santé Publique et Sécurité Sanitaire",
    "Éducation, Recherche, Culture et Médias",
)

CONSTITUTIONNEL = CANONICAL_DOMAINS[0]
ADMINISTRATIF = CANONICAL_DOMAINS[1]
FONCTION_PUBLIQUE = CANONICAL_DOMAINS[2]
INTERNATIONAL = CANONICAL_DOMAINS[3]
FINANCES = CANONICAL_DOMAINS[4]
PENAL = CANONICAL_DOMAINS[5]
CIVIL = CANONICAL_DOMAINS[6]
FAMILLE = CANONICAL_DOMAINS[7]
TRAVAIL = CANONICAL_DOMAINS[8]
AFFAIRES = CANONICAL_DOMAINS[9]
FONCIER = CANONICAL_DOMAINS[10]
ENVIRONNEMENT = CANONICAL_DOMAINS[11]
SANTE = CANONICAL_DOMAINS[12]
EDUCATION = CANONICAL_DOMAINS[13]

# Ce que chaque domaine couvre : les descriptions de la migration 013, mot
# pour mot (un test le verifie). Le modele classe sur la definition meme que
# l'interface affiche.
DESCRIPTIONS_DOMAINES: Dict[str, str] = {
    CONSTITUTIONNEL: "Constitution, révisions, élections, mandats et institutions publiques",
    ADMINISTRATIF: "Organisation et fonctionnement des services publics et collectivités",
    FONCTION_PUBLIQUE: "Statut général des agents publics, carrières, nominations et distinctions",
    INTERNATIONAL: "Traités, conventions et accords bilatéraux ou multilatéraux ratifiés",
    FINANCES: "Budget de l'État, fiscalité, douanes, emprunts publics et accords de prêts",
    PENAL: "Infractions, peines, poursuites judiciaires et justice militaire",
    CIVIL: "Personnes, obligations, contrats, instances civiles et voies d'exécution",
    FAMILLE: "Mariage, filiation, successions, actes d'état civil, CNI et nationalité",
    TRAVAIL: "Relations de travail, conventions collectives, syndicats et prévoyance sociale",
    AFFAIRES: "Sociétés, commerce, secret bancaire, marchés publics et droit OHADA",
    FONCIER: "Domaine de l'État, titres fonciers, expropriations et cadastre",
    ENVIRONNEMENT: "Environnement, mines, forêts, eau, hydrocarbures et biodiversité",
    SANTE: "Santé publique, médecine, pharmacie, sécurité sanitaire et sûreté radiologique",
    EDUCATION: "Enseignement, universités, recherche, culture, patrimoine, médias et langues officielles",
}

# Le code que le modele ecrit pour chaque domaine.
CODES_DOMAINES: Dict[str, str] = {
    "CONST": CONSTITUTIONNEL,
    "ADMIN": ADMINISTRATIF,
    "FP": FONCTION_PUBLIQUE,
    "INTL": INTERNATIONAL,
    "FIN": FINANCES,
    "PENAL": PENAL,
    "CIVIL": CIVIL,
    "FAMILLE": FAMILLE,
    "TRAVAIL": TRAVAIL,
    "AFFAIRES": AFFAIRES,
    "FONCIER": FONCIER,
    "ENV": ENVIRONNEMENT,
    "SANTE": SANTE,
    "EDUC": EDUCATION,
}

# Les cas ou deux domaines se disputent un texte, tranches une fois pour toutes.
# Chaque regle vient d'une erreur observee sur le corpus.
REGLES_D_ARBITRAGE: Tuple[str, ...] = (
    "Accord de prêt, de crédit ou de financement, garantie ou aval de l'État : FIN, "
    "même quand c'est un accord international.",
    "Nomination, promotion, décoration, intégration, mise à la retraite, "
    "sanction d'agents publics : FP.",
    "Code minier, forestier, pétrolier, gazier ; titre ou permis minier ; "
    "exploitation des ressources naturelles : ENV.",
    "Organisation d'un ministère, d'un établissement public ou d'un service : "
    "le domaine de son secteur s'il en a un (hôpital : SANTE, université : EDUC, "
    "caisse de sécurité sociale : TRAVAIL), sinon ADMIN.",
    "Ratification d'un traité ou d'une convention : INTL, sauf accord de prêt.",
    "Les visas (« Vu la Constitution… ») ne comptent pas : seul l'objet du texte décide.",
)

# Inscrite avec chaque verdict du reclassement : un verdict obtenu par une
# autre version des consignes se reconnait, et se refait.
VERSION_DES_CONSIGNES = "2026-10-07"

# Taille d'extrait envoyee par document.
LONGUEUR_EXTRAIT = 500

# Jetons de sortie : un verdict en coute une trentaine ; la marge couvre la
# reflexion que gpt-oss ne permet pas de couper tout a fait.
_JETONS_PAR_VERDICT = 45
_JETONS_DE_MARGE = 400


class ClassementIndisponible(Exception):
    """
    Le classement n'a pas pu se faire : quota, panne, ou reponse inexploitable.

    `retry_after` : secondes avant qu'un nouvel essai ait une chance (None si
    inconnu). `quota` : le quota du modele est epuise, inutile de reessayer
    avant l'echeance.
    """

    def __init__(self, raison: str, retry_after: Optional[float] = None, quota: bool = False):
        super().__init__(raison)
        self.raison = raison
        self.retry_after = retry_after
        self.quota = quota


@dataclass(frozen=True)
class DomainResult:
    """
    Verdict de classement.

    `domain` est TOUJOURS l'un des CANONICAL_DOMAINS.
    `rule` nomme le modele qui a decide.
    """

    domain: str
    confidence: float
    rule: str
    source: Literal["groq"]
    runners_up: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True)
class DocumentAClasser:
    """Un document soumis au classement : son titre d'abord, un extrait ensuite."""

    titre: str
    extrait: str = ""
    type_acte: Optional[str] = None


@dataclass
class ResultatLot:
    """
    Un verdict par document, dans l'ordre du lot. None : document « a revoir »,
    que le modele n'a pas classe malgre les nouveaux essais.
    """

    verdicts: List[Optional[DomainResult]]
    jetons: int
    requetes: int
    modele: str


# ==================== EXTRAIT ====================

# L'article premier, en debut de ligne : un visa qui le cite (« Vu la loi
# n°..., notamment son article 1er ») ne commence pas une ligne par « Article ».
_ARTICLE_PREMIER = re.compile(
    r"^\s*Art(?:icle)?\.?\s*(?:1\s*(?:er)?|premier|unique)\b", re.IGNORECASE | re.MULTILINE
)
_LIGNE_DE_PREAMBULE = re.compile(
    r"^\s*(?:Vu|Sur (?:proposition|le rapport)|Apr[eè]s avis)\b", re.IGNORECASE
)


def extrait_pour_classement(texte: Optional[str], longueur: int = LONGUEUR_EXTRAIT) -> str:
    """
    Les `longueur` caracteres qui disent l'objet du texte.

    A partir de l'article premier s'il se trouve ; a defaut, le texte prive de
    ses visas. Les espaces sont replies : ils couteraient des jetons.
    """
    texte = texte or ""
    debut = _ARTICLE_PREMIER.search(texte)
    if debut:
        extrait = texte[debut.start(): debut.start() + longueur * 2]
    else:
        extrait = "\n".join(
            ligne for ligne in texte.splitlines() if not _LIGNE_DE_PREAMBULE.match(ligne)
        )
    return " ".join(extrait.split())[:longueur]


# ==================== CLASSIFIEUR ====================


class LegalDomainClassifier:
    """Classe des lots de documents par Groq. Sans etat : un service partage."""

    def __init__(self, groq: Optional[Any] = None, modele: Optional[str] = None):
        self._groq = groq
        self._modele = modele

    @property
    def modele(self) -> str:
        return self._modele or settings.GROQ_MODEL_CLASSEMENT

    def _service(self):
        if self._groq is not None:
            return self._groq
        from app.services.groq_service import get_groq_service

        return get_groq_service()

    # ---------------------------------------------------------- consignes

    @staticmethod
    def consignes() -> str:
        domaines = "\n".join(
            f"{code} — {nom} : {DESCRIPTIONS_DOMAINES[nom]}"
            for code, nom in CODES_DOMAINES.items()
        )
        regles = "\n".join(f"- {regle}" for regle in REGLES_D_ARBITRAGE)
        return (
            "Tu classes des textes juridiques camerounais (lois, ordonnances, "
            "décrets, arrêtés) dans UN domaine parmi 14, d'après leur OBJET.\n\n"
            f"Domaines (code — nom : contenu) :\n{domaines}\n\n"
            f"Arbitrages :\n{regles}\n\n"
            "Pour chaque document numéroté, rends : n (son numéro), domaine (un "
            "code), secondaires (0 à 2 autres codes vraiment pertinents, sinon "
            "[]) et confiance (entre 0 et 1). Un verdict par document, aucun oubli."
        )

    @staticmethod
    def message(documents: Sequence[DocumentAClasser]) -> str:
        lignes = []
        for n, document in enumerate(documents, start=1):
            type_acte = f"[{document.type_acte}] " if document.type_acte else ""
            lignes.append(f"{n}. {type_acte}{' '.join((document.titre or '').split())}")
            if document.extrait:
                lignes.append(f"   Extrait : {document.extrait}")
        return "\n".join(lignes)

    @staticmethod
    def schema() -> Dict[str, Any]:
        """Schema strict : enum des codes, tous les champs requis, rien d'autre."""
        codes = list(CODES_DOMAINES)
        verdict = {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "domaine": {"type": "string", "enum": codes},
                "secondaires": {"type": "array", "items": {"type": "string", "enum": codes}},
                "confiance": {"type": "number"},
            },
            "required": ["n", "domaine", "secondaires", "confiance"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"verdicts": {"type": "array", "items": verdict}},
            "required": ["verdicts"],
            "additionalProperties": False,
        }

    def _lire(self, donnees: Dict[str, Any], taille: int) -> Dict[int, DomainResult]:
        """
        Les verdicts exploitables, par numero. Un numero hors du lot, en double
        ou sans code connu est ignore : son document sera redemande.
        """
        verdicts: Dict[int, DomainResult] = {}
        for brut in donnees.get("verdicts") or []:
            if not isinstance(brut, dict):
                continue
            n = brut.get("n")
            domaine = CODES_DOMAINES.get(str(brut.get("domaine", "")).upper())
            if not isinstance(n, int) or not 1 <= n <= taille or n in verdicts or not domaine:
                continue
            try:
                confiance = float(brut.get("confiance", 0.5))
            except (TypeError, ValueError):
                confiance = 0.5
            confiance = round(max(0.0, min(1.0, confiance)), 3)
            secondaires: List[str] = []
            for code in brut.get("secondaires") or []:
                nom = CODES_DOMAINES.get(str(code).upper())
                if nom and nom != domaine and nom not in secondaires:
                    secondaires.append(nom)
            verdicts[n] = DomainResult(
                domain=domaine,
                confidence=confiance,
                rule=f"groq:{self.modele}",
                source="groq",
                runners_up=tuple((nom, round(confiance * 0.8, 3)) for nom in secondaires[:2]),
            )
        return verdicts

    # ----------------------------------------------------------- appels

    def _demande(self, documents: Sequence[DocumentAClasser]) -> Dict[str, Any]:
        return {
            "systeme": self.consignes(),
            "message": self.message(documents),
            "schema": self.schema(),
            "nom_schema": "classement_domaines",
            "max_jetons": _JETONS_DE_MARGE + _JETONS_PAR_VERDICT * len(documents),
            "modele": self.modele,
        }

    @staticmethod
    def _indisponible(erreur: Exception) -> ClassementIndisponible:
        from app.services.groq_service import GroqQuotaError

        return ClassementIndisponible(
            str(erreur),
            retry_after=getattr(erreur, "retry_after", None),
            quota=isinstance(erreur, GroqQuotaError),
        )

    def _suite(self, documents, verdicts, resultat, demander) -> List[List[int]]:
        """
        Range les verdicts obtenus, et rend les sous-lots a redemander : les
        documents manquants ensemble, ou le lot coupe en deux si RIEN n'est
        revenu. Un document seul qui manque encore est « a revoir ».
        """
        manquants = [i for i in range(len(documents)) if i + 1 not in verdicts]
        for i in range(len(documents)):
            if i + 1 in verdicts:
                resultat[demander[i]] = verdicts[i + 1]
        if not manquants or len(documents) == 1:
            return []
        if len(manquants) == len(documents):
            milieu = len(documents) // 2
            return [list(range(milieu)), list(range(milieu, len(documents)))]
        return [manquants]

    def classer_lot(
        self, documents: Sequence[DocumentAClasser], *, attente_max: Optional[float] = None
    ) -> ResultatLot:
        """
        Classe un lot (synchrone : pipeline et scripts).

        Leve ClassementIndisponible si Groq ne peut pas repondre (quota, panne,
        cle). Une reponse incomplete ou illisible n'en est pas une : les
        documents manquants sont redemandes.
        """
        from app.services.groq_service import GroqReponseInvalideError, GroqServiceError

        resultat: List[Optional[DomainResult]] = [None] * len(documents)
        jetons = requetes = 0
        a_traiter = [list(range(len(documents)))] if documents else []
        while a_traiter:
            indices = a_traiter.pop(0)
            lot = [documents[i] for i in indices]
            requetes += 1
            try:
                reponse = self._service().completer_json_sync(
                    **self._demande(lot), attente_max=attente_max
                )
                jetons += reponse.jetons
                verdicts = self._lire(reponse.donnees, len(lot))
            except GroqReponseInvalideError as e:
                logger.warning("🧭 Lot de %d illisible (%s) : redemande", len(lot), e)
                verdicts = {}
            except GroqServiceError as e:
                raise self._indisponible(e) from e
            for sous_lot in self._suite(lot, verdicts, resultat, indices):
                a_traiter.append([indices[i] for i in sous_lot])
        return ResultatLot(resultat, jetons, requetes, self.modele)

    async def classer_lot_async(
        self, documents: Sequence[DocumentAClasser], *, attente_max: Optional[float] = None
    ) -> ResultatLot:
        """Version asynchrone de `classer_lot`, pour l'API."""
        from app.services.groq_service import GroqReponseInvalideError, GroqServiceError

        resultat: List[Optional[DomainResult]] = [None] * len(documents)
        jetons = requetes = 0
        a_traiter = [list(range(len(documents)))] if documents else []
        while a_traiter:
            indices = a_traiter.pop(0)
            lot = [documents[i] for i in indices]
            requetes += 1
            try:
                reponse = await self._service().completer_json(
                    **self._demande(lot), attente_max=attente_max
                )
                jetons += reponse.jetons
                verdicts = self._lire(reponse.donnees, len(lot))
            except GroqReponseInvalideError as e:
                logger.warning("🧭 Lot de %d illisible (%s) : redemande", len(lot), e)
                verdicts = {}
            except GroqServiceError as e:
                raise self._indisponible(e) from e
            for sous_lot in self._suite(lot, verdicts, resultat, indices):
                a_traiter.append([indices[i] for i in sous_lot])
        return ResultatLot(resultat, jetons, requetes, self.modele)

    # ------------------------------------------------- un seul document

    @staticmethod
    def _document(title: str, content: str, doc_type: Optional[str]) -> DocumentAClasser:
        titre = (title or "").strip()
        extrait = extrait_pour_classement(content)
        if not titre and not extrait:
            raise ClassementIndisponible("document vide : ni titre ni contenu")
        return DocumentAClasser(titre=titre, extrait=extrait, type_acte=doc_type or None)

    def classify(
        self, title: str, content: str = "", doc_type: Optional[str] = None
    ) -> DomainResult:
        """Classe un document (synchrone), ou leve ClassementIndisponible."""
        verdict = self.classer_lot([self._document(title, content, doc_type)]).verdicts[0]
        if verdict is None:
            raise ClassementIndisponible("aucun verdict exploitable du modele")
        return verdict

    async def classify_async(
        self, title: str, content: str = "", doc_type: Optional[str] = None
    ) -> DomainResult:
        """Classe un document (asynchrone), ou leve ClassementIndisponible."""
        resultat = await self.classer_lot_async([self._document(title, content, doc_type)])
        if resultat.verdicts[0] is None:
            raise ClassementIndisponible("aucun verdict exploitable du modele")
        return resultat.verdicts[0]

    def health_check(self) -> Dict[str, object]:
        """Etat declaratif, sans appel : la sonde de Groq est sur /rag/health."""
        return {
            "service": "LegalDomainClassifier",
            "status": "healthy",
            "domains": len(CANONICAL_DOMAINS),
            "model": self.modele,
            "mode": "groq",
        }


_instance: Optional[LegalDomainClassifier] = None


def get_legal_domain_classifier() -> LegalDomainClassifier:
    """Instance singleton partagée."""
    global _instance
    if _instance is None:
        _instance = LegalDomainClassifier()
    return _instance
