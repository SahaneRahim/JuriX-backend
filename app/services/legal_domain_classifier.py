"""
Classement d'un texte juridique dans UN domaine.

Ce module remplace app/services/document_classifier.py, dont trois defauts se
cumulaient :

1. Il rendait un ENTIER qui etait une position dans son propre dictionnaire, et
   le pipeline l'ecrivait tel quel dans laws.category_id — une cle etrangere
   vers une table au tout autre ordre. D'ou des decrets affiches dans "Lois".
2. Son score saturait a `min(score / 10, 1.0)` : sur un texte de plusieurs
   pages, quatre domaines atteignaient 1,00 et c'etait la plus petite cle du
   dictionnaire qui gagnait. Mesure sur le corpus : la Loi de finances 2016
   etait classee en Droit Constitutionnel.
3. Il ne voyait jamais le TITRE, alors que sur ce corpus le titre decide seul
   pour quatre documents sur cinq — mesure sur les 2238 titres de prc.cm.

La sortie est donc un NOM de domaine, jamais un entier : la resolution vers la
table `categories` est faite ailleurs (app/services/category_resolver.py), par
le nom.

Author: JuriX Team
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from app.services.text_features import fold_accents

logger = logging.getLogger(__name__)


# ==================== LES DOMAINES ====================

# Ordre d'affichage ET priorite de departage. Explicite, jamais l'ordre
# d'insertion d'un dictionnaire.
CANONICAL_DOMAINS: Tuple[str, ...] = (
    "Droit Constitutionnel",
    "Droit Administratif",
    "Fonction Publique",
    "Droit International",
    "Finances Publiques et Fiscalité",
    "Droit Pénal",
    "Procédure Pénale",
    "Droit Civil",
    "Procédure Civile",
    "Droit de la Famille",
    "Droit du Travail et Sécurité Sociale",
    "Droit des Affaires et OHADA",
    "Droit Foncier et Domanial",
    "Droit de l'Environnement et des Ressources Naturelles",
)

_DOMAIN_INDEX = {name: i for i, name in enumerate(CANONICAL_DOMAINS)}

CONSTITUTIONNEL = CANONICAL_DOMAINS[0]
ADMINISTRATIF = CANONICAL_DOMAINS[1]
FONCTION_PUBLIQUE = CANONICAL_DOMAINS[2]
INTERNATIONAL = CANONICAL_DOMAINS[3]
FINANCES = CANONICAL_DOMAINS[4]
PENAL = CANONICAL_DOMAINS[5]
PROCEDURE_PENALE = CANONICAL_DOMAINS[6]
CIVIL = CANONICAL_DOMAINS[7]
PROCEDURE_CIVILE = CANONICAL_DOMAINS[8]
FAMILLE = CANONICAL_DOMAINS[9]
TRAVAIL = CANONICAL_DOMAINS[10]
AFFAIRES = CANONICAL_DOMAINS[11]
FONCIER = CANONICAL_DOMAINS[12]
ENVIRONNEMENT = CANONICAL_DOMAINS[13]


@dataclass(frozen=True)
class DomainResult:
    """
    Verdict de classement.

    `domain` est TOUJOURS l'un des CANONICAL_DOMAINS : jamais None, jamais un
    entier. `rule` nomme ce qui a decide, ce qui rend chaque erreur diagnosticable
    et corrigeable par l'ajout d'un motif.
    """

    domain: str
    confidence: float
    rule: str
    source: Literal["title", "content", "doctype-default"]
    runners_up: Tuple[Tuple[str, float], ...] = ()


# ==================== NORMALISATION ====================

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """
    Replie les accents, met en minuscules, remplace toute ponctuation par une
    espace.

    Le remplacement de la ponctuation n'est pas cosmetique : le corpus contient
    des titres de la forme
    `decret_n_2015_055_du_02.02.2015_accord_pret_fad`, illisibles autrement.

    fold_accents vient de text_features, deja partage par le RAG et le
    re-ranking : pas de second replieur d'accents dans le projet.
    """
    if not text:
        return ""
    return _PUNCT.sub(" ", fold_accents(text)).strip()


# ==================== REGLES SUR LE TITRE ====================
#
# Premiere correspondance gagnante. L'ORDRE EST LA CONCEPTION : chaque regle
# porte ce qu'elle doit devancer et pourquoi.

_TITLE_RULES: List[Tuple[str, str, str]] = [
    # ---------- Tier A : instruments nommes ----------
    # A1 avant A3 : "code de procedure penale" contient "penale".
    ("A1:procedure-penale", r"procedure penale|instruction judiciaire", PROCEDURE_PENALE),
    ("A2:procedure-civile", r"procedure civile|voies d execution", PROCEDURE_CIVILE),
    ("A3:code-penal", r"code penal|justice militaire|penitentiaire", PENAL),
    ("A4:code-travail", r"code du travail|prevoyance sociale", TRAVAIL),
    # A5 avant B10 : un texte de finances est bourre du mot "impot", mais aussi
    # de "administration", "president", "societe"...
    ("A5:finances-publiques",
     r"loi de finances|loi de reglement|code general des impots|code des impots"
     r"|code des douanes|regime financier de l etat|budget de l etat"
     r"|calendrier budgetaire|code de transparence",
     FINANCES),
    # A6 avant A5/B10 : les codes miniers et forestiers sont pleins de "taxe".
    ("A6:codes-ressources",
     r"code minier|code forestier|code petrolier|code gazier|code de l eau"
     r"|code de l environnement|code de la peche",
     ENVIRONNEMENT),
    ("A7:code-famille", r"code de la famille|code des personnes|etat civil", FAMILLE),
    ("A8:ohada", r"acte uniforme|\bohada\b|code de commerce", AFFAIRES),
    ("A9:code-foncier", r"code domanial|code foncier|regime foncier", FONCIER),
    ("A10:code-civil", r"code civil|code des obligations", CIVIL),
    ("A11:constitution", r"\bconstitution\b|revision constitutionnelle", CONSTITUTIONNEL),

    # ---------- Tier B : les formes dominantes du corpus ----------
    # B0 avant B7 : "statut general des etablissements publics" contient
    # "etablissement public", que B7 capterait comme du droit des affaires.
    ("B0:statut-etablissements",
     r"statut general des etablissements|statut general de la fonction publique",
     ADMINISTRATIF),
    # B1a avant B1b : 214 titres du corpus sont des ratifications d'accords de
    # PRET. C'est de l'emprunt public, pas de la diplomatie.
    ("B1a:pret-ratifie",
     r"(ratifi\w*|a ratifier|habilitant|autorisant)[^|]*"
     r"(pret|credit|financement|refinancement|emprunt|convention financiere"
     r"|titres publics|facilite|\bbad\b|\bfad\b|\bbid\b|\bbird\b|eximbank|banque)",
     FINANCES),
    ("B2:emprunt-public",
     r"emission de titres|emissions de titres|recourir a un emprunt"
     r"|contracter un (pret|emprunt)|refinancement",
     FINANCES),
    ("B1b:ratification", r"ratifi\w*|a ratifier|adhesion (du cameroun )?a", INTERNATIONAL),
    ("B3:instruments-internationaux",
     r"\btraite\b|protocole|accord cadre|accord general|accord entre"
     r"|convention (de|des|sur|entre|d )|charte (africaine|des nations)|memorandum",
     INTERNATIONAL),
    # B6 avant B4 : "attribution en concession provisoire d'un terrain" doit
    # aller au foncier, pas a la fonction publique.
    ("B6:domanial",
     r"classement au domaine|incorporation au domaine|declassement|expropriation"
     r"|utilite publique|titre foncier|immatriculation (directe|d un terrain)"
     r"|cadastre|concession (domaniale|provisoire)|terrain(s)? (situe|du domaine)",
     FONCIER),
    # B4 : 53,2 % du corpus. Apres B1/B3/B6 — une "nomination d'un Ambassadeur"
    # releve de la fonction publique, une "ratification d'accord" non.
    ("B4:actes-de-carriere",
     r"nomination|nommant|promotion|avancement|integration|reclassement"
     r"|mise a la retraite|admission au corps|admission de |titularisation"
     r"|revocation|mutation|rappel definitif|deuxieme section|bonification"
     r"|inscription (au tableau|de )|ouverture d un concours|recrutement"
     r"|renouvellement du mandat|prorogation du mandat|relevant un responsable",
     FONCTION_PUBLIQUE),
    ("B5:distinctions",
     r"decoration|medaille|ordre national|ordres nationaux|croix de la valeur"
     r"|mention honorable|elevation a la dignite",
     FONCTION_PUBLIQUE),
    ("B7:affaires",
     r"approbation des statuts|approbation des modifications|transformation de la societe"
     r"|societe a capital public|capital social|registre du commerce"
     r"|marches publics|entreprise(s)? publique|placement collectif"
     r"|valeurs mobilieres|concurrence|annonces legales",
     AFFAIRES),
    # B8 apres B4 : "nomination de Senateurs" reste de la fonction publique.
    ("B8:institutions",
     r"corps electoral|elections|referendum|senateur|assemblee nationale"
     r"|circonscriptions electorales|conseil constitutionnel"
     r"|reamenagement du gouvernement|formation du gouvernement",
     CONSTITUTIONNEL),
    ("B10:fiscalite", r"impot|fiscal|\btaxe|douan|\btva\b|tresor public", FINANCES),
    ("B13:ressources",
     r"environnement|forestier|\bfaune\b|\bflore\b|pollution|minier|\bmines\b"
     r"|petrolier|hydrocarbures|eaux et forets|aires protegees|biosecurite"
     r"|deforestation|permis de recherche",
     ENVIRONNEMENT),
    ("B11:travail",
     r"salarie|employeur|licenciement|convention collective|securite sociale"
     r"|remuneration|indemnites|\bconges\b|salaire|formation professionnelle",
     TRAVAIL),
    ("B12:famille", r"\bmariage\b|\bdivorce\b|filiation|\bsuccession\b|autorite parentale|adoption",
     FAMILLE),
    # B14 apres B4 : "Delegation Generale a la Surete Nationale" apparait dans
    # des dizaines de titres de nomination.
    ("B14:penal",
     r"infraction|repression|amnistie|remise de peine|commutation de peine"
     r"|\bgrace\b|piraterie|surete de l aviation|terrorisme",
     PENAL),
    ("B15:civil", r"obligations civiles|responsabilite civile|prescription civile", CIVIL),
    # B9 en dernier : les mots les moins specifiques du corpus.
    ("B9:organisation",
     r"creation|organisation|fonctionnement|reorganisation|fixant les modalites"
     r"|portant application|comite|commission|composition|jour ferie|chomee"
     r"|obseques|deuil national|convocation|recensement|hopital|ecole|academie"
     r"|decoupage|reglementation|ordonnant la publication|rendant executoire"
     r"|instituant|denomination"
     # Les secteurs regules (ferroviaire, postal, telecoms, electricite) sont
     # de la regulation administrative. Cette alternative est ici, en derniere
     # regle, et non plus haut : « regissant la biosecurite » doit d'abord
     # tomber sur B13, sinon un texte environnemental deviendrait administratif.
     r"|regissant le secteur|secteur ferroviaire|transport (ferroviaire|routier|aerien|maritime)"
     r"|telecommunication|energie electrique|service postal",
     ADMINISTRATIF),
]

_COMPILED_TITLE_RULES = [
    (rule_id, re.compile(pattern), domain) for rule_id, pattern, domain in _TITLE_RULES
]


# ==================== LEXIQUE DE CONTENU ====================
#
# Chaque entree fait au moins deux mots, ou est sans ambiguite. Les mots-cles
# de l'ancien classifieur qui polluaient tout sont BANNIS seuls :
#   'sa'            — le possessif francais
#   'is'            — le verbe anglais
#   'president'     — dans un corpus de decrets PRESIDENTIELS
#   'domaine'       — "domaine de competence"
#   'protection', 'administration', 'convention', 'international', 'societe'

_CONTENT_LEXICON: Dict[str, List[Tuple[str, float]]] = {
    CONSTITUTIONNEL: [("conseil constitutionnel", 2.0), ("corps electoral", 2.0),
                      ("assemblee nationale", 1.0), ("pouvoir legislatif", 1.5),
                      ("referendum", 1.5), ("senat", 1.0)],
    ADMINISTRATIF: [("service public", 1.0), ("etablissement public", 1.0),
                    ("autorite administrative", 1.5), ("contentieux administratif", 2.0),
                    ("acte reglementaire", 1.5), ("tutelle technique", 1.0)],
    FONCTION_PUBLIQUE: [("fonctionnaire", 1.5), ("agent public", 1.5),
                        ("avancement de grade", 2.0), ("statut general", 1.0),
                        ("regime disciplinaire", 1.5), ("indice de solde", 2.0)],
    INTERNATIONAL: [("droit international", 2.0), ("nations unies", 1.5),
                    ("accord bilateral", 2.0), ("union africaine", 1.5),
                    ("instrument de ratification", 2.0)],
    FINANCES: [("administration fiscale", 2.0), ("loi de finances", 2.5),
               ("recettes fiscales", 2.0), ("contribuable", 1.5), ("droits de douane", 2.0),
               ("taxe sur la valeur ajoutee", 2.5), ("credit budgetaire", 1.5),
               ("dette publique", 1.5)],
    PENAL: [("code penal", 2.5), ("peine d emprisonnement", 2.0), ("infraction", 1.0),
            ("ministere public", 1.0), ("detention provisoire", 1.5)],
    PROCEDURE_PENALE: [("procedure penale", 2.5), ("garde a vue", 2.0),
                       ("juge d instruction", 2.0), ("mandat de depot", 1.5)],
    CIVIL: [("code civil", 2.5), ("responsabilite civile", 2.0), ("obligation contractuelle", 1.5),
            ("dommages et interets", 1.5), ("debiteur", 1.0), ("creancier", 1.0)],
    PROCEDURE_CIVILE: [("procedure civile", 2.5), ("voies d execution", 2.0),
                       ("saisie conservatoire", 1.5)],
    FAMILLE: [("acte de mariage", 2.0), ("autorite parentale", 2.0), ("filiation", 1.5),
              ("regime matrimonial", 2.0), ("succession ab intestat", 2.0)],
    TRAVAIL: [("contrat de travail", 2.5), ("convention collective", 2.0),
              ("inspection du travail", 2.0), ("securite sociale", 1.5),
              ("licenciement", 1.5), ("salarie", 1.0)],
    AFFAIRES: [("societe commerciale", 2.0), ("acte uniforme", 2.5), ("capital social", 1.5),
               ("registre du commerce", 2.0), ("assemblee generale", 1.0),
               ("conseil d administration", 1.0)],
    FONCIER: [("domaine public", 2.0), ("domaine national", 2.0), ("titre foncier", 2.5),
              ("expropriation", 2.0), ("concession domaniale", 2.0), ("bornage", 1.5)],
    ENVIRONNEMENT: [("protection de l environnement", 2.5), ("ressources naturelles", 2.0),
                    ("permis de recherche", 1.5), ("etude d impact", 2.0),
                    ("aire protegee", 2.0), ("exploitation forestiere", 2.0)],
}

# Bloc de visas : chaque decret camerounais commence par "Vu la Constitution ;".
# C'est l'amplificateur mesure du faux positif "Droit Constitutionnel".
_VISA_LINE = re.compile(r"^\s*vu\b.*$", re.IGNORECASE | re.MULTILINE)
_ARTICLE_NUMBER = re.compile(r"\barticle\s*\d+|\bart\.?\s*\d+")

# Seuils d'acceptation de la passe de contenu.
MIN_CONTENT_SHARE = 0.20      # le domaine de tete doit peser au moins ca
MIN_CONTENT_MARGIN = 0.05     # et devancer le suivant d'au moins ca


class LegalDomainClassifier:
    """
    Classement deterministe : titre d'abord, contenu ensuite, defaut sinon.

    Ne connait pas la table `categories` et ne manipule aucun identifiant.
    """

    def classify(
        self,
        title: str,
        content: str = "",
        doc_type: Optional[str] = None,
    ) -> DomainResult:
        """
        Rend le domaine du document. Toujours un domaine, jamais None.

        Args:
            title: le titre du document — signal principal
            content: le texte integral, utilise seulement si le titre ne tranche pas
            doc_type: loi, decret, arrete... sert uniquement au defaut
        """
        rule_id, domain = self._match_title(title)
        if domain:
            return DomainResult(
                domain=domain, confidence=0.90, rule=rule_id, source="title"
            )

        scored = self._score_content(content)
        if scored:
            top_domain, top_share = scored[0]
            runner_share = scored[1][1] if len(scored) > 1 else 0.0

            if top_share >= MIN_CONTENT_SHARE and (top_share - runner_share) >= MIN_CONTENT_MARGIN:
                return DomainResult(
                    domain=top_domain,
                    confidence=round(min(top_share * 1.5, 0.85), 3),
                    rule="content:margin",
                    source="content",
                    runners_up=tuple(scored[1:3]),
                )

            # Egalite : elle n'est PAS tranchee en silence. Elle est nommee
            # dans `rule`, journalisee, et comptee a part dans le rapport de
            # reclassement — un depart au coude a coude ne doit pas se
            # confondre avec une absence totale de signal.
            if len(scored) > 1:
                logger.warning(
                    "Classement indecis : %s (%.2f) contre %s (%.2f) — titre=%r",
                    scored[0][0], top_share, scored[1][0], runner_share, (title or "")[:80],
                )
                return DomainResult(
                    domain=ADMINISTRATIF,
                    confidence=0.15,
                    rule=f"tie:{scored[0][0]}|{scored[1][0]}",
                    source="doctype-default",
                    runners_up=tuple(scored[:2]),
                )

        return DomainResult(
            domain=ADMINISTRATIF,
            confidence=0.15,
            rule="default:acte-executif",
            source="doctype-default",
            runners_up=tuple(scored[:1]) if scored else (),
        )

    def explain(self, title: str, content: str = "", doc_type: Optional[str] = None) -> str:
        """Une ligne lisible : quel domaine, par quelle regle, avec quelle confiance."""
        result = self.classify(title, content, doc_type)
        suffix = ""
        if result.runners_up:
            suffix = " | suivants: " + ", ".join(
                f"{d} {s:.2f}" for d, s in result.runners_up
            )
        return f"{result.domain} [{result.rule}] {result.confidence:.2f} ({result.source}){suffix}"

    def health_check(self) -> Dict[str, object]:
        return {
            "service": "LegalDomainClassifier",
            "status": "healthy",
            "domains": len(CANONICAL_DOMAINS),
            "title_rules": len(_COMPILED_TITLE_RULES),
            "mode": "rules",
        }

    # ==================== INTERNE ====================

    def _match_title(self, title: str) -> Tuple[str, Optional[str]]:
        normalised = normalise(title)
        if not normalised:
            return "", None
        for rule_id, pattern, domain in _COMPILED_TITLE_RULES:
            if pattern.search(normalised):
                return rule_id, domain
        return "", None

    def _score_content(self, content: str) -> List[Tuple[str, float]]:
        """
        Scores normalises en DISTRIBUTION, avec log1p.

        log1p supprime la dependance a la longueur : l'ancien `min(score/10, 1)`
        saturait a dix occurrences, si bien qu'un texte de cent pages atteignait
        1,00 sur quatre domaines a la fois. La normalisation en distribution
        rend cette egalite arithmetiquement impossible et donne un sens a la
        marge entre le premier et le second.
        """
        if not content:
            return []

        # Le bloc de visas est retire AVANT tout comptage.
        stripped = _VISA_LINE.sub(" ", content)
        text = normalise(_ARTICLE_NUMBER.sub(" ", stripped.lower()))
        if not text:
            return []

        raw: Dict[str, float] = {}
        for domain, entries in _CONTENT_LEXICON.items():
            total = 0.0
            for term, weight in entries:
                folded = normalise(term)
                if not folded:
                    continue
                count = text.count(folded)
                if count:
                    total += weight * math.log1p(count)
            if total > 0:
                raw[domain] = total

        if not raw:
            return []

        denominator = sum(raw.values())
        shares = [(domain, score / denominator) for domain, score in raw.items()]
        # Departage explicite : score decroissant, puis ordre canonique declare.
        shares.sort(key=lambda item: (-item[1], _DOMAIN_INDEX[item[0]]))
        return shares


_instance: Optional[LegalDomainClassifier] = None


def get_legal_domain_classifier() -> LegalDomainClassifier:
    """Instance partagee. Le classifieur est sans etat, une seule suffit."""
    global _instance
    if _instance is None:
        _instance = LegalDomainClassifier()
    return _instance
