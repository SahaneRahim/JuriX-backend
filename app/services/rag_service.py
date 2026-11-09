"""
RAGService - Retrieval-Augmented Generation service for legal Q&A.

Orchestrates the complete RAG pipeline:
1. Document retrieval (SearchService)
2. Conversation history loading
3. Prompt construction (persona-adapted)
4. Answer generation (service retenu par LLM_PROVIDER, voir app/services/llm.py)
5. Citation extraction
6. Persistence (Database)
"""

import asyncio
import logging
import re
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import zip_longest
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import anyio

# `text` est indispensable a _fallback_search. Il etait importe localement
# dans la fonction avant sa reecriture ; sans lui, chaque appel levait un
# NameError avale par le except de la fonction, et le repli multilingue
# ne renvoyait jamais rien.
from sqlalchemy import desc, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.models.conversation import Conversation, Message
from app.schemas.rag import (
    Citation,
    CitationIntrouvable,
    CitationPositionnee,
    Progression,
    RAGRequest,
    RAGResponse,
    RAGStreamChunk,
    SourceCitee,
    StatutProuve,
    TexteConsulte,
    TexteTrouve,
)
from app.schemas.search import ChunkResult, SearchFilters, SearchRequest
from app.services.annuaire import contacts_pour
from app.services.article_reference import normalize_number, parse_reference
from app.services.chat_groq import service_du_chat
from app.services.intent_classifier import (
    INTENT_JURIDIQUE,
    IntentResult,
    classify_intent,
)
from app.services.lecture_des_citations import (
    # Les primitives de lecture des citations vivent la-bas ; elles restent
    # importables d'ici sous leurs noms d'origine.
    CITATION_REGEX,
    Occurrence,
    _citable_article_number,
    _natures_propres,
    _normalize_article_number,
    article_compte,
    indices_utf16,
    lire_les_citations,
    normaliser_le_nom,
)
from app.services.llm import (  # noqa: F401 - mentions reexportees pour les tests
    ERREURS_LLM,
    ERREURS_QUOTA,
    ERREURS_SATURATION,
    FIN_ARRETEE,
    MENTION_REPONSE_INTERROMPUE,
    MENTION_REPONSE_TRONQUEE,
    get_llm_service,
    mention_de_fin,
    mention_de_reponse,
)
from app.services.postgres_search_service import escape_like
from app.services.prompts import (
    CONTEXT_TEMPLATE,
    build_context_string,
    format_conversation_history,
    get_conversational_prompt,
    get_system_prompt,
    message_sans_resultat,
    message_sans_texte_pertinent,
)
from app.services.reperes_textes import dates_de_l_acte, numero_de_l_acte
from app.services.reranker import rerank_with_llm
from app.services.search_service import (
    CHEMIN_ABSENT,
    CHEMIN_ECHEC,
    CHEMIN_HNSW,
    CHEMIN_REPLI_HNSW,
    SearchService,
    VoletsDeLaRecherche,
    question_budgetaire,
)
from app.services.statut_des_sources import (
    liens_qui_touchent,
    references_a_resoudre,
    statut_prouve,
)
from app.services.text_features import STOPWORDS
from app.utils.titre_officiel import numero_du_titre

logger = logging.getLogger(__name__)


# Budget de generation. Doit couvrir la reflexion du modele ET la reponse.
# 4096 ne suffisait pas : mesure sur « sanctions du harcelement sexuel »,
# 2 804 jetons de reflexion pour 695 de reponse, et la meme question, posee
# depuis l'interface, s'arretait au milieu d'une phrase — sans erreur.
ANSWER_MAX_TOKENS = 8192

# Budget d'une reponse conversationnelle. Deux phrases sont attendues, et
# pourtant 2048 : la reflexion du modele est facturee sur ce budget avant la
# premiere phrase, exactement comme ci-dessus. Un plafond serre ne rendrait pas
# une reponse courte, il rendrait une reponse VIDE.
CONVERSATION_MAX_TOKENS = 2048

# Rendu quand aucun service de generation n'est branche (cle absente). Le
# message ne nomme plus Gemini : le fournisseur depend de LLM_PROVIDER.
MESSAGE_LLM_ABSENT = (
    "Service de génération non configuré : renseignez la clé du fournisseur "
    "choisi par LLM_PROVIDER dans .env."
)

# Temps laisse a l'enregistrement d'une reponse arretee par le lecteur. Il se
# fait apres la deconnexion, a l'abri de l'annulation : borne, pour qu'une base
# qui ne repond plus ne retienne pas la requete indefiniment.
DELAI_ENREGISTREMENT_ARRETEE_S = 5.0

# Une reponse conversationnelle est certaine ET sans source : les deux vont
# ensemble. C'est deja le choix de la sortie anticipee « article absent »
# (confidence=1.0, sources=[]). 0.0 serait un mensonge : ce score dit « je ne
# suis pas sur », et une salutation notee 0 % de confiance serait faux le jour
# ou le score sera affiche.
CONFIANCE_CONVERSATIONNELLE = 1.0

# PAS DE SOURCE FABRIQUEE. `sources` (et `Message.sources`, ce que montre une
# conversation rouverte) ne porte que les articles que la reponse CITE et que
# la lecture des citations a RATTACHES a un extrait lu (_sources_citees). Le
# repli qui prenait, faute de citation, le premier extrait ou l'un des trois
# premiers dont l'intitule partageait un mot avec la question est retire, avec
# ses gardes (seuil de pertinence, aveux du modele) : revue du 29/10/2026 (tour
# 3), une reponse sur le Code du travail qui ne citait rien recevait pour
# source PRC-9875 art. 35.5, un tableau de droits d'exportation d'animaux de
# la loi de finances 2024, enregistre dans l'historique et affiche sous
# « Sources » a la reouverture. Le seuil n'y pouvait rien : la fusion
# normalise les scores, et le premier extrait vaut toujours 1. Une reponse qui
# ne cite rien garde ses « Textes consultés, non cités »
# (`textes_consultes_non_cites`), qui disent ce que le modele a lu sans en
# faire une source.


def _pluriel(nombre: int, singulier: str, pluriel: str) -> str:
    return f"{nombre} {singulier if nombre <= 1 else pluriel}"


# Les lignes de progression du chat en flux, dans la langue de la question.
# Elles disent ce qui se passe, pas ce qu'on espere : « Rédaction » n'est
# emis qu'au moment ou le modele est appele.
_LIBELLES_DE_PROGRESSION = {
    "fr": {
        "recherche": "Recherche dans les textes…",
        "textes_trouves": lambda articles, textes: (
            f"{_pluriel(articles, 'article trouvé', 'articles trouvés')} "
            f"dans {_pluriel(textes, 'texte', 'textes')}"
        ),
        "textes_seuls": lambda textes: _pluriel(textes, "texte trouvé", "textes trouvés"),
        "aucun": "Aucun texte trouvé",
        "redaction": "Rédaction de la réponse…",
    },
    "en": {
        "recherche": "Searching the texts…",
        "textes_trouves": lambda articles, textes: (
            f"{_pluriel(articles, 'article found', 'articles found')} "
            f"in {_pluriel(textes, 'text', 'texts')}"
        ),
        "textes_seuls": lambda textes: _pluriel(textes, "text found", "texts found"),
        "aucun": "No text found",
        "redaction": "Writing the answer…",
    },
}


def _sans_citation() -> Dict[str, list]:
    """
    Les champs de citation d'une reponse que le serveur ecrit lui-meme
    (aucun resultat, article absent du document ouvert) : vides, mais
    presents, pour que le dernier evenement ait toujours la meme forme.
    """
    return {
        "citations": [], "citations_introuvables": [],
        "articles_cites": [], "textes_consultes_non_cites": [], "contacts": [],
    }


@asynccontextmanager
async def _ferme_a_la_sortie(flux):
    """
    `contextlib.aclosing`, tolerant a un flux qui n'a pas d'`aclose`.

    Le flux du modele est un generateur asynchrone : le fermer coupe sa
    connexion HTTP au fournisseur. Sans fermeture explicite, un flux
    abandonne (lecteur parti pendant que le serveur envoyait un morceau) ne
    l'etait qu'au passage du ramasse-miettes. Les doublures de tests rendent
    parfois un objet sans `aclose` : rien a fermer, rien a casser.
    """
    try:
        yield flux
    finally:
        fermer = getattr(flux, "aclose", None)
        if fermer is not None:
            await fermer()


# « la Loi N°2008/001… » -> « Loi N°2008/001… » : l'intitule d'un texte nomme
# par un lien, comme titre de citation. Un espace exige apres « la », « le »,
# « les » : sans lui, « le » mordait sur « les » et laissait « s textes ».
_SANS_ARTICLE_DEFINI = re.compile(r"^(?:(?:la|les|le)\s+|l['’]\s*)", re.IGNORECASE)


def _date_affichee(jour: Optional[date], titre: Optional[str]) -> Optional[date]:
    """
    La date d'un texte dans les champs du chat (`date_du_texte`), selon la
    regle de sa page et des listes (reperes_textes.dates_de_l_acte) : None
    pour un 1er janvier de remplissage ou pour la seule date de publication au
    Journal officiel. La date brute en base faisait lire « 1er janvier 2024 »
    sous un decret dont l'intitule ne donne que l'annee, la ou sa page dit
    « 2024 ».
    """
    return dates_de_l_acte(jour, titre).date_signature


def _requete_des_identifiants(references: List[str]):
    """
    (reference, identifiant) des textes PUBLIES parmi `references`. Une
    fonction a part pour qu'un test la fasse echouer en SQL, sur la vraie base.
    Import local, comme dans _get_priority_document_context.

    Publies seulement, comme `TexteLie` sur la page d'un texte
    (statut_juridique.texte_lie) : un texte archive, ou encore en brouillon,
    devenait un lien dans le chat, vers une page qui ne le montre pas, alors
    que sa page ne le liait pas.
    """
    from app.models.law import Law

    return select(Law.reference, Law.id).where(
        Law.reference.in_(references), Law.status == "published"
    )


# ==================== CE QUE LE CHAT GARDE DE LA RECHERCHE ====================

# Origine (ChunkResult.source) d'un article que la question NOMME avec son
# texte (« l'article 39 du Code du travail »), lu directement en base et non
# trouve par la recherche : voir RAGService._article_nomme.
SOURCE_ARTICLE_NOMME = "article_nomme"

# Langue de la QUESTION, et non de l'interface : une question anglaise posee
# depuis l'interface francaise est anglaise. Des mots-outils, des elisions et
# des accents, sans modele : la question est courte, et le detecteur fastText
# de app/services/language_detector.py exige 50 caracteres et charge 131 Mo.
# Mesure sur les 102 questions du jeu, leurs traductions et les questions
# suggerees du front (29/10/2026) : aucune question anglaise lue francaise ni
# l'inverse ; les mots-cles nus (« ratification rallonge 2019 ») restent
# INCERTAINS (None), 16 francaises et 37 anglaises sur 102.
_MOTS_FRANCAIS = frozenset(
    "le la les du de des au aux un une et ou est sont que qui quel quelle quels quelles "
    "quoi quand comment pourquoi combien dans sur sous avec sans pour par ce cette ces "
    "mon ma mes sa ses leur leurs il elle ils elles je nous vous peut peuvent doit faut "
    "ne pas être avoir dit prévoit selon entre".split()
)
_MOTS_ANGLAIS = frozenset(
    "the of and or is are was were be been what which who whom whose when where why how "
    "does do did can could should would must may might shall will my your his her its our "
    "their to in at for with by from under about into this that these those there if".split()
)
_ELISIONS_FRANCAISES = re.compile(r"\b(?:l|d|qu|n|s|j|c|m|t)['’]\w", re.IGNORECASE)
_ELISIONS_ANGLAISES = re.compile(r"\w['’](?:s|t|re|ve|ll|d)\b", re.IGNORECASE)
_ACCENTS = re.compile(r"[éèêëàâîïôûùüçœ]")


def langue_de_la_question(question: str) -> Optional[str]:
    """
    « fr », « en », ou None quand rien ne tranche.

    Les accents ne comptent que dans un mot en minuscule : « Yaoundé »,
    « Misérables » sont des noms propres, ils faisaient lire francaises des
    questions anglaises.
    """
    mots = re.findall(r"[A-Za-zÀ-ÿœŒ]+", question or "")
    minuscules = [m.lower() for m in mots]
    francais = (
        sum(1 for m in minuscules if m in _MOTS_FRANCAIS)
        + len(_ELISIONS_FRANCAISES.findall(question or ""))
        + sum(1 for m in mots if m[0].islower() and _ACCENTS.search(m))
    )
    anglais = (
        sum(1 for m in minuscules if m in _MOTS_ANGLAIS)
        + len(_ELISIONS_ANGLAISES.findall(question or ""))
    )
    if francais > anglais:
        return "fr"
    if anglais > francais:
        return "en"
    return None


# ---------------------- L'ARTICLE NOMME : QUEL TEXTE ? ----------------------
#
# Un article nomme avec son texte est lu directement en base et AJOUTE en tete
# du contexte (RAGService._article_nomme). Le texte n'est reconnu que de deux
# facons, toutes deux exactes :
# 1. son NOM D'USAGE, dans la table ci-dessous, relue texte par texte ;
# 2. son NUMERO propre, nature comprise (« loi n° 92/007 », « décret
#    n° 2018/366 ») : le numero que porte l'en-tete de son intitule, pas un
#    numero qu'il ne fait que citer.
# Rien d'autre. Le tour precedent prenait la premiere loi dont l'intitule
# CONTENAIT tous les mots du nom : revue du 29/10/2026, 13 noms courants sur
# 55 menaient a un autre texte (« loi sur le terrorisme » -> une adhesion a une
# convention, « loi n° 2013/004 » -> la loi de 2017 qui la modifie, « Acte
# uniforme relatif au droit des sociétés » -> les societes cooperatives), et
# l'article de cet autre texte passait en tete du contexte. Un nom absent de
# la table n'ajoute rien : la recherche fait son travail, comme avant.


@dataclass(frozen=True)
class TexteNommable:
    """Un texte que la question peut designer par son nom d'usage."""

    reference: str
    """Reference en base."""
    intitule: str
    """Debut de son intitule, relu en base le 29/10/2026 (scripts/eval/evaluer_contexte_chat.py --table)."""
    noms: Tuple[str, ...]
    """Ses noms d'usage, francais et anglais. Compares mot pour mot (_cle_du_nom)."""
    annees: Tuple[int, ...] = ()
    """Les annees qu'un nom peut lui ajouter (« Code du travail de 1992 ») ; une autre annee : rien."""


# Les grands textes du corpus, relus en base le 29/10/2026. Absents du corpus,
# donc absents d'ici : le Code general des impots (ses articles ne sont que
# recopies dans les lois de finances), le Code de la famille, le Code de la
# route. Le Code minier est celui de 2023 (loi n° 2023/014, PRC-9866), dont
# l'article 200 abroge les dispositions contraires de la loi n° 2016/017
# (PRC-5621) : « Code minier de 2016 » ne designe donc rien ici. Le Code penal
# est la loi n° 2016/007 ; sa partie reglementaire (decret n° 2016/319) n'en
# prend pas le nom.
TEXTES_NOMMABLES: Tuple[TexteNommable, ...] = (
    TexteNommable(
        "LOI-1992-007", "Loi N°92/007 du 14 août 1992 portant Code du travail",
        ("code du travail", "code de travail", "labour code", "labor code"), (1992,),
    ),
    TexteNommable(
        "PRC-4722", "Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal",
        ("code pénal", "penal code", "criminal code"), (2016,),
    ),
    TexteNommable(
        "LOI-2005-007", "Loi N°2005/007 du 27 juillet 2005 portant Code de procédure pénale",
        ("code de procédure pénale", "cpp", "criminal procedure code",
         "code of criminal procedure"), (2005,),
    ),
    TexteNommable(
        "CODE-CIVIL", "Code civil applicable au Cameroun, annoté de la jurisprudence de la "
        "Cour suprême",
        ("code civil", "civil code"),
    ),
    TexteNommable(
        "PRC-6283", "Loi N°96/06 du 18 janvier 1996 portant révision de la Constitution du "
        "02 juin 1972",
        ("constitution",), (1972, 1996),
    ),
    TexteNommable(
        "LOI-2012-001", "Loi N°2012/001 du 19 avril 2012 portant Code électoral, modifiée et "
        "complétée par la loi n°2012/017 du 21 décembre 2012",
        ("code électoral", "electoral code"), (2012,),
    ),
    TexteNommable(
        "PRC-9866", "Loi N°2023/014 du 19 décembre 2023 portant Code Minier",
        ("code minier", "mining code"), (2023,),
    ),
    TexteNommable(
        "CIMA-CODE-2019", "Code des assurances des États membres de la CIMA, édition 2019",
        ("code des assurances", "code cima", "code des assurances cima", "cima code",
         "insurance code"), (2019,),
    ),
    TexteNommable(
        "LOI-1968-LF-3", "Loi N°1968-LF-3 du 11 juin 1968 portant Code de la nationalité "
        "camerounaise",
        ("code de la nationalité", "nationality code"), (1968,),
    ),
    TexteNommable(
        "LOI-1967-LF-7", "Loi N°67-LF-7 du 12 juin 1967 instituant un Code des prestations "
        "familiales",
        ("code des prestations familiales",), (1967,),
    ),
    TexteNommable(
        "PRC-5695", "Loi N° 2017/012 du 12 juillet 2017 portant code de justice militaire",
        ("code de justice militaire", "military justice code"), (2017,),
    ),
    TexteNommable(
        "PRC-7487", "Loi N° 2019/008 du 25 avril 2019 portant code pétrolier",
        ("code pétrolier", "petroleum code"), (2019,),
    ),
    TexteNommable(
        "PRC-8036", "Loi 2019/024 du 24 décembre 2019 portant code général des collectivités "
        "territoriales décentralisées",
        ("code général des collectivités territoriales décentralisées",
         "code général des collectivités territoriales"), (2019,),
    ),
    TexteNommable(
        "PRC-6521", "Décret N°2018/366 du 20 juin 2018 portant Code des Marchés Publics",
        ("code des marchés publics", "public procurement code"), (2018,),
    ),
    TexteNommable(
        "OHADA-AUDCG-2010", "Acte uniforme révisé portant sur le droit commercial général du "
        "15 décembre 2010",
        ("acte uniforme portant sur le droit commercial général", "audcg",
         "uniform act on general commercial law", "uniform act relating to general commercial law"),
        (2010,),
    ),
    TexteNommable(
        "OHADA-AUSCGIE-2014", "Acte uniforme révisé relatif au droit des sociétés commerciales et "
        "du groupement d'intérêt économique du 30 janvier 2014",
        ("acte uniforme relatif au droit des sociétés commerciales et du groupement d'intérêt "
         "économique", "acte uniforme relatif au droit des sociétés commerciales",
         "acte uniforme relatif au droit des sociétés", "auscgie",
         "uniform act relating to commercial companies and economic interest groups",
         "uniform act on commercial companies"),
        (2014,),
    ),
    TexteNommable(
        "OHADA-AUSCOOP-2010", "Acte uniforme relatif au droit des sociétés coopératives du "
        "15 décembre 2010",
        ("acte uniforme relatif au droit des sociétés coopératives", "auscoop",
         "uniform act on cooperative societies"), (2010,),
    ),
    TexteNommable(
        "OHADA-AUS-2010", "Acte uniforme révisé portant organisation des sûretés du "
        "15 décembre 2010",
        ("acte uniforme portant organisation des sûretés", "acte uniforme relatif aux sûretés",
         "uniform act organising securities", "uniform act on securities"), (2010,),
    ),
    TexteNommable(
        "OHADA-AUPSRVE-2023", "Acte uniforme portant organisation des procédures simplifiées de "
        "recouvrement et des voies d'exécution, révisé",
        ("acte uniforme portant organisation des procédures simplifiées de recouvrement et des "
         "voies d'exécution", "acte uniforme relatif aux voies d'exécution", "aupsrve",
         "uniform act organising simplified recovery procedures and measures of execution"),
        (2023,),
    ),
    TexteNommable(
        "OHADA-AUPCAP-2015", "Acte uniforme portant organisation des procédures collectives "
        "d'apurement du passif du 10 septembre 2015",
        ("acte uniforme portant organisation des procédures collectives d'apurement du passif",
         "acte uniforme relatif aux procédures collectives", "aupcap",
         "uniform act organising collective proceedings for wiping off debts"), (2015,),
    ),
    TexteNommable(
        "OHADA-AUA-2017", "Acte uniforme relatif au droit de l'arbitrage du 23 novembre 2017",
        ("acte uniforme relatif au droit de l'arbitrage", "uniform act on arbitration law",
         "uniform act on arbitration"), (2017,),
    ),
    TexteNommable(
        "OHADA-AUM-2017", "Acte uniforme relatif à la médiation du 23 novembre 2017",
        ("acte uniforme relatif à la médiation", "uniform act on mediation"), (2017,),
    ),
    TexteNommable(
        "OHADA-AUCTMR-2003", "Acte uniforme relatif aux contrats de transport de marchandises "
        "par route du 22 mars 2003",
        ("acte uniforme relatif aux contrats de transport de marchandises par route",
         "auctmr"), (2003,),
    ),
    TexteNommable(
        "OHADA-AUDCIF-2017", "Acte uniforme relatif au droit comptable et à l'information "
        "financière du 26 janvier 2017",
        ("acte uniforme relatif au droit comptable et à l'information financière", "audcif"),
        (2017,),
    ),
    TexteNommable(
        "OHADA-SYCEBNL-2022", "Acte uniforme relatif au système comptable des entités à but non "
        "lucratif",
        ("acte uniforme relatif au système comptable des entités à but non lucratif",
         "sycebnl"), (2022, 2023),
    ),
    TexteNommable(
        "OHADA-TRAITE-1993", "Traité relatif à l'harmonisation du droit des affaires en Afrique "
        "du 17 octobre 1993",
        ("traité ohada", "traité relatif à l'harmonisation du droit des affaires en afrique",
         "ohada treaty", "treaty on the harmonisation of business law in africa"), (1993,),
    ),
)

# Les mots qu'un nom peut porter sans changer de texte : articles, liaisons,
# et quelques qualificatifs (« le Code pénal camerounais », « l'Acte uniforme
# OHADA relatif au droit de l'arbitrage », « le nouveau Code pénal »). Tout
# autre mot change la cle, et une cle absente de la table ne designe rien :
# « Code du travail français » n'est pas le Code du travail camerounais.
_LIAISONS_DU_NOM = frozenset(
    "le la les l du de des d au aux a en et relatif relative relatifs relatives portant sur "
    "revise revisee of the on relating to and".split()
)
_QUALIFICATIFS_DU_NOM = frozenset(
    "cameroun camerounais camerounaise ohada republique nouveau nouvelle actuel actuelle "
    "vigueur".split()
)
_ANNEE = re.compile(r"(?:19|20)\d\d")


def _sans_accents(texte: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texte or "") if unicodedata.category(c) != "Mn"
    ).lower()


def _cle_du_nom(nom: str) -> Tuple[frozenset, frozenset, Tuple[int, ...]]:
    """
    (mots du nom, les memes sans qualificatifs, annees), liaisons retirees.
    Les deux cles : « Traité OHADA » se lit avec « ohada », « l'Acte uniforme
    OHADA relatif à la médiation » sans.
    """
    mots = [m for m in re.findall(r"[a-z0-9]+", _sans_accents(nom)) if m not in _LIAISONS_DU_NOM]
    annees = tuple(int(m) for m in mots if _ANNEE.fullmatch(m))
    entiere = frozenset(m for m in mots if not _ANNEE.fullmatch(m))
    return entiere, entiere - _QUALIFICATIFS_DU_NOM, annees


def _table_des_noms() -> Dict[frozenset, TexteNommable]:
    table: Dict[frozenset, TexteNommable] = {}
    for texte in TEXTES_NOMMABLES:
        for nom in texte.noms:
            cle, _, _ = _cle_du_nom(nom)
            # Deux textes sous la meme cle : la table serait ambigue.
            assert table.get(cle, texte) is texte, f"nom en double dans la table : {nom}"
            table[cle] = texte
    return table


_TABLE_DES_NOMS = _table_des_noms()


def texte_de_la_table(nom: str) -> Optional[TexteNommable]:
    """Le texte que `nom` designe dans TEXTES_NOMMABLES, mot pour mot ; None sinon."""
    entiere, sans_qualificatifs, annees = _cle_du_nom(nom)
    texte = _TABLE_DES_NOMS.get(entiere) or _TABLE_DES_NOMS.get(sans_qualificatifs)
    if texte is None or any(annee not in texte.annees for annee in annees):
        return None
    return texte


# « loi n° 92/007 », « décret n°2018/366 », « Law No. 2013/004 », « loi
# 2016-007 » : une nature et un numero fait d'au moins deux parties. Lu sur le
# nom sans accents ni majuscules.
_NATURES_NUMEROTEES = {
    "loi": "loi", "law": "loi", "decret": "decret", "decree": "decret",
    "ordonnance": "ordonnance", "ordinance": "ordonnance", "arrete": "arrete",
}
_NOM_NUMEROTE = re.compile(
    r"\b(loi|law|decret|decree|ordonnance|ordinance|arrete)\b(?:\s+(?:organique|cadre))?"
    r"\s*(?:n\s*[°ºo]?\s*\.?|no\.?|numero|number)?\s*"
    r"(\d{1,4}\s*[/-]\s*[0-9a-z]+(?:\s*[/-]\s*[0-9a-z]+)*)"
)


def forme_du_numero(numero: str) -> Tuple[Any, ...]:
    """
    Le numero d'un texte sous une forme comparable : « 92/007 » et
    « 1992-07 » donnent (1992, 7). Une annee sur deux chiffres en tete d'un
    numero en deux parties ou plus est completee (92 -> 1992, 16 -> 2016).
    """
    parties = [p for p in re.split(r"\s*[/-]\s*", _sans_accents(numero).strip()) if p]
    forme: List[Any] = []
    for rang, partie in enumerate(parties):
        if partie.isdigit():
            valeur = int(partie)
            if rang == 0 and len(parties) > 1 and len(partie) <= 2:
                valeur += 1900 if valeur >= 30 else 2000
            forme.append(valeur)
        else:
            forme.append(partie)
    return tuple(forme)


def numero_du_nom(nom: str) -> Optional[Tuple[str, Tuple[Any, ...]]]:
    """(nature, numero comparable) quand le nom designe un texte par son numero."""
    trouve = _NOM_NUMEROTE.search(_sans_accents(nom))
    if trouve is None:
        return None
    return _NATURES_NUMEROTEES[trouve.group(1)], forme_du_numero(trouve.group(2))


def numero_de_l_intitule(titre: Optional[str]) -> Optional[Tuple[str, Tuple[Any, ...]]]:
    """
    (nature, numero comparable) que porte l'EN-TETE d'un intitule
    (titre_officiel.numero_du_titre, reperes_textes.numero_de_l_acte), jamais
    un numero que son objet cite : « Loi N° 2017/015 … modifiant la loi
    N° 2013/004 » est la loi 2017/015.
    """
    numero = numero_du_titre(titre) or numero_de_l_acte(titre)
    premier = re.match(r"\s*([a-z]+)", _sans_accents(titre or ""))
    if not numero or premier is None or premier.group(1) not in _NATURES_NUMEROTEES:
        return None
    return _NATURES_NUMEROTEES[premier.group(1)], forme_du_numero(numero)


# « Article 39 of the Labour Code », « Article 2 of Law No. 2013/004 » :
# parse_reference ne lit que la forme francaise (« de », « du »).
_ARTICLE_ANGLAIS = re.compile(
    r"\bart(?:icle|\.)?\s*(\d+(?:[-.]\d+)*|first)\s+of\s+(.+)", re.IGNORECASE
)
# Ce qui suit le nom d'un texte dans une question : « du Code du travail SUR le
# licenciement », « ... POUR un CDD ». Le nom est essaye entier d'abord, puis
# coupe ici : « portant sur le droit commercial général » est un nom.
_SUITE_DU_NOM = re.compile(
    r"\s+(?:sur|pour|concernant|lorsque|lorsqu|quand|si|s'il|qui|que|quant|en\s+cas|"
    r"en\s+matière|à\s+propos|au\s+sujet|dans\s+le\s+cas|et\s+l['’]article|et\s+les?\b|"
    r"say|says|provide|provides|state|states|stipulate|stipulates|mean|means|on|about|for)\b",
    re.IGNORECASE,
)


def reference_d_article(question: str) -> Optional[Tuple[str, List[str]]]:
    """
    (numero normalise, noms du texte a essayer) quand la question nomme un
    article ET un texte, sinon None.

    Les noms : celui de la question, nettoye, puis le meme coupe a ce qui
    suit le nom (_SUITE_DU_NOM). Le texte qu'ils designent, s'il y en a un,
    est cherche ensuite : TEXTES_NOMMABLES ou numero propre.

    Un RENVOI au texte de la conversation (« de ce code », « du même code »,
    « de la loi », « of this law ») n'est pas un nom : None, comme pour
    « l'article 38 » seul (voir renvoi_au_texte).
    """
    trouve = _article_et_noms(question)
    if trouve is None or _noms_de_renvoi(trouve[1]):
        return None
    return trouve


def _article_et_noms(question: str) -> Optional[Tuple[str, List[str]]]:
    """(numero normalise, noms qui suivent l'article) ; renvoi compris (voir reference_d_article)."""
    reference = parse_reference(question or "")
    if reference is not None and reference.has_hint:
        numero, nom = reference.number, reference.doc_hint
    else:
        anglais = _ARTICLE_ANGLAIS.search(question or "")
        if anglais is None:
            return None
        numero, nom = normalize_number(anglais.group(1)), anglais.group(2)
    nom = re.split(r"[?!;,:(]", nom)[0].strip(" .'’\"«»")
    noms = [nom] if nom else []
    coupe = _SUITE_DU_NOM.split(nom, maxsplit=1)[0].strip(" .'’\"«»")
    if coupe and coupe != nom:
        noms.append(coupe)
    if not numero or not noms:
        return None
    return numero, noms


# ---------------------- L'ARTICLE D'UN TEXTE DEJA NOMME ----------------------
#
# « Que dit l'article 38 de ce code ? », « … du même code », « … dudit code »,
# « … de cette loi », « … de la loi », « What does Article 40 of this law
# say? », « And Article 40 of the same code? » : l'article est nomme avec un
# RENVOI au texte dont parle la conversation, et non avec un texte. Revue du
# 30/10/2026 (tour 4) : reference_d_article y voyait un nom (« ce code »), que
# la table ne resout pas. La question etait cherchee SEULE, sans texte herite
# : apres une question sur le licenciement, le modele lisait l'article 38 ou
# 40 de six a huit textes sans rapport (Code electoral, Code de la
# nationalite, loi sur la concurrence...), aucun extrait du Code du travail ;
# « dudit code » remontait meme en tete l'article 38 du Code de la
# nationalite, par son seul numero (_pin_requested_article). C'est une
# RELANCE, exactement comme « Et l'article 38 ? » : cherchee jointe au sujet,
# l'article lu dans le texte du tour precedent (_article_de_la_relance),
# sinon rien ; jamais un article choisi par son seul numero.
#
# Un nom de renvoi ne porte qu'une designation (« ce », « même », « ledit »,
# « this », « the same »...) ou un article defini (« la loi », « the code »),
# et UNE nature de texte : rien d'autre. « la loi sur le terrorisme », « le
# Code pénal », « la loi de 2024 » nomment un texte, et restent ce qu'ils
# etaient (la table, ou rien).
_DESIGNATIONS_DE_RENVOI = frozenset(
    "ce cet cette ces ledit ladite lesdits lesdites dudit desdits desdites audit auxdits "
    "auxdites meme memes susdit susdite susdits susdites precite precitee precites "
    "precitees susvise susvisee susvises susvisees presente "
    "this that these those same said aforesaid aforementioned present".split()
)
_ARTICLES_DEFINIS = frozenset("le la les l the".split())
_NATURES_DE_TEXTE = frozenset(
    "code codes loi lois texte textes decret decrets acte actes ordonnance ordonnances "
    "law laws act acts text texts decree decrees ordinance ordinances".split()
)
# Ce qui suit le nom sans en faire partie : « Article 40 of this law SAY? »,
# « … of the code PROVIDE on notice? », « l'article 40 de ce code DIT que… ».
# Le nom s'arrete au premier de ces verbes (_mot_de_nature_du_nom) : sans cela, « the
# code provide on notice » n'etait pas un renvoi, et la question etait
# cherchee seule (l'article 40 de huit textes sans rapport, revue du tour 4).
# Pas « said » : « the said Act » est une designation.
_VERBES_APRES_LE_NOM = frozenset(
    "say says provide provides state states stipulate stipulates mean means "
    "dit disent prevoit prevoient dispose disposent precise precisent stipule stipulent "
    "enonce enoncent indique indiquent".split()
)
# Les mots d'un renvoi : ni un nom de texte, ni un sujet (mots_propres).
_MOTS_D_UN_RENVOI = _DESIGNATIONS_DE_RENVOI | _NATURES_DE_TEXTE | _VERBES_APRES_LE_NOM
# « dudit code », « desdites lois » : parse_reference n'y lit pas
# l'introduction d'un nom (« du », « de »), la question n'a donc pas de nom.
# Lu sur la question sans accents ; le groupe est le mot de nature.
_RENVOI_CONTRACTE_AU_TEXTE = re.compile(
    r"\b(?:dudit|desdits|desdites|audit|auxdits|auxdites)\s+"
    r"((?:code|loi|texte|decret|acte|ordonnance)s?)\b"
)


def _mot_de_nature_du_nom(nom: str, coupe: bool) -> Optional[str]:
    """
    Le mot de nature de `nom` (« code », « decret », « law »...) quand il
    renvoie au texte de la conversation sans en nommer aucun ; None sinon.
    Un nom COUPE a ce qui le suit (_SUITE_DU_NOM : « la loi | sur le
    terrorisme ») ne l'est qu'avec une designation (« ce code | sur le
    préavis ») : un article defini seul y nommerait un texte par sa suite.
    Un verbe (_VERBES_APRES_LE_NOM) clot le nom : « the law | say about
    notice » est un renvoi, comme « this law say ».
    """
    mots = _mots(nom)
    verbe = next((i for i, m in enumerate(mots) if m in _VERBES_APRES_LE_NOM), None)
    if verbe is not None:
        mots = mots[:verbe]
    natures = [m for m in mots if m in _NATURES_DE_TEXTE]
    if len(natures) != 1:
        return None
    if any(m not in _DESIGNATIONS_DE_RENVOI | _ARTICLES_DEFINIS | _NATURES_DE_TEXTE for m in mots):
        return None
    if coupe and not any(m in _DESIGNATIONS_DE_RENVOI for m in mots):
        return None
    return natures[0]


def _mot_de_nature_des_noms(noms: List[str]) -> Optional[str]:
    """Les noms d'un article (_article_et_noms) : le mot de nature du nom entier, ou coupe, qui est un renvoi."""
    if not noms:
        return None
    return _mot_de_nature_du_nom(noms[0], coupe=False) or next(
        (mot for mot in (_mot_de_nature_du_nom(nom, coupe=True) for nom in noms[1:]) if mot), None
    )


def _noms_de_renvoi(noms: List[str]) -> bool:
    """Les noms d'un article (_article_et_noms) : le nom entier, ou coupe, est un renvoi."""
    return _mot_de_nature_des_noms(noms) is not None


def mot_de_nature_du_renvoi(question: str) -> Optional[str]:
    """
    Le mot de nature du renvoi au texte de la conversation (renvoi_au_texte),
    sans accents : « decret » pour « l'article 2 de ce décret », « code »
    pour « dudit code », « law » pour « of this law », « texte » pour « de ce
    texte ». None si la question n'en porte pas.
    """
    trouve = _article_et_noms(question)
    if trouve is not None:
        return _mot_de_nature_des_noms(trouve[1])
    if not _NUMERO_D_ARTICLE.search(question or ""):
        return None
    contracte = _RENVOI_CONTRACTE_AU_TEXTE.search(_sans_accents(question))
    return contracte.group(1) if contracte else None


def renvoi_au_texte(question: str) -> bool:
    """
    La question nomme-t-elle un article avec un RENVOI au texte de la
    conversation (« l'article 38 de ce code », « du même code », « dudit
    code », « de cette loi », « de la loi », « of this law », « of the same
    code ») ? Alors elle ne nomme aucun texte : reference_d_article rend None,
    numero_sans_texte le numero, et c'est une relance (est_une_relance).
    """
    return mot_de_nature_du_renvoi(question) is not None


# ---------------------- LA NATURE D'UN RENVOI ----------------------
#
# « Que dit l'article 2 de ce décret ? », dans une conversation qui nomme le
# Code du travail et dont la reponse parle du decret n° 2024/056 : le renvoi
# porte une NATURE, que n'a pas le texte de la conversation. Revue du tour 5
# (sonde_renvoi_nature, sonde_flux_decret) : le renvoi heritait du Code du
# travail quelle que soit sa nature, et l'article 2 de ce code (« Le droit au
# travail est reconnu… ») passait en tete comme article nomme, pertinence
# 1,00 ; une reponse qui l'attribuait au decret etait RATTACHEE a l'article 2
# du decret, qui dit en realite « Sont abrogées toutes dispositions
# antérieures contraires ». Et « décret », retire des mots-cles avec le
# renvoi, ne faisait plus trouver le decret : « Et l'article 1er dudit
# décret ? » ne le lisait plus (il etait lu au rang 2 au tour 4). Le cas est
# courant : 1 702 des 2 275 textes publies sont des decrets, et les reponses
# sur un code renvoient souvent a un decret.
#
# Un renvoi qui porte une nature n'herite donc du texte de la conversation
# que si ce texte est DE CETTE NATURE, lue dans son intitule comme le fait la
# lecture des citations (lecture_des_citations._natures_propres) : la nature
# qui l'ouvre (décret, loi, ordonnance, arrêté, code, acte uniforme, traité,
# constitution), et celle de l'instrument qu'il porte (« Loi … portant Code
# du travail » est une loi ET un code : « ce code » et « cette loi » le
# designent tous deux ; « Loi … modifiant … la loi … portant code électoral »
# n'est pas un code). Sinon il n'herite de rien, et il est cherche comme toute
# relance, joint au sujet, son mot de nature garde dans les mots-cles
# (RAGService._une_recherche).
#
# POURQUOI AUSSI L'INSTRUMENT PORTE, et pas la seule tete de l'intitule : 13
# des 16 codes publies sont portes par une loi ou un decret (« Loi n° 92/007
# … portant Code du travail ») ; sur la seule tete, « ce code » n'heriterait
# plus du Code du travail, et les 9 renvois des jeux qui portent une nature
# (« ce code », « cette loi », « of this law »...) perdraient l'article qu'ils
# demandent.
#
# Mesure du tour 6 (sondes de la revue, jurix_dev) : plus aucun article du
# Code du travail en tete pour « de ce décret », « dudit décret », « du
# décret », « of that decree » ; le decret n° 2024/056 lu aux rangs 3 et 4
# pour les deux premiers (au tour 5 : son seul article 2, au rang 7), et la
# reponse juste « l'article 1er du décret n° 2024/056 » de nouveau
# rattachee. « texte » et « acte » ne disent pas de nature : « ce texte »
# designe tout texte, « the said Act » une loi en anglais, « ledit acte » un
# acte uniforme comme un autre acte.
_NATURE_DU_MOT_DE_RENVOI = {
    "code": "code", "codes": "code",
    "loi": "loi", "lois": "loi", "law": "loi", "laws": "loi",
    "decret": "decret", "decrets": "decret", "decree": "decret", "decrees": "decret",
    "ordonnance": "ordonnance", "ordonnances": "ordonnance",
    "ordinance": "ordonnance", "ordinances": "ordonnance",
}


def nature_du_renvoi(question: str) -> Optional[str]:
    """
    La NATURE du texte que designe le renvoi de la question, sous la forme
    des natures d'un intitule (lecture_des_citations._natures_propres) :
    « decret » pour « l'article 2 de ce décret » ou « … of that decree »,
    « loi » pour « de cette loi » ou « of this law », « code » pour « du même
    code ». None sans renvoi, ou pour un renvoi sans nature (« de ce texte »,
    « of the said Act »).
    """
    return _NATURE_DU_MOT_DE_RENVOI.get(mot_de_nature_du_renvoi(question) or "")


def de_la_nature(intitule: Optional[str], nature: str) -> bool:
    """L'intitule d'un texte dit-il cette nature (lecture_des_citations._natures_propres) ?"""
    return isinstance(intitule, str) and nature in _natures_propres(normaliser_le_nom(intitule))


# ---------------------- LES RELANCES : DE QUOI PARLE-T-ON ? ----------------------
#
# Une RELANCE continue la conversation sans nommer son propre sujet : « Peux-tu
# détailler ? », « Et l'article 38 ? », « Combien ça coûte ? », « What about
# the notice? ». Cherchee seule, elle ramene n'importe quoi. Revue du
# 29/10/2026 (tour 3), apres « Quelles indemnités en cas de licenciement abusif
# ? » : « Et l'article 38 ? » passait la barriere seule (0,486) et le modele
# lisait l'article 38 de huit textes sans rapport (concurrence, education, loi
# de finances...) ; « Quel article exactement ? », sept articles du Code CIMA ;
# « Combien ça coûte ? », huit extraits de lois de finances. Et la recherche
# jointe, qui ne jouait qu'apres un refus, joignait le DERNIER message : une
# seconde relance de suite etait jointe a la premiere (« Peux-tu détailler ?
# Donne-moi un exemple concret », 0,186) et refusee, 6 sur 6.
#
# Une relance est donc cherchee JOINTE AU SUJET de la conversation : la
# derniere question de fond de l'utilisateur (pas une relance), et les
# relances qui l'ont suivie quand elles ajoutent un mot a elles (« Et pour un
# CDD ? »). Une question qui a son propre sujet est cherchee seule, comme
# avant. Sans modele : quelques mots, des regles, et une mesure
# (scripts/eval/evaluer_contexte_chat.py : relances, relances en chaine, et
# AUCUNE question de fond des jeux prise pour une relance).
#
# Mais une relance OUVERTE (« Et », « Mais », « What about ») qui apporte ses
# propres mots peut changer de sujet (« Et la nationalité ? ») : revue du tour
# 4, jointe seulement, elle lisait les textes de l'ancien sujet a la place des
# siens. Elle est cherchee seule ET jointe, les deux listes alternees
# (peut_changer_de_sujet, RAGService._chercher_seule_et_jointe) ; la relance
# sans mot propre qui la suit (« Peux-tu détailler ? ») relit aussi la fin du
# sujet, seule (SujetDeLaConversation.depuis_un_changement_possible).

# Les mots qui ne font pas un sujet : demander une suite ou une precision
# (detailler, un exemple, l'article, le montant, le delai, la procedure, les
# sanctions, saisir « le tribunal »...) sans dire de quoi, et les mots d'une
# conversation (merci, bonjour). Sans accents. Une question qui n'a QUE ces
# mots, et des mots vides, n'a pas de sujet propre : « Quel est le délai pour
# saisir le tribunal ? », apres une question sur le licenciement, parle du
# tribunal du travail, et cherchee seule elle lisait d'autres procedures.
_MOTS_SANS_SUJET = frozenset("""
    detailler detaille detaillez details detail developper developpe developpez
    preciser precise precisez precision precisions approfondir eclaircir
    expliquer explique expliquez explication explications simplement simple
    simplifier clairement clair claire mieux encore autrement exactement precisement
    exemple exemples concret concrete concrets concretes illustrer
    article articles alinea alineas texte textes loi lois disposition dispositions
    reference references source sources citer cite citez quote
    resumer resume resumez resumes reformuler reformule reformulez repeter repete
    reponse reponses difference differences rapport lien
    peux peut peuvent pouvez puis pourrais pourriez veux voudrais voulez dois doit
    devez faut fallait faudrait sais savez savoir
    maintenant ensuite apres alors aussi ici ceci cela celui celle ceux celles
    contacter adresser joindre aller voir
    suite suivant suivante suivants suivantes precedent precedente
    vigueur toujours applicable applicables valable valables valide actuel actuelle
    montant montants somme sommes cout couts coute coutent couter coutera prix
    tarif tarifs frais payer combien
    temps duree durees dure durer delai delais jours mois ans annees
    procedure procedures demarche demarches etape etapes faire suivre plait
    sanction sanctions peine peines risque risques risquer encourt encourue
    condition conditions cas situation question questions chose choses
    droit droits recours possibilite possibilites option options solution solutions
    saisir saisine tribunal tribunaux juridiction juridictions
    prevu prevue prevus prevues prevoit
    merci bonjour bonsoir salut coucou super parfait genial cool okay accord vrai
    beaucoup aide aider compris comprends entendu
    more detail details elaborate explain explanation clarify simply simpler
    example examples concrete instance article articles section provision provisions
    law laws text texts source sources cite quote reference exactly precisely
    summarize summarise summary rephrase reword repeat answer difference
    now next then still also there here force valid applicable current
    amount amounts sum cost costs price fee fees much many pay
    long time period periods deadline deadlines days months years
    procedure procedures step steps process
    penalty penalties sanction sanctions punishment punishments risk risks
    condition conditions case situation question thing things
    right rights remedy remedies option options solution solutions
    contact provided provide provides
    thanks thank hello please yes okay great perfect true
""".split())
# Les ouvertures d'une suite : « Et ... ? », « Mais si... », « What about... ».
_OUVERTURES_DE_RELANCE = (
    ("et",), ("and",), ("mais",), ("but",), ("ok", "et"), ("okay", "and"),
    ("what", "about"), ("how", "about"), ("what", "if"), ("and", "what", "about"),
)
# Un renvoi a ce qui precede : « cette décision », « ce délai », « this decision ».
_RENVOI = re.compile(
    r"\b(?:cette|cet|ces|ce)\s+(?:decision|article|texte|loi|cas|delai|montant|somme|procedure"
    r"|sanction|peine|regle|disposition|situation|jugement|indemnite|contrat|droit|infraction"
    r"|amende|mesure|condamnation|plainte|recours)s?\b"
    r"|\b(?:this|that|these|those)\s+(?:decision|article|law|text|rule|case|provision|amount"
    r"|procedure|penalty|sanction|judgment|judgement|contract|right|offence|offense|fine"
    r"|measure|sentence|complaint|remedy)s?\b"
)
# Au plus autant de mots propres pour qu'une question ouverte par « Et »
# reste une relance (« Et si l'employeur refuse de payer ? » : 3), ou qu'une
# question qui renvoie a ce qui precede en soit une (« Peut-on faire appel de
# cette décision ? » : 2). Au-dela, elle a son sujet : « Et quelles sont les
# conditions pour acquérir la nationalité camerounaise par mariage ? » ;
# « Combien l'État avait-il prévu pour les CAS en 2023, et comment ce montant
# se répartit-il ? », ou « ce montant » renvoie a la question elle-meme.
MOTS_PROPRES_D_UNE_RELANCE = 3
MOTS_PROPRES_D_UN_RENVOI = 2
# Le numero d'un article nomme SANS son texte : « Et l'article 38 ? ».
_NUMERO_D_ARTICLE = re.compile(
    r"\bart(?:icle|\.)?\s*(\d+(?:[-.]\d+)*|premi[eè]re?|1[eè]re?|first)\b", re.IGNORECASE
)
_STOPWORDS_SANS_ACCENTS = frozenset(_sans_accents(m) for m in STOPWORDS)


def _mots(texte: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _sans_accents(texte))


def mots_propres(question: str) -> List[str]:
    """
    Les mots qui font le sujet d'une question : ni mots vides, ni mots d'une
    relance (_MOTS_SANS_SUJET), ni nombres, ni ceux d'un renvoi au texte de la
    conversation (« l'article 40 du même code » : « meme », « code » ne sont
    pas un sujet). Sans accents, dans l'ordre.
    """
    mots = _mots(question)
    if renvoi_au_texte(question):
        mots = [m for m in mots if m not in _MOTS_D_UN_RENVOI]
    return [
        m for m in mots
        if len(m) > 2 and not m.isdigit() and m not in _STOPWORDS_SANS_ACCENTS
        and m not in _MOTS_SANS_SUJET
    ]


def est_une_relance(question: str) -> bool:
    """
    La question continue-t-elle la conversation sans nommer son propre sujet ?
    - un article ET son texte (« l'article 38 du Code pénal »), ou un texte
      par son numero (« la loi n° 2013/004 ») : non, elle a son sujet ;
    - un article avec un renvoi au texte de la conversation (« l'article 40
      du même code », « of this law ») : oui, comme « Et l'article 40 ? »
      (renvoi_au_texte) ;
    - aucun mot propre (« Peux-tu détailler ? », « Et l'article 38 ? »,
      « Combien ça coûte ? », « Can you explain more? ») : oui ;
    - ouverte par « Et », « Mais », « What about »... avec au plus
      MOTS_PROPRES_D_UNE_RELANCE mots propres (« Et pour un CDD ? »), ou qui
      renvoie a ce qui precede avec au plus MOTS_PROPRES_D_UN_RENVOI
      (« Peut-on faire appel de cette décision ? ») : oui. Ouverte et munie
      de ses mots, elle peut aussi changer de sujet : peut_changer_de_sujet ;
    - sinon non : « divorce », « Combien de temps dure le préavis ? » ont leur
      sujet, et sont cherchees seules.
    """
    mots = _mots(question)
    if not mots or reference_d_article(question) is not None or numero_du_nom(question):
        return False
    if renvoi_au_texte(question):
        return True
    propres = mots_propres(question)
    if not propres:
        return True
    ouverte = any(tuple(mots[:len(o)]) == o for o in _OUVERTURES_DE_RELANCE)
    if ouverte and len(propres) <= MOTS_PROPRES_D_UNE_RELANCE:
        return True
    return _renvoie_a_ce_qui_precede(mots, propres)


def _renvoie_a_ce_qui_precede(mots: List[str], propres: List[str]) -> bool:
    """« Peut-on faire appel de cette décision ? » : un renvoi (_RENVOI), et peu de mots propres."""
    return len(propres) <= MOTS_PROPRES_D_UN_RENVOI and bool(_RENVOI.search(" ".join(mots)))


def peut_changer_de_sujet(question: str) -> bool:
    """
    Une relance ouverte par « Et », « Mais », « What about », « But what
    if »... qui apporte ses PROPRES mots (« Et la nationalité ? », « What
    about inheritance? », « Et pour un CDD ? ») : une suite de la
    conversation, ou un NOUVEAU sujet. Les mots seuls ne permettent pas de
    trancher : elle est cherchee seule ET jointe au sujet
    (RAGService._chercher_seule_et_jointe).

    POURQUOI. Revue du 30/10/2026 (tour 4) : jointe seulement, « Et la
    nationalité ? », apres une question sur le licenciement, lisait 0 extrait
    sur 8 du Code de la nationalite (8 sur 8 cherchee seule), et 7 changements
    de sujet sur 13 perdaient nettement le texte de leur sujet ; le modele
    repondait de memoire.

    Non : une relance sans mot propre (« Et l'article 38 ? », « Combien ça
    coûte ? ») ; un renvoi a ce qui precede (« Et cette décision ? »,
    « l'article 40 du même code ») ; une relance qui parle d'un article sans
    nommer de texte (« Et l'article suivant du code ? », « Et l'article 38
    sur la faute lourde ? ») : c'est un article du texte de la conversation,
    et cherchee seule elle lirait n'importe quel article de n'importe quel
    code en tete ; une question qui n'est pas une relance. Celles-la parlent
    de la conversation : jointes au sujet, comme avant.
    """
    if not est_une_relance(question) or renvoi_au_texte(question):
        return False
    if _MENTION_D_ARTICLE.search(question or ""):
        return False
    propres = mots_propres(question)
    return bool(propres) and not _renvoie_a_ce_qui_precede(_mots(question), propres)


# « l'article », « les articles », « art. 38 », « art 38 » : une question qui
# en parle sans nommer de texte parle du texte de la conversation.
_MENTION_D_ARTICLE = re.compile(r"\barticles?\b|\bart\.|\bart\s*\d", re.IGNORECASE)


def numero_sans_texte(question: str) -> Optional[str]:
    """
    Le numero d'un article que la question nomme SANS son texte (« Et
    l'article 38 ? », « l'article 40 du même code » : un renvoi n'est pas un
    texte), normalise ; None s'il n'y en a pas, ou si un texte est nomme avec
    lui (_article_nomme s'en charge).
    """
    if reference_d_article(question) is not None:
        return None
    trouve = _NUMERO_D_ARTICLE.search(question or "")
    return normalize_number(trouve.group(1)) if trouve else None


@dataclass(frozen=True)
class SujetDeLaConversation:
    """
    Le sujet d'une relance : la derniere question de fond de l'utilisateur,
    et les relances qui l'ont suivie avec des mots a elles. `questions` les
    porte dans l'ordre de la conversation.
    """

    questions: Tuple[str, ...]

    @property
    def texte(self) -> str:
        return " ".join(self.questions)

    def depuis_un_changement_possible(self) -> Optional["SujetDeLaConversation"]:
        """
        La fin du sujet, depuis sa derniere relance qui pouvait changer de sujet
        (peut_changer_de_sujet : « Et la nationalité ? »), la question de fond
        exclue ; None s'il n'y en a pas. Voir RAGService._retrieve_chunks.
        """
        for rang in range(len(self.questions) - 1, 0, -1):
            if peut_changer_de_sujet(self.questions[rang]):
                return SujetDeLaConversation(self.questions[rang:])
        return None


# Au plus autant de relances intermediaires gardees dans le sujet.
RELANCES_DU_SUJET = 2


def sujet_de_la_conversation(history: Optional[List[Any]]) -> Optional[SujetDeLaConversation]:
    """
    Le sujet de la conversation, d'apres les messages de l'utilisateur de
    `history` (du plus ancien au plus recent) : la derniere question qui
    n'est pas une relance, puis les relances qui l'ont suivie et qui ajoutent
    des mots propres (« Et pour un CDD ? » ; pas « Peux-tu détailler ? »).
    None sans question de fond.
    """
    questions = [
        (getattr(m, "content", None) or "").strip() for m in (history or [])
        if getattr(m, "role", None) == "user" and (getattr(m, "content", None) or "").strip()
    ]
    relances: List[str] = []
    for question in reversed(questions):
        if est_une_relance(question):
            if mots_propres(question):
                relances.insert(0, question)
            continue
        return SujetDeLaConversation((question,) + tuple(relances[-RELANCES_DU_SUJET:]))
    return None


def textes_de_la_table_dans(texte: str) -> Set[str]:
    """
    Les textes de TEXTES_NOMMABLES que `texte` nomme, quelque part : « selon le
    Code du travail », « the Labour Code ». « constitution » suivi de « de »,
    « d' », « du », « des » n'est pas la Constitution (« constitution d'une
    société », « constitution de partie civile »).
    """
    mots = _mots(texte)
    if not mots:
        return set()
    trouves: Set[str] = set()
    for nommable in TEXTES_NOMMABLES:
        for nom in nommable.noms:
            cible = _mots(nom)
            n = len(cible)
            for i in range(len(mots) - n + 1):
                if mots[i:i + n] != cible:
                    continue
                if cible == ["constitution"] and i + 1 < len(mots) and \
                        mots[i + 1] in ("de", "d", "du", "des", "of"):
                    continue
                trouves.add(nommable.reference)
                break
    return trouves


# L'article demande dans UNE loi, par trois lectures, de la plus sure a la
# moins sure : le numero exact ; ses morceaux (« 35.1 » : un article long
# decoupe) ; l'en-tete « Article 37.- » DANS un morceau qui en porte plusieurs
# (le Code du travail du corpus range les articles 35 a 39 sous « 35.1 »). Un
# renvoi (« l'article 37 ci-dessus ») n'est pas un en-tete : majuscule, et un
# signe d'en-tete apres le numero. Assez de lignes pour voir un doublon
# (RAGService._sans_doute).
_SQL_ARTICLE_NOMME = text("""
    SELECT a.id AS article_id, a.number, a.title AS article_title, a.section,
           a.page_number, a.content, a.law_id, a."order" AS ordre, l.reference,
           l.title AS law_title, l.type, l.language, l.status, l.category_id,
           l.publication_date,
           CASE
               WHEN upper(btrim(a.number, ' .:-')) = :numero THEN 0
               WHEN upper(a.number) LIKE :morceau ESCAPE '\\' THEN 1
               ELSE 2
           END AS lecture
    FROM articles a
    JOIN laws l ON l.id = a.law_id
    WHERE a.law_id = :law_id
      AND (
          upper(btrim(a.number, ' .:-')) = :numero
          OR upper(a.number) LIKE :morceau ESCAPE '\\'
          OR (:entete <> '' AND a.content ~ :entete)
      )
    ORDER BY lecture, a."order", a.id
    LIMIT 8
""")

# Les textes publies, pour reconnaitre un numero propre (numero_de_l_intitule).
# Une lecture de quelques millisecondes, seulement quand la question nomme un
# texte par son numero.
_SQL_TEXTES_PUBLIES = text("SELECT id, reference, title FROM laws WHERE status = 'published'")


# Les articles, parmi des identifiants, qui portent TOUS les mots-cles : la
# requete du premier essai du plein texte (search_articles_pg), avant ses
# replis. Seulement si les mots-cles font au moins DEUX lexemes francais
# distincts : un seul (« cas », « prime ») est dans des milliers d'articles et
# ne prouve rien (revue du 29/10/2026 : « Et dans mon cas ? » faisait lire
# au modele un decret d'adhesion a des conventions sur l'apatridie).
_SQL_ARTICLES_STRICTS = text("""
    SELECT a.id
    FROM articles a
    WHERE a.id = ANY(:ids)
      AND coalesce(array_length(tsvector_to_array(to_tsvector('french', :requete)), 1), 0) >= 2
      AND (a.search_vector @@ websearch_to_tsquery('french', :requete)
           OR a.search_vector @@ websearch_to_tsquery('english', :requete))
""")


# « LIRE AUSSI L'ARTICLE RENVOYE ». Un article qui ne dit sa peine que par renvoi
# (« Les peines de l'article 318 ci-dessus sont doublées », Code penal art. 320 ;
# « Est puni des peines prévues à l'alinéa 1 », cybersecurite art. 65.2) arrive en
# tete sans la regle : le modele lisait le renvoi, puis le vol d'effets militaires
# d'un autre texte, jamais l'art. 318. L'article vise est lu juste apres celui qui
# le vise. Borne : les RANGS_DES_RENVOIS premiers extraits, RENVOIS_MAX lectures
# (une requete, aucun appel au modele), le MEME texte, les seules peines : un
# renvoi ordinaire (« dans les conditions de l'article 12 ») ne deplace rien.
RANGS_DES_RENVOIS = 4
RENVOIS_MAX = 2
SOURCE_ARTICLE_RENVOYE = "article_renvoye"


def _place_de_l_article_renvoye(extraits: List[ChunkResult], renvoi: ChunkResult) -> int:
    """
    Ou lire l'article que vise `renvoi` : juste apres le dernier extrait du
    MEME texte qui le precede, donc devant les autres textes lus entre les
    deux ; juste apres lui sinon (voir RAGService._avec_les_renvois).
    """
    rang = next(i for i, c in enumerate(extraits) if c.fusion_key == renvoi.fusion_key)
    meme_texte = [i for i in range(rang) if extraits[i].law_id == renvoi.law_id]
    # Sans extrait du meme texte avant lui, l'ordre de la recherche tient :
    # les autres textes y sont tous mieux classes que le texte du renvoi.
    if not meme_texte or meme_texte[-1] == rang - 1:
        return rang + 1
    return meme_texte[-1] + 1
_RENVOI_A_UN_ARTICLE = re.compile(
    r"\bpeines?\b[^.;:\n]{0,40}?\bl[’']article\s+(\d+)(?![\d-])", re.IGNORECASE
)
_RENVOI_A_UN_ALINEA = re.compile(
    r"\bpeines?\s+pr[ée]vues?\s+(?:à|a|par)\s+l[’']alin[ée]a\s+(?:\(?(\d+)\)?|premier)"
    r"(?!\s+de\s+l)", re.IGNORECASE,
)
_SQL_ARTICLE_RENVOYE = text("""
    SELECT a.id AS article_id, a.number, a.title AS article_title, a.section,
           a.page_number, a.content, a.law_id, l.reference, l.title AS law_title,
           l.type, l.language, l.status, l.category_id, l.publication_date
    FROM articles a
    JOIN laws l ON l.id = a.law_id
    WHERE a.law_id = :law_id AND a.number = :numero AND l.status = 'published'
    ORDER BY a."order"
    LIMIT 1
""")


def numero_renvoye(extrait: ChunkResult) -> Optional[str]:
    """
    Le numero que cet extrait vise pour sa peine, tel qu'il est en base ; None
    sans renvoi, ou s'il vise son propre article. Un alinea vise le morceau de
    meme rang du meme article : « alinéa 1 » de 65.2 est 65.1.
    """
    numero = str(extrait.number or "")
    contenu = extrait.content or ""
    alinea = _RENVOI_A_UN_ALINEA.search(contenu)
    if alinea and re.fullmatch(r"\d+\.\d+", numero):
        vise = f"{numero.split('.')[0]}.{alinea.group(1) or 1}"
        return vise if vise != numero else None
    article = _RENVOI_A_UN_ARTICLE.search(contenu)
    if article and article.group(1) != numero.split(".")[0]:
        return article.group(1)
    return None


def _sans_doublons(extraits: List[ChunkResult]) -> List[ChunkResult]:
    """Les extraits dans leur ordre, chacun une fois (cle de fusion)."""
    vus: Set[tuple] = set()
    resultat = []
    for extrait in extraits:
        if extrait.fusion_key not in vus:
            vus.add(extrait.fusion_key)
            resultat.append(extrait)
    return resultat


def _entrelaces(premiers: List[ChunkResult], seconds: List[ChunkResult]) -> List[ChunkResult]:
    """
    Les deux listes ALTERNEES, chacune dans son ordre : le 1er de `premiers`,
    le 1er de `seconds`, le 2e de `premiers`, le 2e de `seconds`... Un extrait
    present dans les deux n'est garde qu'a sa premiere place (_sans_doublons) ;
    la plus longue finit seule. Voir RAGService._chercher_seule_et_jointe.
    """
    return _sans_doublons([
        extrait for paire in zip_longest(premiers, seconds) for extrait in paire
        if extrait is not None
    ])


def _motif_d_en_tete(numero: str) -> str:
    """Expression reguliere PostgreSQL de l'en-tete « Article 37.- », '' si le numero n'est pas simple."""
    if not numero.isdigit():
        return ""
    ordinal = "(premier|PREMIER|1er)|" if numero == "1" else ""
    return (
        r"(^|[^[:alnum:]'’])(Article|ARTICLE|Art\.)[[:space:]]+"
        rf"({ordinal}{numero}(er)?)[[:space:]]*(\.[[:space:]]*[-–—]|[-–—:]|\.[[:space:]]*\()"
    )


# Au plus autant de morceaux d'un article nomme ajoutes au contexte : un
# article tres long ne doit pas en chasser tout le reste.
MORCEAUX_DE_L_ARTICLE_NOMME = 4

# Les chemins du volet semantique servis par l'index HNSW, que le chat ne
# croit pas seul (RAGService._retenir_le_pertinent) : le repli d'une recherche
# exacte trop lente, et l'index lui-meme (la mesure, /search).
VOLETS_SERVIS_PAR_L_INDEX = (CHEMIN_REPLI_HNSW, CHEMIN_HNSW)


@dataclass
class EtatDeLaRecuperation:
    """
    Ce que la recuperation du chat a decide pour la question en cours : le
    journal le dit, la mesure le compte (scripts/eval/evaluer_contexte_chat.py).
    """

    langue: Optional[str] = None
    chemin: Optional[str] = None
    similarite: Optional[float] = None
    seuil: Optional[float] = None
    # « sain » : le volet semantique suffit ; « faible » : seuls les textes
    # confirmes restent ; « sans volets » : recherche doublee, rien a juger.
    mode: Optional[str] = None
    # La question porte sur le budget : les tableaux de programmes restent.
    budgetaire: bool = False
    tableaux_exclus: int = 0
    # Le contexte de la recherche hybride, avant les regles du chat ; puis le
    # contexte JUGE (article nomme en tete) et l'article nomme lui-meme.
    hybride: List[ChunkResult] = field(default_factory=list)
    juge: List[ChunkResult] = field(default_factory=list)
    nommes: List[ChunkResult] = field(default_factory=list)
    article_nomme: Optional[str] = None
    # Les mots-cles de la confirmation stricte (_textes_confirmes) : ceux de
    # la question, ou ceux d'une relance et de son sujet ; '' s'ils ne font
    # pas deux mots.
    mots_cles: str = ""
    stricts: Set[int] = field(default_factory=set)
    ajouts: List[str] = field(default_factory=list)
    refus: bool = False
    # La question cherchee JOINTE au sujet de la conversation : une relance
    # (est_une_relance), ou une question refusee seule
    # (RAGService._chercher_la_relance). `relance` : jointe des la premiere
    # recherche ; `sujet` : ce qui a ete joint ; `texte_herite` : le texte du
    # tour precedent, pour un article nomme sans son texte (« Et l'article 38
    # ? »).
    question_jointe: Optional[str] = None
    relance: bool = False
    sujet: Optional[str] = None
    texte_herite: Optional[str] = None
    # Une relance qui peut changer de sujet (peut_changer_de_sujet), cherchee
    # SEULE puis JOINTE (_chercher_seule_et_jointe) : les deux etats, pour le
    # journal et la mesure. `jointe_ecartee` : refusee seule, rien de la
    # recherche jointe ne portait ses mots ni ne figurait parmi ses voisins,
    # la recherche jointe a ete ecartee.
    seule_et_jointe: bool = False
    seule: Optional["EtatDeLaRecuperation"] = None
    jointe: Optional["EtatDeLaRecuperation"] = None
    jointe_ecartee: bool = False
    # La relance sans mot propre qui suit un changement de sujet possible : sa
    # recherche « seule » est celle de la fin du sujet, ce texte-ci, cherche
    # seul (SujetDeLaConversation.depuis_un_changement_possible).
    sujet_recent: Optional[str] = None


# Une forme de mot comparable : sans accents, sans la marque du pluriel
# (« indemnités » et « indemnité »). Pour la corroboration d'une recherche
# jointe par les mots propres de la question (_corrobores).
def _forme_du_mot(mot: str) -> str:
    return mot[:-1] if len(mot) > 3 and mot[-1] in "sx" else mot


def porte_un_mot_propre(extrait: ChunkResult, propres: Set[str]) -> bool:
    """
    L'extrait porte-t-il l'un des mots `propres` (formes de _forme_du_mot), dans
    son texte, l'intitule de sa loi ou son titre ? Mot pour mot : « dot »
    n'est pas « dotation ».
    """
    texte = " ".join(filter(None, (extrait.content, extrait.law_title, extrait.article_title)))
    return bool(propres & {_forme_du_mot(m) for m in _mots(texte)})


class RAGQuotaError(Exception):
    """
    Quota de generation epuise chez le fournisseur.

    Merite son propre type parce qu'elle appelle un 429 et un message qui dit
    la verite — « quota du jour atteint » — la ou un 500 disait « erreur
    interne » pour une cause parfaitement connue.
    """


class RAGOverloadedError(Exception):
    """
    Le fournisseur de generation est momentanement sature.

    Distincte de RAGServiceError parce qu'elle appelle une autre reponse HTTP
    (503 et non 500) et un autre message : « reessaie » plutot que « le service
    est en panne ». Volontairement PAS une sous-classe : le gestionnaire de
    RAGServiceError doit la laisser passer.
    """


def titre_de_conversation(question: str, longueur_max: int = 80) -> str:
    """
    Titre affiche dans la liste, tire de la premiere question.

    Coupe sur une FRONTIERE DE MOT : une troncature brute au milieu d'un mot
    donne des titres illisibles dans un panneau etroit.
    """
    propre = " ".join((question or "").split())
    if not propre:
        return "Conversation"
    if len(propre) <= longueur_max:
        return propre
    coupe = propre[:longueur_max].rsplit(" ", 1)[0]
    return (coupe or propre[:longueur_max]) + "…"


class ConversationInterdite(Exception):
    """
    La conversation demandee appartient a quelqu'un d'autre.

    Volontairement PAS une sous-classe de `RAGServiceError` : la route la
    traduit en 404, et le `except RAGServiceError` generique qui rend un 500 ne
    doit pas l'avaler.
    """


class RAGServiceError(Exception):
    """Base exception for RAG service."""
    pass


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for legal Q&A.

    Orchestrates:
    1. Document retrieval (SearchService)
    2. Conversation history loading
    3. Prompt construction (persona-adapted)
    4. Answer generation (service retenu par LLM_PROVIDER)
    5. Citation extraction
    6. Persistence (Database)

    Stateful service - requires database session per request.
    """

    MAX_HISTORY_MESSAGES = 5
    TOP_K_DOCUMENTS = 5
    # Nombre de CHUNKS envoyes au modele. Un chunk est un article, pas un
    # document : huit articles entiers representent un contexte plus precis, et
    # souvent plus court, que cinq lois resumees a 400 caracteres chacune.
    TOP_K_CHUNKS = 8
    # Similarite cosinus EXACTE (voyage-4-large) sous laquelle le volet
    # semantique est juge TRES FAIBLE, selon la langue de la QUESTION
    # (langue_de_la_question ; une question incertaine prend le seuil anglais,
    # le plus bas). Dessous, seuls les textes confirmes restent
    # (_retenir_le_pertinent), et sans eux le modele n'est pas appele.
    #
    # Des seuils BAS : la barriere ne refuse que ce qui est tres loin de tout
    # le corpus ET que le plein texte ne confirme pas. Le tour precedent
    # (0,37 et 0,26), calibre sur des questions formelles, refusait des
    # questions de tous les jours (« Mon fils a été enrôlé de force par des
    # séparatistes », 0,344) et des relances.
    #
    # Calibre le 29/10/2026 (scripts/eval/evaluer_contexte_chat.py, recherche
    # exacte sur jurix_dev, memes articles et memes vecteurs qu'en
    # production, tableaux de programmes retires), questions groupees par le
    # seuil qu'elles prennent, aucune refusee :
    # - seuil francais, 305 questions legitimes : le jeu relu, les suggestions
    #   du front, 36 questions qui nomment un article, 110 questions de tous
    #   les jours (revue du 29/10/2026, fautes comprises), 15 de montants, 39
    #   relances (jugees seules puis jointes a la question precedente),
    #   mots-cles nus. La plus faible : 0,344 (« Mon fils a été enrôlé de force
    #   par des séparatistes ») : MARGE 0,044 ; premier refus legitime a 0,35.
    #   Hors sujet : max 0,413, 6 sur 20 refuses a ce seuil.
    # - seuil anglais, 199 questions legitimes (traductions du jeu,
    #   suggestions, 30 questions de tous les jours, relances, mots-cles nus
    #   de langue incertaine) : la plus faible 0,253 (« inheritance ») : MARGE
    #   0,033 ; premier refus legitime a 0,26. Hors sujet : max 0,387, 3 sur 20
    #   refuses.
    # Index faux simule (les voisins de l'incident, ou des voisins quelconques)
    # : la barriere en refuse 127 et 132 sur 267 en francais, 94 et 86 sur 188
    # en anglais. Les distributions se chevauchent : la similarite n'est pas
    # un juge de la pertinence, et le hors sujet est d'abord ecarte par le
    # classificateur d'intention, avant toute recherche. Un ajout de textes au
    # corpus ou un changement de modele d'embeddings impose de relancer la
    # mesure (--etats normal leurre leurre2).
    SEUILS_DE_SIMILARITE = {"fr": 0.30, "en": 0.22}
    # Premiers articles du plein texte essayes quand le volet semantique est
    # faible, sauf sur une question anglaise : ceux qui portent tous les
    # mots-cles passent EN TETE du contexte (voir _textes_confirmes).
    PLACES_PLEIN_TEXTE = 4
    # Le volet semantique de la recherche vectorise la question ENTIERE, et
    # non ses mots-cles : les mots-cles servent le plein texte, un modele
    # d'embeddings comprend mieux une phrase. Mesure par ce chemin meme
    # (data/mesure_voyage/evaluer_chat.py, 102 questions, voyage-4-large,
    # 16/10/2026), article attendu parmi les 8 extraits / MRR@8 :
    #   francais : mots-cles 0,794 / 0,417 ; question entiere 0,922 / 0,504
    #   anglais  : mots-cles 0,716 / 0,411 ; question entiere 0,873 / 0,538
    # Ecart de MRR@8 en faveur de la question entiere : +0,087 [+0,035 ;
    # +0,143] en francais, +0,127 [+0,067 ; +0,191] en anglais (IC 95 %).
    vectoriser_la_question_entiere = True
    # La grammaire des citations, commentee dans lecture_des_citations.
    # L'attribut reste : les tests des prompts lui appliquent leurs exemples.
    CITATION_REGEX = CITATION_REGEX

    # Mots vides deplaces dans app/services/text_features.py : le re-ranking en
    # a besoin aussi, et deux listes auraient diverge. L'attribut de classe est
    # conserve, plusieurs appelants le lisent.
    STOPWORDS = STOPWORDS

    @staticmethod
    def extract_keywords(question: str) -> str:
        """
        Extract important keywords from a natural language question.
        Removes stopwords pour ameliorer la recherche plein texte.
        """
        import re
        # Normalize: lowercase and remove punctuation except apostrophes
        words = re.findall(r"[a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]+", question.lower())
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in RAGService.STOPWORDS and len(w) > 2]
        
        # Return at least 2 keywords, or original if too few keywords found
        if len(keywords) < 2:
            return question
        
        return " ".join(keywords[:10])  # Limit to 10 keywords

    def __init__(self, db: AsyncSession, user_id: Optional[int] = None):
        """
        Initialize RAG service.

        Args:
            db: Async database session
            user_id: proprietaire des conversations creees ou reprises. `None`
                = requete anonyme. LE DEFAUT EST INDISPENSABLE : les tests
                existants construisent `RAGService(mock_db_session)`.
        """
        self.db = db
        self.user_id = user_id
        # Groq d'abord pour rediger, puis LLM_PROVIDER (app/services/chat_groq.py).
        self.llm = service_du_chat(get_llm_service())
        # Recherche semantique EXACTE pour le chat : il fait rediger une
        # reponse juridique sur ce qu'il trouve, et un index approche peut
        # perdre son rappel sans rien dire (incident du 29/10/2026, voir
        # SearchService._lire_vecteurs_exacts). /search garde l'index.
        self.search_service = SearchService(db, semantique_exacte=True)
        # Les extraits places dans le dernier prompt construit (_build_prompt).
        # Le service vit le temps d'une requete : c'est la question en cours.
        self._extraits_du_prompt: Optional[List[ChunkResult]] = None
        # Ce que la recuperation a decide pour la question en cours
        # (_retrieve_chunks) : la barriere, le journal, la mesure.
        self._recuperation = EtatDeLaRecuperation()

    async def ask(self, request: RAGRequest) -> RAGResponse:
        """
        Main entry point for RAG question answering.

        Pipeline: load history → retrieve context → generate answer → save.

        Args:
            request: RAGRequest with question, persona, etc.

        Returns:
            RAGResponse with answer and citations
        """
        assert request and request.question, "RAGRequest with question is required"
        start_time = time.time()

        logger.info(
            f"🔍 RAG request: persona={request.persona}, "
            f"question=\"{request.question[:50]}...\""
        )

        try:
            # 1. Load conversation history
            conversation, history = await self._load_or_create_conversation(
                request.session_id, request.persona, request.language
            )

            # 1bis. Aiguillage d'intention. APRES l'historique, parce que
            # « et l'article 12 ? » n'est juridique que par ce qui precede ;
            # AVANT la recuperation, parce que c'est tout l'interet : un
            # message conversationnel ne doit declencher ni embedding, ni
            # recherche hybride, ni re-ranking.
            #
            # `law_id` court-circuite le classificateur : l'utilisateur lit un
            # document precis, la preuve contextuelle vaut mieux que n'importe
            # quel verdict de modele, et l'appel est economise.
            if request.law_id:
                intention = IntentResult(INTENT_JURIDIQUE, 1.0, "court-circuit-law-id")
            else:
                intention = await classify_intent(
                    request.question, llm=self.llm, history=history
                )
            logger.info("🧭 Intention=%s (%s)", intention.intent, intention.rule)

            if intention.intent != INTENT_JURIDIQUE:
                return await self._repondre_conversationnel(
                    request, conversation, history, intention, start_time
                )

            # 2. Retrieve and merge document context (relances, repli compris)
            retrieval_start = time.time()
            search_results, early_response = await self._recuperer(
                request, conversation, history
            )
            retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

            # Handle early exit (article not found in specific document)
            if early_response:
                early_response.retrieval_time_ms = retrieval_time_ms
                early_response.total_time_ms = retrieval_time_ms
                return early_response

            if not search_results:
                logger.warning("⚠️ No results found")
                return await self._handle_no_results(request, conversation, retrieval_time_ms)

            # 4. Generate answer, extract citations, save
            return await self._generate_and_save_response(
                request, conversation, history, search_results,
                retrieval_time_ms, start_time
            )

        except ConversationInterdite:
            # DOIT PASSER AVANT `except Exception` : sinon le refus
            # d'appartenance est retraduit en RAGServiceError, donc rendu en
            # 500 par la route au lieu du 404 qui ne revele rien.
            raise
        except ERREURS_QUOTA as e:
            logger.warning(f"⚠️ Quota de generation epuise: {e}")
            raise RAGQuotaError(str(e)) from e
        except ERREURS_SATURATION as e:
            logger.warning(f"⚠️ Generation saturee: {e}")
            raise RAGOverloadedError(str(e)) from e
        except ERREURS_LLM as e:
            logger.error(f"❌ Erreur de generation: {e}")
            raise RAGServiceError(f"Erreur de génération: {e}") from e
        except Exception as e:
            logger.error(f"❌ RAG error: {e}", exc_info=True)
            raise RAGServiceError(f"Erreur lors du traitement: {str(e)}") from e

    async def _recuperer(
        self, request: RAGRequest, conversation: Any, history: Optional[List[Any]]
    ) -> Tuple[List[ChunkResult], Optional[RAGResponse]]:
        """
        La recuperation du chat, la MEME pour `ask` et `ask_stream` (et pour la
        mesure, scripts/eval/evaluer_contexte_chat.py) :
        1. le document consulte et la recherche (_retrieve_and_merge_context),
           une relance cherchee avec le sujet de la conversation, et aussi
           seule quand elle peut changer de sujet (_chercher_seule_et_jointe) ;
        2. une question refusee seule, dans une conversation qui a un sujet :
           cherchee de nouveau, jointe a ce sujet, et n'en garde que ce qui
           porte ses mots ou ses voisins (_chercher_la_relance) ;
        3. rien du tout, sans refus de la barriere : le repli multilingue. Pas
           apres un refus : la recherche a trouve des textes et les a juges
           hors sujet, et ce repli, qui prend la Constitution a defaut d'un
           texte nomme, ne doit pas la contredire.

        Rend (extraits, reponse directe ou None).
        """
        search_results, early_response = await self._retrieve_and_merge_context(
            request, conversation, history
        )
        if early_response is not None:
            return [], early_response
        if not search_results and history:
            search_results = await self._chercher_la_relance(request, history, conversation)
        if not search_results and not self._recuperation.refus:
            search_results = await self._fallback_search(request.question, history)
        return search_results, None

    async def _retrieve_and_merge_context(
        self, request: RAGRequest, conversation, history: Optional[List[Any]] = None
    ):
        """
        Retrieve documents: priority doc first, then general search, merged.

        `history` : l'historique de la conversation. Une relance
        (est_une_relance) y prend son sujet ; sans historique, la question est
        cherchee seule (le controle du chat appelle ainsi cette methode).

        Returns:
            Tuple of (search_results, early_response_or_None)
        """
        priority_chunks: List[ChunkResult] = []
        not_found = None
        if request.law_id:
            # Le marqueur "article absent" est renvoye A PART et non porte par
            # un faux resultat : sonder hasattr(results[0], 'article_not_found')
            # obligeait tout le reste du code a manipuler un objet qui n'etait
            # pas un resultat de recherche.
            priority_chunks, not_found = await self._get_priority_document_context(
                request.law_id, request.question
            )
            logger.info(f"📌 Priority doc {request.law_id}: {len(priority_chunks)} chunks")

        if not_found:
            message = (
                f"Ce document ne contient pas d'article {not_found['requested_article']}. "
                f"Le document \"{not_found['law_title']}\" contient "
                f"{not_found['total_articles']} article(s). "
            )
            if not_found["total_articles"] > 0:
                message += (
                    f"Veuillez demander un article entre 1 et {not_found['total_articles']}."
                )
            else:
                message += "Ce document ne contient aucun article extrait."

            logger.info(f"📌 Article not found response: {message}")
            return [], RAGResponse(
                answer=message, confidence=1.0, sources=[],
                session_id=conversation.session_id,
                retrieval_time_ms=0, generation_time_ms=0, total_time_ms=0,
                persona=request.persona
            )

        # General search. Sur la page d'un texte (`law_id`), l'article qu'une
        # relance nomme sans texte est celui du document OUVERT, deja lu
        # ci-dessus : pas d'article herite du tour precedent en plus.
        search_results = await self._retrieve_chunks(
            request.question, request.language, history=history, conversation=conversation,
            heriter_l_article=not request.law_id,
        )

        # Merge: priority first, then general (deduplicated)
        if priority_chunks:
            # Dedoublonnage sur la cle de chunk et non sur law_id : deux
            # articles differents d'une meme loi doivent tous deux survivre.
            seen = {c.fusion_key for c in priority_chunks}
            filtered = [c for c in search_results if c.fusion_key not in seen]
            search_results = priority_chunks + filtered[:self.TOP_K_CHUNKS]
            logger.info(
                f"📚 Merged: {len(priority_chunks)} priority + {len(filtered[:self.TOP_K_CHUNKS])} general"
            )
            if self._recuperation.refus:
                # Le document que le lecteur a ouvert n'est pas soumis a la
                # barriere : il l'a choisi, ce n'est pas une trouvaille de la
                # recherche. Le modele est donc appele, sur lui seul.
                logger.info(
                    "🚧 Barriere du chat : seul le document consulte reste, le modele "
                    "repond sur lui"
                )

        return search_results, None

    async def _generate_and_save_response(
        self, request, conversation, history, search_results,
        retrieval_time_ms, start_time
    ):
        """Build prompt, generate LLM answer, extract citations, save."""
        prompt = self._build_prompt(
            question=request.question, search_results=search_results,
            history=history, persona=request.persona
        )

        generation_start = time.time()
        system_prompt = get_system_prompt(request.persona, request.language)

        if self.llm is None:
            raise RAGServiceError(MESSAGE_LLM_ABSENT)

        await self._valider_avant_la_generation()

        # 1000 jetons ne suffisent pas a un modele a raisonnement : la
        # reflexion consomme le budget avant la reponse, et ce qui remonte est
        # un morceau de monologue interne. Observe sur une question de suivi,
        # ou la reponse commencait par « *Self-Correction during drafting:* ».
        llm_response = await self.llm.generate(
            prompt=prompt, system=system_prompt, temperature=0.7,
            max_tokens=ANSWER_MAX_TOKENS, reflexion=settings.GEMINI_REFLEXION_REPONSE,
        )
        generation_time_ms = int((time.time() - generation_start) * 1000)
        answer = llm_response["response"] + mention_de_reponse(llm_response, request.language)

        # Extract citations, dans ce que le modele a lu et dans le texte
        # rendu, mention de fin comprise.
        # Les citations sont lues UNE fois par reponse : les sources et les
        # champs de citation partent de la meme lecture. Les sources sont les
        # seuls articles cites et rattaches, jamais un extrait choisi a la
        # place du modele (voir « PAS DE SOURCE FABRIQUEE »).
        extraits = self._extraits_lus(search_results)
        occurrences = lire_les_citations(answer, extraits)
        citations = self._sources_citees(occurrences)

        confidence = self._calculate_confidence(answer, citations, search_results)
        champs = await self._champs_des_citations(answer, extraits, occurrences)

        await self._save_interaction(
            conversation=conversation, question=request.question, answer=answer,
            citations=citations, confidence=confidence,
            retrieval_time_ms=retrieval_time_ms, generation_time_ms=generation_time_ms
        )

        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"✅ RAG complete: {total_time_ms}ms "
            f"(retrieval={retrieval_time_ms}ms, generation={generation_time_ms}ms)"
        )

        return RAGResponse(
            answer=answer, confidence=confidence, sources=citations,
            session_id=conversation.session_id,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms, persona=request.persona,
            **champs,
        )

    async def _repondre_conversationnel(
        self,
        request: RAGRequest,
        conversation: Conversation,
        history: List[Message],
        intention: IntentResult,
        start_time: float,
    ) -> RAGResponse:
        """
        Repond a un message qui n'est pas une question de droit.

        AUCUNE RECUPERATION : ni extraction de mots-cles, ni embedding, ni
        recherche hybride, ni re-ranking, ni repli multilingue, ni contexte
        documentaire. `retrieval_time_ms` vaut donc 0, et c'est la valeur
        EXACTE, pas une approximation : rien n'a ete cherche. Rien n'est ecrit
        non plus dans `query_cache`, `embedding_cache` ni `search_events`.

        `sources` est vide PAR CONSTRUCTION, jamais par accident. Le front
        masque le bloc Sources sur une liste vide : c'est ce qui fait
        disparaitre le « Code Minier » affiche sous une salutation.
        """
        generation_start = time.time()

        if self.llm is None:
            raise RAGServiceError(MESSAGE_LLM_ABSENT)

        system_prompt = get_conversational_prompt(
            intention.intent, request.language, request.persona
        )
        prompt = (
            format_conversation_history(history)
            + "\n\nMessage de l'utilisateur :\n"
            + request.question
        )

        llm_response = await self.llm.generate(
            prompt=prompt, system=system_prompt, temperature=0.7,
            max_tokens=CONVERSATION_MAX_TOKENS, reflexion=settings.GEMINI_REFLEXION_REPONSE,
        )
        generation_time_ms = int((time.time() - generation_start) * 1000)
        # Deux phrases attendues, mais un budget de jetons tout de meme : une
        # reponse coupee le dit, ici comme ailleurs.
        answer = llm_response["response"] + mention_de_reponse(llm_response, request.language)

        # `_calculate_confidence` n'est PAS appele : ses quatre facteurs
        # mesurent l'ancrage documentaire d'une reponse (citations, pertinence
        # moyenne, longueur, mots de couverture). Aucun n'a de sens ici, et le
        # facteur longueur penaliserait justement la brievete demandee.
        await self._save_interaction(
            conversation=conversation, question=request.question, answer=answer,
            citations=[], confidence=CONFIANCE_CONVERSATIONNELLE,
            retrieval_time_ms=0, generation_time_ms=generation_time_ms,
            # Une salutation ne fait pas un titre de conversation. Une question
            # sur le produit, si : elle en dit assez pour s'y retrouver.
            titrer=(intention.intent == "meta"),
        )

        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "✅ Reponse conversationnelle (%s) : %sms, aucune recherche",
            intention.intent, total_time_ms,
        )

        # Aucun texte consulte : toute citation que le modele ecrirait malgre
        # la consigne est introuvable, et le dire vaut mieux que la taire.
        return RAGResponse(
            answer=answer, confidence=CONFIANCE_CONVERSATIONNELLE, sources=[],
            session_id=conversation.session_id,
            retrieval_time_ms=0,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms, persona=request.persona,
            intent=intention.intent,
            **await self._champs_des_citations(answer, []),
        )

    async def ask_stream(
        self,
        request: RAGRequest
    ) -> AsyncIterator[str]:
        """
        Streaming version of ask() for better UX.

        Yields answer chunks as they're generated.
        Final chunk includes sources and confidence.

        Avant le premier mot, des evenements de progression (`progression`,
        `chunk` vide, `done` faux) : recherche lancee, textes trouves,
        redaction en cours. Le front deploye les ignore : il n'y trouve ni
        texte, ni fin, ni erreur.

        Yields:
            JSON strings for SSE: {"chunk": "...", "done": false}
        """
        assert request is not None, "RAGRequest must not be None"
        assert isinstance(request.question, str) and len(request.question) > 0, "Question must be non-empty"

        try:
            start_time = time.time()

            # L'historique est charge AVANT la recherche, alors qu'il l'etait
            # apres. Deux raisons : le classificateur en a besoin (« et
            # l'article 12 ? » n'est juridique que par ce qui precede), et ce
            # chemin retournait jusqu'ici sans jamais creer la conversation
            # quand la recherche ne rendait rien, la ou `ask` l'enregistre via
            # `_handle_no_results`. Contrepartie : une conversation est
            # desormais creee meme si la generation echoue ensuite.
            conversation, history = await self._load_or_create_conversation(
                request.session_id,
                request.persona,
                request.language
            )

            # Meme aiguillage que `ask`. C'est le chemin du front (le chat
            # lit la reponse en flux) : une divergence entre les deux portes
            # d'entree du meme service, ce depot l'a deja payee.
            if request.law_id:
                intention = IntentResult(INTENT_JURIDIQUE, 1.0, "court-circuit-law-id")
            else:
                intention = await classify_intent(
                    request.question, llm=self.llm, history=history
                )
            logger.info("🧭 Intention=%s (%s, flux)", intention.intent, intention.rule)

            if intention.intent != INTENT_JURIDIQUE:
                # Une reponse de deux phrases n'a rien a gagner a etre
                # decoupee : un chunk terminal unique, ce que le contrat SSE
                # autorise deja (`chunk` et `done` dans le meme evenement).
                # Aucune progression : rien n'est cherche.
                reponse = await self._repondre_conversationnel(
                    request, conversation, history, intention, start_time
                )
                yield RAGStreamChunk(
                    chunk=reponse.answer, done=True, sources=[],
                    confidence=reponse.confidence,
                    session_id=reponse.session_id,
                    persona=conversation.persona,
                    intent=reponse.intent,
                    citations=reponse.citations,
                    citations_introuvables=reponse.citations_introuvables,
                    articles_cites=reponse.articles_cites,
                    textes_consultes_non_cites=reponse.textes_consultes_non_cites,
                    contacts=reponse.contacts,
                ).en_json()
                return

            # La recherche commence : le dire, avec l'identifiant de la
            # conversation. Une reponse arretee par le lecteur n'a pas de
            # dernier evenement ; sans lui, la question suivante en ouvrait une
            # autre.
            yield self._evenement_de_progression("recherche", request, conversation)

            # Meme recuperation que `ask` : le document ouvert (`law_id`) en
            # tete du contexte, la reponse directe quand l'article demande n'y
            # est pas, puis la recherche de repli. Ce chemin n'appelait que la
            # recherche generale : sur la page d'un document, le chat en flux
            # aurait repondu sans ce document.
            retrieval_start = time.time()
            search_results, early_response = await self._recuperer(
                request, conversation, history
            )
            if early_response:
                yield RAGStreamChunk(
                    chunk=early_response.answer, done=True, sources=[],
                    confidence=early_response.confidence,
                    session_id=early_response.session_id,
                    persona=conversation.persona,
                    intent=INTENT_JURIDIQUE,
                    **_sans_citation(),
                ).en_json()
                return
            retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

            yield self._evenement_de_progression(
                "textes_trouves", request, conversation, search_results
            )

            if not search_results:
                # Enregistre, comme `_handle_no_results` le fait sur `ask` : ce
                # tour disparaissait de l'historique.
                sans_resultat = self._message_sans_resultat(request)
                await self._save_interaction(
                    conversation=conversation, question=request.question,
                    answer=sans_resultat, citations=[],
                    confidence=0.0, retrieval_time_ms=retrieval_time_ms,
                    generation_time_ms=0,
                )
                yield RAGStreamChunk(
                    chunk=sans_resultat,
                    done=True, sources=[], confidence=0.0,
                    session_id=conversation.session_id,
                    persona=conversation.persona,
                    intent=INTENT_JURIDIQUE,
                    **_sans_citation(),
                ).en_json()
                return

            prompt = self._build_prompt(
                request.question,
                search_results,
                history,
                request.persona
            )

            # Stream generation
            generation_start = time.time()
            # SYSTEM_PROMPTS est indexe par LANGUE ("fr"/"en"), pas par persona :
            # SYSTEM_PROMPTS["citoyen"] levait KeyError a chaque appel, donc
            # /rag/ask/stream n'a jamais rien renvoye d'autre qu'une erreur.
            # get_system_prompt(persona, language) est le bon accesseur, deja
            # utilise par le chemin non-streaming (ligne ~240).
            system_prompt = get_system_prompt(request.persona, request.language)

            if self.llm is None:
                yield RAGStreamChunk(
                    chunk="", done=True,
                    error=MESSAGE_LLM_ABSENT,
                ).en_json()
                return

            await self._valider_avant_la_generation()

            yield self._evenement_de_progression("redaction", request, conversation)

            answer_parts = []
            raison = {}
            # Meme budget que ask() : a 1000 jetons, la reflexion du modele
            # epuisait le budget avant ou pendant la reponse.
            #
            # Le flux du modele est FERME des que ce generateur l'est : quand le
            # lecteur arrete la reponse (deconnexion), le serveur annule ou
            # ferme ce generateur, et la connexion au fournisseur se coupe sur
            # le champ, au lieu d'attendre le ramasse-miettes pendant que le
            # modele continue d'ecrire — et de facturer.
            try:
                async with _ferme_a_la_sortie(self.llm.generate_stream(
                    prompt=prompt,
                    system=system_prompt,
                    temperature=0.7,
                    max_tokens=ANSWER_MAX_TOKENS,
                    reflexion=settings.GEMINI_REFLEXION_REPONSE,
                    fin=lambda r: raison.update(fin=r),
                )) as flux_du_modele:
                    async for chunk in flux_du_modele:
                        answer_parts.append(chunk)
                        yield RAGStreamChunk(chunk=chunk, done=False).en_json()
            except (asyncio.CancelledError, GeneratorExit):
                # Arretee par le lecteur : le flux du modele est deja ferme
                # (ci-dessus). Ce qu'il a lu est enregistre, puis l'arret suit
                # son cours.
                await self._garder_la_reponse_arretee(
                    request, conversation, answer_parts, search_results,
                    retrieval_time_ms, generation_start,
                )
                raise

            # Une reponse restee incomplete le dit, comme dans `ask` : coupee
            # par le budget malgre la suite demandee, ou flux rompu.
            mention = mention_de_fin(raison.get("fin"), request.language)
            if mention:
                answer_parts.append(mention)
                yield RAGStreamChunk(chunk=mention, done=False).en_json()

            generation_time_ms = int((time.time() - generation_start) * 1000)

            # Process complete answer. La lecture des citations porte sur ce
            # que le modele a lu, et sur le texte FINAL, mention comprise :
            # c'est lui que le front a recu, morceau par morceau.
            full_answer = "".join(answer_parts)
            extraits = self._extraits_lus(search_results)
            occurrences = lire_les_citations(full_answer, extraits)
            # Comme `ask` : les articles cites et rattaches, rien d'autre.
            citations = self._sources_citees(occurrences)
            confidence = self._calculate_confidence(
                full_answer, citations, search_results
            )
            champs = await self._champs_des_citations(full_answer, extraits, occurrences)

            # Save to DB
            await self._save_interaction(
                conversation, request.question, full_answer,
                citations, confidence,
                retrieval_time_ms, generation_time_ms
            )

            # Final chunk with metadata
            yield RAGStreamChunk(
                chunk="", done=True, sources=citations,
                confidence=confidence, session_id=conversation.session_id,
                persona=conversation.persona,
                intent=INTENT_JURIDIQUE,
                **champs,
            ).en_json()

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            code = (
                "quota" if isinstance(e, ERREURS_QUOTA)
                else "overloaded" if isinstance(e, ERREURS_SATURATION)
                else "server"
            )
            yield RAGStreamChunk(
                chunk="", done=True, error=str(e), error_code=code
            ).en_json()

    async def _garder_la_reponse_arretee(
        self,
        request: RAGRequest,
        conversation: Conversation,
        answer_parts: List[str],
        search_results: List[ChunkResult],
        retrieval_time_ms: int,
        generation_start: float,
    ) -> None:
        """
        Enregistre la reponse que le lecteur a arretee, telle qu'il l'a lue.

        POURQUOI. Sans cela, l'echange disparaissait : la question suivante
        (« et pour un CDD ? ») arrivait sans contexte, et une conversation
        neuve arretee des sa premiere question restait vide et sans titre dans
        la liste du compte. La mention de fin dit que la reponse est incomplete,
        au lecteur qui rouvre la conversation comme au modele qui relit
        l'historique.

        A L'ABRI DE L'ANNULATION. La deconnexion annule la tache, et Starlette
        renouvelle l'annulation a chaque attente : sans protection,
        l'enregistrement serait annule a son premier aller-retour. Borne a
        quelques secondes : une base qui ne repond plus ne retient pas la
        requete. Rien n'est appele cote modele : son flux est deja ferme.
        """
        texte = "".join(answer_parts)
        if not texte.strip():
            return
        texte += mention_de_fin(FIN_ARRETEE, request.language)
        with anyio.CancelScope(shield=True):
            try:
                with anyio.fail_after(DELAI_ENREGISTREMENT_ARRETEE_S):
                    extraits = self._extraits_lus(search_results)
                    citations = self._extract_citations(texte, extraits)
                    confidence = self._calculate_confidence(texte, citations, search_results)
                    await self._save_interaction(
                        conversation, request.question, texte, citations, confidence,
                        retrieval_time_ms, int((time.time() - generation_start) * 1000),
                    )
                logger.info("⏹️ Reponse arretee par le lecteur : %d caracteres gardes", len(texte))
            except Exception as e:
                logger.warning(f"⚠️ Reponse arretee non enregistree : {e!r}")

    def _evenement_de_progression(
        self,
        etape: str,
        request: RAGRequest,
        conversation: Conversation,
        extraits: Optional[List[ChunkResult]] = None,
    ) -> str:
        """Un evenement de progression du flux, pret a partir."""
        return RAGStreamChunk(
            chunk="", done=False,
            progression=self._progression(etape, request.language, extraits),
            session_id=conversation.session_id,
            persona=conversation.persona,
        ).en_json()

    async def _retrieve_chunks(
        self,
        question: str,
        language: str,
        history: Optional[List[Any]] = None,
        conversation: Any = None,
        heriter_l_article: bool = True,
    ) -> List[ChunkResult]:
        """
        Recupere les articles pertinents via SearchService.

        Mode hybride : il etait epingle sur "text" parce que le corpus n'etait
        pas vectorise et que la recherche semantique ne renvoyait rien
        d'exploitable. Elle rend desormais des articles entiers, et l'hybride
        tolere un cote semantique vide. Son volet semantique est EXACT pour le
        chat (SearchService(semantique_exacte=True)).

        Aucun filtre de langue : un texte camerounais peut n'exister qu'en
        anglais, le modele traduit dans la reponse.

        Les tableaux de programmes des lois de finances sont retires des deux
        volets, sauf pour une question sur le budget (voir
        search_service.est_un_tableau_de_programmes).

        Puis les regles propres au chat (_retenir_le_pertinent) : l'article que
        la question nomme avec son texte, AJOUTE en tete ; quand le volet
        semantique est tres faible, les seuls textes confirmes ; et la
        BARRIERE, qui rend une liste vide quand rien n'est pertinent : ask et
        ask_stream repondent alors sans appeler le modele. `language` est la
        langue de l'interface ; ces regles suivent celle de la QUESTION.

        UNE RELANCE (est_une_relance), dans une conversation qui a un sujet
        (sujet_de_la_conversation), est cherchee JOINTE a ce sujet : le volet
        semantique vectorise « sujet + relance » ; le plein texte et la
        confirmation stricte prennent les mots propres de la relance, puis
        ceux du sujet (« peux détailler » n'est dans aucun article). Un
        article qu'elle nomme sans son texte (« Et l'article 38 ? », « l'article
        40 du même code ») est lu dans le texte du tour precedent quand il est
        sans ambiguite (_texte_du_tour_precedent), sinon nulle part : jamais
        l'article 38 du premier texte venu. Une relance ouverte qui apporte
        ses mots (« Et la nationalité ? », peut_changer_de_sujet) est cherchee
        SEULE ET JOINTE (_chercher_seule_et_jointe). Une question qui a son
        sujet est cherchee seule, comme avant.

        LA SUITE D'UN CHANGEMENT DE SUJET POSSIBLE. Une relance sans mot
        propre ni article (« Peux-tu détailler ? ») juste apres « Et la
        nationalité ? » continue peut-etre le NOUVEAU sujet. Jointe au sujet
        entier (« Quelles indemnités en cas de licenciement abusif ? Et la
        nationalité ? »), elle ne lisait aucun extrait du Code de la
        nationalite (sonde du tour 5) : le modele aurait detaille de memoire.
        La fin du sujet (depuis_un_changement_possible) est cherchee de nouveau
        seule, comme a son tour, ET la relance jointe au sujet entier, les
        deux alternees, la fin au premier rang : la meme regle, un tour plus
        tard.
        """
        sujet = (
            await self._sujet(history, conversation)
            if history and est_une_relance(question) else None
        )
        if sujet is not None and peut_changer_de_sujet(question):
            return await self._chercher_seule_et_jointe(
                question, history, sujet, heriter_l_article
            )
        recent = (
            sujet.depuis_un_changement_possible()
            if sujet is not None and not mots_propres(question) and not numero_sans_texte(question)
            else None
        )
        if recent is not None:
            return await self._chercher_seule_et_jointe(
                question, history, sujet, heriter_l_article, propre=recent
            )
        return await self._une_recherche(question, history, sujet, heriter_l_article)

    async def _une_recherche(
        self,
        question: str,
        history: Optional[List[Any]] = None,
        sujet: Optional[SujetDeLaConversation] = None,
        heriter_l_article: bool = True,
    ) -> List[ChunkResult]:
        """
        UNE recherche hybride et son jugement (_retenir_le_pertinent) : la
        question seule (`sujet` None), ou jointe au sujet de la conversation.
        Pose l'etat de la recuperation (self._recuperation).
        """
        etat = self._recuperation = EtatDeLaRecuperation()
        texte_herite: Optional[Tuple[int, str]] = None
        if sujet is not None:
            recherchee = f"{sujet.texte} {question.strip()}"
            # Le texte dont herite un article nomme sans le sien, decide AVANT
            # la recherche : un renvoi qui porte une nature et n'en herite pas
            # (nature_du_renvoi) garde son mot de nature dans les mots-cles.
            # Sans lui, « Et l'article 1er dudit décret ? » etait cherchee sur
            # les seuls mots du sujet, et le decret n'etait plus lu.
            heritier = heriter_l_article and bool(numero_sans_texte(question))
            if heritier:
                texte_herite = await self._texte_herite(question, history, sujet)
            propres = self._mots_d_origine(question)
            if heritier and texte_herite is None and nature_du_renvoi(question):
                propres = f"{propres} {self._mot_de_nature_d_origine(question)}".strip()
            # Les mots propres de la relance d'abord : extract_keywords n'en
            # garde que dix.
            pour_les_mots_cles = f"{propres} {sujet.texte}"
            etat.relance, etat.sujet, etat.question_jointe = True, sujet.texte, recherchee
            logger.info(
                f"🔁 Relance « {question[:60]} » cherchee avec le sujet de la conversation "
                f"(« {sujet.texte[:80]} »)"
            )
        else:
            recherchee = pour_les_mots_cles = question
        search_query = self.extract_keywords(pour_les_mots_cles)
        etat.langue = langue_de_la_question(recherchee)
        # Une relance garde les tableaux d'un sujet budgetaire seulement si
        # elle n'apporte pas de sujet a elle (« Et pour 2024 ? ») : « Et le
        # SMIG ? », dans une conversation sur le budget, est une question de
        # droit. Dans le doute, pas de tableaux.
        etat.budgetaire = question_budgetaire(question) or (
            sujet is not None and not mots_propres(question) and question_budgetaire(sujet.texte)
        )
        self.search_service.tableaux_de_programmes_exclus = not etat.budgetaire
        logger.info(f"🔍 RAG Search: original='{recherchee[:50]}...' keywords='{search_query}'")

        search_response = await self.search_service.search(
            SearchRequest(
                query=search_query,
                semantic_query=recherchee if self.vectoriser_la_question_entiere else None,
                mode="hybrid",
                filters=SearchFilters(status="published"),
                limit=self.TOP_K_CHUNKS
            )
        )

        chunks = search_response.chunks
        logger.info(
            f"📚 Retrieved {len(chunks)} chunks "
            f"in {search_response.search_time_ms}ms"
        )

        # Etage 2 du re-ranking, optionnel. L'etage 1 a deja ete applique par
        # SearchService ; celui-ci fait noter les meilleurs chunks par le
        # modele. Desactive par defaut : il ajoute un appel facture et plusieurs
        # centaines de millisecondes sur le chemin critique.
        if settings.RERANK_LLM_ENABLED and chunks and self.llm is not None:
            chunks = await rerank_with_llm(
                recherchee, chunks,
                llm=self.llm,
                top_n=settings.RERANK_LLM_TOP_N,
                timeout=settings.RERANK_LLM_TIMEOUT_S,
            )

        etat.hybride = list(chunks)
        if sujet is None:
            nommes = await self._article_nomme(question)
        elif heriter_l_article:
            nommes = await self._article_de_la_relance(question, texte_herite)
        else:
            nommes = []
        # Le plein texte strict ne confirme rien sur la question brute : faute
        # de deux mots-cles, extract_keywords la rend telle quelle (« Et dans
        # mon cas ? »), et un seul mot est partout.
        mots_cles = search_query if len(set(self._mots_cles(pour_les_mots_cles))) >= 2 else ""
        etat.mots_cles = mots_cles
        chunks = await self._retenir_le_pertinent(mots_cles, chunks, nommes)

        if chunks:
            first = chunks[0]
            logger.info(
                f"📄 First chunk: {first.reference} art. {first.number} - {first.law_title}"
            )
        elif not etat.refus:
            logger.warning(f"⚠️ No search results for query: {recherchee[:100]}")

        return chunks

    async def _chercher_seule_et_jointe(
        self,
        question: str,
        history: Optional[List[Any]],
        sujet: SujetDeLaConversation,
        heriter_l_article: bool = True,
        propre: Optional[SujetDeLaConversation] = None,
    ) -> List[ChunkResult]:
        """
        Une relance qui peut changer de sujet (peut_changer_de_sujet : « Et la
        nationalité ? », « Et pour un CDD ? ») : cherchee SEULE, puis JOINTE
        au sujet de la conversation, chacune jugee par la barriere ; le
        contexte ALTERNE les deux listes (_entrelaces : 1er extrait seul, 1er
        joint, 2e seul, 2e joint...), chacune dans son ordre, sans doublon.
        L'article qu'elle nomme sans texte, herite du tour precedent, reste en
        tete (c'est elle qui le demande).

        `propre` : la relance sans mot propre qui SUIT un changement de sujet
        possible (« Peux-tu détailler ? » apres « Et la nationalité ? ») n'a
        pas de recherche seule qui vaille ; sa recherche « propre » est celle
        de la fin du sujet (SujetDeLaConversation.depuis_un_changement_possible),
        cherchee seule, comme a son tour. Pas « fin du sujet + relance » :
        « Et l'héritage ? Donne-moi un exemple concret » tombait sous le seuil
        (0,295), les mots de la relance ne disant rien du sujet, et seul
        l'ancien sujet restait (sonde du tour 5).

        POURQUOI. Revue du 30/10/2026 (tour 4) : jointe seulement, la question
        lisait les textes de l'ANCIEN sujet a la place des siens (« Et la
        nationalité ? » : 0 extrait du Code de la nationalite sur 8, 8 sur 8
        seule). Seule seulement, une vraie suite perdrait le texte de la
        conversation (« Et le préavis ? » : aucun extrait du Code du travail
        seule, 4 jointe). Les deux, et le modele, qui a l'historique, lit
        chacun.

        ALTERNEES, ET NON LA RECHERCHE SEULE D'ABORD. Revue du tour 5 : ses
        extraits d'abord, une VRAIE suite qui porte ses mots lisait en tete
        huit extraits d'autres regimes qui portent le meme mot, et le texte de
        la conversation au rang 9 ou 10 (« Et le préavis ? » : la dissolution
        d'une cooperative, OHADA-AUSCOOP art. 179 ; « Et pour une
        démission ? » : le fonctionnaire, DEC-1994-199 art. 117 ; « And what
        about the notice period? » : l'agent commercial, OHADA-AUDCG art.
        228). Une reponse tiree de ce premier extrait devenait une source
        rattachee, un lien qui a l'air verifie. Les mots seuls ne disent pas
        laquelle des deux recherches est la bonne : alternees, le premier
        extrait de chacune est lu aux deux premiers rangs, et le budget du
        contexte (prompts.build_context_string) n'ecarte plus toute la partie
        jointe quand les extraits propres sont longs. Mesure du tour 6 (rang
        du texte attendu dans ce que lit le modele, jurix_dev) : sur les 17
        vraies suites de la revue du tour 5, le Code du travail revient au
        rang 1 ou 2 pour les 11 francaises (4 au tour 5) ; pour 4 des 6
        anglaises, il vient au rang 4 (6 a 10 au tour 5), parce que la
        recherche JOINTE y met elle-meme un autre texte en tete (CIMA art.
        74, statut general de la fonction publique art. 120) : le Code du
        travail est son 2e extrait, que l'alternance place au 4e rang. Les 16
        changements de sujet des jeux gardent leur nouveau texte au rang 1 ou
        2 dans 14 cas, comme au tour 5.

        LA CORROBORATION. Refusee seule par la barriere, la question n'a rien
        trouve sur ses propres mots : la recherche jointe ne rend alors que
        l'ancien sujet (« la dot » jointe au licenciement : huit extraits du
        Code du travail et du statut des fonctionnaires). Elle n'en garde que
        les extraits qui portent l'un de ses mots propres, ou qui figuraient
        aussi parmi les voisins de la question seule (_corrobores) ; aucun :
        refus. Dans le doute, rien.
        """
        contexte_seul = await self._une_recherche(question if propre is None else propre.texte)
        seule = self._recuperation
        contexte_joint = await self._une_recherche(question, history, sujet, heriter_l_article)
        jointe = self._recuperation

        ecartee = False
        if seule.refus and contexte_joint:
            corrobores = self._corrobores(question, contexte_joint, seule.hybride)
            ecartee = not corrobores
            logger.info(
                f"🔁 Relance « {question[:60]} » refusee seule : "
                + (
                    f"{len(corrobores)} extrait(s) joint(s) sur {len(contexte_joint)} portent ses "
                    f"mots ou sont ses voisins"
                    if corrobores else "rien de la recherche jointe ne porte ses mots, ecartee"
                )
            )
            contexte_joint = corrobores

        # L'article nomme (herite) en tete, puis les deux recherches alternees.
        # Il ouvre deja la liste jointe (_retenir_le_pertinent) : retire d'elle
        # avant l'alternance, il ne decale pas d'un rang chacun des extraits
        # joints.
        nommes = {c.fusion_key for c in jointe.nommes}
        contexte = _sans_doublons(jointe.nommes + _entrelaces(
            contexte_seul, [c for c in contexte_joint if c.fusion_key not in nommes]
        ))
        self._recuperation = self._etat_seule_et_jointe(
            seule, jointe, sujet, contexte, ecartee
        )
        if propre is not None:
            self._recuperation.sujet_recent = propre.texte
        logger.info(
            f"🔀 Relance « {question[:60]} » cherchee "
            + ("seule" if propre is None else f"avec la fin du sujet (« {propre.texte[:60]} »)")
            + f" ({len(contexte_seul)} extrait(s){', refusee' if seule.refus else ''}) et jointe "
            f"au sujet ({len(contexte_joint)}{', refusee' if jointe.refus else ''}"
            f"{', ecartee' if ecartee else ''}) : {len(contexte)} extrait(s), alternes"
        )
        return contexte

    @staticmethod
    def _corrobores(
        question: str, extraits: List[ChunkResult], voisins: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Les extraits d'une recherche JOINTE que la question, refusee seule,
        corrobore : ils portent l'un de ses mots propres (porte_un_mot_propre),
        ou figuraient aussi parmi les voisins de sa recherche seule. Sans mot
        propre, la question n'a rien a corroborer : tout reste (une relance
        sans sujet a elle se lit dans celui de la conversation).
        """
        propres = {_forme_du_mot(m) for m in mots_propres(question)}
        if not propres:
            return list(extraits)
        cles = {c.fusion_key for c in voisins}
        return [
            c for c in extraits
            if c.fusion_key in cles or porte_un_mot_propre(c, propres)
        ]

    @staticmethod
    def _etat_seule_et_jointe(
        seule: EtatDeLaRecuperation,
        jointe: EtatDeLaRecuperation,
        sujet: SujetDeLaConversation,
        contexte: List[ChunkResult],
        ecartee: bool,
    ) -> EtatDeLaRecuperation:
        """
        L'etat d'une recherche seule et jointe, tel que le lisent la suite du
        chat (refus, message, pas de remontee par le numero) et la mesure.
        - refus : rien a lire, et la barriere (ou la corroboration) a refuse ;
        - chemin : celui qui dit de reessayer si l'une des deux recherches n'a
          eu qu'un volet en panne ou servi par l'index ;
        - similarite : la meilleure des deux ; mode « sain » si une partie
          gardee l'etait.
        """
        degrades = (CHEMIN_ECHEC, CHEMIN_ABSENT, *VOLETS_SERVIS_PAR_L_INDEX)
        chemin = next(
            (e.chemin for e in (seule, jointe) if e.chemin in degrades),
            seule.chemin or jointe.chemin,
        )
        similarites = [e.similarite for e in (seule, jointe) if e.similarite is not None]
        gardees = [e for e, garde in ((seule, not seule.refus), (jointe, not jointe.refus and not ecartee))
                   if garde]
        if any(e.mode == "sain" for e in gardees):
            mode = "sain"
        elif seule.mode == jointe.mode == "sans volets":
            mode = "sans volets"
        else:
            mode = "faible"
        return EtatDeLaRecuperation(
            langue=seule.langue, chemin=chemin,
            similarite=max(similarites) if similarites else None,
            seuil=seule.seuil, mode=mode,
            budgetaire=seule.budgetaire or jointe.budgetaire,
            tableaux_exclus=seule.tableaux_exclus + jointe.tableaux_exclus,
            hybride=_sans_doublons(seule.hybride + jointe.hybride),
            juge=list(contexte), nommes=list(jointe.nommes),
            article_nomme=jointe.article_nomme,
            mots_cles=jointe.mots_cles, stricts=seule.stricts | jointe.stricts,
            ajouts=seule.ajouts + jointe.ajouts,
            refus=not contexte and (seule.refus or jointe.refus or ecartee),
            question_jointe=jointe.question_jointe, relance=True, sujet=sujet.texte,
            texte_herite=jointe.texte_herite,
            seule_et_jointe=True, seule=seule, jointe=jointe, jointe_ecartee=ecartee,
        )

    @staticmethod
    def _mots_d_origine(question: str) -> str:
        """Les mots propres de la question (mots_propres), tels qu'elle les ecrit, accents compris."""
        propres = set(mots_propres(question))
        return " ".join(
            m for m in re.findall(r"[^\W\d_]+", question or "") if _sans_accents(m) in propres
        )

    @staticmethod
    def _mot_de_nature_d_origine(question: str) -> str:
        """Le mot de nature du renvoi (mot_de_nature_du_renvoi), tel que la question l'ecrit ; '' sans renvoi."""
        mot = mot_de_nature_du_renvoi(question)
        if not mot:
            return ""
        return next(
            (m for m in re.findall(r"[^\W\d_]+", question or "") if _sans_accents(m) == mot), mot
        )

    async def _sujet(
        self, history: Optional[List[Any]], conversation: Any = None
    ) -> Optional[SujetDeLaConversation]:
        """
        Le sujet de la conversation (sujet_de_la_conversation), dans
        l'historique charge ; a defaut, dans les vingt derniers messages de
        l'utilisateur. L'historique ne garde que les MAX_HISTORY_MESSAGES
        derniers messages : a la troisieme relance de suite, la question de
        fond en est sortie. Une lecture, et seulement dans ce cas ; sous un
        SAVEPOINT, un echec n'annule qu'elle et ne trouve rien.
        """
        sujet = sujet_de_la_conversation(history)
        identifiant = getattr(conversation, "id", None)
        if sujet is not None or not isinstance(identifiant, int):
            return sujet
        try:
            async with self.db.begin_nested():
                lignes = (await self.db.execute(
                    select(Message.content)
                    .where(Message.conversation_id == identifiant, Message.role == "user")
                    .order_by(desc(Message.created_at))
                    .limit(20)
                )).scalars().all()
        except Exception as e:
            logger.warning(f"⚠️ Sujet de la conversation non relu : {e!r}")
            return None
        anciens = [SimpleNamespace(role="user", content=c) for c in reversed(list(lignes)) if c]
        return sujet_de_la_conversation(anciens)

    async def _article_de_la_relance(
        self, question: str, texte: Optional[Tuple[int, str]]
    ) -> List[ChunkResult]:
        """
        L'article qu'une relance nomme SANS son texte (« Et l'article 38 ? »,
        « l'article 38 de ce code »), lu dans `texte`, celui dont elle herite
        (_texte_herite : le texte du tour precedent quand il est sans
        ambiguite, et de la nature du renvoi). Rien sinon : dans le doute,
        aucun article en tete du contexte.

        UN RENVOI QUI A SES MOTS (« l'article 38 de cette loi sur la
        nationalité », « of the law say about notice ») : la suite peut
        designer un AUTRE texte que celui de la conversation (« cette loi sur
        la nationalité », dans une conversation sur le Code du travail).
        L'article herite n'est garde que s'il porte l'un de ces mots
        (« … de ce code sur le préavis » : le morceau 35.1 du Code du travail
        parle du préavis) ; sinon rien, et la question reste jointe au sujet.
        Sans cela, l'article 38 du Code du travail passait en tete, comme la
        reponse, et franchissait la barriere a lui seul.
        """
        numero = numero_sans_texte(question)
        if not numero:
            return []
        if texte is None:
            logger.info(
                f"📌 Article {numero} nomme sans texte, texte du tour precedent ambigu, absent "
                f"ou d'une autre nature que le renvoi : rien d'ajoute"
            )
            return []
        logger.info(f"📌 Article {numero} nomme sans texte : celui du tour precedent, {texte[1]}")
        extraits = await self._lire_l_article_nomme(numero, texte)
        propres = (
            {_forme_du_mot(m) for m in mots_propres(question)} if renvoi_au_texte(question) else set()
        )
        if extraits and propres and not any(porte_un_mot_propre(c, propres) for c in extraits):
            logger.info(
                f"📌 Article {numero} de {texte[1]} : ne porte aucun mot de « {question[:60]} », "
                f"rien d'ajoute (la suite du renvoi peut designer un autre texte)"
            )
            return []
        self._recuperation.texte_herite = texte[1]
        return extraits

    async def _texte_herite(
        self, question: str, history: Optional[List[Any]], sujet: SujetDeLaConversation
    ) -> Optional[Tuple[int, str]]:
        """
        Le texte dont herite une relance qui nomme un article sans le sien :
        celui du tour precedent (_texte_du_tour_precedent). Un renvoi qui porte
        une NATURE (nature_du_renvoi : « ce décret », « cette loi », « that
        decree ») n'en herite que si ce texte est de cette nature, lue dans son
        intitule (de_la_nature) ; sinon rien, et rien non plus si l'intitule
        n'est pas lu : dans le doute, aucun article en tete. Voir « LA NATURE
        D'UN RENVOI ».
        """
        texte = await self._texte_du_tour_precedent(history, sujet)
        nature = nature_du_renvoi(question)
        if texte is None or nature is None:
            return texte
        if de_la_nature(await self._intitule_du_texte(texte[0]), nature):
            return texte
        logger.info(
            f"📌 Renvoi « {question[:60]} » ({nature}) : le texte du tour precedent, {texte[1]}, "
            f"n'est pas de cette nature : rien d'herite, cherche joint au sujet avec son mot de "
            f"nature"
        )
        return None

    async def _intitule_du_texte(self, law_id: int) -> Optional[str]:
        """
        L'intitule du texte `law_id`, pour la nature d'un renvoi (_texte_herite).
        Une lecture par la cle, et seulement pour un renvoi qui porte une
        nature ; sous un SAVEPOINT, un echec n'annule qu'elle et ne lit rien.
        """
        from app.models.law import Law

        try:
            async with self.db.begin_nested():
                intitule = (await self.db.execute(
                    select(Law.title).where(Law.id == law_id)
                )).scalar_one_or_none()
        except Exception as e:
            logger.warning(f"⚠️ Intitule du texte {law_id} non lu : {e!r}")
            return None
        return intitule if isinstance(intitule, str) else None

    async def _texte_du_tour_precedent(
        self, history: Optional[List[Any]], sujet: SujetDeLaConversation
    ) -> Optional[Tuple[int, str]]:
        """
        (identifiant, reference) du texte du tour precedent, ou None quand il
        n'y en a pas ou qu'il y en a plusieurs. Dans l'ordre, le premier qui
        designe quelque chose decide :
        1. le texte que NOMME l'utilisateur, dans la question la plus recente
           du sujet qui en nomme un (« l'article 39 du Code du travail », « la
           loi n° 92/007 », « selon le Code pénal ») ;
        2. les textes des SOURCES de la derniere reponse (articles cites et
           rattaches : un texte lu par le modele) ;
        3. les textes de la table que nomme la derniere reponse.
        Plusieurs textes au premier niveau qui en designe : ambigu, rien.

        Apres un changement de sujet possible (« Que dit l'article 39 du Code
        du travail ? », puis « Et la nationalité ? »), le texte qu'une question
        PLUS ANCIENNE nommait n'est plus celui du tour precedent : la regle 1
        ne lit que la fin du sujet (depuis_un_changement_possible). Sans cela,
        « Et l'article 38 ? » lisait l'article 38 du Code du travail en tete,
        dans une conversation passee a la nationalite ; c'est la derniere
        reponse qui dit alors le texte, ou rien.
        """
        derniere = next(
            (m for m in reversed(history or []) if getattr(m, "role", None) == "assistant"), None
        )
        for question in reversed((sujet.depuis_un_changement_possible() or sujet).questions):
            textes = await self._textes_nommes_par(question)
            if textes:
                return textes.pop() if len(textes) == 1 else None
        sources = getattr(derniere, "sources", None)
        if isinstance(sources, list):
            par_texte = {
                (int(x["law_id"]), x["law_reference"]) for x in sources
                if isinstance(x, dict) and x.get("law_id") and x.get("law_reference")
            }
            if par_texte:
                return par_texte.pop() if len(par_texte) == 1 else None
        nommes = textes_de_la_table_dans(getattr(derniere, "content", None) or "")
        if len(nommes) == 1:
            ids = await self._identifiants_des_textes(nommes)
            return next(((law_id, ref) for ref, law_id in ids.items()), None)
        return None

    async def _textes_nommes_par(self, question: str) -> Set[Tuple[int, str]]:
        """
        Les textes publies que `question` nomme : avec un article
        (reference_d_article), par son numero propre, ou par un nom de la
        table, ou qu'il soit dans la question.
        """
        textes: Set[Tuple[int, str]] = set()
        reference = reference_d_article(question)
        noms = list(reference[1]) if reference is not None else []
        if numero_du_nom(question):
            noms.append(question)
        if noms:
            # Sous un SAVEPOINT, comme _article_nomme : un echec n'annule que
            # lui, et ne designe rien.
            try:
                async with self.db.begin_nested():
                    texte = await self._texte_nomme(noms)
            except Exception as e:
                logger.warning(f"⚠️ Texte nomme non lu : {e!r}")
                texte = None
            if texte is not None:
                textes.add(texte)
        dans_la_table = textes_de_la_table_dans(question)
        if dans_la_table:
            ids = await self._identifiants_des_textes(dans_la_table)
            textes.update((law_id, ref) for ref, law_id in ids.items())
        return textes

    @classmethod
    def _mots_cles(cls, question: str) -> List[str]:
        """Les mots-cles de la question, ceux qu'extract_keywords garde."""
        mots = re.findall(r"[a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]+", (question or "").lower())
        return [m for m in mots if m not in cls.STOPWORDS and len(m) > 2]

    async def _chercher_la_relance(
        self, request: RAGRequest, history: List[Any], conversation: Any = None
    ) -> List[ChunkResult]:
        """
        Une question REFUSEE par la barriere, que rien ne designait comme une
        relance, dans une conversation qui a un sujet : cherchee de nouveau,
        jointe a ce sujet (sujet_de_la_conversation), et la barriere juge de
        nouveau.

        POURQUOI. Revue du 29/10/2026 : 7 relances sur 25 etaient refusees,
        la ou la production repondait avec l'historique. Une relance que
        est_une_relance reconnait est deja jointe des la premiere recherche
        (_retrieve_chunks) ; ce second essai couvre les autres (« Combien de
        temps dure le préavis ? », cherchee seule parce qu'elle a son sujet).
        Le SUJET, et non le dernier message : revue du tour 3, une
        seconde relance de suite etait jointe a la premiere (« Peux-tu
        détailler ? Donne-moi un exemple concret », 0,186) et refusee, 6 sur 6.
        Rien apres une relance deja jointe : le meme sujet ne changerait rien.

        CORROBOREE, OU RIEN. Revue du tour 4 : « la dot » (0,235 seule) et
        « c koi la dot » (0,215), dans une conversation sur le licenciement,
        passaient jointes (0,492 et 0,461) avec huit extraits du Code du
        travail, du statut des fonctionnaires et du Code civil : le modele
        aurait repondu sur l'ancien sujet, ou de memoire. Refusee seule, la
        question n'a rien trouve sur ses propres mots ; de la recherche
        jointe ne restent que les extraits qui portent l'un d'eux, ou qui
        figuraient aussi parmi ses voisins (_corrobores). Aucun : refus.
        """
        etat = self._recuperation
        if not etat.refus or etat.relance:
            return []
        sujet = await self._sujet(history, conversation)
        if sujet is None:
            return []
        voisins = list(etat.hybride)
        jointe = f"{sujet.texte} {request.question.strip()}"
        logger.info(
            f"🔁 Barriere du chat : question refusee seule, cherchee avec le sujet de la "
            f"conversation (« {sujet.texte[:60]} »)"
        )
        extraits = await self._retrieve_chunks(jointe, request.language)
        self._recuperation.question_jointe = jointe
        self._recuperation.sujet = sujet.texte
        if not extraits:
            return extraits
        corrobores = self._corrobores(request.question, extraits, voisins)
        if not corrobores:
            # Un refus, avec le message d'une relance (la reposer en entier).
            self._recuperation.refus = True
            self._recuperation.jointe_ecartee = True
            logger.warning(
                f"🚧 Barriere du chat : « {request.question[:60]} », refusee seule, jointe au "
                f"sujet : aucun extrait ne porte ses mots ni ne figurait parmi ses voisins, "
                f"contexte ecarte"
            )
            return []
        if len(corrobores) < len(extraits):
            logger.info(
                f"🔁 Jointe au sujet : {len(corrobores)} extrait(s) sur {len(extraits)} portent "
                f"les mots de la question ou sont ses voisins, les seuls gardes"
            )
        return corrobores

    async def _retenir_le_pertinent(
        self,
        mots_cles: str,
        chunks: List[ChunkResult],
        nommes: List[ChunkResult],
    ) -> List[ChunkResult]:
        """
        Le contexte du chat, tire de celui de la recherche hybride.

        1. L'ARTICLE NOMME par la question avec son texte (_article_nomme)
           s'AJOUTE en tete : rien n'est evince pour lui faire place.
        2. VOLET SEMANTIQUE SAIN : un extrait du contexte atteint le seuil de
           similarite de la langue de la question (SEUILS_DE_SIMILARITE), un
           seuil BAS. Le contexte de la recherche passe tel quel.
        3. VOLET TRES FAIBLE (sous ce seuil), en panne, ou SERVI PAR L'INDEX
           (repli d'une recherche exacte trop lente, quelle que soit sa
           similarite) : ne restent que les textes CONFIRMES, l'article nomme
           et les articles qui portent TOUS les mots-cles, au moins deux
           (_textes_confirmes). Les extraits sous le seuil ne sont pas des
           sources : le modele ne les lit pas.
        4. LA BARRIERE : aucun texte confirme, liste vide. Le modele n'est pas
           appele ; la reponse le dit (_message_sans_resultat). Il faut donc
           les DEUX preuves contre la question : une similarite tres basse ET
           un plein texte qui ne trouve rien.

        POURQUOI. Incident du 29/10/2026, « Quelles indemnités en cas de
        licenciement abusif ? » : l'index HNSW de production n'a rendu que des
        tableaux de lois de finances (similarite ~0,34), le plein texte
        trouvait le Code du travail, et la fusion a donne au modele huit
        tableaux. Il a repondu que les documents ne disaient rien, puis cite de
        memoire un « article 76 du Code du travail ». La recherche exacte
        retire la cause, et les tableaux de programmes sont retires du
        contexte (search_service.est_un_tableau_de_programmes) : cette regle
        retire ce qui reste, un contexte sans rien de proche de la question.

        L'INDEX N'EST PAS CRU SEUL. Sur le repli, la similarite ne dit rien de
        ce que l'index a MANQUE : un index faux rend des voisins moyennement
        proches. Rejeu de l'incident (voisins = les morceaux de lois de
        finances les plus proches, recherche exacte indisponible, revue du
        29/10/2026, tour 3) : similarite 0,337 au-dessus du seuil de 0,30, et
        le modele lisait huit extraits de lois de finances (des baremes
        d'amendes, les garanties des marches publics), aucun du Code du
        travail ; l'anglaise, 0,238 au-dessus de 0,22, de meme. Aucun seuil ne
        separe ces voisins des bons (les reformulations de l'incident vont a
        0,398). Le plein texte, lui, ne depend pas de l'index : sur le repli,
        seuls les textes qu'il confirme sont lus, ou rien, avec le message
        « réessayez » (la recherche par le sens n'a pas pu etre verifiee). Le
        repli ne sert qu'en surcharge (BUDGET_RECHERCHE_EXACTE_MS), ou sur une
        base froide au-dela de BUDGET_RECHERCHE_EXACTE_A_FROID_MS ; ce qu'il
        coute alors est mesure (scripts/eval/evaluer_contexte_chat.py, etat
        « repli »).

        UN SEUIL BAS, PAS UN SEUIL MOYEN. Le tour precedent refusait sous
        0,37 : des questions de tous les jours en etaient a 0,344 (« Mon fils a
        été enrôlé de force par des séparatistes », refusee alors que la
        production lisait le Code penal). Les distributions des questions
        legitimes et hors sujet se chevauchent : le seuil ne sert qu'a ecarter
        ce qui est tres loin de tout, la similarite n'est pas un juge de la
        pertinence. Calibrage : voir SEUILS_DE_SIMILARITE.

        Sans volets (recherche doublee par un test), rien a juger : le
        contexte passe tel quel, article nomme en tete.
        """
        etat = self._recuperation
        if nommes:
            etat.article_nomme = ", ".join(f"{c.reference} art. {c.number}" for c in nommes)
        contexte = _sans_doublons(nommes + chunks)
        etat.juge, etat.nommes = list(contexte), list(nommes)

        volets = getattr(self.search_service, "volets", None)
        if not isinstance(volets, VoletsDeLaRecherche):
            etat.mode = "sans volets"
            return contexte

        etat.chemin = volets.chemin
        etat.tableaux_exclus = volets.tableaux_exclus
        seuil = etat.seuil = self.seuil_de_similarite(etat.langue)
        connues = [
            volets.similarites[c.article_id]
            for c in contexte if c.article_id in volets.similarites
        ]
        etat.similarite = max(connues) if connues else None
        langue = etat.langue or "langue incertaine"
        similarite = f"{etat.similarite:.3f}" if etat.similarite is not None else "aucune"
        deja = {c.fusion_key for c in chunks}
        par_l_index = volets.chemin in VOLETS_SERVIS_PAR_L_INDEX

        if etat.similarite is not None and etat.similarite >= seuil and not par_l_index:
            etat.mode = "sain"
            etat.ajouts = [f"{c.reference} art. {c.number}" for c in nommes if c.fusion_key not in deja]
            logger.info(
                f"🧭 Contexte du chat : volet semantique {volets.chemin}, similarite "
                f"{similarite} >= {seuil} ({langue}) : {len(chunks)} extraits de la recherche"
                + (f", article nomme en tete : {etat.article_nomme}" if nommes else "")
            )
            return await self._avec_les_renvois(contexte)

        etat.mode = "faible"
        confirmes = await self._textes_confirmes(mots_cles, contexte, nommes, volets)
        etat.ajouts = [f"{c.reference} art. {c.number}" for c in confirmes if c.fusion_key not in deja]

        # Ce qui a fait ecarter le volet semantique, pour le journal.
        if par_l_index:
            pourquoi = (
                f"volet semantique servi par l'index ({volets.chemin}), non verifiable, "
                f"similarite {similarite}, {langue}"
            )
        else:
            pourquoi = f"volet semantique {volets.chemin}, similarite {similarite} < {seuil}, {langue}"

        if not confirmes:
            etat.refus = True
            # Alerte : une question posee n'aura pas de reponse redigee. Le
            # journal dit pourquoi, preuve par preuve.
            logger.warning(
                f"🚧 Barriere du chat : aucun texte de la recherche pertinent, contexte "
                f"ecarte ({pourquoi} ; aucun article avec tous les mots-cles « {mots_cles} » ; "
                f"aucun article nomme)"
            )
            return []

        if par_l_index:
            entete = (
                f"📌 Volet semantique servi par l'index ({volets.chemin}, similarite "
                f"{similarite}, {langue}, non verifiable)"
            )
        else:
            entete = (
                f"📌 Volet semantique tres faible ({volets.chemin}, similarite {similarite} < "
                f"{seuil}, {langue})"
            )
        logger.info(
            f"{entete} : {len(confirmes)} texte(s) confirme(s) sur {len(contexte)} : "
            + ", ".join(f"{c.reference} art. {c.number}" for c in confirmes)
        )
        return confirmes

    async def _avec_les_renvois(self, extraits: List[ChunkResult]) -> List[ChunkResult]:
        """
        L'article vise par une peine, lu juste apres l'extrait qui le vise,
        sauf si un AUTRE TEXTE est lu entre les extraits du meme texte et
        celui qui renvoie : la regle passe alors avant cet autre texte.

        Pourquoi (controle de production du 07/11/2026, « What penalty does the
        Penal Code provide for theft? ») : le vol d'effets militaires
        (PRC-5695 art. 54) et le vol aggrave (PRC-4722 art. 320) sont a
        egalite a quelques millièmes pres ; le bruit du vecteur de Voyage d'un
        appel a l'autre (cosinus 0,997) les permute. Lu derriere art. 54,
        art. 320 au rang 4 poussait l'art. 318 au rang 5, hors des extraits
        lus. Place devant l'autre texte, la regle est lue quel que soit
        l'ordre de cette egalite.
        """
        resultat = list(extraits)
        lus = 0
        for extrait in extraits[:RANGS_DES_RENVOIS]:
            numero = numero_renvoye(extrait)
            if numero is None or lus >= RENVOIS_MAX:
                continue
            place = _place_de_l_article_renvoye(resultat, extrait)
            deja = next(
                (c for c in resultat if c.law_id == extrait.law_id and str(c.number) == numero),
                None,
            )
            if deja is not None:
                if resultat.index(deja) > place:
                    resultat.remove(deja)
                    resultat.insert(_place_de_l_article_renvoye(resultat, extrait), deja)
                continue
            ligne = (await self.db.execute(
                _SQL_ARTICLE_RENVOYE, {"law_id": extrait.law_id, "numero": numero}
            )).first()
            lus += 1
            if ligne is not None:
                vise = self._chunk_de_l_article_nomme(ligne)
                resultat.insert(place, vise.model_copy(update={"source": SOURCE_ARTICLE_RENVOYE}))
        return resultat

    def seuil_de_similarite(self, langue: Optional[str]) -> float:
        """Le seuil de SEUILS_DE_SIMILARITE pour la langue de la question (incertaine : anglais)."""
        return self.SEUILS_DE_SIMILARITE["fr" if langue == "fr" else "en"]

    async def _textes_confirmes(
        self,
        mots_cles: str,
        contexte: List[ChunkResult],
        nommes: List[ChunkResult],
        volets: VoletsDeLaRecherche,
    ) -> List[ChunkResult]:
        """
        Les textes CONFIRMES, ceux qui restent quand le volet semantique est
        tres faible : l'article nomme, puis les articles qui portent TOUS les
        mots-cles (_articles_stricts), d'abord les premiers du plein texte
        (sauf question anglaise), puis ceux du contexte. Pose etat.stricts.

        Une methode a part : scripts/eval/evaluer_contexte_chat.py la rejoue sur
        chaque question pour calibrer la barriere (refuserait-elle, a tel
        seuil ?), sans toucher au chemin du chat.
        """
        etat = self._recuperation
        # Pas pour une question anglaise : son plein texte ne retrouve pas les
        # articles d'un corpus francais (premier tour : 0 article attendu sur
        # 102). Une question incertaine, souvent des mots-cles francais
        # (« ratification rallonge 2019 »), les essaie : il faut tous ses mots.
        candidats = (
            list(volets.plein_texte[:self.PLACES_PLEIN_TEXTE]) if etat.langue != "en" else []
        )
        etat.stricts = await self._articles_stricts(mots_cles, candidats + contexte)
        confirmes = [
            c for c in candidats + contexte
            if c.article_id in etat.stricts
            # Avec un article nomme, son texte seul : « article code travail »,
            # les mots-cles de « Que dit l'article 39 du Code du travail ? »,
            # sont dans des centaines d'articles d'autres textes (mesure : le
            # Code des prestations familiales et un texte sur les pensions).
            and (not nommes or c.law_id in {n.law_id for n in nommes})
        ]
        return _sans_doublons(nommes + confirmes)[:len(nommes) + self.TOP_K_CHUNKS]

    async def _articles_stricts(
        self, mots_cles: str, extraits: List[ChunkResult]
    ) -> Set[int]:
        """
        Parmi `extraits`, les articles qui portent TOUS les mots-cles de la
        question, au moins deux lexemes francais distincts : la requete ET du
        plein texte (websearch_to_tsquery, en francais ou en anglais) sur leur
        vecteur de recherche, intitule de la loi compris.

        Les passes de repli du plein texte (les mots rares seuls, puis OU) ne
        prouvent rien : « recette » seul ramene les recettes budgetaires d'une
        loi de finances. Un seul lexeme non plus (voir _SQL_ARTICLES_STRICTS).

        Une requete, par identifiants, et seulement sous le seuil. Sous un
        SAVEPOINT : un echec n'annule que lui, et ne confirme rien.
        """
        ids = sorted({c.article_id for c in extraits if c.article_id is not None})
        if not ids or not (mots_cles or "").strip():
            return set()
        try:
            async with self.db.begin_nested():
                lignes = (await self.db.execute(
                    _SQL_ARTICLES_STRICTS, {"ids": ids, "requete": mots_cles}
                )).all()
        except Exception as e:
            logger.warning(f"⚠️ Plein texte strict indisponible, aucun texte confirme : {e!r}")
            return set()
        return {int(ligne.id) for ligne in lignes}

    async def _article_nomme(self, question: str) -> List[ChunkResult]:
        """
        L'article que la question NOMME avec son texte (« Que dit l'article 39
        du Code du travail ? », « art. 37 du Code du travail », « Article 39 of
        the Labour Code », « l'article 35 de la loi n° 92/007 »), lu
        directement en base, sans dependre d'aucun des deux volets de la
        recherche.

        POURQUOI. extract_keywords retire le numero : le plein texte ne peut
        pas viser l'article, et le volet semantique le cherche par son sens,
        pas par son numero. Premier tour, panne simulee : « Que dit l'article
        39 du Code du travail ? » recevait le Code des prestations familiales
        et un texte sur les pensions, aucun extrait du Code du travail.

        DANS LE DOUTE, RIEN. Le texte : TEXTES_NOMMABLES ou son numero propre
        (_texte_nomme), jamais un intitule qui ne fait que contenir les mots du
        nom. L'article : _SQL_ARTICLE_NOMME, puis _sans_doute, qui ecarte un
        numero porte par deux articles du meme texte (la compilation du Code
        civil reprend les memes numeros dans plusieurs titres) et un article
        qu'un texte plus recent du corpus modifie, abroge ou remplace
        (liens_entre_textes) : la Constitution de 1996 dit encore « mandat de
        sept ans renouvelable une fois » a son article 6, reecrit depuis.
        L'article manquant, la recherche fait son travail ; un article faux en
        tete du contexte, le modele le lirait comme la reponse.

        Sous un SAVEPOINT : un echec n'annule que lui.
        """
        reference = reference_d_article(question)
        if reference is None:
            return []
        numero, noms = reference
        try:
            async with self.db.begin_nested():
                texte = await self._texte_nomme(noms)
        except Exception as e:
            logger.warning(f"⚠️ Article nomme non lu : {e!r}")
            return []
        if texte is None:
            return []
        return await self._lire_l_article_nomme(numero, texte)

    async def _lire_l_article_nomme(
        self, numero: str, texte: Tuple[int, str]
    ) -> List[ChunkResult]:
        """
        L'article `numero` du texte `texte` (identifiant, reference) : les trois
        lectures de _SQL_ARTICLE_NOMME, puis _sans_doute. Pour l'article que
        la question nomme avec son texte, et pour celui qu'une relance nomme
        sans le sien (texte du tour precedent). Sous un SAVEPOINT.
        """
        try:
            async with self.db.begin_nested():
                lignes = (await self.db.execute(_SQL_ARTICLE_NOMME, {
                    "law_id": texte[0],
                    "numero": numero.upper(),
                    "morceau": f"{escape_like(numero.upper())}.%",
                    "entete": _motif_d_en_tete(numero),
                })).all()
        except Exception as e:
            logger.warning(f"⚠️ Article nomme non lu : {e!r}")
            return []
        if not lignes:
            logger.info(f"📌 Article {numero} nomme : absent du texte {texte[1]}")
            return []
        articles = self._sans_doute(numero, texte[1], lignes)
        if articles:
            logger.info(
                f"📌 Article {numero} nomme par la question : "
                + ", ".join(f"{c.reference} art. {c.number}" for c in articles)
                + f" ({['numero', 'morceau', 'en-tete'][lignes[0].lecture]})"
            )
        return articles

    def _sans_doute(self, numero: str, reference: str, lignes: List[Any]) -> List[ChunkResult]:
        """
        Les extraits de l'article nomme, ou rien s'il y a un doute :
        - la lecture la plus sure seulement (un numero exact ne se complete pas
          de l'en-tete d'un autre morceau) ;
        - plusieurs lignes : elles doivent se suivre dans le texte (morceaux
          d'un meme article, ou « 74 » et « 74-1 » du Code penal, ranges sous
          le meme numero) ; sinon le numero designe deux articles ;
        - aucun texte plus recent ne doit toucher l'article, sauf pour le
          completer (liens_qui_touchent).
        """
        lecture = lignes[0].lecture
        memes = [ligne for ligne in lignes if ligne.lecture == lecture]
        ordres = [ligne.ordre for ligne in memes]
        if len(memes) > 1 and (
            None in ordres or max(ordres) - min(ordres) != len(memes) - 1
            or len({ligne.number for ligne in memes}) != len(memes) and lecture == 1
        ):
            logger.info(
                f"📌 Article {numero} de {reference} : {len(memes)} extraits qui ne se suivent "
                f"pas, rien d'ajoute (numero ambigu)"
            )
            return []
        touche = [
            lien for ligne in memes
            for lien in (
                liens_qui_touchent(ligne.reference, ligne.number, ligne.content, ligne.section)
                + liens_qui_touchent(ligne.reference, numero, ligne.content, ligne.section)
            )
            if lien.nature != "complete"
        ]
        if touche:
            logger.info(
                f"📌 Article {numero} de {reference} : {touche[0].nature} par "
                f"{touche[0].par_reference or touche[0].par_titre}, rien d'ajoute (la recherche "
                f"et les lignes de statut en rendent compte)"
            )
            return []
        return [self._chunk_de_l_article_nomme(ligne) for ligne in memes][:MORCEAUX_DE_L_ARTICLE_NOMME]

    async def _texte_nomme(self, noms: List[str]) -> Optional[Tuple[int, str]]:
        """
        (identifiant, reference) du texte publie que l'un des noms designe :
        par son numero propre, ou par TEXTES_NOMMABLES. None sinon : un nom
        inconnu de la table, ou un numero porte par aucun texte ou par
        plusieurs.
        """
        from app.models.law import Law

        for nom in noms:
            numerote = numero_du_nom(nom)
            if numerote is not None:
                textes = [
                    (int(ligne.id), ligne.reference)
                    for ligne in (await self.db.execute(_SQL_TEXTES_PUBLIES)).all()
                    if numero_de_l_intitule(ligne.title) == numerote
                ]
                if len(textes) == 1:
                    return textes[0]
                logger.info(
                    f"📌 « {nom} » : {len(textes)} texte(s) publie(s) portent ce numero, "
                    f"pas d'article nomme"
                )
                return None
            texte = texte_de_la_table(nom)
            if texte is not None:
                ligne = (await self.db.execute(
                    select(Law.id, Law.reference).where(
                        Law.reference == texte.reference, Law.status == "published"
                    )
                )).first()
                return (int(ligne.id), ligne.reference) if ligne is not None else None
        logger.info(f"📌 « {noms[0]} » : texte absent de la table des noms, pas d'article nomme")
        return None

    @staticmethod
    def _chunk_de_l_article_nomme(ligne: Any) -> ChunkResult:
        content = ligne.content or ""
        return ChunkResult(
            article_id=ligne.article_id,
            law_id=ligne.law_id,
            number=str(ligne.number) if ligne.number else None,
            article_title=ligne.article_title,
            section=ligne.section,
            page_number=ligne.page_number,
            content=content,
            excerpt=content[:400],
            reference=ligne.reference or "",
            law_title=ligne.law_title or "",
            type=ligne.type or "loi",
            language=ligne.language,
            status=ligne.status or "published",
            category_id=ligne.category_id,
            publication_date=ligne.publication_date,
            relevance_score=1.0,
            source=SOURCE_ARTICLE_NOMME,
        )

    async def _get_priority_document_context(
        self, law_id: int, question: str
    ) -> Tuple[List[ChunkResult], Optional[Dict[str, Any]]]:
        """
        Contexte issu du document actuellement consulte.

        Quand l'utilisateur lit un document et pose une question, ses articles
        priment sur la recherche generale.

        Returns:
            (chunks, marqueur_article_absent). Le second element vaut None dans
            le cas nominal ; il porte sinon de quoi composer la reponse
            "cet article n'existe pas dans ce document".
        """
        assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"

        from app.models.law import Article, Law

        try:
            query = select(Law).where(Law.id == law_id)
            result = await self.db.execute(query)
            law = result.scalar_one_or_none()

            if not law:
                logger.warning(f"⚠️ Priority law_id {law_id} not found")
                return [], None

            article_query = (
                select(Article).where(Article.law_id == law_id).order_by(Article.order)
            )
            article_result = await self.db.execute(article_query)
            articles = list(article_result.scalars().all())
            total_articles = len(articles)

            article_num = self._extract_article_number(question)

            if article_num:
                logger.info(
                    f"📌 Looking for article {article_num} in doc (has {total_articles} articles)"
                )

                # Un autre document est nomme : laisser la recherche generale faire
                if self._references_other_document(question, law.title):
                    return [], None

                wanted = _normalize_article_number(article_num)
                for article in articles:
                    if article.number and _normalize_article_number(article.number) == wanted:
                        logger.info(f"✅ Found article {article.number} in priority doc")
                        return [self._chunk_from_article(law, article, "priority")], None

                logger.info(
                    f"❌ Article {article_num} NOT found in doc (has {total_articles} articles)"
                )
                return [], {
                    "requested_article": article_num,
                    "law_title": law.title,
                    "total_articles": total_articles,
                }

            # Pas d'article precis : les premiers articles du document, chacun
            # comme un chunk distinct. Ils etaient auparavant concatenes dans un
            # seul bloc de texte, ce qui rendait toute citation impossible.
            chunks = [
                self._chunk_from_article(law, article, "priority")
                for article in articles[:3]
                if article.content
            ]
            if chunks:
                return chunks, None

            # Document sans article extrait : le texte integral, sans identite
            # d'article.
            if law.content:
                return [ChunkResult(
                    article_id=None,
                    law_id=law.id,
                    content=law.content,
                    excerpt=law.content[:400],
                    reference=law.reference or "",
                    law_title=law.title,
                    type=law.type or "loi",
                    language=law.language,
                    status=law.status or "published",
                    category_id=law.category_id,
                    category_name="Document actif",
                    publication_date=law.publication_date,
                    relevance_score=1.0,
                    source="priority",
                )], None

        except Exception as e:
            logger.error(f"⚠️ Error fetching priority document: {e}")

        return [], None

    @staticmethod
    def _chunk_from_article(law, article, source: str) -> ChunkResult:
        """Construit un ChunkResult a partir des lignes ORM Law + Article."""
        content = article.content or ""
        return ChunkResult(
            article_id=article.id,
            law_id=law.id,
            number=str(article.number) if article.number else None,
            article_title=article.title,
            section=article.section,
            page_number=article.page_number,
            content=content,
            excerpt=content[:400],
            reference=law.reference or "",
            law_title=law.title,
            type=law.type or "loi",
            language=law.language,
            status=law.status or "published",
            category_id=law.category_id,
            category_name="Document actif" if source == "priority" else None,
            publication_date=law.publication_date,
            relevance_score=1.0,
            source=source,
        )

    def _references_other_document(self, question: str, current_title: str) -> bool:
        """
        Check if the question mentions a law document different from the current one.

        Args:
            question: User's question
            current_title: Title of the currently viewed law

        Returns:
            True if another document is explicitly mentioned
        """
        import re
        other_doc_patterns = [
            r'constitution', r'code\s+civil', r'code\s+p[ée]nal',
            r'loi\s+ohada', r'charte', r'décret',
        ]
        question_lower = question.lower()
        for pattern in other_doc_patterns:
            if re.search(pattern, question_lower) and pattern not in current_title.lower():
                logger.info("📌 Another document mentioned, skipping priority")
                return True
        return False

    async def _load_or_create_conversation(
        self,
        session_id: Optional[str],
        persona: str,
        language: str
    ) -> Tuple[Conversation, List[Message]]:
        """
        Load existing conversation or create new one.

        Returns:
            (conversation, last_N_messages)
        """
        if session_id:
            # Try to load existing
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .options(joinedload(Conversation.messages))
            )
            result = await self.db.execute(stmt)
            # .unique() obligatoire apres un joinedload sur une collection :
            # sans lui SQLAlchemy leve InvalidRequestError, ce qui faisait
            # echouer TOUTE requete portant un session_id deja existant.
            conversation = result.unique().scalar_one_or_none()

            if conversation:
                # CONTROLE D'APPARTENANCE. Sans lui, poster le `session_id`
                # d'un tiers chargeait sa conversation ET en reinjectait les
                # cinq derniers messages dans le prompt : le contenu d'autrui
                # ressortait dans la reponse generee.
                #
                # `user_id` NULL signifie « anonyme, appartient a qui detient le
                # session_id » — les conversations d'avant les comptes restent
                # donc accessibles, sans quoi cette regle les aurait toutes
                # rendues muettes.
                if conversation.user_id not in (None, self.user_id):
                    logger.warning(
                        "⛔ Conversation %s refusee : appartient a un autre compte",
                        session_id,
                    )
                    raise ConversationInterdite(session_id)

                # UN PROFIL PAR CONVERSATION. Un autre profil que celui de la
                # conversation en ouvre une NOUVELLE (nouveau session_id,
                # historique vide), et l'ancienne n'est pas touchee. Ni 409 : le
                # front ne garde que le session_id, et un profil change entre
                # deux visites ferait echouer la premiere question. Ni ignorer :
                # on repondrait avec un profil que l'utilisateur vient de
                # quitter. Le controle d'appartenance passe AVANT : la
                # conversation d'autrui reste refusee, quel que soit le profil.
                if conversation.persona != persona:
                    # Sans les profils : la ligne voisine relie la conversation
                    # au compte, et un profil est une donnee personnelle.
                    logger.info("🔀 Conversation %s : profil change, nouvelle conversation", session_id)
                    session_id = None
                else:
                    # ADOPTION : une conversation anonyme reprise par un compte
                    # lui est rattachee. C'est ce qui permet de discuter sans
                    # compte, puis de s'inscrire en gardant le fil.
                    if conversation.user_id is None and self.user_id is not None:
                        conversation.user_id = self.user_id
                        logger.info(
                            "🔗 Conversation %s rattachee au compte %s",
                            session_id,
                            self.user_id,
                        )

                    # Get last N messages
                    msg_stmt = (
                        select(Message)
                        .where(Message.conversation_id == conversation.id)
                        .order_by(desc(Message.created_at))
                        .limit(self.MAX_HISTORY_MESSAGES)
                    )
                    msg_result = await self.db.execute(msg_stmt)
                    messages = list(msg_result.scalars().all())

                    logger.info(f"📖 Loaded conversation: {len(messages)} messages")
                    return conversation, list(reversed(messages))

        # Create new conversation
        conversation = Conversation(
            session_id=session_id or str(uuid.uuid4()),
            persona=persona,
            language=language,
            user_id=self.user_id,
        )
        self.db.add(conversation)
        await self.db.flush()

        logger.info(f"🆕 New conversation: {conversation.session_id}")
        return conversation, []

    async def _fallback_search(self, question: str, history: List = None) -> List[ChunkResult]:
        """
        Repli multilingue quand la recherche plein texte ne rend rien.

        1. Detecte la langue de la question (FR/EN)
        2. Extrait le contexte documentaire de l'historique
        3. Cherche d'abord dans les documents de la meme langue, puis dans l'autre
        4. La traduction est laissee au modele

        Tourne sur la session async de la requete. Cette methode ouvrait son
        propre moteur synchrone — dans une coroutine — a chaque appel, sans
        jamais le fermer.
        """
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"
        assert history is None or isinstance(history, list), "history must be a list or None"

        prompt_lang = self._detect_language(question)
        logger.info(f"🌐 Detected prompt language: {prompt_lang}")

        article_num = self._extract_article_number(question)
        if not article_num:
            return []

        doc_context = self._extract_document_context(question, history)
        logger.info(
            f"📌 Multilingual search: article {article_num}, context: {doc_context or 'none'}"
        )

        try:
            search_order = ["fr", "en"] if prompt_lang == "fr" else ["en", "fr"]
            # Metacaracteres echappes : un % ou un _ dans le contexte
            # elargissait le motif au lieu d'etre cherche litteralement.
            raw_pattern = doc_context if doc_context else "constitution"
            search_pattern = f"%{escape_like(raw_pattern.lower())}%"

            sql = text("""
                SELECT l.id, l.title, l.reference, l.language, l.type,
                       l.category_id, l.status, l.publication_date,
                       a.id AS article_id, a.number, a.title AS article_title,
                       a.section, a.page_number, a.content AS article_content
                FROM laws l
                JOIN articles a ON a.law_id = l.id
                WHERE (l.language = :lang OR l.language IS NULL)
                  AND (LOWER(l.title) LIKE :pattern ESCAPE '\\'
                       OR LOWER(l.reference) LIKE :pattern ESCAPE '\\')
                  AND (a.number = :num OR a.number ILIKE :num_pattern ESCAPE '\\')
                ORDER BY a.order
                LIMIT 1
            """)

            for lang in search_order:
                result = await self.db.execute(sql, {
                    "lang": lang,
                    "pattern": search_pattern,
                    "num": article_num,
                    "num_pattern": f"%{escape_like(article_num)}%",
                })
                row = result.fetchone()
                if not row:
                    continue

                logger.info(
                    f"📄 Found in {lang.upper()}: {row.title} - article {row.number}"
                )
                content = row.article_content or ""
                return [ChunkResult(
                    article_id=row.article_id,
                    law_id=row.id,
                    number=str(row.number) if row.number else None,
                    article_title=row.article_title,
                    section=row.section,
                    page_number=row.page_number,
                    content=content,
                    excerpt=content[:400],
                    reference=row.reference or "",
                    law_title=row.title,
                    type=row.type or "loi",
                    language=row.language or lang,
                    status=row.status or "published",
                    category_id=row.category_id,
                    publication_date=row.publication_date,
                    relevance_score=1.0,
                    source="fallback",
                )]

            logger.warning(f"⚠️ Article/Section {article_num} not found in any language")

        except Exception as e:
            logger.warning(f"⚠️ Multilingual search failed: {e}")

        return []

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily French or English."""
        french_words = ["de", "la", "le", "les", "du", "des", "au", "aux", "un", "une", 
                       "et", "ou", "pour", "dans", "sur", "avec", "moi", "explique"]
        english_words = ["the", "of", "and", "or", "for", "in", "on", "with", "me", 
                        "explain", "what", "how", "is", "are", "does", "section"]
        
        text_lower = text.lower()
        fr_count = sum(1 for w in french_words if f" {w} " in f" {text_lower} ")
        en_count = sum(1 for w in english_words if f" {w} " in f" {text_lower} ")
        
        return "fr" if fr_count >= en_count else "en"
    
    def _extract_article_number(self, question: str) -> str:
        """Extract article/section number from question."""
        import re
        
        patterns = [
            r'article\s+(\d+|premier|première)',
            r'section\s+(\d+|first)',
            r"l'article\s+(\d+)",
            r'art\.?\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                num = match.group(1)
                # Normalize
                if num.lower() in ["premier", "première", "first"]:
                    return "1"
                return num
        return None
    
    def _extract_document_context(self, question: str, history: List = None) -> str:
        """
        Extract document context from current question or conversation history.
        Used for follow-up questions like "explique l'article 8" after discussing constitution.
        """
        import re
        
        # Document patterns to look for (FR and EN)
        doc_patterns = [
            (r'constitution', 'constitution'),
            (r'code\s+civil', 'code civil'),
            (r'code\s+p[ée]nal', 'code penal'),
            (r'code\s+du\s+travail', 'code du travail'),
            (r'code\s+de\s+commerce', 'code de commerce'),
            (r'loi\s+ohada', 'loi ohada'),
            (r'civil\s+code', 'code civil'),
            (r'penal\s+code', 'code penal'),
            (r'labor\s+code', 'code du travail'),
        ]
        
        # First check current question
        question_lower = question.lower()
        for pattern, doc_name in doc_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                logger.info(f"📚 Document context from question: {doc_name}")
                return doc_name
        
        # If not in question, check conversation history
        if history:
            for msg in reversed(history):  # Start from most recent
                if hasattr(msg, 'content'):
                    content_lower = msg.content.lower()
                    for pattern, doc_name in doc_patterns:
                        if re.search(pattern, content_lower, re.IGNORECASE):
                            logger.info(f"📚 Document context from history: {doc_name}")
                            return doc_name
                            
                # Also check sources in messages
                if hasattr(msg, 'sources') and msg.sources:
                    try:
                        import json
                        sources = msg.sources if isinstance(msg.sources, list) else json.loads(msg.sources)
                        for source in sources:
                            if isinstance(source, dict):
                                title = source.get('law_title', '').lower()
                                for pattern, doc_name in doc_patterns:
                                    if re.search(pattern, title, re.IGNORECASE):
                                        logger.info(f"📚 Document context from sources: {doc_name}")
                                        return doc_name
                    except Exception as e:
                        # `except: pass` nu attrapait KeyboardInterrupt et
                        # SystemExit en plus des erreurs visees, et n'en
                        # laissait aucune trace. L'echec reste tolere — c'est
                        # une heuristique de confort — mais il se voit.
                        logger.debug(f"Detection de document ignoree: {e}")
        
        return None

    def _build_prompt(
        self,
        question: str,
        search_results: List,
        history: List[Message],
        persona: str
    ) -> str:
        """
        Build complete prompt with context and history.

        Structure:
        1. Context from search results
        2. Specific article content if question mentions an article
        3. Conversation history
        4. User question
        """
        # L'article explicitement demande est remonte en tete du contexte.
        # Une methode dediee allait auparavant le RECHERCHER EN BASE, avec son
        # propre moteur synchrone, parce que le contexte ne contenait que des
        # extraits de 400 caracteres. Le contexte porte desormais le texte
        # integral : un simple reordonnancement suffit.
        chunks = self._pin_requested_article(question, search_results)

        # Ce que le modele lit vraiment : la lecture des citations s'y tient
        # (voir _extraits_lus). Le budget du contexte peut ecarter les derniers.
        retenus: List[ChunkResult] = []
        context_str = build_context_string(chunks, retenus)
        self._extraits_du_prompt = retenus
        context_section = CONTEXT_TEMPLATE.format(context_docs=context_str)

        # Build history
        history_str = format_conversation_history(history) if history else ""

        # Combine
        parts = [context_section]

        if history_str:
            parts.append(f"\n{history_str}\n")

        parts.append(f"\nQuestion actuelle de l'utilisateur:\n{question}")

        return "\n".join(parts)
    
    def _pin_requested_article(
        self, question: str, chunks: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Remonte en tete le chunk correspondant a l'article demande, quand la
        question ne donne qu'un numero (« Que dit l'article 42 ? »).

        Sans acces a la base : le contexte contient deja le texte integral des
        articles retenus, il ne reste qu'a mettre le bon en premier.
        """
        article_num = self._extract_article_number(question)
        if not article_num or not chunks:
            return chunks
        # Une relance (« Et l'article 38 ? ») : son numero est celui du texte
        # dont parle la conversation. Lu dans ce texte, il est deja en tete
        # (_article_de_la_relance) ; sinon, aucun extrait n'est remonte pour
        # son seul numero. Revue du 29/10/2026 (tour 3) : l'article 38 de la
        # loi sur la concurrence remontait en tete, dans une conversation sur
        # le licenciement.
        if getattr(getattr(self, "_recuperation", None), "relance", False):
            return chunks
        # « l'article 38 de ce code », « dudit code », « of this law » : le
        # numero est celui du texte de la conversation. Sans historique (rien
        # n'en dit le texte), aucun extrait n'est remonte pour son seul numero.
        # Revue du tour 4 : « … dudit code » remontait en tete l'article 38 du
        # Code de la nationalite, dans une conversation sur le licenciement.
        if renvoi_au_texte(question):
            return chunks
        # L'article NOMME avec son texte est deja en tete (_article_nomme) :
        # ce reordonnancement par le numero seul y remonterait l'article de
        # meme numero d'un autre texte.
        if chunks[0].source == SOURCE_ARTICLE_NOMME:
            return chunks
        # La question nomme un texte, et _article_nomme n'a rien ajoute (texte
        # hors de la table, article reecrit depuis, numero ambigu) : le numero
        # est celui de CE texte, et le premier extrait de meme numero est
        # souvent celui d'un autre. Mesure du 29/10/2026 (scripts/eval/
        # evaluer_contexte_chat.py, questions nommees de la revue) : en tete,
        # l'article 6 de la loi sur le Conseil constitutionnel pour « l'article
        # 6 de la Constitution », l'article 2 du Code minier de 2016 pour « la
        # loi sur les investissements ». L'ordre de la recherche reste ;
        # l'article demande, s'il est lu, reste dans le contexte.
        if reference_d_article(question) is not None:
            return chunks

        wanted = _normalize_article_number(article_num)
        for index, chunk in enumerate(chunks):
            if chunk.number and _normalize_article_number(chunk.number) == wanted:
                if index == 0:
                    return chunks
                logger.info(f"📌 Article {chunk.number} remonte en tete du contexte")
                return [chunk] + chunks[:index] + chunks[index + 1:]

        return chunks

    def _extract_citations(
        self,
        answer: str,
        chunks: List[ChunkResult]
    ) -> List[Citation]:
        """
        Les sources citees par la reponse, une par article : ce que `sources`
        a toujours porte.

        Cherche "Article 161 du Code OHADA", "article 5 de la Loi...", etc.

        La validation se fait sur les chunks REELLEMENT places dans le prompt :
        couple (loi, numero d'article normalise). Elle se contentait de
        verifier que le nom cite apparaissait dans le titre de la loi, sans
        jamais confronter le NUMERO d'article a ce que le contexte contenait.
        L'extrait et l'identifiant d'article proviennent du chunk : aucun
        aller-retour en base, la ou trois moteurs synchrones etaient ouverts
        pour aller rechercher un texte deja present.

        La lecture elle-meme — position, rattachement, citations introuvables —
        est dans lecture_des_citations ; seules les citations d'ARTICLE
        rattachees deviennent des sources, comme avant.
        """
        return self._sources_citees(lire_les_citations(answer, chunks))

    @staticmethod
    def _sources_citees(occurrences: List[Occurrence]) -> List[Citation]:
        """Une `Citation` par article cite et rattache, dans l'ordre du texte."""
        citations: List[Citation] = []
        vues = set()
        for occurrence in occurrences:
            chunk = occurrence.extrait
            if occurrence.forme != "article" or chunk is None or occurrence.cle in vues:
                continue
            vues.add(occurrence.cle)
            excerpt = (chunk.content or chunk.excerpt or "")[:300]
            citations.append(Citation(
                law_id=chunk.law_id,
                law_reference=chunk.reference,
                law_title=chunk.law_title,
                article_number=occurrence.article,
                article_id=chunk.article_id,
                excerpt=excerpt or "Voir document complet pour détails",
                relevance_score=chunk.relevance_score,
            ))

        logger.info(f"📎 Extracted {len(citations)} citations")
        return citations

    def _extraits_lus(self, search_results: List[ChunkResult]) -> List[ChunkResult]:
        """
        Les extraits que le modele a lus : ceux du prompt, quand il a ete
        construit ; a defaut, ce que la recherche a rendu.
        """
        if self._extraits_du_prompt is not None:
            return self._extraits_du_prompt
        return list(search_results or [])

    async def _identifiants_des_textes(self, references) -> Dict[str, int]:
        """
        L'identifiant en base de chaque reference, pour lier un texte que nomme
        un lien prouve (« Modifié par la Loi N°… »).

        Une requete, et seulement s'il y a quelque chose a resoudre : le corpus
        compte 67 liens, la plupart des reponses n'en touchent aucun. Un echec
        laisse les liens sans identifiant plutot que de faire echouer la
        reponse.

        SOUS UN POINT DE SAUVEGARDE, comme la recherche vectorielle
        (search_service) : un echec n'annule que lui. Un rollback complet
        expirait TOUS les objets de la session, conversation comprise, et
        l'enregistrement de l'echange, qui suit, echouait a la lecture de son
        titre (MissingGreenlet, essai du 2026-10-28 sur PostgreSQL) : la
        reponse, deja affichee, sortait de l'historique.
        """
        references = sorted({r for r in references or () if r})
        if not references:
            return {}

        try:
            async with self.db.begin_nested():
                lignes = (await self.db.execute(_requete_des_identifiants(references))).all()
        except Exception as e:
            logger.warning(f"⚠️ Identifiants des textes lies introuvables : {e}")
            return {}
        return {reference: law_id for reference, law_id in lignes if reference}

    async def _champs_des_citations(
        self, answer: str, extraits: List[ChunkResult],
        occurrences: Optional[List[Occurrence]] = None,
    ) -> Dict[str, list]:
        """
        Les quatre champs nouveaux d'une reponse : `citations`,
        `citations_introuvables`, `articles_cites`, `textes_consultes_non_cites`.

        Calcules sur le texte FINAL, mention de fin comprise : c'est celui que
        le front affiche et que la conversation enregistre. Les indices sont
        rendus en UTF-16, ceux du `slice` de JavaScript.

        `extraits` : ce que le modele a lu (_extraits_lus). `sources` n'en
        depend pas : les seuls articles cites et rattaches. Ce que le modele a
        lu sans le citer est dans `textes_consultes_non_cites`, jamais dans
        `sources`.

        `occurrences` : la lecture deja faite de cette reponse sur ces
        extraits, pour ne pas la refaire (revue du 29/10/2026 : la lecture
        peut couter cher sur une longue reponse, elle etait faite deux fois).
        """
        if occurrences is None:
            occurrences = lire_les_citations(answer, extraits)

        # L'unite dont on donne le statut : l'article, sous le numero de son
        # extrait — celui sous lequel le modele a lu sa ligne de statut — ou le
        # texte entier.
        def unite(occurrence: Occurrence) -> Optional[Tuple]:
            extrait = occurrence.extrait
            if extrait is not None:
                if occurrence.forme == "article" and extrait.number:
                    return (extrait.reference, extrait.number, extrait.content, extrait.section)
                return (extrait.reference, None, None, None)
            if occurrence.texte_nomme is not None and occurrence.texte_nomme.reference:
                return (occurrence.texte_nomme.reference, None, None, None)
            return None

        lois_citees = {o.extrait.law_id for o in occurrences if o.extrait is not None}
        non_cites: Dict[int, Tuple[ChunkResult, List[str]]] = {}
        for extrait in extraits:
            if extrait.law_id in lois_citees:
                continue
            _, numeros = non_cites.setdefault(extrait.law_id, (extrait, []))
            numero = _citable_article_number(extrait.number)
            if numero and numero not in numeros:
                numeros.append(numero)

        unites = {u for u in map(unite, occurrences) if u is not None}
        unites.update((e.reference, None, None, None) for e, _ in non_cites.values())
        a_resoudre = set()
        for u in unites:
            a_resoudre.update(references_a_resoudre(liens_qui_touchent(*u)))
        a_resoudre.update(
            o.texte_nomme.reference for o in occurrences
            if o.texte_nomme is not None and o.texte_nomme.reference
        )
        ids = await self._identifiants_des_textes(a_resoudre)
        statuts = {u: statut_prouve(*u, ids=ids) for u in unites}

        def statut_de(cle: Optional[Tuple]) -> StatutProuve:
            return statuts.get(cle) or StatutProuve()

        vers_js = indices_utf16(answer)
        citations: List[CitationPositionnee] = []
        introuvables: List[CitationIntrouvable] = []
        cites: List[SourceCitee] = []
        vues = set()
        lois_citees_par_article = {
            o.extrait.law_id for o in occurrences if o.forme == "article" and o.extrait is not None
        }
        for occurrence in occurrences:
            bornes = {
                "debut": vers_js(occurrence.debut),
                "fin": vers_js(occurrence.fin),
                "texte_cite": answer[occurrence.debut:occurrence.fin],
            }
            if occurrence.introuvable:
                introuvables.append(CitationIntrouvable(**bornes, raison=occurrence.raison))
                continue
            extrait = occurrence.extrait
            if extrait is None:
                # Texte que seule une ligne de statut nomme : lie s'il est en
                # base, ni lie ni signale sinon.
                nomme = occurrence.texte_nomme
                law_id = ids.get(nomme.reference) if nomme and nomme.reference else None
                if law_id is not None:
                    citations.append(CitationPositionnee(
                        **bornes, law_id=law_id, reference=nomme.reference,
                        titre=_SANS_ARTICLE_DEFINI.sub("", nomme.titre),
                        date_du_texte=_date_affichee(nomme.date, nomme.titre),
                        statut=statut_de(unite(occurrence)),
                    ))
                continue
            article = occurrence.article if occurrence.forme == "article" else None
            statut = statut_de(unite(occurrence))
            citations.append(CitationPositionnee(
                **bornes, law_id=extrait.law_id, article=article,
                article_id=extrait.article_id if occurrence.forme == "article" else None,
                reference=extrait.reference, titre=extrait.law_title,
                date_du_texte=_date_affichee(extrait.publication_date, extrait.law_title),
                statut=statut,
            ))
            # Une source par article cite ; un texte cite sans article n'en est
            # une que si aucun de ses articles ne l'est deja.
            if occurrence.forme == "article":
                cle = occurrence.cle
            elif extrait.law_id in lois_citees_par_article:
                continue
            else:
                cle = (extrait.law_id, None)
            if cle in vues:
                continue
            vues.add(cle)
            cites.append(SourceCitee(
                law_id=extrait.law_id,
                law_reference=extrait.reference,
                law_title=extrait.law_title,
                article_number=article,
                article_id=extrait.article_id if occurrence.forme == "article" else None,
                excerpt=(
                    (extrait.content or extrait.excerpt or "")[:300]
                    if occurrence.forme == "article" else ""
                ),
                relevance_score=extrait.relevance_score,
                date_du_texte=_date_affichee(extrait.publication_date, extrait.law_title),
                statut=statut,
            ))

        textes = [
            TexteConsulte(
                law_id=extrait.law_id,
                law_reference=extrait.reference,
                law_title=extrait.law_title,
                date_du_texte=_date_affichee(extrait.publication_date, extrait.law_title),
                articles=numeros,
                statut=statut_de((extrait.reference, None, None, None)),
            )
            for extrait, numeros in non_cites.values()
        ]
        if introuvables:
            logger.info(
                "🔎 %d citation(s) sans texte consulte : %s",
                len(introuvables), [c.texte_cite for c in introuvables],
            )
        return {
            "citations": citations,
            "citations_introuvables": introuvables,
            "articles_cites": cites,
            "textes_consultes_non_cites": textes,
            # Les institutions vers lesquelles la reponse renvoie : leurs
            # coordonnees viennent de l'annuaire, jamais du modele.
            "contacts": contacts_pour(answer),
        }

    @staticmethod
    def _progression(
        etape: str, langue: str, extraits: Optional[List[ChunkResult]] = None
    ) -> Progression:
        """
        Un evenement de progression, avec sa ligne lisible.

        `textes_trouves` porte les textes dans l'ordre du classement, avec le
        nombre d'articles de chacun : le front en tire « 8 articles trouvés :
        Code du travail, … » avec le nom court qu'il donne deja aux sources.

        Des ARTICLES, et non des extraits : les morceaux d'un article long
        (« 55.1 », « 55.2 ») n'en font qu'un, et la ligne niveau loi d'un texte
        n'en est pas un. Compter les extraits annoncait « 8 articles » pour
        trois articles decoupes.
        """
        libelles = _LIBELLES_DE_PROGRESSION.get(langue, _LIBELLES_DE_PROGRESSION["fr"])
        if etape != "textes_trouves":
            return Progression(etape=etape, libelle=libelles[etape])
        par_loi: Dict[int, TexteTrouve] = {}
        articles: Dict[int, set] = {}
        for extrait in extraits or []:
            if extrait.law_id not in par_loi:
                par_loi[extrait.law_id] = TexteTrouve(
                    law_id=extrait.law_id,
                    law_reference=extrait.reference,
                    law_title=extrait.law_title,
                    date_du_texte=_date_affichee(extrait.publication_date, extrait.law_title),
                    nb_articles=0,
                )
            article = article_compte(extrait.number)
            if article is not None:
                articles.setdefault(extrait.law_id, set()).add(article)
        for law_id, trouve in par_loi.items():
            trouve.nb_articles = len(articles.get(law_id, ()))
        nb_articles = sum(trouve.nb_articles for trouve in par_loi.values())
        nb_textes = len(par_loi)
        if nb_textes == 0:
            libelle = libelles["aucun"]
        elif nb_articles == 0:
            libelle = libelles["textes_seuls"](nb_textes)
        else:
            libelle = libelles["textes_trouves"](nb_articles, nb_textes)
        return Progression(
            etape=etape, libelle=libelle, nb_articles=nb_articles, nb_textes=nb_textes,
            textes=list(par_loi.values()),
        )

    def _calculate_confidence(
        self,
        answer: str,
        citations: List[Citation],
        search_results: List
    ) -> float:
        """
        Calculate confidence score based on multiple factors.

        Factors:
        1. Number of citations (more = better)
        2. Average relevance of cited docs
        3. Answer length (too short/long = lower confidence)
        4. Presence of hedge words (probably, maybe = lower)

        Returns:
            Confidence score 0.0-1.0
        """
        score = 0.0

        # Factor 1: Citations (0-0.4)
        if citations:
            citation_score = min(len(citations) / 3, 1.0) * 0.4
            score += citation_score

        # Factor 2: Relevance (0-0.3)
        if citations:
            avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
            score += avg_relevance * 0.3
        elif search_results:
            # No citations but have results.
            # Le diviseur est borne : un objet qui se dit non vide mais dont
            # len() vaut 0 (une doublure, par exemple) provoquait une
            # ZeroDivisionError remontee en erreur de service.
            top = list(search_results[:3])
            if top:
                avg_relevance = sum(r.relevance_score for r in top) / len(top)
                score += avg_relevance * 0.2

        # Factor 3: Answer length (0-0.2)
        answer_len = len(answer.split())
        if 50 <= answer_len <= 500:
            score += 0.2
        elif 30 <= answer_len < 50 or 500 < answer_len <= 700:
            score += 0.1

        # Factor 4: Hedge detection (0-0.1)
        hedge_words = ["peut-être", "probablement", "possiblement", "éventuellement"]
        hedge_count = sum(1 for word in hedge_words if word in answer.lower())
        if hedge_count == 0:
            score += 0.1
        elif hedge_count <= 2:
            score += 0.05

        return round(min(score, 1.0), 2)

    async def _valider_avant_la_generation(self) -> None:
        """
        Valide la transaction en cours, et rend sa connexion au pool, avant
        d'appeler le modele.

        POURQUOI. La recherche validait jusqu'ici la transaction de cette
        session a chaque question : son journal y faisait un COMMIT. La
        conversation creee etait donc enregistree, et la connexion rendue,
        avant la generation. Le journal part desormais sur sa propre connexion
        (search_service._ecrire_apres_la_reponse) : sans ce COMMIT, la
        transaction ouverte au chargement de la conversation restait ouverte
        pendant toute la generation, plusieurs secondes, parfois une minute.
        Chaque chat en cours tenait une des dix connexions du pool — et une
        connexion serveur derriere le pooler de Neon, en mode transaction —, et
        un echec de la generation annulait la conversation creee.
        """
        await self.db.commit()

    async def _save_interaction(
        self,
        conversation: Conversation,
        question: str,
        answer: str,
        citations: List[Citation],
        confidence: float,
        retrieval_time_ms: int,
        generation_time_ms: int,
        *,
        titrer: bool = True,
    ):
        """
        Save question and answer to database.

        `titrer` est NOMME et vaut True par defaut : les trois appelants
        historiques (`ask`, `ask_stream`, `_handle_no_results`) ne changent pas
        de comportement. Seul le chemin conversationnel le passe a False, pour
        que la barre laterale ne se remplisse pas de conversations intitulees
        « bonjour » et « comment vas tu ? ». Le titre reste alors a None
        jusqu'a la premiere vraie question, et le front sait deja afficher un
        repli pour un titre nul.
        """
        # Titre de la conversation, ecrit UNE SEULE FOIS. `_save_interaction`
        # est le point de passage commun a `ask`, `ask_stream` et
        # `_handle_no_results` : le poser ici couvre les trois chemins.
        if titrer and conversation.title is None:
            conversation.title = titre_de_conversation(question)

        # User message
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=question
        )
        self.db.add(user_msg)

        # Assistant message
        sources_json = [c.model_dump() for c in citations]
        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources=sources_json,
            confidence=confidence,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms
        )
        self.db.add(assistant_msg)

        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()

        await self.db.commit()
        logger.info("💾 Saved interaction to database")

        # Update persona interaction metrics
        try:
            from app.services.persona_service import PersonaService
            persona_service = PersonaService(self.db)
            await persona_service.update_interaction_metrics(
                conversation_id=conversation.id,
                persona=conversation.persona,
                language=conversation.language
            )
            logger.debug("📊 Updated persona interaction metrics")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update persona metrics: {e}")

    def _message_sans_resultat(self, request: RAGRequest) -> str:
        """
        La reponse sans texte : celle de la barriere quand elle a refuse le
        contexte, celle d'une recherche vide sinon.
        """
        if self._recuperation.refus:
            logger.info("🚧 Barriere du chat : reponse sans appel au modele")
            # Volet en panne, ou servi par l'index (la recherche exacte n'a pas
            # repondu a temps) : reessayer, et non reformuler.
            return message_sans_texte_pertinent(
                request.language,
                volet_en_panne=self._recuperation.chemin in (
                    CHEMIN_ECHEC, CHEMIN_ABSENT, *VOLETS_SERVIS_PAR_L_INDEX
                ),
                relance=self._recuperation.question_jointe is not None,
            )
        return message_sans_resultat(request.language, request.persona)

    async def _handle_no_results(
        self,
        request: RAGRequest,
        conversation: Conversation,
        retrieval_time_ms: int
    ) -> RAGResponse:
        """
        Handle case when no documents found.

        Recoit la conversation deja chargee par `ask`. Elle la rechargeait :
        sans session_id, ce second appel creait une DEUXIEME conversation, et
        laissait la premiere vide.

        C'est aussi la reponse d'un refus de la barriere (_retenir_le_pertinent),
        avec son propre message.
        """
        no_results_answer = self._message_sans_resultat(request)

        await self._save_interaction(
            conversation,
            request.question,
            no_results_answer,
            [],
            0.0,
            retrieval_time_ms,
            0
        )

        return RAGResponse(
            answer=no_results_answer,
            confidence=0.0,
            sources=[],
            session_id=conversation.session_id,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=0,
            total_time_ms=retrieval_time_ms,
            persona=request.persona
        )
