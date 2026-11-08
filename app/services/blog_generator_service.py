"""
Rédaction d'un article du blog par le modèle, à partir des SEULS textes de la base.

POURQUOI CE SERVICE A ÉTÉ REFAIT. Essai réel du 29/10/2026 (data/blog-verif,
dépôt du backend) : l'article généré sur l'expulsion d'un locataire avait trois
citations, toutes fausses. Deux noms de textes étaient inventés à partir des
cotes internes, l'« article 1 » du Code pénal (sa table des matières) était
cité pour la filouterie de loyers, le corps citait les articles 286 et 312 du
Code pénal pour la violation de domicile (c'est le 299), et reprenait un préavis
du droit français. Les causes, dans l'ancien code :
- les sources arrivaient au modèle sous la forme « PRC-4722 — Article ? » :
  la cote à la place du titre, aucun numéro, un article par loi, 300
  caractères sur un succès du cache, et sans filtre sur les textes publiés ;
- sans source (recherche vide, en panne, ou requête refusée au-delà de 500
  caractères), le prompt disait « Utilise les dispositions officielles
  applicables » : le modèle écrivait de mémoire ;
- le modèle écrivait lui-même le texte, le numéro et l'extrait de chaque
  citation, recopiés tels quels, et le gabarit lui suggérait « law_id 1,
  article 12 » ;
- une réponse coupée ou illisible devenait un faux article publiable.

CE QUI LES FERME, DANS L'ORDRE DU PARCOURS.
1. La recherche du CHAT (RAGService._retrieve_chunks) : recherche sémantique
   EXACTE, sans l'index HNSW ; tableaux de programmes des lois de finances
   retirés, sauf sujet budgétaire ; textes publiés seulement ; mots-clés pour
   le plein texte, sujet entier pour le sens. Ni journal de recherche (ce n'est
   pas la question d'un lecteur), ni cache.
2. Une PREUVE DE PERTINENCE plus stricte que celle du chat (tour 2) : le sujet
   n'est traité que si un extrait au moins en est très proche par le sens, ou
   proche ET porteur d'un mot distinctif du sujet. Sinon 422 « sans_source »,
   et le modèle n'est pas appelé (voir SEUILS_DU_BLOG).
3. Chaque extrait donné au modèle est UN article (un PASSAGE d'une ligne de la
   base, tour 2) : le découpage a collé des articles dans le morceau d'un
   autre (les articles 36 à 39 du Code du travail dans « 35.1 »), et un morceau
   suivant (« 35.2 ») commence par la FIN du dernier article collé dans le
   précédent. Chaque passage porte le numéro de l'article dont il est le
   texte ; la suite d'un article est annoncée et citée comme telle ; les
   annotations de jurisprudence d'un code annoté ne sont pas du texte de loi
   et ne sont jamais données comme tel.
4. Le modèle ne rend, pour ses citations, que des NUMÉROS d'extraits. Le
   serveur construit chaque citation à partir de la base : texte, numéro,
   intitulé du texte, début MOT POUR MOT du passage, lien.
5. La prose est relue : toute mention d'article qu'aucun extrait gardé ne porte
   (pluriels, intervalles, « premier », sans nom de texte, en anglais :
   mentions_d_articles), un texte nommé qu'aucun extrait ne porte, le droit
   d'un autre pays, sont signalés à l'administrateur.
6. Une réponse coupée, illisible ou incomplète est une erreur (502), jamais un
   article ; et l'article enregistré est un BROUILLON, invisible du public tant
   qu'un administrateur ne l'a pas relu et publié.
7. TOUR 3 : des citations vérifiées ne rendent pas la prose juste. L'essai
   réel du 30/10 a publié, sous quatre cartes « vérifiées » et sans un
   avertissement, une expulsion possible sans juge (bail), une charge de la
   preuve inversée (licenciement) et une explication qui contredit l'article
   1752 du Code civil. Chaque phrase est donc comparée aux extraits par un
   CONTRÔLE IA (app/services/controle_du_billet.py), et chaque partie relue
   par un administrateur, avant toute publication (routes/blog.py). Les
   renvois « (extrait 2) », que le lecteur ne peut pas suivre, sont retirés
   de la prose (sans_renvois_aux_extraits).
8. TOUR 4 : les sources lues par le modèle sont gardées avec l'article
   (sources_des_passages), pour que le contrôle et la relecture se fassent
   contre elles, même des jours après ; celles d'un article qui n'en a pas
   (article d'avant, ou écrit à la main) se reconstruisent par la même
   recherche, à partir de son titre et de son contexte (sources_du_sujet).
"""

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple
from urllib.parse import quote

from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.search import ChunkResult, SearchFilters, SearchRequest
from app.services.article_reference import normalize_number
from app.services.lecture_des_citations import lire_les_citations, porte_l_entete
from app.services.llm import (
    ERREURS_BUDGET,
    ERREURS_LLM,
    ERREURS_QUOTA,
    ERREURS_SATURATION,
    get_llm_service,
)
from app.services.mentions_d_articles import (
    droit_etranger,
    en_toutes_lettres,
    numeros_non_portes,
    textes_nommes_non_portes,
)
from app.services.rag_service import RAGService, langue_de_la_question
from app.services.search_service import (
    CHEMIN_EXACT,
    SearchService,
    VoletsDeLaRecherche,
    question_budgetaire,
)
from app.services.textes_successifs import lignes_en_deux
from app.utils.chunk_refiner import est_pseudo_numero

logger = logging.getLogger(__name__)


# ==================== RÉGLAGES ====================

# Délai de toute la génération (recherche et modèle), au-delà duquel la route
# répond 504. Mistral peut prendre jusqu'à ~120 s pour 4 096 jetons
# (MistralService.delai_hors_flux), plus une attente du limiteur partagé avec
# le chat : l'administrateur ne doit pas attendre plus longtemps sans réponse.
DELAI_GENERATION_S = 150
# Autant d'extraits que le chat (RAGService.TOP_K_CHUNKS).
EXTRAITS_AU_PLUS = 8
# Contexte donné au modèle : un article entier dépasse rarement 3 000
# caractères (TARGET_MAX_CHARS du découpage), le total suit le chat.
CARACTERES_PAR_EXTRAIT = 3_500
CARACTERES_DES_EXTRAITS = 24_000
# Longueur du début d'article recopié dans une citation.
CARACTERES_DE_L_EXTRAIT_CITE = 300
# L'explication d'une citation : la limite de la saisie (CitationIn.title), et
# non 300 : une explication coupée en silence peut perdre sa négation.
CARACTERES_DE_L_EXPLICATION = 500
# L'article fait ~1 500 jetons (essai du 29/10/2026 : 1 527) : de la marge.
GENERATION_MAX_TOKENS = 4096
# Aussi bas que la comparaison : l'aléa ne peut ajouter ici que des règles.
GENERATION_TEMPERATURE = 0.2
MOTS_PAR_MINUTE = 200

# PREUVE DE PERTINENCE DU BLOG (tour 2). Le seuil du chat (0,30 en français,
# RAGService.SEUILS_DE_SIMILARITE) est BAS à dessein : le hors-sujet y est
# d'abord écarté par le classificateur d'intention, que le blog n'a pas. Revue
# du tour 1 : « Recette de cuisine du ndolé » gardait 5 extraits (0,30-0,33),
# « Le SMIG au Cameroun c'est combien ? » 5 extraits de lois de finances
# (0,44-0,46), et le modèle était appelé sur des textes qui ne traitent pas le
# sujet. Un seuil seul ne sépare pas : « Les horaires des trains entre Douala et
# Yaoundé » atteint 0,440 sur la loi ferroviaire, le sujet légitime « Le
# harcèlement sur les réseaux sociaux et la loi sur la cybersécurité » 0,471.
#
# Le sujet n'est traité que si UN extrait au moins est FORT :
# - similarité exacte >= SEUILS_DU_BLOG (le sens seul suffit), ou
# - similarité >= SEUILS_ANCRES_DU_BLOG ET l'article porte un mot DISTINCTIF
#   du sujet (lexème présent dans moins de MOTS_DISTINCTIFS_AU_PLUS articles
#   de la base : « expulsion », « préavis », « SMIG » ; pas « Cameroun », « loi »,
#   « droit »), ou tous ses mots-clés (plein texte strict du chat).
# Une fois le sujet prouvé, restent aussi les extraits ancrés au-dessus du
# seuil du chat : ils partagent un mot distinctif du sujet.
#
# Calibré le 30/10/2026 sur jurix_dev avec ce code même
# (data/blog/tour2/calibrage_pertinence.py relève les extraits,
# mesure_pertinence.py rejoue _extraits_pertinents ; recherche exacte du blog,
# voyage-4-large), 98 sujets : 46 juridiques, 51 hors droit, et le SMIG
# « c'est combien ? », que les extraits trouvés (lois de finances, solde des
# fonctionnaires) ne traitent pas.
# - français : 35 sujets juridiques sur 38 passent ; le plus faible,
#   « Le harcèlement sur les réseaux sociaux… », à 0,471 ancré (MARGE 0,016).
#   Hors droit : 0 sur 31 ; le plus fort ancré, « Les horaires des trains
#   entre Douala et Yaoundé », à 0,440 (MARGE 0,015) ; sans ancre, le SMIG à
#   0,460 (MARGE 0,020), le hors-droit 0,437 (MARGE 0,043). Refusés :
#   « Mon voisin construit sur mon terrain » (0,435), « Les heures
#   supplémentaires et leur paiement » (0,449, servi par un tableau comptable
#   de la CIMA) et « Le vol simple et ses peines » (0,307 : la recherche ne
#   trouve pas l'article du vol) : l'administrateur reformule avec les mots de
#   la loi.
# - anglais (le corpus est français : presque aucun mot anglais n'y ancre) :
#   6 sujets juridiques sur 8 ; le plus faible 0,437 (MARGE 0,017) ; hors
#   droit 0 sur 20, le plus fort 0,387 (MARGE 0,033). Refusés : « Unfair
#   dismissal… » (0,415), « Can a landlord evict… » (0,383).
# Les marges sont minces : les distributions se chevauchent (le chat le
# constate aussi). Dans le doute, refuser : un sujet refusé se reformule, un
# article écrit sur des textes qui ne traitent pas le sujet l'est de mémoire.
# Un ajout de textes au corpus ou un changement de modèle d'embeddings impose
# de relancer la mesure.
SEUILS_DU_BLOG = {"fr": 0.48, "en": 0.42}
SEUILS_ANCRES_DU_BLOG = {"fr": 0.455, "en": 0.42}
MOTS_DISTINCTIFS_AU_PLUS = 1000

# Rubriques que le front sait traduire (clés blog.rubrique.* de
# src/lib/stores/language.ts). La rubrique était libre (« Immobilier &
# Logement ») : un filtre par variante, et des noms que l'interface anglaise
# affichait en français.
RUBRIQUES = (
    "Immobilier", "Travail", "Famille", "Affaires", "Numérique", "Fiscalité",
    "Droit pénal", "Consommation", "Santé", "Actualité juridique",
)
RUBRIQUE_PAR_DEFAUT = "Actualité juridique"


# ==================== ERREURS ====================


class ErreurDeGeneration(Exception):
    """
    Échec de la génération, avec la réponse HTTP qui lui revient. La route la
    traduit telle quelle (statut, code, message en français, Retry-After).
    """

    statut = 500
    code = "erreur_interne"
    retry_after: Optional[int] = None


class SansSourceError(ErreurDeGeneration):
    """Aucun texte publié et pertinent : le modèle n'est pas appelé."""

    statut = 422
    code = "sans_source"


class RechercheIndisponibleError(ErreurDeGeneration):
    """Recherche en panne, ou volet sémantique non vérifiable : réessayer."""

    statut = 503
    code = "recherche_indisponible"
    retry_after = 10


class QuotaError(ErreurDeGeneration):
    statut = 429
    code = "quota"
    retry_after = 60


class SaturationError(ErreurDeGeneration):
    statut = 503
    code = "saturation"
    retry_after = 10


class ReponseInvalideError(ErreurDeGeneration):
    """Réponse coupée, illisible ou incomplète : aucun article n'est enregistré."""

    statut = 502
    code = "reponse_invalide"


class CitationInvalide(ValueError):
    """Une citation que la base ne porte pas. La route la traduit en 422."""


# ==================== ADRESSE (SLUG) ====================

SLUG_LONGUEUR_MAX = 80
SLUG_VALIDE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# Lettres que la décomposition NFKD ne ramène pas à l'ASCII : sans elles,
# « main-d'œuvre » donnait « main-duvre ».
_LIGATURES = str.maketrans({"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae", "ß": "ss"})


def slugify(texte: Optional[str], repli: bool = True) -> str:
    """
    L'adresse d'un article : minuscules ASCII, chiffres et tirets, 80
    caractères au plus, coupée entre deux mots.

    POURQUOI. Le slug du modèle ou de l'administrateur était gardé tel quel :
    « Licenciement abusif : ce que dit l'article 34 ? » (essai hors ligne),
    « Bail/Expulsion Été » (essai réel, enregistré puis introuvable). Le front
    insère le slug dans ses liens sans l'encoder : un « ? », un « # » ou un
    « / » rendait l'article injoignable. Toute adresse passe donc par ici, à la
    génération comme à la création et à la modification.

    `repli` : sans aucune lettre ni chiffre (« ??? »), « article-AAAA-MM-JJ » ;
    sinon "" (l'appelant refuse une adresse explicite vide).
    """
    replie = unicodedata.normalize("NFKD", (texte or "").translate(_LIGATURES))
    ascii_seul = replie.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_seul).strip("-")
    if len(slug) > SLUG_LONGUEUR_MAX:
        coupe = slug.rfind("-", 0, SLUG_LONGUEUR_MAX + 1)
        slug = (slug[:coupe] if coupe > 0 else slug[:SLUG_LONGUEUR_MAX]).strip("-")
    if not slug and repli:
        return f"article-{date.today().isoformat()}"
    return slug


# ==================== TEXTE BRUT ====================

_GRAS = re.compile(r"\*\*+|__+")
_TITRE_MARKDOWN = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.MULTILINE)
_PUCE_MARKDOWN = re.compile(r"^[ \t]*[-*•][ \t]+", re.MULTILINE)
_CODE_MARKDOWN = re.compile(r"`+")
_ESPACES = re.compile(r"[ \t]{2,}")
_LIGNES_VIDES = re.compile(r"\n{3,}")
# Les cotes internes des textes (laws.reference), les formes que le front
# reconnaît aussi (estUneCoteInterne). Ce n'est pas un nom de texte : le
# lecteur ne peut rien en faire.
_COTE_INTERNE = re.compile(
    r"\s*\(\s*(?P<entre>(?:PRC|LOI|ORD|DEC|OHADA|CODE|CIMA)(?:-[A-Z0-9]+)+)\s*\)"
    r"|\b(?P<seule>(?:PRC|LOI|ORD|DEC|OHADA|CODE|CIMA)(?:-[A-Z0-9]+)+)\b"
)


def texte_brut(texte: Optional[str]) -> str:
    """
    Le texte sans les marques de Markdown que le modèle écrit malgré la
    consigne : le front échappe tout ce qu'il affiche (aucun {@html}), et les
    « ** » de l'essai réel s'affichaient tels quels, jusque dans la meta
    description. Les retours à la ligne restent.
    """
    resultat = _GRAS.sub("", texte or "")
    resultat = _TITRE_MARKDOWN.sub("", resultat)
    resultat = _PUCE_MARKDOWN.sub("", resultat)
    resultat = _CODE_MARKDOWN.sub("", resultat)
    resultat = "\n".join(ligne.strip() for ligne in resultat.splitlines())
    resultat = _ESPACES.sub(" ", resultat)
    return _LIGNES_VIDES.sub("\n\n", resultat).strip()


# Les renvois aux NUMÉROS des extraits : « (extrait 2) », « (extraits 5 et 9) »,
# « (cf. extrait 3) », « [2] ». Essai réel du 30/10/2026 (licenciement) : cinq
# points clés publiés finissaient par « (extrait 2) », « (extraits 5 et 9) »…
# Le lecteur voit des cartes sans numéro : ces renvois ne mènent nulle part,
# et laissent croire à une source qu'il ne peut pas lire. Ils sont retirés ;
# le contrôle IA dit, lui, sur quels extraits chaque phrase repose.
_RENVOI_AUX_EXTRAITS = re.compile(
    r"[ \t]*[\(\[][ \t]*(?:(?:cf\.?|voir|v\.|selon)[ \t]+)?(?:l['’][ \t]*|les[ \t]+)?"
    r"(?:extraits?|sources?)[ \t]*(?:n[°o]s?\.?[ \t]*)?\[?\d+\]?"
    r"(?:[ \t]*(?:,|;|et|à|-|–|&|and)[ \t]*\[?\d+\]?)*[ \t]*[\)\]]"
    r"|[ \t]*\[\d+(?:[ \t]*(?:,|;|et|-|–)[ \t]*\d+)*\]",
    re.IGNORECASE,
)


def sans_renvois_aux_extraits(texte: Optional[str]) -> str:
    """
    Le texte sans « (extrait 2) », « (extraits 5 et 9) », « [3] ». L'espace
    qui précède le renvoi part avec lui : « respecté (extrait 2). » devient
    « respecté. », et rien d'autre ne change (l'espace française avant « : »
    reste).
    """
    return _RENVOI_AUX_EXTRAITS.sub("", texte or "").strip()


def prose_publique(texte: Optional[str]) -> str:
    """
    Texte brut (sans Markdown) et sans renvoi aux numéros des extraits.

    Les blancs sont remis en ordre APRÈS le retrait des renvois (tour 4) : un
    renvoi seul sur sa ligne, ou en tête de ligne, laissait trois sauts de
    ligne ou une espace de tête, que le formulaire renvoyé, relu ici, ramenait
    à la normale ; le texte enregistré changeait alors à chaque enregistrement
    (revue du tour 3). Appliquée deux fois, elle rend le même texte.
    """
    return texte_brut(sans_renvois_aux_extraits(texte_brut(texte)))


def sans_cote(texte: str) -> Tuple[str, List[str]]:
    """Le texte sans les cotes internes (« (PRC-4722) »), et les cotes retirées."""
    retirees: List[str] = []

    def retirer(trouve: "re.Match") -> str:
        retirees.append(trouve.group("entre") or trouve.group("seule"))
        return ""

    resultat = _COTE_INTERNE.sub(retirer, texte or "")
    return _ESPACES.sub(" ", resultat).replace(" ,", ",").replace(" .", ".").strip(), retirees


def couper(texte: str, longueur: int) -> str:
    """Au plus `longueur` caractères, coupé entre deux mots."""
    if len(texte) <= longueur:
        return texte
    coupe = texte.rfind(" ", 0, longueur)
    return texte[:coupe if coupe > longueur // 2 else longueur].rstrip(" ,;:")


def duree_de_lecture(textes: Iterable[str]) -> str:
    """« N min » à 200 mots par minute : le gabarit du modèle disait toujours « 3 min »."""
    mots = sum(len((t or "").split()) for t in textes)
    return f"{max(1, round(mots / MOTS_PAR_MINUTE))} min"


def mots_cles_propres(tags: Iterable[Any], au_plus: int = 8) -> List[str]:
    """Mots-clés : des chaînes courtes, sans « # », sans doublon."""
    vus: Set[str] = set()
    resultat: List[str] = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        propre = texte_brut(tag).lstrip("#").strip()[:50]
        if propre and propre.lower() not in vus:
            vus.add(propre.lower())
            resultat.append(propre)
    return resultat[:au_plus]


def rubrique_connue(nom: Optional[str]) -> str:
    """
    La rubrique de RUBRIQUES que le modèle a nommée, en entier ou par l'une de
    ses parties (« Travail & Emploi » : Travail), sinon la rubrique de repli.
    """
    def cle(texte: str) -> str:
        return slugify(texte, repli=False)

    for partie in [nom or "", *re.split(r"\s*(?:&|,|/|\bet\b)\s*", nom or "")]:
        voulue = cle(partie)
        trouvee = next((r for r in RUBRIQUES if voulue and cle(r) == voulue), None)
        if trouvee:
            return trouvee
    return RUBRIQUE_PAR_DEFAUT


# ==================== ARTICLES : PASSAGES, EXTRAIT MOT POUR MOT ====================

# Un article long est rangé en morceaux « 55.1 », « 55.2 »… (chunk_refiner).
# Relevé sur jurix_dev le 29/10/2026 : 6 282 numéros de cette forme dans les
# textes publiés, et AUCUN dont le texte porte son propre en-tête
# « Article 55.1 » : ce sont des morceaux de l'article 55. Le lecteur doit lire
# « article 55 », que la page du texte sait ouvrir (?article=55).
_MORCEAU_D_ARTICLE = re.compile(r"^\s*(\d+)\.\d+\s*$")


def numero_affiche(numero: Optional[str]) -> str:
    """« 1751 » pour le morceau « 1751.1 » ; tout autre numéro tel quel."""
    trouve = _MORCEAU_D_ARTICLE.match(numero or "")
    return trouve.group(1) if trouve else (numero or "").strip()


# Un morceau qui n'ouvre pas son article (« 1751.2 », « 35.12 »).
_MORCEAU_SUIVANT = re.compile(r"^\s*\d+\.(?:[2-9]|\d{2,})\s*$")

# L'en-tête d'un article dans le texte d'un morceau : « Article 39.- »,
# « ARTICLE 6 : », « Art. 5 — », « Art..5- » (OCR). Un séparateur après le
# numéro : « l'article 1736 du Code civil », cité dans une annotation, n'en
# est pas un. Sert à SITUER les articles d'un morceau ; ce qui décide qu'un
# morceau PORTE un article reste porte_l_entete, la preuve de la lecture des
# citations et de l'aperçu d'un article (routes/laws).
_EN_TETE_D_ARTICLE = re.compile(
    r"(?<![\w’'])(?:article|art\.{1,2})\s*(?P<numero>\d+(?:\s*(?:er|ère|re))?(?:[-.]\d+)?)"
    r"(?![\w.]?\d)(?:\.?\s*\(\d+\))?\s*(?:\((?:nouveau|nouvelle|new)\)\s*)?(?:\.\s*[-–—]|[-–—:])"
    # « Article 434-8 » (sommaire du Code CIMA) : le tiret y fait partie du
    # numéro, ce n'est pas l'en-tête d'un article 434.
    r"(?!\d)",
    re.IGNORECASE,
)


def entetes_colles(
    contenu: Optional[str], numero_propre: Optional[str], modificatif: bool = True
) -> List[Tuple[int, str]]:
    """
    (place, numéro) de chaque AUTRE article dont le morceau porte l'en-tête,
    dans l'ordre du texte.

    POURQUOI. Le découpage de la base a collé des articles dans le morceau d'un
    autre : au Code du travail, le morceau « 35.1 » porte les articles 36 à 39
    (relevé sur jurix_dev le 29/10/2026), dont l'article 39, celui des
    dommages-intérêts pour licenciement abusif. Annoncé « Article 35 », ce
    morceau faisait citer l'article 35 pour une règle de l'article 39.

    `modificatif` : le texte modifie un AUTRE texte (est_modificatif). Voir
    est_un_article_cite : la marque « (nouveau) » n'y dit une citation que là.
    Par défaut, dans le doute, vrai.
    """
    contenu = contenu or ""
    propre = normalize_number(numero_affiche(numero_propre))
    entetes: List[Tuple[int, str]] = []
    dernier = _rang_d_article(propre)
    for trouve in _EN_TETE_D_ARTICLE.finditer(contenu):
        numero = normalize_number(trouve.group("numero"))
        if not numero or numero == propre or not porte_l_entete(contenu, numero):
            continue
        # « Article 5 (nouveau).- » : le texte d'une loi modificative CITE
        # l'article qu'elle réécrit ; ce n'est pas un article d'elle collé là.
        if est_un_article_cite(contenu, trouve.start(), trouve.end(), modificatif):
            continue
        # Un article collé SUIT le précédent : « Article 390000 : », relevé
        # dans un décret d'avancement de jurix_dev, est une ligne budgétaire.
        rang = _rang_d_article(numero)
        if dernier is not None and rang is not None and not dernier < rang <= dernier + 100:
            continue
        entetes.append((trouve.start(), numero))
        dernier = rang if rang is not None else dernier
    return entetes


_REECRIT = re.compile(r"\((?:nouveau|nouvelle|new)\)", re.IGNORECASE)
# Un texte qui en MODIFIE un autre : « Décret … modifiant et complétant
# certaines dispositions du décret … ». Pas un texte consolidé : « Loi … portant
# Code électoral, modifiée et complétée par … ».
_MODIFICATIF = re.compile(r"\b(?:modifiant|compl[ée]tant|portant\s+modification)\b", re.IGNORECASE)


def est_modificatif(titre_du_texte: Optional[str]) -> bool:
    return bool(_MODIFICATIF.search(titre_du_texte or ""))
# La marque APRÈS le séparateur, alinéa compris : « ARTICLE 2.- (nouveau) »,
# « ARTICLE 5.- (4) (nouveau) ».
_REECRIT_APRES = re.compile(r"[ \t]*(?:\([ \t]*\d+[ \t]*\)[ \t]*)?\((?:nouveau|nouvelle|new)\)", re.IGNORECASE)
# Ce qui annonce une CITATION juste avant l'en-tête : un guillemet OUVRANT
# (« - «ARTICLE 2.- »), un guillemet droit après deux-points, ou « ainsi
# qu'il suit : », « comme suit : », aux espaces et tirets près. Un guillemet
# droit seul est ambigu (il ferme aussi) : il ne compte qu'après « : ».
_CITATION_ANNONCEE = re.compile(
    r"(?:[«“]|:[\s\-–—]*\"|(?:ainsi[ \t]+qu['’][ \t]*il[ \t]+suit|comme[ \t]+suit"
    r"|ainsi[ \t]+r[ée]dig[ée]e?s?|se[ \t]+lit[ \t]+(?:d[ée]sormais[ \t]+)?(?:comme[ \t]+suit|ainsi))"
    r"[ \t]*:)[\s\-–—«“\"]*$",
    re.IGNORECASE,
)


def est_un_article_cite(contenu: str, debut: int, fin: int, modificatif: bool = True) -> bool:
    """
    L'en-tête trouvé entre `debut` et `fin` est-il celui d'un article qu'un
    texte CITE (la nouvelle rédaction d'un article d'un autre texte), et non un
    article de ce texte collé là ?

    - Une citation ANNONCÉE (guillemet ouvrant, « ainsi qu'il suit : »…) en est
      une, dans tout texte.
    - La marque « (nouveau) », avant ou après le séparateur, n'en dit une que
      dans un texte MODIFICATIF (`modificatif`). Dans un texte consolidé, elle
      marque un article du texte lui-même, réécrit depuis : au Code électoral
      de jurix_dev (« Loi N°2012/001 … portant Code électoral, modifiée et
      complétée par … »), « Art. 166 - (nouveau) (1) Le candidat titulaire… »
      est son article 166, collé dans la ligne 165 ; l'écarter versait son
      texte dans la carte de l'article 165 (sonde du tour 3 sur jurix_dev,
      data/blog/tour3/sonde-r2-r3.json : 4 en-têtes du Code électoral).

    POURQUOI. Revue du tour 2 : « Les dispositions de l'Article 2 du décret
    N°2012/350 … sont modifiées et complétées ainsi qu'il suit : - «ARTICLE
    2.- (nouveau) La Juridiction de compétence dudit Consulat… » » (décret
    2021/597, jurix_dev, ligne 56289) devenait l'article 2 du décret 2021/597,
    sous une carte « vérifiée » ; or son article 2 est « Le présent décret sera
    enregistré… ». La marque « (nouveau) » vient APRÈS le séparateur, que
    _REECRIT ne lisait pas ; et une citation sans la marque (« «ARTiCLE 2.-(1)
    Le classement… », décret 2025/086) n'en a aucune. Sept en-têtes de ce type
    dans sept des 65 textes modificatifs publiés de jurix_dev. Le texte cité
    reste dans le passage de l'article qui le cite (l'article 1er du texte
    modificatif), que le modèle peut citer correctement.
    """
    if _CITATION_ANNONCEE.search(contenu[max(0, debut - 80):debut]):
        return True
    return modificatif and bool(
        _REECRIT.search(contenu, debut, fin) or _REECRIT_APRES.match(contenu, fin)
    )


def _rang_d_article(numero: Optional[str]) -> Optional[int]:
    """Le nombre en tête d'un numéro d'article (« 39 », « 5 » pour « 5-1 »), ou None."""
    trouve = re.match(r"\s*(\d+)", numero or "")
    return int(trouve.group(1)) if trouve else None


def articles_inclus(
    contenu: Optional[str], numero_propre: Optional[str], modificatif: bool = True
) -> List[str]:
    """Les AUTRES articles dont le morceau porte l'en-tête, dans l'ordre, sans doublon."""
    vus: List[str] = []
    for _, numero in entetes_colles(contenu, numero_propre, modificatif):
        if numero not in vus:
            vus.append(numero)
    return vus


def est_citable(extrait: ChunkResult) -> bool:
    """Un article d'un texte publié : ni ligne « niveau loi », ni préambule, ni visas."""
    return (
        extrait.article_id is not None
        and bool((extrait.number or "").strip())
        and not est_pseudo_numero(extrait.number)
        and extrait.status == "published"
    )


# Un code ANNOTÉ (« Code civil applicable au Cameroun, annoté de la
# jurisprudence de la Cour suprême ») range sous chaque article, après son
# texte, des annotations : jurisprudence, doctrine, bibliographie. Relevé sur
# jurix_dev le 30/10/2026 : 29 des 35 morceaux suivants de ce code sont des
# annotations, que l'ancien service présentait au modèle comme « Article
# 1751 (partie d'un article long) » puis citait sous le début de l'article 1751.
# Ce ne sont pas les termes de la loi : elles ne sont jamais données comme tels.
_ANNOTE = re.compile(r"\bannot[ée]e?s?\b", re.IGNORECASE)
_ANNOTATION = re.compile(
    r"^[ \t]*(?:jugements?\s+et\s+arr[êe]ts|jurisprudences?|bibliographie|doctrine)\b"
    r"|^[ \t]*§[ \t]*\d"
    r"|\barr(?:\.|[êe]t)(?!é)\s*(?:n\s*°|n[o\d]|n\b|du\b|\d)"
    r"|\bjuridis\b|\bbull?\.\s*des\s+arr[êe]ts|\blex\s+lata\b|\brev(?:ue)?\.?\s+cam|\bp[ée]nant\b",
    re.IGNORECASE | re.MULTILINE,
)
# L'intitulé d'une partie du texte (« SECT. II Des règles particulières aux
# baux à loyer », « CHAP. IV Des demandes en nullité de mariage ») : ce qui le
# suit n'est plus l'article. En majuscules, en tête de ligne, jamais la
# première ligne du passage.
_STRUCTURE = re.compile(
    r"^[ \t]*(?:SECT(?:ION)?\.?|CHAP(?:ITRE)?\.?|TITRE|LIVRE|SOUS-SECTION)[ \t]+"
    r"(?:[IVXLCDM]+|\d+|PREMIER|PREMI[ÈE]RE|UNIQUE)\b",
    re.MULTILINE,
)
_MOT_DE_TEXTE = re.compile(r"[^\W\d_]{3,}")


def est_annote(titre_du_texte: Optional[str]) -> bool:
    return bool(_ANNOTE.search(titre_du_texte or ""))


def _fin_du_texte(
    contenu: str, debut: int, fin: int, annote: bool, suite: bool
) -> Optional[int]:
    """
    La fin du texte de l'article dans [debut, fin) : avant l'intitulé d'une
    autre partie du texte et, dans un code annoté, avant ses annotations. None
    quand il ne reste rien qui soit le texte d'un article.
    """
    morceau = contenu[debut:fin]
    coupes = [m.start() for m in _STRUCTURE.finditer(morceau) if m.start() > 0][:1]
    if annote:
        marque = _ANNOTATION.search(morceau)
        if marque is not None:
            ligne = morceau.rfind("\n", 0, marque.start()) + 1
            if ligne > 0:
                coupes.append(ligne)
            elif suite:
                # La suite d'un article d'un code annoté qui s'ouvre sur une
                # annotation : rien de ce passage n'est la loi.
                return None
            else:
                # Le texte et la première annotation sur une même ligne : la
                # coupe au début de la phrase qui porte l'annotation.
                phrase = morceau.rfind(". ", 0, marque.start())
                coupes.append(phrase + 2 if phrase >= 0 else 0)
    coupe = min(coupes) if coupes else len(morceau)
    garde = morceau[:coupe].rstrip()
    if len(_MOT_DE_TEXTE.findall(garde)) < 2:
        return None
    return debut + len(garde)


@dataclass(frozen=True)
class Passage:
    """
    UN article, ou la suite d'un article, lu dans une ligne de la base.

    C'est l'unité que lit le modèle et que cite le serveur : une ligne de la
    base peut porter plusieurs articles (articles collés), ou la fin d'un
    article commencé dans la ligne précédente.
    """

    ligne_id: int
    law_id: int
    numero: str
    """Le numéro de l'article dont c'est le texte (« 39 »), tel que le lecteur le lit."""
    debut: int
    fin: int
    suite: bool
    """Le passage ne commence pas au début de l'article : son extrait le dit (« […] »)."""
    propre: bool
    """Le passage est le début de l'article de la ligne elle-même (son intitulé vaut pour lui)."""
    texte: str


def debut_du_corps(contenu: Optional[str], premier: Optional[str]) -> int:
    """
    Où commence le texte PROPRE d'un morceau suivant : après le début de
    l'article qu'il répète, s'il le répète ; 0 sinon.

    POURQUOI. Le découpage répète devant chaque morceau d'un article long les
    200 premiers caractères de son chapeau, puis un retour à la ligne
    (chunk_refiner._split_long_article, LONGUEUR_CHAPEAU_REPETE). Relevé sur
    jurix_dev le 30/10/2026 : « 67.2 » du Code du travail commence par « En
    dehors des prestations prévues à l'article 66… Article 68.- (1) A
    l'exception des professions pou », coupé au milieu d'un mot, puis passe à
    « (3) En cas de résiliation… ». Lue comme du texte suivi, cette tête
    faisait de la reprise du début de l'article 67 la « suite » d'un autre
    article, et de « Article 68.- … pou (3) … » un article 68 qui n'existe pas
    sous cette forme. La tête répétée est reconnue à ce qu'elle ouvre aussi le
    premier morceau (`premier`), et finit sur un retour à la ligne.
    """
    contenu, premier = contenu or "", premier or ""
    commun = 0
    for a, b in zip(contenu, premier):
        if a != b:
            break
        commun += 1
    if commun < 10:
        return 0
    fin_de_ligne = contenu.find("\n", max(0, commun - 1))
    if fin_de_ligne == -1 or fin_de_ligne > commun + 2:
        return 0
    return fin_de_ligne + 1


def decouper(
    contenu: Optional[str],
    numero: Optional[str],
    ouvert: Optional[str],
    annote: bool,
    corps: int = 0,
    modificatif: bool = True,
) -> List[Tuple[str, int, int, bool, bool]]:
    """
    Les passages d'une ligne : (numéro, début, fin, suite, propre).

    - La tête de la ligne (avant le premier article collé) : le début de
      l'article de la ligne (« 55 », « 55.1 ») ; pour un morceau suivant
      (« 55.2 »), la SUITE de l'article `ouvert` à la fin du morceau
      précédent — le dernier article collé dans celui-ci, ou l'article 55.
      `ouvert` inconnu (morceaux manquants), ou code annoté : la tête est
      écartée, jamais rattachée au hasard.
    - Chaque article collé : de son en-tête à l'en-tête suivant.
    - `corps` : début du texte propre d'un morceau suivant (debut_du_corps).
      Ce qui le précède répète le début de l'article : ni passage, ni en-tête.
    """
    contenu = contenu or ""
    entetes = [(p, n) for p, n in entetes_colles(contenu, numero, modificatif) if p >= corps]
    bornes = [place for place, _ in entetes] + [len(contenu)]
    passages: List[Tuple[str, int, int, bool, bool]] = []
    if bornes[0] > corps:
        if not _MORCEAU_SUIVANT.match(numero or ""):
            tete: Optional[Tuple[str, bool, bool]] = (numero_affiche(numero), False, True)
        elif ouvert and not annote:
            tete = (ouvert, True, False)
        else:
            tete = None
        if tete is not None:
            fin = _fin_du_texte(contenu, corps, bornes[0], annote, tete[1])
            if fin is not None:
                passages.append((tete[0], corps, fin, tete[1], tete[2]))
    for rang, (place, numero_colle) in enumerate(entetes):
        fin = _fin_du_texte(contenu, place, bornes[rang + 1], annote, False)
        if fin is not None:
            passages.append((numero_colle, place, fin, False, False))
    return passages


def extrait_verbatim(
    texte: Optional[str], suite: bool = False, longueur: int = CARACTERES_DE_L_EXTRAIT_CITE
) -> str:
    """
    Le DÉBUT MOT POUR MOT du passage, pour la carte de citation.

    Le modèle écrivait lui-même l'« extrait » : une paraphrase (l'article 322
    du Code pénal sans « bail dûment enregistré » ni « un mois après
    sommation »), présentée comme le texte. Ici, rien que la base.

    RIEN N'EST RETIRÉ EN SILENCE (tour 2). La première version retirait en tête
    toute ligne courte sans ponctuation finale, prise pour un intitulé : quand
    l'OCR avait coupé la première phrase, le début de l'article disparaissait
    (Code de procédure pénale, art. 8 : la carte commençait à « innocente
    jusqu'à ce que sa culpabilité… », sans « Toute personne suspectée… est
    présumée » ; 69 articles publiés de jurix_dev dans ce cas). Seuls les
    blancs sont ramenés à une espace ; « […] » marque toute coupe : en fin, le
    texte continue ; en tête, le passage est la SUITE d'un article dont le
    début n'est pas montré.
    """
    entier = " ".join((texte or "").split())
    court = couper(entier, longueur)
    extrait = court if court == entier else f"{court} […]"
    return f"[…] {extrait}" if suite else extrait


def adresse_de_l_article(law_id: int, numero: str) -> str:
    return f"/laws/{law_id}?article={quote(numero, safe='')}"


# ==================== CITATIONS VÉRIFIÉES EN BASE ====================


@dataclass
class DemandeDeCitation:
    """Une citation à vérifier : un texte, et l'article par son identifiant ou son numéro."""

    law_id: int
    article_id: Optional[int] = None
    article_number: Optional[str] = None
    title: str = ""
    # Dans une même ligne, la SUITE d'un article plutôt que son début (et
    # inversement) ; None : le premier passage de ce numéro.
    suite: Optional[bool] = None


_SQL_TEXTES = text("SELECT id, title, reference, status FROM laws WHERE id = ANY(:ids)")
_COLONNES_D_ARTICLE = (
    'a.id, a.law_id, a.number, a.title, a.content, a.kind, a."order" AS ordre'
)
_SQL_ARTICLES_PAR_ID = text(
    f"SELECT {_COLONNES_D_ARTICLE} FROM articles a WHERE a.id = ANY(:ids)"
)
# Comme l'aperçu d'un article (GET /laws/{id}/articles/{numero}) : le numéro
# exact, ses morceaux (« 55 » trouve « 55.1 »), et les écritures de l'article
# premier.
_SQL_ARTICLES_PAR_NUMERO = text(
    f"SELECT {_COLONNES_D_ARTICLE} FROM articles a "
    "WHERE a.law_id = :law_id "
    "  AND (upper(btrim(a.number)) = ANY(:candidats) "
    "       OR split_part(upper(btrim(a.number)), '.', 1) = ANY(:candidats)) "
    'ORDER BY a."order", a.id'
)
_PREMIER = ("1ER", "1ERE", "1ÈRE", "PREMIER", "PREMIERE", "PREMIÈRE")
# Les morceaux d'un article (« 1751 », « 1751.1 », « 1751.2 »…), pour retrouver
# ce qui précède un morceau suivant.
_SQL_MORCEAUX = text(
    f"SELECT {_COLONNES_D_ARTICLE} FROM articles a "
    "WHERE a.law_id = :law_id AND (a.number = :base OR a.number LIKE :morceaux) "
    'ORDER BY a."order", a.id'
)
# Les lignes d'un texte qui nomment un article dans leur contenu : candidates
# pour un article collé dans le morceau d'un autre, confirmées ensuite par
# porte_l_entete. Large à dessein : la preuve vient après.
_SQL_ENGLOBANTS = text(
    f"SELECT {_COLONNES_D_ARTICLE} FROM articles a "
    "WHERE a.law_id = :law_id AND a.content ~* :motif "
    'ORDER BY a."order", a.id LIMIT 20'
)


async def _suite_de(
    db: AsyncSession, law_id: int, ligne: Any, modificatif: bool = True
) -> Tuple[Optional[str], int]:
    """
    Pour le morceau suivant `ligne` (« 35.2 ») : (l'article OUVERT à la fin du
    morceau précédent, début du texte propre de `ligne` — debut_du_corps).

    L'article ouvert est le dernier article collé dans le texte propre des
    morceaux précédents de la même suite, ou l'article de la suite (« 35 »).
    None si la suite est incomplète (un morceau manque, l'ordre saute) : on ne
    sait pas de quel article ce texte est la suite, et il n'est pas donné.

    POURQUOI. Revue du tour 1 : 43 morceaux suivants de textes publiés de
    jurix_dev commencent par la FIN du dernier article collé dans le morceau
    précédent. « 35.2 » du Code du travail s'ouvre sur les alinéas (6) et (7)
    de l'article 39 ; le blog le citait « article 35 », avec pour extrait le
    début de l'article 35 (le préavis).
    """
    base = numero_affiche(ligne.number)
    if not base.isdigit():
        return None, 0
    lignes = (await db.execute(
        _SQL_MORCEAUX, {"law_id": law_id, "base": base, "morceaux": f"{base}.%"}
    )).all()
    premier = next((x for x in lignes if not _MORCEAU_SUIVANT.match(x.number or "")), None)
    corps = debut_du_corps(ligne.content, premier.content if premier is not None else None)
    rang = next((i for i, x in enumerate(lignes) if x.id == ligne.id), None)
    if rang is None:
        return None, corps
    precedentes: List[Any] = []
    while True:
        if rang == 0:
            return None, corps
        avant, courante = lignes[rang - 1], lignes[rang]
        if (avant.ordre or 0) != (courante.ordre or 0) - 1:
            return None, corps
        precedentes.append(avant)
        if not _MORCEAU_SUIVANT.match(avant.number or ""):
            break
        rang -= 1
    ouvert = base
    for avant in reversed(precedentes):
        debut = 0
        if _MORCEAU_SUIVANT.match(avant.number or "") and premier is not None:
            debut = debut_du_corps(avant.content, premier.content)
        entetes = [
            (p, n) for p, n in entetes_colles(avant.content, avant.number, modificatif) if p >= debut
        ]
        if entetes:
            ouvert = entetes[-1][1]
    return ouvert, corps


# La ligne qui suit une autre dans le texte (ordre + 1).
_SQL_LIGNE_SUIVANTE = text(
    f"SELECT {_COLONNES_D_ARTICLE} FROM articles a "
    'WHERE a.law_id = :law_id AND a."order" = :ordre ORDER BY a.id LIMIT 1'
)


async def _l_article_existe(
    db: AsyncSession, law_id: int, numero: str, modificatif: bool = True
) -> bool:
    """Le texte a-t-il l'article `numero` : une ligne à ce numéro, ou un en-tête lisible collé ?"""
    if await _lignes_du_numero(db, law_id, numero):
        return True
    motif = rf"art(icle)?\.{{0,2}}[[:space:]]*{numero}([^0-9]|$)"
    for ligne in (await db.execute(_SQL_ENGLOBANTS, {"law_id": law_id, "motif": motif})).all():
        if numero in (n for _, n in entetes_colles(ligne.content, ligne.number, modificatif)):
            return True
    return False


async def _suite_douteuse(
    db: AsyncSession, law_id: int, ligne: Any, ouvert: str, corps: int, modificatif: bool = True
) -> bool:
    """
    Le rattachement de la tête du morceau `ligne` à l'article `ouvert` est-il
    DOUTEUX ? Oui quand l'article qui suit cette tête (premier en-tête lisible
    de la ligne, sinon la ligne suivante du texte) est `ouvert` + 2 ou plus, et
    que l'article `ouvert` + 1 n'existe nulle part dans le texte (ni ligne, ni
    en-tête lisible).

    POURQUOI. Revue du tour 2 : l'en-tête de l'article 73 du Code du travail
    est abîmé par l'OCR à la fin de « 67.2 » (« Art e me,l n (t - ton
    judiciaire ou en faillite… », jurix_dev, ligne 97435). _suite_de croyait
    l'article 72 encore ouvert : la tête de « 67.3 », l'alinéa (2) de l'ARTICLE
    73 (« L'assistance judiciaire lui est acquise d'office… »), devenait une
    carte « article 72 (suite) », vérifiée. De même, la tête de « 52.3 » de la
    loi sur la cybersécurité citée « article 56 (suite) » au lieu de l'article
    57 (2). Trois lignes de jurix_dev sont dans ce cas : 52.3 et 67.3,
    fausses, et 161.2 (Code du travail), juste, qu'on perd aussi. Dans le
    doute, écarter : une tête écartée n'est ni donnée au modèle ni citée ; une
    tête mal rattachée devient une citation vérifiée sous un faux numéro.
    """
    if not (ouvert or "").strip().isdigit():
        return False
    attendu = int(ouvert) + 1
    entetes = [n for p, n in entetes_colles(ligne.content, ligne.number, modificatif) if p >= corps]
    if entetes:
        suivant = _rang_d_article(entetes[0])
    else:
        prochaine = (await db.execute(
            _SQL_LIGNE_SUIVANTE, {"law_id": law_id, "ordre": (ligne.ordre or 0) + 1}
        )).first()
        if (
            prochaine is None or est_pseudo_numero(prochaine.number)
            or _MORCEAU_SUIVANT.match(prochaine.number or "")
        ):
            return False
        suivant = _rang_d_article(numero_affiche(prochaine.number))
    if suivant is None or suivant <= attendu:
        return False
    return not await _l_article_existe(db, law_id, str(attendu), modificatif)


async def passages_de_la_ligne(db: AsyncSession, texte: Any, ligne: Any) -> List[Passage]:
    """Les passages (articles, suites d'articles) d'une ligne de la base."""
    annote = est_annote(texte.title)
    modificatif = est_modificatif(texte.title)
    ouvert, corps = None, 0
    if _MORCEAU_SUIVANT.match(ligne.number or ""):
        ouvert, corps = await _suite_de(db, texte.id, ligne, modificatif)
        if ouvert is not None and await _suite_douteuse(
            db, texte.id, ligne, ouvert, corps, modificatif
        ):
            logger.info(
                f"📰 Blog : tête de la ligne {ligne.id} ({ligne.number}) écartée, rattachement "
                f"douteux à l'article {ouvert}"
            )
            ouvert = None
    contenu = ligne.content or ""
    return [
        Passage(ligne.id, texte.id, numero, debut, fin, suite, propre, contenu[debut:fin])
        for numero, debut, fin, suite, propre in decouper(
            contenu, ligne.number, ouvert, annote, corps, modificatif
        )
    ]


def _choisir(
    passages: Sequence[Passage], voulu: str, suite: Optional[bool], ligne: Any
) -> Optional[Passage]:
    """Le passage de ce numéro (le début de l'article plutôt que sa suite, sauf demande)."""
    if not voulu:
        return passages[0] if passages else None
    # Le numéro de la ligne elle-même (« 35.1 ») désigne son propre article.
    if voulu == normalize_number(ligne.number or "") and voulu != normalize_number(
        numero_affiche(ligne.number)
    ):
        return next((p for p in passages if p.propre), None)
    du_numero = [p for p in passages if normalize_number(p.numero) == voulu]
    if suite is not None:
        du_numero.sort(key=lambda p: p.suite != suite)
    else:
        du_numero.sort(key=lambda p: p.suite)
    return du_numero[0] if du_numero else None


async def citations_verifiees(
    db: AsyncSession, demandes: Sequence[DemandeDeCitation], strict: bool = True
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]]]:
    """
    Les citations, construites à partir de la BASE : (citations, refus).

    Une citation n'est gardée que si son texte existe et est publié, et si
    l'article existe dans ce texte : un PASSAGE d'une ligne qui est le texte de
    cet article (son début, un article collé, ou sa suite). Tout ce qui
    s'affiche sur la carte vient de la base : l'intitulé du texte, le numéro,
    le début mot pour mot du passage, le lien. Seule l'explication (`title`)
    vient de l'auteur. `suite` dit que l'extrait n'est pas le début de
    l'article.

    `strict` : un refus lève CitationInvalide (création, modification,
    publication : 422) ; sinon il est rendu dans `refus` (rang, raison) et la
    citation est écartée (génération : un avertissement le dit).
    """
    ids_textes = sorted({d.law_id for d in demandes})
    ids_articles = sorted({d.article_id for d in demandes if d.article_id})
    textes = {
        ligne.id: ligne
        for ligne in (await db.execute(_SQL_TEXTES, {"ids": ids_textes})).all()
    } if ids_textes else {}
    articles = {
        ligne.id: ligne
        for ligne in (await db.execute(_SQL_ARTICLES_PAR_ID, {"ids": ids_articles})).all()
    } if ids_articles else {}

    citations: List[Dict[str, Any]] = []
    refus: List[Tuple[int, str]] = []
    vues: Set[Tuple[int, int, int]] = set()
    for rang, demande in enumerate(demandes, 1):
        try:
            texte, ligne, passage = await _resoudre(db, demande, textes, articles)
        except CitationInvalide as e:
            if strict:
                raise CitationInvalide(f"Citation {rang} : {e}") from None
            refus.append((rang, str(e)))
            continue
        # Le même passage demandé deux fois fait une seule carte.
        cle = (texte.id, ligne.id, passage.debut)
        if cle in vues:
            continue
        vues.add(cle)
        intitule = (ligne.title or "") if passage.propre else ""
        citations.append({
            "law_id": texte.id,
            "law_reference": texte.reference,
            "article_id": ligne.id,
            "article_number": passage.numero,
            "reference": texte.title,
            "title": couper(prose_publique(demande.title), CARACTERES_DE_L_EXPLICATION) or intitule,
            "excerpt": extrait_verbatim(passage.texte, suite=passage.suite),
            "url": adresse_de_l_article(texte.id, passage.numero),
            "verifie": True,
            "suite": passage.suite,
        })
    return citations, refus


async def passages_des_citations(
    db: AsyncSession, citations: Sequence[Dict[str, Any]]
) -> List[Optional[Tuple[str, Passage]]]:
    """
    Pour chaque citation VÉRIFIÉE, (intitulé du texte, passage entier) tel que
    la base le porte aujourd'hui ; None pour une citation non vérifiée ou que
    la base ne porte plus. Le contrôle IA lit ainsi le texte de l'article
    cité, et non les 300 caractères de sa carte.
    """
    demandes: List[Optional[DemandeDeCitation]] = []
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("verifie") is not True:
            demandes.append(None)
            continue
        try:
            demandes.append(DemandeDeCitation(
                int(citation["law_id"]),
                int(citation["article_id"]) if citation.get("article_id") else None,
                str(citation.get("article_number") or "") or None,
                suite=bool(citation.get("suite")),
            ))
        except (KeyError, TypeError, ValueError):
            demandes.append(None)
    valables = [d for d in demandes if d is not None]
    ids_textes = sorted({d.law_id for d in valables})
    ids_articles = sorted({d.article_id for d in valables if d.article_id})
    textes = {
        ligne.id: ligne for ligne in (await db.execute(_SQL_TEXTES, {"ids": ids_textes})).all()
    } if ids_textes else {}
    articles = {
        ligne.id: ligne
        for ligne in (await db.execute(_SQL_ARTICLES_PAR_ID, {"ids": ids_articles})).all()
    } if ids_articles else {}
    resultat: List[Optional[Tuple[str, Passage]]] = []
    for demande in demandes:
        if demande is None:
            resultat.append(None)
            continue
        try:
            texte, _, passage = await _resoudre(db, demande, textes, articles)
        except CitationInvalide:
            resultat.append(None)
            continue
        resultat.append((str(texte.title or ""), passage))
    return resultat


async def _resoudre(
    db: AsyncSession, demande: DemandeDeCitation, textes: Dict, articles: Dict
) -> Tuple[Any, Any, Passage]:
    """(texte, ligne, passage) de la citation, ou CitationInvalide."""
    texte = textes.get(demande.law_id)
    if texte is None:
        raise CitationInvalide(f"le texte {demande.law_id} n'existe pas dans la base.")
    if texte.status != "published":
        raise CitationInvalide(f"le texte « {texte.title} » n'est pas publié.")
    voulu = normalize_number(demande.article_number or "")

    if demande.article_id:
        ligne = articles.get(demande.article_id)
        if ligne is None or ligne.law_id != texte.id:
            raise CitationInvalide(
                f"l'article {demande.article_id} n'appartient pas au texte « {texte.title} »."
            )
        if est_pseudo_numero(ligne.number):
            raise CitationInvalide(
                f"le passage {ligne.number} du texte « {texte.title} » n'est pas un article."
            )
        passage = _choisir(await passages_de_la_ligne(db, texte, ligne), voulu, demande.suite, ligne)
        if passage is None:
            if voulu:
                raise CitationInvalide(
                    f"le numéro « {demande.article_number} » ne correspond à aucun article du "
                    f"passage {ligne.number} du texte « {texte.title} »."
                )
            raise CitationInvalide(
                f"le passage {ligne.number} du texte « {texte.title} » ne porte aucun texte "
                f"d'article citable."
            )
        return texte, ligne, passage

    if not voulu:
        raise CitationInvalide("indiquez l'article cité (article_id ou article_number).")
    # Par le seul numéro : le DÉBUT de l'article, jamais une suite.
    for ligne in await _lignes_du_numero(db, texte.id, demande.article_number):
        passage = _choisir(await passages_de_la_ligne(db, texte, ligne), voulu, False, ligne)
        if passage is not None and not passage.suite:
            return texte, ligne, passage
    englobe = await _article_englobe(db, texte, voulu)
    if englobe is not None:
        return englobe
    raise CitationInvalide(
        f"le texte « {texte.title} » n'a pas d'article {demande.article_number}."
    )


async def _article_englobe(
    db: AsyncSession, texte: Any, voulu: str
) -> Optional[Tuple[Any, Any, Passage]]:
    """
    (texte, ligne, passage) de l'article `voulu` collé dans le morceau d'un
    autre, s'il y porte son en-tête ; None sinon. Seulement pour un numéro
    simple (« 39 ») : c'est la forme des articles collés relevés.
    """
    if not voulu.isdigit():
        return None
    motif = rf"art(icle)?\.{{0,2}}[[:space:]]*{voulu}([^0-9]|$)"
    for ligne in (await db.execute(_SQL_ENGLOBANTS, {"law_id": texte.id, "motif": motif})).all():
        if est_pseudo_numero(ligne.number) or not porte_l_entete(ligne.content, voulu):
            continue
        passage = _choisir(await passages_de_la_ligne(db, texte, ligne), voulu, False, ligne)
        if passage is not None and not passage.suite:
            return texte, ligne, passage
    return None


async def _lignes_du_numero(db: AsyncSession, law_id: int, numero: str) -> List[Any]:
    """
    Les lignes de ce numéro dans le texte, la meilleure d'abord. Plusieurs
    lignes peuvent le porter (morceaux, homonymes d'une compilation) : un vrai
    article plutôt qu'un bout de tableau, le numéro exact, puis l'ordre du
    texte — la règle de l'aperçu d'un article (routes/laws._choisir).
    """
    vise = normalize_number(numero)
    brut = (numero or "").strip().upper()
    candidats = sorted({vise, brut, brut.split(".")[0], *(_PREMIER if vise == "1" else ())} - {""})
    lignes = [
        ligne for ligne in (await db.execute(
            _SQL_ARTICLES_PAR_NUMERO, {"law_id": law_id, "candidats": candidats}
        )).all()
        if not est_pseudo_numero(ligne.number)
    ]

    def rang(ligne) -> Tuple[bool, bool, int]:
        exact = normalize_number(ligne.number) == vise or ligne.number.strip().upper() == brut
        return ((ligne.kind or "article") != "article", not exact, ligne.ordre or 0)

    return sorted(lignes, key=rang)


# ==================== AVERTISSEMENTS ====================


class Avertissement(NamedTuple):
    """Un point à relire. `mention` sert au dédoublonnage entre deux lectures."""

    champ: Optional[str]
    mention: Optional[str]
    message: str


def champs_de_prose(
    title: str,
    context: str,
    summary: str,
    key_points: Sequence[str],
    citizen_impact: str,
    suggested_question: Optional[str],
    citations: Sequence[Dict[str, Any]],
    tags: Sequence[Any] = (),
) -> List[Tuple[str, str]]:
    """Les textes d'un article que le lecteur voit, nommés comme l'administrateur les voit."""
    champs = [("Titre", title), ("Contexte", context), ("Résumé", summary)]
    champs += [(f"Point clé {i}", point) for i, point in enumerate(key_points or [], 1)]
    champs += [("Impact pour le citoyen", citizen_impact)]
    champs += [("Question suggérée", suggested_question or "")]
    champs += [
        (f"Citation {i} (explication)", str(c.get("title") or ""))
        for i, c in enumerate(citations or [], 1)
        if isinstance(c, dict)
    ]
    # Les mots-clés s'affichent aussi : « Article 312 » en mot-clé est une mention.
    champs += [(f"Mot-clé {i}", t) for i, t in enumerate(tags or [], 1) if isinstance(t, str)]
    return [(nom, valeur) for nom, valeur in champs if (valeur or "").strip()]


# Un renvoi au numéro d'un extrait resté DANS la phrase (« comme le précise
# l'extrait 3, … ») : on ne peut pas le retirer sans casser la phrase
# (sans_renvois_aux_extraits ne retire que les renvois entre parenthèses).
_RENVOI_RESTANT = re.compile(r"\b(?:l['’][ \t]*|les[ \t]+)?extraits?[ \t]+(?:n[°o][ \t]*)?\d+", re.IGNORECASE)


def _chevauche(debut: int, fin: int, places: Sequence[Tuple[int, int]]) -> bool:
    return any(debut < f and d < fin for d, f in places)


def relire_la_prose(
    champs: Sequence[Tuple[str, str]],
    extraits: Sequence[ChunkResult],
    generation: bool,
) -> List[Avertissement]:
    """
    Ce que la prose affirme et qu'aucun extrait gardé ne porte. `extraits` :
    à la génération, les passages lus par le modèle ; sur un article
    enregistré, ses citations vérifiées.

    Quatre lectures, toutes signalées (DANS LE DOUTE, SIGNALER) :
    1. chaque mention d'article (mentions_d_articles : pluriels, intervalles,
       « premier », sans nom, « Le Code pénal, en son article 312 », anglais)
       dont un numéro n'est porté par aucun extrait. C'est le filet ajouté au
       tour 2 : « les articles 286 et 312 du Code pénal », l'invention même de
       l'essai réel, passait sans un mot ;
    2. la lecture des citations du chat (lire_les_citations) : ses citations
       au singulier non portées, et les textes désignés par leur numéro (« le
       décret n° 2018/9387 ») ; une citation d'article qu'une mention de la
       lecture 1 signale déjà n'est pas dite deux fois ;
    3. un texte nommé sans article qu'aucun extrait ne porte (« le Code de la
       famille », « la Constitution ») ;
    4. le droit d'un autre pays (« le code civil français »).
    Une lecture qui échoue rend le champ « à relire à la main » : une
    exception ne doit pas rendre l'administration illisible.
    """
    avertissements: List[Avertissement] = []
    for champ, valeur in champs:
        try:
            signalees = numeros_non_portes(valeur, extraits)
            for mention, manquants in signalees:
                lesquels = en_toutes_lettres(manquants)
                plusieurs = len(manquants) > 1
                if generation:
                    figure = "ne figurent" if plusieurs else "ne figure"
                    message = (
                        f"{champ} : « {mention.texte} » — {lesquels} {figure} dans aucun des "
                        f"extraits consultés ; le modèle l'a peut-être écrit de mémoire. "
                        f"Vérifiez-le dans la base, ou retirez la mention."
                    )
                else:
                    appuye = "ne sont appuyés" if plusieurs else "n'est appuyé"
                    message = (
                        f"{champ} : « {mention.texte} » — {lesquels} {appuye} par aucune "
                        f"citation de l'article. Ajoutez la citation de cet article, ou retirez "
                        f"la mention."
                    )
                avertissements.append(Avertissement(champ, mention.texte, message))
            for renvoi in _RENVOI_RESTANT.finditer(valeur):
                mention = " ".join(renvoi.group().split())
                avertissements.append(Avertissement(
                    champ, mention,
                    f"{champ} : « {mention} » — un renvoi au numéro d'un extrait, que le lecteur "
                    f"ne voit pas. Reformulez sans ce renvoi.",
                ))
            places = [(m.debut, m.fin) for m, _ in signalees]
            for occurrence in lire_les_citations(valeur, extraits):
                if not occurrence.introuvable:
                    continue
                if occurrence.forme == "article" and _chevauche(occurrence.debut, occurrence.fin, places):
                    continue
                mention = " ".join(valeur[occurrence.debut:occurrence.fin].split())
                if generation:
                    message = (
                        f"{champ} : « {mention} » ne figure dans aucun des extraits consultés ; le "
                        f"modèle l'a peut-être écrit de mémoire. Vérifiez-le dans la base, ou "
                        f"retirez-le."
                    )
                else:
                    message = (
                        f"{champ} : « {mention} » n'est appuyé par aucune citation de l'article. "
                        f"Ajoutez la citation de cet article, ou retirez la mention."
                    )
                avertissements.append(Avertissement(champ, mention, message))
            for nom in textes_nommes_non_portes(valeur, extraits):
                avertissements.append(Avertissement(
                    champ, nom,
                    f"{champ} : « {nom} » — aucun "
                    f"{'extrait consulté' if generation else 'texte cité par l’article'} n'est de "
                    f"ce texte. Vérifiez qu'il existe et qu'il dit ce que l'article lui prête, ou "
                    f"retirez-le.",
                ))
            for mention in droit_etranger(valeur):
                avertissements.append(Avertissement(
                    champ, mention,
                    f"{champ} : « {mention} » — le droit d'un autre pays. Le blog ne traite que du "
                    f"droit applicable au Cameroun : retirez ce passage, ou vérifiez qu'il "
                    f"s'applique.",
                ))
        except Exception:
            logger.exception(f"⚠️ Blog : relecture de la prose impossible ({champ})")
            avertissements.append(Avertissement(champ, None, relecture_impossible(champ)))
    return avertissements


def extraits_des_citations(citations: Sequence[Dict[str, Any]]) -> List[ChunkResult]:
    """Les citations VÉRIFIÉES d'un article, sous la forme que lisent les relectures."""
    extraits = []
    for citation in citations or []:
        if not isinstance(citation, dict) or citation.get("verifie") is not True:
            continue
        try:
            extraits.append(ChunkResult(
                article_id=citation.get("article_id"),
                law_id=int(citation["law_id"]),
                number=str(citation.get("article_number") or ""),
                law_title=str(citation.get("reference") or ""),
                reference=str(citation.get("law_reference") or ""),
                content=str(citation.get("excerpt") or ""),
                status="published",
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return extraits


def avertissements_du_billet(post: Any) -> List[Avertissement]:
    """
    Ce qu'un administrateur doit relire dans l'article TEL QU'IL EST, recalculé
    à chaque lecture : rien n'est stocké (la table 017 n'a pas de colonne pour
    cela), et une modification change la réponse.

    - Une citation d'avant la vérification par le serveur (`verifie` absent) :
      l'article n'est pas publiable tant qu'elle y est (tour 3 :
      controle_du_billet.etat_de_publication).
    - Aucune citation du tout : pas publiable non plus.
    Les débuts « Citation N : non vérifiée » et « Aucune citation » sont lus
    par le front (relecture.ts, RECALCULES_PAR_LE_FRONT) : à garder.
    - Ce que la prose affirme et qu'aucune citation VÉRIFIÉE ne porte
      (relire_la_prose) : le lecteur le lirait sans carte pour le vérifier.
    """
    citations = [c for c in (post.citations or []) if isinstance(c, dict)]
    avertissements: List[Avertissement] = []
    for rang, citation in enumerate(citations, 1):
        if citation.get("verifie") is not True:
            avertissements.append(Avertissement(
                f"Citation {rang}", None,
                f"Citation {rang} : non vérifiée (écrite avant la vérification par le "
                f"serveur). L'article ne peut pas être publié tant qu'elle y est : retirez-la, "
                f"ou remplacez-la par un article de la base.",
            ))
    if not citations:
        avertissements.append(Avertissement(
            None, None,
            "Aucune citation : l'article ne s'appuie sur aucun texte de la base, et ne peut pas "
            "être publié.",
        ))
    champs = champs_de_prose(
        post.title, post.context, post.summary, post.key_points or [],
        post.citizen_impact, post.suggested_question, citations, post.tags or [],
    )
    avertissements += relire_la_prose(champs, extraits_des_citations(citations), generation=False)
    return avertissements


def relecture_impossible(champ: str) -> str:
    return (
        f"{champ} : la relecture automatique des références a échoué ; vérifiez à la main "
        f"chaque article et chaque texte cité."
    )


def fusionner(*listes: Sequence[Avertissement]) -> List[str]:
    """Les messages, dans l'ordre, une mention signalée une seule fois par champ."""
    vues: Set[Tuple[Optional[str], str]] = set()
    messages: List[str] = []
    for liste in listes:
        for avertissement in liste:
            cle = (
                (avertissement.champ, avertissement.mention)
                if avertissement.mention is not None
                else (None, avertissement.message)
            )
            if cle not in vues:
                vues.add(cle)
                messages.append(avertissement.message)
    return messages


# ==================== RECHERCHE ====================


class RechercheDuBlog(SearchService):
    """
    La recherche du chat : volet sémantique EXACT (semantique_exacte), sans
    l'index HNSW, dont le rappel peut s'effondrer sans bruit (incident du
    29/10/2026).

    Sans cache : une réponse du cache réduisait le contenu des résultats à 300
    caractères (essai réel du 29/10/2026), et le blog n'a pas à remplir celui
    du chat. Sans journal : un sujet d'article n'est pas la question d'un
    lecteur, et faussait les « requêtes fréquentes » des statistiques.
    """

    def __init__(self, db: AsyncSession):
        super().__init__(db, use_cache=False, semantique_exacte=True)

    async def _ecrire_apres_la_reponse(self, *args: Any, **kwargs: Any) -> None:
        # Ni ligne dans search_events, ni cache (use_cache=False).
        return None


# Les articles, parmi des identifiants, qui portent TOUS les mots-clés : la
# règle de confirmation du chat (rag_service._SQL_ARTICLES_STRICTS), recopiée
# et non importée parce qu'elle est privée là-bas. Au moins DEUX lexèmes
# distincts : un seul (« bail ») est dans des milliers d'articles et ne prouve
# rien.
_SQL_ARTICLES_STRICTS = text("""
    SELECT a.id
    FROM articles a
    WHERE a.id = ANY(:ids)
      AND coalesce(array_length(tsvector_to_array(to_tsvector('french', :requete)), 1), 0) >= 2
      AND (a.search_vector @@ websearch_to_tsquery('french', :requete)
           OR a.search_vector @@ websearch_to_tsquery('english', :requete))
""")

# Les mots DISTINCTIFS du sujet : ses lexèmes (to_tsvector('french')) présents
# dans au moins un article et dans moins de :au_plus articles de la base. Le
# compte s'arrête à :plafond (un mot présent partout ne coûte pas un parcours
# entier). quote_literal : un lexème composé (« dommages-intérêt ») reste un
# seul mot.
_SQL_MOTS_DISTINCTIFS = text("""
    SELECT l.lexeme
    FROM unnest(tsvector_to_array(to_tsvector('french', :sujet))) AS l(lexeme)
    WHERE length(l.lexeme) >= 3
      AND (SELECT count(*) FROM (
             SELECT 1 FROM articles a
             WHERE a.search_vector @@ quote_literal(l.lexeme)::tsquery
             LIMIT :plafond) AS s) BETWEEN 1 AND :au_plus
""")
# Parmi des articles, ceux qui portent l'un de ces mots.
_SQL_ANCRES = text("""
    SELECT a.id
    FROM articles a
    WHERE a.id = ANY(:ids)
      AND EXISTS (SELECT 1 FROM unnest(CAST(:mots AS text[])) AS m(mot)
                  WHERE a.search_vector @@ quote_literal(m.mot)::tsquery)
""")
# « peut-il », « a-t-il » : l'inversion d'une question, pas un mot du sujet.
_INVERSION = re.compile(r"-(?:t-)?(?:il|elle|on|ils|elles)$")


# ==================== CE QUE LE MODÈLE REND ====================


class _CitationDuModele(BaseModel):
    extrait: int
    objet: str = ""


class _ArticleDuModele(BaseModel):
    """
    Le contenu est exigé tel quel : un point clé rendu en objet (`{"point":
    …}`) s'affichait « [object Object] », puis faisait planter la page de
    l'article. La rubrique, la question et les mots-clés, accessoires, sont
    lus avec indulgence (rubrique_connue, mots_cles_propres).
    """

    # Le modèle dit lui-même si les extraits traitent le sujet (tour 2) : un
    # dernier filet derrière la preuve de pertinence. Absent (modèle qui
    # ignore le schéma) : la preuve de pertinence a déjà jugé, on la suit.
    extraits_suffisants: Optional[bool] = None
    title: str
    category: Any = ""
    context: str
    summary: str
    key_points: List[str]
    citations: List[_CitationDuModele]
    citizen_impact: str
    suggested_question: Optional[str] = ""
    tags: List[Any] = []


# Schéma imposé au modèle (json_schema strict chez Mistral, response_schema
# chez Gemini). `citations` ne porte que des NUMÉROS d'extraits : le numéro
# d'article, le texte et l'extrait cités sont écrits par le serveur. La
# rubrique est contrôlée après coup (rubrique_connue) plutôt que par un
# `enum`, que les deux fournisseurs ne lisent pas de la même façon.
SCHEMA_DE_L_ARTICLE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "extraits_suffisants": {"type": "boolean"},
        "title": {"type": "string"},
        "category": {"type": "string"},
        "context": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "extrait": {"type": "integer"},
                    "objet": {"type": "string"},
                },
                "required": ["extrait", "objet"],
            },
        },
        "citizen_impact": {"type": "string"},
        "suggested_question": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "extraits_suffisants", "title", "category", "context", "summary", "key_points",
        "citations", "citizen_impact", "suggested_question", "tags",
    ],
}

SYSTEM_PROMPT = """Tu es le rédacteur juridique de JuriX, un service d'information sur le droit camerounais. Tu écris un article de décryptage pour le grand public, à partir des SEULS extraits de textes officiels qui te sont fournis, numérotés [1], [2], etc.

D'abord, juge les extraits : "extraits_suffisants" vaut false s'ils ne traitent pas le sujet demandé, ou trop peu pour écrire un article exact ; laisse alors tous les autres champs vides. Il vaut true sinon.

Ce que tu peux écrire :
- Uniquement ce que disent les extraits. N'ajoute aucune règle, aucun délai, aucun montant, aucune peine, aucune procédure, aucune exception qu'ils ne contiennent pas.
- N'écris aucun numéro d'article ni aucun nom de texte (loi, code, décret, acte uniforme) qui ne figure pas dans l'en-tête d'un extrait. Ne cite ni jurisprudence ni doctrine qui ne soit pas dans un extrait.
- Ne t'appuie jamais sur ta mémoire, ni sur le droit d'un autre pays (droit français ou autre).
- Une règle ne vaut que pour le cas que vise son extrait (bail professionnel ou bail d'habitation, contrat à durée déterminée ou indéterminée, telle personne, tel manquement) : nomme ce cas, et ne l'étends jamais à un autre.
- Dis exactement ce que l'extrait dit de qui doit prouver, de qui décide (un juge ou non), des délais et des montants ; n'affirme jamais qu'une chose est possible sans juge, ou sans formalité, si aucun extrait ne le dit.
- Si les extraits ne suffisent pas à traiter un point du sujet, dis-le en une phrase (« les textes consultés ne précisent pas… ») au lieu de compléter.

Les extraits :
- Chaque extrait est UN article, nommé dans son en-tête : « Article 39 » est le texte de l'article 39, et de lui seul.
- Un en-tête qui dit « suite » annonce la fin d'un article dont le début n'est pas fourni : n'en déduis rien sur ce que l'article dit avant.

Citations dans le texte :
- Écris exactement « l'article X du Code Y » ou « l'article X de la Loi Y », avec le numéro et le nom du texte tels qu'ils figurent dans l'en-tête de l'extrait, un seul article à la fois. Jamais « art. », jamais « articles 5 et 6 ».
- Le champ "citations" ne contient que les NUMÉROS des extraits sur lesquels tu t'appuies, avec pour chacun un "objet" : une phrase qui dit ce que cet extrait établit pour le lecteur, sans numéro d'article ni nom de texte, et seulement ce que cet extrait-là dit. Le programme ajoute lui-même le texte, le numéro et l'extrait exact à partir de la base.
- N'écris jamais de renvoi aux numéros des extraits dans le texte (« (extrait 2) », « (extraits 5 et 9) », « [3] ») : le lecteur ne voit pas ces numéros.

Textes successifs : si une ligne « Statut » sous l'en-tête d'un extrait dit que l'article a été modifié, abrogé ou remplacé, suis-la, et ne présente pas l'ancienne règle comme en vigueur.

Forme :
- Texte brut uniquement : ni Markdown (pas de **, pas de #, pas de listes à tirets), ni HTML, ni emoji.
- Mots de tous les jours, phrases courtes ; explique aussitôt un terme juridique indispensable.

Réponds uniquement par l'objet JSON demandé."""


# ==================== LE SERVICE ====================


@dataclass
class ArticleRedige:
    """L'article tel que le serveur l'enregistre en brouillon, et ce qui l'accompagne."""

    title: str
    category: str
    context: str
    summary: str
    key_points: List[str]
    citations: List[Dict[str, Any]]
    citizen_impact: str
    suggested_question: str
    tags: List[str]
    reading_time: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    avertissements: List[Avertissement] = field(default_factory=list)


_SANS_SOURCE = (
    "Aucun texte publié de la base ne traite clairement ce sujet : l'article n'a pas été "
    "rédigé. Reformulez le sujet en français, avec les mots de la loi (par exemple « bail », "
    "« licenciement », « succession »)."
)


class BlogGeneratorService:
    """
    Le modèle et la recherche sont des PARAMÈTRES (comme ComparisonService) :
    les tests injectent des doublures, et le vrai modèle n'est jamais appelé.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        llm: Optional[Any] = None,
        recherche: Optional[SearchService] = None,
        llm_du_controle: Optional[Any] = None,
    ):
        self.db = db
        self.llm = llm if llm is not None else get_llm_service()
        self.recherche = recherche if recherche is not None else RechercheDuBlog(db)
        # Le modèle du contrôle qui suit la génération (route /generate).
        # POURQUOI None par défaut : ControleDuBillet prend alors le modèle DU
        # CONTRÔLE (BLOG_CONTROLE_MODELE, tour 6), pas celui de la génération.
        # Une doublure injectée pour la génération (les tests) sert aussi au
        # contrôle, comme avant le tour 6, sauf si on lui en donne une autre.
        self.llm_du_controle = llm_du_controle if llm_du_controle is not None else llm

    async def rediger(
        self,
        topic: str,
        category: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> ArticleRedige:
        """
        Recherche, extraits pertinents, passages, rédaction, citations
        vérifiées, avertissements. N'écrit rien en base : la route enregistre
        le brouillon.

        Raises:
            SansSourceError (422), RechercheIndisponibleError (503),
            QuotaError (429), SaturationError (503), ReponseInvalideError (502)
        """
        debut = time.perf_counter()
        extraits = await self._extraits_pertinents(topic)
        passages = await self._passages(extraits) if extraits else []
        if not passages:
            raise SansSourceError(_SANS_SOURCE)
        contexte, lus, passages = self._contexte(passages)
        duree_recherche = time.perf_counter() - debut

        # Rien n'a été écrit : la transaction ouverte par l'authentification et
        # la recherche est close AVANT le modèle, qui peut prendre deux minutes.
        # Sinon, chaque génération tenait l'une des dix connexions du pool (et
        # une connexion du pooler de Neon), comme le chat avant
        # RAGService._valider_avant_la_generation.
        await self.db.commit()

        debut = time.perf_counter()
        brut = await self._generer(
            self._prompt(topic, additional_context, contexte, len(lus))
        )
        duree_generation = time.perf_counter() - debut
        article = self._lire(brut)
        if article.extraits_suffisants is False:
            logger.info(f"📰 Blog « {topic[:60]} » : le modèle juge les extraits insuffisants")
            raise SansSourceError(_SANS_SOURCE)
        if article.extraits_suffisants is None:
            logger.warning("⚠️ Blog : le modèle n'a pas dit si les extraits suffisent")
        logger.info(
            f"📰 Article « {topic[:60]} » : {len(lus)} extrait(s), recherche "
            f"{duree_recherche:.1f} s, génération {duree_generation:.1f} s"
        )
        return await self._assembler(article, lus, passages, category)

    async def sources_du_sujet(self, sujet: str) -> List[Dict[str, Any]]:
        """
        Les sources d'un article déjà écrit, retrouvées par la MÊME recherche
        que la génération (preuve de pertinence, passages, numérotation), sans
        appeler le modèle : routes/blog, POST /{id}/sources. C'est le chemin
        par lequel un article d'avant, qui n'a pas de sources, peut être
        contrôlé, relu, et revenir en ligne (décision du propriétaire, 30/10).

        Raises:
            SansSourceError (422), RechercheIndisponibleError (503)
        """
        extraits = await self._extraits_pertinents(sujet)
        passages = await self._passages(extraits) if extraits else []
        if not passages:
            raise SansSourceError(_SANS_SOURCE_DE_L_ARTICLE)
        _, lus, gardes = self._contexte(passages)
        logger.info(f"📚 Blog : {len(lus)} source(s) reconstruite(s) pour « {sujet[:60]} »")
        return sources_des_passages(gardes, lus)

    # ------------------------------------------------------------ recherche

    async def _extraits_pertinents(self, topic: str) -> List[ChunkResult]:
        """
        Les articles que la recherche du chat trouve pour le sujet, quand le
        sujet est PROUVÉ (voir SEUILS_DU_BLOG) ; [] sinon (422 « sans_source »,
        le modèle n'est pas appelé).

        Le sujet est prouvé par un extrait FORT : très proche par le sens
        (SEUILS_DU_BLOG), ou proche (SEUILS_ANCRES_DU_BLOG) et porteur d'un mot
        distinctif du sujet ou de tous ses mots-clés. Restent alors les
        extraits forts, et ceux qui portent un mot distinctif du sujet ou tous
        ses mots-clés au-dessus du seuil du chat : un extrait lu est une source
        que le modèle peut citer.

        Volet sémantique en panne ou servi par l'index (repli d'une recherche
        exacte trop lente) : 503, réessayer. Le blog n'est pas pressé ; il ne
        rédige pas sur une recherche qu'il ne peut pas vérifier.
        """
        recherche = self.recherche
        recherche.tableaux_de_programmes_exclus = not question_budgetaire(topic)
        mots_cles = RAGService.extract_keywords(topic)
        try:
            reponse = await recherche.search(SearchRequest(
                query=mots_cles[:500],
                semantic_query=topic[:1000],
                mode="hybrid",
                filters=SearchFilters(status="published"),
                limit=EXTRAITS_AU_PLUS,
            ))
        except Exception as e:
            logger.error(f"❌ Recherche du blog en échec : {e!r}")
            raise RechercheIndisponibleError(
                "La recherche dans les textes est indisponible pour le moment : l'article n'a "
                "pas été rédigé. Réessayez dans quelques instants."
            ) from e

        volets = getattr(recherche, "volets", None)
        if not isinstance(volets, VoletsDeLaRecherche) or volets.chemin != CHEMIN_EXACT:
            chemin = getattr(volets, "chemin", None)
            logger.warning(f"🚧 Blog : volet sémantique {chemin}, génération refusée")
            raise RechercheIndisponibleError(
                "La recherche par le sens n'a pas pu être vérifiée (service saturé ou "
                "indisponible) : l'article n'a pas été rédigé. Réessayez dans quelques instants."
            )

        candidats = [c for c in reponse.chunks if est_citable(c)]
        langue = "fr" if langue_de_la_question(topic) == "fr" else "en"

        def similarite(extrait: ChunkResult) -> float:
            return volets.similarites.get(extrait.article_id, -1.0)

        stricts = await self._articles_stricts(mots_cles, candidats)
        ancres = await self._ancres(topic, candidats) | stricts
        forts = {
            c.article_id for c in candidats
            if similarite(c) >= SEUILS_DU_BLOG[langue]
            or (c.article_id in ancres and similarite(c) >= SEUILS_ANCRES_DU_BLOG[langue])
        }
        seuil_du_chat = RAGService.SEUILS_DE_SIMILARITE[langue]
        retenus = [
            c for c in candidats
            if forts and (
                c.article_id in forts or c.article_id in stricts
                or (c.article_id in ancres and similarite(c) >= seuil_du_chat)
            )
        ]
        meilleure = max((similarite(c) for c in candidats), default=None)
        logger.info(
            f"📚 Blog « {topic[:60]} » : {len(reponse.chunks)} extrait(s) trouvé(s), "
            f"{len(candidats)} article(s) citable(s), meilleure similarité {meilleure}, "
            f"{len(ancres)} ancré(s), {len(forts)} fort(s) (seuils {SEUILS_DU_BLOG[langue]} / "
            f"{SEUILS_ANCRES_DU_BLOG[langue]} ancré), {len(retenus)} retenu(s)"
        )
        return retenus

    async def _articles_stricts(self, mots_cles: str, extraits: List[ChunkResult]) -> Set[int]:
        """Parmi `extraits`, ceux qui portent tous les mots-clés (au moins deux)."""
        ids = sorted({c.article_id for c in extraits if c.article_id is not None})
        if not ids or not mots_cles.strip():
            return set()
        try:
            # Sous un SAVEPOINT : un échec n'annule que lui, et ne confirme rien.
            async with self.db.begin_nested():
                lignes = (await self.db.execute(
                    _SQL_ARTICLES_STRICTS, {"ids": ids, "requete": mots_cles}
                )).all()
        except Exception as e:
            logger.warning(f"⚠️ Plein texte strict indisponible pour le blog : {e!r}")
            return set()
        return {int(ligne.id) for ligne in lignes}

    async def _ancres(self, topic: str, extraits: List[ChunkResult]) -> Set[int]:
        """
        Parmi `extraits`, ceux qui portent un mot DISTINCTIF du sujet. Une
        panne n'ancre rien : seule la similarité seule (le seuil le plus haut)
        peut alors prouver le sujet — dans le doute, plus strict.
        """
        ids = sorted({c.article_id for c in extraits if c.article_id is not None})
        if not ids:
            return set()
        try:
            async with self.db.begin_nested():
                mots = [
                    ligne.lexeme for ligne in (await self.db.execute(_SQL_MOTS_DISTINCTIFS, {
                        "sujet": topic, "au_plus": MOTS_DISTINCTIFS_AU_PLUS,
                        "plafond": MOTS_DISTINCTIFS_AU_PLUS + 1,
                    })).all()
                    if not _INVERSION.search(ligne.lexeme)
                ]
                if not mots:
                    return set()
                lignes = (await self.db.execute(_SQL_ANCRES, {"ids": ids, "mots": mots})).all()
        except Exception as e:
            logger.warning(f"⚠️ Mots distinctifs indisponibles pour le blog : {e!r}")
            return set()
        return {int(ligne.id) for ligne in lignes}

    # ------------------------------------------------------------ passages

    async def _passages(self, extraits: Sequence[ChunkResult]) -> List[Tuple[Passage, ChunkResult]]:
        """
        Chaque extrait retenu découpé en PASSAGES (un article, ou la suite d'un
        article), relus dans la base : c'est la ligne en base, et non la copie
        de la recherche, qui fait foi, comme pour la citation qui en sera tirée.
        """
        ids_articles = sorted({c.article_id for c in extraits if c.article_id is not None})
        ids_textes = sorted({c.law_id for c in extraits})
        lignes = {
            ligne.id: ligne
            for ligne in (await self.db.execute(_SQL_ARTICLES_PAR_ID, {"ids": ids_articles})).all()
        } if ids_articles else {}
        textes = {
            ligne.id: ligne
            for ligne in (await self.db.execute(_SQL_TEXTES, {"ids": ids_textes})).all()
        } if ids_textes else {}
        resultat: List[Tuple[Passage, ChunkResult]] = []
        vus: Set[Tuple[int, int]] = set()
        for extrait in extraits:
            ligne, texte = lignes.get(extrait.article_id), textes.get(extrait.law_id)
            if ligne is None or texte is None or texte.status != "published":
                continue
            for passage in await passages_de_la_ligne(self.db, texte, ligne):
                if (passage.ligne_id, passage.debut) in vus:
                    continue
                vus.add((passage.ligne_id, passage.debut))
                resultat.append((passage, extrait.model_copy(update={
                    "number": passage.numero,
                    "content": passage.texte,
                    "article_title": (ligne.title or None) if passage.propre else None,
                })))
        return resultat

    # ------------------------------------------------------------ prompt

    @staticmethod
    def _contexte(
        passages: Sequence[Tuple[Passage, ChunkResult]],
    ) -> Tuple[str, List[ChunkResult], List[Passage]]:
        """
        Les blocs numérotés donnés au modèle, les extraits TELS QU'IL LES LIT
        (copies coupées : c'est contre eux que la prose est relue), et leurs
        passages (ce que le serveur citera).

        L'en-tête nomme le texte par son INTITULÉ et l'article par son numéro,
        jamais par la cote interne (« PRC-4722 — Article ? » : le modèle en
        tirait des noms de textes inventés). Un extrait est UN article : son
        numéro est celui que la carte affichera.
        """
        blocs: List[str] = []
        lus: List[ChunkResult] = []
        gardes: List[Passage] = []
        place = CARACTERES_DES_EXTRAITS
        for passage, extrait in passages:
            # Le premier extrait passe toujours ; les suivants tant qu'il reste
            # de quoi lire un article (un extrait non donné n'est pas numéroté).
            if lus and place < 500:
                break
            contenu = couper_au_paragraphe(extrait.content or "", min(CARACTERES_PAR_EXTRAIT, place))
            n = len(lus) + 1
            entete = f"[{n}] {extrait.law_title} — Article {passage.numero}"
            if passage.suite:
                entete += (
                    f" (suite : ce passage commence au milieu de l'article {passage.numero}, "
                    f"son début n'est pas fourni)"
                )
            elif extrait.article_title:
                entete += f" — {extrait.article_title}"
            lignes = [entete]
            if extrait.section:
                lignes.append(f"Section : {extrait.section}")
            lignes += [
                sans_cote(str(ligne))[0]
                for ligne in lignes_en_deux(
                    extrait.reference, passage.numero, contenu, extrait.section
                )
            ]
            blocs.append("\n".join(lignes) + "\n\n" + contenu)
            lus.append(extrait.model_copy(update={"content": contenu}))
            gardes.append(passage)
            place -= len(contenu)
        return "\n\n---\n\n".join(blocs), lus, gardes

    @staticmethod
    def _prompt(topic: str, additional_context: Optional[str], contexte: str, n: int) -> str:
        donne = (
            "\nCONTEXTE DONNÉ PAR L'ÉQUIPE DE JURIX (des faits à reprendre, sans rien y ajouter) :\n"
            f"{additional_context}\n"
            if additional_context else ""
        )
        return f"""SUJET : « {topic} »
{donne}
EXTRAITS ({n}, numérotés de 1 à {n} ; les seuls textes que tu peux utiliser) :

{contexte}

Rédige l'article en JSON, avec ces champs :
- extraits_suffisants : true si les extraits traitent le sujet, false sinon (et alors tous les autres champs vides).
- title : un titre clair et exact, 100 caractères au plus.
- category : une rubrique parmi : {", ".join(RUBRIQUES)}.
- context : la situation concrète, en deux ou trois phrases. N'invente aucun fait d'actualité : aucun nom, aucune date, aucun chiffre qui ne vienne du contexte donné par l'équipe.
- summary : ce que prévoient les extraits sur ce sujet, expliqué simplement.
- key_points : trois à cinq points essentiels, une phrase chacun, chacun appuyé sur un extrait.
- citations : pour chaque extrait utilisé, {{"extrait": son numéro entre 1 et {n}, "objet": ce qu'il établit pour le lecteur, en une phrase}}.
- citizen_impact : ce que cela change concrètement pour le lecteur, et la démarche à suivre si les extraits la donnent.
- suggested_question : une question que le lecteur peut poser à l'assistant de JuriX sur son propre cas.
- tags : trois à six mots-clés courts.

LONGUEUR : l'article entier (titre, contexte, résumé, points clés, impact, explications des citations et question) tient en {PHRASES_VISEES} phrases au plus ; c'est la limite du contrôle, qui refuse un article plus long. Fais des phrases complètes, jamais des listes coupées en morceaux."""


    # ------------------------------------------------------------ modèle

    async def _generer(self, prompt: str) -> Dict[str, Any]:
        """L'appel au modèle, ses erreurs traduites (quota, saturation, autre)."""
        try:
            reponse = await self.llm.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=GENERATION_TEMPERATURE,
                max_tokens=GENERATION_MAX_TOKENS,
                response_mime_type="application/json",
                response_schema=SCHEMA_DE_L_ARTICLE,
                # Gemini 3 : la réflexion par défaut entamait le budget de
                # jetons. Mistral l'ignore.
                reflexion="low",
            )
        # Quota et saturation AVANT l'erreur générique, dont elles dérivent.
        except ERREURS_QUOTA as e:
            logger.warning(f"⚠️ Blog : quota de génération épuisé ({e!r})")
            raise QuotaError(
                "Le quota de génération du modèle est épuisé : l'article n'a pas été rédigé. "
                "Réessayez plus tard."
            ) from e
        except ERREURS_SATURATION as e:
            logger.warning(f"⚠️ Blog : génération saturée ({e!r})")
            raise SaturationError(
                "Le service de génération est saturé : l'article n'a pas été rédigé. "
                "Réessayez dans quelques instants."
            ) from e
        except ERREURS_BUDGET as e:
            raise ReponseInvalideError(
                "Le modèle n'a pas pu rédiger l'article dans le budget imparti : aucun article "
                "n'a été enregistré. Relancez la génération."
            ) from e
        except ERREURS_LLM as e:
            logger.error(f"❌ Blog : erreur du modèle ({e!r})")
            raise ReponseInvalideError(
                "Le service de génération a renvoyé une erreur : aucun article n'a été "
                "enregistré. Réessayez plus tard."
            ) from e
        return reponse or {}

    @staticmethod
    def _lire(reponse: Dict[str, Any]) -> _ArticleDuModele:
        """
        L'article du modèle, ou ReponseInvalideError. JAMAIS d'article de repli :
        l'ancien service fabriquait un « Décryptage légal » avec le JSON brut en
        résumé, quatre points génériques et des citations « Article ? », que
        « Publier maintenant » mettait en ligne.
        """
        if reponse.get("tronquee"):
            # Mistral ne complète pas une réponse JSON coupée : un JSON
            # recollé n'est pas fiable.
            logger.warning(f"⚠️ Blog : réponse coupée (fin {reponse.get('fin')})")
            raise ReponseInvalideError(
                "La réponse du modèle a été coupée avant la fin : aucun article n'a été "
                "enregistré. Relancez la génération."
            )
        brut = str(reponse.get("response") or "").strip()
        if brut.startswith("```"):
            brut = re.sub(r"^```(?:json)?\s*|\s*```$", "", brut)
        try:
            donnees = json.loads(brut)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"⚠️ Blog : JSON illisible ({brut[:120]!r})")
            raise ReponseInvalideError(
                "Le modèle n'a pas rendu un article lisible : aucun article n'a été "
                "enregistré. Relancez la génération."
            ) from None
        if not isinstance(donnees, dict):
            raise ReponseInvalideError(
                "Le modèle n'a pas rendu un article : aucun article n'a été enregistré. "
                "Relancez la génération."
            )
        try:
            return _ArticleDuModele.model_validate(donnees)
        except ValidationError as e:
            logger.warning(f"⚠️ Blog : article incomplet ({e.error_count()} erreur(s))")
            raise ReponseInvalideError(
                "Le modèle a rendu un article incomplet : aucun article n'a été enregistré. "
                "Relancez la génération."
            ) from None

    # ------------------------------------------------------------ assemblage

    async def _assembler(
        self,
        article: _ArticleDuModele,
        lus: List[ChunkResult],
        passages: List[Passage],
        category: Optional[str],
    ) -> ArticleRedige:
        avertissements: List[Avertissement] = []

        def propre(champ: str, valeur: str, longueur: Optional[int] = None) -> str:
            nettoye, cotes = sans_cote(prose_publique(valeur))
            # Une cote retirée en tête de ligne y laissait une espace.
            nettoye = texte_brut(nettoye)
            for cote in cotes:
                avertissements.append(Avertissement(
                    champ, cote, f"{champ} : la cote interne {cote} a été retirée ; relisez la phrase."
                ))
            return couper(nettoye, longueur) if longueur else nettoye

        title = propre("Titre", article.title, 300)
        context = propre("Contexte", article.context)
        summary = propre("Résumé", article.summary)
        # Un point clé tient sur une ligne : le formulaire de relecture en fait
        # une ligne par point, et un retour à la ligne le coupait en deux.
        key_points = [
            p for p in (
                " ".join(propre(f"Point clé {i}", point, 1000).split())
                for i, point in enumerate(article.key_points, 1)
            ) if p
        ][:8]
        citizen_impact = propre("Impact pour le citoyen", article.citizen_impact)
        suggested_question = propre("Question suggérée", article.suggested_question or "", 500)
        tags = mots_cles_propres(article.tags)
        if not (title and context and summary and key_points and citizen_impact):
            raise ReponseInvalideError(
                "Le modèle a rendu un article incomplet : aucun article n'a été enregistré. "
                "Relancez la génération."
            )

        # Les citations : des numéros d'extraits, rien d'autre. Chaque extrait
        # est UN passage (un article, ou sa suite) : la carte est celle de ce
        # passage, relue dans la base après la génération (un texte dépublié
        # entre-temps est écarté).
        objets: Dict[int, str] = {}
        for citation in article.citations:
            if not 1 <= citation.extrait <= len(lus):
                avertissements.append(Avertissement(
                    None, None,
                    f"Le modèle a cité l'extrait {citation.extrait}, qui ne fait pas partie des "
                    f"{len(lus)} extraits fournis : citation écartée.",
                ))
                continue
            if citation.extrait not in objets:
                objets[citation.extrait] = propre(
                    f"Citation de l'extrait {citation.extrait}", citation.objet,
                    CARACTERES_DE_L_EXPLICATION,
                )
        demandes = [
            DemandeDeCitation(
                passages[n - 1].law_id, passages[n - 1].ligne_id, passages[n - 1].numero,
                title=objet, suite=passages[n - 1].suite,
            )
            for n, objet in objets.items()
        ]
        origines = list(objets)
        citations, refus = await citations_verifiees(self.db, demandes, strict=False)
        for rang, raison in refus:
            avertissements.append(Avertissement(
                None, None, f"Extrait {origines[rang - 1]} : {raison} Citation écartée."
            ))
        if not citations:
            raise ReponseInvalideError(
                "Le modèle n'a cité aucun des extraits fournis : aucun article n'a été "
                "enregistré. Relancez la génération ou précisez le sujet."
            )

        # La prose relue contre les extraits LUS : ce qu'aucun ne porte vient
        # de la mémoire du modèle.
        key_points, suggested_question = _tenir_le_plafond(
            title, context, summary, key_points, citizen_impact, citations,
            suggested_question, avertissements,
        )

        champs = champs_de_prose(
            title, context, summary, key_points, citizen_impact, suggested_question, [], tags,
        ) + [(f"Citation de l'extrait {n} (explication)", objet) for n, objet in objets.items() if objet]
        avertissements += relire_la_prose(champs, lus, generation=True)

        sources = sources_des_passages(passages, lus)
        return ArticleRedige(
            title=title,
            # La rubrique imposée passe, comme toute prose, par texte_brut.
            category=texte_brut(category) or rubrique_connue(str(article.category or "")),
            context=context,
            summary=summary,
            key_points=key_points,
            citations=citations,
            citizen_impact=citizen_impact,
            suggested_question=suggested_question,
            tags=tags,
            reading_time=duree_de_lecture([context, summary, *key_points, citizen_impact]),
            sources=sources,
            avertissements=avertissements,
        )


# LA LONGUEUR DE L'ARTICLE (option 2 du propriétaire, 08/11).
# POURQUOI : le contrôle refuse (422) un article de plus de PHRASES_AU_PLUS
# phrases, mais rien n'empêchait le modèle d'en écrire plus : un brouillon de
# 26 phrases a été enregistré, puis son contrôle refusé, après un appel payé.
# Deux gardes :
#   1. le prompt annonce PHRASES_VISEES, un peu SOUS le plafond (le modèle
#      compte mal ses phrases, et le découpage du contrôle n'est pas le sien) ;
#   2. après la génération, l'article est compté avec le découpage MÊME du
#      contrôle (phrases_du_billet). Au-delà du plafond, on retire des
#      ÉLÉMENTS ENTIERS, les moins utiles d'abord : la question suggérée (une
#      invitation, pas une règle), puis les points clés du dernier au
#      quatrième (ils redisent le résumé ; trois restent, le minimum demandé).
#      Chaque retrait est un avertissement visible à la relecture.
# Jamais on ne coupe une phrase, le résumé, l'impact ni l'explication d'une
# citation : couper au milieu changerait le sens d'une règle de droit, et
# sans le dire. Si l'article reste trop long, aucun brouillon n'est
# enregistré (ReponseInvalideError) : un brouillon que le contrôle ne peut
# pas juger ne se publierait jamais. Pourquoi pas une seconde génération :
# elle doublerait le coût et la durée (DELAI_GENERATION_S), sans garantie
# d'être plus courte ; l'administrateur peut relancer lui-même.
PHRASES_VISEES = 22
POINTS_CLES_AU_MOINS = 3


def _tenir_le_plafond(
    title: str,
    context: str,
    summary: str,
    key_points: List[str],
    citizen_impact: str,
    citations: Sequence[Dict[str, Any]],
    suggested_question: str,
    avertissements: List[Avertissement],
) -> Tuple[List[str], str]:
    """Les points clés et la question, réduits sous PHRASES_AU_PLUS, ou ReponseInvalideError."""
    # Import tardif : controle_du_billet importe ce module.
    from app.services.controle_du_billet import (
        PHRASES_AU_PLUS, ContenuDuBillet, phrases_du_billet,
    )

    def compte(points: List[str], question: str) -> int:
        return len(phrases_du_billet(ContenuDuBillet(
            title=title, context=context, summary=summary, key_points=points,
            citizen_impact=citizen_impact, citations=list(citations),
            suggested_question=question,
        )))

    total = compte(key_points, suggested_question)
    if total <= PHRASES_AU_PLUS:
        return key_points, suggested_question
    depart = total
    points = list(key_points)
    if suggested_question and total > PHRASES_AU_PLUS:
        suggested_question = ""
        total = compte(points, suggested_question)
        avertissements.append(Avertissement(
            "Question suggérée", None,
            f"Article trop long pour le contrôle ({depart} phrases, {PHRASES_AU_PLUS} au plus) : "
            "la question suggérée a été retirée ; écrivez-en une courte si besoin.",
        ))
    while total > PHRASES_AU_PLUS and len(points) > POINTS_CLES_AU_MOINS:
        retire = points.pop()
        total = compte(points, suggested_question)
        avertissements.append(Avertissement(
            f"Point clé {len(points) + 1}", None,
            f"Article trop long pour le contrôle ({depart} phrases, {PHRASES_AU_PLUS} au plus) : "
            f"le point clé « {couper(retire, 120)} » a été retiré en entier.",
        ))
    if total > PHRASES_AU_PLUS:
        logger.warning(f"⚠️ Blog : article de {depart} phrases, {total} après retraits")
        raise ReponseInvalideError(
            f"Le modèle a rédigé un article trop long pour le contrôle ({depart} phrases, "
            f"{PHRASES_AU_PLUS} au plus) : aucun article n'a été enregistré. Relancez la "
            "génération."
        )
    logger.info(f"📰 Blog : article ramené de {depart} à {total} phrases")
    return points, suggested_question


def sources_des_passages(
    passages: Sequence[Passage], lus: Sequence[ChunkResult]
) -> List[Dict[str, Any]]:
    """
    Les sources, sous la forme du contrat : {n, law_id, law_title,
    article_number, article_id, suite, extrait}, numérotées comme le modèle
    les a lues. Gardées avec l'article (tour 4) : le contrôle IA juge contre
    elles, et l'administrateur relit contre elles, même des jours après.
    """
    return [
        {
            "n": n,
            "law_id": passage.law_id,
            "law_title": extrait.law_title,
            "article_number": passage.numero,
            "article_id": passage.ligne_id,
            "suite": passage.suite,
            "extrait": extrait.content,
        }
        for n, (passage, extrait) in enumerate(zip(passages, lus), 1)
    ]


# Le sujet d'un article déjà écrit, pour en reconstruire les sources : sa
# longueur reste celle d'un sujet de génération (BlogGenerateRequest.topic),
# sur laquelle la preuve de pertinence a été calibrée (SEUILS_DU_BLOG).
CARACTERES_DU_SUJET = 300


def sujet_de_l_article(titre: Optional[str], contexte: Optional[str]) -> str:
    """Le titre, puis le contexte, en une ligne, coupés entre deux mots à 300 caractères."""
    titre = " ".join(prose_publique(titre).split())
    contexte = " ".join(prose_publique(contexte).split())
    if titre and contexte and not titre.endswith(("?", "!", ".", ":", "…")):
        titre += "."
    return couper(" ".join(m for m in (titre, contexte) if m), CARACTERES_DU_SUJET)


_SANS_SOURCE_DE_L_ARTICLE = (
    "Aucun texte publié de la base ne traite clairement le sujet de cet article (son titre et "
    "son contexte) : les sources n'ont pas été reconstruites, rien n'a changé. Reformulez le "
    "titre ou le contexte avec les mots de la loi (par exemple « bail », « licenciement », "
    "« succession »), puis recommencez."
)


def couper_au_paragraphe(contenu: str, longueur: int) -> str:
    """Au plus `longueur` caractères, coupé à la fin d'un paragraphe ou d'une phrase."""
    if len(contenu) <= longueur:
        return contenu
    morceau = contenu[:longueur]
    for separateur in ("\n\n", "\n", ". "):
        coupe = morceau.rfind(separateur)
        if coupe > longueur // 2:
            return morceau[:coupe + (1 if separateur == ". " else 0)].rstrip()
    return couper(morceau, longueur)
