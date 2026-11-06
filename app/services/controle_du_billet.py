"""
Le contrôle IA d'un article du blog, sa relecture humaine, et l'état de sa
publication (tours 3 et 4).

POURQUOI. Des citations vérifiées ne rendent pas la prose juste. Essai réel du
30/10/2026 (data/blog/e2e-tour2, dépôt du backend), deux articles publiés sous
des cartes « vérifiées », sans un avertissement :
- bail : le TITRE (« Expulsion locative sans jugement : quels cas au
  Cameroun ? »), le résumé, deux points clés et l'impact disent qu'on peut
  expulser un locataire sans décision de justice en cas de reprise ou de motif
  grave. Les extraits cités (Acte uniforme sur le droit commercial général,
  art. 127 et 128) parlent du refus de RENOUVELER un bail professionnel, et
  l'art. 133 exige la juridiction. L'explication de la citation de l'art. 1752
  du Code civil l'étend à tout manquement, loyer compris, et invente une mise
  en demeure : l'article ne vise que le locataire qui ne garnit pas la maison ;
- licenciement : le résumé et l'impact disent que « le salarié doit prouver
  le caractère abusif », contre l'art. 39 (3) du Code du travail, cité dans le
  même article (« il appartient à l'employeur d'apporter la preuve »).

DÉCISION DU PROPRIÉTAIRE (29-30/10) : « Relecture + contrôle IA ». Un contrôle
automatique compare chaque phrase aux sources ; une relecture humaine de
chaque partie est obligatoire avant publication, et l'API l'impose.

CE QUE FAIT CE MODULE.
1. Les PARTIES relues d'un article (PARTIES) : le TITRE (tour 4 : la phrase la
   plus lue, h1, carte et <title>, et celle qui portait la règle fausse de
   l'essai du bail), le contexte, le résumé, les points clés, l'impact, les
   explications des citations (avec l'article que chacune accompagne et son
   extrait), la question suggérée. La rubrique, l'adresse et les mots-clés
   n'en sont pas.
2. Les SOURCES : les extraits lus par le modèle à la génération, gardés avec
   l'article (tour 4). Le contrôle juge contre elles, sous leurs numéros, et
   l'administrateur relit contre elles. Un article sans sources (article
   d'avant le 30/10, ou écrit à la main) est « sources_absentes » : on les
   reconstruit d'abord (routes/blog : POST /{id}/sources).
3. Le contrôle : des appels au modèle DU CONTRÔLE (tour 6 : mistral-large-4,
   BLOG_CONTROLE_MODELE ; la génération garde le sien), schéma JSON strict,
   température 0, d'au plus PHRASES_PAR_APPEL phrases chacun, L'UN APRÈS
   L'AUTRE, et un seul contrôle à la fois par modèle (tour 6 : 20 000 jetons
   par minute ; voir PHRASES_AU_PLUS). Revue du tour 6 : il se PLANIFIE avant
   son premier lot, dans l'échéance de la route, attente d'un autre contrôle
   et attentes du budget de jetons comprises, et ne part que si son dernier
   lot finit à temps (BudgetDuControle). controle_ia.modele note le modèle qui
   a VRAIMENT répondu ; jamais de modèle de secours. Pour
   chaque phrase, le modèle réécrit ce qu'elle affirme (renvois résolus,
   présupposés compris), COPIE le passage de l'extrait qui s'en approche, puis
   juge : « soutenue », « contredite », « non_soutenue », « hors_sources ». Le
   serveur vérifie que le passage copié figure bien dans les extraits : une
   phrase dite soutenue ou contredite sans passage réel devient « non
   soutenue », à relire (tour 4 : au tour 3, le modèle inventait l'appui —
   « tandis que le salarié peut prouver le caractère abusif » —, et 4 des 7
   phrases fausses de l'essai passaient « soutenues »). Chaque morceau entre
   guillemets est cherché à part, dans l'extrait qui le porte (mesure réelle
   du 04/11, data/blog/tour4 : lu d'un bloc, un passage vrai copié de deux
   extraits ne figurait dans aucun, et 17 des 18 phrases justes passaient
   « À vérifier »). Tour 5 : chaque mot qui compte d'un morceau, son premier
   et son dernier compris, doit être dans l'extrait, et la raison cite les
   mots DE L'EXTRAIT, jamais ceux du modèle ; une coupure « … » ne saute
   jamais une négation ni une exception de l'extrait, et ne joint jamais deux
   phrases en une citation (voir « LE PASSAGE COPIÉ »).
4. Les EMPREINTES. Celle du contrôle couvre le texte de toutes les parties ET
   les sources : tout changement de l'un ou de l'autre le périme
   (« controle_a_relancer »). Une case de relecture atteste le texte de sa
   partie, les sources, et les verdicts que le contrôle donnait à cette
   partie : elle tombe si l'un d'eux change. Cochée avant le contrôle, elle ne
   compte donc plus après lui ; un nouveau contrôle qui juge autrement une
   partie la fait relire (revue du tour 3 : on pouvait cocher, puis contrôler,
   et publier sans avoir vu les nouveaux verdicts). Tour 5 : une case ne se
   coche que contre le contrôle ENREGISTRÉ, que la requête nomme par son
   `fait_le` (relecture.controle) ; chaque contrôle a le sien
   (relecture_perimee, fait_le_apres). Une case qui tombe avec un nouveau
   contrôle est EFFACÉE (relecture_apres_un_controle) : un contrôle suivant
   qui redonne les verdicts d'avant ne la fait pas revivre.
5. L'état de publication : « publiable » seulement si les sources sont là,
   toutes les citations vérifiées (au moins une), le contrôle à jour, aucune
   phrase « contredite », chaque phrase de règle non trouvée dans les
   extraits réécrite ou confirmée (tour 7) et chaque partie relue. Sinon, le
   premier blocage dans l'ordre où l'administrateur peut le lever : sources,
   citations, contrôle, phrases contredites, phrases à confirmer, relecture.

TOUR 7 (essai réel du 04/11, data/blog/tour6-resultat.json).
- UN SEUL APPEL PAR ARTICLE pour un modèle à budget de jetons (mistral-large-4).
  POURQUOI : sa réflexion domine le temps d'un appel, quelle que soit sa
  taille. Mesuré : des lots de 11-12 phrases en 40,5 à 76,0 s, mais un appel
  de 18 phrases en 73,1 s, et les 23 phrases du bail en UN appel en 78,7 s.
  En deux lots, l'article payait deux réflexions (100,9 et 106,4 s, et un 504
  à 120 s après deux lots payés, le plan les croyant à 3,42 s par phrase). Le
  plan compte désormais un appel au pire mesuré (DUREE_D_UN_APPEL_AU_PIRE_S),
  et non des secondes par phrase.
- DÉCISION DU PROPRIÉTAIRE (04/11) : « Confirmer phrase par phrase ». Une
  phrase qui énonce une RÈGLE (nature « regle ») et que le contrôle n'a pas
  trouvée dans les extraits (« non_soutenue », « À vérifier » compris, ou
  « hors_sources ») est réécrite (un nouveau contrôle la juge alors), ou
  CONFIRMÉE une à une par l'administrateur (« J'ai vérifié cette phrase dans
  le texte ») avant publication (phrases_a_confirmer). POURQUOI : à l'essai,
  les 10 phrases du bail qui disaient ou présupposaient l'expulsion sans juge
  étaient toutes « À vérifier », du même gris que les fausses alertes, et 7
  cases cochées suffisaient à publier « peut être expulsé sans décision de
  justice ». Une confirmation atteste la phrase, son verdict et les sources ;
  un nouveau contrôle la fait tomber comme une case (relecture_apres_un_controle).

Ce module ne publie rien et n'écrit rien en base : la route enregistre.
"""

import asyncio
import bisect
import hashlib
import json
import logging
import math
import re
import time
import unicodedata
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.limiteur import Limiteur
from app.services.article_reference import normalize_number
from app.services.blog_generator_service import (
    CARACTERES_PAR_EXTRAIT,
    ErreurDeGeneration,
    QuotaError,
    ReponseInvalideError,
    SansSourceError,
    SaturationError,
    couper,
    couper_au_paragraphe,
    passages_des_citations,
    texte_brut,
)
from app.services.llm import (
    ERREURS_BUDGET,
    ERREURS_LLM,
    ERREURS_QUOTA,
    ERREURS_SATURATION,
    get_llm_service,
)
from app.services.mistral_service import MistralService, mistral_du_controle, modele_du_controle

logger = logging.getLogger(__name__)


# ==================== RÉGLAGES ====================

# Délai du contrôle (route POST /{id}/controle, et au plus après la génération).
# POURQUOI 165 s (essai réel du 06/11) : UN appel au pire, marge comprise
# (DUREE_D_UN_APPEL_AU_PIRE_S + MARGE_DU_PLAN_S = 135 s), et 30 s pour attendre
# un autre contrôle ou que le budget de jetons se refasse. Au-delà, le plan
# refuse avant tout envoi (503 et Retry-After) : une attente plus longue tient
# l'administrateur devant un sablier sans rien juger, quand une relance au
# Retry-After juge en 40 à 105 s (essais réels du 04/11 et du 06/11).
DELAI_CONTROLE_S = 165
# Le contrôle n'invente rien : il juge.
CONTROLE_TEMPERATURE = 0.0
# La sortie d'un appel, réflexion comprise. POURQUOI 14 000 : l'essai réel du
# 06/11 a buté sur l'ancien plafond (fin=MAX_TOKENS, 4 093 + 12 000 jetons en
# 105 s, donc aucun verdict) ; le plus mesuré jusque-là était 10 720 (le bail).
# Pas davantage : la plus grosse entrée mesurée (5 582 jetons) et ce plafond
# tiennent encore ensemble dans les 20 000 jetons par minute de la clé, et chaque
# millier de jetons coûte ~9 s de plus au pire. Le besoin réel de l'appel coupé
# reste inconnu : s'il dépasse 14 000, il échoue sans verdict, comme avant.
CONTROLE_MAX_TOKENS = 14_000
# LES APPELS (tour 5, défaut D2 de la revue du tour 4 ; tour 6, mistral-large-4).
# TOUR 7 : pour un modèle à budget de jetons (mistral-large-4), UN SEUL appel
# par article jusqu'à PHRASES_AU_PLUS (lots_du_controle(…, un_seul_appel=True)) ;
# ce qui suit sur les lots ne vaut plus que pour les autres modèles.
# Le modèle juge au plus PHRASES_PAR_APPEL phrases par appel, sans jamais
# couper une partie (lots_du_controle).
# Tour 5 (ministral-14b, 937 500 jetons par minute) : les appels partaient
# ENSEMBLE, l'article se jugeait dans le temps de son plus gros appel, d'où
# 80 phrases permises en 4 appels de 20 (~79 s au pire mesuré, 3,96 s par
# phrase ; data/blog/tour5/api/mesure_d2.py).
# TOUR 6 : LES APPELS PARTENT L'UN APRÈS L'AUTRE. POURQUOI : mistral-large-4,
# le modèle du contrôle depuis la décision du 04/11, n'a que 20 000 jetons
# par minute à la clé, et chaque appel reprend le système et TOUS les
# extraits ; deux appels ensemble se disputeraient ce budget, et le second
# recevrait un 429 au milieu de l'article. En série, l'article se juge dans
# la SOMME de ses appels : le plafond de 80 phrases ne tient plus.
# MESURÉ (04/11 à 18:06, data/blog/tour5-modeles/large4-low.log ; effort
# « low », un appel par article, le prompt et le lecteur d'aujourd'hui) :
# - bail : 23 phrases en 78,7 s, soit 3,4 s par phrase ; 4 046 jetons
#   d'entrée, 10 720 de sortie dont 6 591 de réflexion (466 par phrase, 136
#   par seconde) ;
# - travail : 21 phrases en 51,2 s, 2,4 s par phrase ; 5 582 + 7 992.
# Au pire mesuré (3,42 s par phrase), 80 phrases en série prendraient ~274 s,
# bien au-delà des 165 s de DELAI_CONTROLE_S.
# PHRASES_AU_PLUS = 25, au-dessus des 20 à 24 phrases d'un article généré
# (essais réels du 04/11). Pour un modèle à budget de jetons, c'est aussi la
# taille de son UN appel (tour 7). Pour les autres, un article de 21 à 25
# phrases se juge en deux appels, au plus trois si ses parties tombent mal
# (jamais quatre : deux appels qui se suivent comptent ensemble plus de
# PHRASES_PAR_APPEL phrases, sinon ils n'en feraient qu'un).
# LA PREUVE DU PLAFOND (tour 7 ; TestLesLots la refait). Seul, un contrôle est
# UN appel : DUREE_D_UN_APPEL_AU_PIRE_S + MARGE_DU_PLAN_S = 135 s, quelle que
# soit sa taille, sous les 165 s de DELAI_CONTROLE_S ; il reste 30 s pour
# attendre un autre contrôle (le verrou du modèle) ou le budget de jetons. Un
# contrôle qui devrait attendre plus est refusé AVANT tout envoi, avec le
# Retry-After où son plan tient ; relancé alors, il finit dans le délai.
# PHRASES_PAR_APPEL reste 20 pour les modèles sans budget de jetons (leurs
# appels partent l'un après l'autre, chacun loin de CONTROLE_MAX_TOKENS).
PHRASES_PAR_APPEL = 20
PHRASES_AU_PLUS = 25
# LE PLAN DU CONTRÔLE (revue du tour 6, BudgetDuControle ; tour 7) : UN APPEL,
# au pire MESURÉ, quelle que soit sa taille. POURQUOI par appel et non par
# phrase : la réflexion du modèle domine. Mesuré (04/11, data/blog/
# tour5-modeles/large4-low.log et l'essai réel du tour 6) : 78,7 s pour les 23
# phrases du bail en un appel, 73,1 s pour 18, 76,0 s pour un lot de 12, 40,5
# à 60,3 s pour des lots de 10 ou 11. Le plan à 3,42 s par phrase prévoyait
# 41 s pour le lot de 12 : il en a pris 76, et le contrôle a fini en 504
# après deux lots payés.
# POURQUOI 125 s (essai réel du 06/11) : 80 s ne tenait pas, un appel a pris
# 105 s pour 12 000 jetons. Au plafond et à cette vitesse, 14 000 jetons
# prennent ~122,5 s : c'est le plafond qui borne la durée, plus une mesure.
DUREE_D_UN_APPEL_AU_PIRE_S = 125.0
# La sortie qu'un appel peut prendre au budget de jetons, réflexion comprise :
# son plafond, CONTROLE_MAX_TOKENS (12 000 atteints le 06/11). Elle ne sert
# qu'à prévoir l'attente du contrôle suivant dans la file.
JETONS_DE_SORTIE_D_UN_APPEL_AU_PIRE = CONTROLE_MAX_TOKENS
# La vitesse de sortie la plus lente mesurée : 12 000 jetons en 105 s (114 par
# seconde, essai réel du 06/11 ; les appels du 04/11, 136 à 169). POURQUOI elle
# compte : un appel rend au plus CONTROLE_MAX_TOKENS, donc dure au plus ~122,5 s
# à cette vitesse, sous les 125 s du plan, et les 135 s qu'il garde avec la marge.
JETONS_PAR_SECONDE_AU_PIRE = 12_000 / 105
# POURQUOI une marge : le pire n'est que le pire de quelques mesures ; un
# contrôle prévu au bord du délai finissait en 504 après des appels payés
# (revue du tour 6). Elle reste à l'appel : il ne part jamais s'il ne peut
# finir MARGE_DU_PLAN_S avant l'échéance au pire mesuré (avant_le_lot).
MARGE_DU_PLAN_S = 10.0
# L'attente que le plan laisse à l'appel, en plus de ce qu'il prévoit.
# POURQUOI (tour 7) : le plan se fait un instant avant que le client réserve,
# et la place du limiteur (une requête à la fois) se prend en temps réel. Un
# plan accepté au bord (relancé au Retry-After annoncé, ou la part de
# /generate tout juste suffisante) laissait 0 s d'attente, et le client le
# refusait pour quelques microsecondes : 503 « saturé » après un plan accepté.
# Une seconde de plus au plus : l'appel finit encore 9 s avant l'échéance.
TOLERANCE_D_ATTENTE_S = 1.0
# UN CONTRÔLE À LA FOIS PAR MODÈLE (tour 6, un_controle_a_la_fois). Un
# contrôle qui en trouve un autre en cours l'attend 20 s au plus, et pas
# davantage que la marge de son plan (revue du tour 6) : la file ne le garde
# que s'il peut finir dans son délai APRÈS ceux de devant, dans les jetons
# qu'ils auront pris. Sinon, 503 « saturation » aussitôt, avec le Retry-After
# où son plan tiendrait. POURQUOI garder 20 s au plus : c'est l'attente que la
# route promet (avenant du tour 6) ; le plan, lui, ne fait que la raccourcir.
# Sans plan (un modèle sans budget de jetons, une doublure), 20 s, puis une
# minute de Retry-After, le temps que le budget se refasse.
ATTENTE_D_UN_AUTRE_CONTROLE_S = 20.0
REPRISE_APRES_UN_AUTRE_CONTROLE_S = 60
# Les extraits de la génération en font 24 000 au plus ; les citations
# ajoutées depuis prennent la suite, et au moins de quoi lire un article.
CARACTERES_DES_EXTRAITS_DU_CONTROLE = 36_000
CARACTERES_D_UN_EXTRAIT_AJOUTE_AU_MOINS = 800
# La raison montrée à l'administrateur : celle du modèle, puis le passage
# copié qui l'appuie (« Extrait 5 : « … » »).
CARACTERES_DE_LA_RAISON = 600
CARACTERES_DU_PASSAGE_CITE = 240


# ==================== PARTIES, VERDICTS, ÉTATS ====================

# Les sept parties que l'administrateur relit (relecture.parties).
PARTIES = ("titre", "contexte", "resume", "points_cles", "impact", "explications", "question")
# La partie relue de chaque partie de phrase (phrases[].partie).
PARTIE_RELUE = {
    "titre": "titre", "contexte": "contexte", "resume": "resume", "point_cle": "points_cles",
    "impact": "impact", "explication": "explications", "question": "question",
}
NOMS_DES_PARTIES = {
    "titre": "titre", "contexte": "contexte", "resume": "résumé", "points_cles": "points clés",
    "impact": "impact pour le citoyen", "explications": "explications des citations",
    "question": "question suggérée",
}
VERDICTS = ("soutenue", "non_soutenue", "contredite", "hors_sources")
# Les verdicts d'une phrase de règle à réécrire ou à confirmer (tour 7) : le
# contrôle ne l'a pas trouvée dans les extraits. « contredite » n'en est pas :
# elle bloque, et se corrige (phrases_contredites).
VERDICTS_A_CONFIRMER = ("non_soutenue", "hors_sources")
# Dans l'ordre où l'administrateur peut lever les blocages, « publiable » à part.
ETATS = (
    "publiable", "sources_absentes", "citations_non_verifiees", "controle_a_relancer",
    "phrases_contredites", "phrases_a_confirmer", "relecture_incomplete",
)


class TexteTropLongError(ErreurDeGeneration):
    """Aucune phrase, plus de PHRASES_AU_PLUS, ou une partie de plus de PHRASES_PAR_APPEL (lots_du_controle)."""

    statut = 422
    code = "entree_invalide"


class SourcesAbsentesError(ErreurDeGeneration):
    """L'article n'a pas de sources enregistrées : rien contre quoi juger."""

    statut = 422
    code = "sources_absentes"


def maintenant_iso() -> str:
    """La date et l'heure, en UTC, à la seconde (« 2026-10-30T10:15:00+00:00 »)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _instant(valeur: Any) -> Optional[datetime]:
    try:
        instant = datetime.fromisoformat(str(valeur))
    except (TypeError, ValueError):
        return None
    return instant if instant.tzinfo is not None else instant.replace(tzinfo=timezone.utc)


def fait_le_apres(fait_le: str, *precedents: Any) -> str:
    """
    Le `fait_le` d'un contrôle qu'on enregistre : le sien, ou une seconde après
    le plus récent des `precedents` (le contrôle remplacé, celui contre lequel
    la relecture enregistrée a été cochée) s'il ne vient pas après.

    POURQUOI (avenant du tour 5). Une case cochée nomme le contrôle relu par son
    `fait_le` : deux contrôles finis dans la même seconde (deux onglets qui
    relancent ensemble) porteraient le même, et une case cochée contre l'un
    passerait pour cochée contre l'autre. À la seconde, la forme du contrat
    ne change pas.
    """
    nouveau = _instant(fait_le)
    avant = [i for i in (_instant(p) for p in precedents if p) if i is not None]
    if nouveau is None or not avant or nouveau > max(avant):
        return fait_le
    return (max(avant) + timedelta(seconds=1)).astimezone(timezone.utc).isoformat(timespec="seconds")


# ==================== LE TEXTE CONTRÔLÉ ====================


@dataclass
class ContenuDuBillet:
    """
    Les textes qu'un article montre au lecteur et que le contrôle juge, et les
    SOURCES contre lesquelles il les juge. Lu d'un article enregistré (`de`),
    ou bâti par la route à partir d'un article ET d'une modification, avant de
    l'appliquer : l'état de publication d'une modification se juge avant de
    toucher à l'article.
    """

    title: str = ""
    context: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    citizen_impact: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    suggested_question: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def de(cls, post: Any) -> "ContenuDuBillet":
        return cls(
            title=str(getattr(post, "title", None) or ""),
            context=str(post.context or ""),
            summary=str(post.summary or ""),
            key_points=[p for p in (post.key_points or []) if isinstance(p, str)],
            citizen_impact=str(post.citizen_impact or ""),
            citations=[c for c in (post.citations or []) if isinstance(c, dict)],
            suggested_question=str(post.suggested_question or ""),
            sources=[s for s in (getattr(post, "sources", None) or []) if isinstance(s, dict)],
        )


def _entier(valeur: Any) -> Optional[int]:
    if isinstance(valeur, bool):
        return None
    try:
        return int(valeur)
    except (TypeError, ValueError):
        return None


def _canon(texte: Any) -> str:
    """
    Un texte sous sa forme canonique : les blancs ramenés à une espace. Une
    différence de blancs n'est pas une modification (revue du tour 3 : un
    renvoi « (extrait 2) » retiré seul sur sa ligne laissait trois sauts de
    ligne, que le formulaire renvoyé ramenait à deux ; chaque enregistrement
    « modifiait » la partie et périmait le contrôle, en boucle).
    """
    return " ".join(str(texte or "").split())


def cle_d_article(law_id: Any, article_id: Any, numero: Any, suite: Any) -> Tuple[Any, ...]:
    """Un passage de la base : texte, ligne, numéro d'article, début ou suite."""
    return (_entier(law_id), _entier(article_id), normalize_number(str(numero or "")), bool(suite))


def _identite_de_citation(citation: Dict[str, Any]) -> List[Any]:
    """
    Une explication ne se relit qu'avec l'article qu'elle accompagne, et
    l'extrait qu'en montre la carte (revue du tour 3 : une citation renvoyée
    par le formulaire était relue en base, et redevenait « vérifiée » avec un
    NOUVEL extrait si l'article avait changé, sans contrôle ni relecture).
    """
    return [
        *cle_d_article(
            citation.get("law_id"), citation.get("article_id"), citation.get("article_number"),
            citation.get("suite"),
        ),
        citation.get("verifie") is True,
        _canon(citation.get("title")),
        _canon(citation.get("excerpt")),
    ]


def valeurs_des_parties(contenu: ContenuDuBillet) -> Dict[str, Any]:
    """Le texte de chaque partie relue, sous sa forme canonique (_canon)."""
    return {
        "titre": _canon(contenu.title),
        "contexte": _canon(contenu.context),
        "resume": _canon(contenu.summary),
        "points_cles": [_canon(p) for p in contenu.key_points if _canon(p)],
        "impact": _canon(contenu.citizen_impact),
        # Une citation ajoutée, retirée ou remplacée change aussi les explications.
        "explications": [_identite_de_citation(c) for c in contenu.citations],
        "question": _canon(contenu.suggested_question),
    }


def _empreinte(valeur: Any) -> str:
    brut = json.dumps(valeur, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(brut.encode("utf-8")).hexdigest()


# ==================== LES SOURCES ====================


def sources_valables(sources: Any) -> List[Dict[str, Any]]:
    """
    Les sources gardées avec l'article, sous la forme du contrat — {n, law_id,
    law_title, article_number, article_id, suite, extrait} —, triées par
    numéro. Une entrée sans numéro, sans extrait ou au numéro déjà vu est
    ignorée.
    """
    resultat: Dict[int, Dict[str, Any]] = {}
    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, dict):
            continue
        n = _entier(source.get("n"))
        extrait = str(source.get("extrait") or "").strip()
        if n is None or n < 1 or n in resultat or not extrait:
            continue
        resultat[n] = {
            "n": n,
            "law_id": _entier(source.get("law_id")),
            "law_title": str(source.get("law_title") or ""),
            "article_number": str(source.get("article_number") or ""),
            "article_id": _entier(source.get("article_id")),
            "suite": source.get("suite") is True,
            "extrait": extrait,
        }
    return [resultat[n] for n in sorted(resultat)]


def _sources_canon(sources: Any) -> List[List[Any]]:
    return [
        [s["n"], *cle_d_article(s["law_id"], s["article_id"], s["article_number"], s["suite"]),
         _canon(s["extrait"])]
        for s in sources_valables(sources)
    ]


def memes_sources(une: Any, autre: Any) -> bool:
    return _sources_canon(une) == _sources_canon(autre)


def sources_affichees(sources: Any, controle: Any) -> List[Dict[str, Any]]:
    """
    Les sources que l'administration montre (tour 4) : celles de l'article,
    puis les extraits que le dernier contrôle a ajoutés pour les citations
    ajoutées depuis la génération (leurs numéros sont ceux des `extraits` de
    ses verdicts). Jamais dans une réponse publique.
    """
    affichees = sources_valables(sources)
    numeros = {s["n"] for s in affichees}
    ajoutes = controle.get("extraits_ajoutes") if isinstance(controle, dict) else None
    for extrait in sources_valables(ajoutes):
        if extrait["n"] not in numeros:
            numeros.add(extrait["n"])
            affichees.append(extrait)
    return sorted(affichees, key=lambda s: s["n"])


# ==================== EMPREINTES ====================


def empreinte_du_controle(contenu: ContenuDuBillet) -> str:
    """
    L'empreinte que garde controle_ia : le texte de TOUTES les parties, et les
    sources. Un changement de l'un ou de l'autre périme le contrôle.
    """
    return _empreinte({
        "parties": valeurs_des_parties(contenu), "sources": _sources_canon(contenu.sources),
    })


def empreintes_des_parties(contenu: ContenuDuBillet) -> Dict[str, Optional[str]]:
    """L'empreinte du TEXTE de chaque partie ; None pour une partie vide (rien à relire)."""
    return {
        partie: _empreinte([partie, valeur]) if valeur else None
        for partie, valeur in valeurs_des_parties(contenu).items()
    }


def _verdicts_de_la_partie(controle: Any, partie: str) -> Optional[List[List[str]]]:
    """Les (phrase, verdict) que le contrôle enregistré donne à la partie ; None sans contrôle."""
    if not isinstance(controle, dict) or not isinstance(controle.get("phrases"), list):
        return None
    return [
        [str(p.get("texte") or ""), str(p.get("verdict") or "")]
        for p in controle["phrases"]
        if isinstance(p, dict) and PARTIE_RELUE.get(str(p.get("partie"))) == partie
    ]


def empreintes_de_relecture(contenu: ContenuDuBillet, controle: Any) -> Dict[str, Optional[str]]:
    """
    Ce qu'atteste la case cochée d'une partie : son texte, les sources, et les
    verdicts que le contrôle enregistré lui donne. None pour une partie vide.
    """
    sources = _sources_canon(contenu.sources)
    return {
        partie: (
            _empreinte([partie, valeur, sources, _verdicts_de_la_partie(controle, partie)])
            if valeur else None
        )
        for partie, valeur in valeurs_des_parties(contenu).items()
    }


# ==================== LES PHRASES ====================

# Une fin de phrase : « . », « ! », « ? », « … », guillemets ou parenthèse
# fermants compris (l'espace française avant « » » aussi), devant une
# majuscule, un chiffre ou un guillemet ouvrant.
_FIN_DE_PHRASE = re.compile(
    r"[.!?…]+(?:[   ]*[\"»”’)\]])*(?=\s+[«\"“(\[]?[A-ZÀ-ÖØ-Þ0-9])"
)
# Le mot qui précède le point, s'il en finit un (« art. 39 », « M. Biya »).
_DERNIER_MOT = re.compile(r"([^\W\d_]+)\W*$")
_ABREVIATIONS = frozenset({
    "art", "arts", "al", "alin", "no", "nos", "cf", "ex", "pp", "mm", "mme", "mmes", "mlle",
    "dr", "me", "st", "ste", "chap", "sect", "vol", "ss", "op", "cit", "ord", "jr", "cass",
    "civ", "soc", "crim", "bull", "ref", "réf", "spéc", "spec", "env", "max", "min", "doc",
    "ann", "anc", "nouv",
})
_DES_LETTRES = re.compile(r"[^\W\d_]")


def decouper_en_phrases(texte: Optional[str]) -> List[str]:
    """
    Les phrases d'un texte, dans l'ordre. Une phrase ne passe jamais un retour
    à la ligne ; « art. 39 », « al. 3 » ou « M. » ne coupent pas.
    """
    phrases: List[str] = []
    for ligne in (texte or "").splitlines():
        debut = 0
        for fin in _FIN_DE_PHRASE.finditer(ligne):
            mot = _DERNIER_MOT.search(ligne[debut:fin.start()])
            if mot and (len(mot.group(1)) == 1 or mot.group(1).lower() in _ABREVIATIONS):
                continue
            phrase = ligne[debut:fin.end()].strip()
            if _DES_LETTRES.search(phrase):
                phrases.append(phrase)
            debut = fin.end()
        reste = ligne[debut:].strip()
        if _DES_LETTRES.search(reste):
            phrases.append(reste)
    return phrases


@dataclass(frozen=True)
class PhraseAJuger:
    id: str
    partie: str
    """titre, contexte, resume, point_cle, impact, explication ou question."""
    index: int
    """Le rang de l'élément dans sa partie, à partir de 0 : le point clé, la citation ; 0 sinon."""
    texte: str


def phrases_du_billet(contenu: ContenuDuBillet) -> List[PhraseAJuger]:
    """Les phrases de l'article, dans l'ordre de lecture (le titre d'abord), identifiées P1, P2…"""
    elements: List[Tuple[str, int, str]] = [
        ("titre", 0, contenu.title), ("contexte", 0, contenu.context),
        ("resume", 0, contenu.summary),
    ]
    elements += [("point_cle", i, p) for i, p in enumerate(contenu.key_points)]
    elements += [("impact", 0, contenu.citizen_impact)]
    elements += [
        ("explication", i, str(c.get("title") or "")) for i, c in enumerate(contenu.citations)
    ]
    elements += [("question", 0, contenu.suggested_question)]
    phrases: List[PhraseAJuger] = []
    for partie, index, texte in elements:
        for phrase in decouper_en_phrases(texte):
            phrases.append(PhraseAJuger(f"P{len(phrases) + 1}", partie, index, phrase))
    return phrases


# ==================== LES EXTRAITS ====================


@dataclass(frozen=True)
class ExtraitDuControle:
    n: int
    intitule: str
    numero: str
    suite: bool
    texte: str


def _porte(source_texte: str, passage_texte: str) -> bool:
    """Le texte d'une source contient-il le début du passage que la base porte aujourd'hui ?"""
    debut = _canon(passage_texte)[:200]
    return bool(debut) and debut in _canon(source_texte)


async def extraits_du_controle(
    db: AsyncSession, contenu: ContenuDuBillet
) -> Tuple[List[ExtraitDuControle], Dict[int, int], List[Dict[str, Any]]]:
    """
    Les extraits contre lesquels juger ; pour chaque citation (son rang), le
    numéro de l'extrait qu'elle accompagne ; et les extraits AJOUTÉS, sous la
    forme des sources, que le contrôle garde (sources_affichees).

    - Les SOURCES de l'article, sous leurs numéros : ce que le modèle a lu
      pour écrire, et ce que l'administrateur voit numéroté pendant la
      relecture (les `extraits` d'une phrase renvoient à ces numéros).
    - Le passage de chaque citation VÉRIFIÉE qu'aucune source ne porte
      (citation ajoutée depuis), relu en base, à la suite. De même quand
      l'article a changé dans la base depuis la génération : l'explication se
      juge sur le texte d'aujourd'hui, celui de la carte.
    Une citation non vérifiée n'apporte aucun extrait : son texte n'a jamais
    été lu dans la base.
    """
    extraits: List[ExtraitDuControle] = []
    par_article: Dict[Tuple[Any, ...], Tuple[int, str]] = {}
    place = CARACTERES_DES_EXTRAITS_DU_CONTROLE
    sources = sources_valables(contenu.sources)
    for source in sources:
        texte = couper_au_paragraphe(source["extrait"], CARACTERES_PAR_EXTRAIT)
        extraits.append(ExtraitDuControle(
            source["n"], source["law_title"], source["article_number"], source["suite"], texte,
        ))
        par_article.setdefault(cle_d_article(
            source["law_id"], source["article_id"], source["article_number"], source["suite"],
        ), (source["n"], texte))
        place -= len(texte)

    rattachement: Dict[int, int] = {}
    ajoutes: List[Dict[str, Any]] = []
    prochain = max((s["n"] for s in sources), default=0) + 1
    passages = await passages_des_citations(db, contenu.citations)
    for rang, trouve in enumerate(passages):
        if trouve is None:
            continue
        intitule, passage = trouve
        cle = cle_d_article(passage.law_id, passage.ligne_id, passage.numero, passage.suite)
        connu = par_article.get(cle)
        if connu is None or not _porte(connu[1], passage.texte):
            longueur = min(CARACTERES_PAR_EXTRAIT, max(place, CARACTERES_D_UN_EXTRAIT_AJOUTE_AU_MOINS))
            texte = couper_au_paragraphe(passage.texte.strip(), longueur)
            extraits.append(ExtraitDuControle(prochain, intitule, passage.numero, passage.suite, texte))
            ajoutes.append({
                "n": prochain, "law_id": passage.law_id, "law_title": intitule,
                "article_number": passage.numero, "article_id": passage.ligne_id,
                "suite": passage.suite, "extrait": texte,
            })
            connu = par_article[cle] = (prochain, texte)
            prochain += 1
            place -= len(texte)
        rattachement[rang] = connu[0]
    extraits.sort(key=lambda e: e.n)
    return extraits, rattachement, ajoutes


# ==================== LE PROMPT ====================

SYSTEM_DU_CONTROLE = """Tu es le contrôleur de fidélité de JuriX, un service d'information sur le droit camerounais. On te donne des EXTRAITS de textes officiels, numérotés [1], [2]…, puis les PHRASES d'un article écrit pour le grand public, chacune avec son identifiant (P1, P2…), groupées par partie (titre, contexte, résumé, points clés, impact, explications des citations, question). Pour CHAQUE phrase, tu dis si elle est fidèle aux extraits.

Tu ne juges QUE d'après les extraits : jamais d'après ta mémoire, ni d'après le droit d'un autre pays. Tu ne corriges rien. Un extrait dont l'en-tête dit « suite » commence au milieu d'un article.

Pour chaque phrase, remplis dans cet ordre :
1. "affirmation" : ce que la phrase affirme vraiment, en une phrase complète. Remplace les renvois (« ces cas », « des exceptions », « cela », « il ») par ce qu'ils désignent dans les phrases précédentes de la même partie. Une phrase affirme aussi ce qu'elle présuppose ; un titre ou une question aussi : « Expulsion sans jugement : quels cas ? » présuppose que la loi permet, dans certains cas, d'expulser sans jugement. Une phrase qui joint plusieurs affirmations (« et », « ; », « mais », « tandis que ») les donne toutes.
2. "nature" : "regle" si l'affirmation prête quelque chose au droit (un droit, une obligation, une interdiction, une condition, une exception, un délai, un montant, une procédure, qui décide, qui doit prouver), ou dit ce que les textes précisent ou non ; "autre" si elle ne prête rien au droit (un fait de la situation du lecteur, un conseil pratique comme consulter un avocat, une question qui ne présuppose aucune règle).
3. "extraits" : les numéros des extraits qui parlent de cette affirmation, pour l'appuyer ou pour la contredire ; [] s'il n'y en a aucun.
4. "passage" : les mots de ces extraits les plus proches de l'affirmation, COPIÉS MOT POUR MOT, sans rien changer, ajouter ni résumer : un ou deux morceaux, chacun entre guillemets droits, d'un seul tenant (5 à 40 mots, sans « ... » au milieu) et suivi du numéro de son extrait entre crochets, par exemple "Il appartient à l'administration de prouver l'infraction" [1] ; "" s'il n'y en a aucun. Le programme vérifie chaque morceau dans les extraits : un morceau inventé ou retouché fait tomber le verdict. Un extrait peut être abîmé par la numérisation (lettres isolées, mots coupés, comme « le preneur doit e o n u ss rnon dans le mois ») : ne répare jamais ces mots et n'en ajoute aucun à leur place ; copie les mots lisibles qui les entourent.
5. "raison" : 30 mots au plus, qui comparent les deux : « L'extrait 5 dit que … ; la phrase dit que … ». Si ta raison relève qu'un élément de la phrase manque aux extraits (« ne précise pas », « ne mentionne pas »), le verdict n'est pas "soutenue".
6. "verdict", en dernier : l'un de ces quatre mots, et aucun autre (une phrase en partie soutenue est "non_soutenue") :
- "soutenue" : le passage dit la même chose que l'affirmation, pour la même personne, le même cas, avec les mêmes conditions, les mêmes exceptions et les mêmes chiffres. Une reformulation fidèle est soutenue. Une phrase de nature "autre" est soutenue si elle ne présuppose rien que les extraits ne disent pas.
- "contredite" : le passage dit le contraire de l'affirmation, ou une chose incompatible : une autre personne doit prouver ; un juge est exigé là où la phrase s'en passe, ou l'inverse ; ce qui est permis est dit interdit, ou l'inverse ; un autre délai, un autre montant ; une condition ou une exception inversée.
- "non_soutenue" : aucun passage ne dit ce qu'affirme la phrase, sans la contredire : un ajout, une exception ou une procédure que les extraits ne donnent pas, ou une règle étendue à un autre cas que le sien (un manquement précis étendu à tout manquement ; un bail professionnel étendu au bail d'habitation ; un refus de renouveler présenté comme une expulsion).
- "hors_sources" : la phrase s'appuie sur une règle, un texte, une juridiction ou une procédure qu'aucun extrait ne mentionne.

Règles :
- Chaque élément de l'affirmation doit être appuyé. Si un élément est contredit, la phrase est "contredite", même si les autres sont justes ; si un élément n'a aucun appui, elle est au moins "non_soutenue".
- Une exception est une affirmation à part entière : « X est interdit, sauf en cas de Y » affirme aussi que X est permis en cas de Y, et « des exceptions existent » après « on ne peut pas X sans Z » affirme que, dans ces cas, on peut X sans Z. L'exception doit être appuyée comme le reste.
- Vérifie toujours : qui doit prouver ; qui décide (un juge, une juridiction, ou une partie seule) ; ce qui est permis ou interdit ; les délais, les montants, les plafonds et les planchers ; les conditions et les exceptions ; la personne et le cas visés.
- La preuve : si un passage la met à la charge de quelqu'un (« il appartient à B de prouver », « la charge de la preuve incombe à B »), une affirmation qui la met à la charge d'un autre est "contredite", même si elle ajoute que B doit aussi se justifier. Relis ton passage avant de juger : s'il nomme un autre que la phrase, elle est contredite.
- Si tu ne peux pas copier un passage qui dit ce que dit la phrase, elle n'est pas "soutenue".
- L'explication d'une citation se juge d'abord contre l'extrait qu'elle accompagne : elle ne doit rien lui faire dire de plus que ce qu'il dit.

Exemples, sur un extrait inventé pour l'exemple : [1] Loi sur les marchés — Article 10 : « (1) Le commerçant dont l'étal est saisi en obtient la restitution sur décision du juge, après paiement de l'amende. (2) Il appartient à l'administration de prouver l'infraction. »
- « Une fois l'amende payée, le commerçant peut reprendre lui-même son étal. » : l'affirmation est que le commerçant reprend son étal sans décision du juge ; passage "en obtient la restitution sur décision du juge, après paiement de l'amende" [1] ; verdict "contredite".
- « Le commerçant doit prouver qu'il n'a commis aucune infraction. » : passage "Il appartient à l'administration de prouver l'infraction" [1] ; verdict "contredite".
- « Le commerçant doit prouver sa bonne foi, et l'administration doit justifier la saisie. » : deux affirmations ; la première met la preuve à la charge du commerçant, et le passage "Il appartient à l'administration de prouver l'infraction" [1] la met à la charge de l'administration ; verdict "contredite".
- « Un étal saisi n'est rendu que sur décision du juge. Des exceptions existent toutefois en cas d'urgence. » : la seconde phrase affirme qu'en cas d'urgence l'étal est rendu sans décision du juge ; aucun passage ne le dit ; verdict "non_soutenue".
- « Les textes ne disent pas si l'étal peut être repris sans juge en dehors des cas d'urgence. » : nature "regle" ; elle tient pour acquis qu'en cas d'urgence l'étal se reprend sans juge, ce qu'aucun passage ne dit ; verdict "non_soutenue".
- « Renseignez-vous auprès de votre mairie. » : nature "autre" ; verdict "soutenue".

Réponds uniquement par l'objet JSON demandé : un objet par identifiant, dans l'ordre, sans en oublier ni en ajouter."""

# Schéma imposé (json_schema strict chez Mistral, response_schema chez
# Gemini). L'ordre compte : le modèle dit ce que la phrase affirme, copie le
# passage, compare, PUIS juge (Gemini range les champs par ordre
# alphabétique, qui garde aussi le verdict en dernier). Pas d'`enum` (les deux
# fournisseurs ne le lisent pas de la même façon, voir SCHEMA_DE_L_ARTICLE) :
# verdict et nature sont relus après coup, et un verdict illisible ne vaut
# jamais « soutenue ».
SCHEMA_DU_CONTROLE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "affirmation": {"type": "string"},
                    "nature": {"type": "string"},
                    "extraits": {"type": "array", "items": {"type": "integer"}},
                    "passage": {"type": "string"},
                    "raison": {"type": "string"},
                    "verdict": {"type": "string"},
                },
                "required": ["id", "affirmation", "nature", "extraits", "passage", "raison", "verdict"],
            },
        },
    },
    "required": ["phrases"],
}

_TITRES_DES_PARTIES = {
    "titre": "Titre de l'article", "contexte": "Contexte", "resume": "Résumé",
    "impact": "Impact pour le lecteur", "question": "Question suggérée au lecteur",
}


def _entete(extrait: ExtraitDuControle) -> str:
    entete = f"[{extrait.n}] {extrait.intitule} — Article {extrait.numero}"
    if extrait.suite:
        entete += (
            f" (suite : ce passage commence au milieu de l'article {extrait.numero}, "
            f"son début n'est pas fourni)"
        )
    return entete


def nom_d_article(citation: Dict[str, Any]) -> str:
    """« Article 39 », « Article 39 (suite) » : une citation telle que la carte la nomme."""
    numero = str(citation.get("article_number") or "?")
    return f"Article {numero}{' (suite)' if citation.get('suite') is True else ''}"


def _titre_du_groupe(
    phrase: PhraseAJuger, contenu: ContenuDuBillet, rattachement: Dict[int, int]
) -> str:
    if phrase.partie == "point_cle":
        return f"Point clé {phrase.index + 1}"
    if phrase.partie == "explication":
        citation = contenu.citations[phrase.index]
        n = rattachement.get(phrase.index)
        if n is not None:
            return (
                f"Explication de la citation {phrase.index + 1}, qui accompagne l'extrait {n} "
                f"({nom_d_article(citation)})"
            )
        return (
            f"Explication de la citation {phrase.index + 1} ({nom_d_article(citation)}, citation "
            f"non vérifiée : aucun extrait ne l'accompagne)"
        )
    return _TITRES_DES_PARTIES[phrase.partie]


def prompt_du_controle(
    extraits: Sequence[ExtraitDuControle],
    phrases: Sequence[PhraseAJuger],
    contenu: ContenuDuBillet,
    rattachement: Dict[int, int],
    autres: int = 0,
) -> str:
    """
    Les extraits numérotés, puis les phrases groupées par partie, chacune « P3 : … ».
    `autres` : les phrases de l'article jugées par les autres appels (lots_du_controle) ;
    un appel qui juge tout l'article garde le prompt du tour 4, mot pour mot.
    """
    blocs = "\n\n---\n\n".join(f"{_entete(e)}\n\n{e.texte}" for e in extraits)
    lignes: List[str] = []
    groupe: Optional[Tuple[str, int]] = None
    for phrase in phrases:
        if (phrase.partie, phrase.index) != groupe:
            groupe = (phrase.partie, phrase.index)
            lignes += ["", _titre_du_groupe(phrase, contenu, rattachement)]
        lignes.append(f"{phrase.id} : {phrase.texte}")
    a_juger = "\n".join(lignes)
    ailleurs = (
        f" ; les {autres} autres phrases de l'article sont jugées à part" if autres > 0 else ""
    )
    return (
        f"EXTRAITS ({len(extraits)}, numérotés ; les seuls textes contre lesquels juger) :\n\n"
        f"{blocs}\n\n"
        f"PHRASES À JUGER ({len(phrases)}, dans l'ordre de l'article{ailleurs}) :\n"
        f"{a_juger}\n\n"
        f"Rends un objet pour chacune des {len(phrases)} phrases, de {phrases[0].id} à "
        f"{phrases[-1].id}, sans en oublier ni en ajouter : affirmation (renvois résolus, "
        f"présupposés compris), nature, extraits, passage copié mot pour mot, raison, verdict."
    )


# ==================== LE PASSAGE COPIÉ ====================
#
# POURQUOI (revue du tour 4, data/blog/revue-api-tour4b). La vérification du
# passage copié laissait passer des mots inventés, puis les montrait comme le
# texte de l'extrait :
# - un morceau dont les DERNIERS mots avaient changé passait, pourvu que 85 %
#   de ses mots se suivent : « la charge de la preuve incombe au
#   travailleur » (18 mots sur 20) contre « … incombe à l'employeur » (Code du
#   travail, art. 40 al. 8). Or une phrase de loi finit souvent par ce qui
#   décide : qui prouve, qui décide, un chiffre ;
# - un tronçon de moins de 3 mots après « … » n'était jamais cherché : 11
#   verdicts « soutenue » de la mesure réelle du 04/11 reposaient sur « 2
#   ans », « prononce », « l'expulsion »…, et « une indemnité », écrit par le
#   modèle à la place d'un passage illisible de l'art. 36, passait ainsi ;
# - la raison citait les mots DU MODÈLE sous « Extrait N : « … » ».
# Puis (revue de l'API du tour 5, data/blog/tour5/revue-api) : tenu mot à mot,
# le passage ne l'était pas pour le SENS. Avec « … », le modèle recollait des
# mots vrais de l'extrait en la règle inverse, et le serveur citait le collage
# comme texte de l'extrait, que le front montrait en citation vérifiée :
# « l'acquéreur […] peut expulser le fermier ou le locataire » (Code civil,
# art. 1743 : « ne peut expulser »), « Ce motif […] peut être invoqué […] si
# les faits se sont poursuivis » (AUDCG, art. 127 : « ne peut être invoqué
# que si »), « Dans tous cas de licenciement, il appartient […] au
# travailleur » (Code du travail, art. 39 : « à l'employeur », et « au
# travailleur » pris 50 mots plus loin, à l'alinéa 4). Ce sont l'expulsion
# sans juge et la preuve inversée de l'essai réel. Trois causes : un tronçon
# court était accepté jusqu'à 80 mots après le précédent, dans une autre
# phrase ; rien ne regardait ce que cachait le trou ; les fenêtres, triées
# dans l'ordre de l'extrait, recollaient en une citation des tronçons cités
# dans l'autre ordre.
# Désormais :
# - un tronçon se trouve MOT POUR MOT (aux blancs, à la casse, aux accents et
#   à la ponctuation près). Sinon, TOLÉRÉ : ses premier et dernier mots
#   ouvrent et ferment une fenêtre de l'extrait où chacun de ses mots se
#   retrouve, dans l'ordre ; seul un mot outil peut y manquer (« de » pour
#   « en »), et le modèle peut avoir sauté quelques mots de l'extrait
#   (« trois mois » pour « trois (3) mois » ; « s'opposer au renouvellement »
#   pour « s'opposer au droit au renouvellement », mesure du 04/11), jamais un
#   mot qui décide (négation, exception, condition, alternative, modalité :
#   MOTS_DECISIFS) ;
# - un tronçon court (moins de TRONCON_COURT mots) se trouve mot pour mot,
#   juste après le tronçon précédent ET DANS SA PHRASE (ni « . », ni « ; »,
#   ni alinéa « (N) », ni retour à la ligne entre les deux : _fins_de_phrase) ;
#   s'il ouvre le morceau, le tronçon suivant le suit de même ;
# - une coupure ne cache jamais ce qui décide : entre deux tronçons, les mots
#   de l'extrait qu'aucun tronçon ne cite, dans la phrase de l'un ou de
#   l'autre, ne portent ni négation ni exception (_decide : « ne », « n' »,
#   « pas », « sans », « sauf », « à moins », le « que » d'un « ne … que »…).
#   Sinon le morceau n'appuie rien, et la raison dit ce que la coupure saute ;
# - la raison cite les mots DE L'EXTRAIT, une citation par phrase, dans
#   l'ordre du modèle : « […] » seulement entre des tronçons d'une même phrase
#   cités dans l'ordre de l'extrait ; cités dans l'autre ordre, la citation
#   donne les mots du trou (_citations_de_l_extrait). Deux phrases ne se
#   recollent jamais en une citation. Gardés : art. 133 (bail P12, deux
#   phrases citées à part) et art. 39 al. 4 (travail P14, le plancher et le
#   plafond, cités dans l'autre ordre).

# « […] », « [...] », « … », « ... » : le modèle saute une partie de l'extrait.
_ELLIPSE = re.compile(r"\[\s*(?:…|\.{2,})\s*\]|…|\.{3,}")
_MOT = re.compile(r"[a-z0-9]+")
_SUITE_DE_LETTRES = re.compile(r"[^\W_]+")
# Un passage d'au moins 4 mots copiés en tout, en tronçons d'au moins 3.
MOTS_DU_PASSAGE_AU_MOINS = 4
TRONCON_COURT = 3
# Mesure réelle du 04/11 (réponses de ministral-14b relues sans appel,
# data/blog/tour5/api) : un tronçon court venait 3 à 52 mots après le
# précédent (« La juridiction compétente… prononce… l'expulsion », art. 133).
# Il doit aussi rester dans la phrase du précédent (revue de l'API du tour 5) :
# la portée ne borne plus qu'une « phrase » que l'OCR a privée de ponctuation.
PORTEE_D_UNE_COUPURE = 80
# Les mots qu'une coupure ne cache jamais (revue de l'API du tour 5) : ceux
# qui retournent une règle. Avec eux, dans _decide : « n' » (élidé, pas le
# « n » de « n° »), « à moins », et « que », « qu' » après un « ne » de la même
# phrase (« ne peut être invoqué que si »).
MOTS_QU_UNE_COUPURE_NE_CACHE_PAS = frozenset({
    "ne", "pas", "sans", "sauf", "ni", "non", "aucun", "aucune", "aucuns", "aucunes", "jamais",
    "hormis", "excepte", "exceptee", "exceptes", "exceptees",
})
# Au plus tant de placements d'un morceau dans un extrait, et tant de pas pour
# les trouver : un morceau de mots courants (« de… la… le… ») ne fait pas
# exploser la recherche ; au-delà, il n'appuie rien (« À vérifier »).
PLACEMENTS_AU_PLUS = 64
PAS_DE_RECHERCHE_AU_PLUS = 4000
# La tolérance d'un tronçon de m mots : max(2, 15 % de m) mots de l'extrait
# sautés, max(1, 15 % de m) mots outils du modèle absents.
PART_TOLEREE = 0.15
# Les mots outils : seuls à pouvoir manquer à l'extrait dans un tronçon toléré
# (jamais en premier ni en dernier).
MOTS_OUTILS = frozenset({
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "a", "au", "aux", "en", "dans",
    "par", "pour", "sur", "chez", "entre", "vers", "ce", "cet", "cette", "ces", "se", "s", "son",
    "sa", "ses", "leur", "leurs", "il", "ils", "elle", "elles", "lui", "y", "on", "dont", "qui",
})
# Les mots qui décident du sens : jamais sautés dans l'extrait (« ne peut
# pas », « sans versement d'indemnité », « que si », « ou », « peut »,
# « contre son gré »…).
MOTS_DECISIFS = frozenset({
    "ne", "n", "pas", "sans", "sauf", "non", "ni", "nul", "nulle", "nuls", "nulles", "aucun",
    "aucune", "aucuns", "aucunes", "jamais", "rien", "hormis", "hors", "excepte", "exceptee",
    "exceptes", "exceptees", "seul", "seule", "seuls", "seules", "seulement", "uniquement", "que",
    "qu", "plus", "moins", "si", "ou", "et", "mais", "contre", "malgre", "avant", "apres",
    "pendant", "depuis", "jusqu", "jusque", "lorsque", "lorsqu", "quand", "toutefois",
    "cependant", "sous", "reserve", "peut", "peuvent", "doit", "doivent", "interdit",
    "interdite", "interdits", "interdites",
})


_SEPARATEURS = re.compile(r"[\W_]+")
# Les guillemets qui encadrent un morceau copié : droits ou typographiques ;
# « » seulement à défaut (les textes de loi en portent eux-mêmes).
_GUILLEMETS = re.compile(r'["“”]')
_GUILLEMETS_FRANCAIS = re.compile(r"[«»]")
# La source qu'un morceau se donne : « (Extrait 2) », « (Art. 39(1)) »,
# « [5] ». Ce n'est pas un mot copié.
_RENVOI_A_LA_SOURCE = re.compile(
    r"\(\s*(?:extraits?|art(?:icles?)?)\b[^()]*(?:\([^()]*\)[^()]*)*\)|\[\s*\d+\s*\]",
    re.IGNORECASE,
)


def _plat_ascii(texte: str) -> str:
    return unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii").lower()


def _mots(texte: Any) -> List[str]:
    """
    Les mots, sans accents ni casse ni ponctuation (« l'employeur » : l,
    employeur). La ponctuation devient une espace AVANT de retirer ce qui
    n'est pas ASCII : l'apostrophe typographique (’) collait sinon les mots.
    """
    return _MOT.findall(_plat_ascii(_SEPARATEURS.sub(" ", str(texte or ""))))


@dataclass(frozen=True)
class _TexteLu:
    """
    Un extrait prêt à chercher : ses mots (comme _mots), la place de chacun
    dans le texte, où est chaque mot, la phrase de chacun (son rang, compté
    par _fins_de_phrase), et s'il est élidé (« n' », « qu' », « l' »).
    """

    texte: str
    mots: Tuple[str, ...]
    places: Tuple[Tuple[int, int], ...]
    rangs: Mapping[str, Tuple[int, ...]]
    phrases: Tuple[int, ...]
    elides: Tuple[bool, ...]


_APOSTROPHES = frozenset("'’ʼ")
# Ce qui sépare deux phrases d'un extrait, pour qu'une coupure n'en joigne
# pas deux (revue de l'API du tour 5) : un retour à la ligne, « ; », « ! »,
# « ? », un point qui finit une phrase (comme decouper_en_phrases : pas celui
# de « art. 39 » ni de « M. »), ou un alinéa « (2) » devant une majuscule
# (l'OCR perd souvent le point d'avant : « … nlecat (2) La juridiction »).
# « trois (3) mois » n'en est pas un.
_SEPARATEUR_DE_PHRASES = re.compile(r"[\n;!?]")
_ALINEA = re.compile(r"\(\s*\d{1,3}\s*\)(?=\s*[A-ZÀ-ÖØ-Þ])")


def _fins_de_phrase(texte: str) -> List[int]:
    """Les places, dans le texte, de ce qui sépare deux phrases ; triées."""
    fins = [m.start() for m in _SEPARATEUR_DE_PHRASES.finditer(texte)]
    fins += [m.start() for m in _ALINEA.finditer(texte)]
    for fin in _FIN_DE_PHRASE.finditer(texte):
        mot = _DERNIER_MOT.search(texte[max(0, fin.start() - 40):fin.start()])
        if mot and (len(mot.group(1)) == 1 or mot.group(1).lower() in _ABREVIATIONS):
            continue
        fins.append(fin.start())
    return sorted(fins)


def _lire(texte: str) -> _TexteLu:
    """Les mêmes mots que _mots, chacun avec la place, dans le texte, de la suite de lettres qui le porte."""
    mots: List[str] = []
    places: List[Tuple[int, int]] = []
    elides: List[bool] = []
    for suite in _SUITE_DE_LETTRES.finditer(texte):
        apostrophe = texte[suite.end():suite.end() + 1] in _APOSTROPHES
        trouves = _MOT.findall(_plat_ascii(suite.group()))
        for rang, mot in enumerate(trouves):
            mots.append(mot)
            places.append((suite.start(), suite.end()))
            elides.append(apostrophe and rang == len(trouves) - 1)
    rangs: Dict[str, List[int]] = {}
    for rang, mot in enumerate(mots):
        rangs.setdefault(mot, []).append(rang)
    fins = _fins_de_phrase(texte)
    return _TexteLu(
        texte, tuple(mots), tuple(places), {m: tuple(r) for m, r in rangs.items()},
        tuple(bisect.bisect_right(fins, debut) for debut, _ in places), tuple(elides),
    )


def morceaux_copies(passage: Any) -> List[str]:
    """
    Les morceaux que le modèle dit avoir copiés : ceux qu'il met entre
    guillemets, s'il en met, sinon le passage entier ; sans les renvois à leur
    source. POURQUOI (mesure réelle du 04/11, data/blog/tour4) : le modèle
    copie volontiers deux extraits à la fois — « "Le bailleur peut, sans
    versement d'indemnité d'éviction, refuser le renouvellement…" (Extrait 2)
    et "Le locataire qui ne garnit pas la maison…" (Extrait 7) » — ; lu d'un
    bloc, ce passage vrai ne figurait dans aucun extrait, et 27 phrases sur 44
    passaient « À vérifier », dont 17 des 18 phrases justes.
    """
    texte = str(passage or "")
    for guillemets in (_GUILLEMETS, _GUILLEMETS_FRANCAIS):
        if guillemets.search(texte):
            # Entre une ouvrante et une fermante : les rangs impairs (le
            # dernier morceau d'un guillemet jamais fermé compris).
            morceaux = guillemets.split(texte)[1::2]
            break
    else:
        morceaux = [texte]
    propres = (" ".join(_RENVOI_A_LA_SOURCE.sub(" ", m).split()) for m in morceaux)
    return [m for m in propres if _mots(m)]


def _troncons(morceau: str) -> List[List[str]]:
    """Les mots d'un morceau, coupé à chaque « […] » : TOUS ses tronçons, les courts compris."""
    return [m for m in (_mots(s) for s in _ELLIPSE.split(morceau)) if m]


def _aligner(troncon: Sequence[str], lu: _TexteLu, debut: int) -> Optional[int]:
    """
    La fin de la fenêtre TOLÉRÉE qui porte le tronçon à partir du mot `debut`
    de l'extrait (le premier mot du tronçon), ou None. Voir plus haut.
    """
    mots = lu.mots
    sauts_permis = max(2, math.ceil(PART_TOLEREE * len(troncon)))
    oublis_permis = max(1, math.floor(PART_TOLEREE * len(troncon)))
    fin, sauts, oublis = debut, 0, 0
    for rang in range(1, len(troncon)):
        mot = troncon[rang]
        borne = min(len(mots), fin + 2 + sauts_permis - sauts)
        suivant = next((k for k in range(fin + 1, borne) if mots[k] == mot), None)
        if suivant is not None and not any(mots[k] in MOTS_DECISIFS for k in range(fin + 1, suivant)):
            sauts += suivant - fin - 1
            fin = suivant
        elif mot in MOTS_OUTILS and rang < len(troncon) - 1 and oublis < oublis_permis:
            oublis += 1
        else:
            return None
    return fin


def _fenetres(
    troncon: Sequence[str], lu: _TexteLu, depuis: int = 0, au_plus_tard: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Les fenêtres (premier mot, dernier mot) de l'extrait qui portent le
    tronçon, en commençant entre les mots `depuis` et `au_plus_tard` : mot pour
    mot d'abord ; sinon tolérées, pour un tronçon d'au moins TRONCON_COURT mots.
    """
    debuts = lu.rangs.get(troncon[0], ())
    debuts = debuts[bisect.bisect_left(debuts, depuis):]
    if au_plus_tard is not None:
        debuts = debuts[:bisect.bisect_right(debuts, au_plus_tard)]
    taille = len(troncon)
    exactes = [(a, a + taille - 1) for a in debuts if lu.mots[a:a + taille] == tuple(troncon)]
    if exactes or taille < TRONCON_COURT:
        return exactes
    tolerees = ((a, _aligner(troncon, lu, a)) for a in debuts)
    return [(a, fin) for a, fin in tolerees if fin is not None]


Fenetre = Tuple[int, int]


def _placements(troncons: Sequence[Sequence[str]], lu: _TexteLu) -> Iterator[List[Fenetre]]:
    """
    Les placements des tronçons d'un morceau dans cet extrait, une fenêtre
    par tronçon, du plus proche de la lecture du modèle au plus lointain. Un
    tronçon court se cherche juste après le précédent, DANS SA PHRASE, et le
    tronçon qui suit un tronçon court de même. Un tronçon long se cherche après
    le précédent, puis ailleurs dans l'extrait : le modèle a pu citer avant ce
    qui vient après (mesure réelle du 04/11, travail P14 : le plancher des
    dommages-intérêts, puis leur plafond, que l'art. 39 donne dans l'autre
    ordre). Chaque occurrence d'un tronçon long est essayée (bail P12 : la
    « juridiction compétente » de l'alinéa 3 ne « prononce » rien, celle de
    l'alinéa 4 si), au plus PAS_DE_RECHERCHE_AU_PLUS pas.
    """
    pas = [0]

    def suite(rang: int, avant: Optional[Tuple[Fenetre, bool]]) -> Iterator[List[Fenetre]]:
        if rang == len(troncons):
            yield []
            return
        pas[0] += 1
        if pas[0] > PAS_DE_RECHERCHE_AU_PLUS:
            return
        troncon = troncons[rang]
        court = len(troncon) < TRONCON_COURT
        if avant is None:
            candidates = _fenetres(troncon, lu)
        elif court or avant[1]:
            fin = avant[0][1]
            candidates = [
                fenetre for fenetre in _fenetres(troncon, lu, fin + 1, fin + PORTEE_D_UNE_COUPURE)
                if lu.phrases[fenetre[1]] == lu.phrases[fin]
            ][:1]
        else:
            apres = _fenetres(troncon, lu, avant[0][1] + 1)
            candidates = apres + [fenetre for fenetre in _fenetres(troncon, lu) if fenetre not in apres]
        for fenetre in candidates:
            for reste in suite(rang + 1, (fenetre, court)):
                yield [fenetre, *reste]

    yield from suite(0, None)


def _decide(lu: _TexteLu, k: int) -> bool:
    """Le mot `k` de l'extrait retourne-t-il la règle qu'on lirait sans lui (MOTS_QU_UNE_COUPURE_NE_CACHE_PAS) ?"""
    mot = lu.mots[k]
    if mot in MOTS_QU_UNE_COUPURE_NE_CACHE_PAS or (mot == "n" and lu.elides[k]):
        return True
    if mot == "moins":
        return k > 0 and lu.mots[k - 1] == "a" and lu.phrases[k - 1] == lu.phrases[k]
    if mot == "que" or (mot == "qu" and lu.elides[k]):
        debut = k
        while debut > 0 and lu.phrases[debut - 1] == lu.phrases[k]:
            debut -= 1
        return any(lu.mots[j] == "ne" or (lu.mots[j] == "n" and lu.elides[j]) for j in range(debut, k))
    return False


def _mots_caches(lu: _TexteLu, fenetres: Sequence[Fenetre]) -> List[int]:
    """
    Les mots de l'extrait qui décident (_decide) et qu'une coupure cache :
    entre deux fenêtres consécutives (dans l'ordre du modèle), ceux qu'aucune
    fenêtre ne couvre, dans la phrase de l'une ou de l'autre. Une phrase
    entière sautée entre deux phrases citées ne compte pas : elles sont alors
    citées à part (_citations_de_l_extrait), sans prétendre la relier (bail
    P12 : « clauses […] non respectées », à l'alinéa sauté de l'art. 133).
    """
    couverts = {k for debut, fin in fenetres for k in range(debut, fin + 1)}
    caches: List[int] = []
    for une, autre in zip(fenetres, fenetres[1:]):
        gauche, droite = sorted((une, autre))
        if droite[0] <= gauche[1] + 1:
            continue
        bords = (lu.phrases[gauche[1]], lu.phrases[droite[0]])
        caches += [
            k for k in range(gauche[1] + 1, droite[0])
            if k not in couverts and k not in caches and lu.phrases[k] in bords and _decide(lu, k)
        ]
    return sorted(caches)


def _meme_phrase(lu: _TexteLu, une: Fenetre, autre: Fenetre) -> bool:
    return max(lu.phrases[une[0]], lu.phrases[autre[0]]) <= min(lu.phrases[une[1]], lu.phrases[autre[1]])


def _suites_de_phrase(lu: _TexteLu, fenetres: Sequence[Fenetre]) -> List[List[Fenetre]]:
    """Les fenêtres, dans l'ordre du modèle, groupées tant qu'elles restent dans la même phrase."""
    suites: List[List[Fenetre]] = []
    for fenetre in fenetres:
        if suites and _meme_phrase(lu, suites[-1][-1], fenetre):
            suites[-1].append(fenetre)
        else:
            suites.append([fenetre])
    return suites


def _placer(troncons: Sequence[Sequence[str]], lu: _TexteLu) -> Tuple[Optional[List[Fenetre]], List[int]]:
    """
    (fenêtres, []) : le placement des tronçons d'un morceau dans cet extrait
    dont aucune coupure ne cache un mot qui décide, le plus lisible (le moins
    de citations) parmi les PLACEMENTS_AU_PLUS premiers. (None, mots cachés) :
    chaque placement trouvé en cache (ceux du premier). (None, []) : un
    tronçon n'y est pas, ou pas où il doit être.
    """
    meilleur: Optional[Tuple[int, List[Fenetre]]] = None
    caches: List[int] = []
    for rang, fenetres in enumerate(_placements(troncons, lu)):
        if rang >= PLACEMENTS_AU_PLUS:
            break
        trouves = _mots_caches(lu, fenetres)
        if trouves:
            caches = caches or trouves
            continue
        citations = len(_suites_de_phrase(lu, fenetres))
        if meilleur is None or citations < meilleur[0]:
            meilleur = (citations, fenetres)
            if citations == 1:
                break
    return (meilleur[1], []) if meilleur else (None, caches)


def _texte_de_l_extrait(lu: _TexteLu, debut: int, fin: int) -> str:
    """
    Le texte de l'extrait du mot `debut` au mot `fin`, ses blancs ramenés à
    une espace. Ses « » deviennent “ ” : le front ferme une citation au » qui
    équilibre son « (controle.ts).
    """
    de, a = lu.places[debut][0], lu.places[fin][1]
    # « deux (2) ans » : une fenêtre qui commence ou finit au chiffre garde sa
    # parenthèse, pas la moitié.
    avant, apres = lu.texte[:de].rstrip(), lu.texte[a:].lstrip()
    if lu.texte[de:a].count(")") > lu.texte[de:a].count("(") and avant.endswith("("):
        de = len(avant) - 1
    if lu.texte[de:a].count("(") > lu.texte[de:a].count(")") and apres.startswith(")"):
        a = len(lu.texte) - len(apres) + 1
    return _GUILLEMET_FERMANT.sub("”", _GUILLEMET_OUVRANT.sub("“", " ".join(lu.texte[de:a].split())))


def _citations_de_l_extrait(lu: _TexteLu, fenetres: Sequence[Fenetre]) -> List[str]:
    """
    Le texte de l'extrait que couvrent les fenêtres d'un morceau, en
    citations : une par suite de fenêtres de la même phrase (_suites_de_phrase),
    dans l'ordre du modèle. Dans une citation, « […] » entre deux fenêtres qui
    ne se touchent pas, si le modèle les a citées dans l'ordre de l'extrait et
    que le trou ne cache rien qui décide ; sinon, les mots du trou, d'un seul
    tenant (travail P14 : le plafond, puis le plancher, rendus dans la phrase
    de l'art. 39).
    """
    citations = []
    for suite in _suites_de_phrase(lu, fenetres):
        jointes: List[List[int]] = []
        for debut, fin in sorted(suite):
            if jointes and debut <= jointes[-1][1] + 1:
                jointes[-1][1] = max(jointes[-1][1], fin)
            else:
                jointes.append([debut, fin])
        bornes = [(debut, fin) for debut, fin in jointes]
        if list(suite) == sorted(suite) and not _mots_caches(lu, bornes):
            citations.append(" […] ".join(_texte_de_l_extrait(lu, debut, fin) for debut, fin in bornes))
        else:
            citations.append(_texte_de_l_extrait(lu, bornes[0][0], max(fin for _, fin in bornes)))
    return citations


def _mots_ecrits(lu: _TexteLu, rangs: Sequence[int]) -> Tuple[str, ...]:
    """Les mots de l'extrait, tels qu'il les écrit (« n’ », « à moins »), sans doublon."""
    ecrits: List[str] = []
    for k in rangs:
        debut, fin = lu.places[k]
        if lu.mots[k] == "moins" and k > 0:
            debut = lu.places[k - 1][0]
        ecrit = lu.texte[debut:fin + 1] if lu.elides[k] else lu.texte[debut:fin]
        if ecrit not in ecrits:
            ecrits.append(ecrit)
    return tuple(ecrits)


_GUILLEMET_OUVRANT = re.compile(r"«\s*")
_GUILLEMET_FERMANT = re.compile(r"\s*»")


@dataclass(frozen=True)
class LectureDuPassage:
    """
    Ce que lire_le_passage_en_detail trouve d'un passage copié : `appuis` et
    `introuvable` (voir lire_le_passage), et `coupure` : (l'extrait, les mots
    que la coupure saute, tels qu'il les écrit) quand le premier morceau
    refusé l'est parce que chacun de ses placements cache un mot qui décide ;
    None sinon.
    """

    appuis: Optional[List[Tuple[int, str]]]
    introuvable: str
    coupure: Optional[Tuple[int, Tuple[str, ...]]] = None


def lire_le_passage_en_detail(
    passage: Any, numeros: Sequence[int], textes: Mapping[int, str]
) -> LectureDuPassage:
    """
    Voir lire_le_passage ; et, d'un morceau dont la coupure saute une
    négation ou une exception, ce qu'elle saute, et où (revue de l'API du
    tour 5 : la raison le dit, la relecture sait quoi lire dans l'extrait).
    """
    ordre = [*(n for n in numeros if n in textes), *(n for n in textes if n not in numeros)]
    lus = {n: _lire(str(textes[n] or "")) for n in ordre}
    appuis: List[Tuple[int, str]] = []
    copies = 0
    for morceau in morceaux_copies(passage):
        troncons = _troncons(morceau)
        trouve: Optional[Tuple[int, List[Fenetre]]] = None
        coupure: Optional[Tuple[int, Tuple[str, ...]]] = None
        for n in ordre:
            fenetres, caches = _placer(troncons, lus[n])
            if fenetres:
                trouve = (n, fenetres)
                break
            if caches and coupure is None:
                coupure = (n, _mots_ecrits(lus[n], caches))
        if trouve is None:
            ecrits = [s.strip() for s in _ELLIPSE.split(morceau) if _mots(s)]
            nulle_part = [
                ecrit for ecrit in ecrits if not any(_fenetres(_mots(ecrit), lu) for lu in lus.values())
            ]
            return LectureDuPassage(None, " … ".join(nulle_part) or morceau, None if nulle_part else coupure)
        # Un morceau de deux phrases, ou cité dans l'autre ordre, donne deux
        # citations du même extrait (« Extrait 6 : « … » « … » »).
        n, fenetres = trouve
        appuis += [(n, citation) for citation in _citations_de_l_extrait(lus[n], fenetres)]
        # Un tronçon court est vérifié, mais ne prouve rien seul (« la
        # preuve », « du motif ») : il ne compte pas dans le minimum.
        copies += sum(len(t) for t in troncons if len(t) >= TRONCON_COURT)
    return LectureDuPassage(appuis if copies >= MOTS_DU_PASSAGE_AU_MOINS else None, "")


def lire_le_passage(
    passage: Any, numeros: Sequence[int], textes: Mapping[int, str]
) -> Tuple[Optional[List[Tuple[int, str]]], str]:
    """
    (appuis, introuvable). `appuis` : pour chaque morceau copié
    (morceaux_copies), l'extrait où il figure, cherché d'abord parmi ceux que
    cite le modèle, et les mots DE CET EXTRAIT qui le portent, une citation
    par phrase : [(numéro, texte)…]. None si UN morceau ne figure dans aucun
    extrait (un appui inventé ou retouché fait tomber tout le passage), si sa
    coupure saute une négation ou une exception de l'extrait, ou s'il y a
    moins de MOTS_DU_PASSAGE_AU_MOINS mots copiés en tout. Un morceau peut
    venir d'un autre extrait que le précédent ; ses tronçons (« […] »), eux,
    d'un seul.

    `introuvable`, tel que le modèle l'a écrit : les tronçons du premier
    morceau introuvable qu'aucun extrait ne porte (« pour motif grave »), ou ce
    morceau entier si chacun est quelque part mais pas ensemble ; "" sinon.
    """
    lecture = lire_le_passage_en_detail(passage, numeros, textes)
    return lecture.appuis, lecture.introuvable


def appuis_du_passage(
    passage: Any, numeros: Sequence[int], textes: Mapping[int, str]
) -> Optional[List[Tuple[int, str]]]:
    """Les appuis du passage (lire_le_passage) : [(numéro, mots de l'extrait)…], ou None."""
    return lire_le_passage(passage, numeros, textes)[0]


def passage_present(passage: Any, texte: Any) -> bool:
    """
    Le passage copié par le modèle figure-t-il dans le texte de l'extrait ?
    Mot pour mot, aux blancs, à la casse, aux accents et à la ponctuation près,
    ou toléré (voir plus haut) ; « […] » sépare des tronçons, chacun cherché ;
    entre guillemets, chaque morceau doit y être.
    """
    return appuis_du_passage(passage, [0], {0: str(texte or "")}) is not None


def extrait_du_passage(
    passage: Any, numeros: Sequence[int], textes: Mapping[int, str]
) -> Optional[int]:
    """Le numéro de l'extrait où figure (le premier morceau du) passage : ceux que cite le modèle d'abord."""
    appuis = appuis_du_passage(passage, numeros, textes)
    return appuis[0][0] if appuis else None


# ==================== LA RÉPONSE DU MODÈLE ====================

_VERDICTS_LUS = {
    "soutenue": "soutenue", "soutenu": "soutenue", "supported": "soutenue",
    "non_soutenue": "non_soutenue", "non_soutenu": "non_soutenue", "nonsoutenue": "non_soutenue",
    "pas_soutenue": "non_soutenue", "unsupported": "non_soutenue", "not_supported": "non_soutenue",
    "contredite": "contredite", "contredit": "contredite", "contradicted": "contredite",
    "hors_sources": "hors_sources", "hors_source": "hors_sources", "horssources": "hors_sources",
    "out_of_sources": "hors_sources",
}
_NATURES_AUTRES = frozenset({"autre", "other", "fait", "faits", "conseil", "question", "situation"})
# Une phrase que le modèle n'a pas jugée (absente, ou verdict illisible) :
# « non soutenue », à relire. Jamais un accord par défaut.
RAISON_NON_JUGEE = (
    "Le contrôle n'a pas rendu de verdict lisible pour cette phrase : relisez-la contre les "
    "extraits."
)
_SANS_PASSAGE = {
    "soutenue": (
        "À vérifier : le contrôle la disait soutenue, sans pouvoir copier un passage des "
        "extraits qui le dise"
    ),
    "contredite": (
        "À vérifier : le contrôle la disait contredite, sans pouvoir copier le passage des "
        "extraits qui la contredit"
    ),
}
# Le morceau introuvable, tel que le modèle l'a écrit, nommé comme tel (jamais
# comme le texte d'un extrait) : la relecture sait quoi chercher.
CARACTERES_DU_MORCEAU_INTROUVABLE = 160


def _sans_passage(
    verdict: str, introuvable: str, raison: str,
    coupure: Optional[Tuple[int, Tuple[str, ...]]] = None,
) -> str:
    """
    La raison d'une phrase rabattue faute de passage réel (tour 5 : le morceau
    introuvable est nommé ; revue du tour 4 : sur un extrait abîmé par la
    numérisation, l'administrateur ne savait pas ce qui manquait). Une coupure
    qui saute une négation ou une exception le dit, avec l'extrait où lire ces
    mots (revue de l'API du tour 5) ; jamais sous la forme « Extrait N : « … » »
    des passages vérifiés.
    """
    morceau = couper(introuvable, CARACTERES_DU_MORCEAU_INTROUVABLE)
    if coupure is not None:
        n, mots = coupure
        sautes = " », « ".join(mots)
        precision = (
            f" (sa coupure « … » saute « {sautes} » dans l'extrait {n}, entre les mots copiés "
            f"« {morceau} »)"
        )
    elif introuvable:
        precision = f" (introuvable tel quel dans les extraits : « {morceau} »)"
    else:
        precision = ""
    return f"{_SANS_PASSAGE[verdict]}{precision}. Sa raison : « {raison or '—'} »"


# Un renvoi du modèle à des extraits : « extrait 5 », « l'extrait n° 36 »,
# « les extraits 7 et 37 », « [36] ».
_RENVOI_A_DES_EXTRAITS = re.compile(
    r"(?P<nom>\b[Ee]xtraits?\s+(?:n[°o]s?\.?\s*)?)(?P<numeros>\d+(?:\s*(?:,|et|ou|à)\s*\d+)*)\b"
    r"|\[(?P<seul>\d+)\]"
)
_NUMERO = re.compile(r"\d+")


def renvois_aux_extraits(raison: str, extraits: Sequence["ExtraitDuControle"]) -> str:
    """
    La raison du modèle, chaque renvoi nommant un extrait qui existe, et sans
    la forme d'un passage cité par le serveur (_FORME_D_UN_PASSAGE_CITE).

    POURQUOI (revue du tour 4). Le modèle confondait numéros d'extraits et
    numéros d'articles : « L'extrait 36 confirme… », « Les extraits 7 et 37 »,
    quand le panneau n'avait que les extraits 1 à 15. Un numéro qui n'est pas
    celui d'un extrait, mais celui de l'ARTICLE d'un extrait, est rendu à cet
    extrait (« extrait 4 (article 36) ») ; sinon, il est dit inconnu.
    """
    valides = {e.n for e in extraits}
    porteurs: Dict[str, List[int]] = {}
    for extrait in extraits:
        porteurs.setdefault(normalize_number(str(extrait.numero or "")), []).append(extrait.n)

    def nommer(numero: int) -> Tuple[str, str]:
        """(le ou les extraits, la précision) d'un numéro qui n'est pas celui d'un extrait."""
        trouves = sorted(set(porteurs.get(normalize_number(str(numero)), [])))
        if trouves:
            return " ou ".join(str(n) for n in trouves), f" (article {numero})"
        return str(numero), " (aucun extrait n'a ce numéro)"

    def corriger(trouve: "re.Match") -> str:
        if trouve.group("seul") is not None:
            numero = int(trouve.group("seul"))
            if numero in valides:
                return trouve.group(0)
            nom, precision = nommer(numero)
            return f"[{nom}]{precision}"
        numeros = trouve.group("numeros")
        if all(int(n) in valides for n in _NUMERO.findall(numeros)):
            return trouve.group(0)
        return trouve.group("nom") + _NUMERO.sub(
            lambda n: n.group() if int(n.group()) in valides else "".join(nommer(int(n.group()))),
            numeros,
        )

    return _FORME_D_UN_PASSAGE_CITE.sub(r"\1,\2", _RENVOI_A_DES_EXTRAITS.sub(corriger, raison))


# « Extrait 5 : « … » » est la forme des passages que le SERVEUR cite à la fin
# d'une raison, et que le front montre comme texte vérifié (controle.ts,
# raisonLue). Écrite par le modèle dans sa propre raison, elle devient
# « Extrait 5, « … » » : ses mots ne passent jamais pour ceux d'un extrait.
_FORME_D_UN_PASSAGE_CITE = re.compile(r"(\b[Ee]xtraits?\s+\d+)\s*:(\s*«)")


def _plat(brut: Any) -> str:
    texte = unicodedata.normalize("NFKD", str(brut or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]+", "_", texte.lower()).strip("_")


_EN_PARTIE = re.compile(r"(?:^|_)(?:partiel\w*|en_partie|partially|partly)(?:_|$)")


def lire_le_verdict(brut: Any) -> Optional[str]:
    """
    « non soutenue », « Non-soutenue », « hors sources »… : le verdict du
    contrat, ou None. « Partiellement soutenue » (mesure réelle du 04/11 : 2
    phrases sur 44) n'est pas soutenue : « non_soutenue » ; « partiellement
    contredite », « contredite ». Jamais « soutenue ».
    """
    if not isinstance(brut, str):
        return None
    plat = _plat(brut)
    if plat in _VERDICTS_LUS:
        return _VERDICTS_LUS[plat]
    if _EN_PARTIE.search(plat):
        if "contredit" in plat or "contradict" in plat:
            return "contredite"
        if "soutenu" in plat or "support" in plat:
            return "non_soutenue"
    return None


def lire_la_nature(brut: Any) -> str:
    """« regle » ou « autre » ; dans le doute, « regle » (le passage est alors exigé)."""
    return "autre" if _plat(brut) in _NATURES_AUTRES else "regle"


def _numeros(valeur: Any) -> List[int]:
    if not isinstance(valeur, list):
        valeur = [valeur] if valeur is not None else []
    numeros = []
    for element in valeur:
        n = _entier(element)
        if n is not None and n not in numeros:
            numeros.append(n)
    return numeros


def _texte_du_modele(brut: Any) -> str:
    return " ".join(texte_brut(str(brut or "")).split())


def _passages_cites(appuis: Sequence[Tuple[int, str]]) -> str:
    """
    « Extrait 5 : « … » « … » ; extrait 10 : « … » » : pour chaque morceau
    vérifié, les mots DE SON EXTRAIT (lire_le_passage), jamais ceux du modèle.
    """
    groupes: List[Tuple[int, List[str]]] = []
    for n, morceau in appuis:
        if groupes and groupes[-1][0] == n:
            groupes[-1][1].append(morceau)
        else:
            groupes.append((n, [morceau]))
    return " ; ".join(
        f"{'Extrait' if rang == 0 else 'extrait'} {n} : "
        + " ".join(f"« {couper(m, CARACTERES_DU_PASSAGE_CITE)} »" for m in morceaux)
        for rang, (n, morceaux) in enumerate(groupes)
    )


def lire_les_verdicts(
    reponse: Dict[str, Any], phrases: Sequence[PhraseAJuger], extraits: Sequence[ExtraitDuControle]
) -> List[Dict[str, Any]]:
    """
    Les phrases jugées, dans l'ordre de l'article, chacune avec son verdict.

    Une réponse coupée, illisible, ou sans un seul verdict lisible est une
    erreur (502) : le contrôle n'a pas eu lieu. Une phrase oubliée, ou au
    verdict illisible, est « non_soutenue » (RAISON_NON_JUGEE) ; un numéro
    d'extrait qui n'existe pas est retiré ; un identifiant inconnu, ignoré.

    LE PASSAGE COPIÉ EST VÉRIFIÉ. Une phrase « soutenue » (sauf de nature
    « autre ») ou « contredite » doit venir avec un passage qui figure dans un
    extrait, sans qu'une coupure « … » y saute une négation ou une exception ;
    sinon elle devient « non_soutenue », à relire, et sa raison nomme le
    morceau introuvable, ou ce que sa coupure saute : un appui inventé ou
    recollé ne rend pas une phrase soutenue, et une contradiction qu'aucun
    texte ne montre ne bloque pas la publication. Le passage trouvé est ajouté
    à la raison, dans les mots de l'extrait, une citation par phrase, pour la
    relecture. Les renvois de la raison à des extraits qui n'existent pas sont
    corrigés (renvois_aux_extraits).
    """
    if reponse.get("tronquee"):
        raise ReponseInvalideError(
            "La réponse du contrôle IA a été coupée avant la fin : l'article est inchangé. "
            "Relancez le contrôle."
        )
    brut = str(reponse.get("response") or "").strip()
    if brut.startswith("```"):
        brut = re.sub(r"^```(?:json)?\s*|\s*```$", "", brut)
    try:
        donnees = json.loads(brut)
    except (json.JSONDecodeError, TypeError):
        donnees = None
    jugees = donnees.get("phrases") if isinstance(donnees, dict) else None
    if not isinstance(jugees, list):
        raise ReponseInvalideError(
            "Le contrôle IA n'a pas rendu une réponse lisible : l'article est inchangé. Relancez "
            "le contrôle."
        )
    textes = {e.n: e.texte for e in extraits}
    par_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for jugee in jugees:
        if not isinstance(jugee, dict):
            continue
        identifiant = str(jugee.get("id") or "").strip().upper()
        verdict = lire_le_verdict(jugee.get("verdict"))
        if identifiant and verdict is not None and identifiant not in par_id:
            par_id[identifiant] = (verdict, jugee)
    if not any(phrase.id.upper() in par_id for phrase in phrases):
        raise ReponseInvalideError(
            "Le contrôle IA n'a jugé aucune phrase de l'article : l'article est inchangé. "
            "Relancez le contrôle."
        )
    resultat = []
    for phrase in phrases:
        verdict, jugee = par_id.get(phrase.id.upper(), ("non_soutenue", None))
        jugement: Dict[str, Any] = {
            "id": phrase.id, "partie": phrase.partie, "index": phrase.index, "texte": phrase.texte,
        }
        if jugee is None:
            jugement.update(verdict="non_soutenue", extraits=[], raison=RAISON_NON_JUGEE)
            resultat.append(jugement)
            continue
        numeros = [n for n in _numeros(jugee.get("extraits")) if n in textes]
        nature = lire_la_nature(jugee.get("nature"))
        passage = _texte_du_modele(jugee.get("passage"))
        raison = renvois_aux_extraits(_texte_du_modele(jugee.get("raison")), extraits)
        if verdict == "contredite" or (verdict == "soutenue" and nature == "regle"):
            lecture = lire_le_passage_en_detail(passage, numeros, textes)
            if lecture.appuis is None:
                raison = _sans_passage(verdict, lecture.introuvable, raison, lecture.coupure)
                verdict = "non_soutenue"
            else:
                trouves = list(dict.fromkeys(n for n, _ in lecture.appuis))
                numeros = [*trouves, *(n for n in numeros if n not in trouves)]
                cites = _passages_cites(lecture.appuis)
                raison = f"{raison} — {cites}" if raison else cites
        jugement.update(
            verdict=verdict, extraits=numeros, raison=couper(raison, CARACTERES_DE_LA_RAISON),
            # Gardés pour comprendre un verdict ; pas dans la réponse de l'API.
            affirmation=couper(_texte_du_modele(jugee.get("affirmation")), CARACTERES_DE_LA_RAISON),
            nature=nature, passage=couper(passage, CARACTERES_DE_LA_RAISON),
        )
        resultat.append(jugement)
    return resultat


# ==================== LE CONTRÔLE ====================

SANS_SOURCES = (
    "L'article n'a pas de sources enregistrées (article d'avant le contrôle IA, ou écrit à la "
    "main) : rien contre quoi juger ses phrases. Reconstruisez d'abord ses sources, citez des "
    "articles parmi elles, puis relancez le contrôle."
)

_NOMS_DES_GROUPES = {
    "titre": "le titre", "contexte": "le contexte", "resume": "le résumé",
    "impact": "l'impact pour le citoyen", "question": "la question suggérée",
}


def _nom_du_groupe(phrase: PhraseAJuger) -> str:
    if phrase.partie == "point_cle":
        return f"le point clé {phrase.index + 1}"
    if phrase.partie == "explication":
        return f"l'explication de la citation {phrase.index + 1}"
    return _NOMS_DES_GROUPES.get(phrase.partie, phrase.partie)


def lots_du_controle(
    phrases: Sequence[PhraseAJuger], *, un_seul_appel: bool = False,
) -> List[List[PhraseAJuger]]:
    """
    Les phrases de l'article en lots d'au plus PHRASES_PAR_APPEL : un appel au
    modèle par lot, l'un après l'autre (voir PHRASES_PAR_APPEL). Un lot ne
    coupe jamais un groupe — le titre, le contexte, le résumé, un point clé,
    l'impact, une explication, la question — : le modèle résout les renvois
    (« ces cas », « il ») dans la même partie. Le moins de lots possible
    (chacun reprend tous les extraits, donc des jetons d'entrée), puis les
    plus égaux possible (chacun loin de CONTROLE_MAX_TOKENS). Trois au plus
    sous PHRASES_AU_PLUS (voir ce réglage).

    `un_seul_appel` (tour 7, un modèle à budget de jetons) : tout l'article en
    UN lot, jusqu'à PHRASES_AU_PLUS. POURQUOI : chaque appel de mistral-large-4
    paie sa réflexion (35 à 50 s), quelle que soit sa taille ; deux lots
    l'ont payée deux fois, jusqu'au 504 de l'essai réel du 04/11.

    Raises:
        TexteTropLongError (422) : aucune phrase ; plus de PHRASES_AU_PLUS ; un
        groupe de plus de PHRASES_PAR_APPEL (sauf en un seul appel).
    """
    if not phrases:
        raise TexteTropLongError("L'article n'a aucune phrase à contrôler : écrivez-le d'abord.")
    if len(phrases) > PHRASES_AU_PLUS:
        raise TexteTropLongError(
            f"L'article compte {len(phrases)} phrases, {PHRASES_AU_PLUS} au plus pour le "
            f"contrôle IA : raccourcissez-le, puis relancez le contrôle."
        )
    if un_seul_appel:
        return [list(phrases)]
    groupes: List[List[PhraseAJuger]] = []
    for phrase in phrases:
        if groupes and (groupes[-1][0].partie, groupes[-1][0].index) == (phrase.partie, phrase.index):
            groupes[-1].append(phrase)
        else:
            groupes.append([phrase])
    for groupe in groupes:
        if len(groupe) > PHRASES_PAR_APPEL:
            nom = _nom_du_groupe(groupe[0])
            raise TexteTropLongError(
                f"{nom[0].upper()}{nom[1:]} compte {len(groupe)} phrases, {PHRASES_PAR_APPEL} au "
                f"plus par partie pour le contrôle IA : raccourcissez cette partie, puis relancez "
                f"le contrôle."
            )

    def ranger(limite: int) -> List[List[PhraseAJuger]]:
        lots: List[List[PhraseAJuger]] = []
        for groupe in groupes:
            if lots and len(lots[-1]) + len(groupe) <= limite:
                lots[-1].extend(groupe)
            else:
                lots.append(list(groupe))
        return lots

    lots = ranger(PHRASES_PAR_APPEL)
    # Autant de lots, les plus égaux possible : la plus petite taille qui suffit.
    for limite in range(math.ceil(len(phrases) / len(lots)), PHRASES_PAR_APPEL):
        egaux = ranger(limite)
        if len(egaux) <= len(lots):
            return egaux
    return lots


# ==================== LE PLAN DU CONTRÔLE (REVUE DU TOUR 6) ====================
#
# POURQUOI (revue du tour 6, data/blog/tour6/revue-large4). Les lots partaient
# l'un après l'autre, chacun avec les 20 s d'attente du CHAT
# (MISTRAL_ATTENTE_MAX_S) pour que le budget de jetons du modèle (20 000 par
# minute) se refasse. Dès que le coût réel du lot 1 et l'entrée du lot 2
# dépassaient 20 000 (l'article réel du travail avec 18 563 caractères
# d'extraits, ou avec ses 10 052 si le fournisseur annonçait 0 jeton restant,
# comme le 04/11), le lot 2 devait attendre ~33 s : 503 APRÈS le lot 1 payé,
# et la relance au Retry-After refaisait exactement de même. L'article n'était
# jamais contrôlé, donc jamais publiable. Et rien ne comptait ENSEMBLE
# l'attente d'un autre contrôle, celles du budget et la durée des lots : un
# contrôle lancé pendant la fin d'un autre envoyait ses deux lots, puis
# l'échéance de la route le coupait (504, deux lots payés, rien d'enregistré).
# Désormais le contrôle se PLANIFIE, une fois le verrou du modèle pris et AVANT
# son premier lot : pour chaque lot, ses jetons d'entrée (estimés comme le
# client les réserve), sa sortie et sa durée au pire mesuré ; les attentes du
# budget de jetons en découlent, calculées par le limiteur lui-même sur une
# copie de son état (Limiteur.copie : ce qu'il compte déjà, recalage du
# fournisseur compris). Un plan qui finit après l'échéance de la route, moins
# MARGE_DU_PLAN_S, est refusé avant tout envoi : 503, Retry-After = le premier
# instant où il tiendrait. Sinon chaque lot reçoit l'attente que le plan lui
# laisse (attente_max du client), et le plan est refait sur l'état réel avant
# chaque lot suivant : un lot qui finirait après l'échéance ne part pas.
# Seulement pour un modèle qui a un budget de jetons (mistral-large-4) : ses
# durées et sa sortie sont les seules mesurées (ControleDuBillet._limiteur_du_plan).
# TOUR 7 : ce modèle juge l'article en UN appel, et chaque appel compte pour
# DUREE_D_UN_APPEL_AU_PIRE_S, quelle que soit sa taille (essai réel du 04/11 :
# un lot de 12 phrases prévu à 41 s en a pris 76, et le contrôle a fini en 504
# après deux lots payés). Le plan garde sa forme générale (des appels qui se
# suivent) ; il n'en a plus qu'un.


class _Instant:
    """L'horloge d'un plan : l'instant où il en est (Limiteur.copie)."""

    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


@dataclass(frozen=True)
class LotPrevu:
    """Un appel du plan, au pire mesuré : ses phrases, ses jetons d'entrée, son départ, sa fin."""

    phrases: int
    entree: int
    depart: float
    fin: float


@dataclass(frozen=True)
class PlanDuControle:
    """
    Les appels d'un contrôle, sur l'horloge du limiteur, le premier partant au
    plus tôt à `debut`. `limiteur` : la copie où ils sont réservés, donc l'état
    prévu une fois le dernier parti, d'où part le contrôle suivant de la file.
    """

    debut: float
    lots: Tuple[LotPrevu, ...]
    limiteur: Limiteur

    @property
    def fin(self) -> float:
        return self.lots[-1].fin if self.lots else self.debut

    @property
    def duree(self) -> float:
        return self.fin - self.debut


def planifier(
    limiteur: Limiteur, debut: float, appels: Sequence[Tuple[int, int]], *, vide: bool = False,
) -> PlanDuControle:
    """
    Le plan des `appels` — (phrases, jetons d'entrée) de chaque lot, dans
    l'ordre d'envoi —, sur une copie de `limiteur` (`vide` : sans rien de ce
    qu'il compte). Comme le client : chaque lot réserve son entrée quand le
    précédent a fini, puis compte son coût (ici au pire mesuré, comme _compter
    l'usage réel) ; le suivant attend s'il ne tient pas dans la minute. Chaque
    appel dure DUREE_D_UN_APPEL_AU_PIRE_S et rend
    JETONS_DE_SORTIE_D_UN_APPEL_AU_PIRE, quel que soit son nombre de phrases
    (tour 7 : la réflexion du modèle domine).
    """
    horloge = _Instant(debut)
    simule = limiteur.copie(horloge, vide=vide)
    lots: List[LotPrevu] = []
    for phrases, entree in appels:
        reservation = simule.reserver(entree)
        reservation.corriger(entree + JETONS_DE_SORTIE_D_UN_APPEL_AU_PIRE)
        fin = reservation.depart + DUREE_D_UN_APPEL_AU_PIRE_S
        lots.append(LotPrevu(phrases, entree, reservation.depart, fin))
        horloge.t = fin
    return PlanDuControle(debut, tuple(lots), simule)


class BudgetDuControleError(SaturationError):
    """
    Le plan du contrôle ne tient pas dans son délai, avec le budget de jetons
    de son modèle : 503 « saturation » et Retry-After, comme une saturation,
    mais le message parle du budget de l'article ; aucun lot n'est parti, ou
    le contrôle s'arrête avant un lot qui finirait trop tard.
    """


def _secondes(valeur: float) -> int:
    return max(0, math.ceil(valeur - 1e-9))


def _nombre(valeur: int) -> str:
    """« 20 000 » : les milliers séparés comme dans le reste des messages."""
    return f"{valeur:,}".replace(",", " ")


def _relancez(reprise: Optional[int]) -> str:
    if reprise is None:
        return "Relancez le contrôle."
    return f"Relancez le contrôle dans {reprise} s."


# Le Retry-After d'un plan refusé se cherche pas à pas (BudgetDuControle.reprise) ;
# chaque pas franchit au moins une entrée de la minute, il y en a peu.
PAS_DE_LA_REPRISE_AU_PLUS = 32
# POURQUOI une tolérance : un plan relancé au premier instant où il tient finit
# EXACTEMENT à sa limite, et l'arrondi des flottants le mettait parfois une
# fraction de microseconde au-delà (refus, puis recherche sans fin).
TOLERANCE_DU_PLAN_S = 1e-6


@dataclass(frozen=True)
class BudgetDuControle:
    """
    Ce qu'un contrôle sait pour se planifier : le limiteur de son modèle (qui
    tient le budget de jetons), ses appels (phrases, jetons d'entrée de chaque
    lot), son échéance sur l'horloge de ce limiteur (`fin_permise` : l'entrée
    dans la route, plus le délai qu'elle donne), et de quoi le dire
    (`caracteres` des extraits, `phrases` de l'article).
    """

    modele: str
    limiteur: Limiteur
    appels: Tuple[Tuple[int, int], ...]
    fin_permise: float
    caracteres: int
    phrases: int

    def plan(self, debut: float, *, base: Optional[Limiteur] = None, depuis: int = 0) -> PlanDuControle:
        """Le plan des lots `depuis` et suivants, partant de `debut`, sur `base` (le limiteur réel par défaut)."""
        return planifier(base if base is not None else self.limiteur, debut, self.appels[depuis:])

    def _budget(self) -> str:
        return f"{_nombre(self.limiteur.jetons_par_minute or 0)} jetons par minute"

    def _ce_qu_il_demande(self) -> str:
        """« ses 25 phrases se jugent en 2 appels au modèle du contrôle (…), qui n'a que … »"""
        n = len(self.appels)
        appels = "un appel" if n == 1 else f"{n} appels"
        chacun = "il reprend" if n == 1 else "chacun reprend"
        return (
            f"ses {self.phrases} phrases se jugent en {appels} au modèle du contrôle "
            f"({self.modele}), qui n'a que {self._budget()}, et {chacun} tous ses extraits "
            f"({_nombre(self.caracteres)} caractères)"
        )

    def verifier_seul(self) -> None:
        """
        TexteTropLongError (422) si l'article ne peut tenir dans DELAI_CONTROLE_S,
        marge comprise, même seul (aucun autre contrôle, rien dans la minute) :
        aucune relance n'y changerait rien, il faut le raccourcir.
        """
        plan = planifier(self.limiteur, 0.0, self.appels, vide=True)
        if plan.duree + MARGE_DU_PLAN_S <= DELAI_CONTROLE_S + TOLERANCE_DU_PLAN_S:
            return
        raise TexteTropLongError(
            f"L'article ne peut pas être contrôlé dans le délai, même seul : "
            f"{self._ce_qu_il_demande()}. Il faudrait ~{_secondes(plan.duree + MARGE_DU_PLAN_S)} s "
            f"au pire mesuré, attentes du budget de jetons et marge comprises, plus que les "
            f"{DELAI_CONTROLE_S} s du contrôle. Raccourcissez-le, puis relancez le contrôle."
        )

    def reprise(self, base: Limiteur, a_partir_de: float) -> Optional[float]:
        """
        Le premier instant, à partir de `a_partir_de`, où ce contrôle RELANCÉ
        (DELAI_CONTROLE_S entiers, marge comprise) tiendrait son plan, sur
        l'état `base` : le limiteur réel, ou ce que prévoient ceux de devant
        dans la file. None si la recherche n'aboutit pas.
        """
        permis = DELAI_CONTROLE_S - MARGE_DU_PLAN_S
        debut = a_partir_de
        for _ in range(PAS_DE_LA_REPRISE_AU_PLUS):
            plan = planifier(base, debut, self.appels)
            if plan.duree <= permis + TOLERANCE_DU_PLAN_S:
                return debut
            # POURQUOI ce pas : la fin d'un plan ne recule jamais quand il part
            # plus tard (la minute ne fait que se vider) ; partir avant
            # fin − permis ne peut donc pas tenir, et partir à cet instant le peut.
            debut = plan.fin - permis
        return None

    def _retry_after(self, base: Limiteur, a_partir_de: float, maintenant: float) -> Optional[int]:
        """Le Retry-After d'un refus : None si une relance tiendrait dès maintenant."""
        quand = self.reprise(base, a_partir_de)
        if quand is None:
            return REPRISE_APRES_UN_AUTRE_CONTROLE_S
        secondes = _secondes(quand - maintenant)
        return secondes or None

    def avant_le_lot(self, k: int, publier: Callable[[PlanDuControle], None]) -> float:
        """
        AVANT d'envoyer le lot `k` (à partir de 0) : le plan des lots k et
        suivants, sur l'état réel du limiteur, publié pour la file. Il doit
        finir avant l'échéance, et MARGE_DU_PLAN_S avant elle pour le premier
        lot (rien n'est encore payé) ; sinon BudgetDuControleError, et le lot
        ne part pas. Rend l'attente permise au lot (son attente_max) : celle que
        le plan prévoit, plus ce qui reste avant cette limite.

        POURQUOI la marge reste à l'appel (tour 7) : l'attente permise allait
        jusqu'à l'échéance elle-même ; un 429 ou un recalage pendant l'attente
        pouvait faire partir l'appel à DUREE_D_UN_APPEL_AU_PIRE_S de la fin, et
        un appel à peine plus lent que le pire mesuré était coupé, payé (504).
        """
        maintenant = self.limiteur.maintenant()
        plan = self.plan(maintenant, depuis=k)
        publier(plan)
        limite = self.fin_permise - (MARGE_DU_PLAN_S if k == 0 else 0.0)
        if plan.fin > limite + TOLERANCE_DU_PLAN_S:
            raise self._refus(plan, k, maintenant)
        return (plan.lots[0].depart - maintenant) + max(0.0, limite - plan.fin) + TOLERANCE_D_ATTENTE_S

    def _refus(self, plan: PlanDuControle, envoyes: int, maintenant: float) -> BudgetDuControleError:
        reprise = self._retry_after(self.limiteur, maintenant, maintenant)
        restant = _secondes(self.fin_permise - maintenant)
        if envoyes == 0:
            texte = (
                f"Le contrôle IA de cet article ne tient pas dans son délai pour l'instant : "
                f"{self._ce_qu_il_demande()}. Il lui faudrait "
                f"~{_secondes(plan.duree + MARGE_DU_PLAN_S)} s au pire mesuré, attentes du budget "
                f"de jetons et marge comprises, et il en reste {restant}. Rien n'a été envoyé au "
                f"modèle, l'article est inchangé. {_relancez(reprise)}"
            )
        else:
            texte = (
                f"Le contrôle IA s'est arrêté après {envoyes} appel(s) sur {len(self.appels)} : la "
                f"suite finirait après son délai (~{_secondes(plan.duree)} s au pire mesuré, "
                f"attentes du budget de jetons comprises, et il en reste {restant}), le modèle du "
                f"contrôle ({self.modele}) n'ayant que {self._budget()}. Rien n'est enregistré, "
                f"l'article est inchangé. {_relancez(reprise)}"
            )
        logger.warning(
            f"⚠️ Blog : contrôle refusé avant le lot {envoyes + 1}/{len(self.appels)}, plan de "
            f"{plan.duree:.1f} s pour {self.fin_permise - maintenant:.1f} s restantes "
            f"(départs {[round(lot.depart - maintenant, 1) for lot in plan.lots]}) ; reprise {reprise}"
        )
        erreur = BudgetDuControleError(texte)
        erreur.retry_after = reprise
        return erreur

    def dans_la_file(self, devant: "_Prevu") -> Tuple[Optional[PlanDuControle], float]:
        """
        Le plan de ce contrôle s'il part APRÈS ceux de devant (`devant` : ce que
        prévoit le dernier d'entre eux), dans les jetons qu'ils auront pris ; et
        combien de temps il peut attendre le verrou. (None, 0) s'il ne tiendrait
        pas dans son délai, marge comprise : la file ne le garde pas.
        """
        maintenant = self.limiteur.maintenant()
        plan = self.plan(max(maintenant, devant.fin), base=devant.limiteur)
        marge = self.fin_permise - MARGE_DU_PLAN_S - plan.fin
        if marge < -TOLERANCE_DU_PLAN_S:
            return None, 0.0
        return plan, plan.debut - maintenant + max(0.0, marge)

    def refus_dans_la_file(self, devant: Optional["_Prevu"]) -> BudgetDuControleError:
        """
        503 pour un contrôle que la file ne garde pas, ou qui a attendu le
        verrou plus que son plan ne le permettait : Retry-After = quand son plan
        tiendra, une fois ceux de devant finis (`devant`, s'ils l'ont dit).
        """
        maintenant = self.limiteur.maintenant()
        if devant is None:
            reprise: Optional[int] = REPRISE_APRES_UN_AUTRE_CONTROLE_S
            fin = ""
        elif devant.fin >= maintenant:
            reprise = self._retry_after(devant.limiteur, devant.fin, maintenant)
            fin = f" (sa fin est prévue dans ~{max(1, _secondes(devant.fin - maintenant))} s au pire mesuré)"
        else:
            # POURQUOI la minute d'avant : celui de devant dépasse son plan (plus lent que le pire
            # mesuré), sa fin n'est plus connue ; une relance immédiate retomberait dans la file.
            reprise = REPRISE_APRES_UN_AUTRE_CONTROLE_S
            fin = " (il dure plus que prévu)"
        logger.warning(f"⚠️ Blog : contrôle refusé, un autre contrôle de {self.modele} est en cours ; reprise {reprise}")
        erreur = BudgetDuControleError(
            f"Un autre contrôle IA est en cours avec le modèle {self.modele}, qui n'en fait qu'un à "
            f"la fois{fin} : après lui, avec ce qu'il laissera des {self._budget()} du modèle, "
            f"celui-ci ne finirait pas dans son délai. Il n'a pas été fait, rien n'a été envoyé au "
            f"modèle, l'article est inchangé. {_relancez(reprise)}"
        )
        erreur.retry_after = reprise
        return erreur


def llm_du_controle() -> Any:
    """
    Le modèle du contrôle (tour 6) : chez Mistral, BLOG_CONTROLE_MODELE, sans
    modèle de secours (mistral_du_controle) ; avec Gemini (LLM_PROVIDER), le
    service du chat, comme avant.
    """
    if settings.LLM_PROVIDER == "mistral":
        return mistral_du_controle()
    return get_llm_service()


@dataclass(frozen=True)
class _Prevu:
    """Ce que prévoit un contrôle de la file : sa fin, et l'état du limiteur une fois ses lots partis."""

    fin: float
    limiteur: Limiteur


class _FileDuModele:
    """
    Le verrou d'un modèle, et ce que prévoient les contrôles qui le tiennent ou
    l'attendent (`prevus`, dans l'ordre de la file : celui qui le tient d'abord).
    """

    def __init__(self) -> None:
        self.verrou = asyncio.Lock()
        self.prevus: Dict[object, _Prevu] = {}

    def devant(self, place: object) -> Optional[_Prevu]:
        """Ce que prévoit le dernier contrôle devant `place` dans la file, s'il l'a dit."""
        dernier: Optional[_Prevu] = None
        for cle, prevu in self.prevus.items():
            if cle is place:
                break
            dernier = prevu
        return dernier


# Une file par modèle ET par boucle d'événements. POURQUOI par boucle : un
# asyncio.Lock appartient à la boucle où il a fait attendre en premier, et
# chaque test (ou script) a la sienne ; comme les sémaphores du Limiteur. En
# production, une boucle par processus : une file par modèle et par processus.
_FILES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[str, _FileDuModele]]" = (
    weakref.WeakKeyDictionary()
)


def _file_du_modele(modele: str) -> _FileDuModele:
    files = _FILES.setdefault(asyncio.get_running_loop(), {})
    return files.setdefault(modele, _FileDuModele())


@asynccontextmanager
async def un_controle_a_la_fois(
    modele: str, budget: Optional[BudgetDuControle] = None,
) -> AsyncIterator[Callable[[PlanDuControle], None]]:
    """
    Jamais deux contrôles du même modèle ensemble dans le processus (tour 6).

    POURQUOI : mistral-large-4 n'a que 20 000 jetons par minute à la clé, et
    un contrôle d'article, en UN appel (tour 7), en coûte 13 600 à 15 600
    (mesures du 04/11 ; en deux lots, l'essai du tour 6 en a payé 25 000) ; deux
    contrôles ensemble se prendraient le budget, et chacun pourrait recevoir
    un 429 au milieu de son article. Le second attend la fin du premier
    (file d'attente du verrou), au plus ATTENTE_D_UN_AUTRE_CONTROLE_S.

    LA FILE NE GARDE UN CONTRÔLE QUE S'IL RESTE LA PLACE DE SON PLAN (revue du
    tour 6). Avec un `budget`, le contrôle qui arrive derrière d'autres se
    planifie APRÈS eux, d'après ce qu'ils prévoient (leur fin, et les jetons
    qu'ils auront pris) : s'il ne finirait pas dans son délai, il est refusé
    aussitôt ; sinon il attend le verrou, au plus la marge de son plan. Refusé
    ou lassé : BudgetDuControleError (503), avec le Retry-After où son plan
    tiendra, sans avoir rien envoyé. Sans budget (une doublure, un modèle sans
    budget de jetons), 20 s, puis SaturationError avec une minute de Retry-After.

    Rend `publier` : le contrôle qui tient le verrou y publie son plan (avant
    chaque lot), pour ceux qui arrivent derrière lui. Le verrou est rendu, et
    la place quittée, quoi qu'il arrive, échéance de la route comprise.
    """
    file = _file_du_modele(modele)
    place = object()
    attente = ATTENTE_D_UN_AUTRE_CONTROLE_S
    try:
        devant = file.devant(place)
        if budget is not None and devant is not None:
            plan, permise = budget.dans_la_file(devant)
            if plan is None:
                raise budget.refus_dans_la_file(devant)
            attente = min(attente, permise)
            file.prevus[place] = _Prevu(plan.fin, plan.limiteur)
        try:
            if attente <= 0 and file.verrou.locked():
                raise TimeoutError
            async with asyncio.timeout(attente):
                await file.verrou.acquire()
        except TimeoutError:
            if budget is not None and file.devant(place) is not None:
                raise budget.refus_dans_la_file(file.devant(place)) from None
            logger.warning(f"⚠️ Blog : contrôle refusé, un autre contrôle de {modele} est en cours")
            erreur = SaturationError(
                f"Un autre contrôle IA est en cours avec le modèle {modele}, qui n'en fait qu'un à la "
                "fois : celui-ci n'a pas été fait, l'article est inchangé. Relancez le contrôle dans "
                "une minute."
            )
            erreur.retry_after = REPRISE_APRES_UN_AUTRE_CONTROLE_S
            raise erreur from None
        try:
            yield lambda plan: file.prevus.__setitem__(place, _Prevu(plan.fin, plan.limiteur))
        finally:
            file.verrou.release()
    finally:
        file.prevus.pop(place, None)


def _modele_servi(reponse: Any) -> Optional[str]:
    """Le modèle qui a répondu à un appel (« modele » de generate), s'il le dit."""
    nom = reponse.get("modele") if isinstance(reponse, dict) else None
    return nom.strip() if isinstance(nom, str) and nom.strip() else None


class ControleDuBillet:
    """
    Le modèle est un PARAMÈTRE (comme BlogGeneratorService) : les tests
    injectent une doublure, et le vrai modèle n'est jamais appelé. Par défaut,
    celui du contrôle (llm_du_controle), pas celui de la génération.
    """

    def __init__(self, db: AsyncSession, *, llm: Optional[Any] = None):
        self.db = db
        self.llm = llm if llm is not None else llm_du_controle()

    @property
    def modele(self) -> str:
        """Le modèle DEMANDÉ ; controle_ia note celui qui a répondu (controler)."""
        nom = getattr(self.llm, "model_name", None)
        if isinstance(nom, str) and nom.strip():
            return nom.strip()
        return modele_du_controle() if settings.LLM_PROVIDER == "mistral" else settings.GEMINI_MODEL

    def _limiteur_du_plan(self) -> Optional[Limiteur]:
        """
        Le limiteur du modèle du contrôle, s'il tient un budget de jetons par
        minute (mistral-large-4, limites_du_modele) : le contrôle se planifie
        alors (BudgetDuControle). POURQUOI seulement lui : les durées et la
        sortie du plan sont celles mesurées sur ce modèle, et c'est son budget
        qui fait attendre les lots. None pour un modèle sans budget de jetons,
        pour Gemini ou une doublure : rien à planifier, l'échéance de la route
        reste leur seule borne, comme avant.
        """
        if not isinstance(self.llm, MistralService):
            return None
        limiteur = self.llm.limiteur()
        return limiteur if limiteur.jetons_par_minute else None

    async def controler(self, post: Any, *, delai: Optional[float] = None) -> Dict[str, Any]:
        """
        Le contrôle de l'article TEL QU'IL EST : {fait_le, modele, empreinte,
        phrases, empreintes_des_parties, extraits_ajoutes, appels}. N'écrit
        rien : la route enregistre.

        Un appel au modèle par lot de phrases (lots_du_controle), l'un après
        l'autre, un seul contrôle à la fois par modèle (un_controle_a_la_fois),
        sous l'échéance unique de la route ; un SEUL appel pour tout l'article
        avec un modèle à budget de jetons (tour 7). `modele` est celui qui a VRAIMENT
        répondu (le champ « model » de la réponse ; tour 6), à défaut celui
        qu'on a demandé. La transaction est close AVANT le modèle, qui peut
        prendre une minute (comme la génération) : rien n'a été écrit, et la
        connexion est rendue au pool.

        `delai` : ce que la route laisse au contrôle, compté d'ici (son
        échéance) ; DELAI_CONTROLE_S par défaut. Avec un modèle à budget de
        jetons, le contrôle s'y planifie, attente d'un autre contrôle comprise
        (revue du tour 6, BudgetDuControle) : il ne part que si son appel
        finit à temps au pire mesuré, marge comprise.

        Raises:
            SourcesAbsentesError (422), TexteTropLongError (422 : trop de
            phrases, ou un plan qui ne tiendrait jamais dans le délai),
            QuotaError (429), SaturationError (503 : modèle saturé, ou un autre
            contrôle en cours), BudgetDuControleError (503 : le plan ne tient
            pas dans le délai, ou après un autre contrôle ; Retry-After où il
            tiendra), ReponseInvalideError (502)
        """
        limiteur = self._limiteur_du_plan()
        # L'échéance, sur l'horloge du limiteur, prise UNE fois, à l'entrée :
        # l'attente du verrou et chaque lot s'y comptent ensuite.
        fin_permise = (
            limiteur.maintenant() + (DELAI_CONTROLE_S if delai is None else delai)
            if limiteur is not None else None
        )
        contenu = ContenuDuBillet.de(post)
        if not sources_valables(contenu.sources):
            raise SourcesAbsentesError(SANS_SOURCES)
        phrases = phrases_du_billet(contenu)
        # Tour 7 : tout l'article en un appel pour un modèle à budget de jetons.
        lots = lots_du_controle(phrases, un_seul_appel=limiteur is not None)
        extraits, rattachement, ajoutes = await extraits_du_controle(self.db, contenu)
        if not extraits:  # pragma: no cover - des sources valables donnent toujours un extrait
            raise SansSourceError(SANS_SOURCES)
        prompts = [
            prompt_du_controle(extraits, lot, contenu, rattachement, autres=len(phrases) - len(lot))
            for lot in lots
        ]
        budget: Optional[BudgetDuControle] = None
        if limiteur is not None and fin_permise is not None:
            budget = BudgetDuControle(
                modele=self.modele, limiteur=limiteur,
                appels=tuple(
                    (len(lot), self.llm.jetons_d_entree(prompt, SYSTEM_DU_CONTROLE))
                    for lot, prompt in zip(lots, prompts)
                ),
                fin_permise=fin_permise, caracteres=sum(len(e.texte) for e in extraits),
                phrases=len(phrases),
            )
        empreinte = empreinte_du_controle(contenu)
        empreintes = empreintes_des_parties(contenu)
        await self.db.commit()
        if budget is not None:
            budget.verifier_seul()

        debut = time.perf_counter()
        async with un_controle_a_la_fois(self.modele, budget) as publier:
            juges = await self._juger(prompts, lots, extraits, budget, publier)
        jugees = [phrase for phrases_du_lot, _, _ in juges for phrase in phrases_du_lot]
        compte = {v: sum(p["verdict"] == v for p in jugees) for v in VERDICTS}
        # Le modèle qui a VRAIMENT répondu (tour 6) : jamais le nom demandé à
        # la place d'un autre. Une doublure, ou Gemini, ne le dit pas : le nom
        # demandé, comme avant.
        servis = list(dict.fromkeys(servi for _, _, servi in juges if servi))
        modele = " + ".join(servis) if servis else self.modele
        if servis and servis != [self.modele]:
            logger.warning(f"⚠️ Blog : contrôle demandé à {self.modele}, rendu par {modele}")
        logger.info(
            f"🔎 Blog : contrôle de {len(jugees)} phrase(s) sur {len(extraits)} extrait(s) par "
            f"{modele} en {time.perf_counter() - debut:.1f} s, {len(lots)} appel(s) l'un après "
            f"l'autre ({', '.join(f'{len(lot)} phrases {duree:.1f} s' for lot, (_, duree, _) in zip(lots, juges))}) "
            f": {compte}"
        )
        return {
            "fait_le": maintenant_iso(),
            "modele": modele,
            "empreinte": empreinte,
            "phrases": jugees,
            # Gardées pour dire QUELLES parties ont changé depuis le contrôle,
            # et quels extraits il a ajoutés ; pas dans `controle_ia` de l'API.
            "empreintes_des_parties": empreintes,
            "extraits_ajoutes": ajoutes,
            # La durée de chaque appel, pour mesurer le contrôle sur les vrais
            # articles (voir PHRASES_PAR_APPEL) ; pas dans l'API non plus.
            "appels": [
                {"phrases": len(lot), "duree_s": round(duree, 1)} for lot, (_, duree, _) in zip(lots, juges)
            ],
        }

    async def _juger(
        self, prompts: Sequence[str], lots: Sequence[Sequence[PhraseAJuger]],
        extraits: Sequence[ExtraitDuControle], budget: Optional[BudgetDuControle] = None,
        publier: Callable[[PlanDuControle], None] = lambda plan: None,
    ) -> List[Tuple[List[Dict[str, Any]], float, Optional[str]]]:
        """
        Les verdicts de chaque lot, la durée de son appel et le modèle qui a
        répondu. Les appels partent L'UN APRÈS L'AUTRE (tour 6, voir
        PHRASES_AU_PLUS) : chacun ne part qu'une fois le précédent lu, dans le
        budget de jetons du modèle (le limiteur du client). Avec un `budget`
        (revue du tour 6), chacun ne part que si le plan des lots qui restent
        tient dans l'échéance, et attend au plus ce que ce plan lui laisse
        (BudgetDuControle.avant_le_lot). Le premier qui échoue fait échouer le
        contrôle, avec son erreur, et les suivants ne partent pas : un contrôle
        à moitié fait n'en est pas un (rien n'est enregistré). L'échéance de la
        route coupe l'appel en cours.
        """
        juges: List[Tuple[List[Dict[str, Any]], float, Optional[str]]] = []
        for k, (prompt, lot) in enumerate(zip(prompts, lots)):
            attente_max = budget.avant_le_lot(k, publier) if budget is not None else None
            debut = time.perf_counter()
            reponse = await self._appeler(prompt, attente_max) or {}
            juges.append(
                (lire_les_verdicts(reponse, lot, extraits), time.perf_counter() - debut, _modele_servi(reponse))
            )
        return juges

    async def _appeler(self, prompt: str, attente_max: Optional[float] = None) -> Dict[str, Any]:
        """
        L'appel au modèle, ses erreurs traduites comme celles de la génération.
        Jamais d'autre modèle en secours (tour 6) : un 429 qui dure, ou un
        refus, devient une erreur que la route rend à l'administrateur (503 ou
        429 avec Retry-After, 502), et rien n'est enregistré. `attente_max` :
        l'attente que le plan laisse à cet appel (revue du tour 6), au lieu des
        20 s du chat ; passée au seul client qui la connaît (MistralService).
        """
        try:
            return await self.llm.generate(
                prompt=prompt,
                system=SYSTEM_DU_CONTROLE,
                temperature=CONTROLE_TEMPERATURE,
                max_tokens=CONTROLE_MAX_TOKENS,
                response_mime_type="application/json",
                response_schema=SCHEMA_DU_CONTROLE,
                # L'effort de raisonnement du modèle du contrôle (tour 6) ; « low »
                # pour Gemini 3 aussi, comme avant.
                reflexion=settings.BLOG_CONTROLE_REFLEXION,
                # Le modèle qui a VRAIMENT répondu, que controle_ia note.
                rendre_le_modele=True,
                **({"attente_max": attente_max} if attente_max is not None else {}),
            )
        # Quota et saturation AVANT l'erreur générique, dont elles dérivent.
        except ERREURS_QUOTA as e:
            logger.warning(f"⚠️ Blog : quota épuisé pendant le contrôle ({e!r})")
            raise QuotaError(
                "Le quota du modèle est épuisé : le contrôle IA n'a pas été fait, l'article est "
                "inchangé. Relancez le contrôle plus tard."
            ) from e
        except ERREURS_SATURATION as e:
            logger.warning(f"⚠️ Blog : contrôle saturé ({e!r})")
            # Le délai que le client connaît (Retry-After du fournisseur, attente
            # du budget de jetons), sinon celui de SaturationError.
            reprise = getattr(e, "retry_after", None)
            reprise = math.ceil(reprise) if isinstance(reprise, (int, float)) and reprise > 0 else None
            erreur = SaturationError(
                f"Le modèle du contrôle ({self.modele}) est saturé, ou a atteint sa limite de "
                "débit : le contrôle IA n'a pas été fait, l'article est inchangé. Relancez le "
                f"contrôle {f'dans {reprise} s' if reprise else 'dans quelques instants'}."
            )
            if reprise:
                erreur.retry_after = reprise
            raise erreur from e
        except ERREURS_BUDGET as e:
            raise ReponseInvalideError(
                "Le modèle n'a pas pu rendre le contrôle dans le budget imparti : l'article est "
                "inchangé. Relancez le contrôle."
            ) from e
        except ERREURS_LLM as e:
            logger.error(f"❌ Blog : erreur du modèle pendant le contrôle ({e!r})")
            raise ReponseInvalideError(
                f"Le service du modèle du contrôle ({self.modele}) a renvoyé une erreur : le "
                "contrôle IA n'a pas été fait, l'article est inchangé. Relancez le contrôle plus tard."
            ) from e


# ==================== CE QUE L'API EN MONTRE ====================


def _verdict_de(phrase: Mapping[str, Any]) -> str:
    """Le verdict enregistré d'une phrase ; illisible, « non_soutenue » (jamais un accord)."""
    verdict = phrase.get("verdict")
    return verdict if verdict in VERDICTS else "non_soutenue"


def _nature_de(phrase: Mapping[str, Any]) -> str:
    """
    « autre » si le contrôle l'a dit ; sinon « regle » : dans le doute, une
    règle (lire_la_nature), y compris pour une phrase que le modèle n'a pas
    jugée (RAISON_NON_JUGEE), qui n'a pas de nature.
    """
    return "autre" if phrase.get("nature") == "autre" else "regle"


def controle_affiche(controle: Any) -> Optional[Dict[str, Any]]:
    """
    `controle_ia` tel que le contrat le décrit, ou None (pas de contrôle, ou
    illisible). `nature` (avenant du tour 7) : « regle » ou « autre », pour
    que l'administration sache quelles phrases sont à confirmer
    (phrases_a_confirmer). Jamais dans une réponse publique.
    """
    if not isinstance(controle, dict) or not isinstance(controle.get("phrases"), list):
        return None
    phrases = []
    for phrase in controle["phrases"]:
        if not isinstance(phrase, dict):
            continue
        phrases.append({
            "id": str(phrase.get("id") or ""),
            "partie": str(phrase.get("partie") or ""),
            "index": _entier(phrase.get("index")) or 0,
            "texte": str(phrase.get("texte") or ""),
            "verdict": _verdict_de(phrase),
            "extraits": _numeros(phrase.get("extraits")),
            "raison": str(phrase.get("raison") or ""),
            "nature": _nature_de(phrase),
        })
    return {
        "fait_le": str(controle.get("fait_le") or ""),
        "modele": str(controle.get("modele") or ""),
        "empreinte": str(controle.get("empreinte") or ""),
        "phrases": phrases,
    }


# ==================== LES PHRASES À CONFIRMER (TOUR 7) ====================
#
# DÉCISION DU PROPRIÉTAIRE (04/11) : « Confirmer phrase par phrase ». Une phrase
# de RÈGLE (nature « regle ») que le contrôle n'a pas trouvée dans les extraits
# (« non_soutenue », « À vérifier » compris, ou « hors_sources ») se réécrit (un
# nouveau contrôle la juge alors) ou se CONFIRME, une à une, par une personne.
# relecture.phrases_confirmees enregistre, pour chaque phrase confirmée, son
# empreinte (empreintes_a_confirmer) ; elle ne compte que si la phrase, son
# verdict et les sources n'ont pas changé depuis, comme une case de partie.


def phrases_a_confirmer(controle: Any) -> List[Dict[str, Any]]:
    """Les phrases de règle du contrôle enregistré qu'il n'a pas trouvées dans les extraits, dans l'ordre."""
    if not isinstance(controle, dict) or not isinstance(controle.get("phrases"), list):
        return []
    return [
        p for p in controle["phrases"]
        if isinstance(p, dict) and _nature_de(p) == "regle" and _verdict_de(p) in VERDICTS_A_CONFIRMER
    ]


def _id_de(phrase: Mapping[str, Any]) -> str:
    return str(phrase.get("id") or "")


def empreintes_a_confirmer(contenu: ContenuDuBillet, controle: Any) -> Dict[str, str]:
    """
    Ce qu'atteste la confirmation de chaque phrase à confirmer, par
    identifiant : la phrase (sa partie, son texte), son verdict, et les
    sources. Comme une case (empreintes_de_relecture), elle tombe si l'un
    d'eux change.
    """
    sources = _sources_canon(contenu.sources)
    return {
        _id_de(p): _empreinte([
            "phrase", _id_de(p), str(p.get("partie") or ""), _canon(p.get("texte")), _verdict_de(p), sources,
        ])
        for p in phrases_a_confirmer(controle) if _id_de(p)
    }


def _confirmations(relecture: Any) -> Dict[str, Any]:
    """Les confirmations enregistrées : {identifiant: empreinte}, {} sinon."""
    confirmees = relecture.get("phrases_confirmees") if isinstance(relecture, dict) else None
    return dict(confirmees) if isinstance(confirmees, dict) else {}


def phrases_confirmees(contenu: ContenuDuBillet, controle: Any, relecture: Any) -> List[str]:
    """Les identifiants des phrases dont la confirmation compte aujourd'hui, dans l'ordre de l'article."""
    enregistrees = _confirmations(relecture)
    return [i for i, empreinte in empreintes_a_confirmer(contenu, controle).items() if enregistrees.get(i) == empreinte]


def phrases_non_confirmees(contenu: ContenuDuBillet, controle: Any, relecture: Any) -> List[Dict[str, Any]]:
    """Les phrases à confirmer qui ne le sont pas (ou plus) : elles bloquent la publication."""
    confirmees = set(phrases_confirmees(contenu, controle, relecture))
    return [p for p in phrases_a_confirmer(controle) if _id_de(p) not in confirmees]


def parties_relues(contenu: ContenuDuBillet, controle: Any, relecture: Any) -> Dict[str, bool]:
    """
    Chaque partie : relue si sa case atteste le texte, les sources et les
    verdicts d'AUJOURD'HUI (empreintes_de_relecture). Une partie vide n'a rien
    à relire.
    """
    empreintes = empreintes_de_relecture(contenu, controle)
    relues = relecture.get("parties") if isinstance(relecture, dict) else None
    relues = relues if isinstance(relues, dict) else {}
    return {
        partie: empreintes[partie] is None or relues.get(partie) == empreintes[partie]
        for partie in PARTIES
    }


def relecture_apres_un_controle(
    contenu: ContenuDuBillet, ancien: Any, nouveau: Any, relecture: Any
) -> Any:
    """
    La relecture à enregistrer avec un NOUVEAU contrôle : une case ne reste
    cochée que si elle comptait avec le contrôle remplacé (`ancien`) ET compte
    avec le nouveau, c'est-à-dire si sa partie garde ses verdicts ; les autres
    sont effacées. `par`, `le` et `controle` restent ceux de la dernière case
    cochée par une personne. Rien à effacer : la relecture telle quelle.

    POURQUOI (revue du front du tour 5, data/blog/tour5/revue-front, par la
    vraie API). Une case garde l'empreinte de ce qu'elle atteste (texte,
    sources, verdicts de sa partie) : un contrôle qui jugeait autrement sa
    partie la faisait seulement ne plus compter, et un contrôle suivant qui
    redonnait les verdicts d'avant la faisait revivre. Un article en ligne,
    caché par le contrôle Y (contexte « contredite » ou « À vérifier »),
    revenait ainsi en ligne au contrôle Z (les verdicts changent d'un contrôle
    à l'autre, à la température 0), sans que personne ait lu ce que Y avait
    trouvé, ni relu, ni publié. Une case tombée est effacée : seule une
    personne la recoche, contre le contrôle enregistré (relecture_perimee).
    Une case qui ne comptait déjà plus (cochée avant une reconstruction des
    sources, ou avant ce correctif) est effacée de même.

    LES PHRASES CONFIRMÉES (avenant du tour 7) tombent EXACTEMENT de même : une
    confirmation ne reste que si elle comptait avec le contrôle remplacé et
    compte avec le nouveau (la même phrase, au même rang, le même verdict, les
    mêmes sources) ; sinon elle est effacée, et ne revit pas au contrôle
    suivant.
    """
    if not isinstance(relecture, dict):
        return relecture
    nouvelle = dict(relecture)
    cochees = relecture.get("parties")
    if isinstance(cochees, dict):
        avant = empreintes_de_relecture(contenu, ancien)
        apres = empreintes_de_relecture(contenu, nouveau)
        gardees = {
            partie: cochees.get(partie) if cochees.get(partie) == avant[partie] == apres[partie] else None
            for partie in PARTIES
        }
        if gardees != {partie: cochees.get(partie) for partie in PARTIES}:
            nouvelle["parties"] = gardees
    confirmees = relecture.get("phrases_confirmees")
    if isinstance(confirmees, dict):
        avant_c = empreintes_a_confirmer(contenu, ancien)
        apres_c = empreintes_a_confirmer(contenu, nouveau)
        gardees_c = {i: e for i, e in confirmees.items() if e == avant_c.get(i) == apres_c.get(i)}
        if gardees_c != confirmees:
            nouvelle["phrases_confirmees"] = gardees_c
    return relecture if nouvelle == relecture else nouvelle


def relecture_affichee(contenu: ContenuDuBillet, controle: Any, relecture: Any) -> Dict[str, Any]:
    """
    `relecture` telle que le contrat la décrit (réponses d'administration
    seulement depuis le tour 7). `controle` (avenant du tour 5) : le `fait_le`
    du contrôle contre lequel les cases enregistrées ont été cochées, ou null.
    `phrases_confirmees` (avenant du tour 7) : les identifiants des phrases de
    règle dont la confirmation compte avec le contrôle enregistré, dans l'ordre
    de l'article.
    """
    relecture = relecture if isinstance(relecture, dict) else {}
    return {
        "parties": parties_relues(contenu, controle, relecture),
        "par": str(relecture.get("par") or "") or None,
        "le": str(relecture.get("le") or "") or None,
        "controle": str(relecture.get("controle") or "") or None,
        "phrases_confirmees": phrases_confirmees(contenu, controle, relecture),
    }


def relecture_perimee(cochees: Mapping[str, bool], controle_relu: Optional[str], controle: Any) -> Optional[str]:
    """
    Le `detail` du refus 422 « relecture_perimee », ou None si les cases
    peuvent s'enregistrer.

    AVENANT DU TOUR 5 (des deux côtés). Une requête qui coche une partie nomme
    le contrôle que l'administrateur a relu : `relecture.controle`, son
    `fait_le`. Absent, ou autre que celui du contrôle ENREGISTRÉ : refusé, et
    rien n'est appliqué. POURQUOI (revue du tour 4) : l'API attestait le
    contrôle enregistré au moment de la requête ; relancé entre-temps dans un
    autre onglet, par un autre administrateur, ou après une réponse perdue, un
    contrôle dont personne n'avait lu les verdicts passait pour relu, et
    l'article partait en ligne en une requête. Sans contrôle enregistré, aucune
    case n'a rien à attester. Décocher n'atteste rien : pas de contrôle à nommer.
    """
    if not any(cochee is True for cochee in cochees.values()):
        return None
    affiche = controle_affiche(controle)
    enregistre = affiche["fait_le"] if affiche else ""
    relu = (controle_relu or "").strip()
    if not enregistre:
        return (
            "Relecture refusée : aucun contrôle IA n'est enregistré pour cet article, les cases "
            "n'ont rien à attester. Rien n'a été enregistré : lancez le contrôle IA, puis relisez "
            "chaque partie contre ses verdicts."
        )
    if not relu:
        return (
            "Relecture refusée : les cases cochées ne disent pas contre quel contrôle IA elles "
            "l'ont été (relecture.controle, le fait_le du contrôle relu). Rien n'a été enregistré : "
            "relisez l'article, puis cochez de nouveau."
        )
    if relu != enregistre:
        return (
            f"Relecture périmée : les cases cochées portent sur un autre contrôle IA que celui "
            f"enregistré (fait le {enregistre}, relancé depuis). Rien n'a été enregistré : relisez "
            f"l'article avec les verdicts du contrôle enregistré, puis cochez de nouveau."
        )
    return None


def confirmations_perimees(
    ids: Optional[Sequence[str]], controle_relu: Optional[str], controle: Any,
) -> Optional[str]:
    """
    Le `detail` du refus 422 « relecture_perimee » de `relecture.phrases_confirmees`,
    ou None si elles peuvent s'enregistrer.

    AVENANT DU TOUR 7 (des deux côtés). La liste des phrases confirmées part
    TOUJOURS avec `relecture.controle`, le `fait_le` du contrôle affiché :
    absent, ou autre que celui du contrôle ENREGISTRÉ (ou aucun contrôle) :
    refusé, rien n'est appliqué, comme une case (relecture_perimee). Un
    identifiant que ce contrôle n'a pas (« P31 » d'un contrôle d'avant, ou
    inventé) aussi : la personne a confirmé une phrase qu'elle n'a pas sous
    les yeux. Une phrase connue qui n'est pas à confirmer (soutenue,
    contredite, ou d'une autre nature) n'est pas refusée : elle n'est pas
    gardée (nouvelle_relecture).
    """
    if ids is None:
        return None
    affiche = controle_affiche(controle)
    enregistre = affiche["fait_le"] if affiche else ""
    relu = (controle_relu or "").strip()
    if not enregistre:
        return (
            "Relecture refusée : aucun contrôle IA n'est enregistré pour cet article, il n'a aucune "
            "phrase à confirmer. Rien n'a été enregistré : lancez le contrôle IA, puis confirmez les "
            "phrases de règle qu'il n'a pas trouvées dans les extraits."
        )
    if not relu:
        return (
            "Relecture refusée : les phrases confirmées ne disent pas contre quel contrôle IA elles "
            "l'ont été (relecture.controle, le fait_le du contrôle relu). Rien n'a été enregistré : "
            "relisez l'article, puis confirmez de nouveau."
        )
    if relu != enregistre:
        return (
            f"Relecture périmée : les phrases confirmées portent sur un autre contrôle IA que celui "
            f"enregistré (fait le {enregistre}, relancé depuis). Rien n'a été enregistré : relisez les "
            f"phrases avec les verdicts du contrôle enregistré, puis confirmez de nouveau."
        )
    connues = {p["id"] for p in affiche["phrases"]}
    inconnues = [i for i in dict.fromkeys(ids) if i not in connues]
    if inconnues:
        quelles = f"la phrase {inconnues[0]}" if len(inconnues) == 1 else f"les phrases {_en_liste(inconnues)}"
        return (
            f"Relecture refusée : le contrôle IA enregistré (fait le {enregistre}) n'a pas {quelles}. "
            f"Rien n'a été enregistré : relisez l'article avec les verdicts du contrôle enregistré, "
            f"puis confirmez de nouveau."
        )
    return None


def controle_a_jour(contenu: ContenuDuBillet, controle: Any) -> bool:
    return (
        isinstance(controle, dict) and isinstance(controle.get("phrases"), list)
        and controle.get("empreinte") == empreinte_du_controle(contenu)
    )


def phrases_contredites(controle: Any) -> List[Dict[str, Any]]:
    if not isinstance(controle, dict) or not isinstance(controle.get("phrases"), list):
        return []
    return [p for p in controle["phrases"] if isinstance(p, dict) and p.get("verdict") == "contredite"]


def etat_de_publication(
    contenu: ContenuDuBillet, controle: Any, relecture: Any, valides: Sequence[bool]
) -> str:
    """
    « publiable », ou le PREMIER blocage, dans l'ordre où l'administrateur peut
    le lever : les sources (le contrôle juge contre elles, l'administrateur
    cite parmi elles), les citations, le contrôle, les phrases contredites (les
    corriger périme le contrôle), les phrases de règle à réécrire ou à
    confirmer (tour 7 ; les réécrire périme aussi le contrôle), la relecture.

    `valides` : pour chaque citation, vérifiée par le serveur ET toujours
    portée par la base (routes/blog._validite_des_citations). Un article sans
    aucune citation n'est pas publiable : un décryptage juridique s'appuie sur
    un texte de la base (revue du tour 2 : on pouvait retirer les citations
    d'un article d'avant et publier sa prose écrite de mémoire).
    """
    if not sources_valables(contenu.sources):
        return "sources_absentes"
    if not contenu.citations or len(valides) != len(contenu.citations) or not all(valides):
        return "citations_non_verifiees"
    if not controle_a_jour(contenu, controle):
        return "controle_a_relancer"
    if phrases_contredites(controle):
        return "phrases_contredites"
    if phrases_non_confirmees(contenu, controle, relecture):
        return "phrases_a_confirmer"
    if not all(parties_relues(contenu, controle, relecture).values()):
        return "relecture_incomplete"
    return "publiable"


def _en_liste(noms: Sequence[str]) -> str:
    return noms[0] if len(noms) == 1 else f"{', '.join(noms[:-1])} et {noms[-1]}"


def motif_du_refus(
    etat: str, contenu: ContenuDuBillet, controle: Any, relecture: Any, valides: Sequence[bool]
) -> str:
    """Le `detail` en français d'une publication refusée (422, code = l'état)."""
    if etat == "sources_absentes":
        return (
            "Publication refusée : l'article n'a pas les sources sur lesquelles il a été écrit "
            "(article d'avant le contrôle IA, ou écrit à la main). Reconstruisez-les, citez des "
            "articles parmi elles, relancez le contrôle IA, puis relisez chaque partie."
        )
    if etat == "citations_non_verifiees":
        if not contenu.citations:
            return (
                "Publication refusée : l'article ne cite aucun article de la base. Citez au moins "
                "un article, relancez le contrôle IA, puis relisez chaque partie."
            )
        rangs = [str(i) for i, v in enumerate(valides, 1) if not v] or ["?"]
        if len(rangs) == 1:
            quelles, verbe, pronom = f"la citation {rangs[0]}", "n'est pas vérifiée", "la"
        else:
            quelles, verbe, pronom = f"les citations {_en_liste(rangs)}", "ne sont pas vérifiées", "les"
        return (
            f"Publication refusée : {quelles} {verbe} dans la base (écrite avant la vérification "
            f"par le serveur, texte dépublié ou article modifié depuis). Retirez-{pronom} ou citez "
            f"à nouveau l'article, relancez le contrôle IA, puis relisez les explications."
        )
    if etat == "controle_a_relancer":
        if not isinstance(controle, dict):
            return (
                "Publication refusée : le contrôle IA n'a pas été fait sur cet article (il a "
                "échoué, l'article a été écrit à la main, ou ses sources ont été reconstruites). "
                "Lancez-le, puis relisez chaque partie."
            )
        avant = controle.get("empreintes_des_parties")
        aujourd_hui = empreintes_des_parties(contenu)
        changees = [
            NOMS_DES_PARTIES[p] for p in PARTIES
            if isinstance(avant, dict) and avant.get(p) != aujourd_hui[p]
        ]
        lesquelles = f" (parties modifiées : {_en_liste(changees)})" if changees else ""
        return (
            f"Publication refusée : le texte ou les sources ont changé depuis le contrôle "
            f"IA{lesquelles}. Relancez le contrôle, puis relisez les parties concernées."
        )
    if etat == "phrases_contredites":
        contredites = phrases_contredites(controle)
        exemples = " ; ".join(
            f"{NOMS_DES_PARTIES.get(PARTIE_RELUE.get(str(p.get('partie')), ''), 'texte')} : "
            f"« {couper(str(p.get('texte') or ''), 120)} »"
            for p in contredites[:3]
        )
        nombre = (
            "une phrase que les extraits contredisent" if len(contredites) == 1
            else f"{len(contredites)} phrases que les extraits contredisent"
        )
        return (
            f"Publication refusée : le contrôle IA a trouvé {nombre} ({exemples}). Corrigez-les, "
            f"relancez le contrôle, puis relisez les parties corrigées."
        )
    if etat == "phrases_a_confirmer":
        restantes = phrases_non_confirmees(contenu, controle, relecture)
        exemples = " ; ".join(
            f"{_id_de(p)}, {NOMS_DES_PARTIES.get(PARTIE_RELUE.get(str(p.get('partie')), ''), 'texte')} : "
            f"« {couper(str(p.get('texte') or ''), 120)} »"
            for p in restantes[:3]
        )
        if len(restantes) == 1:
            quoi = "une phrase de règle que le contrôle IA n'a pas trouvée dans les extraits n'est ni réécrite ni confirmée"
        else:
            quoi = (
                f"{len(restantes)} phrases de règle que le contrôle IA n'a pas trouvées dans les extraits "
                f"ne sont ni réécrites ni confirmées"
            )
        return (
            f"Publication refusée : {quoi} ({exemples}). Réécrivez-les et relancez le contrôle, ou "
            f"confirmez chacune après l'avoir vérifiée dans le texte (« J'ai vérifié cette phrase "
            f"dans le texte »)."
        )
    if etat == "relecture_incomplete":
        a_relire = [
            NOMS_DES_PARTIES[p]
            for p, relue in parties_relues(contenu, controle, relecture).items() if not relue
        ]
        return (
            f"Publication refusée : la relecture n'est pas complète. À relire contre les sources "
            f"et les verdicts du contrôle, puis à cocher : {_en_liste(a_relire)}."
        )
    return "Publication refusée."


def nouvelle_relecture(
    avant: ContenuDuBillet,
    apres: ContenuDuBillet,
    controle: Any,
    enregistree: Any,
    demandee: Optional[Dict[str, bool]],
    par: Optional[str],
    par_id: Optional[int],
    le: str,
    controle_relu: Optional[str] = None,
    confirmees: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    La relecture à enregistrer après une modification, de `avant` en `apres`
    (le contrôle enregistré, `controle`, ne change pas avec elle).

    - Une partie dont le texte CHANGE est décochée, même si la même requête la
      coche : la case a été cochée pour l'ancien texte (le front décoche, lui
      aussi, une partie modifiée avant de l'envoyer).
    - Les cases envoyées par l'administrateur (`demandee`) sont prises pour
      les autres parties : cochée, la partie garde l'empreinte de ce qu'elle
      atteste (empreintes_de_relecture : texte, sources, verdicts).
    - `confirmees` (avenant du tour 7) : la liste ENTIÈRE des phrases de règle
      que l'administrateur a confirmées (une phrase absente ne l'est plus) ;
      None : inchangées. Seules les phrases à confirmer du contrôle sont
      gardées, chacune avec l'empreinte de ce qu'elle atteste
      (empreintes_a_confirmer). Une phrase d'une partie dont le texte change
      perd sa confirmation, comme la case de sa partie : sa réécriture se fait
      juger par un nouveau contrôle.
    - `controle` (avenant du tour 5) : le contrôle contre lequel des cases
      viennent d'être cochées, ou des phrases confirmées (`controle_relu`, que
      la route a vérifié : relecture_perimee, confirmations_perimees) ; sinon,
      celui d'avant.
    - `par` et `le` sont notés par le serveur quand une case ou une
      confirmation change par le fait de l'administrateur ; ceux du corps sont
      ignorés.
    Rien à noter (aucune case, rien de coché, rien de confirmé) : {} (un
    article d'avant garde sa forme).
    """
    enregistree = enregistree if isinstance(enregistree, dict) else {}
    anciennes = enregistree.get("parties")
    anciennes = anciennes if isinstance(anciennes, dict) else {}
    textes_avant = empreintes_des_parties(avant)
    textes_apres = empreintes_des_parties(apres)
    modifiees = {p for p in PARTIES if textes_avant[p] != textes_apres[p]}
    attestees = empreintes_de_relecture(apres, controle)
    parties: Dict[str, Optional[str]] = {p: anciennes.get(p) for p in PARTIES}
    for partie in modifiees:
        parties[partie] = None
    cochees = {p: c for p, c in (demandee or {}).items() if p in PARTIES and p not in modifiees}
    for partie, cochee in cochees.items():
        parties[partie] = attestees[partie] if cochee else None

    partie_de = {_id_de(p): PARTIE_RELUE.get(str(p.get("partie"))) for p in phrases_a_confirmer(controle)}
    confirmations_avant = _confirmations(enregistree)
    if confirmees is None:
        confirmations = {
            i: e for i, e in confirmations_avant.items() if partie_de.get(i) not in modifiees
        }
    else:
        a_confirmer = empreintes_a_confirmer(apres, controle)
        confirmations = {
            i: a_confirmer[i] for i in dict.fromkeys(confirmees)
            if i in a_confirmer and partie_de.get(i) not in modifiees
        }

    nouvelle: Dict[str, Any] = {
        "parties": parties,
        "par": enregistree.get("par"), "par_id": enregistree.get("par_id"),
        "le": enregistree.get("le"),
    }
    atteste = any(c is True for c in cochees.values()) or bool(confirmees is not None and confirmations)
    relu = controle_relu if controle_relu and atteste else None
    # Une relecture d'avant le tour 5 garde sa forme tant que rien n'est coché.
    if relu or enregistree.get("controle"):
        nouvelle["controle"] = relu or enregistree.get("controle")
    # Et une relecture d'avant le tour 7, tant que rien n'est confirmé.
    if confirmations or "phrases_confirmees" in enregistree:
        nouvelle["phrases_confirmees"] = confirmations
    relues_avant = parties_relues(avant, controle, enregistree)
    relues_apres = parties_relues(apres, controle, nouvelle)
    confirmations_changees = confirmees is not None and (
        phrases_confirmees(avant, controle, enregistree) != phrases_confirmees(apres, controle, nouvelle)
    )
    if any(relues_avant[p] != relues_apres[p] for p in cochees) or confirmations_changees:
        nouvelle.update(par=par, par_id=par_id, le=le)
    if not enregistree and not any(parties.values()) and not confirmations and nouvelle["le"] is None:
        return {}
    return nouvelle
