"""
Routes du blog juridique : des décryptages d'actualité rédigés par le modèle à
partir des textes de la base, CONTRÔLÉS phrase à phrase par le modèle et RELUS
partie par partie par un administrateur avant publication.

Public :
- GET  /api/v1/blog            les articles PUBLIÉS ET PUBLIABLES (rubrique facultative)
- GET  /api/v1/blog/{slug}     un article publié et publiable ; 404 sinon
Administration (compte admin ou superadmin) :
- POST   /api/v1/blog/generate        rédige un article, le CONTRÔLE, l'enregistre en BROUILLON (201)
- POST   /api/v1/blog/{id}/controle   (re)lance le contrôle IA sur le texte actuel
- POST   /api/v1/blog/{id}/sources    reconstruit les sources (article d'avant, écrit à la main)
- GET    /api/v1/blog/admin/all       tous les articles, brouillons compris
- GET    /api/v1/blog/admin/{id}      un article, brouillon compris
- POST   /api/v1/blog                 crée un brouillon à la main (201)
- PUT    /api/v1/blog/{id}            modifie, relit, publie ou dépublie
- DELETE /api/v1/blog/{id}            supprime (204)

TOUTE ERREUR rend `{"detail": "<message en français>", "code": "<code>"}`,
depuis la route elle-même (RouteDuBlog), donc DANS le middleware CORS.
POURQUOI : une exception non rattrapée (quota du modèle, saturation, conflit
d'adresse) remontait en 500 par ServerErrorMiddleware, EN DEHORS du CORS : le
navigateur bloquait la réponse et l'administrateur lisait « Failed to fetch »
(essai réel du 29/10/2026, clé refusée). Et une 422 de FastAPI porte un
`detail` en TABLEAU, que le front affichait « [object Object] ».

Codes : entree_invalide (422), sans_source (422), sources_absentes (422,
contrôle d'un article sans sources), relecture_perimee (422, avenants des
tours 5 et 7), introuvable (404), slug_pris (409), article_en_ligne (409,
avenant du tour 5), quota (429), saturation (503), recherche_indisponible
(503), reponse_invalide (502), delai (504), erreur_interne (500) ; et, pour
une publication refusée (422), l'état qui la bloque : sources_absentes,
citations_non_verifiees, controle_a_relancer, phrases_contredites,
phrases_a_confirmer (avenant du tour 7), relecture_incomplete. 401 et 403
(authentification) gardent la forme habituelle de FastAPI. Retry-After (429,
503) est lisible par le front (app/main.py, expose_headers).

CONTRAT DES CITATIONS (tour 2, des deux côtés) :
- toute citation de toute réponse porte `verifie` (booléen) : vrai seulement
  pour une citation construite par le serveur ET que la base porte encore
  (texte publié, article présent, extrait inchangé) ;
- dans les réponses PUBLIQUES, une citation dont verifie est faux n'a ni
  `excerpt`, ni `url`, ni explication (défense de plus : depuis le tour 3, un
  article qui en a une n'est plus servi) ; une citation vérifiée n'y montre
  pas `law_reference`, la cote interne ;
- `suite` (booléen) : l'extrait est la suite d'un article, pas son début ; il
  commence alors par « […] ». À afficher « (suite) » à côté du numéro.

CONTRAT DU CONTRÔLE ET DE LA RELECTURE (tour 3, 30/10/2026, des deux côtés) :
- tout article de toute réponse d'administration porte (de toute réponse
  jusqu'à l'avenant du tour 7) :
  - `controle_ia` : null, ou {fait_le, modele, empreinte, phrases: [{id,
    partie, index, texte, verdict, extraits, raison}]} ; `partie` vaut
    titre, contexte, resume, point_cle, impact, explication ou question ;
    `index` est le rang, à partir de 0, du point clé ou de la citation (0 pour
    une partie d'un seul texte) ; `verdict` vaut soutenue, non_soutenue,
    contredite ou hors_sources ; `extraits` sont des numéros d'extraits (ceux
    des `sources`, puis ceux des citations ajoutées depuis) ;
  - `relecture` : {parties: {titre, contexte, resume, points_cles, impact,
    explications, question: booléen}, par, le}. Une partie n'est cochée que
    si son texte, les sources et ses verdicts n'ont pas changé depuis ; une
    partie vide n'a rien à relire et l'est toujours. Dans une réponse
    PUBLIQUE, `par` vaut null ;
  - `etat_publication` : publiable, sources_absentes, citations_non_verifiees,
    controle_a_relancer, phrases_contredites ou relecture_incomplete (le
    premier blocage, dans cet ordre).
- le public ne voit que les articles publiés ET publiables : un article
  d'avant (citations jamais vérifiées, lignes du 26/10 de jurix_dev) n'est
  jamais servi (404), mais reste visible et corrigeable dans l'administration ;
- POST /generate contrôle l'article juste après l'avoir enregistré ; si le
  contrôle échoue, le brouillon reste (controle_ia null) et `avertissements`
  dit de relancer le contrôle ;
- PUT : exactement {"is_published": false} retire aussitôt l'article du site,
  toujours. Tout autre corps est validé EN ENTIER, puis appliqué d'un bloc,
  ou pas du tout : un refus ne retire rien du site. Publier (ou rester en
  ligne) exige l'état « publiable », sinon 422 avec code = l'état. `relecture`
  ({parties: {...}}) porte les cases de l'administrateur ; le serveur note
  `par` et `le`.

AVENANT DU TOUR 4 (des deux côtés) :
- le TITRE est une partie comme les autres : « titre » dans les `partie` de
  controle_ia.phrases (en premier) et dans relecture.parties. Changer le titre
  d'un article en ligne périme son contrôle, comme toute partie relue ;
- les SOURCES sont gardées avec l'article (dans la colonne JSON, sans
  colonne nouvelle) et rendues par TOUTE réponse d'administration (generate,
  admin/all, admin/{id}, POST, PUT, /controle, /sources), jamais par le public :
  `sources` : [{n, law_id, law_title, article_number, article_id, suite,
  extrait}], puis les extraits que le contrôle a ajoutés pour les citations
  ajoutées depuis (mêmes numéros que ses verdicts) ;
- un article sans sources (article d'avant, ou écrit à la main) est
  « sources_absentes », le PREMIER blocage ; POST /{id}/controle y répond 422
  sources_absentes. POST /{id}/sources les reconstruit par la recherche de la
  génération (titre et contexte), sans modèle : 200 {post, sources,
  avertissements}. Une citation n'est gardée que si elle est vérifiée et
  qu'une source reconstruite la porte (relue alors en base) ; les autres sont
  retirées, chacune avec un avertissement. Si les sources changent, le
  contrôle d'avant est effacé (ses numéros d'extraits ne valent plus).
  L'article repasse en BROUILLON : il ne revient en ligne que par « Publier »,
  après citations, contrôle et relecture. Erreurs : 404, 422 sans_source,
  503 recherche_indisponible, 504 delai ; rien n'est changé ;
- une case de relecture atteste le texte de sa partie, les sources ET les
  verdicts que le contrôle donnait à cette partie : cochée avant le contrôle,
  elle ne compte plus après ; un nouveau contrôle qui juge autrement une
  partie la décoche ; reconstruire les sources décoche tout ;
- le contrôle vérifie le passage que le modèle dit avoir lu : une phrase
  « soutenue » ou « contredite » sans passage réel des extraits devient
  « non_soutenue », et sa raison commence par « À vérifier : » ; le passage
  trouvé est cité à la fin de la raison, morceau par morceau, chacun avec
  son extrait (« — Extrait 5 : « … » « … » ; extrait 10 : « … » »), et ces
  extraits viennent en tête de `extraits`.

AVENANT DU TOUR 5 (des deux côtés) :
- un PUT dont `relecture.parties` coche une partie porte `relecture.controle`,
  le `controle_ia.fait_le` du contrôle que l'administrateur a relu. Absent, ou
  autre que celui du contrôle enregistré (ou aucun contrôle enregistré) : 422
  `relecture_perimee`, et RIEN n'est appliqué. Décocher n'en demande pas.
  Toute réponse rend `relecture.controle` : le `fait_le` contre lequel les
  cases enregistrées ont été cochées, ou null. Chaque contrôle a son propre
  `fait_le` (fait_le_apres) ;
- POST /{id}/controle d'un article PUBLIÉ répond 409 `article_en_ligne`, sans
  appeler le modèle, sauf si le corps est {"confirmer": true} : un nouveau
  contrôle peut juger autrement une partie, qui est alors à relire, et
  l'article n'est plus servi au public jusqu'à la relecture. Un article sans
  sources répond d'abord 422 `sources_absentes`. Publié PENDANT le contrôle
  (un autre onglet), rien n'est enregistré : 409 `article_en_ligne` aussi ;
- une case qu'un nouveau contrôle fait tomber est EFFACÉE (revue du front du
  tour 5) : un contrôle suivant qui redonne les verdicts d'avant ne la fait
  pas revivre, et l'article qu'un contrôle a caché ne revient au public
  qu'une fois la partie recochée par une personne, contre le contrôle
  enregistré ;
- le contrôle juge l'article en appels d'au plus 20 phrases, sans couper une
  partie : au-delà du plafond, 422 `entree_invalide` (« raccourcissez ») ;
  depuis le tour 6, les appels se suivent et le plafond est de 25 phrases
  (voir l'avenant du tour 6) ;
- le passage cité à la fin d'une raison est fait des mots DE L'EXTRAIT, jamais
  de ceux du modèle ; un morceau copié dont un mot qui compte n'est pas dans
  l'extrait (son premier et son dernier compris), un tronçon court après
  « … » qu'on ne retrouve pas juste après le précédent, DANS SA PHRASE, ou une
  coupure « … » qui saute une négation ou une exception de l'extrait (« ne »,
  « pas », « sans », « sauf », « à moins », le « que » d'un « ne … que »…),
  rabat la phrase en « non_soutenue » (« À vérifier : … », qui dit ce que la
  coupure saute). Une citation ne joint jamais deux phrases : un morceau qui
  en couvre deux, ou cité dans un autre ordre que celui de l'extrait, est
  cité en plusieurs « … », dans l'ordre du modèle (« — Extrait 6 : « … » « … » ») ;
- une raison ne renvoie qu'à des extraits qui existent : « l'extrait 36 »
  (l'article 36) devient « l'extrait 4 (article 36) ».

AVENANT DU TOUR 6 (le modèle du contrôle, décision du 04/11) :
- le contrôle, après /generate comme à la demande, juge avec son propre
  modèle, BLOG_CONTROLE_MODELE (mistral-large-4) ; la génération garde
  MISTRAL_MODEL. `controle_ia.modele` est le modèle qui a VRAIMENT répondu
  (le champ « model » de sa réponse), jamais un autre sous son nom : le
  contrôle n'a pas de modèle de secours ;
- ses appels se suivent, et un seul contrôle par modèle à la fois dans le
  processus (20 000 jetons par minute pour mistral-large-4). Un contrôle
  lancé pendant un autre l'attend 20 s au plus, puis 503 `saturation` avec
  Retry-After (60) ; un 429 du fournisseur, ou le budget de jetons épuisé :
  503 `saturation` avec le délai connu en Retry-After. Rien n'est enregistré ;
- 25 phrases au plus (un appel, 140 s au pire, pour 180 s) : au-delà, 422
  `entree_invalide` (« raccourcissez-le »).

REVUE DU TOUR 6 (le budget de jetons de mistral-large-4) :
- le contrôle se PLANIFIE avant son premier lot : attente d'un autre
  contrôle, attentes du budget de jetons entre ses lots et durée de ses lots
  au pire mesuré, comptées ensemble depuis l'entrée dans la route (le délai
  de la route, 180 s, ou ce que /generate laisse). Un plan qui ne finit pas
  10 s avant l'échéance est refusé AVANT tout envoi : 503 `saturation`, un
  message sur le budget de jetons de l'article, et Retry-After = quand il
  tiendra (pas de Retry-After si une relance tient tout de suite) ; relancé
  alors, il aboutit. Sinon chaque lot attend ce que le plan lui laisse, et
  non plus les 20 s du chat : le second lot d'un gros article attend que le
  premier sorte de la minute (le 503 de chaque relance, lot 1 payé, est fini) ;
- un contrôle lancé pendant un autre n'attend que si son plan tient APRÈS
  lui ; sinon 503 aussitôt, avec le Retry-After où il tiendra ;
- un article dont le plan ne tient pas, même seul (des lots mal répartis et
  de gros extraits) : 422 `entree_invalide` (« raccourcissez-le »).

AVENANT DU TOUR 7 (des deux côtés ; essai réel du 04/11 et décision du
propriétaire, « Confirmer phrase par phrase ») :
- UN APPEL PAR ARTICLE pour le modèle du contrôle (budget de jetons) : 25
  phrases au plus, en un appel ; le plan compte un appel au pire mesuré (140 s,
  plus 10 s de marge), et refuse avant tout envoi ce qui ne tient pas (503
  `saturation` et Retry-After, comme au tour 6 ; au-delà de 25 phrases, 422
  `entree_invalide`). Délais : POST /{id}/controle répond en DELAI_CONTROLE_S
  (180 s) au plus ; POST /generate en DELAI_DE_LA_REDACTION_S (205 s) au plus
  quand le contrôle suit (génération et contrôle), ou 504 à DELAI_GENERATION_S
  (150 s) de génération ;
- `controle_ia.phrases[].nature` : « regle » ou « autre » ;
- une phrase de règle (nature « regle ») que le contrôle juge `non_soutenue`
  (« À vérifier » compris) ou `hors_sources` est À CONFIRMER : réécrite (un
  nouveau contrôle la juge), ou confirmée par l'administrateur. Le PUT peut
  porter `relecture.phrases_confirmees`, la liste ENTIÈRE des `id`
  (controle_ia.phrases) confirmés, TOUJOURS avec `relecture.controle` (le
  `fait_le` du contrôle relu, comme au tour 5). Un `id` que le contrôle
  enregistré n'a pas, un autre contrôle, ou aucun : 422 `relecture_perimee`,
  RIEN n'est appliqué. Un `id` connu qui n'est pas à confirmer n'est pas
  gardé. Les réponses d'administration rendent `relecture.phrases_confirmees`
  (les confirmations qui comptent, dans l'ordre de l'article) ;
- `etat_publication` « phrases_a_confirmer », entre `phrases_contredites` et
  `relecture_incomplete` : publier exige chaque phrase à confirmer réécrite
  ou confirmée. Un nouveau contrôle fait tomber une confirmation EXACTEMENT
  comme une case (elle ne reste que si la phrase, son verdict et les sources
  sont les mêmes) ; une phrase d'une partie modifiée perd la sienne ;
- les réponses PUBLIQUES (GET /blog, GET /blog/{slug}) ne portent plus ni
  `controle_ia`, ni `relecture`, ni `etat_publication` : la route publique ne
  sert que des articles publiés ET publiables ;
- un BROUILLON JAMAIS PUBLIÉ dont l'adresse est encore celle que le serveur
  a tirée de son titre prend, quand le titre change, l'adresse du nouveau
  titre (libre, comme à la génération) ; une adresse écrite à la main est
  gardée.
"""

import asyncio
import logging
import re
import time
from dataclasses import replace
from typing import Annotated, Any, Callable, Coroutine, Dict, List, Optional, Sequence, Set, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import desc, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.auth import get_current_admin_user
from app.core.database import get_db
from app.models.blog import BlogPost
from app.models.user import User
from app.services.article_reference import normalize_number
from app.services.blog_generator_service import (
    DELAI_GENERATION_S,
    SLUG_LONGUEUR_MAX,
    Avertissement,
    BlogGeneratorService,
    CitationInvalide,
    DemandeDeCitation,
    ErreurDeGeneration,
    avertissements_du_billet,
    citations_verifiees,
    duree_de_lecture,
    fusionner,
    mots_cles_propres,
    prose_publique,
    slugify,
    sujet_de_l_article,
    texte_brut,
)
from app.services.controle_du_billet import (
    DELAI_CONTROLE_S,
    PARTIES,
    PHRASES_AU_PLUS,
    ContenuDuBillet,
    ControleDuBillet,
    cle_d_article,
    confirmations_perimees,
    controle_affiche,
    etat_de_publication,
    fait_le_apres,
    maintenant_iso,
    memes_sources,
    motif_du_refus,
    nom_d_article,
    nouvelle_relecture,
    relecture_affichee,
    relecture_apres_un_controle,
    relecture_perimee,
    sources_affichees,
    sources_valables,
)

logger = logging.getLogger(__name__)


# ==================== ERREURS ====================


class ErreurDuBlog(HTTPException):
    """Une réponse d'erreur du blog : un statut, un code stable, un message en français."""

    def __init__(
        self, status_code: int, code: str, detail: str, retry_after: Optional[int] = None
    ):
        super().__init__(
            status_code=status_code,
            detail=detail,
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )
        self.code = code


def _reponse_d_erreur(
    statut: int, code: str, detail: str, headers: Optional[Dict[str, str]] = None
) -> JSONResponse:
    return JSONResponse(status_code=statut, content={"detail": detail, "code": code}, headers=headers)


# Les champs, nommés comme dans l'administration.
_NOMS_DES_CHAMPS = {
    "topic": "sujet", "category": "rubrique", "additional_context": "contexte fourni",
    "title": "titre", "slug": "adresse (slug)", "reading_time": "durée de lecture",
    "context": "contexte", "bot_analysis": "analyse", "summary": "résumé",
    "key_points": "points clés", "keyPoints": "points clés", "citations": "citations",
    "citizen_impact": "impact pour le citoyen", "citizenImpact": "impact pour le citoyen",
    "suggested_question": "question suggérée", "tags": "mots-clés",
    "is_published": "publication", "law_id": "texte", "lawId": "texte",
    "article_id": "article", "articleId": "article", "article_number": "numéro d'article",
    "articleNumber": "numéro d'article", "limit": "nombre d'articles", "offset": "décalage",
    "post_id": "identifiant de l'article", "relecture": "relecture", "parties": "parties",
    "controle": "contrôle relu", "phrases_confirmees": "phrases confirmées",
    "phrasesConfirmees": "phrases confirmées",
}


def _raison(erreur: Dict[str, Any]) -> str:
    """La raison d'une erreur de validation de Pydantic, en français."""
    genre = erreur.get("type", "")
    ctx = erreur.get("ctx") or {}
    if genre == "missing":
        return "champ obligatoire"
    if genre == "string_too_short":
        return f"{ctx.get('min_length')} caractère(s) au minimum"
    if genre == "string_too_long":
        return f"{ctx.get('max_length')} caractères au maximum"
    if genre == "too_short":
        return f"{ctx.get('min_length')} élément(s) au minimum"
    if genre == "too_long":
        return f"{ctx.get('max_length')} éléments au maximum"
    if genre in ("string_type", "string_unicode"):
        return "texte attendu"
    if genre in ("int_parsing", "int_type", "int_from_float"):
        return "nombre entier attendu"
    if genre in ("bool_parsing", "bool_type"):
        return "vrai ou faux attendu"
    if genre == "greater_than":
        return f"doit être supérieur à {ctx.get('gt')}"
    if genre == "greater_than_equal":
        return f"doit valoir au moins {ctx.get('ge')}"
    if genre == "less_than":
        return f"doit être inférieur à {ctx.get('lt')}"
    if genre == "less_than_equal":
        return f"doit valoir au plus {ctx.get('le')}"
    if genre == "list_type":
        return "liste attendue"
    if genre in ("dict_type", "model_type", "model_attributes_type"):
        return "objet attendu"
    if genre == "json_invalid":
        return "JSON illisible"
    if genre == "value_error":
        return str(erreur.get("msg", "")).removeprefix("Value error, ")
    return str(erreur.get("msg", "valeur invalide"))


def message_de_validation(erreurs: Sequence[Dict[str, Any]]) -> str:
    """Le `detail` d'une 422 : une chaîne lisible, et non le tableau de FastAPI."""
    morceaux = []
    for erreur in list(erreurs)[:5]:
        chemin = [
            _NOMS_DES_CHAMPS.get(str(p), f"n° {p + 1}" if isinstance(p, int) else str(p))
            for p in erreur.get("loc", ())
            if p not in ("body", "query", "path")
        ]
        morceaux.append(f"{' › '.join(chemin) or 'requête'} : {_raison(erreur)}")
    reste = len(erreurs) - len(morceaux)
    suite = f" ; et {reste} autre(s) erreur(s)" if reste > 0 else ""
    return f"Requête invalide — {' ; '.join(morceaux)}{suite}."


class RouteDuBlog(APIRoute):
    """
    Les routes du blog, avec leurs erreurs mises en forme par la route
    elle-même : la réponse passe alors par CORS, comme une réponse normale.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        traiter = super().get_route_handler()

        async def route(request: Request) -> Response:
            try:
                return await traiter(request)
            except ErreurDuBlog as e:
                return _reponse_d_erreur(e.status_code, e.code, e.detail, e.headers)
            except RequestValidationError as e:
                return _reponse_d_erreur(422, "entree_invalide", message_de_validation(e.errors()))
            except StarletteHTTPException:
                # 401 et 403 de l'authentification : la réponse habituelle.
                raise
            except Exception:
                # Jamais le message : il peut porter une requête SQL ou une clé.
                logger.exception(f"❌ Blog : erreur inattendue ({request.method} {request.url.path})")
                return _reponse_d_erreur(
                    500, "erreur_interne",
                    "Erreur interne du serveur. Réessayez ; si elle persiste, signalez-la.",
                )

        return route


router = APIRouter(route_class=RouteDuBlog)


# ==================== SCHÉMAS ====================


def _prose(valeur: str) -> str:
    """
    Texte brut (sans les « ** » que le front afficherait tels quels), sans
    renvoi « (extrait 2) » que le lecteur ne peut pas suivre, jamais vide.
    """
    propre = prose_publique(valeur)
    if not propre:
        raise ValueError("ne peut pas être vide")
    return propre


Prose = Annotated[str, AfterValidator(_prose)]
# La rédaction est libre, mais bornée : sans limite, un corps de plusieurs
# mégaoctets passait jusqu'à la base.
LONGUEUR_D_UN_TEXTE = 20_000


class _Entree(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)


class BlogGenerateRequest(_Entree):
    """
    `topic` est la requête de recherche : 300 caractères au plus. Sans borne,
    un sujet de plus de 500 caractères faisait refuser la requête de recherche,
    l'erreur était avalée, et le modèle écrivait sans aucune source.
    `additional_context` va au modèle, jamais à la recherche.
    """

    topic: str = Field(..., min_length=3, max_length=300)
    category: Optional[str] = Field(None, max_length=100)
    additional_context: Optional[str] = Field(None, max_length=4000)


class CitationIn(_Entree):
    """
    Une citation envoyée par l'administration : un texte, un article (par son
    identifiant ou son numéro), une explication. Le reste (intitulé du texte,
    extrait, lien) est relu en base par le serveur : les autres clés envoyées
    (`reference`, `excerpt`, `url`, `verifie`...) sont ignorées. Une citation
    non vérifiée renvoyée telle quelle le reste (_citations_envoyees).
    """

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True, extra="ignore")

    law_id: int = Field(..., gt=0, validation_alias=AliasChoices("law_id", "lawId"))
    article_id: Optional[int] = Field(
        None, gt=0, validation_alias=AliasChoices("article_id", "articleId")
    )
    article_number: Optional[str] = Field(
        None, max_length=64, validation_alias=AliasChoices("article_number", "articleNumber")
    )
    title: Optional[str] = Field(None, max_length=500)
    # La SUITE d'un article plutôt que son début, quand une même ligne porte
    # les deux (citation renvoyée telle que le serveur l'a écrite).
    suite: Optional[bool] = None

    @model_validator(mode="after")
    def _un_article(self) -> "CitationIn":
        if not self.article_id and not (self.article_number or "").strip():
            raise ValueError("indiquez l'article cité (article_id ou article_number)")
        return self

    def demande(self) -> DemandeDeCitation:
        return DemandeDeCitation(
            self.law_id, self.article_id, self.article_number, self.title or "", suite=self.suite
        )


def _points(valeur: Optional[List[str]]) -> Optional[List[str]]:
    """
    Les points clés en texte brut, sans renvoi aux extraits, sans les lignes
    vides, un point par ligne : le formulaire en fait une ligne par point.
    """
    if valeur is None:
        return None
    points = [" ".join(prose_publique(p).split()) for p in valeur]
    if any(len(p) > 1000 for p in points):
        raise ValueError("1000 caractères au maximum par point")
    return [p for p in points if p]


class BotAnalysisIn(_Entree):
    summary: Prose = Field(..., max_length=LONGUEUR_D_UN_TEXTE)
    key_points: List[str] = Field(
        default_factory=list, max_length=12, validation_alias=AliasChoices("key_points", "keyPoints")
    )
    citations: List[CitationIn] = Field(default_factory=list, max_length=20)
    citizen_impact: Prose = Field(
        ..., max_length=LONGUEUR_D_UN_TEXTE,
        validation_alias=AliasChoices("citizen_impact", "citizenImpact"),
    )

    _points_propres = field_validator("key_points")(_points)


class BotAnalysisUpdate(_Entree):
    """Chaque partie est facultative : l'administration renvoie ce qu'elle modifie."""

    summary: Optional[Prose] = Field(None, max_length=LONGUEUR_D_UN_TEXTE)
    key_points: Optional[List[str]] = Field(
        None, max_length=12, validation_alias=AliasChoices("key_points", "keyPoints")
    )
    # Absent : les citations ne changent pas. Une liste (même vide) les remplace.
    citations: Optional[List[CitationIn]] = Field(None, max_length=20)
    citizen_impact: Optional[Prose] = Field(
        None, max_length=LONGUEUR_D_UN_TEXTE,
        validation_alias=AliasChoices("citizen_impact", "citizenImpact"),
    )

    _points_propres = field_validator("key_points")(_points)


# L'identifiant d'une phrase du contrôle (« P12 »), tel que controle_ia le rend.
IdDePhrase = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1, max_length=16)]


class RelectureIn(BaseModel):
    """
    Les cases de relecture de l'administrateur : {"parties": {"resume": true,
    …}, "controle": "<fait_le du contrôle relu>", "phrases_confirmees":
    ["P5", …]}. Les parties absentes ne changent pas. `controle` est exigé dès
    qu'une partie est cochée (avenant du tour 5 : relecture_perimee), et avec
    toute liste de phrases confirmées (avenant du tour 7 :
    confirmations_perimees). `phrases_confirmees` : la liste ENTIÈRE des
    phrases de règle confirmées ; absente, elles ne changent pas. `par` et
    `le` envoyés sont ignorés : le serveur les note.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    parties: Dict[str, StrictBool] = Field(default_factory=dict)
    controle: Optional[str] = Field(None, max_length=64)
    phrases_confirmees: Optional[List[IdDePhrase]] = Field(
        None, max_length=4 * PHRASES_AU_PLUS,
        validation_alias=AliasChoices("phrases_confirmees", "phrasesConfirmees"),
    )

    @field_validator("parties", mode="before")
    @classmethod
    def _noms_du_contrat(cls, valeur: Any) -> Any:
        if isinstance(valeur, dict) and "pointsCles" in valeur and "points_cles" not in valeur:
            valeur = {("points_cles" if k == "pointsCles" else k): v for k, v in valeur.items()}
        return valeur

    @field_validator("parties")
    @classmethod
    def _parties_connues(cls, valeur: Dict[str, bool]) -> Dict[str, bool]:
        inconnues = sorted(set(valeur) - set(PARTIES))
        if inconnues:
            raise ValueError(
                f"partie(s) inconnue(s) : {', '.join(inconnues)} (attendues : {', '.join(PARTIES)})"
            )
        return valeur


class BlogPostCreate(_Entree):
    """
    Un brouillon écrit à la main. `is_published` est IGNORÉ : tout article naît
    brouillon, et seule la modification (PUT is_published=true) le publie.
    """

    title: Prose = Field(..., max_length=500)
    slug: Optional[str] = Field(None, max_length=255)
    category: Prose = Field(..., max_length=100)
    reading_time: Optional[str] = Field(None, max_length=50)
    context: Prose = Field(..., max_length=LONGUEUR_D_UN_TEXTE)
    bot_analysis: BotAnalysisIn
    suggested_question: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list, max_length=20)
    is_published: Optional[bool] = None


class BlogPostUpdate(_Entree):
    """Chaque champ est facultatif ; `null` vaut « inchangé », sauf pour la question suggérée."""

    title: Optional[Prose] = Field(None, max_length=500)
    slug: Optional[str] = Field(None, max_length=255)
    category: Optional[Prose] = Field(None, max_length=100)
    reading_time: Optional[str] = Field(None, max_length=50)
    context: Optional[Prose] = Field(None, max_length=LONGUEUR_D_UN_TEXTE)
    bot_analysis: Optional[BotAnalysisUpdate] = None
    suggested_question: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = Field(None, max_length=20)
    relecture: Optional[RelectureIn] = None
    is_published: Optional[bool] = None


# ==================== OUTILS ====================

# La génération et le contrôle qui la suit, ensemble, quand c'est possible
# (le front abandonne à 215 s) ; le contrôle garde au moins de quoi juger un
# article court. TOUR 7 (mistral-large-4, UN appel) : une génération de 15 à
# 20 s (essai réel du 04/11 : brouillon enregistré 18,0 et 20,7 s après le
# clic) laisse au contrôle tout DELAI_CONTROLE_S, soit son appel au pire
# mesuré et sa marge (150 s) et 30 s d'attente ; une génération de plus de 55 s
# lui en laisse moins de 150 (plafond relevé à 15 500 jetons le 08/11 :
# 180 + 25 s ; sous les 300 s d'inactivité de Railway, avec 85 s de marge) : le plan le refuse avant tout envoi, et le
# brouillon reste à contrôler (rien n'est payé pour rien). Le minimum de 30 s
# ne vaut que pour un modèle sans plan (doublure, Gemini, ministral).
DELAI_DE_LA_REDACTION_S = 205
DELAI_CONTROLE_APRES_GENERATION_AU_MOINS_S = 30


def delai_du_controle_apres_la_generation(ecoule: float) -> float:
    """Ce que /generate laisse au contrôle, `ecoule` secondes après le début de la requête."""
    reste = DELAI_DE_LA_REDACTION_S - ecoule
    return min(DELAI_CONTROLE_S, max(DELAI_CONTROLE_APRES_GENERATION_AU_MOINS_S, reste))


# La reconstruction des sources : la recherche seule, sans modèle.
DELAI_SOURCES_S = 60


def get_blog_generator(db: AsyncSession = Depends(get_db)) -> BlogGeneratorService:
    """Le service de génération, sur la session de la requête (les tests le doublent)."""
    return BlogGeneratorService(db)


def get_controleur(db: AsyncSession = Depends(get_db)) -> ControleDuBillet:
    """Le contrôle IA, sur la session de la requête (les tests le doublent)."""
    return ControleDuBillet(db)


# Ce que la base dit AUJOURD'HUI des citations vérifiées : le texte est-il
# toujours publié, l'article toujours dans ce texte, et son contenu porte-t-il
# toujours l'extrait affiché ?
_SQL_TEXTES_PUBLIES = text("SELECT id FROM laws WHERE id = ANY(:ids) AND status = 'published'")
_SQL_CONTENUS = text("SELECT id, law_id, content FROM articles WHERE id = ANY(:ids)")
_COUPE = "[…]"


def _extrait_present(extrait: str, contenu: Optional[str]) -> bool:
    """L'extrait (sans ses « […] ») est-il toujours, mot pour mot, dans le contenu ?"""
    coeur = extrait.strip()
    if coeur.startswith(_COUPE):
        coeur = coeur[len(_COUPE):].strip()
    if coeur.endswith(_COUPE):
        coeur = coeur[:-len(_COUPE)].strip()
    return bool(coeur) and coeur in " ".join((contenu or "").split())


async def _validite_des_citations(
    db: AsyncSession, listes: Sequence[Sequence[Dict[str, Any]]]
) -> List[List[bool]]:
    """
    Pour chaque liste de citations, et chaque citation : est-elle encore une
    citation VÉRIFIÉE ? Vérifiée par le serveur ET, aujourd'hui, texte publié,
    article dans ce texte, extrait toujours dans son contenu.

    POURQUOI. Revue du tour 1 : une citation dont le texte était dépublié APRÈS
    la publication de l'article restait servie au public comme vérifiée,
    jusqu'à la modification suivante. La vérification était celle d'un jour ;
    le lecteur lit aujourd'hui. Deux requêtes pour toute une liste.
    """
    textes: Set[int] = set()
    lignes: Set[int] = set()
    for citations in listes:
        for citation in citations:
            if isinstance(citation, dict) and citation.get("verifie") is True:
                try:
                    textes.add(int(citation["law_id"]))
                    lignes.add(int(citation["article_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
    publies = {
        ligne.id for ligne in (await db.execute(_SQL_TEXTES_PUBLIES, {"ids": sorted(textes)})).all()
    } if textes else set()
    contenus = {
        ligne.id: ligne for ligne in (await db.execute(_SQL_CONTENUS, {"ids": sorted(lignes)})).all()
    } if lignes else {}

    def valide(citation: Dict[str, Any]) -> bool:
        if citation.get("verifie") is not True:
            return False
        try:
            law_id, article_id = int(citation["law_id"]), int(citation["article_id"])
        except (KeyError, TypeError, ValueError):
            return False
        ligne = contenus.get(article_id)
        return (
            law_id in publies and ligne is not None and ligne.law_id == law_id
            and _extrait_present(str(citation.get("excerpt") or ""), ligne.content)
        )

    return [[valide(c) for c in citations if isinstance(c, dict)] for citations in listes]


SANS_SOURCES = (
    "Sources absentes : l'article n'a pas les extraits sur lesquels il a été écrit (article "
    "d'avant le contrôle IA, ou écrit à la main). Reconstruisez ses sources, citez des articles "
    "parmi elles, puis lancez le contrôle IA et relisez chaque partie : il n'est pas publiable "
    "avant."
)


def _avertissements_d_etat(post: BlogPost, valides: Sequence[bool]) -> List[Avertissement]:
    """L'absence de sources, et les citations vérifiées que la base ne porte plus telles quelles."""
    avertissements = []
    if not sources_valables(post.sources):
        avertissements.append(Avertissement(None, None, SANS_SOURCES))
    for rang, (citation, valide) in enumerate(zip(post.citations, valides), 1):
        if citation.get("verifie") is True and not valide:
            avertissements.append(Avertissement(
                f"Citation {rang}", None,
                f"Citation {rang} : son texte n'est plus publié, ou l'article a changé dans la "
                f"base depuis sa vérification. L'article n'est plus publiable tant qu'elle y "
                f"est : retirez-la, ou citez à nouveau l'article.",
            ))
    return avertissements


def _etat(post: BlogPost, valides: Sequence[bool]) -> str:
    return etat_de_publication(ContenuDuBillet.de(post), post.controle_ia, post.relecture, valides)


async def _posts_json(
    db: AsyncSession,
    posts: Sequence[BlogPost],
    admin: bool,
    du_billet: Optional[List[Avertissement]] = None,
    valides: Optional[Dict[int, List[bool]]] = None,
) -> List[Dict[str, Any]]:
    """
    La forme du front (BlogPost.to_dict), les citations jugées sur la base
    d'aujourd'hui. Routes publiques : rien d'autre (une citation non vérifiée
    n'y garde que sa référence et son numéro). Administration : plus
    `controle_ia`, `relecture`, `etat_publication`, les `sources` (tour 4) et
    les `avertissements` de l'article tel qu'il est, recalculés à chaque
    lecture.

    POURQUOI rien de la relecture au public (avenant du tour 7) : les réponses
    publiques rendaient `controle_ia` (chaque verdict, « À vérifier » compris,
    et ses raisons), la relecture et l'état : des notes internes, lisibles par
    quiconque appelle l'API. La route publique ne sert que des articles
    publiés ET publiables ; le front n'en a pas besoin.
    """
    if valides is None:
        listes = await _validite_des_citations(db, [post.citations for post in posts])
        valides = {post.id: liste for post, liste in zip(posts, listes)}
    resultat = []
    for post in posts:
        citations_valides = valides[post.id]
        donnees = post.to_dict(publique=not admin, valides=citations_valides)
        if admin:
            contenu = ContenuDuBillet.de(post)
            controle, relecture = post.controle_ia, post.relecture
            donnees["controle_ia"] = controle_affiche(controle)
            donnees["relecture"] = relecture_affichee(contenu, controle, relecture)
            donnees["etat_publication"] = etat_de_publication(contenu, controle, relecture, citations_valides)
            donnees["sources"] = sources_affichees(contenu.sources, controle)
            donnees["avertissements"] = fusionner(
                avertissements_du_billet(post) if du_billet is None else du_billet,
                _avertissements_d_etat(post, citations_valides),
            )
        resultat.append(donnees)
    return resultat


async def _post_json(
    db: AsyncSession, post: BlogPost, admin: bool, du_billet: Optional[List[Avertissement]] = None
) -> Dict[str, Any]:
    return (await _posts_json(db, [post], admin, du_billet))[0]


async def _article_ou_404(db: AsyncSession, post_id: int, verrou: bool = False) -> BlogPost:
    """
    L'article, relu en base. `verrou` : la ligne est verrouillée jusqu'à la fin
    de la transaction (SELECT … FOR UPDATE). Une modification et l'écriture
    d'un contrôle lisent puis réécrivent le même dossier JSON : sans verrou,
    la seconde effaçait la première.
    """
    post = await db.get(
        BlogPost, post_id, populate_existing=True, with_for_update=True if verrou else None
    )
    if post is None:
        raise ErreurDuBlog(404, "introuvable", "Article introuvable.")
    return post


async def _citations_ou_422(db: AsyncSession, citations: Sequence[CitationIn]) -> List[Dict[str, Any]]:
    """Les citations relues en base ; la première qui n'y est pas fait répondre 422."""
    try:
        verifiees, _ = await citations_verifiees(db, [c.demande() for c in citations], strict=True)
    except CitationInvalide as e:
        raise ErreurDuBlog(422, "entree_invalide", str(e)) from None
    return verifiees


def _cle_de_citation(law_id: Any, article_id: Any, numero: Any) -> Tuple[Any, Any, str]:
    return (law_id, article_id or None, normalize_number(str(numero or "")))


async def _citations_envoyees(
    db: AsyncSession, envoyees: Sequence[CitationIn], post: BlogPost
) -> List[Dict[str, Any]]:
    """
    Les citations envoyées par l'administration, relues en base (422 sinon).

    Une citation d'AVANT la vérification par le serveur (verifie faux, law_id
    1 du gabarit de l'ancien prompt) que l'administration renvoie telle qu'elle
    l'a reçue reste ce qu'elle est : non vérifiée, gardée telle quelle (l'état
    « citations_non_verifiees » refuse alors de publier). Jamais relue sous son
    law_id : celui-ci vient du gabarit, et une relecture réussie rattacherait
    l'explication du modèle à un article sans rapport, marqué vérifié. Avant
    le tour 2, renvoyer une telle citation faisait répondre 422 à TOUTE
    modification, y compris celle qui retirait l'article du site.
    """
    anciennes = {
        _cle_de_citation(c.get("law_id"), c.get("article_id"), c.get("article_number")): c
        for c in post.citations
        if c.get("verifie") is not True
    }
    resultat: List[Dict[str, Any]] = []
    vues: Set[Tuple[Any, ...]] = set()
    for rang, citation in enumerate(envoyees, 1):
        ancienne = anciennes.get(
            _cle_de_citation(citation.law_id, citation.article_id, citation.article_number)
        )
        if ancienne is not None:
            gardee = {**ancienne, "verifie": False}
            if citation.title is not None:
                gardee["title"] = prose_publique(citation.title)
            resultat.append(gardee)
            continue
        try:
            (verifiee, *_), _ = await citations_verifiees(db, [citation.demande()], strict=True)
        except CitationInvalide as e:
            raise ErreurDuBlog(
                422, "entree_invalide", str(e).replace("Citation 1 :", f"Citation {rang} :", 1)
            ) from None
        cle = (verifiee["law_id"], verifiee["article_id"], verifiee["article_number"], verifiee["suite"])
        if cle not in vues:
            vues.add(cle)
            resultat.append(verifiee)
    return resultat


# Les adresses des articles de la rédaction, écrits et vérifiés à la main dans
# le front (src/lib/data/blog.ts). Un article de l'API à la même adresse les
# masquait, dans la liste comme à l'adresse partagée, sans un mot (revue du
# tour 1) : le lecteur d'un lien vers un article vérifié lisait un article
# généré. Aucun article de l'API ne les prend.
SLUGS_DE_LA_REDACTION = frozenset({
    "litiges-locatifs-resiliation-bail-expulsion-abusive",
    "rupture-contrat-travail-preavis-indemnites-licenciement",
    "arnaques-mobile-money-e-commerce-recours-legaux",
})


def _slug_explicite(valeur: str) -> str:
    slug = slugify(valeur, repli=False)
    if not slug:
        raise ErreurDuBlog(
            422, "entree_invalide",
            "Requête invalide — adresse (slug) : elle doit contenir au moins une lettre ou un chiffre.",
        )
    return slug


async def _slug_pris(db: AsyncSession, slug: str, sauf: Optional[int] = None) -> bool:
    if slug in SLUGS_DE_LA_REDACTION:
        return True
    requete = select(BlogPost.id).where(BlogPost.slug == slug)
    if sauf is not None:
        requete = requete.where(BlogPost.id != sauf)
    return (await db.execute(requete.limit(1))).first() is not None


async def _slug_libre(db: AsyncSession, base: str, sauf: Optional[int] = None) -> str:
    """`base`, ou `base-2`, `base-3`... : la première adresse libre (`sauf` : l'article lui-même)."""
    # Un préfixe de la base : un suffixe peut raccourcir une base de 80 caractères.
    requete = select(BlogPost.slug).where(BlogPost.slug.like(f"{base[:60]}%"))
    if sauf is not None:
        requete = requete.where(BlogPost.id != sauf)
    pris = set((await db.execute(requete)).scalars()) | SLUGS_DE_LA_REDACTION
    if base not in pris:
        return base
    for rang in range(2, 10_000):
        suffixe = f"-{rang}"
        candidat = f"{base[:SLUG_LONGUEUR_MAX - len(suffixe)].rstrip('-')}{suffixe}"
        if candidat not in pris:
            return candidat
    raise ErreurDuBlog(409, "slug_pris", "Aucune adresse libre pour cet article.")


# « base-2 », « base-3 »… : le suffixe que _slug_libre ajoute à une adresse prise.
_SUFFIXE_D_ADRESSE = re.compile(r"^(?P<base>.+)-(?P<rang>[2-9]|[1-9][0-9]+)$")


def adresse_tiree_du_titre(slug: str, titre: str) -> bool:
    """
    L'adresse est-elle celle que le serveur a tirée de ce titre : slugify(titre),
    ou la suivante libre (« -2 », « -3 »…, _slug_libre) ? Sinon, quelqu'un l'a
    écrite à la main (ou le titre a changé depuis sans elle).
    """
    base = slugify(titre, repli=False)
    if not base or not slug:
        return False
    if slug == base:
        return True
    trouve = _SUFFIXE_D_ADRESSE.match(slug)
    if trouve is None:
        return False
    suffixe = f"-{trouve.group('rang')}"
    return trouve.group("base") == base[:SLUG_LONGUEUR_MAX - len(suffixe)].rstrip("-")


async def _adresse_du_nouveau_titre(db: AsyncSession, post: BlogPost, titre: Optional[str]) -> Optional[str]:
    """
    L'adresse que prend un BROUILLON JAMAIS PUBLIÉ dont le titre change
    (avenant du tour 7), ou None : elle suit le nouveau titre, libre comme à
    la génération, si le serveur avait tiré l'actuelle de l'ancien titre.

    POURQUOI : un brouillon généré gardait l'adresse de son premier titre ; à
    l'essai réel du 04/11, le titre du bail, corrigé (il portait la fausse
    prémisse de l'expulsion sans juge), laissait l'adresse la dire encore,
    jusqu'à ce que l'administrateur la réécrive à la main. Personne ne connaît
    l'adresse d'un brouillon jamais publié : la changer ne casse aucun lien.
    Celle d'un article publié une fois a pu circuler, et une adresse écrite à
    la main est un choix : gardées.
    """
    if titre is None or post.deja_publie or not adresse_tiree_du_titre(post.slug, post.title):
        return None
    base = slugify(titre, repli=False)
    # Le même titre à l'adresse près (blancs, ponctuation) : l'adresse ne bouge pas.
    if not base or base == slugify(post.title, repli=False):
        return None
    suivante = await _slug_libre(db, base, sauf=post.id)
    return suivante if suivante != post.slug else None


def _slug_deja_pris(slug: str) -> ErreurDuBlog:
    return ErreurDuBlog(
        409, "slug_pris",
        f"L'adresse « {slug} » est déjà prise par un autre article : choisissez-en une autre.",
    )


def _conflit_d_adresse(erreur: IntegrityError) -> bool:
    return "blog_posts_slug_key" in str(erreur)


async def _inserer(
    db: AsyncSession, valeurs: Dict[str, Any], base: str, imposee: bool
) -> BlogPost:
    """
    Enregistre un nouvel article sous une adresse libre. L'unicité est vérifiée
    AVANT, puis tenue par la contrainte : deux enregistrements simultanés sous
    la même adresse faisaient une IntegrityError, donc une 500. Une adresse
    choisie (`imposee`) déjà prise : 409 ; une adresse tirée du titre : la
    suivante libre.
    """
    for _ in range(3):
        if imposee:
            if await _slug_pris(db, base):
                raise _slug_deja_pris(base)
            slug = base
        else:
            slug = await _slug_libre(db, base)
        post = BlogPost(slug=slug, **valeurs)
        db.add(post)
        try:
            await db.commit()
        except IntegrityError as e:
            await db.rollback()
            if not _conflit_d_adresse(e):
                raise
            if imposee:
                raise _slug_deja_pris(slug) from None
            continue
        await db.refresh(post)
        return post
    raise _slug_deja_pris(base)


def _nom_de_l_administrateur(admin: Any) -> str:
    """Le nom noté dans `relecture.par` : nom complet, sinon identifiant (jamais l'e-mail)."""
    return str(getattr(admin, "full_name", None) or getattr(admin, "username", None) or "administrateur")


def _id_de_l_administrateur(admin: Any) -> Optional[int]:
    identifiant = getattr(admin, "id", None)
    return identifiant if isinstance(identifiant, int) else None


# ==================== ROUTES PUBLIQUES ====================


@router.get("")
async def list_blog_posts(
    category: Optional[str] = Query(None, max_length=100),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Les articles publiés ET PUBLIABLES, du plus récent au plus ancien. Jamais
    un brouillon, jamais un article d'avant le contrôle (décision du
    propriétaire du 30/10 : les trois lignes du 26/10 de jurix_dev, dont la
    prose écrite de mémoire est fausse, sont cachées du public), jamais un
    article devenu non publiable (texte cité dépublié depuis, par exemple).

    La publiabilité se juge en Python (empreintes, citations relues en base) :
    la requête ne garde d'abord que les articles publiés qui ont des sources et
    un contrôle, puis la page est coupée APRÈS le jugement, pour que `limit` et
    `offset` comptent des articles servis.

    Au moins aussi strict que le correctif du propriétaire sur main (5bdcb4f,
    04/11 : publié, au moins une citation, toutes `verifie: true`) :
    « publiable » exige en plus ses sources, un contrôle à jour sans phrase
    contredite, chaque phrase de règle à confirmer réécrite ou confirmée (tour
    7), la relecture de chaque partie, et des citations que la base porte
    encore aujourd'hui.
    """
    requete = select(BlogPost).where(
        BlogPost.is_published.is_(True),
        func.jsonb_typeof(BlogPost.verification) == "object",
        func.jsonb_typeof(BlogPost.verification["sources"]) == "array",
        func.jsonb_typeof(BlogPost.verification["controle_ia"]) == "object",
    )
    rubrique = (category or "").strip()
    if rubrique and rubrique.lower() != "tous":
        # Égalité, et non `ilike '%…%'` : « Travail » ne doit pas ramener
        # « Travail & Emploi », et un « % » envoyé ne doit rien filtrer.
        requete = requete.where(func.lower(BlogPost.category) == rubrique.lower())
    requete = requete.order_by(desc(BlogPost.published_at), desc(BlogPost.id))
    posts = (await db.execute(requete)).scalars().all()
    listes = await _validite_des_citations(db, [post.citations for post in posts])
    valides = {post.id: liste for post, liste in zip(posts, listes)}
    publiables = [post for post in posts if _etat(post, valides[post.id]) == "publiable"]
    return await _posts_json(db, publiables[offset:offset + limit], admin=False, valides=valides)


# ==================== ROUTES D'ADMINISTRATION ====================
#
# Déclarées AVANT GET /{slug} et /admin/all avant /admin/{post_id} : les
# routes sont essayées dans l'ordre.


async def _annuler_la_transaction_ouverte(db: AsyncSession) -> None:
    """
    Annule la transaction encore ouverte : un délai dépassé pendant une lecture
    peut l'avoir laissée inutilisable. Seulement s'il y en a une : annuler
    périme TOUS les objets de la session (une lecture d'attribut ensuite est
    une requête implicite, interdite en asynchrone). Un contrôle refusé (aucun
    extrait, quota…) laisse, lui, une transaction saine : rien à annuler.
    """
    if db.in_transaction():
        await db.rollback()


async def _controler_apres_la_generation(
    db: AsyncSession, post_id: int, llm: Optional[Any], delai: float
) -> Tuple[BlogPost, Optional[Avertissement]]:
    """
    Le contrôle IA du brouillon tout juste enregistré : (l'article relu, et
    l'avertissement qui dit de relancer le contrôle s'il a échoué). Un échec
    ne perd rien : le brouillon reste, controle_ia null. `llm` None : le
    modèle du contrôle (BLOG_CONTROLE_MODELE).
    """
    probleme: Optional[str] = None
    try:
        post = await _article_ou_404(db, post_id)
        # Le même délai pour l'échéance et pour le plan du contrôle (revue du
        # tour 6) : il ne part que si son dernier lot finit avant elle.
        async with asyncio.timeout(delai):
            controle = await ControleDuBillet(db, llm=llm).controler(post, delai=delai)
    except TimeoutError:
        await _annuler_la_transaction_ouverte(db)
        probleme = f"il a dépassé {delai:g} secondes"
    except ErreurDeGeneration as e:
        probleme = str(e)
    except Exception:
        logger.exception(f"❌ Blog : contrôle IA du brouillon {post_id} en échec")
        await _annuler_la_transaction_ouverte(db)
        probleme = "erreur interne"
    if probleme is not None:
        logger.warning(f"⚠️ Blog : contrôle IA du brouillon {post_id} non fait ({probleme[:120]})")
        return await _article_ou_404(db, post_id), Avertissement(
            None, None,
            f"Le contrôle IA n'a pas pu être fait ({probleme.rstrip('.')}). Le brouillon est "
            f"enregistré, sans contrôle : relancez le contrôle IA avant de publier.",
        )
    post = await _article_ou_404(db, post_id, verrou=True)
    _enregistrer_le_controle(post, controle)
    await db.commit()
    await db.refresh(post)
    return post, None


def _avec_son_fait_le(controle: Dict[str, Any], post: BlogPost) -> Dict[str, Any]:
    """
    Le contrôle à enregistrer, son `fait_le` après celui du contrôle qu'il
    remplace et de celui que la relecture enregistrée nomme (fait_le_apres) :
    chaque contrôle a le sien, que les cases cochées nomment (avenant du tour 5).
    """
    return {
        **controle,
        "fait_le": fait_le_apres(
            controle["fait_le"], (post.controle_ia or {}).get("fait_le"), post.relecture.get("controle"),
        ),
    }


def _enregistrer_le_controle(post: BlogPost, controle: Dict[str, Any]) -> None:
    """
    Range le nouveau contrôle (son `fait_le` à lui), et EFFACE les cases qu'il
    fait tomber (relecture_apres_un_controle) : un contrôle suivant qui
    redonnerait les verdicts d'avant ne les fait pas revivre, et l'article
    qu'il a caché du public n'y revient qu'après une relecture (revue du front
    du tour 5).
    """
    nouveau = _avec_son_fait_le(controle, post)
    avant = post.relecture
    relecture = relecture_apres_un_controle(ContenuDuBillet.de(post), post.controle_ia, nouveau, avant)
    if relecture is not avant:
        post.relecture = relecture
    post.controle_ia = nouveau


ARTICLE_EN_LIGNE = (
    "Contrôle non relancé : l'article est en ligne. Un nouveau contrôle peut juger autrement des "
    "phrases : leurs parties seraient à relire, leurs phrases de règle à confirmer, et l'article "
    "ne serait plus servi au public jusque-là. Rien n'a été fait : confirmez la relance "
    "({\"confirmer\": true}) pour la lancer."
)
ARTICLE_PUBLIE_PENDANT_LE_CONTROLE = (
    "Contrôle non enregistré : l'article a été publié pendant le contrôle. Ses nouveaux verdicts "
    "pourraient faire relire des parties, ou confirmer des phrases de règle, et le retirer du "
    "public jusque-là. Rien n'a changé : relancez le contrôle en confirmant "
    "({\"confirmer\": true}) si vous le voulez."
)


def _relance_confirmee(corps: Any) -> bool:
    """Le corps est-il {"confirmer": true}, le booléen JSON (et non « oui », ni 1) ?"""
    return isinstance(corps, dict) and corps.get("confirmer") is True


@router.post("/generate", status_code=201)
async def generate_blog_post(
    req: BlogGenerateRequest,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    service: BlogGeneratorService = Depends(get_blog_generator),
):
    """
    Rédige un article sur `topic`, l'enregistre en BROUILLON, puis le
    CONTRÔLE (avec le modèle du contrôle, BLOG_CONTROLE_MODELE ; tour 6).

    Rend `{post, sources, avertissements}` :
    - `post` : l'article enregistré (isPublished false), avec ses
      `avertissements`, son `controle_ia` (null si le contrôle a échoué), sa
      `relecture` (rien de coché), son `etat_publication` et ses `sources`
      (tour 4 : gardées avec l'article, rendues par toute lecture de
      l'administration) ;
    - `sources` : les extraits donnés au modèle, `{n, law_id, law_title,
      article_number, article_id, suite, extrait}` — un extrait par article,
      ou par suite d'article (`suite` vrai) ; ce sont les numéros des
      `extraits` du contrôle, et ceux de `post.sources` ;
    - `avertissements` : ce que la génération a relevé (mentions qu'aucun
      extrait ne porte, citations écartées, cotes retirées), un contrôle IA à
      relancer, puis ceux de l'article.

    Raises:
        422 sans_source : aucun texte publié ne traite clairement le sujet
        (modèle non appelé), ou le modèle juge les extraits insuffisants
        429 quota, 503 saturation ou recherche_indisponible, 502 reponse_invalide,
        504 delai (au-delà de DELAI_GENERATION_S) : rien n'est enregistré
    """
    debut = time.monotonic()
    try:
        async with asyncio.timeout(DELAI_GENERATION_S):
            article = await service.rediger(req.topic, req.category, req.additional_context)
    except TimeoutError:
        logger.warning(f"⏱️ Blog : génération de « {req.topic[:60]} » au-delà de {DELAI_GENERATION_S} s")
        raise ErreurDuBlog(
            504, "delai",
            f"La génération a dépassé {DELAI_GENERATION_S} secondes : aucun article n'a été "
            "enregistré. Réessayez dans quelques instants.",
        ) from None
    except ErreurDeGeneration as e:
        raise ErreurDuBlog(e.statut, e.code, str(e), e.retry_after) from None

    post = await _inserer(
        db,
        {
            "title": article.title,
            "category": article.category,
            "reading_time": article.reading_time,
            "context": article.context,
            "summary": article.summary,
            "key_points": article.key_points,
            "citations": article.citations,
            # Les extraits lus : le contrôle les relit sous les mêmes numéros.
            "sources": article.sources,
            "citizen_impact": article.citizen_impact,
            "suggested_question": article.suggested_question or None,
            "tags": article.tags,
            "is_published": False,
        },
        base=slugify(article.title),
        imposee=False,
    )
    logger.info(f"📝 Blog : brouillon {post.id} « {post.slug} » enregistré")
    # Génération et contrôle tiennent ensemble dans DELAI_DE_LA_REDACTION_S
    # quand c'est possible : le front abandonne à 215 s (revue du tour 3 : 150
    # puis 120 s côté serveur). Un contrôle trop court échoue sans rien
    # perdre : le brouillon reste, à contrôler.
    delai = delai_du_controle_apres_la_generation(time.monotonic() - debut)
    # Le modèle DU CONTRÔLE (tour 6), pas celui qui vient de rédiger.
    post, controle_manque = await _controler_apres_la_generation(
        db, post.id, service.llm_du_controle, delai
    )
    du_billet = avertissements_du_billet(post)
    return {
        "post": await _post_json(db, post, admin=True, du_billet=du_billet),
        "sources": article.sources,
        "avertissements": fusionner(
            article.avertissements, [controle_manque] if controle_manque else [], du_billet
        ),
    }


@router.post("/{post_id}/controle")
async def controler_un_article(
    post_id: int,
    corps: Any = Body(None, description='{"confirmer": true} pour relancer le contrôle d\'un article en ligne.'),
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    controleur: ControleDuBillet = Depends(get_controleur),
):
    """
    (Re)lance le contrôle IA sur le texte ACTUEL de l'article, contre ses
    sources, et rend l'article. Un échec ne change rien : le contrôle d'avant
    reste (périmé, ou absent).

    UN ARTICLE EN LIGNE (avenant du tour 5) ne se recontrôle que sur
    {"confirmer": true}. POURQUOI (essai réel du tour 4) : à la température 0,
    des verdicts changent d'un contrôle à l'autre ; relancé sans un mot, un
    contrôle qui jugeait autrement une partie la décochait, et l'article
    quittait le public sans que l'administrateur l'ait voulu ni su.

    LES CASES QU'IL FAIT TOMBER SONT EFFACÉES (_enregistrer_le_controle). Elles
    revivaient au contrôle suivant s'il redonnait les verdicts d'avant, et
    l'article caché revenait en ligne sans relecture ni « Publier » (revue du
    front du tour 5).

    Raises:
        404 introuvable ; 422 sources_absentes (l'article n'a pas de sources :
        les reconstruire d'abord, POST /{id}/sources ; le modèle n'est pas
        appelé) ; 409 article_en_ligne (publié, sans {"confirmer": true} : le
        modèle n'est pas appelé) ; 422 entree_invalide (aucune phrase, ou trop
        pour les appels du contrôle) ; 429 quota, 503 saturation (modèle
        saturé, budget de jetons épuisé, ou un autre contrôle en cours ; avec
        Retry-After), 502 reponse_invalide (réponse coupée, illisible, sans
        verdict, ou requête refusée), 504 delai (au-delà de DELAI_CONTROLE_S),
        500 erreur_interne
    """
    post = await _article_ou_404(db, post_id)
    # Sans sources, rien à relancer : le 422 sources_absentes du contrôle dit
    # quoi faire d'abord (reconstruire les sources, ce qui dépublie).
    if post.is_published and sources_valables(post.sources) and not _relance_confirmee(corps):
        raise ErreurDuBlog(409, "article_en_ligne", ARTICLE_EN_LIGNE)
    try:
        # Le même délai pour l'échéance et pour le plan du contrôle (revue du
        # tour 6) : attente d'un autre contrôle, attentes du budget de jetons
        # et lots comptés ensemble depuis ici.
        async with asyncio.timeout(DELAI_CONTROLE_S):
            controle = await controleur.controler(post, delai=DELAI_CONTROLE_S)
    except TimeoutError:
        await _annuler_la_transaction_ouverte(db)
        logger.warning(f"⏱️ Blog : contrôle de l'article {post_id} au-delà de {DELAI_CONTROLE_S} s")
        raise ErreurDuBlog(
            504, "delai",
            f"Le contrôle IA a dépassé {DELAI_CONTROLE_S} secondes : l'article est inchangé. "
            "Relancez le contrôle dans quelques instants.",
        ) from None
    except ErreurDeGeneration as e:
        raise ErreurDuBlog(e.statut, e.code, str(e), e.retry_after) from None
    # Relu et verrouillé : une modification enregistrée pendant le contrôle
    # n'est pas effacée (le contrôle, lui, porte l'empreinte du texte jugé :
    # il est alors périmé, et le dit). De deux contrôles finis dans la même
    # seconde, le second enregistré a un `fait_le` plus tardif que le premier.
    post = await _article_ou_404(db, post_id, verrou=True)
    if post.is_published and sources_valables(post.sources) and not _relance_confirmee(corps):
        # Publié PENDANT le contrôle (un autre onglet) : ses nouveaux verdicts
        # pourraient le retirer du public sans que personne l'ait confirmé.
        raise ErreurDuBlog(409, "article_en_ligne", ARTICLE_PUBLIE_PENDANT_LE_CONTROLE)
    _enregistrer_le_controle(post, controle)
    await db.commit()
    await db.refresh(post)
    logger.info(f"🔎 Blog : contrôle IA de l'article {post_id} enregistré")
    return await _post_json(db, post, admin=True)


async def _citations_portees(
    db: AsyncSession, citations: Sequence[Dict[str, Any]], sources: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Les citations qu'on garde après une reconstruction des sources, et un
    avertissement par citation retirée.

    Gardée : vérifiée par le serveur ET portée par une source reconstruite
    (même texte, même ligne, même numéro, début ou suite) ; elle est alors
    relue en base, comme à la génération (extrait et lien d'aujourd'hui),
    avec son explication. Une citation d'avant la vérification n'est jamais
    portée : son texte n'a jamais été lu dans la base (law_id 1 du gabarit de
    l'ancien prompt).
    """
    portees = {
        cle_d_article(s["law_id"], s["article_id"], s["article_number"], s["suite"])
        for s in sources_valables(list(sources))
    }
    demandes: List[DemandeDeCitation] = []
    rangs: List[int] = []
    avertissements: List[str] = []
    for rang, citation in enumerate(citations, 1):
        nom = f"Citation {rang} ({nom_d_article(citation)} · {citation.get('reference') or 'texte inconnu'})"
        cle = cle_d_article(
            citation.get("law_id"), citation.get("article_id"), citation.get("article_number"),
            citation.get("suite"),
        )
        if citation.get("verifie") is not True:
            avertissements.append(
                f"{nom} : écrite avant la vérification par le serveur, elle ne renvoie à aucun "
                f"article lu dans la base. Retirée : citez un article parmi les sources."
            )
        elif cle not in portees:
            avertissements.append(
                f"{nom} : aucune des sources reconstruites ne porte cet article. Retirée : citez "
                f"un article parmi les sources, ou ajoutez-le à nouveau par son numéro."
            )
        else:
            demandes.append(DemandeDeCitation(
                int(citation["law_id"]), int(citation["article_id"]),
                str(citation.get("article_number") or "") or None,
                title=str(citation.get("title") or ""), suite=citation.get("suite") is True,
            ))
            rangs.append(rang)
    gardees, refus = await citations_verifiees(db, demandes, strict=False)
    for rang_demande, raison in refus:
        avertissements.append(f"Citation {rangs[rang_demande - 1]} : {raison} Retirée.")
    return gardees, avertissements


@router.post("/{post_id}/sources")
async def reconstruire_les_sources(
    post_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
    service: BlogGeneratorService = Depends(get_blog_generator),
):
    """
    Reconstruit les SOURCES de l'article par la recherche de la génération, à
    partir de son titre et de son contexte, sans appeler le modèle (tour 4).

    POURQUOI. Décision du propriétaire (30/10) : les articles d'avant restent
    en base, visibles et CORRIGEABLES dans l'administration, et ne reviennent
    en ligne que par le nouveau parcours. Sans sources, ils ne pouvaient être
    ni contrôlés ni relus contre des textes de la base (revue du tour 3 :
    impasse ; seule issue, supprimer et régénérer). Même chemin pour un
    article écrit à la main.

    Ensuite : les citations non portées par une source sont retirées (chacune
    avec un avertissement) ; si les sources changent, le contrôle d'avant est
    effacé ; l'article repasse en BROUILLON. Il reste à citer, contrôler,
    relire, puis publier.

    Rend 200 `{post, sources, avertissements}` (la forme de /generate).

    Raises:
        404 introuvable ; 422 sans_source (aucun texte ne traite clairement le
        sujet), 503 recherche_indisponible, 504 delai : rien n'est changé
    """
    post = await _article_ou_404(db, post_id)
    sujet = sujet_de_l_article(post.title, post.context)
    try:
        async with asyncio.timeout(DELAI_SOURCES_S):
            sources = await service.sources_du_sujet(sujet)
    except TimeoutError:
        await _annuler_la_transaction_ouverte(db)
        logger.warning(f"⏱️ Blog : sources de l'article {post_id} au-delà de {DELAI_SOURCES_S} s")
        raise ErreurDuBlog(
            504, "delai",
            f"La recherche des sources a dépassé {DELAI_SOURCES_S} secondes : l'article est "
            "inchangé. Réessayez dans quelques instants.",
        ) from None
    except ErreurDeGeneration as e:
        raise ErreurDuBlog(e.statut, e.code, str(e), e.retry_after) from None

    # Relu et verrouillé : une modification enregistrée pendant la recherche
    # n'est pas effacée.
    post = await _article_ou_404(db, post_id, verrou=True)
    gardees, avertissements = await _citations_portees(db, post.citations, sources)
    changees = not memes_sources(post.sources, sources)
    post.sources = sources
    post.citations = gardees
    if changees and post.controle_ia is not None:
        # Ses numéros d'extraits renvoyaient aux anciennes sources.
        post.controle_ia = None
        avertissements.append(
            "Les sources ont changé : le contrôle IA d'avant, fait sur les anciennes, est effacé. "
            "Relancez-le, puis relisez chaque partie."
        )
    if post.is_published:
        post.noter_publie()
        logger.info(f"🗑️ Blog : article {post.id} « {post.slug} » repassé en brouillon (sources)")
        avertissements.append(
            "L'article repasse en brouillon : il ne revient en ligne que par « Publier », une fois "
            "cité, contrôlé et relu."
        )
    post.is_published = False
    await db.commit()
    await db.refresh(post)
    logger.info(
        f"📚 Blog : sources de l'article {post_id} reconstruites ({len(sources)}), "
        f"{len(gardees)} citation(s) gardée(s)"
    )
    du_billet = avertissements_du_billet(post)
    donnees = await _post_json(db, post, admin=True, du_billet=du_billet)
    return {
        "post": donnees,
        "sources": donnees["sources"],
        "avertissements": fusionner([Avertissement(None, None, a) for a in avertissements], du_billet),
    }


@router.get("/admin/all")
async def list_all_posts_admin(
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Tous les articles, brouillons et articles d'avant compris, du plus récent au plus ancien."""
    requete = select(BlogPost).order_by(desc(BlogPost.created_at), desc(BlogPost.id))
    posts = (await db.execute(requete)).scalars().all()
    return await _posts_json(db, posts, admin=True)


@router.get("/admin/{post_id}")
async def get_blog_post_admin(
    post_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Un article, brouillon compris : l'aperçu avant publication."""
    return await _post_json(db, await _article_ou_404(db, post_id), admin=True)


@router.post("", status_code=201)
async def create_blog_post(
    body: BlogPostCreate,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Crée un brouillon à la main ; ses citations sont relues en base (422
    sinon). Ni sources, ni contrôle, ni relecture : il est
    « sources_absentes ». Avant de publier : POST /{id}/sources, puis le
    contrôle et la relecture.
    """
    citations = await _citations_ou_422(db, body.bot_analysis.citations)
    analyse = body.bot_analysis
    valeurs = {
        "title": body.title,
        "category": body.category,
        "reading_time": texte_brut(body.reading_time) or duree_de_lecture(
            [body.context, analyse.summary, *analyse.key_points, analyse.citizen_impact]
        ),
        "context": body.context,
        "summary": analyse.summary,
        "key_points": analyse.key_points,
        "citations": citations,
        "citizen_impact": analyse.citizen_impact,
        "suggested_question": prose_publique(body.suggested_question) or None,
        "tags": mots_cles_propres(body.tags, au_plus=20),
        "is_published": False,
    }
    if body.slug is not None:
        post = await _inserer(db, valeurs, base=_slug_explicite(body.slug), imposee=True)
    else:
        post = await _inserer(db, valeurs, base=slugify(body.title), imposee=False)
    return await _post_json(db, post, admin=True)


def _retrait_seul(corps: Any) -> bool:
    """Le corps est-il EXACTEMENT {"is_published": false} (le booléen JSON) ?"""
    return isinstance(corps, dict) and set(corps) == {"is_published"} and corps["is_published"] is False


@router.put("/{post_id}")
async def update_blog_post(
    post_id: int,
    corps: Dict[str, Any] = Body(..., description="Les champs de BlogPostUpdate, tous facultatifs."),
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Modifie, relit, publie ou dépublie un article.

    RETIRER DU SITE : un corps qui est EXACTEMENT {"is_published": false} retire
    l'article aussitôt, toujours, sans rien relire d'autre (citations d'avant,
    champs vides, rien ne l'en empêche). C'est ce qu'envoie « Retirer du
    site ».

    TOUT AUTRE CORPS est validé EN ENTIER (schéma, cases de relecture, phrases
    confirmées et le contrôle qu'elles nomment — 422 relecture_perimee —,
    adresse, citations relues en base, et l'état de publication s'il doit être
    ou rester en ligne), puis appliqué d'un bloc, ou pas du tout. POURQUOI
    (revue du tour 2) : le tour 2 dépubliait d'abord, puis validait ; « Publier les
    modifications » d'un article en ligne, refusé pour une adresse prise,
    laissait l'article HORS LIGNE alors que le front le croyait en ligne.
    Désormais un refus ne retire rien.

    PUBLIER, ou RESTER EN LIGNE (is_published vrai, ou absent sur un article
    publié) exige l'état « publiable » APRÈS la modification : sources,
    citations vérifiées, contrôle IA à jour sans phrase contredite, chaque
    phrase de règle à confirmer confirmée (tour 7), chaque partie relue.
    Sinon 422, code = l'état. Modifier une partie relue d'un
    article en ligne (le titre compris, tour 4) périme son contrôle : envoyer
    is_published false avec la modification, puis relancer le contrôle,
    relire, et republier. La rubrique, l'adresse, la durée et les mots-clés
    se changent en ligne. Le passage de brouillon à publié remet
    `published_at` à la date du jour.

    L'ADRESSE D'UN BROUILLON JAMAIS PUBLIÉ suit son titre tant que le serveur
    l'a tirée de lui (avenant du tour 7, _adresse_du_nouveau_titre).
    """
    if _retrait_seul(corps):
        post = await _article_ou_404(db, post_id, verrou=True)
        if post.is_published:
            post.noter_publie()
            post.is_published = False
            await db.commit()
            await db.refresh(post)
            logger.info(f"🗑️ Blog : article {post.id} « {post.slug} » retiré du site")
        else:
            await db.commit()
        return await _post_json(db, post, admin=True)
    post = await _article_ou_404(db, post_id, verrou=True)
    return await _modifier(db, post, corps, current_admin)


async def _modifier(
    db: AsyncSession, post: BlogPost, corps: Dict[str, Any], admin: Any
) -> Dict[str, Any]:
    """La modification elle-même : tout validé AVANT de toucher à l'article, puis d'un bloc."""
    try:
        body = BlogPostUpdate.model_validate(corps)
    except ValidationError as e:
        raise ErreurDuBlog(422, "entree_invalide", message_de_validation(e.errors())) from None
    # Avenant du tour 5 : une case cochée nomme le contrôle relu, qui doit être
    # celui enregistré ; sinon rien n'est appliqué, pas même le reste du corps.
    # Avenant du tour 7 : de même pour toute liste de phrases confirmées, dont
    # chaque identifiant doit être une phrase de ce contrôle.
    if body.relecture is not None:
        refus = relecture_perimee(
            body.relecture.parties, body.relecture.controle, post.controle_ia,
        ) or confirmations_perimees(
            body.relecture.phrases_confirmees, body.relecture.controle, post.controle_ia,
        )
        if refus:
            raise ErreurDuBlog(422, "relecture_perimee", refus)
    analyse = body.bot_analysis
    avant = ContenuDuBillet.de(post)

    # 1. Tout ce qui peut refuser la modification, sans toucher à l'article.
    citations = None
    if analyse is not None and analyse.citations is not None:
        citations = await _citations_envoyees(db, analyse.citations, post)
    slug = None
    if body.slug is not None:
        slug = _slug_explicite(body.slug)
        if slug != post.slug and await _slug_pris(db, slug, sauf=post.id):
            raise _slug_deja_pris(slug)
    if slug is None or slug == post.slug:
        # Avenant du tour 7 : un brouillon jamais publié suit son nouveau titre.
        slug = await _adresse_du_nouveau_titre(db, post, body.title) or slug

    question = avant.suggested_question
    if "suggested_question" in body.model_fields_set:
        question = prose_publique(body.suggested_question)
    apres = replace(
        avant,
        title=body.title if body.title is not None else avant.title,
        context=body.context if body.context is not None else avant.context,
        summary=analyse.summary if analyse and analyse.summary is not None else avant.summary,
        key_points=(
            list(analyse.key_points) if analyse and analyse.key_points is not None
            else avant.key_points
        ),
        citizen_impact=(
            analyse.citizen_impact if analyse and analyse.citizen_impact is not None
            else avant.citizen_impact
        ),
        citations=citations if citations is not None else avant.citations,
        suggested_question=question,
    )
    relecture = nouvelle_relecture(
        avant, apres, post.controle_ia, post.relecture,
        body.relecture.parties if body.relecture else None,
        par=_nom_de_l_administrateur(admin), par_id=_id_de_l_administrateur(admin),
        le=maintenant_iso(),
        controle_relu=body.relecture.controle if body.relecture else None,
        confirmees=body.relecture.phrases_confirmees if body.relecture else None,
    )
    publie = post.is_published if body.is_published is None else body.is_published
    if publie:
        (valides,) = await _validite_des_citations(db, [apres.citations])
        etat = etat_de_publication(apres, post.controle_ia, relecture, valides)
        if etat != "publiable":
            raise ErreurDuBlog(
                422, etat, motif_du_refus(etat, apres, post.controle_ia, relecture, valides)
            )

    # 2. Tout est valide : la modification, d'un bloc.
    for champ in ("title", "category"):
        valeur = getattr(body, champ)
        if valeur is not None:
            setattr(post, champ, valeur)
    if body.tags is not None:
        post.tags = mots_cles_propres(body.tags, au_plus=20)
    prose_modifiee = (apres.context, apres.summary, apres.key_points, apres.citizen_impact) != (
        avant.context, avant.summary, avant.key_points, avant.citizen_impact
    )
    if prose_modifiee:
        post.context, post.summary = apres.context, apres.summary
        post.key_points, post.citizen_impact = apres.key_points, apres.citizen_impact
    if apres.suggested_question != avant.suggested_question:
        post.suggested_question = apres.suggested_question or None
    if citations is not None:
        post.citations = citations
    if relecture != post.relecture:
        post.relecture = relecture
    if slug is not None:
        post.slug = slug

    if body.reading_time:
        post.reading_time = texte_brut(body.reading_time) or post.reading_time
    elif prose_modifiee:
        post.reading_time = duree_de_lecture(
            [post.context, post.summary, *(post.key_points or []), post.citizen_impact]
        )

    if publie and not post.is_published:
        post.published_at = func.current_date()
        logger.info(f"📰 Blog : article {post.id} « {post.slug} » publié")
    elif not publie and post.is_published:
        logger.info(f"🗑️ Blog : article {post.id} « {post.slug} » retiré du site (avec modifications)")
    if publie or post.is_published:
        # Son adresse a pu circuler : elle ne suivra plus le titre (tour 7).
        post.noter_publie()
    post.is_published = publie

    try:
        await db.commit()
    except IntegrityError as e:
        # Après le rollback, l'article est expiré : ne plus lire ses attributs.
        await db.rollback()
        if _conflit_d_adresse(e):
            raise _slug_deja_pris(slug or "") from None
        raise
    await db.refresh(post)
    return await _post_json(db, post, admin=True)


@router.delete("/{post_id}", status_code=204)
async def delete_blog_post(
    post_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Supprime un article, publié ou non."""
    post = await _article_ou_404(db, post_id)
    await db.delete(post)
    await db.commit()
    return Response(status_code=204)


# ==================== ROUTE PUBLIQUE PAR ADRESSE ====================


@router.get("/{slug}")
async def get_blog_post(
    slug: str = Path(..., max_length=255),
    db: AsyncSession = Depends(get_db),
):
    """
    Un article PUBLIÉ ET PUBLIABLE. Un brouillon, un article dépublié, un
    article d'avant le contrôle ou devenu non publiable répond 404 : la route
    servait tout article par son adresse, et un article retiré pour erreur
    restait lisible par les liens déjà partagés (essai réel du 29/10/2026,
    capture 11) ; puis la prose fausse des articles d'avant restait publique
    (revue du tour 2). L'aperçu passe par GET /admin/{id}.
    """
    requete = select(BlogPost).where(BlogPost.slug == slug, BlogPost.is_published.is_(True))
    post = (await db.execute(requete)).scalar_one_or_none()
    if post is None:
        raise ErreurDuBlog(404, "introuvable", "Article introuvable.")
    (valides,) = await _validite_des_citations(db, [post.citations])
    if _etat(post, valides) != "publiable":
        raise ErreurDuBlog(404, "introuvable", "Article introuvable.")
    (donnees,) = await _posts_json(db, [post], admin=False, valides={post.id: valides})
    return donnees
