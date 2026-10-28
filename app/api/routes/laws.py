"""
API routes for Laws management.

Endpoints:
- GET /api/v1/laws - List laws with filters
- GET /api/v1/laws/{id} - Get law detail with articles
- POST /api/v1/laws/{id}/explain-article - Explain one article (Gemini)
- POST /api/v1/admin/laws - Create law (admin only)
- PUT /api/v1/admin/laws/{id} - Update law (admin only)
- DELETE /api/v1/admin/laws/{id} - Delete law (admin only)

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Union, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import Integer, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer, load_only, selectinload

from app.core.auth import get_current_admin_user
from app.core.database import AsyncSessionLocal, get_db
from app.models.law import Article, Category, Law
from app.models.user import User
from app.schemas.law import (
    ArticleExplanationRequest,
    ArticleExplanationResponse,
    LawCreate,
    LawDetailResponse,
    LawResponse,
    LawUpdate,
)
from app.schemas.law_list import LawListLightItem, LawListPage, TriDesListes
from app.schemas.reperes import ReperesDuTexte
from app.schemas.statut import ArticleApercuResponse, ArticleHomonyme, StatutJuridique
from app.services.article_reference import normalize_number
from app.services.chiffres_du_corpus import invalider_cache_corpus
from app.services.document_storage import (
    DocumentInaccessible,
    DocumentIntrouvable,
    IdentifiantInvalide,
    chemin_local,
)
from app.services.explanation_service import (
    ArticleNotFoundError,
    ExplanationError,
    ExplanationOverloadedError,
    ExplanationQuotaError,
    ExplanationService,
)
from app.services.search_service import invalidate_search_cache
from app.services.statut_juridique import (
    InfoTexte,
    MorceauDuTexte,
    contient_le_repere,
    invalider_cache_statut,
    morceaux_etablis,
    statut_d_un_article,
    statut_du_texte,
    texte_lie,
)
from app.tasks.process_law import delete_from_search_index
from app.utils.chunk_refiner import est_pseudo_numero


class LawIngestRequest(BaseModel):
    file_id: str
    title: Optional[str] = None
    original_filename: Optional[str] = None
    reference: Optional[str] = None  # Optional custom reference
    category_id: Optional[int] = None  # Optional category ID


logger = logging.getLogger(__name__)

# Références fortes vers les tâches de fond (cf. create_task plus bas).
_background_tasks: set = set()

router = APIRouter(tags=["Laws"])


# ==================== DEPENDENCIES ====================


def _oublier_les_chiffres_du_corpus() -> None:
    """
    Vide les caches memoire tires du corpus : les chiffres de GET /corpus et les
    textes nommes par le statut juridique. Sans cela, un texte qu'un
    administrateur publie, archive ou supprime resterait compte, ou lien
    consultable, jusqu'a dix minutes. Une ingestion par script, dans un autre
    processus, n'y passe pas : la duree du cache la couvre.
    """
    invalider_cache_corpus()
    invalider_cache_statut()


# L'authentification reelle vit dans app/core/auth.py (JWT + bcrypt + roles).
# Un stub renvoyant {"id": 1, "role": "admin"} occupait cette place : les quatre
# endpoints d'administration ci-dessous etaient donc ouverts a tous.
# get_current_admin_user est importe en tete de fichier.


# ==================== PUBLIC ENDPOINTS ====================


class LawListItem(LawResponse, ReperesDuTexte):
    """
    Ligne de GET /laws/ : un LawResponse dont le texte peut etre allege.

    `content` y devient facultatif, pour `content_max_chars` (voir la route).
    Sans ce parametre, le texte reste ENTIER, comme avant : l'ecran
    d'administration des documents pre-remplit son formulaire d'edition avec la
    ligne de liste et renvoie `content` tel quel par PUT /laws/admin/{id}.
    Tronquer par defaut ferait ecraser le texte d'un document par son extrait
    au premier enregistrement.

    Les reperes d'une ligne dense (objet, numero, date_signature, annee,
    statut) s'y ajoutent, calcules : voir app/schemas/reperes.py.
    """

    content: Optional[str] = Field(
        None,
        description=(
            "Texte du document : entier par defaut, tronque a content_max_chars "
            "caracteres, ou null avec content_max_chars=0"
        ),
    )


# Tous les champs d'une ligne, sauf le texte : il est fourni a part, entier,
# tronque ou absent. Lire `law.content` sur une colonne differee declencherait
# un chargement implicite, interdit en asynchrone.
_CHAMPS_LIGNE_SANS_TEXTE = tuple(nom for nom in LawListItem.model_fields if nom != "content")


def _ligne_de_liste(law: Law, contenu: Optional[str]) -> LawListItem:
    """Construit une ligne de liste sans jamais lire `law.content`."""
    donnees = {nom: getattr(law, nom) for nom in _CHAMPS_LIGNE_SANS_TEXTE}
    donnees["content"] = contenu
    return LawListItem.model_validate(donnees, from_attributes=True)


def _nombre_d_articles():
    """
    Nombre d'articles de la loi courante, en sous-requete correlee.

    Compte par l'index de articles.law_id, sans charger un seul article. La
    liste tirait ce nombre de `len(law.articles)` : elle chargeait TOUS les
    articles des lois de la page — contenu et embed_text compris — pour n'en
    garder que le compte.
    """
    return (
        select(func.count(Article.id))
        .where(Article.law_id == Law.id)
        .correlate(Law)
        .scalar_subquery()
    )


async def _avec_nombre_d_articles(db: AsyncSession, law: Law) -> Law:
    """
    Pose `article_count` par un COUNT avant de rendre une loi en LawResponse.

    Law.articles ne se charge plus d'office : lire le nombre sur la collection
    declencherait un chargement implicite, que l'asynchrone refuse
    (MissingGreenlet). Une instruction, quelle que soit la taille de la loi.
    """
    law.article_count = (
        await db.execute(select(func.count(Article.id)).where(Article.law_id == law.id))
    ).scalar_one()
    return law


# Sans barre oblique finale aussi : GET /api/v1/laws repondait par une
# redirection 307 vers /api/v1/laws/ — vers http://, derriere le proxy de
# Railway, d'ou une seconde redirection 301 vers https (3,45 s mesurees avant
# le premier octet). Servie directement, l'adresse ne coute plus de detour.
# Hors du schema OpenAPI : c'est la meme route.
def _ordre_des_lois(tri: Optional[str], colonnes=None) -> tuple:
    """
    ORDER BY d'une liste : date de l'acte, puis numero du releve PRC, puis id,
    tous dans le meme sens. Les textes sans date passent en dernier.

    Des dizaines de decrets partagent une date. A date egale, l'id (ordre
    d'ingestion) melangeait les numeros d'actes : « 2026/282, 2026/281,
    2026/284 ». Le numero de la reference PRC (« PRC-10758 », ordre de mise en
    ligne sur prc.cm) suit celui des actes : mesure sur jurix_dev, pour deux
    textes de meme date et de meme nature, l'ordre PRC est celui des numeros
    d'actes dans 93,1 % des paires, l'ordre des id dans 68,4 %. Une reference
    d'une autre forme (« LOI-2012-001 ») passe apres, a date egale. L'id
    termine un ordre TOTAL : des pages qui ne se chevauchent pas.
    """
    jour = colonnes.publication_date if colonnes is not None else Law.publication_date
    ident = colonnes.id if colonnes is not None else Law.id
    reference = colonnes.reference if colonnes is not None else Law.reference
    # .cast() et non sqlalchemy.cast : ce module importe typing.cast.
    releve = case(
        (reference.op("~")("^PRC-[0-9]{1,9}$"), func.substr(reference, 5).cast(Integer))
    )
    if tri == "date_asc":
        return (jour.asc().nulls_last(), releve.asc().nulls_last(), ident.asc())
    return (jour.desc().nulls_last(), releve.desc().nulls_last(), ident.desc())


async def _page_de_lois(
    db: AsyncSession, filtres: list, tri: str, page: int, page_size: int
) -> LawListPage:
    """
    Une page legere et le total, en UNE instruction dans le cas courant.

    Le total vient de count(*) OVER () : calcule sur toutes les lignes
    filtrees avant le LIMIT, il evite un second aller-retour de comptage. Le
    nombre d'articles est compte a l'exterieur, sur les seules lignes de la
    page : dans la meme sous-requete que la fenetre, il le serait pour chaque
    texte du domaine (1 277 pour la fonction publique). Une page au-dela de la
    derniere ne rend aucune ligne, donc aucun total : un COUNT le donne alors.
    """
    tranche = (
        select(
            Law.id, Law.reference, Law.title, Law.type, Law.language, Law.status,
            Law.category_id, Category.name.label("category_name"), Law.publication_date,
            func.count().over().label("total"),
        )
        .select_from(Law)
        .outerjoin(Category, Category.id == Law.category_id)
        .where(*filtres)
        .order_by(*_ordre_des_lois(tri))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .subquery("tranche")
    )
    articles = (
        select(func.count(Article.id))
        .where(Article.law_id == tranche.c.id)
        .correlate(tranche)
        .scalar_subquery()
    )
    rangees = (
        await db.execute(
            select(tranche, articles.label("article_count")).order_by(
                *_ordre_des_lois(tri, tranche.c)
            )
        )
    ).all()
    if rangees:
        total = rangees[0].total
    else:
        total = (
            await db.execute(select(func.count()).select_from(Law).where(*filtres))
        ).scalar_one()
    return LawListPage(
        items=[
            LawListLightItem.model_validate(rangee, from_attributes=True) for rangee in rangees
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=-(-total // page_size),
        sort=tri,
    )


@router.get("/", response_model=Union[List[LawListItem], LawListPage])
@router.get("", response_model=Union[List[LawListItem], LawListPage], include_in_schema=False)
async def get_laws(
    language: Optional[str] = Query(None, description="Filter by language (fr or en)"),
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    law_status: str = Query(
        "published", description="Filter by status (published, draft, or archived)"
    ),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of records to return"),
    content_max_chars: Optional[int] = Query(
        None,
        ge=0,
        le=20000,
        description=(
            "Longueur maximale de `content`. Absent : le texte entier (defaut, "
            "dont depend l'edition d'administration). 0 : aucun texte, la "
            "colonne n'est meme pas lue. N : les N premiers caracteres, coupes "
            "par la base."
        ),
    ),
    sort: Optional[TriDesListes] = Query(
        None,
        description=(
            "date_desc ou date_asc (date de l'acte, puis id). Absent : l'ordre "
            "historique de la liste nue, sans tri ; date_desc avec `page`."
        ),
    ),
    page: Optional[int] = Query(
        None,
        ge=1,
        le=10_000_000,
        description=(
            "Page, a partir de 1. Present (ou `page_size`), la reponse devient "
            "une page legere {items, total, page, page_size, pages, sort} ; "
            "`skip`, `limit` et `content_max_chars` sont alors ignores."
        ),
    ),
    page_size: Optional[int] = Query(
        None, ge=1, le=100, description="Taille de page (defaut 20) ; voir `page`"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Get list of laws with optional filters.

    **Deux formes de reponse.** Sans `page` ni `page_size` : la liste nue
    d'origine (tableau de LawListItem), pour le front deja deploye et
    l'administration, avec `skip`/`limit`, `content_max_chars` et, s'il est
    donne, `sort`. Avec `page` ou `page_size` : une page legere (LawListPage),
    sans texte, avec le vrai total et un tri par date decroissante par defaut ;
    c'est ce qui permet une pagination par lien (`?page=2`) sans JavaScript.
    Les deux formes portent les reperes d'une ligne dense : objet, numero,
    date_signature, annee, statut.

    **Filters:**
    - `language`: Filter by language code (fr or en)
    - `category_id`: Filter by category ID
    - `law_status`: Filter by status (active or archived)

    **Pagination:**
    - `skip`: Number of records to skip (default: 0)
    - `limit`: Maximum records to return (default: 50, max: 100)

    **Example:**
    ```
    GET /api/v1/laws?language=fr&category_id=1&skip=0&limit=10
    ```

    Returns:
        List of laws matching the filters
    """
    logger.info(
        f"📋 GET /laws - language={language}, category_id={category_id}, "
        f"status={law_status}, skip={skip}, limit={limit}, sort={sort}, page={page}"
    )

    if page is not None or page_size is not None:
        filtres = []
        if language:
            filtres.append(Law.language == language)
        if category_id:
            filtres.append(Law.category_id == category_id)
        if law_status:
            filtres.append(Law.status == law_status)
        try:
            return await _page_de_lois(
                db, filtres, sort or "date_desc", page or 1, page_size or 20
            )
        except Exception as e:
            logger.error(f"❌ Error fetching laws page: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching laws"
            )

    try:
        # UNE instruction : les lois, leur categorie (jointure, Law.category
        # est en lazy="joined") et leur nombre d'articles (sous-requete). Elle
        # en coutait trois — lois, puis tous leurs articles, puis categories.
        colonnes = [Law, _nombre_d_articles()]
        if content_max_chars:
            # substr et non left : PostgreSQL ne lit alors que le debut d'un
            # texte stocke hors ligne (TOAST), au lieu de le decompresser en
            # entier.
            colonnes.append(func.substr(Law.content, 1, content_max_chars))
        query = select(*colonnes)
        if content_max_chars is not None:
            # Le texte entier ne quitte pas la base. raiseload : une lecture
            # oubliee leverait une erreur au lieu d'un chargement implicite.
            query = query.options(defer(Law.content, raiseload=True))

        # Apply filters
        if language:
            query = query.where(Law.language == language)

        if category_id:
            # Filter by category ID directly
            query = query.where(Law.category_id == category_id)

        if law_status:
            query = query.where(Law.status == law_status)

        # Sans `sort`, aucun ORDER BY, comme avant : le front deploye pagine
        # sur cet ordre-la, et l'administration aussi.
        if sort:
            query = query.order_by(*_ordre_des_lois(sort))

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        lignes = []
        for rangee in result.all():
            law = rangee[0]
            law.article_count = rangee[1]
            if content_max_chars is None:
                contenu = law.content
            elif content_max_chars == 0:
                contenu = None
            else:
                contenu = rangee[2]
            lignes.append(_ligne_de_liste(law, contenu))

        logger.info(f"✅ Found {len(lignes)} laws")
        return lignes

    except Exception as e:
        logger.error(f"❌ Error fetching laws: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching laws"
        )


@router.get("/{law_id}", response_model=LawDetailResponse)
async def get_law(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get law detail by ID with articles.

    **Includes:**
    - Law metadata
    - All articles (ordered)
    - Categories
    - Auto-detection results (language, category suggestions)

    Args:
        law_id: Law ID

    Returns:
        Law detail with articles

    Raises:
        404: Law not found
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"

    logger.info(f"📄 GET /laws/{law_id}")

    try:
        # Deux instructions : la loi avec sa categorie (jointure), puis ses
        # articles. Des articles, seules les colonnes du sommaire
        # (ArticleSummary) et de son tri : `content` et `embed_text` en etaient
        # aussi charges, soit pour le plus gros document (2 134 articles) deux
        # fois son texte de plus, pour rien — la page lit le texte dans
        # `law.content`.
        query = (
            select(Law)
            .options(
                selectinload(Law.articles).load_only(
                    Article.id,
                    Article.law_id,
                    Article.number,
                    Article.title,
                    Article.section,
                    Article.page_number,
                    Article.kind,
                    Article.order,
                )
            )
            .where(Law.id == law_id)
        )

        result = await db.execute(query)
        law = result.scalar_one_or_none()

        if not law:
            logger.warning(f"⚠️  Law {law_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Law with ID {law_id} not found"
            )

        # Les articles sont charges par le selectinload ci-dessus ; ils
        # etaient autrefois jetes par LawResponse, qui n'expose que
        # `article_count`. La page de lecture les reconstruisait donc par
        # expression reguliere sur le contenu — 193 articles la ou la base en
        # compte 230, et aucune page de PDF. `article_count` se lit sur cette
        # collection chargee.
        law.articles.sort(key=lambda a: (a.order or 0, a.id))
        reponse = LawDetailResponse.model_validate(law)
        reponse.statut_juridique = await _statut_ou_rien(db, law)
        logger.info(f"✅ Law {law_id} found: {law.title} ({len(law.articles)} articles)")
        return reponse

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching law"
        )


async def _statut_ou_rien(db: AsyncSession, law: Law) -> Optional[StatutJuridique]:
    """
    Le statut juridique du texte, ou None s'il n'a pas pu etre calcule.

    Le bloc est facultatif : une erreur dans son calcul ne doit pas priver le
    lecteur du texte lui-meme. Le front deploye ne le lit pas, le nouveau sait
    qu'il peut manquer.
    """
    try:
        return await statut_du_texte(db, law.reference, law.articles)
    except Exception as e:
        logger.error(f"❌ Statut juridique de la loi {law.id} non calcule: {e}", exc_info=True)
        return None


# Un article peut etre tres long (listes de nominations, tableaux) : l'apercu en
# garde le debut et renvoie au texte integral, au lieu de transporter 100 Ko
# sur un reseau mobile pour une feuille d'apercu.
APERCU_MAX_CARACTERES = 20_000

# L'apercu change aussi peu que le texte : le navigateur le garde cinq minutes,
# un cache partage dix. Meme logique que CACHE_CONTROL_CATEGORIES.
CACHE_CONTROL_APERCU = "public, max-age=300, s-maxage=600, stale-while-revalidate=3600"

# Variantes ecrites en base pour l'article premier : la recherche normalise
# « 1er » en « 1 », mais des numeros « 1ER » ou « PREMIER » restent possibles.
_PREMIER = ("1ER", "1ERE", "1ÈRE", "PREMIER", "PREMIERE", "PREMIÈRE")


# Colonnes d'un morceau pour l'apercu : jamais l'embedding ni embed_text.
_COLONNES_APERCU = (
    Article.id, Article.number, Article.title, Article.section,
    Article.page_number, Article.content, Article.order, Article.kind,
)


def _base_normalisee(numero: Optional[str]) -> str:
    """« 55.2 » -> « 55 », « 1er » -> « 1 » ; un pseudo-numero reste tel quel."""
    if not numero:
        return ""
    if est_pseudo_numero(numero):
        return str(numero).strip().upper()
    return normalize_number(str(numero).split(".")[0])


def _sequences(lignes: List) -> List[List]:
    """
    Les morceaux candidats, regroupes en articles.

    Un article long est coupe en morceaux qui se suivent (« 55.1 », « 55.2 »,
    ordres consecutifs) ; une compilation repete un numero dans plusieurs
    titres (« 53 » du Code civil), loin l'un de l'autre. Une sequence est une
    suite d'ordres consecutifs ; un numero sans suffixe en ouvre toujours une.
    """
    sequences: List[List] = []
    for ligne in sorted(lignes, key=lambda candidat: (candidat.order, candidat.id)):
        precedente = sequences[-1][-1] if sequences else None
        if (
            precedente is not None
            and "." in str(ligne.number)
            and ligne.order == precedente.order + 1
        ):
            sequences[-1].append(ligne)
        else:
            sequences.append([ligne])
    return sequences


def _choisir(sequences: List[List], demande: str, article_id: Optional[int]) -> List:
    """
    La sequence a montrer quand le numero en designe plusieurs.

    D'abord celle qui contient le morceau cite (`article_id`, porte par la
    citation du chat). Sinon, un vrai article plutot qu'un bout de tableau ou de
    liste qui porte le meme numero (le Code CIMA a un morceau « 13 » qui n'est
    qu'une ligne de table des matieres) ; puis le numero exact demande (« 55.2 »)
    ; puis l'ordre du texte.
    """
    for sequence in sequences:
        if any(m.id == article_id for m in sequence):
            return sequence

    def rang(sequence):
        vrai_article = (sequence[0].kind or "article") == "article"
        exact = any(str(m.number).strip().upper() == demande.upper() for m in sequence)
        return (not vrai_article, not exact, sequence[0].order)

    return min(sequences, key=rang)


def _recoller(contenus: List[str]) -> str:
    """
    Le texte d'un article coupe en morceaux, sans le chapeau recopie.

    Le decoupage (chunk_refiner) recopie en tete de chaque morceau suivant les
    200 premiers caracteres du chapeau, suivis d'un saut de ligne : utiles au
    modele, ils feraient lire deux fois la meme phrase dans l'apercu. On retire
    du morceau suivant le plus long debut qu'il partage avec le premier, coupe
    a un saut de ligne.
    """
    if not contenus:
        return ""
    premier = contenus[0] or ""
    parties = [premier.strip()]
    for suite in contenus[1:]:
        suite = suite or ""
        commun = 0
        for a, b in zip(premier, suite):
            if a != b:
                break
            commun += 1
        coupe = suite.rfind("\n", 0, commun + 1)
        parties.append((suite[coupe + 1:] if coupe > 0 else suite).strip())
    return "\n".join(p for p in parties if p)


@router.get("/{law_id}/articles/{numero}", response_model=ArticleApercuResponse)
async def get_law_article(
    law_id: int,
    response: Response,
    numero: str = PathParam(
        ..., min_length=1, max_length=64,
        description="Numéro de l'article, tel qu'affiché ou cité (« 6 », « 1er », « L 94 bis »)",
    ),
    article_id: Optional[int] = Query(
        None,
        ge=1,
        description=(
            "Identifiant d'un article de ce numéro (citation du chat) : départage les "
            "homonymes d'une compilation"
        ),
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Un article seul, avec son texte entier et son statut juridique.

    Pour la feuille d'apercu d'une citation du chat : jamais le texte entier de
    la loi, seulement les morceaux de l'article (trois instructions au plus). Un
    article coupe en morceaux (« 55.1 », « 55.2 ») est recolle ; un article dont
    la fin est rangee sous un pseudo-numero (fin de l'article 6 de la
    Constitution, morceau SECTION_2) est complete quand le releve des liens l'a
    etabli.

    Raises:
        404: texte ou article introuvable
    """
    loi = (
        await db.execute(
            select(
                Law.id, Law.reference, Law.type, Law.title, Law.publication_date, Law.status
            ).where(Law.id == law_id)
        )
    ).one_or_none()
    if loi is None:
        raise HTTPException(status_code=404, detail=f"Law with ID {law_id} not found")

    demande = numero.strip()
    vise = _base_normalisee(demande)
    # La forme demandee elle-meme aussi : normalize_number ramene « TROISIÈME »
    # a « 3 », alors que les lois de finances gardent en base leurs numeros en
    # toutes lettres.
    brut = demande.upper()
    candidats = {vise, brut, brut.split(".")[0], *(_PREMIER if vise == "1" else ())}
    numero_majuscule = func.upper(Article.number)
    lignes = (
        await db.execute(
            select(*_COLONNES_APERCU).where(
                Article.law_id == law_id,
                numero_majuscule.in_(candidats)
                | func.split_part(numero_majuscule, ".", 1).in_(candidats),
            )
        )
    ).all()

    englobant = False
    if lignes:
        sequences = _sequences(lignes)
        choisie = _choisir(sequences, demande, article_id)
    else:
        # Aucun morceau de ce numero : l'article est colle dans un autre
        # (l'article 5 de la loi 90/053 est a la fin du morceau « 2 »). On ne le
        # rend que si la citation designe ce morceau, ou si le releve des liens
        # l'a etabli ; jamais en devinant.
        sequences = []
        englobe = await _morceau_englobant(db, loi.reference, law_id, vise, article_id)
        if englobe is None:
            raise HTTPException(
                status_code=404, detail=f"Article {numero} introuvable dans le document {law_id}"
            )
        choisie, englobant = [englobe], True

    morceaux = [MorceauDuTexte(m.id, m.number, m.section, m.content) for m in choisie]
    suite = None if englobant else await _suite_etablie(db, loi.reference, law_id, vise, choisie[-1])
    if suite is not None:
        morceaux.append(suite)

    contenu = _recoller([m.contenu for m in morceaux])
    tronque = len(contenu) > APERCU_MAX_CARACTERES
    if tronque:
        contenu = contenu[:APERCU_MAX_CARACTERES].rstrip()

    premier = choisie[0]
    infos = {
        loi.reference: InfoTexte(
            loi.id, loi.reference, loi.type, loi.title, loi.publication_date, loi.status
        )
    }
    response.headers["Cache-Control"] = CACHE_CONTROL_APERCU
    return ArticleApercuResponse(
        law_id=law_id,
        texte=texte_lie(loi.reference, loi.title, loi.publication_date, infos),
        numero=vise if englobant else _numero_affiche(premier.number, len(choisie)),
        titre=premier.title,
        section=premier.section,
        page=premier.page_number,
        article_ids=[m.id for m in morceaux],
        contenu=contenu,
        tronque=tronque,
        englobant=englobant,
        homonymes=[
            ArticleHomonyme(article_id=s[0].id, section=s[0].section)
            for s in sequences
            if s is not choisie
        ],
        statut=await statut_d_un_article(db, loi.reference, morceaux),
    )


def _numero_affiche(numero: str, morceaux: int) -> str:
    """« 55.1 » ouvre un article en plusieurs morceaux : c'est l'article 55."""
    numero = str(numero).strip()
    if morceaux > 1 and "." in numero and not est_pseudo_numero(numero):
        return numero.split(".")[0]
    return numero


async def _morceau_englobant(
    db: AsyncSession, reference: str, law_id: int, vise: str, article_id: Optional[int]
):
    """
    Le morceau qui contient l'article, quand celui-ci n'a pas de morceau propre.

    Deux preuves seulement : la citation qui designe ce morceau (`article_id`,
    du meme texte), ou le releve des liens (`morceaux` : l'article 8 de la loi
    sur la Cour supreme est a la fin du morceau « 7 », repere « ARTICLE 8 »).
    """
    if article_id is not None:
        ligne = (
            await db.execute(
                select(*_COLONNES_APERCU).where(Article.id == article_id, Article.law_id == law_id)
            )
        ).first()
        if ligne is not None:
            return ligne
    for morceau, article, repere in morceaux_etablis(reference):
        if _base_normalisee(article) != vise:
            continue
        for ligne in (
            await db.execute(
                select(*_COLONNES_APERCU)
                .where(Article.law_id == law_id, Article.number == morceau)
                .order_by(Article.order)
            )
        ).all():
            if contient_le_repere(ligne.content, repere):
                return ligne
    return None


async def _suite_etablie(
    db: AsyncSession, reference: str, law_id: int, vise: str, dernier
) -> Optional[MorceauDuTexte]:
    """
    Le morceau qui suit l'article et en porte la fin, quand le releve l'a etabli.

    Seul le releve des liens le sait (`morceaux` : « SECTION_2 » contient la fin
    de l'article 6 de la Constitution, repere « par intérim »). On ne devine
    rien : ni un pseudo-numero voisin ni un intertitre ne suffisent.
    """
    attendus = [
        (morceau, repere)
        for morceau, article, repere in morceaux_etablis(reference)
        if _base_normalisee(article) == vise
    ]
    if not attendus:
        return None
    suivant = (
        await db.execute(
            select(Article.id, Article.number, Article.section, Article.content)
            .where(Article.law_id == law_id, Article.order == dernier.order + 1)
            .order_by(Article.id)
            .limit(1)
        )
    ).first()
    if suivant is None:
        return None
    for morceau, repere in attendus:
        if str(suivant.number).strip() == morceau and contient_le_repere(suivant.content, repere):
            return MorceauDuTexte(suivant.id, suivant.number, suivant.section, suivant.content)
    return None


def get_explanation_service(db: AsyncSession = Depends(get_db)) -> ExplanationService:
    """Injection d'ExplanationService, calquee sur get_rag_service."""
    return ExplanationService(db)


@router.post("/{law_id}/explain-article", response_model=ArticleExplanationResponse)
async def explain_article(
    law_id: int,
    request: ArticleExplanationRequest,
    service: ExplanationService = Depends(get_explanation_service),
) -> ArticleExplanationResponse:
    """
    Explique un article en langage courant, sur la page du document.

    Le numero voyage dans le corps et non dans le chemin : `articles.number`
    contient « 1er », « L 94 bis » ou « PREAMBULE », qu'il faudrait sinon
    encoder.

    Route PUBLIQUE, comme `GET /laws/{id}` et `POST /rag/ask` : l'explication
    ne revele rien que la page ne montre deja. Elle depense en revanche un
    appel Gemini a chaque fois, sur un quota partage avec le chat — il n'y a
    volontairement aucun cache, c'est un choix produit assume.

    Raises:
        404: document ou article introuvable
        429: quota de generation epuise
        503: service de generation sature
    """
    try:
        return await service.explain(
            law_id=law_id,
            number=request.number,
            language=request.language,
            excerpt=request.excerpt,
        )

    except ArticleNotFoundError as e:
        logger.warning(f"⚠️ Explication impossible: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ExplanationQuotaError as e:
        # 429 et non 500 : la cause est connue, elle se dit, et elle porte un
        # delai. Meme traitement que /rag/ask, dont le quota est le meme.
        logger.warning(f"⚠️ Quota epuise: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": "60"},
        )
    except ExplanationOverloadedError as e:
        # 503 et non 500 : le client sait qu'un nouvel essai a un sens.
        logger.warning(f"⚠️ Generation saturee: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "10"},
        )
    except ExplanationError as e:
        logger.error(f"❌ Explication en echec: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    except Exception as e:
        # `detail` ne reprend PAS le message : une erreur inattendue peut porter
        # une cle ou un identifiant de projet, et cette route est publique.
        logger.error(f"❌ Erreur inattendue: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur",
        )


def _download_filename(law: Law, file_path: Path) -> str:
    """
    Nom de fichier propose a l'utilisateur, construit sur le TITRE du document.

    original_filename est le nom du fichier tel qu'aspire depuis prc.cm :
    "10691_decret-n-2026-164-du-4-mai-2026-portant-approbation-des-statuts-...".
    Le prefixe numerique est l'identifiant interne du site source et les tirets
    remplacent une ponctuation qui existait — cela n'a aucun sens pour qui
    telecharge le texte.

    Le titre est donc repris tel quel, accents compris (Content-Disposition
    encode l'UTF-8 via filename*, RFC 5987), en ne retirant que les caracteres
    interdits dans un nom de fichier.
    """
    title = (law.title or "").strip()
    if not title:
        title = law.reference or file_path.stem

    # Caracteres interdits par les systemes de fichiers courants, plus les
    # caracteres de controle. Le reste, accents inclus, est conserve.
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")

    # 150 caracteres : sous la limite de 255 octets de la plupart des systemes,
    # meme apres encodage UTF-8 des accents.
    if len(cleaned) > 150:
        cleaned = cleaned[:150].rstrip()

    return f"{cleaned or 'document'}{file_path.suffix or '.pdf'}"


async def _law_file_path(law: Law) -> Path:
    """
    Chemin du fichier d'origine d'une loi, resolu UNE seule fois.

    Les endpoints qui servent un PDF repetaient la meme construction a la main,
    avec un repli qui joignait `law.file_id` au repertoire d'upload sans aucune
    verification : ni motif, ni resolve(), ni confinement. Ce n'etait pas
    exploitable — la valeur vient de la base et non de la requete — mais c'etait
    autant de copies de la faiblesse que resolve_upload_path a ete ecrite pour
    fermer.

    C'est desormais AUSSI la couture entre les deux magasins de documents. En
    mode local, rien ne change. En mode distant (`DOCUMENTS_BASE_URL` non vide),
    le document est telecharge en flux dans le cache disque, puis son chemin est
    rendu — les bibliotheques de rendu prennent un chemin, pas une URL, et un
    fichier deja sur disque se sert ensuite sans repasser en memoire.

    Raises:
        HTTPException: 404 si la loi n'a pas de fichier, si l'identifiant ne
            respecte pas le motif, ou si le document est introuvable ;
            503 si le magasin distant est en panne.
    """
    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    try:
        return await chemin_local(law.file_id)
    except IdentifiantInvalide as exc:
        logger.warning(f"⚠️ file_id invalide pour la loi {law.id}: {exc}")
        raise HTTPException(status_code=404, detail="File not found on server") from exc
    except DocumentIntrouvable as exc:
        raise HTTPException(status_code=404, detail="File not found on server") from exc
    except DocumentInaccessible as exc:
        # 503 et non 404 : c'est NOTRE magasin qui est en panne. Un 404 ferait
        # disparaitre le document de l'interface au lieu de signaler une panne
        # passagere, et accuserait l'utilisateur d'un defaut qui n'est pas le
        # sien — meme raisonnement que pour GoogleInjoignable dans /auth/google.
        logger.error(f"⚠️ Magasin de documents injoignable pour la loi {law.id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Document momentanement indisponible",
            headers={"Retry-After": "30"},
        ) from exc


async def _loi_pour_fichier(db: AsyncSession, law_id: int) -> Optional[Law]:
    """
    La loi, reduite aux colonnes que lisent les routes de fichier.

    Elles chargeaient la ligne ENTIERE, texte integral compris, pour n'en lire
    que `file_id` et le titre : jusqu'a 1,76 million de caracteres venus de la
    base a chaque page de PDF affichee (/page/{n} est appele une fois par
    page), et, avant que Law.articles ne soit plus charge d'office, tous les
    articles en plus.
    """
    query = (
        select(Law)
        .options(
            load_only(Law.id, Law.file_id, Law.title, Law.reference, Law.original_filename)
        )
        .where(Law.id == law_id)
    )
    return (await db.execute(query)).scalar_one_or_none()


@router.get("/{law_id}/download")
async def download_law_file(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Download the original uploaded file for a law."""
    law = await _loi_pour_fichier(db, law_id)

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    file_path = await _law_file_path(law)

    return FileResponse(
        path=str(file_path), 
        filename=_download_filename(law, file_path),
        media_type="application/pdf" if file_path.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # attachment et non inline : un bouton nomme "Telecharger" doit
        # enregistrer le fichier, pas l'afficher dans l'onglet.
        content_disposition_type="attachment"
    )


# /pdf-data et /pdf-stream ont ete SUPPRIMES.
#
# Aucun client ne les appelait — ni le front, ni les tests, ni un script. Mais
# tous deux lisaient le fichier entier en memoire (`f.read()`), et /pdf-data y
# ajoutait un encodage base64, qui gonfle de 33 %. Sur le document le plus lourd
# du corpus (26 Mo), un seul appel demandait donc une pointe de ~61 Mo dans un
# conteneur qui n'en a que 512 — le plus gros risque de saturation memoire du
# service, au profit de personne.
#
# Le besoin d'origine (contourner les gestionnaires de telechargement qui
# interceptent les requetes) est couvert par /download, qui sert le fichier en
# flux depuis le disque sans jamais le charger en memoire.

@router.get("/{law_id}/pdf-info")
async def get_law_pdf_info(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get PDF information including page count.
    Used by the frontend to know how many pages to request.
    """
    from pypdf import PdfReader

    law = await _loi_pour_fichier(db, law_id)

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    file_path = await _law_file_path(law)

    # Get page count
    try:
        reader = PdfReader(str(file_path))
        page_count = len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")
    
    return {
        "law_id": law_id,
        "filename": law.original_filename or file_path.name,
        "page_count": page_count
    }


@router.get("/{law_id}/page/{page_num}")
async def get_law_pdf_page_image(
    law_id: int,
    page_num: int,
    dpi: int = Query(default=120, ge=72, le=200, description="DPI for rendering (72-200)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Render a specific PDF page as a JPEG image.
    
    Uses Poppler (pdf2image) for reliable rendering of all image formats
    including JPEG2000 which pdf.js cannot handle.
    
    Args:
        law_id: ID of the law
        page_num: Page number (1-indexed)
        dpi: Resolution for rendering (default 120, max 200 for performance)
    
    Returns:
        JPEG image of the specified page
    """
    from io import BytesIO

    from fastapi.responses import Response

    law = await _loi_pour_fichier(db, law_id)

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    file_path = await _law_file_path(law)

    # Convert page to image using Poppler
    try:
        from pdf2image import convert_from_path
        
        # Convert only the specific page (1-indexed)
        images = convert_from_path(
            str(file_path),
            first_page=page_num,
            last_page=page_num,
            dpi=dpi,
            fmt="jpeg",
        )
        
        if not images:
            raise HTTPException(status_code=404, detail=f"Page {page_num} not found")
        
        # Convert to JPEG bytes
        img_buffer = BytesIO()
        images[0].save(img_buffer, format="JPEG", quality=85)
        img_bytes = img_buffer.getvalue()
        
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                "X-Page-Number": str(page_num),
            }
        )
        
    except Exception as e:
        logger.error(f"Error rendering page {page_num} for law {law_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error rendering page: {str(e)}")


# ==================== INGESTION ENDPOINTS ====================


@router.post("/admin/ingest", response_model=LawResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_law(
    request: LawIngestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Ingest a new law from an uploaded file.

    1. Creates a Law record (status=processing)
    2. Declenche le traitement du fichier en tache de fond

    Args:
        request: Ingest request (file_id, title)

    Returns:
        Created Law object (pending)
    """
    assert request is not None, "LawIngestRequest must not be None"
    assert isinstance(request.file_id, str) and len(request.file_id) > 0, "file_id must be a non-empty string"

    logger.info(f"📥 POST /admin/laws/ingest - File: {request.file_id}")

    try:
        # Validate file_id
        if not request.file_id or len(request.file_id) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file_id: must be at least 8 characters",
            )

        # Check if law with this file_id already exists (idempotency)
        ref_prefix = f"PENDING-{request.file_id[:8]}"

        # Check for existing pending law to avoid duplicates
        query = select(Law).where(Law.reference == ref_prefix)
        result = await db.execute(query)
        existing_law = result.scalar_one_or_none()

        if existing_law and existing_law.status == "processing":
            logger.info(f"ℹ️ Law already exists for file {request.file_id}: ID={existing_law.id}")
            return await _avec_nombre_d_articles(db, existing_law)

        # If existing law failed, create a new one with timestamp
        if existing_law:
            ref_prefix = f"{ref_prefix}-{int(time.time())}"

        # Use custom reference if provided, otherwise use PENDING prefix
        law_reference = request.reference if request.reference else ref_prefix

        # Check if reference already exists (collision check)
        existing_ref_query = select(Law).where(Law.reference == law_reference)
        existing_ref_result = await db.execute(existing_ref_query)
        existing_ref_law = existing_ref_result.scalar_one_or_none()

        if existing_ref_law:
            logger.info(f"ℹ️ Law with reference '{law_reference}' already exists: ID={existing_ref_law.id}. Returning existing law.")
            
            # If it was stuck or previous attempt failed, we might want to trigger processing again
            # For now, just return it so the frontend sees it's there
            if existing_ref_law.status == "processing" or existing_ref_law.content == "Document en cours de traitement par le système.":
                 # Potentially restart failed processing here if needed
                 pass

            return await _avec_nombre_d_articles(db, existing_ref_law)

        # Create initial Law record
        new_law = Law(
            reference=law_reference,
            title=request.title or "Document en cours de traitement",
            type="autre",  # Changed from "unknown" to valid type
            content="Document en cours de traitement par le système.",  # Min 10 chars required
            # "processing" et non "published" : le document n'a pas encore ete
            # extrait. Le publier d'emblee le rendait visible dans le corpus
            # public avec son contenu de remplacement ("Document en cours de
            # traitement par le systeme."), et un echec d'OCR l'y laissait
            # indefiniment. Le pipeline le passe a "published" en cas de succes,
            # a "refused" en cas d'echec.
            status="processing",
            category_id=request.category_id,  # Category selected by admin
            file_id=request.file_id,
            original_filename=request.original_filename,
        )

        db.add(new_law)
        await db.commit()
        await db.refresh(new_law)
        # Creee a l'instant : aucun article ne la reference encore (le
        # pipeline les ecrira). Pose ici, la collection n'etant plus chargee
        # d'office.
        new_law.article_count = 0

        # Lance le traitement en arrière-plan (BackgroundTasks FastAPI)
        # Traitement dans une tache asyncio, sans courtier de messages
        # sans bloquer la réponse HTTP
        law_id_for_bg = cast(int, new_law.id)
        file_id_for_bg = request.file_id

        async def _process_and_invalidate():
            from app.tasks.process_law import process_law_async
            try:
                result = await process_law_async(law_id_for_bg, file_id_for_bg)
                logger.info(f"✅ Background processing completed: {result}")
                # Invalider le cache après traitement
                async with AsyncSessionLocal() as cache_db:
                    await invalidate_search_cache(cache_db)
                _oublier_les_chiffres_du_corpus()
            except Exception as bg_err:
                logger.error(f"❌ Background processing failed: {bg_err}", exc_info=True)

        import asyncio as _asyncio

        # Reference forte conservee : asyncio ne garde qu'une reference FAIBLE
        # sur les taches, une tache non referencee peut etre collectee en plein
        # traitement et le document rester bloque en "processing".
        _task = _asyncio.create_task(_process_and_invalidate())
        _background_tasks.add(_task)
        _task.add_done_callback(_background_tasks.discard)
        logger.info(f"🚀 Background task started for Law ID={new_law.id}")

        return new_law

    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        await db.rollback()
        error_msg = f"Error starting ingestion: {type(e).__name__}: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)


# ==================== ADMIN ENDPOINTS ====================


@router.post("/admin", response_model=LawResponse, status_code=status.HTTP_201_CREATED)
async def create_law(
    law: LawCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Create a new law (admin only).

    **Auto-Detection (v2.1):**
    - If `language` not provided, auto-detected from content
    - Category suggestions provided based on content

    **Required Fields:**
    - reference: Unique law reference
    - title: Law title
    - type: Law type (loi, décret, ordonnance, etc.)
    - content: Full text content

    Args:
        law: Law creation data

    Returns:
        Created law with auto-detection results

    Raises:
        400: Validation error
        401: Unauthorized
        409: Law reference already exists
    """
    assert law is not None, "LawCreate must not be None"
    assert isinstance(law.reference, str) and len(law.reference) > 0, "Law reference must be a non-empty string"

    logger.info(f"➕ POST /admin/laws - Creating law: {law.title}")

    try:
        # Check if reference already exists
        existing_query = select(Law).where(Law.reference == law.reference)
        existing_result = await db.execute(existing_query)
        existing_law = existing_result.scalar_one_or_none()

        if existing_law:
            logger.warning(f"⚠️  Law reference {law.reference} already exists")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Law with reference '{law.reference}' already exists",
            )

        # Create law instance
        new_law = Law(**law.model_dump())

        # TODO: Integrate auto-detection services
        # if not new_law.language:
        #     language_service = LanguageDetector()
        #     detection = language_service.detect(new_law.content)
        #     new_law.detected_language = detection["language"]
        #     new_law.language_confidence = detection["confidence"]

        # Add to database
        db.add(new_law)
        await db.commit()
        await db.refresh(new_law)
        # Creee a l'instant : aucun article ne la reference encore.
        new_law.article_count = 0
        _oublier_les_chiffres_du_corpus()

        logger.info(f"✅ Law created: ID={new_law.id}, reference={new_law.reference}")
        return new_law

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Error creating law: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error creating law"
        )


@router.put("/admin/{law_id}", response_model=LawResponse)
async def update_law(
    law_id: int,
    law_update: LawUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Update an existing law (admin only).

    **Partial Updates:**
    - Only provided fields are updated
    - Omitted fields remain unchanged

    Args:
        law_id: Law ID to update
        law_update: Fields to update

    Returns:
        Updated law

    Raises:
        404: Law not found
        401: Unauthorized
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
    assert law_update is not None, "LawUpdate must not be None"

    logger.info(f"✏️  PUT /admin/laws/{law_id}")

    try:
        # Fetch existing law
        query = select(Law).where(Law.id == law_id)
        result = await db.execute(query)
        existing_law = result.scalar_one_or_none()

        if not existing_law:
            logger.warning(f"⚠️  Law {law_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Law with ID {law_id} not found"
            )

        # Update fields
        update_data = law_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing_law, field, value)

        await db.commit()
        await db.refresh(existing_law)
        _oublier_les_chiffres_du_corpus()

        logger.info(f"✅ Law {law_id} updated")
        return await _avec_nombre_d_articles(db, existing_law)

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Error updating law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error updating law"
        )


@router.delete("/admin/{law_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_law(
    law_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Delete a law (admin only).

    **Cascade Delete:**
    - Deletes law and all associated articles
    - Removes from search index

    Args:
        law_id: Law ID to delete

    Returns:
        204 No Content on success

    Raises:
        404: Law not found
        401: Unauthorized
    """
    logger.info(f"🗑️  DELETE /admin/laws/{law_id}")

    try:
        # Fetch law
        query = select(Law).where(Law.id == law_id)
        result = await db.execute(query)
        law = result.scalar_one_or_none()

        if not law:
            logger.warning(f"⚠️  Law {law_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Law with ID {law_id} not found"
            )

        # Delete law (cascade deletes articles)
        await db.delete(law)
        await db.commit()

        # Vide le search_vector PostgreSQL (remplace suppression la recherche plein texte)
        delete_from_search_index(law_id)

        # Invalider le cache de recherche
        await invalidate_search_cache(db)
        _oublier_les_chiffres_du_corpus()
        
        logger.info(f"✅ Law {law_id} deleted")
        return None

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Error deleting law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error deleting law"
        )
