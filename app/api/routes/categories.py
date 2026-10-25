"""
API Routes for Category Management (Service 8).

Endpoints:
- POST /api/v1/categories - Create category
- GET /api/v1/categories - List categories
- GET /api/v1/categories/{id} - Get single category
- PUT /api/v1/categories/{id} - Update category
- DELETE /api/v1/categories/{id} - Delete category
- GET /api/v1/categories/{id}/stats - Get category statistics
- GET /api/v1/categories/stats/all - Get all category statistics
- GET /api/v1/categories/mapping/id-to-name - Get ID → Name mapping
- GET /api/v1/categories/mapping/name-to-id - Get Name → ID mapping
- GET /api/v1/categories/health - Health check

Author: JuriX Team
Version: 2.1.0
"""

import itertools
import logging
import threading
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import event
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.auth import get_current_admin_user
from app.core.database import get_db
from app.models.law import Category, Law
from app.models.user import User
from app.schemas.law import CategoryCreate, CategoryResponse, CategoryStats, CategoryUpdate
from app.services.category_service import (
    CategoryInUseError,
    CategoryNotFoundError,
    CategoryService,
    CategoryServiceError,
    DuplicateCategoryNameError,
)

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# CACHE DE LA LISTE DES CATEGORIES
# ============================================================================
#
# POURQUOI. GET /categories est appele par l'accueil, /laws et /search, et sa
# reponse ne change qu'avec une ecriture d'administration. Mesure en
# production : 1,7 ms de SQL, mais 8 allers-retours de 83 ms vers le pooler
# (0,55 a 0,95 s cote serveur). En memoire, la reponse ne coute plus rien
# pendant la duree de vie du cache. GET /categories/{id} lit le meme cache : les
# 14 domaines y sont, comptes compris.
#
# 600 s : les comptes ne bougent qu'a l'ajout, la suppression ou le
# reclassement d'une loi. Les ecritures faites PAR CE PROCESSUS invalident le
# cache sur-le-champ (routes ci-dessous, et crochets ORM en fin de section).
# La duree ne borne donc que les ecritures faites ailleurs — un script de
# maintenance lance contre la base de production — et un compte en retard de
# dix minutes n'y trompe personne. Un seul processus sert l'API
# (`--workers 1`, Dockerfile) : un cache par processus suffit.
_DUREE_CACHE_S = 600.0

# La limite maximale de la route. Une liste lue en entier en contient moins ;
# si elle l'atteint, elle a pu etre tronquee, et les pages au-dela se lisent en
# base.
_PLAFOND_CACHE = 1000

# En-tete des reponses REUSSIES (une erreur n'est jamais mise en cache).
#   public  : la reponse ne depend ni du compte ni d'un cookie. Les ecrans
#             d'administration l'appellent avec un jeton ; sans `public`, un
#             cache partage ne garderait pas une reponse a une requete
#             authentifiee.
#   max-age=60 : le navigateur la reutilise une minute — la page de recherche
#             la redemande a chaque visite, et depuis le Cameroun chaque appel
#             a Railway coute 0,5 a 1 s. Court pour qu'une modification
#             d'administration se voie vite.
#   s-maxage=300 : un cache partage (CDN, proxy) peut la garder cinq minutes.
#             Aucun n'est sur le chemin aujourd'hui (Railway ne met rien en
#             cache, les rendus serveur de Vercel non plus) ; la valeur reste
#             en dessous de la duree du cache memoire, et un cache partage ne
#             recoit pas nos invalidations.
#   stale-while-revalidate=600 : passe ce delai, la copie perimee sert encore
#             pendant que la nouvelle se charge en arriere-plan. Un compte en
#             retard est sans consequence ; une attente ne l'est pas.
CACHE_CONTROL_CATEGORIES = "public, max-age=60, s-maxage=300, stale-while-revalidate=600"


class _EntreeCache(NamedTuple):
    lue_a: float
    categories: Tuple[CategoryResponse, ...]


_cache_categories: Optional[_EntreeCache] = None
# Generation : incrementee a chaque invalidation. Une lecture lancee AVANT une
# ecriture et terminee APRES ne doit pas reinstaller l'etat d'avant pour dix
# minutes : elle ne garde son resultat que si la generation n'a pas bouge.
_generations = itertools.count(1)
_generation = 0
# Les ecritures du pipeline arrivent depuis des fils d'executor (sessions
# synchrones) : invalidation et installation passent par ce verrou.
_verrou_cache = threading.Lock()


def invalider_cache_categories() -> None:
    """Oublie la liste en cache : la prochaine lecture repassera par la base."""
    global _cache_categories, _generation
    with _verrou_cache:
        _generation = next(_generations)
        _cache_categories = None


async def _categories_avec_comptes(db: AsyncSession) -> Tuple[CategoryResponse, ...]:
    """
    Toutes les categories, avec leur nombre de lois, depuis le cache si possible.

    En base : UNE instruction (le GROUP BY de CategoryService.list_categories).
    """
    global _cache_categories
    entree = _cache_categories
    maintenant = time.monotonic()
    if entree is not None and maintenant - entree.lue_a < _DUREE_CACHE_S:
        return entree.categories

    generation = _generation
    categories = tuple(
        await CategoryService(db).list_categories(skip=0, limit=_PLAFOND_CACHE)
    )
    with _verrou_cache:
        if generation == _generation:
            # Horodatee au DEBUT de la lecture : la donnee a cet age-la.
            _cache_categories = _EntreeCache(maintenant, categories)
    return categories


# --- Invalidation par les ecritures de l'ORM --------------------------------
#
# Les lois s'ecrivent depuis plusieurs chemins de ce processus : routes
# d'administration (laws.py, batch_upload.py), pipeline d'ingestion (sessions
# synchrones), suppression forcee d'une categorie. Plutot que d'appeler
# l'invalidation dans chacun — et d'en oublier un —, deux crochets sur la classe
# Session les voient tous, sessions asynchrones comprises (AsyncSession
# s'appuie sur une Session). Seules les ecritures qui changent la reponse
# comptent : une loi ajoutee ou supprimee, une loi qui change de categorie, une
# categorie creee, modifiee ou supprimee. Le statut d'une loi n'y est pas : les
# comptes portent sur toutes les lois, quel que soit leur statut.
#
# Le marquage se fait au flush, l'invalidation au COMMIT : invalider au flush
# laisserait une lecture concurrente remettre en cache l'etat d'avant, encore
# seul visible tant que la transaction n'est pas validee.
#
# Ce que l'ORM ne voit pas (SQL brut, autre processus) reste borne par
# _DUREE_CACHE_S.
_CLE_COMPTES_PERIMES = "jurix_categories_perimees"


def _ecriture_touche_les_categories(session: Session) -> bool:
    for objet in itertools.chain(session.new, session.deleted):
        if isinstance(objet, (Law, Category)):
            return True
    for objet in session.dirty:
        if isinstance(objet, Category):
            return True
        if isinstance(objet, Law) and sa_inspect(objet).attrs.category_id.history.has_changes():
            return True
    return False


@event.listens_for(Session, "after_flush")
def _noter_ecriture_categories(session: Session, _flush_context) -> None:
    # after_flush voit encore l'etat d'AVANT le flush : new, dirty, deleted et
    # l'historique des attributs.
    if _ecriture_touche_les_categories(session):
        session.info[_CLE_COMPTES_PERIMES] = True


@event.listens_for(Session, "after_commit")
def _invalider_apres_commit(session: Session) -> None:
    if session.info.pop(_CLE_COMPTES_PERIMES, False):
        invalider_cache_categories()


@event.listens_for(Session, "after_rollback")
def _oublier_apres_annulation(session: Session) -> None:
    session.info.pop(_CLE_COMPTES_PERIMES, None)

# Create router
router = APIRouter(
    tags=["categories"],
    responses={
        400: {"description": "Bad Request - Validation error"},
        404: {"description": "Not Found - Category doesn't exist"},
        409: {"description": "Conflict - Category in use or duplicate name"},
        500: {"description": "Internal Server Error"}
    }
)


# ============================================================================
# CRUD ENDPOINTS
# ============================================================================

@router.post(
    "",
    response_model=CategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new category"
)
async def create_category(
    category_data: CategoryCreate,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> CategoryResponse:
    """
    Create a new legal category.

    **Parameters:**
    - **name**: Category name (unique, 1-100 characters)
    - **description**: Optional description

    **Returns:**
    - Created category with ID and metadata

    **Raises:**
    - 400: Invalid input data
    - 409: Category name already exists
    - 500: Server error
    """
    assert category_data is not None, "CategoryCreate must not be None"
    assert isinstance(category_data.name, str) and len(category_data.name) > 0, "Category name must be a non-empty string"

    try:
        logger.info(f"📥 POST /categories - Creating: {category_data.name}")
        service = CategoryService(db)
        category = await service.create_category(category_data)
        # Explicite, en plus du crochet ORM : la route qui ecrit dit ce
        # qu'elle perime.
        invalider_cache_categories()

        # Build response
        law_count = await service._get_law_count_for_category(category.id)
        response = CategoryResponse(
            id=category.id,
            name=category.name,
            description=category.description,
            created_at=category.created_at,
            law_count=law_count
        )

        logger.info(f"✅ Category created: {category.name} (ID: {category.id})")
        return response

    except DuplicateCategoryNameError as e:
        logger.warning(f"⚠️  Duplicate category name: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create category: {str(e)}"
        )


@router.get(
    "",
    response_model=List[CategoryResponse],
    summary="List all categories"
)
async def list_categories(
    response: Response,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: AsyncSession = Depends(get_db)
) -> List[CategoryResponse]:
    """
    List all legal categories with law counts.

    **Parameters:**
    - **skip**: Pagination offset (default: 0)
    - **limit**: Max results (default: 100, max: 1000)

    **Returns:**
    - List of categories ordered by name
    - Each category includes law_count

    **Example Response:**
    ```json
    [
        {
            "id": 1007,
            "name": "Droit Civil et Procédure Civile",
            "description": "Personnes, obligations, contrats, instances civiles et voies d'exécution",
            "created_at": "2024-01-01T00:00:00",
            "law_count": 45
        }
    ]
    ```
    """
    try:
        logger.info(f"📥 GET /categories (skip={skip}, limit={limit})")
        toutes = await _categories_avec_comptes(db)
        if len(toutes) >= _PLAFOND_CACHE and skip + limit > len(toutes):
            # Liste peut-etre tronquee au plafond : la page demandee se lit en
            # base, telle que la route la servait avant le cache.
            categories = await CategoryService(db).list_categories(skip=skip, limit=limit)
        else:
            # Meme decoupage qu'OFFSET/LIMIT sur la meme requete, triee par
            # display_order.
            categories = list(toutes[skip:skip + limit])

        response.headers["Cache-Control"] = CACHE_CONTROL_CATEGORIES
        logger.info(f"✅ Returning {len(categories)} categories")
        return categories

    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list categories: {str(e)}"
        )


# NOTE: declaree avant "/{category_id}" — FastAPI resout dans l'ordre de
# declaration, la route parametree capturait sinon ce chemin.
@router.get(
    "/health",
    response_model=Dict[str, str],
    summary="Health check"
)
async def health_check(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Check CategoryService health.

    **Returns:**
    - Service status and metadata

    **Example Response:**
    ```json
    {
        "service": "CategoryService",
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": "2024-01-11T12:00:00"
    }
    ```
    """
    try:
        logger.debug("📥 GET /categories/health")
        service = CategoryService(db)
        health = service.health_check()

        logger.debug("✅ Health check passed")
        return health

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "service": "CategoryService",
            "status": "unhealthy",
            "error": str(e)
        }


@router.get(
    "/{category_id}",
    response_model=CategoryResponse,
    summary="Get a single category"
)
async def get_category(
    category_id: int,
    response: Response,
    db: AsyncSession = Depends(get_db)
) -> CategoryResponse:
    """
    Get a category by ID.

    **Parameters:**
    - **category_id**: Category ID

    **Returns:**
    - Category details with law count

    **Raises:**
    - 404: Category not found
    - 500: Server error
    """
    assert isinstance(category_id, int) and category_id > 0, "Category ID must be a positive integer"

    try:
        logger.info(f"📥 GET /categories/{category_id}")
        # La liste en cache porte exactement les champs de cette reponse,
        # compte compris : la categorie s'y lit sans aller en base. Cette route
        # chargeait auparavant la categorie, TOUTES ses lois puis tous leurs
        # articles (relations en selectin) avant de compter : 4 instructions
        # et 1 262 articles hydrates pour 206 octets.
        toutes = await _categories_avec_comptes(db)
        trouvee = next((c for c in toutes if c.id == category_id), None)
        if trouvee is None:
            # Absente du cache : inconnue, ou creee par un autre processus
            # depuis la lecture. La base tranche — deux instructions, la
            # categorie seule puis un COUNT, aucune loi chargee.
            service = CategoryService(db)
            category = await service.get_category(category_id)
            law_count = await service._get_law_count_for_category(category.id)
            trouvee = CategoryResponse(
                id=category.id,
                name=category.name,
                description=category.description,
                icon=category.icon,
                created_at=category.created_at,
                law_count=law_count
            )
            # Le cache ignorait une categorie qui existe : il est en retard.
            invalider_cache_categories()

        response.headers["Cache-Control"] = CACHE_CONTROL_CATEGORIES
        logger.info(f"✅ Category found: {trouvee.name}")
        return trouvee

    except CategoryNotFoundError as e:
        logger.warning(f"⚠️  Category not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch category: {str(e)}"
        )


@router.put(
    "/{category_id}",
    response_model=CategoryResponse,
    summary="Update a category"
)
async def update_category(
    category_id: int,
    category_data: CategoryUpdate,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> CategoryResponse:
    """
    Update a category.

    **Parameters:**
    - **category_id**: Category ID
    - **name**: New name (optional)
    - **description**: New description (optional)

    **Returns:**
    - Updated category

    **Raises:**
    - 404: Category not found
    - 409: New name conflicts with existing category
    - 500: Server error
    """
    assert isinstance(category_id, int) and category_id > 0, "category_id must be a positive integer"
    assert category_data is not None, "CategoryUpdate must not be None"

    try:
        logger.info(f"📥 PUT /categories/{category_id}")
        service = CategoryService(db)
        category = await service.update_category(category_id, category_data)
        invalider_cache_categories()

        # Build response
        law_count = await service._get_law_count_for_category(category.id)
        response = CategoryResponse(
            id=category.id,
            name=category.name,
            description=category.description,
            created_at=category.created_at,
            law_count=law_count
        )

        logger.info(f"✅ Category updated: {category.name}")
        return response

    except CategoryNotFoundError as e:
        logger.warning(f"⚠️  Category not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DuplicateCategoryNameError as e:
        logger.warning(f"⚠️  Duplicate category name: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update category: {str(e)}"
        )


@router.delete(
    "/{category_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a category"
)
async def delete_category(
    category_id: int,
    force: bool = Query(False, description="Force delete even if laws are associated"),
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
):
    """
    Delete a category.

    **Parameters:**
    - **category_id**: Category ID
    - **force**: If true, deletes category even if laws are associated
                 (laws will have category_id set to NULL)

    **Returns:**
    - 204 No Content on success

    **Raises:**
    - 404: Category not found
    - 409: Category has associated laws and force=false
    - 500: Server error

    **Warning:**
    Using force=true will set category_id to NULL for all associated laws.
    Use with caution in production!
    """
    assert isinstance(category_id, int) and category_id > 0, "category_id must be a positive integer"

    try:
        logger.info(f"📥 DELETE /categories/{category_id} (force={force})")
        service = CategoryService(db)
        await service.delete_category(category_id, force=force)
        invalider_cache_categories()

        logger.info(f"✅ Category deleted: ID {category_id}")
        return None  # 204 No Content

    except CategoryNotFoundError as e:
        logger.warning(f"⚠️  Category not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except CategoryInUseError as e:
        logger.warning(f"⚠️  Category in use: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete category: {str(e)}"
        )


# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@router.get(
    "/{category_id}/stats",
    response_model=CategoryStats,
    summary="Get category statistics"
)
async def get_category_stats(
    category_id: int,
    db: AsyncSession = Depends(get_db)
) -> CategoryStats:
    """
    Get statistics for a single category.

    **Parameters:**
    - **category_id**: Category ID

    **Returns:**
    - Category statistics (law count, percentage)

    **Example Response:**
    ```json
    {
        "category_id": 1007,
        "category_name": "Droit Civil et Procédure Civile",
        "law_count": 45,
        "percentage": 26.47
    }
    ```

    **Raises:**
    - 404: Category not found
    - 500: Server error
    """
    try:
        logger.info(f"📥 GET /categories/{category_id}/stats")
        service = CategoryService(db)
        stats = await service.get_category_stats(category_id)

        logger.info(f"✅ Stats returned: {stats.law_count} laws ({stats.percentage:.2f}%)")
        return stats

    except CategoryNotFoundError as e:
        logger.warning(f"⚠️  Category not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch category stats: {str(e)}"
        )


@router.get(
    "/stats/all",
    response_model=List[CategoryStats],
    summary="Get statistics for all categories"
)
async def get_all_category_stats(
    db: AsyncSession = Depends(get_db)
) -> List[CategoryStats]:
    """
    Get statistics for all categories.

    **Returns:**
    - List of category statistics ordered by law count (descending)
    - Each includes: category ID, name, law count, percentage

    **Example Response:**
    ```json
    [
        {
            "category_id": 1007,
            "category_name": "Droit Civil et Procédure Civile",
            "law_count": 45,
            "percentage": 26.47
        },
        {
            "category_id": 1006,
            "category_name": "Droit Pénal et Procédure Pénale",
            "law_count": 38,
            "percentage": 22.35
        }
    ]
    ```

    **Use Case:**
    - Dashboard analytics
    - Category distribution charts
    - Admin overview
    """
    try:
        logger.info("📥 GET /categories/stats/all")
        service = CategoryService(db)
        stats_list = await service.get_all_category_stats()

        logger.info(f"✅ Stats returned for {len(stats_list)} categories")
        return stats_list

    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch all category stats: {str(e)}"
        )


# ============================================================================
# MAPPING UTILITY ENDPOINTS
# ============================================================================

@router.get(
    "/mapping/id-to-name",
    response_model=Dict[int, str],
    summary="Get ID → Name mapping"
)
async def get_id_to_name_mapping(
    db: AsyncSession = Depends(get_db)
) -> Dict[int, str]:
    """
    Get ID → Name mapping for all categories.

    **Returns:**
    - Dictionary mapping category IDs to names

    **Example Response:**
    ```json
    {
        "1007": "Droit Civil et Procédure Civile",
        "1006": "Droit Pénal et Procédure Pénale",
        "1010": "Droit des Affaires, Banque et OHADA"
    }
    ```

    **Use Case:**
    - Fast lookups without database queries
    - Frontend dropdown populations
    - DocumentClassifier integration
    """
    try:
        logger.info("📥 GET /categories/mapping/id-to-name")
        service = CategoryService(db)
        mapping = await service.get_category_mapping()

        logger.info(f"✅ Mapping returned: {len(mapping)} categories")
        return mapping

    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch ID → Name mapping: {str(e)}"
        )


@router.get(
    "/mapping/name-to-id",
    response_model=Dict[str, int],
    summary="Get Name → ID mapping"
)
async def get_name_to_id_mapping(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, int]:
    """
    Get Name → ID mapping for all categories.

    **Returns:**
    - Dictionary mapping category names to IDs

    **Example Response:**
    ```json
    {
        "Droit Civil et Procédure Civile": 1007,
        "Droit Pénal et Procédure Pénale": 1006,
        "Droit des Affaires, Banque et OHADA": 1010
    }
    ```

    **Use Case:**
    - Reverse lookups by name
    - Category name validation
    - Import/export utilities
    """
    try:
        logger.info("📥 GET /categories/mapping/name-to-id")
        service = CategoryService(db)
        mapping = await service.get_reverse_category_mapping()

        logger.info(f"✅ Reverse mapping returned: {len(mapping)} categories")
        return mapping

    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch Name → ID mapping: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

