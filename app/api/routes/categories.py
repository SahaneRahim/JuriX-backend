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

import logging
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.category_service import (
    CategoryService,
    CategoryNotFoundError,
    DuplicateCategoryNameError,
    CategoryInUseError,
    CategoryServiceError
)
from app.schemas.law import CategoryCreate, CategoryUpdate, CategoryResponse, CategoryStats

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/categories",
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
    db: AsyncSession = Depends(get_db)
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
            "id": 1,
            "name": "Droit Civil",
            "description": "Civil law",
            "created_at": "2024-01-01T00:00:00",
            "law_count": 45
        }
    ]
    ```
    """
    try:
        logger.info(f"📥 GET /categories (skip={skip}, limit={limit})")
        service = CategoryService(db)
        categories = await service.list_categories(skip=skip, limit=limit)

        # DEBUG: Print first category details
        if categories:
            first = categories[0]
            logger.info(f"DEBUG: First category: ID={first.id}, Name={first.name}, Icon={getattr(first, 'icon', 'MISSING')}")

        logger.info(f"✅ Returning {len(categories)} categories")
        return categories

    except CategoryServiceError as e:
        logger.error(f"❌ Service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list categories: {str(e)}"
        )


@router.get(
    "/{category_id}",
    response_model=CategoryResponse,
    summary="Get a single category"
)
async def get_category(
    category_id: int,
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
        logger.info(f"�� GET /categories/{category_id}")
        service = CategoryService(db)
        category = await service.get_category(category_id)

        # Build response
        law_count = await service._get_law_count_for_category(category.id)
        response = CategoryResponse(
            id=category.id,
            name=category.name,
            description=category.description,
            icon=category.icon,
            created_at=category.created_at,
            law_count=law_count
        )

        logger.info(f"✅ Category found: {category.name}")
        return response

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
    db: AsyncSession = Depends(get_db)
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
    db: AsyncSession = Depends(get_db)
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
        "category_id": 1,
        "category_name": "Droit Civil",
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
            "category_id": 1,
            "category_name": "Droit Civil",
            "law_count": 45,
            "percentage": 26.47
        },
        {
            "category_id": 2,
            "category_name": "Droit Pénal",
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
        "1": "Droit Civil",
        "2": "Droit Pénal",
        "3": "Droit Commercial"
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
        "Droit Civil": 1,
        "Droit Pénal": 2,
        "Droit Commercial": 3
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
