"""
Service 8: CategoryService - Gestion des catégories juridiques.

Features:
- CRUD catégories (create, read, update, delete, list)
- Mapping ID ↔ Name (bidirectional lookups)
- Statistiques par catégorie (law counts, distributions)
- Health check endpoint

Author: JuriX Team
Version: 2.1.0
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from app.models.law import Category, Law
from app.schemas.law import CategoryCreate, CategoryUpdate, CategoryResponse, CategoryStats

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class CategoryServiceError(Exception):
    """Base exception for CategoryService operations."""
    pass


class CategoryNotFoundError(CategoryServiceError):
    """Raised when a category is not found."""
    pass


class DuplicateCategoryNameError(CategoryServiceError):
    """Raised when trying to create/update a category with duplicate name."""
    pass


class CategoryInUseError(CategoryServiceError):
    """Raised when trying to delete a category that has associated laws."""
    pass


# ============================================================================
# CATEGORY SERVICE
# ============================================================================

class CategoryService:
    """
    Service for managing legal categories.

    Provides CRUD operations, mapping utilities, and statistics
    for the 12 Cameroonian legal categories.

    Features:
    - Create, read, update, delete, list categories
    - Bidirectional ID ↔ Name mapping
    - Statistics: law counts and distribution percentages
    - Smart delete: prevents deletion of categories with laws
    - Health check endpoint

    Usage:
        service = CategoryService(db_session)
        categories = await service.list_categories()
        mapping = await service.get_category_mapping()
        stats = await service.get_all_category_stats()
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize CategoryService.

        Args:
            db: Async database session
        """
        self.db = db
        logger.info("📦 CategoryService initialized")

    # ========================================================================
    # CRUD OPERATIONS
    # ========================================================================

    async def create_category(self, category_data: CategoryCreate) -> Category:
        """
        Create a new category.

        Args:
            category_data: Category creation data

        Returns:
            Created category

        Raises:
            DuplicateCategoryNameError: If category name already exists
            CategoryServiceError: On database errors
        """
        try:
            logger.info(f"📝 Creating category: {category_data.name}")

            # Validate unique name
            await self._validate_unique_name(category_data.name)

            # Create category
            category = Category(
                name=category_data.name,
                description=category_data.description
            )

            self.db.add(category)
            await self.db.commit()
            await self.db.refresh(category)

            logger.info(f"✅ Category created: {category.name} (ID: {category.id})")
            return category

        except DuplicateCategoryNameError:
            await self.db.rollback()
            raise

        except IntegrityError as e:
            await self.db.rollback()
            if "unique constraint" in str(e).lower() or "duplicate" in str(e).lower():
                raise DuplicateCategoryNameError(
                    f"Category with name '{category_data.name}' already exists"
                )
            raise CategoryServiceError(f"Database integrity error: {e}") from e

        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error creating category: {e}")
            raise CategoryServiceError(f"Failed to create category: {e}") from e

    async def get_category(self, category_id: int) -> Category:
        """
        Get a category by ID.

        Args:
            category_id: Category ID

        Returns:
            Category entity

        Raises:
            CategoryNotFoundError: If category doesn't exist
        """
        try:
            logger.debug(f"🔍 Fetching category ID: {category_id}")

            stmt = select(Category).where(Category.id == category_id)
            result = await self.db.execute(stmt)
            category = result.scalar_one_or_none()

            if not category:
                raise CategoryNotFoundError(f"Category with ID {category_id} not found")

            logger.debug(f"✅ Category found: {category.name}")
            return category

        except CategoryNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ Error fetching category: {e}")
            raise CategoryServiceError(f"Failed to fetch category: {e}") from e

    async def list_categories(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[CategoryResponse]:
        """
        List all categories with law counts.

        Args:
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return

        Returns:
            List of categories with law counts
        """
        try:
            logger.info(f"📋 Listing categories (skip={skip}, limit={limit})")

            # Query categories with law counts
            stmt = (
                select(
                    Category.id,
                    Category.name,
                    Category.description,
                    Category.icon,
                    Category.display_order,
                    Category.created_at,
                    func.count(Law.id).label("law_count")
                )
                .outerjoin(Law, Category.id == Law.category_id)
                .group_by(Category.id, Category.name, Category.description, Category.icon, Category.display_order, Category.created_at)
                .order_by(Category.display_order)
                .offset(skip)
                .limit(limit)
            )

            result = await self.db.execute(stmt)
            rows = result.all()

            # Build response
            categories = [
                CategoryResponse(
                    id=row.id,
                    name=row.name,
                    description=row.description,
                    icon=row.icon,
                    created_at=row.created_at,
                    law_count=row.law_count
                )
                for row in rows
            ]

            logger.info(f"✅ Found {len(categories)} categories")
            return categories

        except Exception as e:
            logger.error(f"❌ Error listing categories: {e}")
            raise CategoryServiceError(f"Failed to list categories: {e}") from e

    async def update_category(
        self,
        category_id: int,
        category_data: CategoryUpdate
    ) -> Category:
        """
        Update a category.

        Args:
            category_id: Category ID
            category_data: Update data

        Returns:
            Updated category

        Raises:
            CategoryNotFoundError: If category doesn't exist
            DuplicateCategoryNameError: If new name conflicts
        """
        try:
            logger.info(f"✏️ Updating category ID: {category_id}")

            # Fetch category
            category = await self.get_category(category_id)

            # Validate unique name if changing
            if category_data.name and category_data.name != category.name:
                await self._validate_unique_name(category_data.name, exclude_id=category_id)
                category.name = category_data.name

            # Update description
            if category_data.description is not None:
                category.description = category_data.description

            await self.db.commit()
            await self.db.refresh(category)

            logger.info(f"✅ Category updated: {category.name}")
            return category

        except (CategoryNotFoundError, DuplicateCategoryNameError):
            await self.db.rollback()
            raise

        except IntegrityError as e:
            await self.db.rollback()
            if "unique constraint" in str(e).lower():
                raise DuplicateCategoryNameError(
                    f"Category with name '{category_data.name}' already exists"
                )
            raise CategoryServiceError(f"Database integrity error: {e}") from e

        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error updating category: {e}")
            raise CategoryServiceError(f"Failed to update category: {e}") from e

    async def delete_category(self, category_id: int, force: bool = False) -> bool:
        """
        Delete a category.

        Args:
            category_id: Category ID
            force: If True, allows deletion even if laws are associated
                   (laws will have category_id set to NULL)

        Returns:
            True if deleted successfully

        Raises:
            CategoryNotFoundError: If category doesn't exist
            CategoryInUseError: If category has laws and force=False
        """
        try:
            logger.info(f"🗑️  Deleting category ID: {category_id} (force={force})")

            # Fetch category
            category = await self.get_category(category_id)

            # Check if category has associated laws
            law_count = await self._get_law_count_for_category(category_id)

            if law_count > 0 and not force:
                raise CategoryInUseError(
                    f"Cannot delete category '{category.name}': "
                    f"{law_count} law(s) are associated. Use force=True to delete anyway."
                )

            # If force=True and laws exist, set their category_id to NULL
            if law_count > 0 and force:
                logger.warning(
                    f"⚠️  Force deleting category with {law_count} law(s). "
                    f"Setting category_id to NULL for affected laws."
                )
                stmt = (
                    select(Law)
                    .where(Law.category_id == category_id)
                )
                result = await self.db.execute(stmt)
                laws = result.scalars().all()

                for law in laws:
                    law.category_id = None

            # Delete category
            await self.db.delete(category)
            await self.db.commit()

            logger.info(f"✅ Category deleted: {category.name}")
            return True

        except (CategoryNotFoundError, CategoryInUseError):
            await self.db.rollback()
            raise

        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error deleting category: {e}")
            raise CategoryServiceError(f"Failed to delete category: {e}") from e

    # ========================================================================
    # MAPPING UTILITIES (ID ↔ Name)
    # ========================================================================

    async def get_category_mapping(self) -> Dict[int, str]:
        """
        Get ID → Name mapping for all categories.

        Returns:
            Dictionary mapping category IDs to names

        Example:
            {1: "Droit Civil", 2: "Droit Pénal", ...}
        """
        try:
            logger.debug("🗺️  Fetching category ID → Name mapping")

            stmt = select(Category.id, Category.name).order_by(Category.name)
            result = await self.db.execute(stmt)
            rows = result.all()

            mapping = {row.id: row.name for row in rows}
            logger.debug(f"✅ Mapping fetched: {len(mapping)} categories")
            return mapping

        except Exception as e:
            logger.error(f"❌ Error fetching mapping: {e}")
            raise CategoryServiceError(f"Failed to fetch category mapping: {e}") from e

    async def get_reverse_category_mapping(self) -> Dict[str, int]:
        """
        Get Name → ID mapping for all categories.

        Returns:
            Dictionary mapping category names to IDs

        Example:
            {"Droit Civil": 1, "Droit Pénal": 2, ...}
        """
        try:
            logger.debug("🗺️  Fetching category Name → ID mapping")

            stmt = select(Category.name, Category.id).order_by(Category.name)
            result = await self.db.execute(stmt)
            rows = result.all()

            mapping = {row.name: row.id for row in rows}
            logger.debug(f"✅ Reverse mapping fetched: {len(mapping)} categories")
            return mapping

        except Exception as e:
            logger.error(f"❌ Error fetching reverse mapping: {e}")
            raise CategoryServiceError(f"Failed to fetch reverse category mapping: {e}") from e

    async def get_category_by_name(self, name: str) -> Optional[Category]:
        """
        Get a category by name (case-insensitive).

        Args:
            name: Category name

        Returns:
            Category entity or None if not found
        """
        try:
            logger.debug(f"🔍 Fetching category by name: {name}")

            stmt = select(Category).where(func.lower(Category.name) == name.lower())
            result = await self.db.execute(stmt)
            category = result.scalar_one_or_none()

            if category:
                logger.debug(f"✅ Category found: {category.name}")
            else:
                logger.debug(f"⚠️  Category not found: {name}")

            return category

        except Exception as e:
            logger.error(f"❌ Error fetching category by name: {e}")
            raise CategoryServiceError(f"Failed to fetch category by name: {e}") from e

    # ========================================================================
    # STATISTICS
    # ========================================================================

    async def get_category_stats(self, category_id: int) -> CategoryStats:
        """
        Get statistics for a single category.

        Args:
            category_id: Category ID

        Returns:
            Category statistics with law count and percentage

        Raises:
            CategoryNotFoundError: If category doesn't exist
        """
        try:
            logger.debug(f"📊 Fetching stats for category ID: {category_id}")

            # Fetch category
            category = await self.get_category(category_id)

            # Get total laws
            total_laws_stmt = select(func.count(Law.id))
            total_laws = (await self.db.execute(total_laws_stmt)).scalar_one()

            # Get law count for this category
            law_count = await self._get_law_count_for_category(category_id)

            # Calculate percentage
            percentage = (law_count / total_laws * 100) if total_laws > 0 else 0.0

            stats = CategoryStats(
                category_id=category.id,
                category_name=category.name,
                law_count=law_count,
                percentage=percentage
            )

            logger.debug(f"✅ Stats fetched: {category.name} - {law_count} laws ({percentage:.2f}%)")
            return stats

        except CategoryNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ Error fetching category stats: {e}")
            raise CategoryServiceError(f"Failed to fetch category stats: {e}") from e

    async def get_all_category_stats(self) -> List[CategoryStats]:
        """
        Get statistics for all categories.

        Returns:
            List of category statistics ordered by law count (descending)
        """
        try:
            logger.info("📊 Fetching stats for all categories")

            # Get total laws
            total_laws_stmt = select(func.count(Law.id))
            total_laws = (await self.db.execute(total_laws_stmt)).scalar_one()

            # Get law counts per category
            stmt = (
                select(
                    Category.id,
                    Category.name,
                    func.count(Law.id).label("law_count")
                )
                .outerjoin(Law, Category.id == Law.category_id)
                .group_by(Category.id, Category.name)
                .order_by(func.count(Law.id).desc(), Category.name)
            )

            result = await self.db.execute(stmt)
            rows = result.all()

            # Build stats list
            stats_list = [
                CategoryStats(
                    category_id=row.id,
                    category_name=row.name,
                    law_count=row.law_count,
                    percentage=(row.law_count / total_laws * 100) if total_laws > 0 else 0.0
                )
                for row in rows
            ]

            logger.info(f"✅ Stats fetched for {len(stats_list)} categories")
            return stats_list

        except Exception as e:
            logger.error(f"❌ Error fetching all category stats: {e}")
            raise CategoryServiceError(f"Failed to fetch all category stats: {e}") from e

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status dictionary
        """
        return {
            "service": "CategoryService",
            "status": "healthy",
            "version": "2.1.0",
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================

    async def _validate_unique_name(
        self,
        name: str,
        exclude_id: Optional[int] = None
    ) -> None:
        """
        Validate that category name is unique.

        Args:
            name: Category name to validate
            exclude_id: Category ID to exclude from check (for updates)

        Raises:
            DuplicateCategoryNameError: If name already exists
        """
        conditions = [func.lower(Category.name) == name.lower()]
        if exclude_id is not None:
            conditions.append(Category.id != exclude_id)

        stmt = select(Category).where(and_(*conditions))
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            raise DuplicateCategoryNameError(
                f"Category with name '{name}' already exists"
            )

    async def _get_law_count_for_category(self, category_id: int) -> int:
        """
        Get number of laws associated with a category.

        Args:
            category_id: Category ID

        Returns:
            Number of associated laws
        """
        stmt = select(func.count(Law.id)).where(Law.category_id == category_id)
        count = (await self.db.execute(stmt)).scalar_one()
        return count


# ============================================================================
# MODULE-LEVEL UTILITIES
# ============================================================================

# Note: CategoryService is stateful (requires db session per request)
# No singleton pattern needed - instances created per request via dependency injection
