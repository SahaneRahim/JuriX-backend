"""Tests for CategoryService."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.category_service import (
    CategoryService,
    CategoryNotFoundError,
    DuplicateCategoryNameError,
    CategoryInUseError
)
from app.schemas.law import CategoryCreate, CategoryUpdate
from app.models.law import Category, Law


class TestCreateCategory:
    """Tests for create_category method."""

    @pytest.mark.asyncio
    async def test_create_category_success(self, db_session: AsyncSession):
        """Test successful category creation."""
        service = CategoryService(db_session)
        category_data = CategoryCreate(
            name="Test Category",
            description="A test category for unit testing"
        )

        result = await service.create_category(category_data)

        assert result.id is not None
        assert result.name == "Test Category"
        assert result.description == "A test category for unit testing"
        assert result.created_at is not None

    @pytest.mark.asyncio
    async def test_create_category_duplicate_name(self, db_session: AsyncSession):
        """Test creating category with duplicate name fails."""
        service = CategoryService(db_session)
        category_data = CategoryCreate(name="Droit Civil", description="Test")

        # "Droit Civil" should already exist in seed data
        with pytest.raises(DuplicateCategoryNameError) as exc_info:
            await service.create_category(category_data)

        assert "already exists" in str(exc_info.value).lower()
        assert "Droit Civil" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_category_empty_name(self, db_session: AsyncSession):
        """Test creating category with empty name fails validation."""
        service = CategoryService(db_session)

        # Pydantic should validate this, but test the service behavior
        with pytest.raises((ValueError, DuplicateCategoryNameError)):
            category_data = CategoryCreate(name="", description="Test")
            await service.create_category(category_data)

    @pytest.mark.asyncio
    async def test_create_category_without_description(self, db_session: AsyncSession):
        """Test creating category without description succeeds."""
        service = CategoryService(db_session)
        category_data = CategoryCreate(name="Category Without Desc")

        result = await service.create_category(category_data)

        assert result.id is not None
        assert result.name == "Category Without Desc"
        assert result.description is None


class TestGetCategory:
    """Tests for get_category method."""

    @pytest.mark.asyncio
    async def test_get_category_success(self, db_session: AsyncSession):
        """Test successfully retrieving an existing category."""
        service = CategoryService(db_session)

        # Create a category first
        category_data = CategoryCreate(name="Get Test Category")
        created = await service.create_category(category_data)

        # Now get it
        result = await service.get_category(created.id)

        assert result.id == created.id
        assert result.name == "Get Test Category"

    @pytest.mark.asyncio
    async def test_get_category_not_found(self, db_session: AsyncSession):
        """Test getting non-existent category raises error."""
        service = CategoryService(db_session)

        with pytest.raises(CategoryNotFoundError) as exc_info:
            await service.get_category(999999)

        assert "not found" in str(exc_info.value).lower()
        assert "999999" in str(exc_info.value)


class TestListCategories:
    """Tests for list_categories method."""

    @pytest.mark.asyncio
    async def test_list_categories_all(self, db_session: AsyncSession):
        """Test listing all categories."""
        service = CategoryService(db_session)

        result = await service.list_categories()

        # Should have at least the 12 seed categories
        assert len(result) >= 12
        assert all(hasattr(cat, "id") for cat in result)
        assert all(hasattr(cat, "name") for cat in result)

    @pytest.mark.asyncio
    async def test_list_categories_pagination(self, db_session: AsyncSession):
        """Test pagination with skip and limit."""
        service = CategoryService(db_session)

        # Get first 5 categories
        page1 = await service.list_categories(skip=0, limit=5)
        assert len(page1) == 5

        # Get next 5 categories
        page2 = await service.list_categories(skip=5, limit=5)
        assert len(page2) == 5

        # Ensure they're different
        page1_ids = {cat.id for cat in page1}
        page2_ids = {cat.id for cat in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_categories_empty_result(self, db_session: AsyncSession):
        """Test listing with skip beyond available categories."""
        service = CategoryService(db_session)

        result = await service.list_categories(skip=1000, limit=10)

        assert len(result) == 0


class TestUpdateCategory:
    """Tests for update_category method."""

    @pytest.mark.asyncio
    async def test_update_category_success(self, db_session: AsyncSession):
        """Test successfully updating a category."""
        service = CategoryService(db_session)

        # Create a category
        category_data = CategoryCreate(name="Update Test", description="Original")
        created = await service.create_category(category_data)

        # Update it
        update_data = CategoryUpdate(name="Updated Name", description="Updated Desc")
        result = await service.update_category(created.id, update_data)

        assert result.id == created.id
        assert result.name == "Updated Name"
        assert result.description == "Updated Desc"

    @pytest.mark.asyncio
    async def test_update_category_not_found(self, db_session: AsyncSession):
        """Test updating non-existent category raises error."""
        service = CategoryService(db_session)
        update_data = CategoryUpdate(name="New Name")

        with pytest.raises(CategoryNotFoundError):
            await service.update_category(999999, update_data)

    @pytest.mark.asyncio
    async def test_update_category_duplicate_name(self, db_session: AsyncSession):
        """Test updating to duplicate name fails."""
        service = CategoryService(db_session)

        # Create first category
        cat1_data = CategoryCreate(name="Category 1")
        cat1 = await service.create_category(cat1_data)

        # Create second category
        cat2_data = CategoryCreate(name="Category 2")
        cat2 = await service.create_category(cat2_data)

        # Try to update cat2 to have same name as cat1
        update_data = CategoryUpdate(name="Category 1")
        with pytest.raises(DuplicateCategoryNameError):
            await service.update_category(cat2.id, update_data)

    @pytest.mark.asyncio
    async def test_update_category_partial(self, db_session: AsyncSession):
        """Test partial update (only name or description)."""
        service = CategoryService(db_session)

        # Create a category
        category_data = CategoryCreate(name="Partial Test", description="Original Desc")
        created = await service.create_category(category_data)

        # Update only description
        update_data = CategoryUpdate(description="New Description")
        result = await service.update_category(created.id, update_data)

        assert result.name == "Partial Test"  # Name unchanged
        assert result.description == "New Description"  # Description updated


class TestDeleteCategory:
    """Tests for delete_category method."""

    @pytest.mark.asyncio
    async def test_delete_category_success(self, db_session: AsyncSession):
        """Test successfully deleting a category without laws."""
        service = CategoryService(db_session)

        # Create a category
        category_data = CategoryCreate(name="Delete Test")
        created = await service.create_category(category_data)

        # Delete it
        result = await service.delete_category(created.id)

        assert result is True

        # Verify it's gone
        with pytest.raises(CategoryNotFoundError):
            await service.get_category(created.id)

    @pytest.mark.asyncio
    async def test_delete_category_not_found(self, db_session: AsyncSession):
        """Test deleting non-existent category raises error."""
        service = CategoryService(db_session)

        with pytest.raises(CategoryNotFoundError):
            await service.delete_category(999999)

    @pytest.mark.asyncio
    async def test_delete_category_with_laws_fails(self, db_session: AsyncSession, sample_law):
        """Test deleting category with associated laws fails."""
        service = CategoryService(db_session)

        # sample_law fixture should have a category_id
        if sample_law.category_id is None:
            pytest.skip("Sample law has no category")

        # Try to delete the category
        with pytest.raises(CategoryInUseError) as exc_info:
            await service.delete_category(sample_law.category_id, force=False)

        assert "cannot delete" in str(exc_info.value).lower()
        assert "law(s) are associated" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_delete_category_with_laws_force_success(self, db_session: AsyncSession, sample_law):
        """Test force deleting category with laws succeeds."""
        service = CategoryService(db_session)

        # sample_law fixture should have a category_id
        if sample_law.category_id is None:
            pytest.skip("Sample law has no category")

        category_id = sample_law.category_id

        # Force delete the category
        result = await service.delete_category(category_id, force=True)

        assert result is True

        # Verify category is deleted
        with pytest.raises(CategoryNotFoundError):
            await service.get_category(category_id)


class TestCategoryMapping:
    """Tests for category mapping utilities."""

    @pytest.mark.asyncio
    async def test_get_category_mapping(self, db_session: AsyncSession):
        """Test getting ID to name mapping."""
        service = CategoryService(db_session)

        mapping = await service.get_category_mapping()

        # Should be a dict with int keys and string values
        assert isinstance(mapping, dict)
        assert len(mapping) >= 12  # At least seed categories
        assert all(isinstance(k, int) for k in mapping.keys())
        assert all(isinstance(v, str) for v in mapping.values())

        # Should include known categories
        assert "Droit Civil" in mapping.values()
        assert "Droit Pénal" in mapping.values()

    @pytest.mark.asyncio
    async def test_get_reverse_category_mapping(self, db_session: AsyncSession):
        """Test getting name to ID mapping."""
        service = CategoryService(db_session)

        mapping = await service.get_reverse_category_mapping()

        # Should be a dict with string keys and int values
        assert isinstance(mapping, dict)
        assert len(mapping) >= 12  # At least seed categories
        assert all(isinstance(k, str) for k in mapping.keys())
        assert all(isinstance(v, int) for v in mapping.values())

        # Should include known categories
        assert "Droit Civil" in mapping.keys()
        assert "Droit Pénal" in mapping.keys()

    @pytest.mark.asyncio
    async def test_get_category_by_name(self, db_session: AsyncSession):
        """Test getting category by name."""
        service = CategoryService(db_session)

        # Test with existing category
        result = await service.get_category_by_name("Droit Civil")
        assert result is not None
        assert result.name == "Droit Civil"

        # Test with non-existent category
        result = await service.get_category_by_name("Non Existent Category")
        assert result is None

    @pytest.mark.asyncio
    async def test_mapping_consistency(self, db_session: AsyncSession):
        """Test that forward and reverse mappings are consistent."""
        service = CategoryService(db_session)

        forward_mapping = await service.get_category_mapping()
        reverse_mapping = await service.get_reverse_category_mapping()

        # Should have same number of entries
        assert len(forward_mapping) == len(reverse_mapping)

        # Every entry in forward should have reverse
        for cat_id, cat_name in forward_mapping.items():
            assert cat_name in reverse_mapping
            assert reverse_mapping[cat_name] == cat_id


class TestCategoryStats:
    """Tests for category statistics methods."""

    @pytest.mark.asyncio
    async def test_get_category_stats(self, db_session: AsyncSession):
        """Test getting statistics for single category."""
        service = CategoryService(db_session)

        # Get first category
        categories = await service.list_categories(limit=1)
        category = categories[0]

        stats = await service.get_category_stats(category.id)

        assert stats.category_id == category.id
        assert stats.category_name == category.name
        assert stats.law_count >= 0
        assert 0 <= stats.percentage <= 100

    @pytest.mark.asyncio
    async def test_get_category_stats_not_found(self, db_session: AsyncSession):
        """Test getting stats for non-existent category."""
        service = CategoryService(db_session)

        with pytest.raises(CategoryNotFoundError):
            await service.get_category_stats(999999)

    @pytest.mark.asyncio
    async def test_get_all_category_stats(self, db_session: AsyncSession):
        """Test getting statistics for all categories."""
        service = CategoryService(db_session)

        stats_list = await service.get_all_category_stats()

        # Should have stats for all categories
        assert len(stats_list) >= 12

        # All stats should have valid data
        for stats in stats_list:
            assert stats.category_id is not None
            assert stats.category_name is not None
            assert stats.law_count >= 0
            assert 0 <= stats.percentage <= 100

        # Total percentages should sum to ~100% if there are laws, or 0% if no laws
        total_percentage = sum(s.percentage for s in stats_list)
        total_laws = sum(s.law_count for s in stats_list)

        if total_laws > 0:
            # If there are laws, percentages should sum to ~100%
            assert 99 <= total_percentage <= 101
        else:
            # If there are no laws, all percentages should be 0
            assert total_percentage == 0

    @pytest.mark.asyncio
    async def test_category_stats_ordering(self, db_session: AsyncSession):
        """Test that stats are ordered by law count descending."""
        service = CategoryService(db_session)

        stats_list = await service.get_all_category_stats()

        # Check ordering (descending by law_count, then by name)
        for i in range(len(stats_list) - 1):
            current = stats_list[i]
            next_stat = stats_list[i + 1]

            # If law counts are different, should be descending
            if current.law_count != next_stat.law_count:
                assert current.law_count >= next_stat.law_count


class TestHealthCheck:
    """Tests for health check method."""

    def test_health_check(self, db_session: AsyncSession):
        """Test health check returns correct structure."""
        service = CategoryService(db_session)

        health = service.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "service" in health
        assert "timestamp" in health
        assert health["status"] == "healthy"
        assert health["service"] == "CategoryService"


class TestPrivateHelpers:
    """Tests for private helper methods."""

    @pytest.mark.asyncio
    async def test_validate_unique_name_new(self, db_session: AsyncSession):
        """Test validating a unique name that doesn't exist."""
        service = CategoryService(db_session)

        # Should not raise any exception
        await service._validate_unique_name("Totally New Category Name")

    @pytest.mark.asyncio
    async def test_validate_unique_name_duplicate(self, db_session: AsyncSession):
        """Test validating a duplicate name raises error."""
        service = CategoryService(db_session)

        with pytest.raises(DuplicateCategoryNameError):
            await service._validate_unique_name("Droit Civil")

    @pytest.mark.asyncio
    async def test_validate_unique_name_with_exclusion(self, db_session: AsyncSession):
        """Test validating name with exclusion for updates."""
        service = CategoryService(db_session)

        # Get an existing category
        categories = await service.list_categories(limit=1)
        category = categories[0]

        # Should not raise error when excluding the same category
        await service._validate_unique_name(category.name, exclude_id=category.id)

    @pytest.mark.asyncio
    async def test_get_law_count_for_category(self, db_session: AsyncSession, sample_law):
        """Test counting laws for a category."""
        service = CategoryService(db_session)

        if sample_law.category_id is None:
            pytest.skip("Sample law has no category")

        count = await service._get_law_count_for_category(sample_law.category_id)

        assert count >= 1  # At least the sample law


# Summary:
# - 4 test classes for CRUD operations (Create, Get, List, Update, Delete)
# - 3 test classes for utilities (Mapping, Stats, Health)
# - 1 test class for private helpers
# - Total: 30+ test methods covering all functionality
# - Tests cover success cases, error cases, edge cases, and validation
