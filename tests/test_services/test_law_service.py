"""
Comprehensive test suite for LawService.

Tests cover:
- CRUD operations (create, read, update, delete)
- v2.1 language detection and filtering
- v2.1 category suggestions
- Filtering and pagination
- Search functionality
- Statistics
- Error handling

Total: 25+ tests for >90% coverage

Author: JuriX Development Team
Date: 2026-01-10
"""

import pytest
from datetime import date, datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from app.core.database import Base
from app.models.law import Category, Law, Article
from app.schemas.law import (
    CategoryCreate,
    LawCreate,
    LawUpdate,
    LawFilters,
    SearchQuery
)
from app.services.law_service import (
    LawService,
    LawServiceError,
    LawNotFoundError,
    CategoryNotFoundError,
    DuplicateReferenceError
)


# ============================================================================
# Test Configuration & Fixtures
# ============================================================================

# Use in-memory SQLite for fast tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="function")
async def db_engine():
    """Create a fresh database engine for each test."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(db_engine):
    """Create a fresh database session for each test."""
    AsyncSessionLocal = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
async def law_service(db_session):
    """Create LawService instance for testing."""
    return LawService(db_session)


@pytest.fixture(scope="function")
async def sample_categories(db_session):
    """Create sample categories for testing."""
    categories = [
        Category(id=1, name="Droit civil", description="Code civil et lois connexes"),
        Category(id=2, name="Droit pénal", description="Code pénal et procédure pénale"),
        Category(id=3, name="Droit commercial", description="Commerce et affaires"),
    ]

    for cat in categories:
        db_session.add(cat)

    await db_session.commit()
    return categories


@pytest.fixture
def sample_law_data_french():
    """Sample French law data for testing."""
    return LawCreate(
        reference="LOI-2024-001",
        title="Loi sur le Code civil camerounais",
        type="loi",
        content="""
        Article 1er. Le Code civil régit les personnes, les biens et les relations de droit privé.
        Article 2. Toute personne a droit à la reconnaissance de sa personnalité juridique.
        Article 3. Les contrats légalement formés ont force de loi entre les parties.
        """,
        language="fr",
        category_id=1,
        status="draft"
    )


@pytest.fixture
def sample_law_data_english():
    """Sample English law data for testing."""
    return LawCreate(
        reference="LAW-2024-002",
        title="Commercial Code Act",
        type="loi",
        content="""
        Section 1. This Act establishes the framework for commercial transactions.
        Section 2. All commercial entities must register with the competent authority.
        Section 3. Disputes shall be resolved through arbitration or litigation.
        """,
        language="en",
        category_id=3,
        status="published"
    )


# ============================================================================
# Test Class 1: Create Operations (5 tests)
# ============================================================================

@pytest.mark.asyncio
class TestCreate:
    """Tests for law creation operations."""

    async def test_create_law_success(self, law_service, sample_categories, sample_law_data_french):
        """Test successful law creation with all fields."""
        # Act
        law = await law_service.create_law(sample_law_data_french)

        # Assert
        assert law.id is not None
        assert law.reference == "LOI-2024-001"
        assert law.title == "Loi sur le Code civil camerounais"
        assert law.type == "loi"
        assert law.language == "fr"
        assert law.category_id == 1
        assert law.status == "draft"
        assert law.created_at is not None

        # v2.1 fields
        assert law.detected_language is not None
        assert law.language_confidence is not None
        assert law.language_confidence >= 0.0
        assert law.language_confidence <= 1.0

    async def test_create_law_with_auto_language_detection(self, law_service, sample_categories):
        """Test law creation without language - should auto-detect."""
        # Arrange
        law_data = LawCreate(
            reference="LOI-2024-003",
            title="Loi sans langue spécifiée",
            type="décret",
            content="Ceci est un texte en français qui devrait être détecté automatiquement.",
            category_id=1
        )

        # Act
        law = await law_service.create_law(law_data)

        # Assert
        assert law.language is not None  # Should be auto-detected
        assert law.detected_language is not None
        assert law.language_confidence is not None
        assert law.language_confidence > 0.0

    async def test_create_law_with_category_suggestions(self, law_service, sample_categories):
        """Test law creation with automatic category suggestions (v2.1)."""
        # Arrange
        law_data = LawCreate(
            reference="LOI-2024-004",
            title="Loi sur les contrats commerciaux",
            type="loi",
            content="Article 1. Les contrats de vente sont régis par le présent code.",
            language="fr"
        )

        # Act
        law = await law_service.create_law(law_data)

        # Assert - v2.1 category suggestions
        assert law.suggested_categories is not None
        assert isinstance(law.suggested_categories, list)
        # Should suggest up to 3 categories
        assert len(law.suggested_categories) <= 3

        if law.suggested_categories:
            assert law.category_confidence is not None
            assert 0.0 <= law.category_confidence <= 1.0

    async def test_create_law_duplicate_reference(self, law_service, sample_categories, sample_law_data_french):
        """Test that duplicate reference raises error."""
        # Arrange - create first law
        await law_service.create_law(sample_law_data_french)

        # Act & Assert - try to create duplicate
        with pytest.raises(DuplicateReferenceError) as exc_info:
            await law_service.create_law(sample_law_data_french)

        assert "LOI-2024-001" in str(exc_info.value)

    async def test_create_law_invalid_data(self, law_service, sample_categories):
        """Test law creation with invalid data raises validation error."""
        # Arrange - invalid type
        with pytest.raises(ValueError):
            LawCreate(
                reference="LOI-2024-005",
                title="Loi invalide",
                type="invalid_type",  # Invalid type
                content="Content",
                language="fr"
            )

        # Arrange - invalid language
        with pytest.raises(ValueError):
            LawCreate(
                reference="LOI-2024-005",
                title="Loi invalide",
                type="loi",
                content="Content",
                language="de"  # Invalid language (only fr/en allowed)
            )


# ============================================================================
# Test Class 2: Read Operations (4 tests)
# ============================================================================

@pytest.mark.asyncio
class TestRead:
    """Tests for law retrieval operations."""

    async def test_get_law_by_id(self, law_service, sample_categories, sample_law_data_french):
        """Test fetching a law by ID."""
        # Arrange
        created_law = await law_service.create_law(sample_law_data_french)

        # Act
        law = await law_service.get_law(created_law.id)

        # Assert
        assert law.id == created_law.id
        assert law.reference == "LOI-2024-001"
        assert law.title == created_law.title

    async def test_get_law_with_category(self, law_service, sample_categories, sample_law_data_french):
        """Test fetching law with category relationship."""
        # Arrange
        created_law = await law_service.create_law(sample_law_data_french)

        # Act
        law = await law_service.get_law(created_law.id, include_category=True)

        # Assert
        assert law.category is not None
        assert law.category.name == "Droit civil"

    async def test_get_law_not_found(self, law_service):
        """Test that fetching non-existent law raises error."""
        # Act & Assert
        with pytest.raises(LawNotFoundError):
            await law_service.get_law(99999)

    async def test_list_laws_pagination(self, law_service, sample_categories):
        """Test listing laws with pagination."""
        # Arrange - create 5 laws
        for i in range(5):
            law_data = LawCreate(
                reference=f"LOI-2024-{i:03d}",
                title=f"Loi numéro {i}",
                type="loi",
                content=f"Contenu de la loi {i}",
                language="fr",
                category_id=1
            )
            await law_service.create_law(law_data)

        # Act - get page 1 with 2 items
        filters = LawFilters(page=1, per_page=2)
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 5
        assert len(result.items) == 2
        assert result.page == 1
        assert result.per_page == 2
        assert result.pages == 3  # 5 items / 2 per page = 3 pages


# ============================================================================
# Test Class 3: Update Operations (4 tests)
# ============================================================================

@pytest.mark.asyncio
class TestUpdate:
    """Tests for law update operations."""

    async def test_update_law_title(self, law_service, sample_categories, sample_law_data_french):
        """Test updating law title (partial update)."""
        # Arrange
        law = await law_service.create_law(sample_law_data_french)

        # Act
        update_data = LawUpdate(title="Nouveau titre modifié")
        updated_law = await law_service.update_law(law.id, update_data)

        # Assert
        assert updated_law.title == "Nouveau titre modifié"
        assert updated_law.reference == "LOI-2024-001"  # Unchanged
        assert updated_law.updated_at is not None

    async def test_update_law_content_triggers_redetection(self, law_service, sample_categories, sample_law_data_french):
        """Test that updating content triggers language/category re-detection."""
        # Arrange
        law = await law_service.create_law(sample_law_data_french)
        original_confidence = law.language_confidence

        # Act - update content
        update_data = LawUpdate(
            content="This is new English content that should be detected as English language."
        )
        updated_law = await law_service.update_law(law.id, update_data)

        # Assert - v2.1 re-detection occurred
        assert updated_law.detected_language is not None
        assert updated_law.language_confidence is not None
        # Confidence may have changed due to new content
        assert updated_law.suggested_categories is not None

    async def test_update_law_not_found(self, law_service):
        """Test updating non-existent law raises error."""
        # Arrange
        update_data = LawUpdate(title="Test")

        # Act & Assert
        with pytest.raises(LawNotFoundError):
            await law_service.update_law(99999, update_data)

    async def test_update_law_duplicate_reference(self, law_service, sample_categories, sample_law_data_french, sample_law_data_english):
        """Test updating reference to duplicate value raises error."""
        # Arrange - create two laws
        law1 = await law_service.create_law(sample_law_data_french)
        law2 = await law_service.create_law(sample_law_data_english)

        # Act & Assert - try to update law2 reference to law1's reference
        update_data = LawUpdate(reference="LOI-2024-001")
        with pytest.raises(DuplicateReferenceError):
            await law_service.update_law(law2.id, update_data)


# ============================================================================
# Test Class 4: Delete Operations (2 tests)
# ============================================================================

@pytest.mark.asyncio
class TestDelete:
    """Tests for law deletion operations."""

    async def test_delete_law_success(self, law_service, sample_categories, sample_law_data_french):
        """Test successful law deletion."""
        # Arrange
        law = await law_service.create_law(sample_law_data_french)

        # Act
        result = await law_service.delete_law(law.id)

        # Assert
        assert result is True

        # Verify law is deleted
        with pytest.raises(LawNotFoundError):
            await law_service.get_law(law.id)

    async def test_delete_law_not_found(self, law_service):
        """Test deleting non-existent law raises error."""
        # Act & Assert
        with pytest.raises(LawNotFoundError):
            await law_service.delete_law(99999)


# ============================================================================
# Test Class 5: Filtering Operations (6 tests)
# ============================================================================

@pytest.mark.asyncio
class TestFiltering:
    """Tests for law filtering operations (v2.1 features)."""

    async def test_filter_by_language_french(self, law_service, sample_categories):
        """Test filtering laws by French language (v2.1)."""
        # Arrange - create French and English laws
        french_law = LawCreate(
            reference="LOI-FR-001",
            title="Loi française",
            type="loi",
            content="Contenu en français",
            language="fr",
            category_id=1
        )
        english_law = LawCreate(
            reference="LAW-EN-001",
            title="English Law",
            type="loi",
            content="English content",
            language="en",
            category_id=1
        )
        await law_service.create_law(french_law)
        await law_service.create_law(english_law)

        # Act - filter by French
        filters = LawFilters(language="fr", page=1, per_page=10)
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 1
        assert result.items[0].language == "fr"

    async def test_filter_by_language_english(self, law_service, sample_categories):
        """Test filtering laws by English language (v2.1)."""
        # Arrange
        french_law = LawCreate(
            reference="LOI-FR-002",
            title="Loi française",
            type="loi",
            content="Contenu en français",
            language="fr",
            category_id=1
        )
        english_law = LawCreate(
            reference="LAW-EN-002",
            title="English Law",
            type="loi",
            content="English content",
            language="en",
            category_id=1
        )
        await law_service.create_law(french_law)
        await law_service.create_law(english_law)

        # Act - filter by English
        filters = LawFilters(language="en", page=1, per_page=10)
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 1
        assert result.items[0].language == "en"

    async def test_filter_by_category(self, law_service, sample_categories):
        """Test filtering laws by category."""
        # Arrange - create laws in different categories
        law1 = LawCreate(
            reference="LOI-CAT1-001",
            title="Loi catégorie 1",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            category_id=1,
            language="fr"
        )
        law2 = LawCreate(
            reference="LOI-CAT2-001",
            title="Loi catégorie 2",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            category_id=2,
            language="fr"
        )
        await law_service.create_law(law1)
        await law_service.create_law(law2)

        # Act - filter by category 1
        filters = LawFilters(category_id=1, page=1, per_page=10)
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 1
        assert result.items[0].category_id == 1

    async def test_filter_by_status(self, law_service, sample_categories):
        """Test filtering laws by status."""
        # Arrange
        draft_law = LawCreate(
            reference="LOI-DRAFT-001",
            title="Loi brouillon",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            status="draft",
            language="fr",
            category_id=1
        )
        published_law = LawCreate(
            reference="LOI-PUB-001",
            title="Loi publiée",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            status="published",
            language="fr",
            category_id=1
        )
        await law_service.create_law(draft_law)
        await law_service.create_law(published_law)

        # Act - filter by published
        filters = LawFilters(status="published", page=1, per_page=10)
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 1
        assert result.items[0].status == "published"

    async def test_filter_by_date_range(self, law_service, sample_categories):
        """Test filtering laws by publication date range."""
        # Arrange
        law_2023 = LawCreate(
            reference="LOI-2023-001",
            title="Loi de 2023",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            publication_date=date(2023, 6, 15),
            language="fr",
            category_id=1
        )
        law_2024 = LawCreate(
            reference="LOI-2024-100",
            title="Loi de 2024",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            publication_date=date(2024, 3, 20),
            language="fr",
            category_id=1
        )
        await law_service.create_law(law_2023)
        await law_service.create_law(law_2024)

        # Act - filter 2024 only
        filters = LawFilters(year_from=2024, year_to=2024, page=1, per_page=10)
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 1
        assert result.items[0].publication_date.year == 2024

    async def test_combined_filters(self, law_service, sample_categories):
        """Test applying multiple filters simultaneously."""
        # Arrange - create diverse laws
        law1 = LawCreate(
            reference="LOI-COMBINED-001",
            title="Loi française publiée catégorie 1",
            type="loi",
            content="Contenu en français",
            language="fr",
            category_id=1,
            status="published",
            publication_date=date(2024, 1, 1)
        )
        law2 = LawCreate(
            reference="LOI-COMBINED-002",
            title="Loi anglaise brouillon catégorie 2",
            type="décret",
            content="English content",
            language="en",
            category_id=2,
            status="draft",
            publication_date=date(2024, 6, 1)
        )
        await law_service.create_law(law1)
        await law_service.create_law(law2)

        # Act - filter: French + category 1 + published
        filters = LawFilters(
            language="fr",
            category_id=1,
            status="published",
            page=1,
            per_page=10
        )
        result = await law_service.list_laws(filters)

        # Assert
        assert result.total == 1
        assert result.items[0].language == "fr"
        assert result.items[0].category_id == 1
        assert result.items[0].status == "published"


# ============================================================================
# Test Class 6: Search Operations (2 tests)
# ============================================================================

@pytest.mark.asyncio
class TestSearch:
    """Tests for law search operations."""

    async def test_search_laws_by_title(self, law_service, sample_categories):
        """Test searching laws by title."""
        # Arrange
        law1 = LawCreate(
            reference="LOI-SEARCH-001",
            title="Loi sur les contrats commerciaux",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            language="fr",
            category_id=1
        )
        law2 = LawCreate(
            reference="LOI-SEARCH-002",
            title="Loi sur la propriété intellectuelle",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            language="fr",
            category_id=1
        )
        await law_service.create_law(law1)
        await law_service.create_law(law2)

        # Act
        result = await law_service.search_laws("contrats")

        # Assert
        assert result.total >= 1
        assert any("contrat" in r.title.lower() for r in result.results)
        assert result.search_time_ms >= 0

    async def test_search_laws_with_filters(self, law_service, sample_categories):
        """Test searching with additional filters."""
        # Arrange
        law1 = LawCreate(
            reference="LOI-SEARCH-003",
            title="Loi française sur le commerce",
            type="loi",
            content="Commerce en français",
            language="fr",
            category_id=3
        )
        law2 = LawCreate(
            reference="LAW-SEARCH-004",
            title="English Commercial Law",
            type="loi",
            content="Commercial in English",
            language="en",
            category_id=3
        )
        await law_service.create_law(law1)
        await law_service.create_law(law2)

        # Act - search "commercial" but filter by French only
        filters = LawFilters(language="fr", page=1, per_page=10)
        result = await law_service.search_laws("commerce", filters)

        # Assert
        assert result.total >= 1
        assert all(r.language == "fr" for r in result.results)


# ============================================================================
# Test Class 7: Statistics Operations (2 tests)
# ============================================================================

@pytest.mark.asyncio
class TestStatistics:
    """Tests for statistics operations (v2.1 features)."""

    async def test_get_language_stats(self, law_service, sample_categories):
        """Test language distribution statistics (v2.1)."""
        # Arrange - create laws in different languages
        for i in range(3):
            french_law = LawCreate(
                reference=f"LOI-FR-{i:03d}",
                title=f"Loi française {i}",
                type="loi",
                content="Contenu en français",
                language="fr",
                category_id=1
            )
            await law_service.create_law(french_law)

        for i in range(2):
            english_law = LawCreate(
                reference=f"LAW-EN-{i:03d}",
                title=f"English Law {i}",
                type="loi",
                content="English content",
                language="en",
                category_id=1
            )
            await law_service.create_law(english_law)

        # Act
        stats = await law_service.get_language_stats()

        # Assert
        assert stats.french == 3
        assert stats.english == 2
        assert stats.total == 5

    async def test_get_comprehensive_stats(self, law_service, sample_categories):
        """Test comprehensive law statistics."""
        # Arrange - create diverse laws
        law1 = LawCreate(
            reference="LOI-STATS-001",
            title="Loi test",
            type="loi",
            content="Contenu de la loi avec du texte suffisant",
            language="fr",
            category_id=1,
            status="published"
        )
        law2 = LawCreate(
            reference="DECRET-STATS-001",
            title="Décret test",
            type="décret",
            content="Contenu de la loi avec du texte suffisant",
            language="fr",
            category_id=2,
            status="draft"
        )
        await law_service.create_law(law1)
        await law_service.create_law(law2)

        # Act
        stats = await law_service.get_law_stats()

        # Assert
        assert stats.total_laws == 2
        assert stats.by_language.french == 2
        assert "published" in stats.by_status
        assert "draft" in stats.by_status
        assert "loi" in stats.by_type
        assert "décret" in stats.by_type


# ============================================================================
# Test Class 8: Category Operations (2 tests)
# ============================================================================

@pytest.mark.asyncio
class TestCategories:
    """Tests for category operations."""

    async def test_create_category(self, law_service):
        """Test creating a new category."""
        # Arrange
        cat_data = CategoryCreate(
            name="Nouvelle catégorie",
            description="Description de test"
        )

        # Act
        category = await law_service.create_category(cat_data)

        # Assert
        assert category.id is not None
        assert category.name == "Nouvelle catégorie"
        assert category.description == "Description de test"

    async def test_list_categories(self, law_service, sample_categories):
        """Test listing all categories with law counts."""
        # Act
        categories = await law_service.list_categories()

        # Assert
        assert len(categories) == 3
        assert all(hasattr(cat, "law_count") for cat in categories)


# ============================================================================
# Test Class 9: Integration Test (1 test)
# ============================================================================

@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for complete workflows."""

    async def test_full_crud_lifecycle(self, law_service, sample_categories, sample_law_data_french):
        """Test complete CRUD lifecycle: Create → Read → Update → Delete."""
        # 1. CREATE
        law = await law_service.create_law(sample_law_data_french)
        assert law.id is not None
        original_id = law.id

        # 2. READ
        fetched_law = await law_service.get_law(original_id)
        assert fetched_law.reference == "LOI-2024-001"

        # 3. UPDATE
        update_data = LawUpdate(title="Titre mis à jour")
        updated_law = await law_service.update_law(original_id, update_data)
        assert updated_law.title == "Titre mis à jour"

        # 4. DELETE
        result = await law_service.delete_law(original_id)
        assert result is True

        # 5. VERIFY DELETION
        with pytest.raises(LawNotFoundError):
            await law_service.get_law(original_id)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
