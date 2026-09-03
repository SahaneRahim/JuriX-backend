"""
LawService - Service for managing legal documents (laws) with CRUD operations.

Provides comprehensive law management with v2.1 features:
- Complete CRUD operations (create, read, update, delete, list, search)
- v2.1 language filtering and auto-detection
- v2.1 automatic category suggestions with confidence scores
- Integration with LanguageDetector and DocumentClassifier
- Advanced filtering and pagination
- Search functionality

Author: JuriX Development Team
Date: 2026-01-10
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, extract, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.core.dependencies import get_document_classifier, get_language_detector
from app.models.law import Article, Category, Law
from app.schemas.law import (
    CategoryCreate,
    CategoryResponse,
    CategoryStats,
    LanguageStats,
    LawCreate,
    LawFilters,
    LawListResponse,
    LawResponse,
    LawStats,
    LawUpdate,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class LawServiceError(Exception):
    """Base exception for LawService operations."""
    pass


class LawNotFoundError(LawServiceError):
    """Raised when a law is not found."""
    pass


class CategoryNotFoundError(LawServiceError):
    """Raised when a category is not found."""
    pass


class DuplicateReferenceError(LawServiceError):
    """Raised when trying to create/update a law with duplicate reference."""
    pass


# ============================================================================
# LawService Implementation
# ============================================================================

class LawService:
    """
    Service for managing legal documents (laws) with CRUD operations.

    This is a stateful service that requires a database session.
    Instantiate per request via dependency injection.

    Features:
    - Complete CRUD operations
    - v2.1 language detection and filtering
    - v2.1 category suggestions with confidence
    - Advanced filtering and search
    - Integration with ML services

    Usage:
        async with get_db() as db:
            service = LawService(db)
            law = await service.create_law(law_data)
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize LawService with database session.

        Args:
            db: AsyncSession for database operations
        """
        self.db = db
        self.language_detector = get_language_detector()
        self.document_classifier = get_document_classifier()
        logger.info("✅ LawService initialized")

    # ========================================================================
    # Law CRUD Operations
    # ========================================================================

    async def create_law(self, law_data: LawCreate) -> Law:
        """
        Create a new law with automatic language and category detection.

        v2.1 Features:
        - Auto-detects language if not provided
        - Auto-suggests top 3 categories
        - Stores confidence scores

        Args:
            law_data: Law creation data

        Returns:
            Created Law object with v2.1 fields populated

        Raises:
            DuplicateReferenceError: If reference already exists
            LawServiceError: If creation fails
        """
        assert law_data is not None, "law_data must not be None"
        assert hasattr(law_data, "reference") and law_data.reference, "law_data must have a reference"

        try:
            logger.info(f"📝 Creating law: {law_data.reference}")

            # Validate unique reference
            await self._validate_unique_reference(law_data.reference)

            # Auto-detect language (v2.1)
            detected_lang, lang_confidence = await self._detect_language_and_confidence(
                law_data.content
            )

            # Auto-suggest categories (v2.1)
            suggested_cats, cat_confidence = await self._suggest_categories(
                law_data.content
            )

            # Prepare law data
            law_dict = law_data.model_dump(exclude_unset=True)

            # Use provided language or detected language
            final_language = law_data.language if law_data.language else detected_lang

            # Update law_dict with v2.1 detection fields
            law_dict.update({
                "language": final_language,
                "detected_language": detected_lang,
                "language_confidence": lang_confidence,
                "suggested_categories": suggested_cats,
                "category_confidence": cat_confidence,
            })

            # Create Law object
            law = Law(**law_dict)

            self.db.add(law)
            await self.db.commit()
            await self.db.refresh(law)

            logger.info(
                f"✅ Law created: {law.reference} (id={law.id}, "
                f"lang={final_language}, confidence={lang_confidence:.2f})"
            )

            # Auto-index in la recherche plein texte (v2.1 SearchService integration)
            try:
                from app.services.search_service import SearchService
                search_service = SearchService(self.db, use_cache=True)
                await search_service.index_law(law)
                logger.info(f"🔍 Auto-indexed law {law.id} in search engine")
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-index law {law.id}: {e}")

            return law

        except DuplicateReferenceError:
            await self.db.rollback()
            raise  # Re-raise without wrapping

        except IntegrityError as e:
            await self.db.rollback()
            if "unique constraint" in str(e).lower():
                raise DuplicateReferenceError(
                    f"Law with reference '{law_data.reference}' already exists"
                )
            raise LawServiceError(f"Database integrity error: {str(e)}")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error creating law: {str(e)}")
            raise LawServiceError(f"Failed to create law: {str(e)}")

    async def get_law(
        self,
        law_id: int,
        include_articles: bool = False,
        include_category: bool = True
    ) -> Law:
        """
        Get a single law by ID with optional eager loading.

        Args:
            law_id: Law ID
            include_articles: Whether to include articles
            include_category: Whether to include category

        Returns:
            Law object

        Raises:
            LawNotFoundError: If law not found
        """
        try:
            logger.debug(f"🔍 Fetching law id={law_id}")

            # Build query with eager loading
            query = select(Law).where(Law.id == law_id)

            if include_category:
                query = query.options(joinedload(Law.category))

            if include_articles:
                query = query.options(selectinload(Law.articles))

            result = await self.db.execute(query)
            law = result.scalar_one_or_none()

            if not law:
                raise LawNotFoundError(f"Law with id={law_id} not found")

            logger.debug(f"✅ Law found: {law.reference}")
            return law

        except LawNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ Error fetching law: {str(e)}")
            raise LawServiceError(f"Failed to fetch law: {str(e)}")

    async def get_law_by_reference(self, reference: str) -> Optional[Law]:
        """
        Get a law by its unique reference.

        Args:
            reference: Law reference (e.g., "LOI-2024-001")

        Returns:
            Law object or None if not found
        """
        try:
            query = select(Law).where(Law.reference == reference)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"❌ Error fetching law by reference: {str(e)}")
            raise LawServiceError(f"Failed to fetch law: {str(e)}")

    async def list_laws(self, filters: LawFilters) -> LawListResponse:
        """
        List laws with filtering and pagination.

        v2.1 Features:
        - Filter by language (fr/en)
        - Filter by category, status, type
        - Date range filtering
        - Sorting and pagination

        Args:
            filters: Filter and pagination parameters

        Returns:
            LawListResponse with paginated results
        """
        try:
            logger.info(f"📋 Listing laws (page={filters.page}, filters={filters.model_dump(exclude_unset=True)})")

            # Build base query
            query = select(Law).options(joinedload(Law.category))

            # Apply filters
            query = self._build_filters_query(query, filters)

            # Count total (before pagination)
            count_query = select(func.count()).select_from(Law)
            count_query = self._build_filters_query(count_query, filters)
            total_result = await self.db.execute(count_query)
            total = total_result.scalar_one()

            # Apply sorting
            query = self._apply_sorting(query, filters)

            # Apply pagination
            offset = (filters.page - 1) * filters.per_page
            query = query.offset(offset).limit(filters.per_page)

            # Execute query
            result = await self.db.execute(query)
            laws = result.scalars().unique().all()

            # Calculate pages
            pages = (total + filters.per_page - 1) // filters.per_page

            logger.info(f"✅ Found {total} laws ({len(laws)} on page {filters.page})")

            return LawListResponse(
                items=[LawResponse.model_validate(law) for law in laws],
                total=total,
                page=filters.page,
                per_page=filters.per_page,
                pages=pages,
                filters_applied=filters.model_dump(exclude_unset=True)
            )

        except Exception as e:
            logger.error(f"❌ Error listing laws: {str(e)}")
            raise LawServiceError(f"Failed to list laws: {str(e)}")

    async def update_law(self, law_id: int, law_data: LawUpdate) -> Law:
        """
        Update an existing law with partial data.

        v2.1 Features:
        - Re-detects language if content changed
        - Re-suggests categories if content changed

        Args:
            law_id: Law ID to update
            law_data: Partial update data

        Returns:
            Updated Law object

        Raises:
            LawNotFoundError: If law not found
            DuplicateReferenceError: If new reference conflicts
        """
        assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
        assert law_data is not None, "law_data must not be None"

        try:
            logger.info(f"✏️ Updating law id={law_id}")

            # Fetch existing law
            law = await self.get_law(law_id, include_category=True)

            # Get update data
            update_dict = law_data.model_dump(exclude_unset=True)

            # Check if reference changed
            if "reference" in update_dict and update_dict["reference"] != law.reference:
                await self._validate_unique_reference(
                    update_dict["reference"],
                    exclude_id=law_id
                )

            # Check if content changed (requires re-detection)
            content_changed = "content" in update_dict

            if content_changed:
                new_content = update_dict["content"]

                # Re-detect language
                detected_lang, lang_confidence = await self._detect_language_and_confidence(
                    new_content
                )

                # Re-suggest categories
                suggested_cats, cat_confidence = await self._suggest_categories(
                    new_content
                )

                # Update v2.1 fields
                update_dict["detected_language"] = detected_lang
                update_dict["language_confidence"] = lang_confidence
                update_dict["suggested_categories"] = suggested_cats
                update_dict["category_confidence"] = cat_confidence

                # Update language if not explicitly provided
                if "language" not in update_dict:
                    update_dict["language"] = detected_lang

                logger.info(
                    f"🔄 Content changed, re-detected: lang={detected_lang}, "
                    f"confidence={lang_confidence:.2f}"
                )

            # Apply updates
            for key, value in update_dict.items():
                setattr(law, key, value)

            # Update timestamp
            law.updated_at = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(law)

            logger.info(f"✅ Law updated: {law.reference}")

            # Update search index (v2.1 SearchService integration)
            try:
                from app.services.search_service import SearchService
                search_service = SearchService(self.db, use_cache=True)
                await search_service.update_law_index(law_id, law)
                logger.info(f"🔍 Updated search index for law {law_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update search index for law {law_id}: {e}")

            return law

        except (LawNotFoundError, DuplicateReferenceError):
            raise
        except IntegrityError as e:
            await self.db.rollback()
            raise LawServiceError(f"Database integrity error: {str(e)}")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error updating law: {str(e)}")
            raise LawServiceError(f"Failed to update law: {str(e)}")

    async def delete_law(self, law_id: int) -> bool:
        """
        Delete a law by ID.

        Note: Articles are cascade deleted automatically.

        Args:
            law_id: Law ID to delete

        Returns:
            True if deleted successfully

        Raises:
            LawNotFoundError: If law not found
        """
        try:
            logger.info(f"🗑️ Deleting law id={law_id}")

            # Fetch law (raises if not found)
            law = await self.get_law(law_id)

            # Delete from search index first (v2.1 SearchService integration)
            try:
                from app.services.search_service import SearchService
                search_service = SearchService(self.db, use_cache=True)
                await search_service.delete_law_index(law_id)
                logger.info(f"🔍 Deleted law {law_id} from search index")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete law {law_id} from search index: {e}")

            # Delete law (cascade deletes articles)
            await self.db.delete(law)
            await self.db.commit()

            logger.info(f"✅ Law deleted: {law.reference}")
            return True

        except LawNotFoundError:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error deleting law: {str(e)}")
            raise LawServiceError(f"Failed to delete law: {str(e)}")

    # ========================================================================
    # Search Operations
    # ========================================================================

    async def search_laws(
        self,
        query: str,
        filters: Optional[LawFilters] = None
    ) -> SearchResponse:
        """
        Search laws by title and content with optional filters.

        Args:
            query: Search query string
            filters: Optional filters to apply

        Returns:
            SearchResponse with matching laws and relevance scores
        """
        try:
            import time
            start_time = time.time()

            logger.info(f"🔍 Searching laws: query='{query}'")

            # Build search query
            search_filter = or_(
                Law.title.ilike(f"%{query}%"),
                Law.content.ilike(f"%{query}%"),
                Law.reference.ilike(f"%{query}%")
            )

            base_query = select(Law).where(search_filter).options(joinedload(Law.category))

            # Apply additional filters if provided
            if filters:
                base_query = self._build_filters_query(base_query, filters)

            # Execute search
            result = await self.db.execute(base_query)
            laws = result.scalars().unique().all()

            # Calculate simple relevance scores
            search_results = []
            for law in laws:
                score = self._calculate_relevance_score(query, law)
                search_results.append(SearchResult(
                    **LawResponse.model_validate(law).model_dump(),
                    relevance_score=score
                ))

            # Sort by relevance
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)

            search_time_ms = (time.time() - start_time) * 1000

            logger.info(f"✅ Search complete: {len(search_results)} results in {search_time_ms:.1f}ms")

            return SearchResponse(
                query=query,
                results=search_results,
                total=len(search_results),
                search_time_ms=search_time_ms
            )

        except Exception as e:
            logger.error(f"❌ Error searching laws: {str(e)}")
            raise LawServiceError(f"Failed to search laws: {str(e)}")

    # ========================================================================
    # Category Operations
    # ========================================================================

    async def create_category(self, category_data: CategoryCreate) -> Category:
        """Create a new category."""
        try:
            logger.info(f"📝 Creating category: {category_data.name}")

            category = Category(**category_data.model_dump())
            self.db.add(category)
            await self.db.commit()
            await self.db.refresh(category)

            logger.info(f"✅ Category created: {category.name} (id={category.id})")
            return category

        except IntegrityError as e:
            await self.db.rollback()
            raise LawServiceError(f"Category creation failed: {str(e)}")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error creating category: {str(e)}")
            raise LawServiceError(f"Failed to create category: {str(e)}")

    async def get_category(self, category_id: int) -> Category:
        """Get a category by ID."""
        try:
            query = select(Category).where(Category.id == category_id)
            result = await self.db.execute(query)
            category = result.scalar_one_or_none()

            if not category:
                raise CategoryNotFoundError(f"Category with id={category_id} not found")

            return category
        except CategoryNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ Error fetching category: {str(e)}")
            raise LawServiceError(f"Failed to fetch category: {str(e)}")

    async def list_categories(self) -> List[CategoryResponse]:
        """List all categories with law counts."""
        try:
            query = (
                select(
                    Category,
                    func.count(Law.id).label("law_count")
                )
                .outerjoin(Law, Category.id == Law.category_id)
                .group_by(Category.id)
                .order_by(Category.name)
            )

            result = await self.db.execute(query)
            rows = result.all()

            categories = []
            for category, law_count in rows:
                cat_dict = CategoryResponse.model_validate(category).model_dump()
                cat_dict["law_count"] = law_count
                categories.append(CategoryResponse(**cat_dict))

            logger.info(f"✅ Listed {len(categories)} categories")
            return categories

        except Exception as e:
            logger.error(f"❌ Error listing categories: {str(e)}")
            raise LawServiceError(f"Failed to list categories: {str(e)}")

    # ========================================================================
    # Statistics Operations
    # ========================================================================

    async def get_language_stats(self) -> LanguageStats:
        """
        Get statistics on law distribution by language (v2.1 feature).

        Returns:
            LanguageStats with counts by language
        """
        try:
            logger.info("📊 Calculating language statistics")

            # Count by language
            query = (
                select(
                    Law.language,
                    func.count(Law.id).label("count")
                )
                .group_by(Law.language)
            )

            result = await self.db.execute(query)
            rows = result.all()

            # Build stats
            french = 0
            english = 0
            unknown = 0

            for lang, count in rows:
                if lang == "fr":
                    french = count
                elif lang == "en":
                    english = count
                else:
                    unknown += count

            total = french + english + unknown

            stats = LanguageStats(
                french=french,
                english=english,
                unknown=unknown,
                total=total
            )

            logger.info(f"✅ Language stats: fr={french}, en={english}, unknown={unknown}")
            return stats

        except Exception as e:
            logger.error(f"❌ Error calculating language stats: {str(e)}")
            raise LawServiceError(f"Failed to calculate language stats: {str(e)}")

    async def get_law_stats(self) -> LawStats:
        """Get comprehensive law statistics."""
        assert self.db is not None, "Database session must be initialized"
        assert hasattr(self, "db"), "LawService must have a db attribute"

        try:
            logger.info("📊 Calculating comprehensive law statistics")

            # Total laws and articles
            total_laws_query = select(func.count(Law.id))
            total_laws = (await self.db.execute(total_laws_query)).scalar_one()

            total_articles_query = select(func.count(Article.id))
            total_articles = (await self.db.execute(total_articles_query)).scalar_one()

            # Language stats
            lang_stats = await self.get_language_stats()

            # Status distribution
            status_query = (
                select(Law.status, func.count(Law.id))
                .group_by(Law.status)
            )
            status_result = await self.db.execute(status_query)
            by_status = {status: count for status, count in status_result.all()}

            # Type distribution
            type_query = (
                select(Law.type, func.count(Law.id))
                .group_by(Law.type)
            )
            type_result = await self.db.execute(type_query)
            by_type = {law_type: count for law_type, count in type_result.all()}

            # Top categories
            cat_query = (
                select(
                    Category.id,
                    Category.name,
                    func.count(Law.id).label("law_count")
                )
                .outerjoin(Law, Category.id == Law.category_id)
                .group_by(Category.id, Category.name)
                .order_by(func.count(Law.id).desc())
                .limit(5)
            )
            cat_result = await self.db.execute(cat_query)
            top_categories = [
                CategoryStats(
                    category_id=cat_id,
                    category_name=cat_name,
                    law_count=count,
                    percentage=(count / total_laws * 100) if total_laws > 0 else 0.0
                )
                for cat_id, cat_name, count in cat_result.all()
            ]

            # Latest publication
            latest_query = (
                select(Law.publication_date)
                .where(Law.publication_date.isnot(None))
                .order_by(Law.publication_date.desc())
                .limit(1)
            )
            latest_result = await self.db.execute(latest_query)
            latest_publication = latest_result.scalar_one_or_none()

            # Average articles per law
            avg_articles = total_articles / total_laws if total_laws > 0 else 0.0

            stats = LawStats(
                total_laws=total_laws,
                total_articles=total_articles,
                by_language=lang_stats,
                by_status=by_status,
                by_type=by_type,
                top_categories=top_categories,
                avg_articles_per_law=avg_articles,
                latest_publication=latest_publication
            )

            logger.info(f"✅ Law stats calculated: {total_laws} laws, {total_articles} articles")
            return stats

        except Exception as e:
            logger.error(f"❌ Error calculating law stats: {str(e)}")
            raise LawServiceError(f"Failed to calculate law stats: {str(e)}")

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    async def _detect_language_and_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detect language and confidence score using LanguageDetector.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence_score)
        """
        try:
            result = self.language_detector.detect(text)
            return result["language"], result["confidence"]
        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {str(e)}, defaulting to 'fr'")
            return "fr", 0.0

    async def _suggest_categories(self, text: str) -> Tuple[List[int], Optional[float]]:
        """
        Suggest top 3 categories using DocumentClassifier.

        Args:
            text: Text to classify

        Returns:
            Tuple of (category_ids, top_confidence)
        """
        try:
            # Get top 3 category suggestions (using top_k parameter)
            predictions = self.document_classifier.classify(text, top_k=3)

            if not predictions:
                return [], None

            # Extract category IDs and top confidence
            category_ids = [pred[0] for pred in predictions]
            top_confidence = predictions[0][1] if predictions else None

            return category_ids, top_confidence

        except Exception as e:
            logger.warning(f"⚠️ Category suggestion failed: {str(e)}")
            return [], None

    async def _validate_unique_reference(
        self,
        reference: str,
        exclude_id: Optional[int] = None
    ) -> None:
        """
        Validate that reference is unique.

        Args:
            reference: Reference to check
            exclude_id: Optional law ID to exclude (for updates)

        Raises:
            DuplicateReferenceError: If reference exists
        """
        query = select(Law).where(Law.reference == reference)

        if exclude_id:
            query = query.where(Law.id != exclude_id)

        result = await self.db.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            raise DuplicateReferenceError(
                f"Law with reference '{reference}' already exists (id={existing.id})"
            )

    def _build_filters_query(self, query, filters: LawFilters):
        """
        Apply filters to a SQLAlchemy query.

        Args:
            query: Base SQLAlchemy query
            filters: LawFilters object

        Returns:
            Modified query with filters applied
        """
        conditions = []

        # Language filter (v2.1)
        if filters.language:
            conditions.append(Law.language == filters.language)

        # Category filter
        if filters.category_id:
            conditions.append(Law.category_id == filters.category_id)

        # Status filter
        if filters.status:
            conditions.append(Law.status == filters.status)

        # Type filter
        if filters.type:
            conditions.append(Law.type == filters.type)

        # Date range filters
        if filters.year_from:
            conditions.append(extract("year", Law.publication_date) >= filters.year_from)

        if filters.year_to:
            conditions.append(extract("year", Law.publication_date) <= filters.year_to)

        # Apply all conditions
        if conditions:
            query = query.where(and_(*conditions))

        return query

    def _apply_sorting(self, query, filters: LawFilters):
        """
        Apply sorting to a query.

        Args:
            query: Base query
            filters: Filters with sorting params

        Returns:
            Query with sorting applied
        """
        # Map sort fields to model attributes
        sort_field_map = {
            "created_at": Law.created_at,
            "updated_at": Law.updated_at,
            "publication_date": Law.publication_date,
            "title": Law.title,
            "reference": Law.reference,
        }

        sort_field = sort_field_map.get(filters.sort_by, Law.created_at)

        if filters.sort_order == "asc":
            query = query.order_by(sort_field.asc())
        else:
            query = query.order_by(sort_field.desc())

        return query

    def _calculate_relevance_score(self, query: str, law: Law) -> float:
        """
        Calculate simple relevance score for search results.

        Args:
            query: Search query
            law: Law object

        Returns:
            Relevance score (0.0-1.0)
        """
        query_lower = query.lower()
        score = 0.0

        # Title match (highest weight)
        if query_lower in law.title.lower():
            score += 0.5
            if law.title.lower().startswith(query_lower):
                score += 0.2

        # Reference match
        if query_lower in law.reference.lower():
            score += 0.3

        # Content match (lower weight)
        if query_lower in law.content.lower():
            score += 0.2

        # Normalize to 0.0-1.0
        return min(score, 1.0)
