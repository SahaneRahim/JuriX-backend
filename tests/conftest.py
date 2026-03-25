"""
Pytest configuration and fixtures for JuriX tests.

This module provides shared fixtures for all tests including:
- Database session management (async)
- Test client setup
- Mock services
- Sample test data

Author: JuriX Team
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings


# Test database URL (use in-memory SQLite for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_db_engine():
    """Create async engine for tests."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_db_session(async_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create async database session for tests.

    This fixture provides a clean database session for each test,
    with automatic rollback after the test completes.
    """
    async_session = async_sessionmaker(
        async_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """
    Create async HTTP client for API tests with test database.

    This fixture provides an async HTTP client that can make requests
    to the FastAPI application for integration testing, using the test database
    with seeded categories.
    """
    # Override get_db to use test database (with seeded categories)
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

    # Clean up override
    app.dependency_overrides.clear()


@pytest.fixture
def override_get_db(async_db_session: AsyncSession):
    """Override the get_db dependency to use test database."""
    async def _override_get_db():
        yield async_db_session

    return _override_get_db


@pytest_asyncio.fixture
async def db_session(async_db_session):
    """Alias for async_db_session to support existing tests with seed data."""
    from app.models.law import Category

    # Seed categories for tests that expect them
    categories = [
        Category(id=1, name="Droit Civil", description="Droit civil camerounais"),
        Category(id=2, name="Droit Commercial OHADA", description="Droit commercial OHADA"),
        Category(id=3, name="Droit Pénal", description="Droit pénal camerounais"),
        Category(id=4, name="Droit Administratif", description="Droit administratif"),
        Category(id=5, name="Droit du Travail", description="Code du travail"),
        Category(id=6, name="Droit Foncier", description="Droit foncier"),
        Category(id=7, name="Droit de la Famille", description="Droit de la famille"),
        Category(id=8, name="Droit Fiscal", description="Droit fiscal"),
        Category(id=9, name="Droit des Affaires", description="Droit des affaires"),
        Category(id=10, name="Droit International", description="Droit international"),
        Category(id=11, name="Droit Constitutionnel", description="Droit constitutionnel"),
        Category(id=12, name="Procédure Civile", description="Procédure civile"),
    ]

    for cat in categories:
        async_db_session.add(cat)
    await async_db_session.commit()

    return async_db_session


@pytest_asyncio.fixture
async def sample_law(db_session):
    """Create a sample law with category for testing."""
    from app.models.law import Category, Law
    from datetime import date

    # Create a category first
    category = Category(
        name="Test Category for Law",
        description="Category for sample law fixture"
    )
    db_session.add(category)
    await db_session.flush()

    # Create a law
    law = Law(
        reference="TEST-001",
        title="Sample Test Law",
        type="loi",
        content="This is a sample law content for testing purposes",
        publication_date=date(2024, 1, 15),
        status="published",
        category_id=category.id,
        language="fr"
    )
    db_session.add(law)
    await db_session.commit()
    await db_session.refresh(law)

    return law


# Mark configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as unit test")
