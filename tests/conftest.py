"""
Configuration et fixtures pytest pour JuriX.

## Pourquoi PostgreSQL et non SQLite

L'ancien harnais utilisait SQLite en mémoire. C'était intenable : le cœur du
produit est du SQL PostgreSQL brut — `websearch_to_tsquery`, `ts_rank_cd`,
`DISTINCT ON`, `= ANY()`, `similarity()`, l'extension `vector` — et les objets
dont il dépend (colonnes `search_vector`, index GIN, triggers, tables
`query_cache` / `embedding_cache`) n'existent QUE dans les migrations, jamais
dans `Base.metadata`. `create_all` ne pouvait donc pas produire un schéma
utilisable, et une suite verte sur SQLite aurait certifié du code qui renvoie
500 en production.

Le harnais applique désormais `alembic upgrade head` sur une vraie base
PostgreSQL, une fois par session.

## Mise en place

    docker run -d --name jurix-pg-test -p 5433:5432 \
      -e POSTGRES_USER=jurix -e POSTGRES_PASSWORD=jurix -e POSTGRES_DB=jurix_test \
      pgvector/pgvector:pg16

Puis, si l'URL diffère du défaut :

    export TEST_DATABASE_URL=postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_test

Sans base joignable, les tests qui en dépendent sont **ignorés avec un message
explicite** — jamais silencieusement verts.

## Isolation

Chaque test reçoit une session neuve, et les tables de données sont purgées par
`TRUNCATE ... RESTART IDENTITY CASCADE` après chaque test.

L'isolation transactionnelle (transaction externe + rollback) a été essayée et
écartée : le code testé appelle `commit()` en interne, ce qui impose des
SAVEPOINT, or le dialecte asyncpg les implémente via `Connection.transaction()`,
qui refuse de s'exécuter dans une transaction ouverte manuellement. Le mode
`rollback_only` évite les SAVEPOINT mais casse les tests qui relisent ce qu'ils
viennent d'écrire. TRUNCATE est plus lent, mais sans surprise.

Author: JuriX Team
"""

import os
from datetime import date
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.database import get_db
from app.main import app

# ==================== CONFIGURATION ====================

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_test",
)

_PG_HINT = (
    f"PostgreSQL injoignable sur {TEST_DATABASE_URL}.\n"
    "  docker run -d --name jurix-pg-test -p 5433:5432 \\\n"
    "    -e POSTGRES_USER=jurix -e POSTGRES_PASSWORD=jurix "
    "-e POSTGRES_DB=jurix_test \\\n"
    "    pgvector/pgvector:pg16\n"
    "  (ou definissez TEST_DATABASE_URL)"
)

# Fixtures qui exigent une base : sert au marquage automatique en integration.
_DB_FIXTURES = {
    "pg_engine",
    "db_session",
    "async_db_session",
    "client",
    "admin_client",
    "sample_law",
    "test_user",
    "test_admin_user",
    "test_superadmin_user",
    "auth_headers",
    "admin_headers",
    "as_admin",
}

_pg_available: bool | None = None


def _check_pg() -> bool:
    """Teste une connexion TCP + handshake PostgreSQL, une seule fois."""
    global _pg_available
    if _pg_available is not None:
        return _pg_available
    try:
        import asyncio

        import asyncpg

        dsn = TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

        async def _probe():
            conn = await asyncpg.connect(dsn, timeout=5)
            await conn.close()

        asyncio.run(_probe())
        _pg_available = True
    except Exception:
        _pg_available = False
    return _pg_available


# ==================== HOOKS PYTEST ====================


def pytest_configure(config):
    """Déclare les marqueurs et neutralise les dépendances externes."""
    config.addinivalue_line("markers", "integration: nécessite une base PostgreSQL")
    config.addinivalue_line("markers", "slow: test lent")
    config.addinivalue_line("markers", "unit: test unitaire, sans base")

    # Clé factice si aucune n'est configurée : GeminiService et EmbeddingService
    # refusent de se construire sans clé, ce qui faisait echouer au *setup* des
    # tests qui ne touchent jamais l'API. Construire un client Gemini ne declenche
    # aucun appel reseau — les tests qui appellent reellement l'API la simulent.
    from app.core.config import settings

    if not settings.GEMINI_API_KEY:
        settings.GEMINI_API_KEY = "test-key-not-a-real-credential"


def pytest_collection_modifyitems(config, items):
    """
    Marque `integration` tout test utilisant une fixture de base, et l'ignore
    si PostgreSQL n'est pas joignable.

    Le marquage est automatique plutôt que déclaré fichier par fichier : la
    dépendance à la base est déjà exprimée par les fixtures demandées, la
    dupliquer dans 21 fichiers ne ferait que créer une source de divergence.
    """
    skip_pg = pytest.mark.skip(reason=_PG_HINT)
    available = None

    for item in items:
        needs_db = bool(_DB_FIXTURES & set(getattr(item, "fixturenames", ())))
        if not needs_db:
            continue
        item.add_marker(pytest.mark.integration)
        if available is None:
            available = _check_pg()
        if not available:
            item.add_marker(skip_pg)


# ==================== BASE DE DONNEES ====================


@pytest.fixture(scope="session")
def _migrated_schema():
    """
    Applique `alembic upgrade head` une seule fois par session.

    Fixture SYNCHRONE volontairement : Alembic utilise un moteur psycopg2, donc
    aucune boucle asyncio n'est impliquée et le schéma peut être construit une
    fois pour toute la session sans lier quoi que ce soit à une boucle donnée.
    """
    from alembic import command
    from alembic.config import Config

    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    try:
        command.upgrade(Config("alembic.ini"), "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
    return True


@pytest_asyncio.fixture
async def pg_engine(_migrated_schema):
    """
    Moteur async, créé DANS la boucle du test.

    Portée fonction et non session : un moteur asyncpg est lié à la boucle
    d'événements qui l'a créé. Une fixture de session le rattachait à la boucle
    des fixtures, tandis que chaque test tourne dans la sienne — d'où des
    "attached to a different loop" qui remontaient en HTTP 500 depuis les routes.
    Créer le moteur est peu coûteux ; c'est la migration qui l'était, et elle
    reste faite une seule fois par _migrated_schema.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
    try:
        yield engine
    finally:
        await engine.dispose()


# Tables purgées entre deux tests. alembic_version en est exclue : elle porte
# l'état des migrations, pas des données de test.
_DATA_TABLES = (
    "message_feedback", "messages", "conversations", "persona_interactions",
    "persona_stats", "articles", "laws", "categories", "users",
    "query_cache", "embedding_cache",
)


@pytest_asyncio.fixture
async def async_db_session(pg_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Session isolée par test, nettoyage par TRUNCATE en fin de test.

    L'isolation transactionnelle a été essayée puis écartée : le code testé
    appelle `commit()` en interne (`store_in_pg_cache`, `update_law_search_vector`,
    `reindex_all_laws`), ce qui impose des SAVEPOINT — or le dialecte asyncpg les
    implémente via `Connection.transaction()`, qui refuse de s'exécuter dans une
    transaction ouverte manuellement. Le mode `rollback_only` évite les SAVEPOINT
    mais casse les tests qui relisent ce qu'ils viennent d'écrire.

    TRUNCATE ... RESTART IDENTITY CASCADE est plus lent mais sans surprise, et
    remet aussi les séquences à zéro — ce dont les tests qui écrivent des ids
    explicites (les 12 catégories de référence) ont besoin.
    """
    truncate = text(f"TRUNCATE {', '.join(_DATA_TABLES)} RESTART IDENTITY CASCADE")

    # Purge AVANT et APRES : purger seulement en sortie laisse la base sale si un
    # test est interrompu, et le test suivant echoue alors sur des donnees qui ne
    # lui appartiennent pas — un mode de panne trompeur.
    async with pg_engine.begin() as conn:
        await conn.execute(truncate)

    session_factory = async_sessionmaker(
        pg_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

    async with pg_engine.begin() as conn:
        await conn.execute(truncate)


@pytest_asyncio.fixture
async def db_session(async_db_session: AsyncSession) -> AsyncSession:
    """Session avec les 12 catégories de référence pré-insérées."""
    from app.models.law import Category

    noms = [
        "Droit Civil", "Droit Commercial OHADA", "Droit Pénal",
        "Droit Administratif", "Droit du Travail", "Droit Foncier",
        "Droit de la Famille", "Droit Fiscal", "Droit des Affaires",
        "Droit International", "Droit Constitutionnel", "Procédure Civile",
    ]
    for i, nom in enumerate(noms, start=1):
        async_db_session.add(Category(id=i, name=nom, description=nom))
    await async_db_session.commit()

    # Insérer avec des ids explicites ne fait PAS avancer la séquence : le
    # prochain id auto-généré repartirait à 1 et heurterait "Droit Civil".
    # C'est ce qui faisait échouer tous les tests créant une catégorie via l'API.
    await async_db_session.execute(
        text("SELECT setval('categories_id_seq', (SELECT max(id) FROM categories))")
    )
    await async_db_session.commit()
    return async_db_session


@pytest_asyncio.fixture
async def sample_law(db_session: AsyncSession):
    """Une loi d'exemple rattachée à une catégorie."""
    from app.models.law import Category, Law

    category = Category(name="Test Category for Law", description="Fixture")
    db_session.add(category)
    await db_session.flush()

    law = Law(
        reference="TEST-001",
        title="Sample Test Law",
        type="loi",
        content="This is a sample law content for testing purposes",
        publication_date=date(2024, 1, 15),
        status="published",
        category_id=category.id,
        language="fr",
    )
    db_session.add(law)
    await db_session.commit()
    await db_session.refresh(law)
    return law


# ==================== CLIENT HTTP ====================


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Client HTTP anonyme, branché sur la base de test."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac
    finally:
        # .pop et non .clear() : clear() supprimerait aussi les surcharges
        # posees par d'autres fixtures (authentification notamment).
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def override_get_db(async_db_session: AsyncSession):
    """Surcharge de get_db réutilisable par les tests qui pilotent l'app."""

    async def _override_get_db():
        yield async_db_session

    return _override_get_db


# ==================== AUTHENTIFICATION ====================
#
# Surcharger `get_current_user` suffit à débloquer les trois niveaux d'accès :
# `get_current_active_user`, `get_current_admin_user` et
# `get_current_superadmin_user` en dépendent tous. Les surcharger un par un
# obligerait à traiter chaque endpoint séparément.
#
# `client` reste volontairement anonyme, pour que les tests qui vérifient les
# 401 continuent de fonctionner.


async def _make_user(db: AsyncSession, email: str, username: str, role: str):
    from app.core.auth import hash_password
    from app.models.user import User

    user = User(
        email=email,
        username=username,
        hashed_password=hash_password("MotDePasseTest123"),
        role=role,
        is_active=True,
        is_verified=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Compte sans privilège."""
    return await _make_user(db_session, "user@test.cm", "usertest", "user")


@pytest_asyncio.fixture
async def test_admin_user(db_session: AsyncSession):
    """Compte administrateur."""
    return await _make_user(db_session, "admin@test.cm", "admintest", "admin")


@pytest_asyncio.fixture
async def test_superadmin_user(db_session: AsyncSession):
    """Compte superadministrateur."""
    return await _make_user(db_session, "super@test.cm", "supertest", "superadmin")


@pytest.fixture
def auth_headers(test_user):
    """En-tête Authorization pour un compte sans privilège (jeton réel)."""
    from app.core.auth import create_access_token

    return {"Authorization": f"Bearer {create_access_token({'sub': test_user.email})}"}


@pytest.fixture
def admin_headers(test_admin_user):
    """En-tête Authorization pour un administrateur (jeton réel)."""
    from app.core.auth import create_access_token

    return {"Authorization": f"Bearer {create_access_token({'sub': test_admin_user.email})}"}


@pytest.fixture
def as_admin(test_admin_user):
    """
    Court-circuite l'authentification en se faisant passer pour un administrateur.

    Utile aux tests qui portent sur la logique métier et non sur l'authentification :
    ils n'ont pas à fabriquer un jeton ni à gérer son expiration.
    """
    from app.core.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: test_admin_user
    try:
        yield test_admin_user
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest_asyncio.fixture
async def admin_client(client: AsyncClient, as_admin) -> AsyncClient:
    """Client HTTP authentifié comme administrateur."""
    return client
