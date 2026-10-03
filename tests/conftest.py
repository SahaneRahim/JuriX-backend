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

La base de test vit sur le MEME serveur que la base de developpement (le
conteneur `jurix-pg`, port 5433). Il suffit d'y creer la base une fois :

    docker exec jurix-pg psql -U jurix -d jurix_dev -c "CREATE DATABASE jurix_test"

Puis, si l'URL diffère du défaut :

    export TEST_DATABASE_URL=postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_test

Sans base joignable, les tests qui en dépendent sont **ignorés avec un message
explicite** — jamais silencieusement verts. Le message distingue un serveur
injoignable d'une base absente : les confondre a laisse environ 93 tests
ignores sous « PostgreSQL injoignable » alors que le serveur repondait.

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

# ---------------------------------------------------------------------------
# AVANT tout import de app.* : settings est instancie au chargement de
# app.core.config, et le moteur SYNCHRONE partage lit settings.DATABASE_URL a
# l'import de app.core.database. Sans cette surcharge posee ici, ce moteur
# pointerait la base de DEVELOPPEMENT (port 5432) pendant toute la suite, et
# les tests du pipeline y ecriraient pour de bon. pytest_configure serait trop
# tard : les imports ont deja eu lieu.
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

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

# Deux causes, deux remedes : les confondre a coute environ 93 tests ignores
# sous « PostgreSQL injoignable » alors que le serveur repondait tres bien et
# qu'il ne manquait que la base.
_PG_HINTS = {
    "serveur": (
        f"Serveur PostgreSQL injoignable pour {TEST_DATABASE_URL}.\n"
        "  Le conteneur de developpement est-il demarre ?  docker start jurix-pg\n"
        "  (ou definissez TEST_DATABASE_URL)"
    ),
    "base": (
        f"Le serveur repond, mais la base de test n'existe pas ({TEST_DATABASE_URL}).\n"
        '  docker exec jurix-pg psql -U jurix -d jurix_dev -c "CREATE DATABASE jurix_test"'
    ),
}

# Fixtures qui exigent une base : sert au marquage automatique en integration.
_DB_FIXTURES = {
    "pg_engine",
    "db_session",
    "async_db_session",
    "category_ids",
    "sync_db_session",
    "migrated_categories",
    "client",
    "admin_client",
    "sample_law",
    "test_user",
    "test_admin_user",
    "test_superadmin_user",
    "superadmin_client",
    "auth_headers",
    "admin_headers",
    "as_admin",
}

# None tant que la sonde n'a pas tourne ; ensuite "" si la base est
# disponible, ou la cle de _PG_HINTS qui decrit ce qui manque.
_pg_probleme: str | None = None


def _check_pg() -> str:
    """
    Sonde la base de test une seule fois.

    Rend "" si elle est utilisable, sinon "base" ou "serveur". La distinction
    passe par le TYPE d'exception : asyncpg leve InvalidCatalogNameError quand
    le serveur repond mais que la base n'existe pas.
    """
    global _pg_probleme
    if _pg_probleme is not None:
        return _pg_probleme
    try:
        import asyncio

        import asyncpg

        dsn = TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

        async def _probe():
            conn = await asyncpg.connect(dsn, timeout=5)
            await conn.close()

        asyncio.run(_probe())
        _pg_probleme = ""
    except Exception as exc:
        import asyncpg

        _pg_probleme = (
            "base" if isinstance(exc, asyncpg.InvalidCatalogNameError) else "serveur"
        )
    return _pg_probleme


# ==================== HOOKS PYTEST ====================


def pytest_configure(config):
    """Déclare les marqueurs et neutralise les dépendances externes."""
    config.addinivalue_line("markers", "integration: nécessite une base PostgreSQL")
    config.addinivalue_line("markers", "slow: test lent")
    config.addinivalue_line("markers", "unit: test unitaire, sans base")
    config.addinivalue_line(
        "markers",
        "docling: execute le vrai Docling (modeles dans le cache Hugging Face requis)",
    )
    config.addinivalue_line(
        "markers",
        "gemma: execute le vrai modele EmbeddingGemma local (onnxruntime et fichiers requis)",
    )

    # Cle factice IMPOSEE, et pas seulement quand aucune n'est configuree.
    #
    # L'ancienne regle ne posait la cle factice que si la variable etait vide.
    # Or une vraie cle vit dans .env : tout test qui atteignait le vrai service
    # d'embeddings — ComparisonService construit un SearchService reel —
    # appelait donc l'API Gemini PAYANTE, et pouvait meme passer. Une suite de
    # tests ne doit jamais rien couter.
    #
    # JURIX_E2E=1 garde la vraie cle, pour un essai de bout en bout delibere.
    from app.core.config import settings

    if os.environ.get("JURIX_E2E") != "1":
        settings.GEMINI_API_KEY = "test-key-not-a-real-credential"


def pytest_collection_modifyitems(config, items):
    """
    Marque `integration` tout test utilisant une fixture de base, et l'ignore
    si PostgreSQL n'est pas joignable.

    Le marquage est automatique plutôt que déclaré fichier par fichier : la
    dépendance à la base est déjà exprimée par les fixtures demandées, la
    dupliquer dans 21 fichiers ne ferait que créer une source de divergence.
    """
    probleme = None

    for item in items:
        needs_db = bool(_DB_FIXTURES & set(getattr(item, "fixturenames", ())))
        if not needs_db:
            continue
        item.add_marker(pytest.mark.integration)
        if probleme is None:
            probleme = _check_pg()
        if probleme:
            item.add_marker(pytest.mark.skip(reason=_PG_HINTS[probleme]))


# ==================== BASE DE DONNEES ====================


@pytest.fixture(scope="session", autouse=True)
def _bind_sync_engine_to_test_db():
    """
    Rebranche le moteur synchrone partage sur la base de TEST.

    Ceinture et bretelles : la surcharge de DATABASE_URL en tete de fichier
    suffit normalement, mais elle depend de l'ordre des imports. Un moteur
    synchrone pointe sur la base de developpement ferait ecrire les tests du
    pipeline dans de vraies donnees ; le rebranchement est explicite et
    verifiable.
    """
    from sqlalchemy import create_engine

    from app.core import database

    sync_url = TEST_DATABASE_URL.replace("+asyncpg", "")
    if str(database.sync_engine.url) == sync_url:
        yield
        return

    engine = create_engine(sync_url, pool_pre_ping=True, future=True)
    previous = database.sync_engine
    database.sync_engine = engine
    database.SyncSessionLocal.configure(bind=engine)
    try:
        yield
    finally:
        engine.dispose()
        database.sync_engine = previous
        database.SyncSessionLocal.configure(bind=previous)


# ==================== DOUBLURE DU SERVICE D'EMBEDDINGS ====================


class _EmbeddingsDeterministes:
    """
    Remplace le service d'embeddings reel derriere SearchService.

    POURQUOI. Un SearchService construit dans un test prend le singleton du
    module, donc le VRAI service : hier un client Gemini (avec la vraie cle du
    .env, donc des appels payants), demain le modele local (1,5 Go en memoire).
    La recherche hybride avalant les echecs semantiques, ces tests restaient
    verts dans tous les cas sans rien prouver.

    Les vecteurs sont deterministes — derives d'une empreinte stable du texte,
    jamais de hash(), qui change d'une execution a l'autre — et unitaires, comme
    ceux du vrai service. Un test qui veut un comportement precis remplace
    lui-meme le singleton : sa fixture s'execute apres celle-ci.
    """

    TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    TASK_QUERY = "RETRIEVAL_QUERY"

    class _Fournisseur:
        label = "doublure de test"
        fingerprint = "doublure|tests"

    provider = _Fournisseur()

    def __init__(self, dim: int):
        self.EMBEDDING_DIM = dim

    def prechauffer(self):
        pass

    def generate_embedding(self, text, normalize=True, task_type=TASK_DOCUMENT):
        import hashlib

        import numpy as np

        graine = int.from_bytes(
            hashlib.sha256(f"{task_type}\x00{text}".encode()).digest()[:8], "big"
        )
        v = np.random.default_rng(graine).standard_normal(self.EMBEDDING_DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)

    async def generate_embedding_async(self, text, task_type=TASK_QUERY):
        return self.generate_embedding(text, True, task_type)


@pytest.fixture(autouse=True)
def _epingler_le_fournisseur_d_embeddings(request):
    """
    Les tests construisent un EmbeddingService sur Gemini, sauf ceux marques
    `gemma`.

    Le defaut de production est EmbeddingGemma, en local. Sans cet
    epinglage, tout test qui construit un service par defaut exigerait le
    modele sur disque — 300 Mo, absents d'une machine d'integration — et le
    chargerait en memoire. Sur Gemini, la cle de test est fausse et les
    tests doublent le client : rien ne part sur le reseau.

    Les tests qui portent sur le choix du fournisseur le reglent eux-memes ;
    ceux du vrai modele sont marques `gemma`.

    Affectation directe et non `monkeypatch` : une fixture automatique qui le
    demanderait l'instancierait AVANT la doublure ci-dessous, donc le
    demonterait APRES elle. Les `monkeypatch` des tests sur les singletons de
    search_service seraient alors defaits apres la restauration, et la
    doublure fuirait d'un test a l'autre.
    """
    if request.node.get_closest_marker("gemma"):
        yield
        return
    from app.core.config import settings

    avant = settings.EMBEDDING_PROVIDER
    avant_classement = settings.INTENT_CLASSIFIER
    settings.EMBEDDING_PROVIDER = "gemini"
    # Le classement local d'intention passerait par la doublure d'embeddings
    # et rendrait des verdicts arbitraires : les tests du routage doublent le
    # modele (Gemini) ; ceux du classement local le reglent eux-memes.
    settings.INTENT_CLASSIFIER = "gemini"
    try:
        yield
    finally:
        settings.EMBEDDING_PROVIDER = avant
        settings.INTENT_CLASSIFIER = avant_classement


@pytest.fixture(autouse=True)
def _doubler_le_service_d_embeddings(request):
    """
    Installe la doublure avant chaque test, et restaure l'etat ensuite.

    Les tests marques `gemma` sont exemptes : ils existent precisement pour
    executer le vrai modele.
    """
    if request.node.get_closest_marker("gemma"):
        yield
        return

    from app.services import search_service
    from app.services.embedding_service import EmbeddingService

    avant = (search_service._embedding_service_instance, search_service._singletons_initialized)
    search_service._embedding_service_instance = _EmbeddingsDeterministes(
        EmbeddingService.EMBEDDING_DIM
    )
    search_service._singletons_initialized = True
    try:
        yield
    finally:
        search_service._embedding_service_instance, search_service._singletons_initialized = avant


@pytest.fixture(scope="session")
def _migrated_schema():
    """
    Applique `alembic upgrade head` une seule fois par session.

    Fixture SYNCHRONE volontairement : Alembic utilise un moteur psycopg2, donc
    aucune boucle asyncio n'est impliquée et le schéma peut être construit une
    fois pour toute la session sans lier quoi que ce soit à une boucle donnée.
    """
    from alembic.config import Config

    from alembic import command

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
    "query_cache", "embedding_cache", "search_events",
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
    """
    Session avec les 14 domaines juridiques canoniques pré-insérés.

    Les identifiants sont attribués dans l'ordre INVERSE de la liste canonique,
    et la séquence démarre à 1000. Ce n'est pas un détail de mise en place :
    l'ancien pipeline écrivait dans `laws.category_id` la position d'une
    catégorie dans un dictionnaire Python, et l'ancienne version de cette
    fixture — qui insérait les catégories avec `id=1, 2, 3...` dans l'ordre —
    rendait ce bug invisible. Avec des identifiants désordonnés, tout code qui
    résout une catégorie par position échoue immédiatement.
    """
    from app.models.law import Category
    from app.services.legal_domain_classifier import CANONICAL_DOMAINS

    await async_db_session.execute(text("SELECT setval('categories_id_seq', 1000, true)"))
    for order, nom in enumerate(reversed(CANONICAL_DOMAINS)):
        async_db_session.add(Category(name=nom, description=nom, display_order=order))
    await async_db_session.commit()
    return async_db_session


@pytest_asyncio.fixture
async def category_ids(db_session: AsyncSession) -> dict:
    """
    {nom du domaine: identifiant}, lu en base.

    Les fixtures de test ecrivaient `category_id=1` en dur. C'etait la meme
    hypothese que celle du bug d'origine — que la position vaut identifiant —
    et elle rendait toute la suite aveugle a ce defaut.
    """
    from sqlalchemy import select as sa_select

    from app.models.law import Category

    rows = (await db_session.execute(sa_select(Category.name, Category.id))).all()
    return {name: identifier for name, identifier in rows}


@pytest.fixture
def sync_db_session(request):
    """
    Session SYNCHRONE sur la base de test, pour le pipeline et les scripts.

    Le pipeline d'ingestion et les scripts de maintenance sont synchrones
    (`SyncSessionLocal`) : les tester à travers une session async ne testerait
    pas le chemin réel. Le nettoyage est fait ici même, la fixture async
    n'ayant pas de prise sur les connexions psycopg2.
    """
    from sqlalchemy import text as sa_text

    from app.core.database import SyncSessionLocal

    request.getfixturevalue("_migrated_schema")
    truncate = sa_text(f"TRUNCATE {', '.join(_DATA_TABLES)} RESTART IDENTITY CASCADE")

    with SyncSessionLocal() as session:
        session.execute(truncate)
        session.commit()
        try:
            yield session
        finally:
            session.rollback()
            session.execute(truncate)
            session.commit()


@pytest.fixture
def migrated_categories(sync_db_session):
    """
    Base dont la table `categories` a la forme d'AVANT la migration
    d9e0f1a2b3c4 — types de documents inclus, lois rattachées — puis remise à
    l'état canonique par cette migration.

    C'est le seul test qui exerce réellement l'ordre des étapes : sans le
    re-pointage des lois avant la suppression, l'étape 4 lèverait
    ForeignKeyViolation.
    """
    from sqlalchemy import text as sa_text

    from app.models.law import Category, Law

    sync_db_session.execute(sa_text("DELETE FROM laws"))
    sync_db_session.execute(sa_text("DELETE FROM categories"))
    legacy = [
        "Droit Civil", "Droit Pénal", "Droit Commercial", "Droit du Travail",
        "Droit Fiscal", "Droit Administratif", "Droit de la Famille",
        "Droit des Affaires", "Lois Internationales Ratifiées", "Lois",
        "Ordonnances", "Décrets", "Arrêtés", "Autres", "Droit OHADA",
    ]
    rows = {}
    for order, nom in enumerate(legacy):
        category = Category(name=nom, description=nom, display_order=order)
        sync_db_session.add(category)
        sync_db_session.flush()
        rows[nom] = category.id
    # Des lois pointant sur une ligne vouee a disparaitre et sur une ligne
    # vouee a etre fusionnee : les deux chemins de l'etape 3.
    for reference, nom in [("LEG-1", "Lois"), ("LEG-2", "Droit des Affaires"),
                           ("LEG-3", "Droit OHADA"), ("LEG-4", "Décrets")]:
        sync_db_session.add(Law(
            reference=reference, title=f"Loi {reference}", type="loi",
            content="Contenu.", language="fr", status="published",
            category_id=rows[nom],
        ))
    sync_db_session.commit()

    from alembic.config import Config

    from alembic import command

    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    config = Config("alembic.ini")
    try:
        command.downgrade(config, "c8d9e0f1a2b3")
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous

    sync_db_session.expire_all()
    sync_db_session.commit()
    return sync_db_session


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


@pytest.fixture
def as_superadmin(test_superadmin_user):
    """
    Se fait passer pour un superadministrateur.

    Nécessaire parce que certaines opérations — la suppression d'un compte, par
    exemple — exigent ce rôle et non simplement `admin`. Sans ce niveau, on ne
    peut pas tester la différence entre les deux.
    """
    from app.core.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: test_superadmin_user
    try:
        yield test_superadmin_user
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest_asyncio.fixture
async def superadmin_client(client: AsyncClient, as_superadmin) -> AsyncClient:
    """Client HTTP authentifié comme superadministrateur."""
    return client
