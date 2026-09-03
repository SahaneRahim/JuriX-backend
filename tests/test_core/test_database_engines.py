"""
Tests structurels sur les moteurs de base.

Onze appels a create_engine() etaient dissemines dans le code, un par fonction
ou par instance, et aucun n'appelait jamais dispose() : chaque loi traitee et
chaque question posee au RAG laissaient un pool de connexions ouvert jusqu'a
l'arret du processus. Ces tests encodent la regle plutot que de la documenter.
"""

import ast
import pathlib

import pytest

APP_ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
ALLOWED = {APP_ROOT / "core" / "database.py"}


def _modules_calling_create_engine():
    offenders = []
    for path in APP_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in {"create_engine", "create_async_engine"} and path not in ALLOWED:
                offenders.append(f"{path.relative_to(APP_ROOT.parent)}:{node.lineno}")
    return offenders


def test_no_module_builds_its_own_engine():
    """Un seul endroit a le droit de creer un moteur."""
    offenders = _modules_calling_create_engine()

    assert offenders == [], (
        "Moteur cree hors de app/core/database.py : "
        + ", ".join(offenders)
        + ". Utiliser SyncSessionLocal() ou AsyncSessionLocal()."
    )


def test_sync_session_factory_is_shared():
    """Toutes les sessions synchrones partagent le meme moteur."""
    from app.core.database import SyncSessionLocal, sync_engine

    with SyncSessionLocal() as first, SyncSessionLocal() as second:
        assert first.get_bind() is second.get_bind()
        assert first.get_bind() is sync_engine


def test_embedding_service_borrows_the_shared_factory():
    """Le service d'embeddings ne possede plus de moteur en propre."""
    from app.core.database import SyncSessionLocal
    from app.services.embedding_service import EmbeddingService

    service = EmbeddingService(use_cache=True)

    assert service._sync_session_factory is SyncSessionLocal
    assert not hasattr(service, "_sync_engine")


def test_sync_engine_points_at_the_test_database():
    """
    Garde-fou : le moteur synchrone lit settings.DATABASE_URL, soit la base de
    DEVELOPPEMENT si la surcharge de conftest n'a pas pris. Les tests du
    pipeline y ecriraient alors de vraies donnees.
    """
    from app.core.database import sync_engine
    from tests.conftest import TEST_DATABASE_URL

    expected_port = int(TEST_DATABASE_URL.rsplit(":", 1)[1].split("/")[0])
    assert sync_engine.url.port == expected_port


@pytest.mark.asyncio
async def test_close_db_disposes_both_engines(monkeypatch):
    """close_db() doit fermer le moteur async ET le moteur synchrone."""
    from app.core import database

    disposed = {"async": False, "sync": False}

    class _FakeAsyncEngine:
        async def dispose(self):
            disposed["async"] = True

    class _FakeSyncEngine:
        def dispose(self):
            disposed["sync"] = True

    # Les moteurs sont remplaces en entier : AsyncEngine.dispose est en lecture
    # seule, on ne peut pas la substituer attribut par attribut.
    monkeypatch.setattr(database, "engine", _FakeAsyncEngine())
    monkeypatch.setattr(database, "sync_engine", _FakeSyncEngine())

    await database.close_db()

    assert disposed == {"async": True, "sync": True}
