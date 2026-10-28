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


def test_une_erreur_du_moteur_synchrone_est_signalee_une_fois(monkeypatch, caplog):
    """
    Une URL refusee par psycopg2 a coupe le cache d'embeddings en production
    sans rien dire : l'erreur etait avalee au niveau DEBUG. Elle est signalee
    une fois par cause, premiere ligne seulement, et jamais pour un sondage.
    """
    import logging

    from app.core import database

    monkeypatch.setattr(database, "_CAUSES_SIGNALEES", set())

    class _Contexte:
        def __init__(self, message, sondage=False):
            self.is_pre_ping = sondage
            self.original_exception = RuntimeError(message)

    with caplog.at_level(logging.WARNING, logger="app.core.database"):
        for _ in range(3):
            database._signaler_une_fois(
                _Contexte("invalid dsn: invalid URI query parameter\nDETAIL: valeur privee")
            )
        database._signaler_une_fois(_Contexte("server closed the connection", sondage=True))

    assert caplog.text.count("Moteur synchrone (psycopg2) en echec") == 1
    assert "invalid URI query parameter" in caplog.text
    assert "valeur privee" not in caplog.text
    assert "server closed" not in caplog.text


# ==================== VERIFICATION DES CONNEXIONS A LA SORTIE DU POOL ====================


class _ConnexionFactice:
    """Ce que _verifier_a_la_sortie lit d'une connexion asyncpg."""

    def __init__(self, fermee=False, panne=None):
        self.fermee = fermee
        self.panne = panne
        self.envois = []

    def is_closed(self):
        return self.fermee

    async def execute(self, sql, timeout=None):
        self.envois.append((sql, timeout))
        if self.panne is not None:
            raise self.panne
        return "SELECT 1"


def _fiche(connexion, rendue_il_y_a=None):
    import time

    from app.core import database

    class _Fiche:
        pass

    fiche = _Fiche()
    fiche.driver_connection = connexion
    fiche.info = {} if rendue_il_y_a is None else {
        database._RENDUE_A: time.monotonic() - rendue_il_y_a
    }
    return fiche


class TestVerificationALaSortie:
    """
    pool_pre_ping coutait QUATRE allers-retours a chaque emprunt (BEGIN, « ; »
    prepare puis execute, ROLLBACK) : 8 par recherche, a 83 ms chacun. La
    verification qui le remplace n'en coute aucun pour une connexion recente,
    un seul pour une connexion inactive, et ecarte une connexion fermee sans
    rien envoyer.
    """

    @staticmethod
    async def _verifier(fiche):
        from sqlalchemy.util import greenlet_spawn

        from app.core import database

        await greenlet_spawn(database._verifier_a_la_sortie, None, fiche, None)

    @pytest.mark.asyncio
    async def test_une_connexion_fermee_est_ecartee_sans_rien_envoyer(self):
        from sqlalchemy import exc

        connexion = _ConnexionFactice(fermee=True)

        with pytest.raises(exc.DisconnectionError):
            await self._verifier(_fiche(connexion, rendue_il_y_a=1000))
        assert connexion.envois == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rendue_il_y_a", [None, 1.0], ids=["neuve", "recente"])
    async def test_une_connexion_neuve_ou_recente_ne_coute_rien(self, rendue_il_y_a):
        connexion = _ConnexionFactice()

        await self._verifier(_fiche(connexion, rendue_il_y_a))

        assert connexion.envois == []

    @pytest.mark.asyncio
    async def test_une_connexion_inactive_est_sondee_une_fois(self):
        """Protocole simple, sans transaction ni preparation : un aller-retour, borne."""
        from app.core import database

        connexion = _ConnexionFactice()

        await self._verifier(_fiche(connexion, database.PING_APRES_INACTIVITE_S + 1))

        assert connexion.envois == [("SELECT 1", database.DELAI_PING_S)]

    @pytest.mark.asyncio
    async def test_un_sondage_en_echec_fait_remplacer_la_connexion(self):
        from sqlalchemy import exc

        from app.core import database

        connexion = _ConnexionFactice(panne=ConnectionResetError("connexion perdue"))

        with pytest.raises(exc.DisconnectionError):
            await self._verifier(_fiche(connexion, database.PING_APRES_INACTIVITE_S + 1))

    def test_le_moteur_de_l_application_est_ainsi_regle(self):
        from sqlalchemy import event

        from app.core import database, mesure_sql

        assert database.engine.pool._pre_ping is False
        assert database.engine.pool._pool.use_lifo is True
        assert database.engine.pool._recycle == -1
        for nom, ecouteur in (
            ("checkout", database._verifier_a_la_sortie),
            ("checkin", database._noter_le_retour),
            ("before_cursor_execute", mesure_sql._avant_instruction),
        ):
            assert event.contains(database.engine.sync_engine, nom, ecouteur), nom

    @pytest.mark.asyncio
    async def test_une_connexion_coupee_par_le_serveur_est_remplacee(self, pg_engine):
        """
        Ce que fait Neon a sa mise en veille : il ferme les connexions du pool.
        La requete suivante doit repondre, sur une connexion neuve, sans erreur.
        """
        import asyncio

        import asyncpg
        from sqlalchemy import text

        from app.core import database
        from tests.conftest import TEST_DATABASE_URL

        database.brancher_verification_des_connexions(pg_engine.sync_engine)
        async with pg_engine.connect() as connexion:
            pid = (await connexion.execute(text("SELECT pg_backend_pid()"))).scalar()

        tiers = await asyncpg.connect(TEST_DATABASE_URL.replace("+asyncpg", ""))
        try:
            assert await tiers.fetchval("SELECT pg_terminate_backend($1)", pid)
        finally:
            await tiers.close()
        await asyncio.sleep(0.2)  # la fin de flux arrive a la boucle

        async with pg_engine.connect() as connexion:
            nouveau = (await connexion.execute(text("SELECT pg_backend_pid()"))).scalar()

        assert nouveau != pid
