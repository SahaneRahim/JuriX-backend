"""
Compteur des instructions SQL par requete (app/core/mesure_sql.py) et sa
metrique `db` dans l'en-tete Server-Timing.

Les allers-retours vers la base ont ete la premiere cause de lenteur mesuree,
et ne se comptaient qu'en local derriere un proxy TCP. Ces tests verifient ce
que l'en-tete permet de lire en production : le nombre d'instructions et de
transactions de CETTE requete, ni plus (une autre requete, une tache qui lui
survit) ni moins (le volet lance dans une tache).
"""

import asyncio
import re

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import mesure_sql


@pytest.fixture
async def moteur_compte(pg_engine):
    """Le moteur de test, avec le compteur pose comme sur celui de l'application."""
    mesure_sql.brancher(pg_engine.sync_engine)
    try:
        yield pg_engine
    finally:
        mesure_sql.debrancher(pg_engine.sync_engine)


def _db(en_tete: str) -> dict:
    """{n, tx, ...} et dur, lus dans la metrique `db` d'un Server-Timing."""
    metrique = re.search(r'db;desc="([^"]*)";dur=([\d.]+)', en_tete)
    assert metrique, en_tete
    valeurs = dict(paire.split("=") for paire in metrique.group(1).split())
    valeurs = {cle: int(v) for cle, v in valeurs.items()}
    valeurs["dur"] = float(metrique.group(2))
    return valeurs


class TestCompteur:

    @pytest.mark.asyncio
    async def test_instructions_et_transactions_de_la_requete(self, moteur_compte):
        mesure, jeton = mesure_sql.commencer()
        try:
            async with AsyncSession(moteur_compte) as session:
                await session.execute(text("SELECT 1"))
                await session.execute(text("SELECT 2"))
                await session.commit()
                await session.execute(text("SELECT 3"))
        finally:
            mesure_sql.terminer(jeton)

        assert (mesure.instructions, mesure.transactions) == (3, 2)
        assert mesure.duree_s > 0
        assert mesure_sql.courante() is None

    @pytest.mark.asyncio
    async def test_hors_requete_rien_n_est_compte(self, moteur_compte):
        """Les boucles de fond (purge des caches) n'ont pas de mesure : rien ne casse."""
        async with AsyncSession(moteur_compte) as session:
            await session.execute(text("SELECT 1"))

        assert mesure_sql.courante() is None

    @pytest.mark.asyncio
    async def test_une_instruction_refusee_compte_et_ne_laisse_rien(self, moteur_compte):
        mesure, jeton = mesure_sql.commencer()
        try:
            async with moteur_compte.connect() as connexion:
                with pytest.raises(Exception):
                    await connexion.execute(text("SELECT * FROM table_qui_n_existe_pas"))
                info = (await connexion.get_raw_connection()).info
        finally:
            mesure_sql.terminer(jeton)

        assert mesure.instructions == 1
        assert mesure_sql._DEBUT not in info

    @pytest.mark.asyncio
    async def test_une_tache_lancee_par_la_requete_compte_pour_elle(self, moteur_compte):
        """Le volet semantique d'une recherche tourne dans une tache, sur sa connexion."""

        async def _volet():
            async with AsyncSession(moteur_compte) as session:
                await session.execute(text("SELECT 1"))

        async def _survivante():
            mesure_sql.ne_plus_compter()
            async with AsyncSession(moteur_compte) as session:
                await session.execute(text("SELECT 1"))

        mesure, jeton = mesure_sql.commencer()
        try:
            await asyncio.gather(asyncio.create_task(_volet()), asyncio.create_task(_survivante()))
            # ne_plus_compter() n'a touche que la copie de contexte de sa tache.
            assert mesure_sql.courante() is mesure
        finally:
            mesure_sql.terminer(jeton)

        assert (mesure.instructions, mesure.transactions) == (1, 1)

    @pytest.mark.asyncio
    async def test_deux_requetes_simultanees_ont_chacune_leur_compte(self, moteur_compte):
        async def _requete(n):
            mesure, jeton = mesure_sql.commencer()
            try:
                async with AsyncSession(moteur_compte) as session:
                    for _ in range(n):
                        await session.execute(text("SELECT pg_sleep(0.01)"))
                return mesure
            finally:
                mesure_sql.terminer(jeton)

        une, trois = await asyncio.gather(_requete(1), _requete(3))

        assert (une.instructions, trois.instructions) == (1, 3)

    @pytest.mark.asyncio
    async def test_une_connexion_ouverte_pendant_la_requete_est_comptee(self, moteur_compte):
        """Une connexion neuve, c'est une vingtaine d'allers-retours avec TLS."""
        await moteur_compte.dispose()
        mesure, jeton = mesure_sql.commencer()
        try:
            async with AsyncSession(moteur_compte) as session:
                await session.execute(text("SELECT 1"))
        finally:
            mesure_sql.terminer(jeton)

        assert mesure.connexions == 1
        assert 'cx=1' in mesure.en_tete()

    def test_la_mesure_ne_garde_que_des_nombres(self):
        """Ni texte SQL ni parametre : une question d'utilisateur n'a rien a y faire."""
        mesure = mesure_sql.MesureSQL()

        assert all(isinstance(v, (int, float)) for v in vars(mesure).values())
        assert mesure.en_tete() == 'db;desc="n=0 tx=0";dur=0.0'

    @pytest.mark.asyncio
    async def test_brancher_deux_fois_ne_compte_pas_double(self, moteur_compte):
        from sqlalchemy import event

        mesure_sql.brancher(moteur_compte.sync_engine)
        mesure, jeton = mesure_sql.commencer()
        try:
            async with AsyncSession(moteur_compte) as session:
                await session.execute(text("SELECT 1"))
        finally:
            mesure_sql.terminer(jeton)

        assert (mesure.instructions, mesure.transactions) == (1, 1)
        mesure_sql.debrancher(moteur_compte.sync_engine)
        assert not event.contains(
            moteur_compte.sync_engine, "before_cursor_execute", mesure_sql._avant_instruction
        )


class TestSondage:

    @pytest.mark.asyncio
    async def test_un_sondage_de_connexion_inactive_est_compte(self):
        """database._verifier_a_la_sortie : un SELECT 1 de plus pour la requete."""
        import time

        from sqlalchemy.util import greenlet_spawn

        from app.core import database

        class _Connexion:
            def is_closed(self):
                return False

            async def execute(self, sql, timeout=None):
                return "SELECT 1"

        class _Fiche:
            driver_connection = _Connexion()
            info = {database._RENDUE_A: time.monotonic() - database.PING_APRES_INACTIVITE_S - 1}

        mesure, jeton = mesure_sql.commencer()
        try:
            await greenlet_spawn(database._verifier_a_la_sortie, None, _Fiche(), None)
        finally:
            mesure_sql.terminer(jeton)

        assert mesure.sondages == 1
        assert "ping=1" in mesure.en_tete()


class TestEnTete:

    @pytest.mark.asyncio
    async def test_une_route_sans_base_annonce_zero_instruction(self):
        from httpx import ASGITransport, AsyncClient

        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            reponse = await client.get("/health")

        en_tete = reponse.headers["server-timing"]
        assert re.match(r"app;dur=[\d.]+, db;", en_tete)
        assert _db(en_tete) == {"n": 0, "tx": 0, "dur": 0.0}

    @pytest.mark.asyncio
    async def test_une_route_qui_lit_la_base_annonce_ses_instructions(
        self, client, moteur_compte
    ):
        """
        GET /laws : une instruction (lois, categorie et nombre d'articles en
        une seule), dans une transaction.
        """
        reponse = await client.get("/api/v1/laws/?limit=5&content_max_chars=0")

        assert reponse.status_code == 200
        db = _db(reponse.headers["server-timing"])
        assert (db["n"], db["tx"]) == (1, 1)
        assert db["dur"] > 0

    @pytest.mark.asyncio
    async def test_chaque_requete_a_son_propre_compte(self, client, moteur_compte):
        premiere = await client.get("/api/v1/laws/?limit=5&content_max_chars=0")
        seconde = await client.get("/api/v1/laws/?limit=5&content_max_chars=0")

        assert _db(premiere.headers["server-timing"])["n"] == 1
        assert _db(seconde.headers["server-timing"])["n"] == 1

    @pytest.mark.asyncio
    async def test_l_en_tete_reste_lisible_par_le_front(self):
        """CORS l'expose : sans cela, fetch() ne le verrait pas."""
        from httpx import ASGITransport, AsyncClient

        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            reponse = await client.get(
                "/health", headers={"Origin": "https://juri-x-frontend.vercel.app"}
            )

        assert "server-timing" in reponse.headers["access-control-expose-headers"].lower()
        assert "server-timing" in reponse.headers
