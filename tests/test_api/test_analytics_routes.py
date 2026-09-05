"""
Les statistiques doivent etre mesurees, et reservees aux administrateurs.

Deux defauts corriges ici :

1. `GET /analytics/search` et `/analytics/usage` renvoyaient des constantes
   codees en dur — 350 recherches, 150 ms, 1150 appels, 25 utilisateurs actifs —
   accompagnees d'un champ `note: "Mock data"` que le front ne lisait pas. Le
   tableau de bord admin les affichait comme des mesures.
2. Les quatre routes d'analytics repondaient 200 SANS jeton, exposant la
   composition du corpus et les dernieres lois traitees. Le front envoyait
   pourtant un jeton.

Usage:
    pytest tests/test_api/test_analytics_routes.py -v
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import text


ROUTES = ("overview", "laws", "search", "usage")


class TestAuthentification:
    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.asyncio
    async def test_sans_jeton_c_est_401(self, client: AsyncClient, route):
        assert (await client.get(f"/api/v1/analytics/{route}")).status_code == 401

    @pytest.mark.asyncio
    async def test_avec_un_jeton_admin_c_est_200(self, admin_client: AsyncClient, db_session):
        for route in ROUTES:
            reponse = await admin_client.get(f"/api/v1/analytics/{route}")
            assert reponse.status_code == 200, f"/{route} rend {reponse.status_code}"


class TestPlusDeDonneesInventees:
    @pytest.mark.asyncio
    async def test_aucun_champ_note_mock_data(self, admin_client: AsyncClient, db_session):
        """
        Le champ `note: "Mock data - integrate with search logs in production"`
        etait la seule trace que les chiffres etaient faux, et personne ne le
        lisait. Sa disparition est ce qui prouve que les chiffres ne le sont
        plus.
        """
        for route in ("search", "usage"):
            corps = (await admin_client.get(f"/api/v1/analytics/{route}")).json()
            assert "note" not in corps, f"/{route} porte encore un champ note"

    @pytest.mark.asyncio
    async def test_une_base_vide_rend_zero_et_non_350(
        self, admin_client: AsyncClient, db_session
    ):
        """Sans aucune recherche enregistree, le total est 0. Il valait 350."""
        corps = (await admin_client.get("/api/v1/analytics/search")).json()

        assert corps["total_searches"] == 0
        assert corps["modes_usage"] == {}
        assert corps["median_response_time_ms"] == 0

    @pytest.mark.asyncio
    async def test_les_recherches_enregistrees_sont_comptees(
        self, admin_client: AsyncClient, db_session
    ):
        await db_session.execute(text("""
            INSERT INTO search_events (query, mode, results_count, duration_ms, cached)
            VALUES ('nomination', 'text', 5, 120, false),
                   ('nomination', 'text', 5, 100, true),
                   ('code minier', 'hybrid', 3, 300, false),
                   ('xyzintrouvable', 'text', 0, 40, false)
        """))
        await db_session.commit()

        corps = (await admin_client.get("/api/v1/analytics/search")).json()

        assert corps["total_searches"] == 4
        assert corps["modes_usage"] == {"text": 3, "hybrid": 1}
        # Mediane des NON mises en cache : 40, 120, 300 -> 120
        assert corps["median_response_time_ms"] == 120
        assert corps["cache_hit_rate_percent"] == 25.0

    @pytest.mark.asyncio
    async def test_les_requetes_sans_resultat_remontent(
        self, admin_client: AsyncClient, db_session
    ):
        """
        C'est la statistique la plus utile du lot : ce que les gens cherchent
        et que le corpus ne contient pas.
        """
        await db_session.execute(text("""
            INSERT INTO search_events (query, mode, results_count, duration_ms)
            VALUES ('code penal', 'text', 0, 30), ('code penal', 'text', 0, 25),
                   ('nomination', 'text', 5, 100)
        """))
        await db_session.commit()

        corps = (await admin_client.get("/api/v1/analytics/search")).json()

        assert corps["queries_without_results"] == [{"query": "code penal", "count": 2}]

    @pytest.mark.asyncio
    async def test_la_fenetre_d_observation_est_respectee(
        self, admin_client: AsyncClient, db_session
    ):
        await db_session.execute(text("""
            INSERT INTO search_events (query, mode, results_count, duration_ms, created_at)
            VALUES ('ancienne', 'text', 1, 10, now() - interval '30 days'),
                   ('recente',  'text', 1, 10, now())
        """))
        await db_session.commit()

        assert (await admin_client.get("/api/v1/analytics/search?days=7")).json()["total_searches"] == 1
        assert (await admin_client.get("/api/v1/analytics/search?days=60")).json()["total_searches"] == 2

    @pytest.mark.asyncio
    async def test_usage_compte_de_vraies_conversations(
        self, admin_client: AsyncClient, db_session
    ):
        from app.models.conversation import Conversation, Message

        conversation = Conversation(session_id="s-1", persona="citoyen", language="fr")
        db_session.add(conversation)
        await db_session.flush()
        db_session.add_all([
            Message(conversation_id=conversation.id, role="user", content="Une question ?"),
            Message(conversation_id=conversation.id, role="assistant", content="Une reponse.",
                    retrieval_time_ms=100, generation_time_ms=400),
        ])
        await db_session.commit()

        corps = (await admin_client.get("/api/v1/analytics/usage")).json()

        assert corps["conversations"] == 1
        assert corps["questions_asked"] == 1
        assert corps["answers_generated"] == 1
        assert corps["median_answer_time_ms"] == 500
        assert corps["personas_usage"] == {"citoyen": 1}
