"""
Le service de generation doit nommer ses pannes.

Trois defauts corriges ici, tous constates en production :

1. La sonde de sante interrogeait le modele avec `max_output_tokens=10`. Les
   modeles a raisonnement consomment ce budget en reflexion interne, donc
   `response.text` revenait vide et la sonde repondait « unhealthy » sur une API
   parfaitement joignable. Mesure sur trois modeles : les trois echouaient a
   10 jetons, les trois repondaient a 512.
2. `GeminiServiceError` etait levee en trois endroits et attrapee NULLE PART.
   Une saturation passagere (503) et un quota epuise (429) remontaient
   identiquement en HTTP 500 avec le JSON brut du fournisseur dans `detail`.
3. La reflexion interne du modele fuyait dans la reponse : une question de
   suivi rendait un texte commencant par « *Self-Correction during drafting:* ».

Usage:
    pytest tests/test_services/test_gemini_failure_modes.py -v
"""

from types import SimpleNamespace

import pytest

from app.services.gemini_service import (
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiServiceError,
    _finish_reason,
    _is_overloaded,
    _is_quota_exhausted,
    _visible_text,
    retry_after_seconds,
)


def _reponse(parts, finish="STOP"):
    """Reponse minimale a la forme de celle du SDK google-genai."""
    contenu = SimpleNamespace(parts=[SimpleNamespace(**p) for p in parts])
    candidat = SimpleNamespace(content=contenu, finish_reason=SimpleNamespace(name=finish))
    texte = "".join(p.get("text") or "" for p in parts)
    return SimpleNamespace(candidates=[candidat], text=texte)


class TestClassementDesPannes:
    """Une saturation et un quota epuise n'appellent pas la meme reponse."""

    SATURATION = (
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
        "currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
    )
    QUOTA = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
        "your current quota', 'status': 'RESOURCE_EXHAUSTED', 'details': "
        "[{'retryDelay': '42s'}]}}"
    )

    def test_la_saturation_est_reconnue(self):
        assert _is_overloaded(Exception(self.SATURATION)) is True

    def test_le_quota_n_est_pas_une_saturation(self):
        """
        Les confondre faisait retenter en boucle sur un quota deja epuise,
        brulant des appels pour rien.
        """
        assert _is_overloaded(Exception(self.QUOTA)) is False
        assert _is_quota_exhausted(Exception(self.QUOTA)) is True

    def test_la_saturation_n_est_pas_un_quota(self):
        assert _is_quota_exhausted(Exception(self.SATURATION)) is False

    def test_une_erreur_ordinaire_n_est_ni_l_un_ni_l_autre(self):
        banale = Exception("400 INVALID_ARGUMENT: model not found")
        assert _is_overloaded(banale) is False
        assert _is_quota_exhausted(banale) is False

    def test_le_delai_conseille_est_extrait(self):
        assert retry_after_seconds(Exception(self.QUOTA)) == 42

    def test_le_delai_a_un_defaut_quand_le_fournisseur_n_en_donne_pas(self):
        assert retry_after_seconds(Exception("429 RESOURCE_EXHAUSTED"), defaut=90) == 90

    def test_les_deux_types_restent_attrapables_comme_avant(self):
        """
        Sous-classes de GeminiServiceError : tout appelant existant continue de
        fonctionner sans modification.
        """
        assert issubclass(GeminiQuotaError, GeminiServiceError)
        assert issubclass(GeminiOverloadedError, GeminiServiceError)


class TestTexteVisible:
    """La reflexion interne du modele ne doit jamais atteindre l'utilisateur."""

    def test_la_reflexion_est_retiree(self):
        reponse = _reponse([
            {"text": "*Self-Correction during drafting:* the user said…", "thought": True},
            {"text": "Le permis de recherche couvre au plus 500 km².", "thought": False},
        ])

        assert _visible_text(reponse) == "Le permis de recherche couvre au plus 500 km²."

    def test_une_reponse_sans_reflexion_passe_intacte(self):
        reponse = _reponse([{"text": "Reponse simple.", "thought": False}])

        assert _visible_text(reponse) == "Reponse simple."

    def test_plusieurs_morceaux_visibles_sont_recolles(self):
        reponse = _reponse([
            {"text": "reflexion", "thought": True},
            {"text": "Premiere partie. ", "thought": False},
            {"text": "Seconde partie.", "thought": False},
        ])

        assert _visible_text(reponse) == "Premiere partie. Seconde partie."

    def test_repli_quand_il_n_y_a_que_de_la_reflexion(self):
        """
        Mieux vaut une reponse bavarde que pas de reponse : si TOUT est marque
        comme reflexion, on rend quand meme quelque chose plutot que du vide.
        """
        reponse = _reponse([{"text": "monologue interne", "thought": True}])

        assert _visible_text(reponse) == "monologue interne"

    def test_repli_quand_la_structure_en_parts_est_absente(self):
        sans_parts = SimpleNamespace(candidates=[], text="texte de secours")

        assert _visible_text(sans_parts) == "texte de secours"


class TestRaisonDArret:
    def test_la_raison_est_lisible(self):
        assert _finish_reason(_reponse([{"text": "x"}], finish="MAX_TOKENS")) == "MAX_TOKENS"

    def test_absence_de_candidat_ne_leve_pas(self):
        assert _finish_reason(SimpleNamespace(candidates=[])) == "UNKNOWN"


class TestSondeDeSante:
    """
    La sonde ne doit pas confondre « budget epuise » et « service en panne ».
    C'est le defaut qui faisait afficher `llm: unhealthy` en permanence.
    """

    @pytest.mark.asyncio
    async def test_un_budget_epuise_ne_veut_pas_dire_en_panne(self, monkeypatch):
        from app.services.gemini_service import GeminiService

        service = GeminiService.__new__(GeminiService)
        service.model_name = "modele-test"
        service.client = SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **kw: _reponse([{"text": "", "thought": True}],
                                                   finish="MAX_TOKENS")
        ))

        rapport = await service.health_check()

        assert rapport["status"] == "healthy"
        assert "joignable" in rapport["note"]

    @pytest.mark.asyncio
    async def test_une_saturation_est_degradee_pas_en_panne(self):
        from app.services.gemini_service import GeminiService

        def _sature(**kw):
            raise Exception(TestClassementDesPannes.SATURATION)

        service = GeminiService.__new__(GeminiService)
        service.model_name = "modele-test"
        service.client = SimpleNamespace(models=SimpleNamespace(generate_content=_sature))

        rapport = await service.health_check()

        assert rapport["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_une_vraie_panne_reste_une_panne(self):
        from app.services.gemini_service import GeminiService

        def _casse(**kw):
            raise Exception("400 INVALID_ARGUMENT")

        service = GeminiService.__new__(GeminiService)
        service.model_name = "modele-test"
        service.client = SimpleNamespace(models=SimpleNamespace(generate_content=_casse))

        assert (await service.health_check())["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_une_reponse_pleine_est_saine(self):
        from app.services.gemini_service import GeminiService

        service = GeminiService.__new__(GeminiService)
        service.model_name = "modele-test"
        service.client = SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **kw: _reponse([{"text": "OK", "thought": False}])
        ))

        assert (await service.health_check())["status"] == "healthy"
