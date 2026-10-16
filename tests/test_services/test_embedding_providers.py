"""
Tests du contrat de fournisseur d'embeddings.

Le service delegue la production des vecteurs a un fournisseur interchangeable
et garde tout le reste. Ces tests verrouillent les quatre garanties qui
rendent l'echange sur :

- deux fournisseurs ne se servent JAMAIS leurs vecteurs par le cache, meme a
  dimension egale — c'est le cas que la garde de dimension ne voit pas ;
- une dimension que le fournisseur ne sait pas produire est refusee a la
  construction, pas au premier appel apres trois tentatives ;
- ce qu'un nouvel essai reproduirait n'est jamais rejoue, et une question
  n'a droit qu'a un essai ;
- le service normalise et verifie, quel que soit ce que rend le fournisseur.

Aucun vrai modele, aucun appel reseau : les fournisseurs sont des doublures.
"""

import numpy as np
import pytest

from app.services import embedding_service as module
from app.services.embedding_service import (
    EmbeddingDefinitiveError,
    EmbeddingService,
    EmbeddingServiceError,
    GeminiProvider,
)


class _Fournisseur:
    """Fournisseur factice, configurable, qui compte ses appels."""

    label = "Fournisseur de test"
    model = "modele-de-test"
    retryable = True
    inter_batch_delay_s = 0.0

    def __init__(self, name="test", native_dim=None, echec=None, norme=7.0):
        self.name = name
        self.native_dim = native_dim or EmbeddingService.EMBEDDING_DIM
        self.echec = echec
        self.norme = norme
        self.appels = 0

    @property
    def fingerprint(self):
        return f"{self.name}|{self.model}"

    def embed(self, texts, task_type, dim):
        self.appels += 1
        if self.echec:
            raise self.echec
        rng = np.random.default_rng(3)
        # Vecteurs NON unitaires : c'est au service de normaliser.
        return [(rng.random(dim) * self.norme).astype(np.float32) for _ in texts]


def _service(fournisseur):
    return EmbeddingService(use_cache=False, provider=fournisseur)


class TestEmpreinteDansLaCle:

    def test_deux_fournisseurs_a_dimension_egale_ont_des_cles_distinctes(self):
        """
        Le cas que la garde de dimension ne voit pas : meme texte, meme
        dimension, meme tache, deux modeles. Sans l'empreinte dans la cle, le
        vecteur de l'un serait resservi a l'autre pendant sept jours.
        """
        a = _service(_Fournisseur(name="a"))
        b = _service(_Fournisseur(name="b"))

        cle_a = a._cache_key("Article premier", EmbeddingService.TASK_DOCUMENT)
        cle_b = b._cache_key("Article premier", EmbeddingService.TASK_DOCUMENT)

        assert cle_a != cle_b

    def test_meme_fournisseur_meme_cle(self):
        a = _service(_Fournisseur(name="a"))
        a2 = _service(_Fournisseur(name="a"))

        assert a._cache_key("x", EmbeddingService.TASK_DOCUMENT) == a2._cache_key(
            "x", EmbeddingService.TASK_DOCUMENT
        )

    def test_la_cle_lit_l_empreinte_de_l_instance(self):
        """Pas une constante de classe figee a l'import."""
        f = _Fournisseur(name="avant")
        s = _service(f)
        avant = s._cache_key("x", EmbeddingService.TASK_DOCUMENT)

        f.name = "apres"

        assert s._cache_key("x", EmbeddingService.TASK_DOCUMENT) != avant

    def test_empreinte_gemini_porte_le_modele(self, monkeypatch):
        monkeypatch.setattr(module.genai, "Client", lambda *a, **k: object())

        assert GeminiProvider(api_key="factice").fingerprint == (
            f"gemini|{module.settings.GEMINI_EMBEDDING_MODEL}"
        )


class TestDimensionNative:

    def test_dimension_trop_grande_refusee_a_la_construction(self):
        """
        Sans ce controle, l'erreur n'apparait qu'au premier appel, apres trois
        tentatives, et la recherche hybride l'avale sans bruit.
        """
        trop_petit = _Fournisseur(native_dim=EmbeddingService.EMBEDDING_DIM - 1)

        with pytest.raises(EmbeddingServiceError, match="dimension native"):
            _service(trop_petit)

    def test_dimension_egale_a_la_native_acceptee(self):
        _service(_Fournisseur(native_dim=EmbeddingService.EMBEDDING_DIM))

    def test_dimension_hors_des_dimensions_permises_refusee(self):
        """
        Voyage n'accepte que 256, 512, 1024 et 2048 : une dimension sous la
        native mais hors liste n'echouerait qu'au premier appel, en HTTP 400,
        que la recherche hybride avalerait.
        """
        f = _Fournisseur(native_dim=2048)
        f.dimensions_permises = (256, 512, 2048)

        with pytest.raises(EmbeddingServiceError, match="permises"):
            _service(f)

    def test_dimension_permise_acceptee(self):
        f = _Fournisseur(native_dim=2048)
        f.dimensions_permises = (EmbeddingService.EMBEDDING_DIM,)

        _service(f)


class TestLimiteDeTexte:

    def test_la_limite_du_fournisseur_s_impose_a_l_instance(self):
        f = _Fournisseur()
        f.max_caracteres = 50

        s = _service(f)

        assert s.MAX_TEXT_LENGTH == 50
        with pytest.raises(ValueError, match="trop long"):
            s.generate_embedding("x" * 51)

    def test_sans_limite_declaree_le_defaut_du_service(self):
        assert _service(_Fournisseur()).MAX_TEXT_LENGTH == EmbeddingService.MAX_TEXT_LENGTH

    def test_gemini_garde_sa_limite(self):
        """2 048 jetons de contexte chez gemini-embedding-001."""
        assert GeminiProvider.max_caracteres == 10_000


class TestReprises:

    def test_fournisseur_non_rejouable_echoue_au_premier_appel(self, monkeypatch):
        """
        Un modele local qui echoue echouera pareil a la tentative suivante :
        rejouer ne ferait que dormir avant d'echouer quand meme.
        """
        dodos = []
        monkeypatch.setattr(module.time, "sleep", lambda s: dodos.append(s))
        f = _Fournisseur(echec=RuntimeError("modele indisponible"))
        f.retryable = False

        with pytest.raises(EmbeddingServiceError):
            _service(f).generate_embedding("Article premier")

        assert f.appels == 1
        assert dodos == []

    def test_fournisseur_rejouable_est_rejoue(self, monkeypatch):
        monkeypatch.setattr(module.time, "sleep", lambda s: None)
        f = _Fournisseur(echec=RuntimeError("503 passager"))

        with pytest.raises(EmbeddingServiceError):
            _service(f).generate_embedding("Article premier")

        assert f.appels == EmbeddingService.MAX_RETRIES

    def test_une_erreur_definitive_n_est_pas_rejouee(self, monkeypatch):
        """Une cle refusee le sera encore : la relancer telle quelle."""
        dodos = []
        monkeypatch.setattr(module.time, "sleep", lambda s: dodos.append(s))
        f = _Fournisseur(echec=EmbeddingDefinitiveError("HTTP 401 : cle refusee"))

        with pytest.raises(EmbeddingDefinitiveError):
            _service(f).generate_embedding("Article premier")
        with pytest.raises(EmbeddingDefinitiveError):
            _service(f).generate_batch_embeddings(["a", "b"])

        assert f.appels == 2
        assert dodos == []

    def test_une_question_n_a_qu_un_essai(self, monkeypatch):
        """
        Pendant ce temps, l'utilisateur attend : un echec passager ne lui
        coute que le volet semantique de sa recherche.
        """
        dodos = []
        monkeypatch.setattr(module.time, "sleep", lambda s: dodos.append(s))
        f = _Fournisseur(echec=RuntimeError("503 passager"))

        with pytest.raises(EmbeddingServiceError):
            _service(f).generate_embedding("Une question", task_type=EmbeddingService.TASK_QUERY)

        assert f.appels == 1
        assert dodos == []

    def test_pas_de_pause_entre_lots_si_le_fournisseur_n_en_veut_pas(self, monkeypatch):
        dodos = []
        monkeypatch.setattr(module.time, "sleep", lambda s: dodos.append(s))
        f = _Fournisseur()  # inter_batch_delay_s = 0.0

        _service(f).generate_batch_embeddings(["a", "b", "c"], batch_size=1)

        assert dodos == []


class TestLeServiceGardeSesControles:

    def test_le_service_normalise_quel_que_soit_le_fournisseur(self):
        vecteur = _service(_Fournisseur(norme=9.0)).generate_embedding("Article premier")

        assert vecteur.shape == (EmbeddingService.EMBEDDING_DIM,)
        assert abs(float(np.linalg.norm(vecteur)) - 1.0) < 1e-5

    def test_reponse_incomplete_detectee_par_le_service(self):
        class _Court(_Fournisseur):
            def embed(self, texts, task_type, dim):
                return super().embed(texts[:-1], task_type, dim)

        with pytest.raises(EmbeddingServiceError, match="incomplète"):
            _service(_Court()).generate_batch_embeddings(["a", "b", "c"])

    def test_le_fournisseur_injecte_est_celui_qui_sert(self):
        f = _Fournisseur()
        s = _service(f)

        s.generate_embedding("Article premier")

        assert s.provider is f
        assert f.appels == 1

    def test_health_check_decrit_le_fournisseur(self):
        sante = _service(_Fournisseur()).health_check()

        assert sante["provider"] == "Fournisseur de test"
        assert sante["model"] == "modele-de-test"


class TestGeminiRefus:
    """
    Une cle Gemini invalide arrive en 400 (API_KEY_INVALID) : elle doit
    arreter un traitement par lots, pas faire sauter chaque lot apres trois
    essais.
    """

    @staticmethod
    def _fournisseur(monkeypatch, erreur):
        class _Modeles:
            def embed_content(self, **kwargs):
                raise erreur

        class _Client:
            def __init__(self, *a, **k):
                self.models = _Modeles()

        monkeypatch.setattr(module.genai, "Client", _Client)
        return GeminiProvider(api_key="factice")

    def test_une_cle_invalide_vise_la_configuration(self, monkeypatch):
        from google.genai import errors

        erreur = errors.ClientError(400, {"error": {
            "code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT",
            "details": [{"reason": "API_KEY_INVALID"}],
        }})

        with pytest.raises(EmbeddingDefinitiveError) as refus:
            self._fournisseur(monkeypatch, erreur).embed(["a"], EmbeddingService.TASK_DOCUMENT, 1024)
        assert refus.value.code == 401
        assert refus.value.vise_la_configuration

    def test_un_429_reste_rejouable(self, monkeypatch):
        """Le quota par minute se rejoue ; _is_daily_quota_error lit son message."""
        from google.genai import errors

        erreur = errors.ClientError(429, {"error": {
            "code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED",
        }})

        with pytest.raises(errors.ClientError):
            self._fournisseur(monkeypatch, erreur).embed(["a"], EmbeddingService.TASK_DOCUMENT, 1024)
