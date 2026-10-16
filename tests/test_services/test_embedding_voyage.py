"""
Tests du fournisseur Voyage AI.

Deux familles :

- Sur DOUBLURE HTTP (httpx.MockTransport pose sur
  `embedding_service.transport_http`) : le corps exact de chaque requete, les
  lots, les delais, la lecture des erreurs, l'empreinte, l'echeance des
  questions. Aucun appel reseau : la garde de conftest refuserait le moindre.
- Sur le VRAI Voyage, marques `voyage_live` (lances par
  `pytest -m voyage_live` seulement) : la preuve que les questions de
  production tombent dans l'espace des vecteurs deja calcules du corpus
  (data/mesure_voyage). Ils sautent si ces fichiers manquent.

Chaque piege verifie ici casse EN SILENCE : un vecteur de 1024, unitaire,
d'apparence saine, mais dans un autre espace.
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.core.config import settings
from app.services import embedding_service as module
from app.services.embedding_service import (
    EmbeddingDefinitiveError,
    EmbeddingService,
    EmbeddingServiceError,
    GeminiProvider,
    SuiviSemantique,
    VoyageProvider,
    empreinte_voyage,
    motif_court,
    suivi_semantique,
)

QUERY = EmbeddingService.TASK_QUERY
DOCUMENT = EmbeddingService.TASK_DOCUMENT
DIM = EmbeddingService.EMBEDDING_DIM
CLE = "cle-voyage-de-test"


# ==================== DOUBLURE HTTP ====================


class FauxVoyage:
    """
    Rend un vecteur par texte recu, a la dimension demandee, et garde chaque
    requete. `reponse(requete, corps)` peut etre remplacee pour simuler un
    refus ; elle rend alors un httpx.Response.
    """

    def __init__(self, reponse=None, melanger=False, dim=None):
        self.requetes = []
        self.corps = []
        self.reponse = reponse
        self.melanger = melanger
        self.dim = dim

    def __call__(self, requete: httpx.Request) -> httpx.Response:
        corps = json.loads(requete.content)
        self.requetes.append(requete)
        self.corps.append(corps)
        if self.reponse is not None:
            rendu = self.reponse(requete, corps)
            if rendu is not None:
                return rendu
        dim = self.dim or corps.get("output_dimension") or 1024
        donnees = []
        for i, texte in enumerate(corps["input"]):
            rng = np.random.default_rng(len(texte) + 7 * i)
            # Vecteurs NON unitaires : c'est au service de normaliser.
            donnees.append({"object": "embedding", "index": i,
                            "embedding": (rng.random(dim) * 3.0).tolist()})
        if self.melanger:
            donnees.reverse()
        return httpx.Response(200, json={
            "object": "list", "data": donnees, "model": corps["model"],
            "usage": {"total_tokens": sum(len(t) // 3 for t in corps["input"])},
        })


@pytest.fixture
def faux(monkeypatch):
    """Pose une doublure Voyage, et la rend pour inspection."""
    def _poser(**options):
        doublure = FauxVoyage(**options)
        monkeypatch.setattr(module, "transport_http", httpx.MockTransport(doublure))
        return doublure
    return _poser


def _refus(code, detail):
    return lambda requete, corps: httpx.Response(code, json={"detail": detail})


# ==================== CORPS DE LA REQUETE ====================


class TestCorps:
    """
    Le corps doit produire l'espace des vecteurs du corpus, calcules par
    data/mesure_voyage/vectoriser_voyage.py avec les defauts de l'API : 1024
    dimensions, sortie float, troncature cote serveur.
    """

    def test_corps_exact_d_une_question(self, faux):
        doublure = faux()
        VoyageProvider(api_key=CLE).embed(["Quelle peine pour le vol ?"], QUERY, DIM)

        (requete,) = doublure.requetes
        assert str(requete.url) == module.VOYAGE_URL == "https://api.voyageai.com/v1/embeddings"
        assert requete.headers["authorization"] == f"Bearer {CLE}"
        assert doublure.corps == [{
            "input": ["Quelle peine pour le vol ?"],
            "model": "voyage-4-large",
            "input_type": "query",
            "output_dimension": 1024,
            "output_dtype": "float",
            "truncation": True,
        }]

    def test_un_document_part_en_input_type_document(self, faux):
        doublure = faux()
        VoyageProvider(api_key=CLE).embed(["Article premier."], DOCUMENT, DIM)

        assert doublure.corps[0]["input_type"] == "document"

    def test_aucun_prefixe_cote_client(self, faux):
        """Voyage ajoute lui-meme son gabarit : un prefixe ici le doublerait."""
        doublure = faux()
        VoyageProvider(api_key=CLE).embed(["texte brut", "autre"], DOCUMENT, DIM)

        assert doublure.corps[0]["input"] == ["texte brut", "autre"]

    def test_tache_inconnue_refusee_sans_appel(self, faux):
        """
        Un input_type absent est accepte par l'API mais encode SANS gabarit,
        dans un espace voisin, sans erreur : il faut refuser avant.
        """
        doublure = faux()
        with pytest.raises(EmbeddingDefinitiveError, match="task_type inconnu"):
            VoyageProvider(api_key=CLE).embed(["X"], "CLASSIFICATION", DIM)
        assert doublure.requetes == []

    @pytest.mark.parametrize("dim", [768, 1536, 3072])
    def test_dimension_hors_liste_refusee_sans_appel(self, faux, dim):
        doublure = faux()
        with pytest.raises(EmbeddingDefinitiveError, match=str(dim)):
            VoyageProvider(api_key=CLE).embed(["X"], DOCUMENT, dim)
        assert doublure.requetes == []

    def test_ordre_retabli_par_index(self, faux):
        """La reponse peut arriver dans le desordre : `index` fait foi."""
        faux(melanger=True)
        textes = ["a", "bb", "ccc"]
        vecteurs = VoyageProvider(api_key=CLE).embed(textes, DOCUMENT, DIM)

        attendus = [(np.random.default_rng(len(t) + 7 * i).random(DIM) * 3.0).astype(np.float32)
                    for i, t in enumerate(textes)]
        for v, a in zip(vecteurs, attendus):
            np.testing.assert_allclose(v, a, rtol=1e-6)

    def test_reponse_incomplete_refusee(self, faux):
        def _deux_sur_trois(requete, corps):
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [0.1] * DIM} for i in range(2)
            ]})
        faux(reponse=_deux_sur_trois)

        with pytest.raises(EmbeddingServiceError, match="incomplète"):
            VoyageProvider(api_key=CLE).embed(["a", "b", "c"], DOCUMENT, DIM)


# ==================== LOTS ====================


class TestLots:

    def test_textes_longs_repartis_sous_le_budget_de_jetons(self, faux):
        doublure = faux()
        p = VoyageProvider(api_key=CLE)
        textes = ["x" * p.max_caracteres] * 5
        vecteurs = p.embed(textes, DOCUMENT, DIM)

        assert len(vecteurs) == 5
        assert len(doublure.corps) > 1
        for corps in doublure.corps:
            estimes = sum(p._estimer(t) for t in corps["input"])
            assert estimes <= p.JETONS_PAR_REQUETE or len(corps["input"]) == 1

    def test_au_plus_900_textes_par_requete(self, faux):
        doublure = faux()
        VoyageProvider(api_key=CLE).embed(["t"] * 1000, DOCUMENT, DIM)

        assert [len(c["input"]) for c in doublure.corps] == [900, 100]

    def test_refus_trop_de_jetons_coupe_le_lot_en_deux(self, faux):
        """Un tableau de chiffres compte plus de jetons que prevu : on coupe."""
        def _trop_gros(requete, corps):
            if len(corps["input"]) > 2:
                return httpx.Response(400, json={
                    "detail": "Request to model 'voyage-4-large' failed. The max allowed "
                              "tokens per submitted batch is 120000."})
            return None
        doublure = faux(reponse=_trop_gros)
        textes = [f"t{i}" * (i + 1) for i in range(7)]
        vecteurs = VoyageProvider(api_key=CLE).embed(textes, DOCUMENT, DIM)

        assert len(vecteurs) == 7
        acceptes = [c["input"] for c in doublure.corps if len(c["input"]) <= 2]
        assert sum(acceptes, []) == textes, "ordre et completude preserves"

    def test_refus_trop_de_jetons_sur_un_texte_seul_est_definitif(self, faux):
        faux(reponse=_refus(400, "The example at index 0 has too many tokens"))

        with pytest.raises(EmbeddingDefinitiveError, match="HTTP 400"):
            VoyageProvider(api_key=CLE).embed(["un seul"], DOCUMENT, DIM)


# ==================== DELAIS ====================


class TestDelais:

    def test_une_question_a_un_delai_court(self, faux, monkeypatch):
        monkeypatch.setattr(settings, "VOYAGE_TIMEOUT_QUESTION_S", 3.0)
        doublure = faux()
        VoyageProvider(api_key=CLE).embed(["q"], QUERY, DIM)

        delais = doublure.requetes[0].extensions["timeout"]
        assert delais["read"] == 3.0
        assert delais["connect"] == 2.0

    def test_un_lot_de_documents_a_un_delai_long(self, faux, monkeypatch):
        monkeypatch.setattr(settings, "VOYAGE_TIMEOUT_DOCUMENT_S", 120.0)
        doublure = faux()
        VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)

        delais = doublure.requetes[0].extensions["timeout"]
        assert delais["read"] == 120.0
        assert delais["connect"] == 2.0

    def test_delai_question_relu_a_chaque_acces(self, monkeypatch):
        p = VoyageProvider(api_key=CLE)
        monkeypatch.setattr(settings, "VOYAGE_TIMEOUT_QUESTION_S", 1.5)
        assert p.delai_question_s == 1.5


# ==================== ERREURS ====================


class TestErreurs:

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_passager_rejouable_avec_son_code(self, faux, code):
        """
        Le code doit figurer dans le message : regenerate_embeddings ne
        reattend un lot que s'il y lit « 429 ».
        """
        faux(reponse=_refus(code, "Rate limit exceeded" if code == 429 else "Internal error"))

        with pytest.raises(EmbeddingServiceError, match=f"HTTP {code}") as erreur:
            VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)
        assert not isinstance(erreur.value, EmbeddingDefinitiveError)

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_definitif_jamais_rejoue(self, faux, code):
        faux(reponse=_refus(code, "Provided API key is invalid."))

        with pytest.raises(EmbeddingDefinitiveError, match=f"HTTP {code}"):
            VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)

    def test_une_erreur_reseau_ne_porte_jamais_la_cle(self, monkeypatch):
        """h11 recopie un en-tete refuse dans son message : masque, et sans chaine."""
        def _refus_d_en_tete(requete):
            raise httpx.LocalProtocolError(f"Illegal header value b'Bearer {CLE} '")
        monkeypatch.setattr(module, "transport_http", httpx.MockTransport(_refus_d_en_tete))

        with pytest.raises(EmbeddingServiceError) as erreur:
            VoyageProvider(api_key=CLE).embed(["q"], QUERY, DIM)

        assert CLE not in str(erreur.value)
        assert erreur.value.__cause__ is None
        assert erreur.value.__suppress_context__

    @pytest.mark.parametrize("panne, motif", [
        (httpx.ConnectError, "injoignable"),
        (httpx.ConnectTimeout, "delai depasse"),
        (httpx.ReadTimeout, "delai depasse"),
    ])
    def test_reseau_en_panne_est_passager(self, monkeypatch, panne, motif):
        """Une lenteur n'est pas une coupure : le motif les distingue."""
        def _coupe(requete):
            raise panne("echec reseau", request=requete)
        monkeypatch.setattr(module, "transport_http", httpx.MockTransport(_coupe))

        with pytest.raises(EmbeddingServiceError, match=motif) as erreur:
            VoyageProvider(api_key=CLE).embed(["q"], QUERY, DIM)
        assert not isinstance(erreur.value, EmbeddingDefinitiveError)
        assert motif_court(erreur.value) == motif

    def test_la_cle_n_apparait_jamais_dans_un_message(self, faux):
        faux(reponse=_refus(401, f"Provided API key {CLE} is invalid."))

        with pytest.raises(EmbeddingDefinitiveError) as erreur:
            VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)
        assert CLE not in str(erreur.value)
        assert "***" in str(erreur.value)

    def test_une_cle_a_cheval_sur_la_coupe_ne_fuit_pas(self, faux):
        """Coupe a 200 caracteres AVANT de masquer, le debut de la cle passait."""
        faux(reponse=_refus(401, "x" * 190 + CLE + " y"))

        with pytest.raises(EmbeddingDefinitiveError) as erreur:
            VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)
        assert CLE[:6] not in str(erreur.value)

    @pytest.mark.parametrize("code, configuration", [
        (401, True), (403, True), (404, True), (400, False), (422, False),
    ])
    def test_le_refus_porte_son_code(self, faux, code, configuration):
        """
        400 et 422 visent un texte : un traitement par lots saute le lot. Les
        autres visent la cle, les droits ou le modele : il faut s'arreter.
        """
        faux(reponse=_refus(code, "refus"))

        with pytest.raises(EmbeddingDefinitiveError) as erreur:
            VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)
        assert erreur.value.code == code
        assert erreur.value.vise_la_configuration is configuration

    def test_un_refus_avant_envoi_vise_la_configuration(self, faux):
        faux()
        with pytest.raises(EmbeddingDefinitiveError) as erreur:
            VoyageProvider(api_key=CLE).embed(["X"], "CLASSIFICATION", DIM)
        assert erreur.value.code is None
        assert erreur.value.vise_la_configuration

    def test_reponse_illisible(self, faux):
        faux(reponse=lambda requete, corps: httpx.Response(200, text="<html>proxy</html>"))

        with pytest.raises(EmbeddingServiceError, match="illisible"):
            VoyageProvider(api_key=CLE).embed(["d"], DOCUMENT, DIM)


class TestReprisesDuService:
    """Le service rejoue le passager, jamais le definitif, et jamais une question."""

    @pytest.fixture(autouse=True)
    def _sans_pause(self, monkeypatch):
        monkeypatch.setattr(module.time, "sleep", lambda *_: None)

    def _service(self):
        return EmbeddingService(use_cache=False, provider=VoyageProvider(api_key=CLE))

    def test_un_document_passager_est_rejoue(self, faux):
        doublure = faux(reponse=_refus(503, "Service unavailable"))

        with pytest.raises(EmbeddingServiceError, match="503"):
            self._service().generate_embedding("article", task_type=DOCUMENT)
        assert len(doublure.requetes) == EmbeddingService.MAX_RETRIES

    def test_un_refus_definitif_n_est_pas_rejoue(self, faux):
        doublure = faux(reponse=_refus(401, "Provided API key is invalid."))

        with pytest.raises(EmbeddingDefinitiveError):
            self._service().generate_embedding("article", task_type=DOCUMENT)
        assert len(doublure.requetes) == 1

    def test_un_lot_definitif_n_est_pas_rejoue(self, faux):
        doublure = faux(reponse=_refus(403, "Forbidden"))

        with pytest.raises(EmbeddingDefinitiveError):
            self._service().generate_batch_embeddings(["a", "b"])
        assert len(doublure.requetes) == 1

    def test_une_question_n_a_qu_un_essai(self, faux):
        """
        Trois essais et leurs pauses feraient attendre l'utilisateur plus de
        dix secondes avant le repli sur le plein texte.
        """
        doublure = faux(reponse=_refus(503, "Service unavailable"))

        with pytest.raises(EmbeddingServiceError, match="503"):
            self._service().generate_embedding("question", task_type=QUERY)
        assert len(doublure.requetes) == 1

    def test_vecteurs_normalises_par_le_service(self, faux):
        faux()
        v = self._service().generate_embedding("article", task_type=DOCUMENT)

        assert v.shape == (DIM,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


# ==================== CONNEXION GARDEE ====================


class TestConnexion:
    """
    Un client par appel rouvrait TCP et TLS a chaque question : 1,14 s de
    mediane contre 0,42 s avec une connexion reutilisee (mesure).
    """

    def test_le_client_est_garde_d_un_appel_a_l_autre(self, faux, monkeypatch):
        doublure = faux()
        vrai = httpx.Client
        constructions = []
        monkeypatch.setattr(
            module.httpx, "Client", lambda *a, **k: constructions.append(1) or vrai(*a, **k)
        )
        p = VoyageProvider(api_key=CLE)

        p.embed(["a"], QUERY, DIM)
        premier = p._client_http
        assert premier is not None and not premier.is_closed
        p.embed(["b"], QUERY, DIM)

        assert p._client_http is premier
        assert len(constructions) == 1
        assert len(doublure.requetes) == 2

    def test_un_nouveau_transport_reconstruit_le_client(self, faux, monkeypatch):
        """La garde reseau des tests change de transport : l'ancien client irait au reseau."""
        premiere = faux()
        p = VoyageProvider(api_key=CLE)
        p.embed(["a"], QUERY, DIM)
        ancien = p._client_http

        seconde = FauxVoyage()
        monkeypatch.setattr(module, "transport_http", httpx.MockTransport(seconde))
        p.embed(["b"], QUERY, DIM)

        assert len(premiere.requetes) == 1 and len(seconde.requetes) == 1
        assert ancien.is_closed
        assert p._client_http is not ancien

    def test_fermer(self, faux):
        faux()
        p = VoyageProvider(api_key=CLE)
        p.embed(["a"], QUERY, DIM)
        client = p._client_http

        p.fermer()

        assert client.is_closed
        assert p._client_http is None

    def test_les_connexions_inactives_durent_entre_deux_questions(self, monkeypatch):
        """
        5 s par defaut chez httpx : la connexion serait refermee entre deux
        questions. Verifie sur le VRAI pool (avec une doublure de transport,
        httpx ignore les limites) ; aucune requete n'est envoyee.
        """
        monkeypatch.setattr(module, "transport_http", None)
        p = VoyageProvider(api_key=CLE)
        try:
            client = p._client()
            assert client._transport._pool._keepalive_expiry == VoyageProvider.CONNEXION_INACTIVE_S
        finally:
            p.fermer()
        assert VoyageProvider.CONNEXION_INACTIVE_S >= 30


# ==================== ECHEANCE DES QUESTIONS ====================


class _Lent:
    """Fournisseur qui dort plus longtemps que son delai declare."""

    name = "lent"
    label = "Fournisseur lent"
    model = "lent"
    fingerprint = "lent|lent"
    native_dim = DIM
    retryable = True
    inter_batch_delay_s = 0.0

    def __init__(self, delai, sommeil):
        self.delai_question_s = delai
        self.sommeil = sommeil
        self.fils = []

    def embed(self, texts, task_type, dim):
        self.fils.append(threading.current_thread().name)
        time.sleep(self.sommeil)
        return [np.ones(dim, dtype=np.float32) for _ in texts]


class TestEcheance:

    @pytest.mark.asyncio
    async def test_une_question_lente_echoue_a_l_echeance(self):
        """
        httpx ne borne pas la resolution DNS : seule l'echeance du service
        garantit le delai, au-dela duquel la recherche repond en plein texte.
        """
        # Echeance a 1,1 s, sommeil de 3 s : une marge large, pour une machine
        # chargee. Le fil finit son sommeil apres le test, sans effet.
        lent = _Lent(delai=0.1, sommeil=3.0)
        service = EmbeddingService(use_cache=False, provider=lent)

        debut = time.monotonic()
        with pytest.raises(EmbeddingServiceError, match="Pas de vecteur pour la question"):
            await service.generate_embedding_async("question", task_type=QUERY)
        assert time.monotonic() - debut < 2.5

    @pytest.mark.asyncio
    async def test_une_question_rapide_passe(self):
        service = EmbeddingService(use_cache=False, provider=_Lent(delai=2.0, sommeil=0.0))

        v = await service.generate_embedding_async("question", task_type=QUERY)
        assert v.shape == (DIM,)

    @pytest.mark.asyncio
    async def test_les_questions_ont_leurs_propres_fils(self):
        """Un fournisseur lent ne doit pas saturer l'executeur par defaut."""
        lent = _Lent(delai=2.0, sommeil=0.0)
        service = EmbeddingService(use_cache=False, provider=lent)

        await service.generate_embedding_async("question", task_type=QUERY)
        assert lent.fils[-1].startswith("embed-question")

    @pytest.mark.asyncio
    async def test_un_document_n_a_pas_d_echeance(self):
        """
        Un lot de documents peut durer : seul son delai HTTP le borne. Sommeil
        de 1,3 s, au-dela de l'echeance qu'aurait une question (1,05 s).
        """
        lent = _Lent(delai=0.05, sommeil=1.3)
        service = EmbeddingService(use_cache=False, provider=lent)

        v = await service.generate_embedding_async("article", task_type=DOCUMENT)
        assert v.shape == (DIM,)
        assert not lent.fils[-1].startswith("embed-question")

    @pytest.mark.asyncio
    async def test_une_question_deja_en_cache_ne_passe_pas_par_les_fils_dedies(self, monkeypatch):
        """
        Quand le fournisseur ne repond plus, ses appels occupent les fils
        dedies : une question deja vectorisee ne doit pas attendre derriere.
        """
        fournisseur = VoyageProvider(api_key=CLE)
        service = EmbeddingService(use_cache=False, provider=fournisseur)
        service.use_cache = True
        attendu = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        lectures = []

        def _depuis_le_cache(cle):
            lectures.append(threading.current_thread().name)
            return attendu

        monkeypatch.setattr(service, "_get_from_pg_cache", _depuis_le_cache)
        monkeypatch.setattr(fournisseur, "embed", lambda *a: pytest.fail("aucun appel attendu"))

        v = await service.generate_embedding_async("question deja posee", task_type=QUERY)

        np.testing.assert_array_equal(v, attendu)
        # Sur les fils du cache : ni la boucle (psycopg2 la gelerait), ni ceux
        # des appels au fournisseur.
        assert lectures == [lectures[0]] and lectures[0].startswith("embed-cache")
        # Servie par le cache : elle ne dit rien du fournisseur.
        assert suivi_semantique.etat()["appels"] == 0

    @pytest.mark.asyncio
    async def test_une_question_en_cache_passe_meme_fils_des_questions_bloques(self, monkeypatch):
        """Voyage ne repond plus : ses appels tiennent les quatre fils des questions."""
        lent = _Lent(delai=0.5, sommeil=2.0)
        service = EmbeddingService(use_cache=False, provider=lent)
        service.use_cache = True
        attendu = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        cle_connue = service._cache_key("deja posee", QUERY)
        monkeypatch.setattr(service, "_get_from_pg_cache", lambda cle: attendu if cle == cle_connue else None)

        bloquees = [
            asyncio.create_task(service.generate_embedding_async(f"nouvelle {i}", task_type=QUERY))
            for i in range(4)
        ]
        await asyncio.sleep(0.2)
        debut = time.monotonic()
        v = await service.generate_embedding_async("deja posee", task_type=QUERY)

        assert time.monotonic() - debut < 0.5
        np.testing.assert_array_equal(v, attendu)
        for tache in bloquees:
            with pytest.raises(EmbeddingServiceError, match="Pas de vecteur"):
                await tache

    @pytest.mark.asyncio
    async def test_une_question_nouvelle_ne_lit_le_cache_qu_une_fois(self, monkeypatch, faux):
        faux()
        service = EmbeddingService(use_cache=False, provider=VoyageProvider(api_key=CLE))
        service.use_cache = True
        lectures, ecritures = [], []
        monkeypatch.setattr(service, "_get_from_pg_cache", lambda cle: lectures.append(cle))
        monkeypatch.setattr(service, "_store_in_pg_cache", lambda cle, v: ecritures.append(cle))

        await service.generate_embedding_async("question nouvelle", task_type=QUERY)
        module._EXECUTEUR_CACHE.submit(lambda: None).result(timeout=5)

        assert len(lectures) == 1
        assert ecritures == lectures


# ==================== SUIVI DU VOLET SEMANTIQUE ====================


class TestSuiviSemantique:
    """
    Ce que GET /search/health montre des questions recentes. Un TAUX sur une
    fenetre glissante : sous un bridage, succes et refus alternent ; et un etat
    qui ne vieillit pas laisserait la sante degradee une nuit entiere.
    """

    @pytest.mark.asyncio
    async def test_les_appels_au_fournisseur_sont_comptes(self, faux):
        doublure = faux(reponse=_refus(429, "You have exceeded the rate limit"))
        service = EmbeddingService(use_cache=False, provider=VoyageProvider(api_key=CLE))

        with pytest.raises(EmbeddingServiceError):
            await service.generate_embedding_async("q1", task_type=QUERY)
        doublure.reponse = None
        await service.generate_embedding_async("q2", task_type=QUERY)

        etat = suivi_semantique.etat()
        assert (etat["appels"], etat["echecs"], etat["derniere_erreur"]) == (2, 1, "HTTP 429")

    @pytest.mark.asyncio
    async def test_l_echeance_est_comptee(self):
        service = EmbeddingService(use_cache=False, provider=_Lent(delai=0.05, sommeil=1.5))

        with pytest.raises(EmbeddingServiceError):
            await service.generate_embedding_async("q", task_type=QUERY)

        assert suivi_semantique.etat()["derniere_erreur"] == "delai depasse"

    def test_degrade_au_dela_du_taux_et_du_minimum(self):
        suivi = SuiviSemantique()
        for _ in range(2):
            suivi.noter(RuntimeError("Voyage HTTP 429"))
        assert not suivi.etat()["degrade"], "deux echecs ne suffisent pas"

        suivi.noter(RuntimeError("Voyage HTTP 429"))
        for _ in range(6):
            suivi.noter(None)
        etat = suivi.etat()
        assert (etat["appels"], etat["echecs"]) == (9, 3)
        assert etat["degrade"], "3 echecs sur 9, au-dela de 30 %"

        for _ in range(2):
            suivi.noter(None)
        assert not suivi.etat()["degrade"], "3 sur 11 : sous 30 %"

    def test_une_sonde_ratee_degrade_jusqu_au_premier_succes(self):
        """
        Echec passager au demarrage : le service est garde, mais rien ne dit
        que le volet semantique fonctionne tant qu'aucun appel n'a reussi.
        """
        suivi = SuiviSemantique()
        suivi.noter_sonde(RuntimeError("Voyage injoignable : ConnectError"))
        etat = suivi.etat()
        assert etat["degrade"]
        assert etat["sonde_demarrage"]["erreur"] == "injoignable"

        suivi.noter(None)
        assert not suivi.etat()["degrade"]

    def test_une_sonde_reussie_ne_degrade_pas(self):
        suivi = SuiviSemantique()
        suivi.noter_sonde(None)
        assert not suivi.etat()["degrade"]
        assert suivi.etat()["sonde_demarrage"]["erreur"] is None

    def test_l_etat_vieillit(self, monkeypatch):
        suivi = SuiviSemantique()
        for _ in range(3):
            suivi.noter(RuntimeError("Voyage HTTP 401"))
        assert suivi.etat()["degrade"]

        maintenant = time.monotonic() + suivi.FENETRE_S + 1
        monkeypatch.setattr(module.time, "monotonic", lambda: maintenant)

        etat = suivi.etat()
        assert etat["appels"] == 0 and not etat["degrade"]

    @pytest.mark.parametrize("erreur, motif", [
        (RuntimeError("Voyage HTTP 401 : Provided API key pa-XXXX is invalid."), "HTTP 401"),
        (RuntimeError("Pas de vecteur pour la question en 4.0 s (Voyage AI)"), "delai depasse"),
        (TimeoutError(), "delai depasse"),
        (RuntimeError("Voyage injoignable : ConnectError: [Errno -3]"), "injoignable"),
    ])
    def test_le_motif_ne_garde_rien_du_message(self, erreur, motif):
        """La route est publique : ni cle, ni detail, seulement le motif."""
        assert motif_court(erreur) == motif

    def test_une_autre_erreur_donne_sa_classe(self):
        erreur = RuntimeError("echec")
        erreur.__cause__ = ValueError("different halfvec dimensions 768 and 1024")
        assert motif_court(erreur) == "ValueError"


# ==================== EMPREINTE, CONSTRUCTION, DEFAUT ====================


class TestEmpreinte:

    def test_empreinte_litterale(self):
        """
        C'est la valeur ecrite en base pour les 32 303 vecteurs charges : un
        caractere de difference, et /search/health les compte « foreign ».
        """
        assert VoyageProvider(api_key=CLE).fingerprint == "voyage|voyage-4-large"

    def test_empreinte_du_chargeur_identique_a_celle_du_fournisseur(self):
        assert empreinte_voyage(settings.VOYAGE_EMBEDDING_MODEL) == VoyageProvider(api_key=CLE).fingerprint

    def test_empreinte_suit_le_modele(self, monkeypatch):
        monkeypatch.setattr(settings, "VOYAGE_EMBEDDING_MODEL", "voyage-4-lite")
        assert VoyageProvider(api_key=CLE).fingerprint == "voyage|voyage-4-lite"

    def test_voyage_et_gemini_ont_des_empreintes_distinctes(self, monkeypatch):
        monkeypatch.setattr(module.genai, "Client", lambda *a, **k: object())
        assert VoyageProvider(api_key=CLE).fingerprint != GeminiProvider(api_key="x").fingerprint


class TestConstruction:

    def test_les_blancs_autour_de_la_cle_sont_retires(self, faux):
        """Un saut de ligne recopie avec la cle rendait l'en-tete illegal."""
        doublure = faux()
        VoyageProvider(api_key=f"  {CLE}\n").embed(["q"], QUERY, DIM)

        assert doublure.requetes[0].headers["authorization"] == f"Bearer {CLE}"

    def test_cle_absente_refusee_a_la_construction(self, monkeypatch):
        monkeypatch.setattr(settings, "VOYAGE_API_KEY", "")
        with pytest.raises(EmbeddingServiceError, match="VOYAGE_API_KEY non configurée"):
            VoyageProvider()

    def test_construire_n_appelle_pas_le_reseau(self, faux):
        doublure = faux()
        EmbeddingService(use_cache=False, provider=VoyageProvider(api_key=CLE))
        assert doublure.requetes == []

    def test_fournisseur_choisi_par_la_configuration(self, monkeypatch):
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "voyage")
        assert isinstance(module._construire_fournisseur(api_key=CLE), VoyageProvider)
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setattr(module.genai, "Client", lambda *a, **k: object())
        assert isinstance(module._construire_fournisseur(api_key="x"), GeminiProvider)

    def test_le_defaut_de_production_est_voyage(self):
        """Lu sur la DECLARATION : un .env peut surcharger le reglage."""
        from app.core.config import Settings

        assert Settings.model_fields["EMBEDDING_PROVIDER"].default == "voyage"
        assert Settings.model_fields["VOYAGE_EMBEDDING_MODEL"].default == "voyage-4-large"

    def test_la_limite_de_texte_est_celle_du_fournisseur(self):
        service = EmbeddingService(use_cache=False, provider=VoyageProvider(api_key=CLE))
        assert service.MAX_TEXT_LENGTH == VoyageProvider.max_caracteres == 78_000


class TestPrechauffage:

    def test_une_seule_question_a_la_dimension_du_service(self, faux):
        """
        Une cle revoquee, ou recopiee d'un exemple, doit se voir au demarrage
        et non a la premiere question.
        """
        doublure = faux()
        EmbeddingService(use_cache=False, provider=VoyageProvider(api_key=CLE)).prechauffer()

        assert len(doublure.corps) == 1
        assert doublure.corps[0]["input_type"] == "query"
        assert doublure.corps[0]["output_dimension"] == DIM

    def test_une_sortie_de_mauvaise_forme_est_definitive(self, faux):
        faux(dim=512)
        with pytest.raises(EmbeddingDefinitiveError, match="attendu"):
            VoyageProvider(api_key=CLE).prechauffer(DIM)

    def test_une_cle_refusee_est_definitive(self, faux):
        faux(reponse=_refus(401, "Provided API key is invalid."))
        with pytest.raises(EmbeddingDefinitiveError):
            VoyageProvider(api_key=CLE).prechauffer(DIM)

    def test_sans_effet_pour_un_fournisseur_sans_sonde(self):
        """Gemini n'a pas de sonde : rien n'est appele."""
        class _Api:
            label = "API doublee"
            model = "modele"
            fingerprint = "api|modele"
            native_dim = DIM
            retryable = True
            inter_batch_delay_s = 0.0

            def embed(self, *args):
                raise AssertionError("aucun appel attendu")

        EmbeddingService(use_cache=False, provider=_Api()).prechauffer()


# ==================== LE VRAI VOYAGE ====================


MESURE = Path(__file__).resolve().parents[2] / "data" / "mesure_voyage"


def _cosinus(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _vecteur_des_lots(article_id):
    for fichier in sorted((MESURE / "lots" / "voyage-4-large").glob("*.npz")):
        with np.load(fichier) as z:
            ids = z["ids"].tolist()
            if article_id in ids:
                return z["v"][ids.index(article_id)]
    return None


@pytest.mark.voyage_live
class TestVoyageReel:
    """
    La preuve, AVANT tout chargement, que le corps de production retrouve
    l'espace des vecteurs calcules par la mesure. Quelques milliers de jetons.
    """

    @pytest.fixture
    def service(self):
        service = EmbeddingService(use_cache=False)
        yield service
        # Une vraie connexion TLS laissee ouverte leverait un ResourceWarning,
        # erreur sous filterwarnings = error.
        service.provider.fermer()

    def test_les_questions_retrouvent_celles_de_la_mesure(self, service):
        reference = MESURE / "questions_voyage-4-large_fr.npy"
        if not reference.exists():
            pytest.skip(f"{reference} absent")
        jeu = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "retrieval_eval_v1.json"
        items = json.loads(jeu.read_text(encoding="utf-8"))["items"]
        attendus = np.load(reference)

        for k in (0, 57):
            v = service.generate_embedding(items[k]["question"], task_type=QUERY)
            assert _cosinus(v, attendus[k]) >= 0.999, items[k]["question"]

    @pytest.mark.parametrize("passe", ["reduite", "complete"])
    def test_les_documents_retrouvent_ceux_des_lots(self, service, passe):
        """
        Un morceau de chaque passe de la mesure : les 38 lois reduites (15/10),
        puis le reste du corpus (16/10), dont le plus long morceau (15 323
        caracteres), au-dela de l'ancienne coupe de 10 000.
        """
        instantane = MESURE / "morceaux.jsonl"
        if not instantane.exists():
            pytest.skip(f"{instantane} absent")
        reduites = set(json.loads((MESURE / "lois_reduites.json").read_text()))
        morceaux = [json.loads(ligne) for ligne in instantane.open(encoding="utf-8")]
        if passe == "reduite":
            morceau = next(m for m in morceaux if m["law_id"] in reduites and 800 < len(m["texte"]) < 3000)
        else:
            morceau = max((m for m in morceaux if m["law_id"] not in reduites), key=lambda m: len(m["texte"]))
            assert len(morceau["texte"]) > 10_000
        attendu = _vecteur_des_lots(morceau["id"])
        assert attendu is not None, f"morceau {morceau['id']} absent des lots"

        v = service.generate_batch_embeddings([morceau["texte"]])[0]
        assert _cosinus(v, attendu) >= 0.999, morceau["id"]
