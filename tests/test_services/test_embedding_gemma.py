"""
Tests du fournisseur EmbeddingGemma.

Deux familles :

- Sur DOUBLURE de session ONNX et de tokenizer, pour la logique : prefixes,
  sortie lue par son nom, troncature journalisee, lots par budget de jetons,
  ordre rendu, verrou, empreinte. Rapides, sans modele.
- Sur le VRAI modele, marques `gemma` : ils n'existent que pour prouver que la
  chaine fonctionne de bout en bout. Ils sautent si onnxruntime ou les
  fichiers manquent — et doivent avoir tourne au moins une fois sur un poste
  qui a le modele.

Chaque piege verifie ici casse EN SILENCE : un vecteur de 768, unitaire,
d'apparence saine, mais faux.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from app.services import embedding_service as module
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    GeminiProvider,
    GemmaProvider,
)

QUERY = EmbeddingService.TASK_QUERY
DOCUMENT = EmbeddingService.TASK_DOCUMENT


# ==================== DOUBLURES ====================


class _Encodage:
    def __init__(self, texte, n=None, tronque=False):
        n = n if n is not None else max(1, len(texte) // 4)
        self.ids = list(range(1, n + 1))
        self.attention_mask = [1] * n
        self.overflowing = [object()] if tronque else []


class _Tokenizer:
    def __init__(self, tronquer=()):
        self.recus = []
        self.tronquer = set(tronquer)

    def encode_batch(self, textes):
        self.recus.extend(textes)
        return [_Encodage(t, tronque=(i in self.tronquer)) for i, t in enumerate(textes)]


class _Session:
    """Rend, pour chaque ligne, un vecteur qui porte la LONGUEUR utile de son entree."""

    def __init__(self):
        self.appels = []

    def run(self, noms, feed):
        self.appels.append((list(noms), feed["input_ids"].shape))
        longueurs = feed["attention_mask"].sum(axis=1)
        sortie = np.zeros((len(longueurs), 768), dtype=np.float32)
        sortie[:, 0] = longueurs
        sortie[:, 1] = 1.0
        return [sortie]


@pytest.fixture
def modele_factice(tmp_path, monkeypatch):
    """Dossier de modele aux fichiers presents, et chargement remplace."""
    (tmp_path / "onnx").mkdir()
    for f in ("onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data", "tokenizer.json"):
        (tmp_path / f).write_bytes(b"")
    monkeypatch.setattr(module.settings, "GEMMA_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(module.settings, "GEMMA_ONNX_FILE", "onnx/model_quantized.onnx")

    etat = {"session": _Session(), "tok": _Tokenizer()}
    monkeypatch.setattr(module, "_charger_gemma", lambda *a, **k: (etat["session"], etat["tok"]))
    return etat


# ==================== PREFIXES ====================


class TestPrefixes:

    def test_requete_et_document_recoivent_leur_prefixe_exact(self, modele_factice):
        p = GemmaProvider()
        p.embed(["X"], QUERY, 768)
        p.embed(["X"], DOCUMENT, 768)

        # Au caractere pres : espace final, « none » en minuscules.
        assert modele_factice["tok"].recus == [
            "task: search result | query: X",
            "title: none | text: X",
        ]

    def test_task_type_inconnu_refuse(self, modele_factice):
        """Ignorer le prefixe encoderait dans un espace voisin, sans erreur."""
        with pytest.raises(EmbeddingServiceError, match="task_type inconnu"):
            GemmaProvider().embed(["X"], "CLASSIFICATION", 768)


# ==================== SORTIE DU MODELE ====================


class TestSortie:

    def test_sortie_demandee_par_son_nom(self, modele_factice):
        """
        La sortie 0 du graphe est last_hidden_state, des etats PAR JETON. Les
        moyenner donne un vecteur quasi orthogonal au bon (cosinus 0,0188
        mesure). Seul sentence_embedding inclut les couches Dense et la
        normalisation.
        """
        GemmaProvider().embed(["a", "b"], DOCUMENT, 768)

        noms = [n for n, _ in modele_factice["session"].appels]
        assert noms and all(n == ["sentence_embedding"] for n in noms)

    def test_dimension_matryoshka_tronquee(self, modele_factice):
        vecteurs = GemmaProvider().embed(["a"], DOCUMENT, 256)

        assert vecteurs[0].shape == (256,)

    def test_dimension_superieure_a_la_native_refusee(self, modele_factice):
        with pytest.raises(EmbeddingServiceError, match="768"):
            GemmaProvider().embed(["a"], DOCUMENT, 1024)


# ==================== TRONCATURE ====================


class TestTroncature:

    def test_troncature_journalisee(self, modele_factice, caplog):
        """
        10 000 caracteres valent ~2 100 a 2 500 jetons : la fin des articles
        les plus longs disparait. Le taire serait mentir sur ce qui est indexe.
        """
        modele_factice["tok"].tronquer = {1}

        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            GemmaProvider().embed(["court", "tres long"], DOCUMENT, 768)

        tronques = [r for r in caplog.records if "tronque" in r.getMessage()]
        assert len(tronques) == 1

    def test_pas_de_journal_sans_troncature(self, modele_factice, caplog):
        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            GemmaProvider().embed(["court"], DOCUMENT, 768)

        assert not [r for r in caplog.records if "tronque" in r.getMessage()]


# ==================== LOTS ====================


class TestLots:

    def test_ordre_d_origine_rendu(self, modele_factice, monkeypatch):
        """Les textes sont tries par longueur pour former les lots : l'ordre
        rendu doit etre celui de l'appelant, sinon chaque vecteur part sur le
        mauvais article."""
        monkeypatch.setattr(GemmaProvider, "JETONS_PAR_APPEL", 40)
        textes = ["x" * 80, "x" * 8, "x" * 40, "x" * 4, "x" * 60]

        vecteurs = GemmaProvider().embed(textes, DOCUMENT, 768)

        # La doublure ecrit la longueur utile en composante 0 ; le fournisseur
        # a prefixe chaque texte avant de le tokeniser.
        prefixe = module._GEMMA_PREFIXES[DOCUMENT]
        assert [int(v[0]) for v in vecteurs] == [len(prefixe + t) // 4 for t in textes]

    def test_budget_de_jetons_borne_chaque_passe(self, modele_factice, monkeypatch):
        monkeypatch.setattr(GemmaProvider, "JETONS_PAR_APPEL", 40)
        textes = ["x" * n for n in (80, 8, 40, 4, 60, 12, 16)]

        GemmaProvider().embed(textes, DOCUMENT, 768)

        passes = [forme for _, forme in modele_factice["session"].appels]
        assert len(passes) > 1
        # Une passe d'un seul texte peut depasser le budget (on ne coupe pas un
        # texte) ; une passe de plusieurs ne le doit jamais.
        assert all(n * longueur <= 40 for n, longueur in passes if n > 1)

    def test_verrou_pose_sur_les_documents_seulement(self, modele_factice, monkeypatch):
        """Une requete ne doit jamais attendre qu'un lot d'ingestion se termine."""
        pris = []

        class _Verrou:
            def __enter__(self):
                pris.append(True)

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(module, "_gemma_verrou_documents", _Verrou())
        p = GemmaProvider()

        p.embed(["question"], QUERY, 768)
        assert pris == []

        p.embed(["article"], DOCUMENT, 768)
        assert pris == [True]


# ==================== EMPREINTE ET CONSTRUCTION ====================


class TestEmpreinte:

    def test_gemma_et_gemini_ont_des_empreintes_distinctes(self, modele_factice, monkeypatch):
        monkeypatch.setattr(module.genai, "Client", lambda *a, **k: object())

        assert GemmaProvider().fingerprint != GeminiProvider(api_key="factice").fingerprint

    @pytest.mark.parametrize(
        "reglage, valeur",
        [
            ("GEMMA_REVISION", "0000000000000000000000000000000000000000"),
            ("GEMMA_MAX_TOKENS", 512),
        ],
    )
    def test_empreinte_suit_la_configuration(self, modele_factice, monkeypatch, reglage, valeur):
        avant = GemmaProvider().fingerprint
        monkeypatch.setattr(module.settings, reglage, valeur)

        assert GemmaProvider().fingerprint != avant

    def test_empreinte_suit_les_prefixes(self, modele_factice, monkeypatch):
        """Changer un prefixe change les vecteurs : l'empreinte doit suivre."""
        avant = GemmaProvider().fingerprint
        monkeypatch.setitem(module._GEMMA_PREFIXES, "RETRIEVAL_QUERY", "task: question answering | query: ")

        assert GemmaProvider().fingerprint != avant

    def test_fichier_du_modele_absent_refuse_a_la_construction(self, tmp_path, monkeypatch):
        """
        Une installation incomplete doit se voir ICI. Sinon la recherche
        hybride avale l'erreur au premier encodage et repond en plein texte.
        """
        monkeypatch.setattr(module.settings, "GEMMA_MODEL_DIR", str(tmp_path / "vide"))

        with pytest.raises(EmbeddingServiceError, match="introuvable"):
            GemmaProvider()

    def test_fournisseur_choisi_par_la_configuration(self, modele_factice, monkeypatch):
        monkeypatch.setattr(module.settings, "EMBEDDING_PROVIDER", "gemma")
        assert isinstance(module._construire_fournisseur(), GemmaProvider)

        monkeypatch.setattr(module.settings, "EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setattr(module.genai, "Client", lambda *a, **k: object())
        assert isinstance(module._construire_fournisseur(api_key="factice"), GeminiProvider)


class TestChargement:

    def test_session_et_tokenizer_charges_une_seule_fois(self, monkeypatch):
        """
        process_law.py construit un EmbeddingService par loi. Un chargement par
        instance relirait 300 Mo et 1,5 s de tokenizer a chaque loi.
        """
        import onnxruntime
        import tokenizers

        constructions = {"session": 0, "tok": 0}

        class _Tok:
            encode_special_tokens = False

            def enable_truncation(self, max_length):
                self.max_length = max_length

        def _from_file(chemin):
            constructions["tok"] += 1
            return _Tok()

        def _session(*a, **k):
            constructions["session"] += 1
            return object()

        monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(_from_file))
        monkeypatch.setattr(onnxruntime, "InferenceSession", _session)
        monkeypatch.setattr(module, "_gemma_ressources", {})

        a = module._charger_gemma("m.onnx", "t.json", 2048, 3)
        b = module._charger_gemma("m.onnx", "t.json", 2048, 3)

        assert a is b
        assert constructions == {"session": 1, "tok": 1}

    def test_tokenizer_configure_a_la_creation(self, monkeypatch):
        """Troncature ET jetons speciaux lus comme du texte, une fois pour toutes."""
        import onnxruntime
        import tokenizers

        cree = {}

        class _Tok:
            encode_special_tokens = False

            def enable_truncation(self, max_length):
                self.max_length = max_length

        def _from_file(chemin):
            cree["tok"] = _Tok()
            return cree["tok"]

        monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(_from_file))
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: object())
        monkeypatch.setattr(module, "_gemma_ressources", {})

        module._charger_gemma("m.onnx", "t.json", 2048, 3)

        assert cree["tok"].max_length == 2048
        assert cree["tok"].encode_special_tokens is True


# ==================== LE VRAI MODELE ====================


def _dossier_reel():
    dossier = Path(os.environ.get("GEMMA_MODEL_DIR") or module.settings.GEMMA_MODEL_DIR)
    if not (dossier / module.settings.GEMMA_ONNX_FILE).is_file():
        pytest.skip(f"modele EmbeddingGemma absent de {dossier}")
    return dossier


@pytest.mark.gemma
class TestModeleReel:
    """
    La preuve de bout en bout. Le reste de cette suite verifie la logique sur
    doublures ; seul ce bloc prouve que le vrai modele produit ce qu'on croit.
    """

    @pytest.fixture
    def service(self, monkeypatch):
        pytest.importorskip("onnxruntime")
        monkeypatch.setattr(module.settings, "GEMMA_MODEL_DIR", str(_dossier_reel()))
        # Fixee AVANT la construction : le service refuse une dimension que
        # le fournisseur ne sait pas produire.
        monkeypatch.setattr(EmbeddingService, "EMBEDDING_DIM", 768)
        return EmbeddingService(use_cache=False, provider=GemmaProvider())

    def test_vecteur_de_768_unitaire(self, service):
        v = service.generate_embedding("Article premier du Code minier", task_type=QUERY)

        assert v.shape == (768,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_la_bonne_loi_sort_en_premier(self, service):
        question = service.generate_embedding(
            "quelle est la duree du permis de recherche miniere ?", task_type=QUERY
        )
        articles = service.generate_batch_embeddings([
            "ARTICLE 12.-Le permis de recherche est accorde pour une duree de "
            "trois ans renouvelable deux fois.",
            "ARTICLE 3.-La presente loi regit les relations de travail entre "
            "les employeurs et les travailleurs.",
            "Les dirigeants de societe sont responsables civilement et "
            "penalement de leurs actes de gestion.",
        ])

        scores = [float(a @ question) for a in articles]

        assert scores[0] == max(scores)
        assert scores[0] - max(scores[1:]) > 0.2

    def test_jeton_special_en_clair_ne_fait_pas_echouer_le_lot(self, service):
        """<image_soft_token> donnerait l'id 262144, hors de la table."""
        v = service.generate_embedding("texte OCR avec <image_soft_token> et <eos>")

        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
