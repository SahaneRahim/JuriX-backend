"""
Tests du service d'extraction LlamaParse.

Aucun appel réseau : `respx` intercepte httpx. Sans cela ces tests
consommeraient des crédits LlamaParse à chaque exécution.
"""

import json

import httpx
import pytest
import respx

from app.services.llama_parse_service import (
    LlamaParseError,
    LlamaParseService,
    _strip_stamp_blocks,
)

BASE = "https://api.cloud.llamaindex.ai/api/v2/parse"


@pytest.fixture
def service(tmp_path):
    return LlamaParseService(api_key="cle-de-test", cache_dir=tmp_path / "cache")


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\ncontenu\n%%EOF")
    return p


class TestNettoyageDesCachets:
    """
    Retrait du cachet officiel apposé sur chaque page du corpus prc.cm.

    LlamaParse restitue ce cachet en texte ; sans nettoyage il pollue le FTS et
    les embeddings de tous les documents.
    """

    def test_bloc_multiligne_retire(self):
        md = (
            "Article 1er. Contenu juridique réel.\n\n"
            "PRESIDENCE DE LA REPUBLIQUE\n"
            "SERVICE DU FICHIER LEGISLATIF ET REGLEMENTAIRE\n"
            "COPIE CERTIFIEE CONFORME\n"
            "CERTIFIED TRUE COPY\n\n"
            "Article 2. Autre contenu."
        )
        out = _strip_stamp_blocks(md)
        assert "COPIE CERTIFIEE CONFORME" not in out
        assert "Article 1er. Contenu juridique réel." in out
        assert "Article 2. Autre contenu." in out

    def test_cachet_sur_une_seule_ligne_retire(self):
        """LlamaParse condense parfois tout le cachet sur une ligne."""
        md = (
            "Article 1er. Contenu.\n"
            "[signature: PRESIDENCE DE LA REPUBLIQUE SECRETARIAT GENERAL "
            "COPIE CERTIFIEE CONFORME CERTIFIED TRUE COPY]\n"
        )
        out = _strip_stamp_blocks(md)
        assert "CERTIFIED TRUE COPY" not in out
        assert "Article 1er. Contenu." in out

    def test_en_tete_legitime_conserve(self):
        """
        Beaucoup de décrets portent « PRESIDENCE DE LA REPUBLIQUE » comme
        autorité émettrice : c'est du contenu, pas un tampon. Une seule phrase
        du vocabulaire ne doit donc rien déclencher.
        """
        md = "PRESIDENCE DE LA REPUBLIQUE\n\nDECRET N° 2024/191 du 4 juin 2024"
        out = _strip_stamp_blocks(md)
        assert "PRESIDENCE DE LA REPUBLIQUE" in out
        assert "DECRET N° 2024/191" in out

    def test_filigrane_retire(self):
        md = "Article 1er.\nw\nw\nw\n.prc\n.cm\nContenu."
        out = _strip_stamp_blocks(md)
        assert "Article 1er." in out and "Contenu." in out
        assert "\nw\n" not in f"\n{out}\n"

    def test_tableaux_conserves(self):
        """Les <table> portent du sens (annexes budgétaires) — jamais retirés."""
        md = "<table><tr><td>Recettes</td><td>66 900 000</td></tr></table>"
        assert "<table>" in _strip_stamp_blocks(md)

    def test_balises_inline_retirees(self):
        """<sup> casse la détection d'article : ARTICLE 1<sup>ER</sup>."""
        assert _strip_stamp_blocks("**ARTICLE 1<sup>ER</sup>**") == "**ARTICLE 1ER**"


class TestDisponibilite:
    def test_indisponible_sans_cle(self, tmp_path, monkeypatch):
        # api_key="" retombe volontairement sur settings.LLAMA_CLOUD_API_KEY :
        # il faut donc neutraliser aussi la configuration pour tester ce cas.
        monkeypatch.setattr(
            "app.services.llama_parse_service.settings.LLAMA_CLOUD_API_KEY", ""
        )
        assert not LlamaParseService(api_key="", cache_dir=tmp_path).is_available()

    def test_disponible_avec_cle(self, service):
        assert service.is_available()


class TestExtraction:
    @pytest.mark.asyncio
    @respx.mock
    async def test_extraction_et_marqueurs_de_page(self, service, pdf):
        respx.post(f"{BASE}/upload").mock(
            return_value=httpx.Response(200, json={"job": {"id": "job-1"}})
        )
        respx.get(f"{BASE}/job-1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "job": {"status": "COMPLETED"},
                    "markdown": [{"markdown": "Page une"}, {"markdown": "Page deux"}],
                },
            )
        )
        service.POLL_INTERVAL = 0
        text = await service.extract_text(pdf)
        # Les marqueurs alimentent Article.page_number via text_chunker
        assert "<<PAGE:1>>" in text and "<<PAGE:2>>" in text
        assert "Page une" in text and "Page deux" in text

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_evite_un_second_appel(self, service, pdf):
        upload = respx.post(f"{BASE}/upload").mock(
            return_value=httpx.Response(200, json={"job": {"id": "job-1"}})
        )
        respx.get(f"{BASE}/job-1").mock(
            return_value=httpx.Response(
                200,
                json={"job": {"status": "COMPLETED"}, "markdown": [{"markdown": "Texte"}]},
            )
        )
        service.POLL_INTERVAL = 0
        first = await service.extract_text(pdf)
        second = await service.extract_text(pdf)
        assert first == second
        assert upload.call_count == 1, "le second appel aurait dû être servi par le cache"

    @pytest.mark.asyncio
    @respx.mock
    async def test_job_en_echec_leve(self, service, pdf):
        respx.post(f"{BASE}/upload").mock(
            return_value=httpx.Response(200, json={"job": {"id": "job-1"}})
        )
        respx.get(f"{BASE}/job-1").mock(
            return_value=httpx.Response(
                200,
                json={"job": {"status": "FAILED", "error_message": "document illisible"}},
            )
        )
        service.POLL_INTERVAL = 0
        with pytest.raises(LlamaParseError, match="document illisible"):
            await service.extract_text(pdf)

    @pytest.mark.asyncio
    @respx.mock
    async def test_cle_refusee_ne_declenche_pas_de_retry(self, service, pdf):
        """Un 401 est définitif : réessayer 4 fois ne ferait que perdre du temps."""
        route = respx.post(f"{BASE}/upload").mock(
            return_value=httpx.Response(401, text="cle invalide")
        )
        service.POLL_INTERVAL = 0
        with pytest.raises(LlamaParseError):
            await service.extract_text(pdf)
        assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_fichier_absent(self, service, tmp_path):
        with pytest.raises(LlamaParseError, match="introuvable"):
            await service.extract_text(tmp_path / "absent.pdf")


class TestNormalisationDesReponses:
    """L'API v2 renvoie le markdown sous plusieurs formes selon les options."""

    def test_liste_de_pages(self):
        pages = LlamaParseService._pages_from(
            {"markdown": [{"markdown": "a"}, {"markdown": "b"}]}
        )
        assert pages == ["a", "b"]

    def test_chaine_unique(self):
        assert LlamaParseService._pages_from({"markdown": "seule"}) == ["seule"]

    def test_champ_pages(self):
        assert LlamaParseService._pages_from({"pages": [{"markdown": "x"}]}) == ["x"]

    def test_markdown_full(self):
        assert LlamaParseService._pages_from({"markdown_full": "tout"}) == ["tout"]

    def test_reponse_vide(self):
        assert LlamaParseService._pages_from({}) == []
