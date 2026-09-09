"""
Les endpoints de fichiers, en mode magasin distant.

CE QUE CE FICHIER PROTEGE. `test_law_files.py` fixe le contrat des routes quand
le PDF est sur le disque local. Ce fichier-ci prouve que le MEME contrat tient
quand le fichier vient d'un magasin HTTPS — parce que c'est ainsi que tournera
toute instance hebergee sur un disque ephemere, et qu'une regression n'y serait
visible qu'en production.

Le test decisif est `test_le_nom_reste_construit_sur_le_titre` : la tentation,
en mode distant, est de rediriger le navigateur vers l'URL publique. Ce serait
gratuit en bande passante, et ce serait un piege — le magasin ne controle pas
`Content-Disposition`, donc l'utilisateur enregistrerait un fichier nomme
`abc123def456.pdf` au lieu du titre de la loi. `_download_filename` a ete ecrite
exactement contre ce defaut.

Le second est `test_magasin_en_panne_donne_503_et_non_404` : un 404 ferait
disparaitre le document de l'interface au lieu de signaler une panne passagere.

AUCUN RESEAU REEL : `respx` intercepte httpx et compte les appels.

Usage:
    pytest tests/test_api/test_law_files_distant.py -v
"""

import hashlib
import io
from urllib.parse import unquote

import httpx
import pytest
import respx
from pypdf import PdfWriter

from app.core.config import settings
from app.models.law import Law
from app.services import document_storage

BASE = "https://magasin.test/documents"
FILE_ID = "distant0001abcdef"
URL_PDF = f"{BASE}/{FILE_ID}.pdf"


def _pdf_valide() -> bytes:
    """PDF reellement valide : /pdf-info l'ouvre avec PdfReader."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    tampon = io.BytesIO()
    writer.write(tampon)
    return tampon.getvalue()


PDF = _pdf_valide()


@pytest.fixture
def magasin(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DOCUMENTS_BASE_URL", BASE)
    monkeypatch.setattr(settings, "DOCUMENTS_CACHE_DIR", str(tmp_path / "cache"))
    document_storage._verrous.clear()
    return tmp_path / "cache"


@pytest.fixture
async def loi_distante(db_session):
    """Une loi dont le fichier n'existe QUE dans le magasin distant."""
    law = Law(
        reference="PRC-TEST-DISTANT",
        title="Décret de test avec fichier",
        content="Article 1. Contenu de test.",
        type="décret",
        language="fr",
        status="published",
        file_id=FILE_ID,
        original_filename="decret-original.pdf",
    )
    db_session.add(law)
    await db_session.commit()
    await db_session.refresh(law)
    return law


class TestTelechargement:
    @pytest.mark.asyncio
    @respx.mock
    async def test_sert_les_octets_de_l_amont(self, client, magasin, loi_distante):
        respx.get(URL_PDF).mock(return_value=httpx.Response(200, content=PDF))

        r = await client.get(f"/api/v1/laws/{loi_distante.id}/download")

        assert r.status_code == 200
        assert hashlib.sha256(r.content).hexdigest() == hashlib.sha256(PDF).hexdigest()

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_nom_reste_construit_sur_le_titre(self, client, magasin, loi_distante):
        """
        LE TEST QUI INTERDIT LA REDIRECTION. Voir le docstring du module : une
        302 vers le magasin perdrait le nom construit sur le titre.
        """
        respx.get(URL_PDF).mock(return_value=httpx.Response(200, content=PDF))

        r = await client.get(f"/api/v1/laws/{loi_distante.id}/download")

        disposition = unquote(r.headers["content-disposition"])
        assert disposition.startswith("attachment")
        assert "Décret de test avec fichier.pdf" in disposition
        assert "decret-original.pdf" not in disposition
        assert FILE_ID not in disposition, "l'identifiant interne a fuite dans le nom"

    @pytest.mark.asyncio
    @respx.mock
    async def test_deux_appels_ne_tirent_le_document_qu_une_fois(
        self, client, magasin, loi_distante
    ):
        route = respx.get(URL_PDF).mock(return_value=httpx.Response(200, content=PDF))

        await client.get(f"/api/v1/laws/{loi_distante.id}/download")
        await client.get(f"/api/v1/laws/{loi_distante.id}/download")

        assert route.call_count == 1, "le cache ne sert a rien"


class TestInformationsPdf:
    @pytest.mark.asyncio
    @respx.mock
    async def test_compte_les_pages_depuis_l_amont(self, client, magasin, loi_distante):
        respx.get(URL_PDF).mock(return_value=httpx.Response(200, content=PDF))

        r = await client.get(f"/api/v1/laws/{loi_distante.id}/pdf-info")

        assert r.status_code == 200
        assert r.json()["page_count"] == 1


class TestPannesEtRefus:
    @pytest.mark.asyncio
    @respx.mock
    async def test_magasin_en_panne_donne_503_et_non_404(self, client, magasin, loi_distante):
        """Un 404 accuserait l'utilisateur d'un defaut qui est le notre."""
        respx.get(URL_PDF).mock(return_value=httpx.Response(502))

        r = await client.get(f"/api/v1/laws/{loi_distante.id}/download")

        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "30"

    @pytest.mark.asyncio
    @respx.mock
    async def test_document_absent_du_magasin_donne_404(self, client, magasin, loi_distante):
        respx.get(url__startswith=BASE).mock(return_value=httpx.Response(404))

        r = await client.get(f"/api/v1/laws/{loi_distante.id}/download")

        assert r.status_code == 404

    @pytest.mark.parametrize("file_id", ["../../../etc/passwd", "sous/dossier/fichier", "a@evil"])
    @pytest.mark.asyncio
    @respx.mock
    async def test_file_id_douteux_donne_404_sans_aucune_requete(
        self, client, db_session, magasin, file_id
    ):
        """
        Le pendant, au niveau HTTP, de la garde SSRF de document_storage.

        L'assertion qui compte n'est pas le 404 — c'est `call_count == 0`. Une
        garde placee trop tard rendrait le meme 404 apres avoir deja contacte
        l'hote arbitraire.
        """
        route = respx.get(url__startswith="http").mock(return_value=httpx.Response(200))
        law = Law(
            reference=f"PRC-TEST-{abs(hash(file_id))}",
            title="Loi au file_id douteux",
            content="Article 1.",
            type="loi",
            language="fr",
            status="published",
            file_id=file_id,
        )
        db_session.add(law)
        await db_session.commit()
        await db_session.refresh(law)

        r = await client.get(f"/api/v1/laws/{law.id}/download")

        assert r.status_code == 404
        assert route.call_count == 0, "une requete est partie vers un hote arbitraire"
