"""
Tests des endpoints qui servent le fichier d'origine d'une loi.

Le bouton "Telecharger" du front appelait window.print(), qui imprime la page
affichee — donc l'unique article lu, jamais le document. L'endpoint qui sert le
vrai fichier existait mais n'etait appele par personne. Ces tests fixent son
contrat : le fichier rendu est l'ORIGINAL, octet pour octet, et il est propose
en telechargement et non affiche dans l'onglet.

Ils couvrent aussi la resolution du chemin, qui etait recopiee a la main dans
cinq endpoints avec un repli joignant la valeur de base au repertoire d'upload
sans aucune verification.
"""

import hashlib

import pytest

from app.models.law import Law
from app.services.file_upload_service import get_upload_service


def _valid_pdf_bytes() -> bytes:
    """
    PDF d'une page, REELLEMENT valide.

    Ecrit par pypdf plutot qu'a la main : /pdf-info ouvre le fichier avec
    PdfReader, et un squelette PDF sans table xref le fait echouer — le test
    mesurerait alors la qualite de sa propre doublure.
    """
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


MINIMAL_PDF = _valid_pdf_bytes()


@pytest.fixture
async def law_with_file(db_session):
    """Une loi dont le PDF est reellement pose dans le repertoire d'upload."""
    file_id = "testfileid0001abcd"
    storage = get_upload_service().storage_path
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{file_id}.pdf"
    path.write_bytes(MINIMAL_PDF)

    law = Law(
        reference="PRC-TEST-FILE",
        title="Décret de test avec fichier",
        content="Article 1. Contenu de test.",
        type="décret",
        language="fr",
        status="published",
        file_id=file_id,
        original_filename="decret-original.pdf",
    )
    db_session.add(law)
    await db_session.commit()
    await db_session.refresh(law)

    yield law

    path.unlink(missing_ok=True)


class TestDownload:

    @pytest.mark.asyncio
    async def test_serves_the_original_bytes(self, client, law_with_file):
        response = await client.get(f"/api/v1/laws/{law_with_file.id}/download")

        assert response.status_code == 200
        assert response.content == MINIMAL_PDF
        assert hashlib.sha256(response.content).hexdigest() == (
            hashlib.sha256(MINIMAL_PDF).hexdigest()
        )

    @pytest.mark.asyncio
    async def test_is_offered_as_a_download_not_displayed(self, client, law_with_file):
        """
        attachment et non inline : un bouton nomme "Telecharger" doit
        enregistrer le fichier, pas l'ouvrir dans l'onglet.
        """
        response = await client.get(f"/api/v1/laws/{law_with_file.id}/download")

        disposition = response.headers["content-disposition"]
        assert disposition.startswith("attachment")
        assert response.headers["content-type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_filename_is_built_on_the_title(self, client, law_with_file):
        """
        Le nom propose vient du TITRE, pas de original_filename.

        original_filename est le nom du fichier aspire depuis prc.cm —
        "10691_decret-n-2026-164-du-4-mai-2026-portant-..." — dont le prefixe
        numerique est l'identifiant interne du site source.
        """
        from urllib.parse import unquote

        response = await client.get(f"/api/v1/laws/{law_with_file.id}/download")

        disposition = unquote(response.headers["content-disposition"])
        assert "Décret de test avec fichier.pdf" in disposition
        assert "decret-original.pdf" not in disposition

    @pytest.mark.asyncio
    async def test_law_without_file_gives_404(self, client, db_session):
        law = Law(
            reference="PRC-TEST-NOFILE",
            title="Loi créée sans téléversement",
            content="Article 1. Contenu.",
            type="loi",
            language="fr",
            status="published",
        )
        db_session.add(law)
        await db_session.commit()
        await db_session.refresh(law)

        response = await client.get(f"/api/v1/laws/{law.id}/download")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_law_gives_404(self, client):
        assert (await client.get("/api/v1/laws/999999/download")).status_code == 404


class TestPathResolution:
    """
    Le chemin passe par resolve_upload_path, qui refuse tout identifiant hors
    motif. Le repli precedent joignait la valeur brute au repertoire d'upload,
    sans motif ni confinement — cinq fois, une par endpoint.
    """

    @pytest.mark.parametrize("bad_file_id", [
        "../../../etc/passwd",
        "sous/dossier/fichier",
        "court",
        "",
    ])
    @pytest.mark.asyncio
    async def test_malformed_file_id_gives_404_not_500(
        self, client, db_session, bad_file_id
    ):
        law = Law(
            reference=f"PRC-BAD-{abs(hash(bad_file_id)) % 10000}",
            title="Loi au file_id douteux",
            content="Article 1. Contenu.",
            type="loi",
            language="fr",
            status="published",
            file_id=bad_file_id or None,
        )
        db_session.add(law)
        await db_session.commit()
        await db_session.refresh(law)

        for endpoint in ("download", "pdf-data", "pdf-info"):
            response = await client.get(f"/api/v1/laws/{law.id}/{endpoint}")
            assert response.status_code == 404, (
                f"{endpoint} devrait repondre 404 pour file_id={bad_file_id!r}, "
                f"pas {response.status_code}"
            )

    @pytest.mark.asyncio
    async def test_all_file_endpoints_agree_on_the_same_file(self, client, law_with_file):
        """Les cinq endpoints resolvent le meme chemin, donc le meme fichier."""
        download = await client.get(f"/api/v1/laws/{law_with_file.id}/download")
        info = await client.get(f"/api/v1/laws/{law_with_file.id}/pdf-info")

        assert download.status_code == 200
        assert info.status_code == 200
        assert info.json()["filename"] == "decret-original.pdf"
