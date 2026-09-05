"""
Les routes qui ecrivent doivent exiger un compte administrateur.

Trois portes etaient ouvertes, constatees par appel direct au serveur :

- `POST /api/v1/upload` acceptait 50 Mo de n'importe qui, sans jeton, et les
  ecrivait sur le disque.
- Les quatre routes d'analytics repondaient 200 sans jeton et exposaient la
  composition du corpus.
- Le WebSocket d'import acceptait la connexion sans jeton (101 verifie), la ou
  son jumeau HTTP rend 401.

Ce fichier couvre aussi les cinq routes d'ECRITURE dont depend l'interface
d'administration et qui n'avaient AUCUN test : upload, ingest, batch-upload, et
la modification et la suppression de comptes.

Usage:
    pytest tests/test_api/test_ecritures_protegees.py -v
"""

import io

import pytest
from httpx import AsyncClient


def _pdf_minimal() -> bytes:
    """Un PDF valide et minuscule, suffisant pour franchir la validation."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    tampon = io.BytesIO()
    writer.write(tampon)
    return tampon.getvalue()


class TestUploadProtege:
    @pytest.mark.asyncio
    async def test_sans_jeton_c_est_401(self, client: AsyncClient):
        """N'importe qui pouvait ecrire 50 Mo sur le disque du serveur."""
        reponse = await client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", _pdf_minimal(), "application/pdf")},
        )

        assert reponse.status_code == 401

    @pytest.mark.asyncio
    async def test_avec_un_jeton_admin_l_upload_aboutit(
        self, admin_client: AsyncClient, db_session
    ):
        reponse = await admin_client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", _pdf_minimal(), "application/pdf")},
        )

        assert reponse.status_code == 201, reponse.text
        assert reponse.json()["file_id"]


class TestIngestionProtegee:
    @pytest.mark.asyncio
    async def test_ingest_sans_jeton_c_est_401(self, client: AsyncClient):
        reponse = await client.post(
            "/api/v1/laws/admin/ingest",
            json={"file_id": "inexistant", "original_filename": "x.pdf", "title": "Titre"},
        )

        assert reponse.status_code == 401

    @pytest.mark.asyncio
    async def test_batch_upload_sans_jeton_c_est_401(self, client: AsyncClient):
        reponse = await client.post(
            "/api/v1/batch-upload/upload",
            files={"files": ("test.pdf", _pdf_minimal(), "application/pdf")},
        )

        assert reponse.status_code == 401


class TestComptesProteges:
    @pytest.mark.asyncio
    async def test_modification_sans_jeton_c_est_401(self, client: AsyncClient):
        reponse = await client.put("/api/v1/admin/users/1", json={"full_name": "X"})

        assert reponse.status_code == 401

    @pytest.mark.asyncio
    async def test_suppression_sans_jeton_c_est_401(self, client: AsyncClient):
        assert (await client.delete("/api/v1/admin/users/1")).status_code == 401

    @pytest.mark.asyncio
    async def test_un_admin_modifie_un_compte(self, admin_client: AsyncClient, test_user):
        reponse = await admin_client.put(
            f"/api/v1/admin/users/{test_user.id}", json={"full_name": "Nom modifie"}
        )

        assert reponse.status_code == 200
        assert reponse.json()["full_name"] == "Nom modifie"

    @pytest.mark.asyncio
    async def test_un_admin_simple_ne_supprime_PAS_un_compte(
        self, admin_client: AsyncClient, test_user
    ):
        """
        La suppression exige le role superadmin. Regle deliberee, et jusqu'ici
        non testee : rien n'empechait de la relacher par inadvertance.
        """
        reponse = await admin_client.delete(f"/api/v1/admin/users/{test_user.id}")

        assert reponse.status_code == 403

    @pytest.mark.asyncio
    async def test_un_superadmin_supprime_un_compte(
        self, superadmin_client: AsyncClient, test_user
    ):
        reponse = await superadmin_client.delete(f"/api/v1/admin/users/{test_user.id}")

        assert reponse.status_code in (200, 204), reponse.text
        relecture = await superadmin_client.get("/api/v1/admin/users")
        assert test_user.id not in [u["id"] for u in relecture.json()]


class TestWebSocketProtege:
    """
    La poignee de main renvoyait 101 sans le moindre jeton, alors que
    `GET /batch-upload/status` rend 401.
    """

    def test_un_jeton_absent_est_refuse(self):
        from app.api.routes.batch_upload import _jeton_valide

        assert _jeton_valide(None) is False
        assert _jeton_valide("") is False

    def test_un_jeton_illisible_est_refuse(self):
        from app.api.routes.batch_upload import _jeton_valide

        assert _jeton_valide("pas.un.jeton") is False

    def test_un_jeton_signe_ailleurs_est_refuse(self):
        """Une signature valide pour une AUTRE cle ne doit pas passer."""
        from jose import jwt

        from app.api.routes.batch_upload import _jeton_valide

        faux = jwt.encode({"sub": "pirate@example.com"}, "une-autre-cle", algorithm="HS256")

        assert _jeton_valide(faux) is False

    def test_un_jeton_valide_est_accepte(self):
        from app.api.routes.batch_upload import _jeton_valide
        from app.core.auth import create_access_token

        assert _jeton_valide(create_access_token({"sub": "admin@example.com"})) is True
