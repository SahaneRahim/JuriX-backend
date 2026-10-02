"""
POST /api/v1/upload/cleanup ne doit retirer que les fichiers orphelins.

Il supprimait tout fichier de plus de 24 h dans data/uploads, y compris le PDF
de chaque loi publiee, que `laws.file_id` designe : l'affichage du document et
toute re-extraction cassaient au premier nettoyage.

Usage:
    pytest tests/test_api/test_nettoyage_uploads.py -v
"""

import os
from datetime import datetime, timedelta

import pytest

from app.api.routes import upload as route_upload
from app.models.law import Law
from app.services.file_upload_service import FileUploadService


@pytest.fixture
def stockage(tmp_path, monkeypatch):
    service = FileUploadService(
        storage_path=str(tmp_path),
        max_size_mb=50,
        allowed_formats=("pdf", "docx"),
        clamav_enabled=False,
        cleanup_hours=24,
    )
    monkeypatch.setattr(route_upload, "_upload_service", service)
    return tmp_path


def _vieux_fichier(dossier, nom):
    chemin = dossier / nom
    chemin.write_bytes(b"%PDF-1.4\n")
    vieux = (datetime.now() - timedelta(hours=48)).timestamp()
    os.utime(chemin, (vieux, vieux))
    return chemin


@pytest.mark.asyncio
async def test_le_pdf_d_une_loi_survit_au_nettoyage(admin_client, db_session, stockage):
    db_session.add(Law(
        reference="LOI-NETTOYAGE", title="Loi au fichier conserve", content="Contenu.",
        type="loi", language="fr", status="published", file_id="f0a1b2c3d4",
    ))
    await db_session.commit()
    reference = _vieux_fichier(stockage, "f0a1b2c3d4.pdf")
    orphelin = _vieux_fichier(stockage, "9e8d7c6b5a.pdf")

    reponse = await admin_client.post("/api/v1/upload/cleanup")

    assert reponse.status_code == 200
    assert reference.exists(), "le PDF d'une loi publiee a ete supprime"
    assert not orphelin.exists()
    assert reponse.json()["deleted_count"] == 1
