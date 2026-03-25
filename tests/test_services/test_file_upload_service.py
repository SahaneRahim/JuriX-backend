"""
Tests unitaires pour le service FileUploadService.

Ces tests vérifient:
- Validation de fichiers (format, taille, structure)
- Scan antivirus (mock)
- Extraction de métadonnées (PDF, DOCX)
- Stockage et nettoyage
- Gestion d'erreurs

Usage:
    pytest backend/tests/test_services/test_file_upload_service.py -v
    pytest backend/tests/test_services/test_file_upload_service.py -v --cov=backend/app/services/file_upload_service
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.services.file_upload_service import FileUploadError, FileUploadService


@pytest.fixture
def temp_storage():
    """Fixture: Temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def upload_service(temp_storage):
    """Fixture: FileUploadService instance with temp storage."""
    return FileUploadService(
        storage_path=str(temp_storage),
        max_size_mb=50,
        allowed_formats=("pdf", "docx"),
        clamav_enabled=False,  # Use mock
        cleanup_hours=24,
    )


@pytest.fixture
def sample_pdf(tmp_path):
    """Fixture: Create a minimal valid PDF file."""
    pdf_path = tmp_path / "test.pdf"
    # Minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
214
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_docx(tmp_path):
    """Fixture: Create a minimal valid DOCX file."""
    docx_path = tmp_path / "test.docx"
    
    # Create a minimal DOCX using python-docx
    from docx import Document
    
    doc = Document()
    doc.add_paragraph("Test document content")
    doc.core_properties.title = "Test Document"
    doc.core_properties.author = "Test Author"
    doc.save(docx_path)
    
    return docx_path


@pytest.fixture
def sample_txt(tmp_path):
    """Fixture: Create a text file (invalid format)."""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("This is a text file, not PDF or DOCX")
    return txt_path


@pytest.fixture
def large_file(tmp_path):
    """Fixture: Create a file larger than 50 MB."""
    large_path = tmp_path / "large.pdf"
    # Create 51 MB file
    large_path.write_bytes(b"x" * (51 * 1024 * 1024))
    return large_path


# ==================== FILE VALIDATION TESTS ====================


class TestFileValidation:
    """Tests de validation de fichiers."""

    @pytest.mark.asyncio
    async def test_validate_pdf_success(self, upload_service, sample_pdf):
        """Test validation d'un PDF valide."""
        result = await upload_service.validate_file(sample_pdf, "test.pdf")
        
        assert result.is_valid is True
        assert result.format == "pdf"
        assert result.mime_type == "application/pdf"
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_docx_success(self, upload_service, sample_docx):
        """Test validation d'un DOCX valide."""
        result = await upload_service.validate_file(sample_docx, "test.docx")
        
        assert result.is_valid is True
        assert result.format == "docx"
        assert result.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_invalid_format_fails(self, upload_service, sample_txt):
        """Test qu'un fichier .txt est rejeté."""
        result = await upload_service.validate_file(sample_txt, "test.txt")
        
        assert result.is_valid is False
        assert result.format is None
        assert len(result.errors) > 0
        assert any("non reconnu" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_file_too_large_fails(self, upload_service, large_file):
        """Test qu'un fichier > 50 MB est rejeté."""
        result = await upload_service.validate_file(large_file, "large.pdf")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("50 MB" in error or "exceeds" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_corrupted_pdf_fails(self, upload_service, tmp_path):
        """Test qu'un PDF corrompu est détecté."""
        corrupted_pdf = tmp_path / "corrupted.pdf"
        # PDF avec magic bytes mais structure invalide
        corrupted_pdf.write_bytes(b"%PDF-1.4\nGarbage data without proper structure")
        
        result = await upload_service.validate_file(corrupted_pdf, "corrupted.pdf")
        
        assert result.is_valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_validate_empty_file_fails(self, upload_service, tmp_path):
        """Test qu'un fichier vide (0 bytes) est rejeté."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")
        
        result = await upload_service.validate_file(empty_file, "empty.pdf")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("empty" in error.lower() or "0 bytes" in error.lower() for error in result.errors)


# ==================== ANTIVIRUS SCANNING TESTS ====================


class TestAntivirusScanning:
    """Tests de scan antivirus."""

    @pytest.mark.asyncio
    async def test_scan_clean_file_passes(self, upload_service, sample_pdf):
        """Test qu'un fichier propre passe le scan."""
        result = await upload_service.scan_virus(sample_pdf)
        
        assert result.is_clean is True
        assert result.scanner == "mock"
        assert result.scan_time_ms > 0
        assert result.threat_found is None

    @pytest.mark.asyncio
    async def test_scan_mock_implementation(self, upload_service, sample_docx):
        """Test que le mock scanner fonctionne."""
        result = await upload_service.scan_virus(sample_docx)
        
        assert result.scanner == "mock"
        assert result.scanner_version == "1.0.0-dev"
        assert result.is_clean is True

    @pytest.mark.asyncio
    async def test_scan_timeout_handling(self, upload_service, sample_pdf):
        """Test que le scan gère les timeouts gracieusement."""
        # Le mock devrait toujours réussir rapidement
        result = await upload_service.scan_virus(sample_pdf)
        
        assert result.scan_time_ms < 5000  # Max 5 secondes
        assert result.is_clean is True


# ==================== METADATA EXTRACTION TESTS ====================


class TestMetadataExtraction:
    """Tests d'extraction de métadonnées."""

    @pytest.mark.asyncio
    async def test_extract_pdf_metadata(self, upload_service, sample_pdf):
        """Test extraction de métadonnées PDF."""
        metadata = await upload_service.extract_metadata(sample_pdf, "pdf")
        
        assert metadata.file_size > 0
        assert len(metadata.file_hash) == 64  # SHA-256
        assert metadata.page_count is not None
        assert metadata.word_count is None  # PDF n'a pas word_count

    @pytest.mark.asyncio
    async def test_extract_docx_metadata(self, upload_service, sample_docx):
        """Test extraction de métadonnées DOCX."""
        metadata = await upload_service.extract_metadata(sample_docx, "docx")
        
        assert metadata.file_size > 0
        assert len(metadata.file_hash) == 64  # SHA-256
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.word_count is not None
        assert metadata.word_count > 0
        assert metadata.page_count is None  # DOCX n'a pas page_count

    @pytest.mark.asyncio
    async def test_extract_metadata_missing_fields(self, upload_service, sample_pdf):
        """Test que les métadonnées manquantes sont gérées."""
        # Le PDF minimal n'a pas de titre/auteur
        metadata = await upload_service.extract_metadata(sample_pdf, "pdf")
        
        # Ces champs peuvent être None
        assert metadata.title is None or isinstance(metadata.title, str)
        assert metadata.author is None or isinstance(metadata.author, str)
        
        # Mais ces champs doivent toujours exister
        assert metadata.file_size > 0
        assert metadata.file_hash is not None

    @pytest.mark.asyncio
    async def test_file_hash_generation(self, upload_service, sample_pdf):
        """Test que le hash SHA-256 est calculé correctement."""
        metadata1 = await upload_service.extract_metadata(sample_pdf, "pdf")
        metadata2 = await upload_service.extract_metadata(sample_pdf, "pdf")
        
        # Le même fichier doit avoir le même hash
        assert metadata1.file_hash == metadata2.file_hash
        assert len(metadata1.file_hash) == 64


# ==================== STORAGE MANAGEMENT TESTS ====================


class TestStorageManagement:
    """Tests de gestion du stockage."""

    @pytest.mark.asyncio
    async def test_upload_file_success(self, upload_service, sample_pdf):
        """Test du workflow complet d'upload."""
        # Créer un UploadFile mock
        with open(sample_pdf, "rb") as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="test_document.pdf",
            file=open(sample_pdf, "rb")
        )
        
        try:
            result = await upload_service.upload_file(upload_file)
            
            assert result.file_id is not None
            assert result.filename == "test_document.pdf"
            assert result.file_size > 0
            assert result.validation.is_valid is True
            assert result.scan_result.is_clean is True
            assert result.metadata is not None
            assert result.uploaded_at is not None
            assert result.expires_at is not None
            
        finally:
            upload_file.file.close()

    @pytest.mark.asyncio
    async def test_cleanup_old_files(self, upload_service, temp_storage):
        """Test que les fichiers > 24h sont supprimés."""
        # Créer des fichiers avec différents âges
        old_file = temp_storage / "old_file.pdf"
        old_file.write_bytes(b"%PDF-1.4\nOld file")
        
        recent_file = temp_storage / "recent_file.pdf"
        recent_file.write_bytes(b"%PDF-1.4\nRecent file")
        
        # Modifier le timestamp du vieux fichier
        old_time = datetime.now() - timedelta(hours=25)
        old_timestamp = old_time.timestamp()
        import os
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Nettoyer les fichiers > 24h
        stats = await upload_service.cleanup_old_files(max_age_hours=24)
        
        assert stats["deleted_count"] >= 1
        assert not old_file.exists()
        assert recent_file.exists()

    @pytest.mark.asyncio
    async def test_unique_filename_generation(self, upload_service, sample_pdf):
        """Test qu'il n'y a pas de collisions de noms."""
        # Upload le même fichier 2 fois
        upload_file1 = UploadFile(
            filename="same_name.pdf",
            file=open(sample_pdf, "rb")
        )
        upload_file2 = UploadFile(
            filename="same_name.pdf",
            file=open(sample_pdf, "rb")
        )
        
        try:
            result1 = await upload_service.upload_file(upload_file1)
            upload_file1.file.close()
            upload_file1.file = open(sample_pdf, "rb")
            
            result2 = await upload_service.upload_file(upload_file2)
            
            # Les IDs doivent être différents
            assert result1.file_id != result2.file_id
            
        finally:
            upload_file1.file.close()
            upload_file2.file.close()


# ==================== ERROR HANDLING TESTS ====================


class TestErrorHandling:
    """Tests de gestion d'erreurs."""

    @pytest.mark.asyncio
    async def test_upload_with_invalid_file(self, upload_service, sample_txt):
        """Test qu'un fichier invalide lève FileUploadError."""
        upload_file = UploadFile(
            filename="invalid.txt",
            file=open(sample_txt, "rb")
        )
        
        try:
            with pytest.raises(FileUploadError) as exc_info:
                await upload_service.upload_file(upload_file)
            
            assert "Validation échouée" in str(exc_info.value)
            
        finally:
            upload_file.file.close()

    @pytest.mark.asyncio
    async def test_concurrent_uploads(self, upload_service, sample_pdf):
        """Test de thread safety avec uploads concurrents."""
        import asyncio
        
        async def upload_task(filename):
            upload_file = UploadFile(
                filename=filename,
                file=open(sample_pdf, "rb")
            )
            try:
                result = await upload_service.upload_file(upload_file)
                return result.file_id
            finally:
                upload_file.file.close()
        
        # Lancer 3 uploads en parallèle
        tasks = [
            upload_task(f"concurrent_{i}.pdf")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Tous les uploads doivent réussir avec des IDs uniques
        assert len(results) == 3
        assert len(set(results)) == 3  # Tous différents


# ==================== HEALTH CHECK TESTS ====================


class TestHealthCheck:
    """Tests du health check."""

    def test_health_check_returns_status(self, upload_service):
        """Test que le health check retourne un statut."""
        health = upload_service.health_check()
        
        assert "service" in health
        assert "status" in health
        assert "storage" in health
        assert "scanner" in health
        assert "timestamp" in health
        
        assert health["service"] == "FileUploadService"
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_storage_status(self, upload_service):
        """Test que le health check vérifie le stockage."""
        health = upload_service.health_check()
        
        storage = health["storage"]
        assert "path" in storage
        assert "exists" in storage
        assert "writable" in storage
        assert "total_files" in storage
        assert "total_size_mb" in storage

    def test_health_check_scanner_status(self, upload_service):
        """Test que le health check vérifie le scanner."""
        health = upload_service.health_check()
        
        scanner = health["scanner"]
        assert "type" in scanner
        assert "available" in scanner
        assert scanner["type"] == "mock"  # En mode développement
        assert scanner["available"] is True
