"""
Tests unitaires pour le service OCRService.

Ces tests vérifient:
- Détection type PDF (natif, scanné, hybride)
- Traitement OCR (Tesseract)
- Nettoyage de texte
- Support multi-langue
- Gestion d'erreurs

Usage:
    pytest backend/tests/test_services/test_ocr_service.py -v
    pytest backend/tests/test_services/test_ocr_service.py -v --cov=backend/app/services/ocr_service
"""

import tempfile
from pathlib import Path

import pytest

from app.schemas.ocr import PDFType
from app.services.ocr_service import OCRError, OCRService
from app.utils.ocr_utils import (
    assess_text_quality,
    clean_ocr_text,
    has_embedded_text,
    merge_text_blocks,
)


@pytest.fixture
def ocr_service():
    """Fixture: OCRService instance."""
    return OCRService(default_language="fra+eng", default_dpi=300)


@pytest.fixture
def sample_native_pdf(tmp_path):
    """Fixture: Create a minimal native PDF with text."""
    pdf_path = tmp_path / "native.pdf"
    # Minimal valid PDF with text.
    # Note: str puis .encode(), et non un litteral b"..." — le flux contient des
    # caracteres accentues, interdits dans un litteral bytes (SyntaxError qui
    # faisait echouer la collecte de TOUTE la suite de tests).
    pdf_content = """%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Article 1. La présente loi régit les sociétés commerciales.) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000262 00000 n 
0000000341 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
435
%%EOF""".encode("latin-1")
    pdf_path.write_bytes(pdf_content)
    return pdf_path


# ==================== PDF TYPE DETECTION TESTS ====================


class TestPDFTypeDetection:
    """Tests de détection de type PDF."""

    @pytest.mark.asyncio
    async def test_detect_native_pdf(self, ocr_service, sample_native_pdf):
        """Test détection d'un PDF natif avec texte."""
        result = await ocr_service.detect_pdf_type(sample_native_pdf)
        
        assert result.pdf_type == PDFType.NATIVE
        assert result.has_text is True
        assert result.text_coverage > 0.5
        assert result.needs_ocr is False

    @pytest.mark.asyncio
    async def test_detect_scanned_pdf_mock(self, ocr_service, tmp_path):
        """Test détection d'un PDF scanné (simulation)."""
        # Create minimal PDF without text
        pdf_path = tmp_path / "scanned.pdf"
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
        
        result = await ocr_service.detect_pdf_type(pdf_path)
        
        # Should detect as scanned or unknown (no text)
        assert result.has_text is False
        assert result.text_coverage < 0.2

    @pytest.mark.asyncio
    async def test_detect_empty_pdf(self, ocr_service, tmp_path):
        """Test détection d'un PDF vide."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [] /Count 0 >>
endobj
xref
0 3
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
trailer
<< /Size 3 /Root 1 0 R >>
startxref
109
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        
        # Should handle gracefully
        with pytest.raises(Exception):  # May raise on empty PDF
            await ocr_service.detect_pdf_type(pdf_path)

    @pytest.mark.asyncio
    async def test_detect_corrupted_pdf(self, ocr_service, tmp_path):
        """Test gestion d'un PDF corrompu."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"Not a valid PDF")
        
        with pytest.raises(OCRError):
            await ocr_service.detect_pdf_type(pdf_path)


# ==================== OCR PROCESSING TESTS ====================


class TestOCRProcessing:
    """Tests de traitement OCR."""

    @pytest.mark.asyncio
    async def test_process_native_pdf(self, ocr_service, sample_native_pdf):
        """Test traitement d'un PDF natif (pas d'OCR nécessaire)."""
        result = await ocr_service.process_pdf(sample_native_pdf)
        
        assert result.pdf_type == PDFType.NATIVE
        assert len(result.text) > 0
        assert "Article 1" in result.text or "loi" in result.text.lower()
        assert result.processing_time_ms > 0
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_ocr_timeout_handling(self, ocr_service, sample_native_pdf):
        """Test gestion des timeouts."""
        # Avec un timeout très court, devrait quand même fonctionner pour PDF natif
        result = await ocr_service.process_pdf(
            sample_native_pdf, timeout_seconds=1
        )
        
        assert result.text_length > 0


# ==================== TEXT CLEANING TESTS ====================


class TestTextCleaning:
    """Tests de nettoyage de texte."""

    @pytest.mark.asyncio
    async def test_clean_extra_spaces(self, ocr_service):
        """Test normalisation des espaces."""
        dirty_text = "Article   1.  La    présente  loi"
        cleaned = await ocr_service.clean_ocr_text(dirty_text)
        
        assert "  " not in cleaned  # No double spaces
        assert cleaned.count(" ") < dirty_text.count(" ")

    def test_clean_broken_words(self):
        """Test reconstruction de mots cassés."""
        text = "res pon sable"
        cleaned = clean_ocr_text(text)
        
        # Should attempt to fix broken words
        assert len(cleaned) <= len(text)

    def test_clean_special_characters(self):
        """Test correction de caractères spéciaux."""
        text = "l'article « test » — suite"
        cleaned = clean_ocr_text(text)
        
        assert cleaned is not None
        assert len(cleaned) > 0

    def test_preserve_structure(self):
        """Test préservation de la structure."""
        text = "Article 1.\nLa présente loi.\n\nArticle 2.\nSuite."
        cleaned = clean_ocr_text(text)
        
        # Should preserve paragraph breaks
        assert "Article 1" in cleaned
        assert "Article 2" in cleaned


# ==================== PERFORMANCE TESTS ====================


class TestPerformance:
    """Tests de performance."""

    @pytest.mark.asyncio
    async def test_processing_time_acceptable(self, ocr_service, sample_native_pdf):
        """Test que le traitement est rapide pour PDF natif."""
        result = await ocr_service.process_pdf(sample_native_pdf)
        
        # Native PDF should be fast (<1s)
        assert result.processing_time_ms < 1000


# ==================== ERROR HANDLING TESTS ====================


class TestErrorHandling:
    """Tests de gestion d'erreurs."""

    @pytest.mark.asyncio
    async def test_invalid_pdf_format(self, ocr_service, tmp_path):
        """Test gestion d'un format invalide."""
        invalid_path = tmp_path / "invalid.txt"
        invalid_path.write_text("Not a PDF")
        
        with pytest.raises(OCRError):
            await ocr_service.process_pdf(invalid_path)

    def test_tesseract_availability_check(self, ocr_service):
        """Test vérification disponibilité Tesseract."""
        # Should not crash
        available = ocr_service.tesseract_available
        assert isinstance(available, bool)


# ==================== UTILITY FUNCTIONS TESTS ====================


class TestUtilityFunctions:
    """Tests des fonctions utilitaires."""

    def test_has_embedded_text(self, sample_native_pdf):
        """Test détection de texte embarqué."""
        has_text, coverage = has_embedded_text(sample_native_pdf)
        
        assert has_text is True
        assert 0.0 <= coverage <= 1.0

    def test_merge_text_blocks(self):
        """Test fusion de blocs de texte."""
        blocks = ["Article 1. Text.", "Article 2. More text."]
        merged = merge_text_blocks(blocks)
        
        assert "Article 1" in merged
        assert "Article 2" in merged
        assert len(merged) > 0

    def test_assess_text_quality_good(self):
        """Test évaluation qualité texte (bon)."""
        good_text = "Article 1. La présente loi régit les sociétés commerciales."
        quality = assess_text_quality(good_text)
        
        assert quality > 0.7  # Good quality

    def test_assess_text_quality_poor(self):
        """Test évaluation qualité texte (mauvais)."""
        poor_text = "@#$% !!! ??? ### ***"
        quality = assess_text_quality(poor_text)
        
        assert quality < 0.5  # Poor quality

    def test_clean_ocr_text_empty(self):
        """Test nettoyage texte vide."""
        cleaned = clean_ocr_text("")
        assert cleaned == ""

    def test_clean_ocr_text_whitespace(self):
        """Test nettoyage espaces uniquement."""
        cleaned = clean_ocr_text("   \n\t   ")
        assert cleaned == ""


# ==================== HEALTH CHECK TESTS ====================


class TestHealthCheck:
    """Tests du health check."""

    def test_health_check_returns_status(self, ocr_service):
        """Test que le health check retourne un statut."""
        health = ocr_service.health_check()
        
        assert "service" in health
        assert "status" in health
        assert "tesseract" in health
        assert "poppler" in health
        
        assert health["service"] == "OCRService"
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_tesseract_status(self, ocr_service):
        """Test vérification statut Tesseract."""
        health = ocr_service.health_check()
        
        tesseract = health["tesseract"]
        assert "available" in tesseract
        assert isinstance(tesseract["available"], bool)


# ==================== INTEGRATION TESTS ====================


class TestIntegration:
    """Tests d'intégration."""

    @pytest.mark.asyncio
    async def test_full_workflow_native_pdf(self, ocr_service, sample_native_pdf):
        """Test workflow complet pour PDF natif."""
        # 1. Detect type
        detection = await ocr_service.detect_pdf_type(sample_native_pdf)
        assert detection.pdf_type == PDFType.NATIVE
        
        # 2. Process
        result = await ocr_service.process_pdf(sample_native_pdf)
        assert result.text_length > 0
        assert result.pdf_type == PDFType.NATIVE
        
        # 3. Verify quality
        quality = assess_text_quality(result.text)
        assert quality > 0.5
