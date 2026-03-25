"""
Service OCR pour extraction de texte depuis PDFs scannés.

Ce service gère:
- Détection type PDF (natif vs scanné vs hybride)
- OCR avec Tesseract (FR + EN)
- Post-processing et nettoyage du texte
- Gestion performance et timeouts

Objectif: Extraction texte fiable en 5-30s selon qualité scan.

Usage:
    service = OCRService()
    result = await service.process_pdf(pdf_path, language="fra+eng")
    # {'pdf_type': 'scanned', 'text': '...', 'confidence': 0.92}
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.ocr import (
    OCRResult,
    PDFType,
    PDFTypeDetectionResult,
)
from app.utils.ocr_utils import (
    assess_text_quality,
    clean_ocr_text,
    convert_pdf_to_images,
    estimate_ocr_time,
    has_embedded_text,
    has_images,
    merge_text_blocks,
)

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Exception levée lors d'erreurs OCR."""

    pass


class OCRService:
    """
    Service d'extraction de texte par OCR.

    Fonctionnalités:
    - Détection automatique type PDF
    - OCR Tesseract multi-langue (FR, EN)
    - Nettoyage et post-processing
    - Gestion timeouts et performance

    Attributes:
        tesseract_path: Chemin vers Tesseract (auto-détecté)
        tesseract_available: Si Tesseract est disponible
        default_language: Langue par défaut
        default_dpi: DPI par défaut
    """

    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        default_language: str = "fra+eng",
        default_dpi: int = 300,
    ):
        """
        Initialise le service OCR.

        Args:
            tesseract_path: Chemin vers Tesseract (optionnel, auto-détecté)
            default_language: Langue par défaut (défaut: fra+eng)
            default_dpi: DPI par défaut (défaut: 300)

        Raises:
            OCRError: Si Tesseract requis mais non disponible
        """
        logger.info("🚀 Initialisation du OCRService...")

        self.default_language = default_language
        self.default_dpi = default_dpi
        self.tesseract_path = tesseract_path

        # Vérifier disponibilité Tesseract
        self.tesseract_available = self._check_tesseract()

        if self.tesseract_available:
            logger.info(f"  ✅ Tesseract disponible: {self._get_tesseract_version()}")
        else:
            logger.warning("  ⚠️  Tesseract non disponible, mode mock activé")

        logger.info(
            f"✅ OCRService initialisé "
            f"(langue: {self.default_language}, DPI: {self.default_dpi})"
        )

    def _check_tesseract(self) -> bool:
        """Vérifie si Tesseract est disponible."""
        try:
            import pytesseract

            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

            # Test rapide
            version = pytesseract.get_tesseract_version()
            return version is not None

        except Exception as e:
            logger.debug(f"Tesseract check failed: {e}")
            return False

    def _get_tesseract_version(self) -> str:
        """Récupère la version de Tesseract."""
        try:
            import pytesseract

            version = pytesseract.get_tesseract_version()
            return str(version)
        except Exception:
            return "unknown"

    async def detect_pdf_type(self, pdf_path: Path) -> PDFTypeDetectionResult:
        """
        Détecte le type de PDF (natif, scanné, hybride).

        Args:
            pdf_path: Chemin vers le PDF

        Returns:
            PDFTypeDetectionResult avec classification

        Example:
            >>> result = await service.detect_pdf_type(Path("doc.pdf"))
            >>> result.pdf_type
            PDFType.SCANNED
            >>> result.needs_ocr
            True
        """
        logger.info(f"🔍 Détection type PDF: {pdf_path.name}")

        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)

            # Vérifier texte embarqué
            has_text, text_coverage = has_embedded_text(pdf_path)

            # Vérifier images
            has_imgs = has_images(pdf_path)

            # Classifier le PDF
            # Classifier le PDF - MODE AGRESSIF
            # User wants "Brutal" extraction. 
            # If text_coverage is NOT perfect (100%), we default to HYBRID/OCR to catch stamps, side-notes, etc.
            # But scanning huge PDFs is slow.
            # Compromise: High threshold.
            if text_coverage >= 0.98:
                # Almost certainly native text throughout
                pdf_type = PDFType.NATIVE
                needs_ocr = False
            else:
                # Even if 80% coverage (0.8), we trigger OCR to be safe ("Brutal")
                # This covers "Valid text BUT header/footer is image"
                # And also covers SCANNED (coverage < 0.2)
                pdf_type = PDFType.HYBRID
                needs_ocr = True

            logger.info(
                f"  ✅ Type détecté: {pdf_type.value} "
                f"(texte: {text_coverage:.0%}, images: {has_imgs})"
            )

            return PDFTypeDetectionResult(
                pdf_type=pdf_type,
                has_text=has_text,
                has_images=has_imgs,
                page_count=page_count,
                text_coverage=text_coverage,
                needs_ocr=needs_ocr,
            )

        except Exception as e:
            logger.error(f"❌ Erreur détection type PDF: {e}")
            raise OCRError(f"Échec détection type PDF: {e}") from e

    async def process_pdf(
        self,
        pdf_path: Path,
        language: Optional[str] = None,
        dpi: Optional[int] = None,
        timeout_seconds: int = 60,
    ) -> OCRResult:
        """
        Traite un PDF avec OCR si nécessaire.

        Workflow:
        1. Détecte type PDF
        2. Si scanné/hybride: OCR
        3. Si natif: extraction texte direct
        4. Nettoyage et post-processing

        Args:
            pdf_path: Chemin vers le PDF
            language: Langue OCR (défaut: fra+eng)
            dpi: DPI conversion (défaut: 300)
            timeout_seconds: Timeout par page (défaut: 60s)

        Returns:
            OCRResult avec texte extrait

        Raises:
            OCRError: Si traitement échoue

        Example:
            >>> result = await service.process_pdf(Path("scan.pdf"))
            >>> result.text[:50]
            'Article 1. La présente loi régit...'
        """
        assert pdf_path is not None, "pdf_path must not be None"
        assert timeout_seconds > 0, "timeout_seconds must be positive"

        start_time = time.time()
        language = language or self.default_language
        dpi = dpi or self.default_dpi

        logger.info(f"📄 Traitement PDF: {pdf_path.name} (langue: {language})")

        # 1. Détecter type PDF
        detection = await self.detect_pdf_type(pdf_path)
        logger.debug(f"  Type: {detection.pdf_type.value}, OCR requis: {detection.needs_ocr}")

        # 2. Extraire texte selon type
        if detection.pdf_type == PDFType.NATIVE:
            # Extraction directe
            text = await self._extract_native_text(pdf_path)
            confidence = 1.0  # Texte natif = confiance maximale
        elif detection.needs_ocr:
            # OCR requis
            if not self.tesseract_available:
                raise OCRError(
                    "Tesseract non disponible. OCR requis mais impossible."
                )
            text = await self.extract_text_ocr(
                pdf_path, language, dpi, timeout_seconds
            )
            confidence = assess_text_quality(text)
        else:
            # Fallback
            text = await self._extract_native_text(pdf_path)
            confidence = 0.5

        # 3. Nettoyage
        cleaned_text = await self.clean_ocr_text(text)

        # Calculer temps de traitement
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"✅ Traitement terminé: {len(cleaned_text)} caractères "
            f"({processing_time_ms}ms, confiance: {confidence:.2%})"
        )

        return OCRResult(
            pdf_type=detection.pdf_type,
            text=cleaned_text,
            page_count=detection.page_count,
            processing_time_ms=processing_time_ms,
            language=language,
            confidence=confidence,
            ocr_engine="tesseract" if self.tesseract_available else "native",
            text_length=len(cleaned_text),
        )

    async def _extract_native_text(self, pdf_path: Path) -> str:
        """Extrait texte d'un PDF natif."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            text_blocks = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    # Add page marker for article extraction
                    text_blocks.append(f"<<PAGE:{page_num}>>\n{text}")

            merged = merge_text_blocks(text_blocks)
            logger.debug(f"  Texte natif extrait: {len(merged)} caractères")

            return merged

        except Exception as e:
            logger.error(f"Erreur extraction texte natif: {e}")
            raise OCRError(f"Échec extraction texte: {e}") from e

    async def extract_text_ocr(
        self,
        pdf_path: Path,
        language: str = "fra+eng",
        dpi: int = 300,
        timeout_seconds: int = 60,
    ) -> str:
        """
        Extrait texte via OCR Tesseract.

        Args:
            pdf_path: Chemin vers le PDF
            language: Langue Tesseract
            dpi: DPI conversion
            timeout_seconds: Timeout par page

        Returns:
            Texte extrait

        Raises:
            OCRError: Si OCR échoue
        """
        logger.info(f"🔍 OCR en cours: {pdf_path.name} ({language}, {dpi} DPI)")

        try:
            import pytesseract

            # 1. Convertir PDF en images
            images = convert_pdf_to_images(pdf_path, dpi=dpi)
            logger.debug(f"  {len(images)} pages converties en images")

            # 2. OCR sur chaque page (parallèle)
            text_blocks = []

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []

                for i, image in enumerate(images):
                    future = executor.submit(
                        self._ocr_single_page, image, language, i + 1
                    )
                    futures.append(future)

                # Collecter résultats avec timeout
                for future in futures:
                    try:
                        page_text = future.result(timeout=timeout_seconds)
                        if page_text.strip():
                            text_blocks.append(page_text)
                    except FuturesTimeoutError:
                        logger.warning(f"  ⚠️  Timeout OCR page")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Erreur OCR page: {e}")

            # 3. Fusionner
            merged = merge_text_blocks(text_blocks)
            logger.info(f"  ✅ OCR terminé: {len(merged)} caractères extraits")

            return merged

        except ImportError as e:
            raise OCRError("pytesseract non disponible") from e
        except Exception as e:
            logger.error(f"❌ Erreur OCR: {e}")
            raise OCRError(f"Échec OCR: {e}") from e

    def _ocr_single_page(self, image, language: str, page_num: int) -> str:
        """OCR sur une seule page (méthode synchrone pour ThreadPoolExecutor)."""
        try:
            import pytesseract

            # Configuration Tesseract
            # Configuration Tesseract
            # psm 3: Fully automatic page segmentation, but no OSD. (Default)
            # psm 6: Assume a single uniform block of text.
            # psm 1: Auto + OSD (can be flaky with layout). 
            # We use psm 6 or 3 to try to keep blocks roughly in order.
            config = "--psm 3"

            text = pytesseract.image_to_string(image, lang=language, config=config)

            logger.debug(f"  Page {page_num}: {len(text)} caractères")
            # Add page marker for article extraction
            return f"<<PAGE:{page_num}>>\n{text}"

        except Exception as e:
            logger.warning(f"  Erreur OCR page {page_num}: {e}")
            return ""

    async def clean_ocr_text(self, text: str) -> str:
        """
        Nettoie et post-traite le texte OCR.

        NOTE: Not recursive — delegates to the imported utility function
        ``clean_ocr_text`` from ``app.utils.ocr_utils``, not to itself.

        Args:
            text: Texte brut OCR

        Returns:
            Texte nettoyé

        Example:
            >>> cleaned = await service.clean_ocr_text("Article   1.  Text...")
            >>> cleaned
            'Article 1. Text...'
        """
        return clean_ocr_text(text)

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du service.

        Returns:
            Dictionnaire avec statut de chaque composant
        """
        status = {
            "service": "OCRService",
            "status": "healthy",
            "tesseract": {},
            "poppler": {},
            "supported_languages": [],
        }

        # Vérifier Tesseract
        try:
            if self.tesseract_available:
                import pytesseract

                version = self._get_tesseract_version()
                langs = pytesseract.get_languages()

                status["tesseract"] = {
                    "available": True,
                    "version": version,
                    "path": str(pytesseract.pytesseract.tesseract_cmd),
                }
                status["supported_languages"] = langs
            else:
                status["tesseract"] = {"available": False}
                status["status"] = "degraded"

        except Exception as e:
            status["tesseract"] = {"available": False, "error": str(e)}
            status["status"] = "degraded"

        # Vérifier Poppler (pdf2image)
        try:
            from pdf2image import pdfinfo_from_path

            # Test simple
            status["poppler"] = {"available": True}

        except Exception as e:
            status["poppler"] = {"available": False, "error": str(e)}
            status["status"] = "degraded"

        return status
