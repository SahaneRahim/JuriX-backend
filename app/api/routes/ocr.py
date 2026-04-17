"""
API endpoints for OCR operations.

Provides REST endpoints for:
- PDF type detection (POST /api/v1/ocr/detect)
- OCR processing (POST /api/v1/ocr/process)
- Health check (GET /api/v1/ocr/health)

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.ocr import (
    OCRRequest,
    OCRResult,
    OCRServiceHealth,
    PDFTypeDetectionRequest,
    PDFTypeDetectionResult,
)
from app.services.ocr_service import OCRError, OCRService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["OCR"])

# Initialize service (singleton pattern)
_ocr_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    """
    Get or create OCRService instance.
    
    Returns:
        OCRService instance
    """
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(
            default_language="fra+eng",
            default_dpi=300,
        )
    return _ocr_service


@router.post(
    "/detect",
    response_model=PDFTypeDetectionResult,
    summary="Detect PDF type",
    description="""
    Detect if a PDF is native (text-based), scanned (image-based), or hybrid.
    
    **Quick check** to determine if OCR processing is needed.
    
    **Returns:**
    - PDF type classification
    - Text coverage percentage
    - OCR recommendation
    """,
)
async def detect_pdf_type(request: PDFTypeDetectionRequest) -> PDFTypeDetectionResult:
    """
    Detect PDF type without processing.
    
    Args:
        request: Detection request with PDF path
        
    Returns:
        PDFTypeDetectionResult with classification
        
    Raises:
        400: Invalid PDF path
        404: PDF file not found
        500: Detection error
    """
    assert request is not None, "PDFTypeDetectionRequest must not be None"
    assert isinstance(request.pdf_path, str) and len(request.pdf_path) > 0, "pdf_path must be a non-empty string"

    logger.info(f"🔍 PDF type detection request: {request.pdf_path}")
    
    pdf_path = Path(request.pdf_path)
    
    # Validate path
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {request.pdf_path}",
        )
    
    try:
        service = get_ocr_service()
        result = await service.detect_pdf_type(pdf_path)
        
        logger.info(f"✅ Detection complete: {result.pdf_type.value}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF type detection failed: {str(e)}",
        )


@router.post(
    "/process",
    response_model=OCRResult,
    summary="Process PDF with OCR",
    description="""
    Process a PDF document with OCR if needed.
    
    **Workflow:**
    1. Auto-detect PDF type
    2. Extract text (native or OCR)
    3. Clean and post-process text
    
    **Performance:** 5-30s depending on scan quality and page count.
    
    **Languages:** French (fra), English (eng), or both (fra+eng)
    """,
)
async def process_pdf(request: OCRRequest) -> OCRResult:
    """
    Process PDF with OCR.
    
    Args:
        request: OCR request with PDF path and options
        
    Returns:
        OCRResult with extracted text
        
    Raises:
        400: Invalid request
        404: PDF file not found
        500: Processing error
    """
    assert request is not None, "OCRRequest must not be None"
    assert isinstance(request.pdf_path, str) and len(request.pdf_path) > 0, "pdf_path must be a non-empty string"

    logger.info(
        f"📄 OCR processing request: {request.pdf_path} "
        f"(lang: {request.language}, dpi: {request.dpi})"
    )
    
    pdf_path = Path(request.pdf_path)
    
    # Validate path
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {request.pdf_path}",
        )
    
    try:
        service = get_ocr_service()
        result = await service.process_pdf(
            pdf_path=pdf_path,
            language=request.language,
            dpi=request.dpi,
            timeout_seconds=request.timeout_seconds,
        )
        
        logger.info(
            f"✅ OCR complete: {result.text_length} chars "
            f"({result.processing_time_ms}ms)"
        )
        return result
        
    except OCRError as e:
        logger.warning(f"⚠️  OCR error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"❌ Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        )


@router.get(
    "/health",
    response_model=OCRServiceHealth,
    summary="Service health check",
    description="Check the health status of the OCR service including Tesseract availability.",
)
async def health_check() -> OCRServiceHealth:
    """
    Check service health.
    
    Returns:
        Health status including Tesseract and Poppler availability
    """
    service = get_ocr_service()
    health = service.health_check()
    
    return OCRServiceHealth(**health)
