"""
Pydantic schemas for OCR operations.

Provides request/response models with validation for:
- PDF type detection (native, scanned, hybrid)
- OCR processing results
- OCR configuration
- Health check

Author: JuriX Development Team
Date: 2026-01-11
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums
# ============================================================================


class PDFType(str, Enum):
    """Type of PDF document."""

    NATIVE = "native"  # Text-based PDF with embedded text
    SCANNED = "scanned"  # Image-based PDF requiring OCR
    HYBRID = "hybrid"  # Mixed content (text + images)
    UNKNOWN = "unknown"  # Unable to determine


# ============================================================================
# OCR Result Schemas
# ============================================================================


class OCRResult(BaseModel):
    """
    Result of OCR processing on a PDF.
    
    Contains:
    - PDF type classification
    - Extracted text (cleaned)
    - Processing metadata
    - Confidence scores
    """

    pdf_type: PDFType = Field(..., description="Detected PDF type")
    text: str = Field(..., description="Extracted and cleaned text")
    page_count: int = Field(..., ge=1, description="Number of pages processed")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    language: str = Field(..., description="OCR language used (e.g., 'fra+eng')")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="OCR confidence score (0-1)"
    )
    ocr_engine: str = Field(default="tesseract", description="OCR engine used")
    text_length: int = Field(..., ge=0, description="Length of extracted text")

    class Config:
        json_schema_extra = {
            "example": {
                "pdf_type": "scanned",
                "text": "Article 1. La présente loi régit...",
                "page_count": 45,
                "processing_time_ms": 15000,
                "language": "fra+eng",
                "confidence": 0.92,
                "ocr_engine": "tesseract",
                "text_length": 25000,
            }
        }


class PDFTypeDetectionResult(BaseModel):
    """
    Result of PDF type detection only (no OCR).
    
    Quick check to determine if OCR is needed.
    """

    pdf_type: PDFType = Field(..., description="Detected PDF type")
    has_text: bool = Field(..., description="Whether PDF has embedded text")
    has_images: bool = Field(..., description="Whether PDF contains images")
    page_count: int = Field(..., ge=1, description="Number of pages")
    text_coverage: float = Field(
        ..., ge=0.0, le=1.0, description="Percentage of pages with text (0-1)"
    )
    needs_ocr: bool = Field(..., description="Whether OCR processing is recommended")

    class Config:
        json_schema_extra = {
            "example": {
                "pdf_type": "scanned",
                "has_text": False,
                "has_images": True,
                "page_count": 45,
                "text_coverage": 0.0,
                "needs_ocr": True,
            }
        }


# ============================================================================
# Request Schemas
# ============================================================================


class OCRRequest(BaseModel):
    """
    Request schema for OCR processing.
    """

    pdf_path: str = Field(..., min_length=1, description="Path to PDF file")
    language: str = Field(
        default="fra+eng",
        description="Tesseract language codes (e.g., 'fra', 'eng', 'fra+eng')",
    )
    dpi: int = Field(
        default=300, ge=72, le=600, description="DPI for PDF to image conversion"
    )
    timeout_seconds: int = Field(
        default=60, ge=5, le=300, description="Maximum processing time per page"
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language codes."""
        allowed_languages = {"fra", "eng", "fra+eng", "eng+fra"}
        if v.lower() not in allowed_languages:
            raise ValueError(
                f"Language must be one of: {', '.join(allowed_languages)}. Got: {v}"
            )
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "pdf_path": "/path/to/document.pdf",
                "language": "fra+eng",
                "dpi": 300,
                "timeout_seconds": 60,
            }
        }


class PDFTypeDetectionRequest(BaseModel):
    """Request schema for PDF type detection only."""

    pdf_path: str = Field(..., min_length=1, description="Path to PDF file")

    class Config:
        json_schema_extra = {"example": {"pdf_path": "/path/to/document.pdf"}}


# ============================================================================
# Health Check Schema
# ============================================================================


class OCRServiceHealth(BaseModel):
    """Health check response for OCR service."""

    service: str = Field(default="OCRService", description="Service name")
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    tesseract: dict = Field(..., description="Tesseract OCR status")
    poppler: dict = Field(..., description="Poppler (pdf2image) status")
    supported_languages: list = Field(
        default_factory=list, description="Available Tesseract languages"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "service": "OCRService",
                "status": "healthy",
                "tesseract": {
                    "available": True,
                    "version": "5.3.0",
                    "path": "/usr/bin/tesseract",
                },
                "poppler": {"available": True, "version": "22.12.0"},
                "supported_languages": ["fra", "eng", "osd"],
            }
        }


# ============================================================================
# Statistics Schema
# ============================================================================


class OCRStatistics(BaseModel):
    """Statistics for OCR processing."""

    total_pages_processed: int = Field(0, ge=0)
    total_processing_time_ms: int = Field(0, ge=0)
    average_time_per_page_ms: float = Field(0.0, ge=0.0)
    total_text_extracted: int = Field(0, ge=0, description="Total characters extracted")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Success rate (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "total_pages_processed": 450,
                "total_processing_time_ms": 180000,
                "average_time_per_page_ms": 400,
                "total_text_extracted": 1250000,
                "success_rate": 0.98,
            }
        }
