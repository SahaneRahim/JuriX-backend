"""
Pydantic schemas for file upload operations.

Provides request/response models with validation for:
- File upload results
- File metadata (PDF, DOCX)
- Validation results
- Antivirus scan results

Author: JuriX Development Team
Date: 2026-01-11
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# File Upload Result Schemas
# ============================================================================


class FileMetadata(BaseModel):
    """
    Metadata extracted from uploaded files.
    
    Fields vary by file type:
    - PDF: title, author, creation_date, page_count
    - DOCX: title, author, creation_date, word_count
    """

    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    creation_date: Optional[datetime] = Field(None, description="Document creation date")
    page_count: Optional[int] = Field(None, ge=0, description="Number of pages (PDF only)")
    word_count: Optional[int] = Field(None, ge=0, description="Number of words (DOCX only)")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Loi N°2024-015",
                "author": "République du Cameroun",
                "creation_date": "2024-01-15T10:30:00",
                "page_count": 45,
                "file_size": 2048576,
                "file_hash": "a" * 64,
            }
        }


class ValidationResult(BaseModel):
    """
    Result of file validation checks.
    
    Validates:
    - File format (PDF, DOCX)
    - File size (max 50 MB)
    - File structure integrity
    - MIME type
    """

    is_valid: bool = Field(..., description="Whether file passed all validation checks")
    format: Optional[str] = Field(None, description="Detected file format (pdf or docx)")
    mime_type: str = Field(..., description="Detected MIME type")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")

    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "format": "pdf",
                "mime_type": "application/pdf",
                "errors": [],
                "warnings": [],
            }
        }


class ScanResult(BaseModel):
    """
    Result of antivirus scan.
    
    Supports:
    - ClamAV (production)
    - Mock scanner (development)
    """

    is_clean: bool = Field(..., description="Whether file is clean (no threats detected)")
    scanner: str = Field(..., description="Scanner used (clamav or mock)")
    scan_time_ms: int = Field(..., ge=0, description="Scan duration in milliseconds")
    threat_found: Optional[str] = Field(None, description="Name of threat if detected")
    scanner_version: Optional[str] = Field(None, description="Scanner version")

    @field_validator("scanner")
    @classmethod
    def validate_scanner(cls, v: str) -> str:
        """Validate scanner type."""
        allowed_scanners = {"clamav", "mock"}
        if v.lower() not in allowed_scanners:
            raise ValueError(f"Scanner must be one of: {', '.join(allowed_scanners)}. Got: {v}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "is_clean": True,
                "scanner": "mock",
                "scan_time_ms": 150,
                "threat_found": None,
                "scanner_version": "1.0.0",
            }
        }


class FileUploadResult(BaseModel):
    """
    Complete result of file upload operation.
    
    Includes:
    - File identification
    - Validation results
    - Scan results
    - Extracted metadata
    - Storage information
    """

    file_id: str = Field(..., description="Unique file identifier (UUID)")
    filename: str = Field(..., min_length=1, description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="Detected MIME type")
    file_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash")
    
    # Nested results
    metadata: FileMetadata = Field(..., description="Extracted file metadata")
    validation: ValidationResult = Field(..., description="Validation results")
    scan_result: ScanResult = Field(..., description="Antivirus scan results")
    
    # Storage info
    storage_path: str = Field(..., description="Relative path to stored file")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp for temp files")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "loi_2024_015.pdf",
                "file_size": 2048576,
                "mime_type": "application/pdf",
                "file_hash": "a" * 64,
                "metadata": {
                    "title": "Loi N°2024-015",
                    "author": "République du Cameroun",
                    "page_count": 45,
                    "file_size": 2048576,
                    "file_hash": "a" * 64,
                },
                "validation": {
                    "is_valid": True,
                    "format": "pdf",
                    "mime_type": "application/pdf",
                    "errors": [],
                    "warnings": [],
                },
                "scan_result": {
                    "is_clean": True,
                    "scanner": "mock",
                    "scan_time_ms": 150,
                },
                "storage_path": "uploads/550e8400-e29b-41d4-a716-446655440000.pdf",
                "uploaded_at": "2024-01-15T10:30:00",
                "expires_at": "2024-01-16T10:30:00",
            }
        }


# ============================================================================
# Request Schemas
# ============================================================================


class FileUploadRequest(BaseModel):
    """
    Request schema for file upload (for documentation).
    
    Note: Actual upload uses multipart/form-data, not JSON.
    This schema is for OpenAPI documentation only.
    """

    file: str = Field(..., description="File to upload (multipart/form-data)")

    class Config:
        json_schema_extra = {
            "example": {
                "file": "binary file data",
            }
        }


# ============================================================================
# Health Check Schema
# ============================================================================


class UploadServiceHealth(BaseModel):
    """Health check response for upload service."""

    service: str = Field(default="FileUploadService", description="Service name")
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    storage: dict = Field(..., description="Storage status and statistics")
    scanner: dict = Field(..., description="Antivirus scanner status")
    timestamp: datetime = Field(..., description="Health check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "service": "FileUploadService",
                "status": "healthy",
                "storage": {
                    "path": "/app/data/uploads",
                    "exists": True,
                    "writable": True,
                    "total_files": 42,
                    "total_size_mb": 128.5,
                },
                "scanner": {
                    "type": "mock",
                    "available": True,
                    "version": "1.0.0",
                },
                "timestamp": "2024-01-15T10:30:00",
            }
        }
