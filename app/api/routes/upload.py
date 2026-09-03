"""
API endpoints for file upload operations.

Provides REST endpoints for:
- File upload (POST /api/v1/upload)
- Upload status (GET /api/v1/upload/{file_id})
- File deletion (DELETE /api/v1/upload/{file_id})
- Health check (GET /api/v1/upload/health)

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
from pathlib import Path
from typing import Dict

from fastapi import Depends, APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.file_upload import FileUploadResult, UploadServiceHealth
from app.services.file_upload_service import FileUploadError, FileUploadService
from app.core.auth import get_current_admin_user
from app.models.user import User

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["File Upload"])

# Initialize service (singleton pattern)
# In production, this should be dependency-injected
_upload_service: FileUploadService | None = None


def get_upload_service() -> FileUploadService:
    """
    Get or create FileUploadService instance.

    Returns:
        FileUploadService instance
    """
    global _upload_service
    if _upload_service is None:
        # Ensure storage directory exists
        storage_path = Path("./data/uploads")
        storage_path.mkdir(parents=True, exist_ok=True)

        # TODO: Load from environment variables
        _upload_service = FileUploadService(
            storage_path="./data/uploads",
            max_size_mb=250,
            allowed_formats=("pdf", "docx"),
            clamav_enabled=False,  # Mock by default
            cleanup_hours=24,
        )
    return _upload_service


@router.post(
    "",
    response_model=FileUploadResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
    description="""
    Upload a PDF or DOCX file for processing.

    **Validations:**
    - Maximum file size: 50 MB
    - Allowed formats: PDF, DOCX
    - File structure validation
    - Antivirus scan

    **Returns:**
    - File ID for tracking
    - Validation results
    - Extracted metadata
    - Scan results
    """,
)
async def upload_file(
    file: UploadFile = File(..., description="File to upload (PDF or DOCX, max 50 MB)"),
) -> FileUploadResult:
    """
    Upload a file with validation and scanning.

    Args:
        file: Uploaded file

    Returns:
        FileUploadResult with complete upload information

    Raises:
        400: Invalid file format or validation failed
        413: File too large
        500: Internal server error
    """
    assert file is not None, "UploadFile must not be None"
    assert file.filename is not None and len(file.filename) > 0, "Filename must not be empty"

    logger.info(f"📤 Upload request: {file.filename} ({file.content_type})")

    try:
        service = get_upload_service()
        result = await service.upload_file(file)

        logger.info(f"✅ Upload successful: {result.file_id}")
        return result

    except FileUploadError as e:
        logger.warning(f"⚠️  Upload validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during file upload",
        )


# NOTE: declaree avant "/{file_id}" — FastAPI resout dans l'ordre de
# declaration, la route parametree capturait sinon ce chemin.
@router.get(
    "/health",
    response_model=UploadServiceHealth,
    summary="Service health check",
    description="Check the health status of the file upload service.",
)
async def health_check() -> UploadServiceHealth:
    """
    Check service health.

    Returns:
        Health status including storage and scanner information
    """
    service = get_upload_service()
    health = service.health_check()

    return UploadServiceHealth(**health)


@router.get(
    "/{file_id}",
    response_model=Dict,
    summary="Get upload status",
    description="Retrieve information about an uploaded file by its ID.",
)
async def get_upload_status(file_id: str) -> Dict:
    """
    Get status and metadata of an uploaded file.

    Args:
        file_id: File identifier

    Returns:
        File information

    Raises:
        404: File not found
    """
    assert isinstance(file_id, str) and len(file_id) > 0, "file_id must be a non-empty string"

    service = get_upload_service()

    # Check if file exists
    file_path = service.storage_path / f"{file_id}.pdf"
    if not file_path.exists():
        file_path = service.storage_path / f"{file_id}.docx"

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_id}",
        )

    # Return basic file info
    return {
        "file_id": file_id,
        "exists": True,
        "file_size": file_path.stat().st_size,
        "storage_path": str(file_path),
    }


@router.delete(
    "/{file_id}",
    response_model=Dict,
    summary="Delete uploaded file",
    description="Delete an uploaded file from temporary storage.",
)
async def delete_upload(
    file_id: str,
    current_admin: User = Depends(get_current_admin_user),
) -> Dict:
    """
    Delete an uploaded file.

    Args:
        file_id: File identifier

    Returns:
        Success confirmation

    Raises:
        404: File not found
    """
    assert isinstance(file_id, str) and len(file_id) > 0, "file_id must be a non-empty string"

    service = get_upload_service()

    # Try to find and delete file
    deleted = False
    for ext in [".pdf", ".docx"]:
        file_path = service.storage_path / f"{file_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            deleted = True
            logger.info(f"🗑️  Deleted file: {file_id}{ext}")
            break

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_id}",
        )

    return {"success": True, "file_id": file_id, "message": "File deleted successfully"}




@router.post(
    "/cleanup",
    response_model=Dict,
    summary="Cleanup old files",
    description="Remove files older than specified hours (default: 24h).",
)
async def cleanup_old_files(
    max_age_hours: int = 24,
    current_admin: User = Depends(get_current_admin_user),
) -> Dict:
    """
    Cleanup old uploaded files.

    Args:
        max_age_hours: Maximum age in hours (default: 24)

    Returns:
        Cleanup statistics
    """
    assert isinstance(max_age_hours, int) and max_age_hours > 0, "max_age_hours must be a positive integer"

    service = get_upload_service()
    stats = await service.cleanup_old_files(max_age_hours)

    logger.info(f"🧹 Cleanup completed: {stats['deleted_count']} files deleted")
    return stats
