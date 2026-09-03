"""
API routes for Laws management.

Endpoints:
- GET /api/v1/laws - List laws with filters
- GET /api/v1/laws/{id} - Get law detail with articles
- POST /api/v1/admin/laws - Create law (admin only)
- PUT /api/v1/admin/laws/{id} - Update law (admin only)
- DELETE /api/v1/admin/laws/{id} - Delete law (admin only)

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import time
from typing import List, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import get_current_admin_user
from app.models.user import User
from app.core.database import get_db, AsyncSessionLocal
from app.models.law import Category, Law
from app.schemas.law import (
    LawCreate,
    LawResponse,
    LawUpdate,
)
from app.services.file_upload_service import get_upload_service
from app.services.search_service import invalidate_search_cache
from app.tasks.process_law import delete_from_search_index


class LawIngestRequest(BaseModel):
    file_id: str
    title: Optional[str] = None
    original_filename: Optional[str] = None
    reference: Optional[str] = None  # Optional custom reference
    category_id: Optional[int] = None  # Optional category ID


logger = logging.getLogger(__name__)

# Références fortes vers les tâches de fond (cf. create_task plus bas).
_background_tasks: set = set()

router = APIRouter(tags=["Laws"])


# ==================== DEPENDENCIES ====================


# L'authentification reelle vit dans app/core/auth.py (JWT + bcrypt + roles).
# Un stub renvoyant {"id": 1, "role": "admin"} occupait cette place : les quatre
# endpoints d'administration ci-dessous etaient donc ouverts a tous.
# get_current_admin_user est importe en tete de fichier.


# ==================== PUBLIC ENDPOINTS ====================


@router.get("/", response_model=List[LawResponse])
async def get_laws(
    language: Optional[str] = Query(None, description="Filter by language (fr or en)"),
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    law_status: str = Query(
        "published", description="Filter by status (published, draft, or archived)"
    ),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get list of laws with optional filters.

    **Filters:**
    - `language`: Filter by language code (fr or en)
    - `category_id`: Filter by category ID
    - `law_status`: Filter by status (active or archived)

    **Pagination:**
    - `skip`: Number of records to skip (default: 0)
    - `limit`: Maximum records to return (default: 50, max: 100)

    **Example:**
    ```
    GET /api/v1/laws?language=fr&category_id=1&skip=0&limit=10
    ```

    Returns:
        List of laws matching the filters
    """
    logger.info(
        f"📋 GET /laws - language={language}, category_id={category_id}, "
        f"status={law_status}, skip={skip}, limit={limit}"
    )

    try:
        # Build query
        query = select(Law).options(selectinload(Law.articles), selectinload(Law.category))

        # Apply filters
        if language:
            query = query.where(Law.language == language)

        if category_id:
            # Filter by category ID directly
            query = query.where(Law.category_id == category_id)

        if law_status:
            query = query.where(Law.status == law_status)

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        laws = result.scalars().all()

        logger.info(f"✅ Found {len(laws)} laws")
        return laws

    except Exception as e:
        logger.error(f"❌ Error fetching laws: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching laws"
        )


@router.get("/{law_id}", response_model=LawResponse)
async def get_law(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get law detail by ID with articles.

    **Includes:**
    - Law metadata
    - All articles (ordered)
    - Categories
    - Auto-detection results (language, category suggestions)

    Args:
        law_id: Law ID

    Returns:
        Law detail with articles

    Raises:
        404: Law not found
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"

    logger.info(f"📄 GET /laws/{law_id}")

    try:
        # Query law with relationships
        query = (
            select(Law)
            .options(selectinload(Law.articles), selectinload(Law.category))
            .where(Law.id == law_id)
        )

        result = await db.execute(query)
        law = result.scalar_one_or_none()

        if not law:
            logger.warning(f"⚠️  Law {law_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Law with ID {law_id} not found"
            )

        logger.info(f"✅ Law {law_id} found: {law.title}")
        return law

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching law"
        )


@router.get("/{law_id}/download")
async def download_law_file(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Download the original uploaded file for a law."""
    query = select(Law).where(Law.id == law_id)
    result = await db.execute(query)
    law = result.scalar_one_or_none()

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    upload_service = get_upload_service()
    
    # Try to find file with extension
    file_path = None
    # Use stored filename or construct one
    filename = law.original_filename or f"{law.reference}.pdf"
    
    # Check for PDF or DOCX using file_id
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{law.file_id}{ext}"
        if p.exists():
            file_path = p
            break
            
    if not file_path:
         # Fallback: maybe file_id IS the filename (legacy behavior?)
         # Or maybe stored without extension? 
         p = upload_service.storage_path / law.file_id
         if p.exists():
             file_path = p
         else:
             raise HTTPException(status_code=404, detail="File not found on server")

    return FileResponse(
        path=str(file_path), 
        filename=str(cast(str, law.original_filename) or file_path.name),
        media_type="application/pdf" if file_path.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        content_disposition_type="inline"
    )


@router.get("/{law_id}/pdf-data")
async def get_law_pdf_data(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Return PDF as Base64-encoded JSON to bypass download manager interception.
    This allows JavaScript to fetch the PDF without IDM/browser extensions intercepting the request.
    """
    import base64
    
    query = select(Law).where(Law.id == law_id)
    result = await db.execute(query)
    law = result.scalar_one_or_none()

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    from app.services.file_upload_service import get_upload_service
    upload_service = get_upload_service()
    
    # Find the file
    file_path = None
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{law.file_id}{ext}"
        if p.exists():
            file_path = p
            break
            
    if not file_path:
        p = upload_service.storage_path / law.file_id
        if p.exists():
            file_path = p
        else:
            raise HTTPException(status_code=404, detail="File not found on server")

    # Read file and encode to Base64
    with open(str(file_path), "rb") as f:
        file_bytes = f.read()
    
    base64_data = base64.b64encode(file_bytes).decode("utf-8")
    
    return {
        "filename": law.original_filename or file_path.name,
        "content_type": "application/pdf" if file_path.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "data": base64_data
    }


@router.post("/{law_id}/pdf-stream")
async def get_law_pdf_stream(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream PDF as binary data for efficient loading of large files.
    
    Uses POST method to bypass download managers (IDM) which typically
    only intercept GET requests. Returns PDF as binary with special headers.
    
    This is more efficient than base64 encoding for large files
    (saves ~33% bandwidth and avoids JavaScript decode overhead).
    """
    from fastapi.responses import Response
    
    query = select(Law).where(Law.id == law_id)
    result = await db.execute(query)
    law = result.scalar_one_or_none()

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    from app.services.file_upload_service import get_upload_service
    upload_service = get_upload_service()
    
    # Find the file
    file_path = None
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{law.file_id}{ext}"
        if p.exists():
            file_path = p
            break
            
    if not file_path:
        p = upload_service.storage_path / law.file_id
        if p.exists():
            file_path = p
        else:
            raise HTTPException(status_code=404, detail="File not found on server")

    # Read file content
    with open(str(file_path), "rb") as f:
        file_bytes = f.read()
    
    # Return as binary stream with headers that prevent IDM interception
    # Using POST method + octet-stream prevents IDM from intercepting
    return Response(
        content=file_bytes,
        media_type="application/octet-stream",  # Not application/pdf - hides from IDM
        headers={
            "Content-Disposition": "inline",
            "X-Content-Type-Options": "nosniff",
            "X-PDF-Content": "true",  # Custom header to identify as PDF
            "Cache-Control": "private, max-age=3600",
        }
    )


@router.get("/{law_id}/pdf-info")
async def get_law_pdf_info(
    law_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get PDF information including page count.
    Used by the frontend to know how many pages to request.
    """
    from pypdf import PdfReader
    
    query = select(Law).where(Law.id == law_id)
    result = await db.execute(query)
    law = result.scalar_one_or_none()

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    from app.services.file_upload_service import get_upload_service
    upload_service = get_upload_service()
    
    # Find the file
    file_path = None
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{law.file_id}{ext}"
        if p.exists():
            file_path = p
            break
            
    if not file_path:
        p = upload_service.storage_path / law.file_id
        if p.exists():
            file_path = p
        else:
            raise HTTPException(status_code=404, detail="File not found on server")

    # Get page count
    try:
        reader = PdfReader(str(file_path))
        page_count = len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")
    
    return {
        "law_id": law_id,
        "filename": law.original_filename or file_path.name,
        "page_count": page_count
    }


@router.get("/{law_id}/page/{page_num}")
async def get_law_pdf_page_image(
    law_id: int,
    page_num: int,
    dpi: int = Query(default=120, ge=72, le=200, description="DPI for rendering (72-200)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Render a specific PDF page as a JPEG image.
    
    Uses Poppler (pdf2image) for reliable rendering of all image formats
    including JPEG2000 which pdf.js cannot handle.
    
    Args:
        law_id: ID of the law
        page_num: Page number (1-indexed)
        dpi: Resolution for rendering (default 120, max 200 for performance)
    
    Returns:
        JPEG image of the specified page
    """
    from io import BytesIO
    from fastapi.responses import Response
    
    query = select(Law).where(Law.id == law_id)
    result = await db.execute(query)
    law = result.scalar_one_or_none()

    if not law:
        raise HTTPException(status_code=404, detail="Law not found")

    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    from app.services.file_upload_service import get_upload_service
    upload_service = get_upload_service()
    
    # Find the file
    file_path = None
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{law.file_id}{ext}"
        if p.exists():
            file_path = p
            break
            
    if not file_path:
        p = upload_service.storage_path / law.file_id
        if p.exists():
            file_path = p
        else:
            raise HTTPException(status_code=404, detail="File not found on server")

    # Convert page to image using Poppler
    try:
        from pdf2image import convert_from_path
        
        # Convert only the specific page (1-indexed)
        images = convert_from_path(
            str(file_path),
            first_page=page_num,
            last_page=page_num,
            dpi=dpi,
            fmt="jpeg",
        )
        
        if not images:
            raise HTTPException(status_code=404, detail=f"Page {page_num} not found")
        
        # Convert to JPEG bytes
        img_buffer = BytesIO()
        images[0].save(img_buffer, format="JPEG", quality=85)
        img_bytes = img_buffer.getvalue()
        
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                "X-Page-Number": str(page_num),
            }
        )
        
    except Exception as e:
        logger.error(f"Error rendering page {page_num} for law {law_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error rendering page: {str(e)}")


# ==================== INGESTION ENDPOINTS ====================


@router.post("/admin/ingest", response_model=LawResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_law(
    request: LawIngestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Ingest a new law from an uploaded file.

    1. Creates a Law record (status=processing)
    2. Declenche le traitement du fichier en tache de fond

    Args:
        request: Ingest request (file_id, title)

    Returns:
        Created Law object (pending)
    """
    assert request is not None, "LawIngestRequest must not be None"
    assert isinstance(request.file_id, str) and len(request.file_id) > 0, "file_id must be a non-empty string"

    logger.info(f"📥 POST /admin/laws/ingest - File: {request.file_id}")

    try:
        # Validate file_id
        if not request.file_id or len(request.file_id) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file_id: must be at least 8 characters",
            )

        # Check if law with this file_id already exists (idempotency)
        ref_prefix = f"PENDING-{request.file_id[:8]}"

        # Check for existing pending law to avoid duplicates
        query = select(Law).where(Law.reference == ref_prefix)
        result = await db.execute(query)
        existing_law = result.scalar_one_or_none()

        if existing_law and existing_law.status == "processing":
            logger.info(f"ℹ️ Law already exists for file {request.file_id}: ID={existing_law.id}")
            return existing_law

        # If existing law failed, create a new one with timestamp
        if existing_law:
            ref_prefix = f"{ref_prefix}-{int(time.time())}"

        # Use custom reference if provided, otherwise use PENDING prefix
        law_reference = request.reference if request.reference else ref_prefix

        # Check if reference already exists (collision check)
        existing_ref_query = select(Law).where(Law.reference == law_reference)
        existing_ref_result = await db.execute(existing_ref_query)
        existing_ref_law = existing_ref_result.scalar_one_or_none()

        if existing_ref_law:
            logger.info(f"ℹ️ Law with reference '{law_reference}' already exists: ID={existing_ref_law.id}. Returning existing law.")
            
            # If it was stuck or previous attempt failed, we might want to trigger processing again
            # For now, just return it so the frontend sees it's there
            if existing_ref_law.status == "processing" or existing_ref_law.content == "Document en cours de traitement par le système.":
                 # Potentially restart failed processing here if needed
                 pass
                 
            return existing_ref_law

        # Create initial Law record
        new_law = Law(
            reference=law_reference,
            title=request.title or "Document en cours de traitement",
            type="autre",  # Changed from "unknown" to valid type
            content="Document en cours de traitement par le système.",  # Min 10 chars required
            # "processing" et non "published" : le document n'a pas encore ete
            # extrait. Le publier d'emblee le rendait visible dans le corpus
            # public avec son contenu de remplacement ("Document en cours de
            # traitement par le systeme."), et un echec d'OCR l'y laissait
            # indefiniment. Le pipeline le passe a "published" en cas de succes,
            # a "refused" en cas d'echec.
            status="processing",
            category_id=request.category_id,  # Category selected by admin
            file_id=request.file_id,
            original_filename=request.original_filename,
        )

        db.add(new_law)
        await db.commit()
        await db.refresh(new_law)

        # Lance le traitement en arrière-plan (BackgroundTasks FastAPI)
        # Traitement dans une tache asyncio, sans courtier de messages
        # sans bloquer la réponse HTTP
        law_id_for_bg = cast(int, new_law.id)
        file_id_for_bg = request.file_id

        async def _process_and_invalidate():
            from app.tasks.process_law import process_law_async
            try:
                result = await process_law_async(law_id_for_bg, file_id_for_bg)
                logger.info(f"✅ Background processing completed: {result}")
                # Invalider le cache après traitement
                async with AsyncSessionLocal() as cache_db:
                    await invalidate_search_cache(cache_db)
            except Exception as bg_err:
                logger.error(f"❌ Background processing failed: {bg_err}", exc_info=True)

        import asyncio as _asyncio

        # Reference forte conservee : asyncio ne garde qu'une reference FAIBLE
        # sur les taches, une tache non referencee peut etre collectee en plein
        # traitement et le document rester bloque en "processing".
        _task = _asyncio.create_task(_process_and_invalidate())
        _background_tasks.add(_task)
        _task.add_done_callback(_background_tasks.discard)
        logger.info(f"🚀 Background task started for Law ID={new_law.id}")

        return new_law

    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        await db.rollback()
        error_msg = f"Error starting ingestion: {type(e).__name__}: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)


# ==================== ADMIN ENDPOINTS ====================


@router.post("/admin", response_model=LawResponse, status_code=status.HTTP_201_CREATED)
async def create_law(
    law: LawCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Create a new law (admin only).

    **Auto-Detection (v2.1):**
    - If `language` not provided, auto-detected from content
    - Category suggestions provided based on content

    **Required Fields:**
    - reference: Unique law reference
    - title: Law title
    - type: Law type (loi, décret, ordonnance, etc.)
    - content: Full text content

    Args:
        law: Law creation data

    Returns:
        Created law with auto-detection results

    Raises:
        400: Validation error
        401: Unauthorized
        409: Law reference already exists
    """
    assert law is not None, "LawCreate must not be None"
    assert isinstance(law.reference, str) and len(law.reference) > 0, "Law reference must be a non-empty string"

    logger.info(f"➕ POST /admin/laws - Creating law: {law.title}")

    try:
        # Check if reference already exists
        existing_query = select(Law).where(Law.reference == law.reference)
        existing_result = await db.execute(existing_query)
        existing_law = existing_result.scalar_one_or_none()

        if existing_law:
            logger.warning(f"⚠️  Law reference {law.reference} already exists")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Law with reference '{law.reference}' already exists",
            )

        # Create law instance
        new_law = Law(**law.model_dump())

        # TODO: Integrate auto-detection services
        # if not new_law.language:
        #     language_service = LanguageDetector()
        #     detection = language_service.detect(new_law.content)
        #     new_law.detected_language = detection["language"]
        #     new_law.language_confidence = detection["confidence"]

        # Add to database
        db.add(new_law)
        await db.commit()
        await db.refresh(new_law)

        logger.info(f"✅ Law created: ID={new_law.id}, reference={new_law.reference}")
        return new_law

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Error creating law: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error creating law"
        )


@router.put("/admin/{law_id}", response_model=LawResponse)
async def update_law(
    law_id: int,
    law_update: LawUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Update an existing law (admin only).

    **Partial Updates:**
    - Only provided fields are updated
    - Omitted fields remain unchanged

    Args:
        law_id: Law ID to update
        law_update: Fields to update

    Returns:
        Updated law

    Raises:
        404: Law not found
        401: Unauthorized
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
    assert law_update is not None, "LawUpdate must not be None"

    logger.info(f"✏️  PUT /admin/laws/{law_id}")

    try:
        # Fetch existing law
        query = select(Law).where(Law.id == law_id)
        result = await db.execute(query)
        existing_law = result.scalar_one_or_none()

        if not existing_law:
            logger.warning(f"⚠️  Law {law_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Law with ID {law_id} not found"
            )

        # Update fields
        update_data = law_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing_law, field, value)

        await db.commit()
        await db.refresh(existing_law)

        logger.info(f"✅ Law {law_id} updated")
        return existing_law

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Error updating law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error updating law"
        )


@router.delete("/admin/{law_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_law(
    law_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
):
    """
    Delete a law (admin only).

    **Cascade Delete:**
    - Deletes law and all associated articles
    - Removes from search index

    Args:
        law_id: Law ID to delete

    Returns:
        204 No Content on success

    Raises:
        404: Law not found
        401: Unauthorized
    """
    logger.info(f"🗑️  DELETE /admin/laws/{law_id}")

    try:
        # Fetch law
        query = select(Law).where(Law.id == law_id)
        result = await db.execute(query)
        law = result.scalar_one_or_none()

        if not law:
            logger.warning(f"⚠️  Law {law_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Law with ID {law_id} not found"
            )

        # Delete law (cascade deletes articles)
        await db.delete(law)
        await db.commit()

        # Vide le search_vector PostgreSQL (remplace suppression la recherche plein texte)
        delete_from_search_index(law_id)

        # Invalider le cache de recherche
        await invalidate_search_cache(db)
        
        logger.info(f"✅ Law {law_id} deleted")
        return None

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Error deleting law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error deleting law"
        )
