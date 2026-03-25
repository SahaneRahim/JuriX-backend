"""
Batch upload router with WebSocket support for real-time progress tracking.

Handles:
- Batch PDF upload (multiple files)
- Real-time progress updates via WebSocket
- Status tracking (PENDING → PROCESSING → PUBLISHED/REFUSED)
"""

from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List
import asyncio
import json
import uuid
from datetime import datetime

from app.core.database import get_db
from app.models.law import Law
from app.tasks.process_law import process_law_sync
from app.core.config import settings

router = APIRouter(prefix="/admin/batch", tags=["batch-upload"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_progress(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except:
                self.disconnect(session_id)

manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time upload progress.
    
    Client receives JSON messages:
    {
        "type": "progress",
        "law_id": 123,
        "filename": "law.pdf",
        "progress": 45,
        "status": "processing",
        "message": "Extracting text..."
    }
    """
    assert session_id and isinstance(session_id, str), "session_id must be a non-empty string"
    await manager.connect(session_id, websocket)
    MAX_KEEPALIVE_CYCLES = 3600  # ~1 hour at 1s keepalive cadence (NASA Rule 2: bounded loops)
    try:
        for _ in range(MAX_KEEPALIVE_CYCLES):
            # Keep connection alive, bounded to prevent infinite hang
            await websocket.receive_text()
        # Gracefully close after max cycles
        await websocket.close(code=1000, reason="Session timeout (max keepalive reached)")
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)


@router.post("/upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    session_id: str = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Batch upload multiple PDF files.
    
    Args:
        files: List of PDF files to upload
        session_id: WebSocket session ID for progress updates
        
    Returns:
        {
            "session_id": "uuid",
            "total_files": 10,
            "created_laws": [{"id": 1, "filename": "law1.pdf"}, ...]
        }
    """
    assert files is not None and len(files) > 0, "At least one file must be uploaded"
    assert all(hasattr(f, 'filename') for f in files), "All uploaded items must be valid files"

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Validate files
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(400, f"File {file.filename} is not a PDF")
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset
        
        if size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"File {file.filename} exceeds {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB limit")
    
    created_laws = []
    
    # Process each file
    for idx, file in enumerate(files):
        try:
            # Send progress update
            await manager.send_progress(session_id, {
                "type": "upload_progress",
                "current": idx + 1,
                "total": len(files),
                "filename": file.filename,
                "status": "uploading"
            })
            
            # Read file content
            content = await file.read()
            
            # Create law entry with PENDING status
            law = Law(
                reference=f"PENDING-{uuid.uuid4().hex[:8]}",
                title=f"PENDING-{uuid.uuid4().hex[:8]}",
                type="loi",
                content="",  # Will be filled by processing
                status="pending",
                processing_progress=0,
                processing_started_at=None
            )
            
            db.add(law)
            await db.flush()  # Get ID
            
            # Save file temporarily
            from pathlib import Path
            from app.services.file_upload_service import get_upload_service
            
            upload_service = get_upload_service()
            file_id = str(uuid.uuid4())
            file_path = upload_service.storage_path / f"{file_id}.pdf"
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            created_laws.append({
                "id": law.id,
                "filename": file.filename,
                "file_id": file_id,
                "status": "pending"
            })
            
            # Send creation confirmation
            await manager.send_progress(session_id, {
                "type": "file_created",
                "law_id": law.id,
                "filename": file.filename,
                "status": "pending"
            })
            
        except Exception as e:
            await manager.send_progress(session_id, {
                "type": "error",
                "filename": file.filename,
                "error": str(e)
            })
    
    await db.commit()
    
    # Start background processing for all files
    asyncio.create_task(process_batch(created_laws, session_id, db))
    
    return {
        "session_id": session_id,
        "total_files": len(files),
        "created_laws": created_laws
    }


async def process_batch(laws: List[dict], session_id: str, db: AsyncSession):
    """
    Background task to process all uploaded files.
    """
    for law_data in laws:
        try:
            law_id = law_data["id"]
            file_id = law_data["file_id"]
            
            # Update status to PROCESSING
            result = await db.execute(select(Law).where(Law.id == law_id))
            law = result.scalar_one_or_none()
            
            if law:
                law.status = "processing"
                law.processing_progress = 0
                law.processing_started_at = datetime.utcnow()
                await db.commit()
            
            # Send processing start
            await manager.send_progress(session_id, {
                "type": "processing_start",
                "law_id": law_id,
                "filename": law_data["filename"],
                "status": "processing"
            })
            
            # Call processing pipeline (synchronous)
            # We'll need to update process_law_sync to emit progress
            result = process_law_sync(law_id, file_id)
            
            # Update final status
            if result.get("status") == "completed":
                law.status = "published"
                law.processing_progress = 100
            else:
                law.status = "refused"
                law.processing_error = str(result.get("errors", []))
            
            await db.commit()
            
            # Send completion
            await manager.send_progress(session_id, {
                "type": "processing_complete",
                "law_id": law_id,
                "filename": law_data["filename"],
                "status": law.status,
                "progress": 100
            })
            
        except Exception as e:
            # Mark as refused
            law.status = "refused"
            law.processing_error = str(e)
            await db.commit()
            
            await manager.send_progress(session_id, {
                "type": "processing_error",
                "law_id": law_id,
                "filename": law_data["filename"],
                "error": str(e)
            })


@router.get("/status")
async def get_batch_status(
    status: str = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all laws with optional status filter.
    
    Args:
        status: Filter by status (pending, processing, published, refused)
        
    Returns:
        List of laws with their processing status
    """
    assert status is None or isinstance(status, str), "Status filter must be a string or None"
    assert status is None or status in ("pending", "processing", "published", "refused"), \
        f"Invalid status: {status}. Must be one of: pending, processing, published, refused"

    query = select(Law)
    
    if status:
        query = query.where(Law.status == status)
    
    query = query.order_by(Law.created_at.desc())
    
    result = await db.execute(query)
    laws = result.scalars().all()
    
    return {
        "total": len(laws),
        "laws": [
            {
                "id": law.id,
                "title": law.title,
                "reference": law.reference,
                "status": law.status,
                "processing_progress": law.processing_progress,
                "processing_error": law.processing_error,
                "created_at": law.created_at.isoformat() if law.created_at else None,
                "processing_started_at": law.processing_started_at.isoformat() if law.processing_started_at else None
            }
            for law in laws
        ]
    }
