"""
API routes for Laws management.

Endpoints:
- GET /api/v1/laws - List laws with filters
- GET /api/v1/laws/{id} - Get law detail with articles
- POST /api/v1/laws/{id}/explain-article - Explain one article (Gemini)
- POST /api/v1/admin/laws - Create law (admin only)
- PUT /api/v1/admin/laws/{id} - Update law (admin only)
- DELETE /api/v1/admin/laws/{id} - Delete law (admin only)

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import get_current_admin_user
from app.core.database import AsyncSessionLocal, get_db
from app.models.law import Law
from app.models.user import User
from app.schemas.law import (
    ArticleExplanationRequest,
    ArticleExplanationResponse,
    LawCreate,
    LawDetailResponse,
    LawResponse,
    LawUpdate,
)
from app.services.document_storage import (
    DocumentInaccessible,
    DocumentIntrouvable,
    IdentifiantInvalide,
    chemin_local,
)
from app.services.explanation_service import (
    ArticleNotFoundError,
    ExplanationError,
    ExplanationOverloadedError,
    ExplanationQuotaError,
    ExplanationService,
)
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


@router.get("/{law_id}", response_model=LawDetailResponse)
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

        # Les articles sont deja charges par le selectinload ci-dessus ; ils
        # etaient simplement jetes par LawResponse, qui n'expose que
        # `article_count`. La page de lecture les reconstruisait donc par
        # expression reguliere sur le contenu — 193 articles la ou la base en
        # compte 230, et aucune page de PDF.
        law.articles.sort(key=lambda a: (a.order or 0, a.id))
        logger.info(f"✅ Law {law_id} found: {law.title} ({len(law.articles)} articles)")
        return law

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching law {law_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching law"
        )


def get_explanation_service(db: AsyncSession = Depends(get_db)) -> ExplanationService:
    """Injection d'ExplanationService, calquee sur get_rag_service."""
    return ExplanationService(db)


@router.post("/{law_id}/explain-article", response_model=ArticleExplanationResponse)
async def explain_article(
    law_id: int,
    request: ArticleExplanationRequest,
    service: ExplanationService = Depends(get_explanation_service),
) -> ArticleExplanationResponse:
    """
    Explique un article en langage courant, sur la page du document.

    Le numero voyage dans le corps et non dans le chemin : `articles.number`
    contient « 1er », « L 94 bis » ou « PREAMBULE », qu'il faudrait sinon
    encoder.

    Route PUBLIQUE, comme `GET /laws/{id}` et `POST /rag/ask` : l'explication
    ne revele rien que la page ne montre deja. Elle depense en revanche un
    appel Gemini a chaque fois, sur un quota partage avec le chat — il n'y a
    volontairement aucun cache, c'est un choix produit assume.

    Raises:
        404: document ou article introuvable
        429: quota de generation epuise
        503: service de generation sature
    """
    try:
        return await service.explain(
            law_id=law_id,
            number=request.number,
            language=request.language,
            excerpt=request.excerpt,
        )

    except ArticleNotFoundError as e:
        logger.warning(f"⚠️ Explication impossible: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ExplanationQuotaError as e:
        # 429 et non 500 : la cause est connue, elle se dit, et elle porte un
        # delai. Meme traitement que /rag/ask, dont le quota est le meme.
        logger.warning(f"⚠️ Quota epuise: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": "60"},
        )
    except ExplanationOverloadedError as e:
        # 503 et non 500 : le client sait qu'un nouvel essai a un sens.
        logger.warning(f"⚠️ Generation saturee: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "10"},
        )
    except ExplanationError as e:
        logger.error(f"❌ Explication en echec: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    except Exception as e:
        # `detail` ne reprend PAS le message : une erreur inattendue peut porter
        # une cle ou un identifiant de projet, et cette route est publique.
        logger.error(f"❌ Erreur inattendue: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur",
        )


def _download_filename(law: Law, file_path: Path) -> str:
    """
    Nom de fichier propose a l'utilisateur, construit sur le TITRE du document.

    original_filename est le nom du fichier tel qu'aspire depuis prc.cm :
    "10691_decret-n-2026-164-du-4-mai-2026-portant-approbation-des-statuts-...".
    Le prefixe numerique est l'identifiant interne du site source et les tirets
    remplacent une ponctuation qui existait — cela n'a aucun sens pour qui
    telecharge le texte.

    Le titre est donc repris tel quel, accents compris (Content-Disposition
    encode l'UTF-8 via filename*, RFC 5987), en ne retirant que les caracteres
    interdits dans un nom de fichier.
    """
    title = (law.title or "").strip()
    if not title:
        title = law.reference or file_path.stem

    # Caracteres interdits par les systemes de fichiers courants, plus les
    # caracteres de controle. Le reste, accents inclus, est conserve.
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")

    # 150 caracteres : sous la limite de 255 octets de la plupart des systemes,
    # meme apres encodage UTF-8 des accents.
    if len(cleaned) > 150:
        cleaned = cleaned[:150].rstrip()

    return f"{cleaned or 'document'}{file_path.suffix or '.pdf'}"


async def _law_file_path(law: Law) -> Path:
    """
    Chemin du fichier d'origine d'une loi, resolu UNE seule fois.

    Les endpoints qui servent un PDF repetaient la meme construction a la main,
    avec un repli qui joignait `law.file_id` au repertoire d'upload sans aucune
    verification : ni motif, ni resolve(), ni confinement. Ce n'etait pas
    exploitable — la valeur vient de la base et non de la requete — mais c'etait
    autant de copies de la faiblesse que resolve_upload_path a ete ecrite pour
    fermer.

    C'est desormais AUSSI la couture entre les deux magasins de documents. En
    mode local, rien ne change. En mode distant (`DOCUMENTS_BASE_URL` non vide),
    le document est telecharge en flux dans le cache disque, puis son chemin est
    rendu — les bibliotheques de rendu prennent un chemin, pas une URL, et un
    fichier deja sur disque se sert ensuite sans repasser en memoire.

    Raises:
        HTTPException: 404 si la loi n'a pas de fichier, si l'identifiant ne
            respecte pas le motif, ou si le document est introuvable ;
            503 si le magasin distant est en panne.
    """
    if not law.file_id:
        raise HTTPException(status_code=404, detail="No source file found for this law")

    try:
        return await chemin_local(law.file_id)
    except IdentifiantInvalide as exc:
        logger.warning(f"⚠️ file_id invalide pour la loi {law.id}: {exc}")
        raise HTTPException(status_code=404, detail="File not found on server") from exc
    except DocumentIntrouvable as exc:
        raise HTTPException(status_code=404, detail="File not found on server") from exc
    except DocumentInaccessible as exc:
        # 503 et non 404 : c'est NOTRE magasin qui est en panne. Un 404 ferait
        # disparaitre le document de l'interface au lieu de signaler une panne
        # passagere, et accuserait l'utilisateur d'un defaut qui n'est pas le
        # sien — meme raisonnement que pour GoogleInjoignable dans /auth/google.
        logger.error(f"⚠️ Magasin de documents injoignable pour la loi {law.id}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Document momentanement indisponible",
            headers={"Retry-After": "30"},
        ) from exc


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

    file_path = await _law_file_path(law)

    return FileResponse(
        path=str(file_path), 
        filename=_download_filename(law, file_path),
        media_type="application/pdf" if file_path.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # attachment et non inline : un bouton nomme "Telecharger" doit
        # enregistrer le fichier, pas l'afficher dans l'onglet.
        content_disposition_type="attachment"
    )


# /pdf-data et /pdf-stream ont ete SUPPRIMES.
#
# Aucun client ne les appelait — ni le front, ni les tests, ni un script. Mais
# tous deux lisaient le fichier entier en memoire (`f.read()`), et /pdf-data y
# ajoutait un encodage base64, qui gonfle de 33 %. Sur le document le plus lourd
# du corpus (26 Mo), un seul appel demandait donc une pointe de ~61 Mo dans un
# conteneur qui n'en a que 512 — le plus gros risque de saturation memoire du
# service, au profit de personne.
#
# Le besoin d'origine (contourner les gestionnaires de telechargement qui
# interceptent les requetes) est couvert par /download, qui sert le fichier en
# flux depuis le disque sans jamais le charger en memoire.

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

    file_path = await _law_file_path(law)

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

    file_path = await _law_file_path(law)

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
