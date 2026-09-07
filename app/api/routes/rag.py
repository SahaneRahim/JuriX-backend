"""
API routes for RAG chatbot service.

Provides endpoints for:
- ask: Standard RAG question answering
- ask/stream: Streaming RAG with SSE
- conversations: Conversation history management
- health: Service health check
"""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.auth import get_current_active_user, get_current_user_optional
from app.core.database import get_db
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.rag import (
    ConversationResponse,
    ConversationSummary,
    MessageResponse,
    RAGRequest,
    RAGResponse,
)
from app.services.rag_service import (
    ConversationInterdite,
    RAGOverloadedError,
    RAGQuotaError,
    RAGService,
    RAGServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_rag_service(
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> RAGService:
    """
    Injection de RAGService, avec le proprietaire quand il y en a un.

    `get_current_user_optional` rend `None` sans en-tete Authorization — le
    chat reste utilisable sans compte — mais leve 401 sur un jeton present et
    invalide, plutot que de degrader en anonyme : sinon l'utilisateur croirait
    sa conversation enregistree alors qu'elle partirait en `user_id` NULL.
    """
    return RAGService(db, user_id=user.id if user else None)


async def _conversation_accessible(
    db: AsyncSession, session_id: str, user_id: Optional[int], *, avec_messages: bool
) -> Conversation:
    """
    Charge une conversation, ou leve 404.

    REGLE D'APPARTENANCE, la meme partout : `user_id` est le proprietaire ;
    NULL signifie « anonyme, appartient a qui detient le session_id ».

    404 ET NON 403 quand elle appartient a quelqu'un d'autre : un 403
    confirmerait l'existence du session_id et permettrait de les enumerer. Meme
    raisonnement que le message unique de `_authenticate`.
    """
    stmt = select(Conversation).where(Conversation.session_id == session_id)
    if avec_messages:
        stmt = stmt.options(joinedload(Conversation.messages))
    resultat = await db.execute(stmt)
    # .unique() obligatoire apres joinedload sur une collection.
    conversation = resultat.unique().scalar_one_or_none() if avec_messages else resultat.scalar_one_or_none()

    introuvable = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Conversation {session_id} not found",
    )
    if conversation is None:
        raise introuvable
    if conversation.user_id not in (None, user_id):
        raise introuvable
    return conversation


async def _refuser_si_conversation_dautrui(
    db: AsyncSession, session_id: Optional[str], user_id: Optional[int]
) -> None:
    """
    Garde du flux SSE, a appeler AVANT d'ouvrir la reponse.

    Une fois la `StreamingResponse` commencee, le code HTTP est fige a 200 :
    l'erreur ne pourrait plus etre qu'un evenement dans le corps, que le client
    devrait savoir distinguer d'une reponse. Le refus doit donc sortir avant.

    A la difference de `_conversation_accessible`, une conversation INTROUVABLE
    n'est pas une erreur ici : le client a le droit de proposer un session_id
    neuf, le service creera la conversation.
    """
    if not session_id:
        return
    conversation = (
        await db.execute(
            select(Conversation.user_id).where(Conversation.session_id == session_id)
        )
    ).scalar_one_or_none()
    if conversation is not None and conversation not in (None, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation introuvable"
        )


@router.post("/ask", response_model=RAGResponse, status_code=status.HTTP_200_OK)
async def ask(
    request: RAGRequest, rag_service: RAGService = Depends(get_rag_service)
) -> RAGResponse:
    """
    Ask a legal question and get RAG-powered answer.

    **Pipeline:**
    1. Retrieve top 5 relevant legal documents (<200ms)
    2. Load conversation history (last 5 messages)
    3. Generate persona-adapted answer with Gemini (2-5s)
    4. Extract and validate source citations
    5. Calculate confidence score
    6. Save interaction to database

    **Performance:** <5s total (specification requirement)

    **Example Request:**
    ```json
    {
        "question": "Quelle est la responsabilité des dirigeants de société?",
        "persona": "avocat",
        "language": "fr",
        "session_id": "abc123",
        "stream": false
    }
    ```

    **Example Response:**
    ```json
    {
        "answer": "Selon l'article 161 du Code OHADA...",
        "confidence": 0.85,
        "sources": [
            {
                "law_id": 156,
                "law_reference": "LOI-2024-001",
                "law_title": "Code OHADA",
                "article_number": "161",
                "excerpt": "Les dirigeants sont responsables...",
                "relevance_score": 0.92
            }
        ],
        "session_id": "abc123",
        "retrieval_time_ms": 180,
        "generation_time_ms": 2400,
        "total_time_ms": 2600,
        "persona": "avocat"
    }
    ```
    """
    assert request is not None, "RAGRequest must not be None"
    assert isinstance(request.question, str) and len(request.question) > 0, "Question must be a non-empty string"

    try:
        logger.info(f"📥 Ask request: persona={request.persona}")

        response = await rag_service.ask(request)

        logger.info(f"📤 Ask response: {response.total_time_ms}ms")
        return response

    except ConversationInterdite:
        # 404 et non 403 : un 403 confirmerait l'existence du session_id et
        # permettrait de les enumerer. PLACE AVANT `except RAGServiceError` —
        # l'ordre porte le sens, sinon le 500 generique avalerait ce cas.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation introuvable"
        )
    except RAGQuotaError as e:
        # 429 et non 500 : la cause est connue, elle se dit, et elle porte un
        # delai. Le message precedent deversait le JSON brut de Google dans
        # `detail`, ce qui exposait des identifiants de quota au client.
        logger.warning(f"⚠️ Quota epuise: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": "60"},
        )
    except RAGOverloadedError as e:
        # 503 et non 500 : le client sait alors qu'un nouvel essai a un sens,
        # et Retry-After lui dit quand. Un 500 disait « c'est casse » pour une
        # minute de charge chez le fournisseur.
        logger.warning(f"⚠️ Generation saturee: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "10"},
        )
    except RAGServiceError as e:
        logger.error(f"❌ RAG error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erreur interne du serveur"
        )


@router.post("/ask/stream", status_code=status.HTTP_200_OK)
async def ask_stream(request: RAGRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Ask question with streaming response (SSE).

    **Benefits:**
    - First token appears in <500ms (faster perceived response)
    - User sees answer being generated in real-time
    - Better UX for longer answers

    **Response Format:** Server-Sent Events (SSE)

    Each event is JSON:
    ```json
    {"chunk": "text fragment", "done": false}
    {"chunk": "", "done": true, "sources": [...], "confidence": 0.85}
    ```

    **Example Usage (JavaScript):**
    ```javascript
    const eventSource = new EventSource('/api/v1/rag/ask/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.done) {
            // Show sources and confidence
        } else {
            // Append chunk to answer
        }
    };
    ```
    """
    try:
        # Force streaming mode
        request.stream = True

        # Le refus d'appartenance doit sortir AVANT l'ouverture du flux : une
        # fois la StreamingResponse commencee, le code HTTP est fige a 200 et
        # l'erreur ne pourrait plus etre qu'un evenement dans le corps.
        await _refuser_si_conversation_dautrui(
            rag_service.db, request.session_id, rag_service.user_id
        )

        async def event_generator():
            try:
                async for chunk_json in rag_service.ask_stream(request):
                    yield f"data: {chunk_json}\n\n"
            except Exception as e:
                logger.error(f"❌ Stream error: {e}")
                error_json = json.dumps({"error": str(e), "done": True})
                yield f"data: {error_json}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    except Exception as e:
        logger.error(f"❌ Streaming setup error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    status_code=status.HTTP_200_OK,
)
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
) -> List[Conversation]:
    """
    Les conversations du compte courant, la plus recente en premier.

    Route AUTHENTIFIEE, sans variante anonyme : une conversation anonyme
    n'appartient a personne, donc aucune liste ne peut la contenir.

    Sans jointure ni agregat : le titre est une colonne, et l'index
    `idx_conversations_user_updated` sert exactement ce filtre et ce tri.
    """
    resultat = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(min(max(limit, 1), 200))
        .offset(max(offset, 0))
    )
    return list(resultat.scalars().all())


@router.get(
    "/conversations/{session_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
)
async def get_conversation(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> ConversationResponse:
    """
    Get conversation history by session ID.

    Returns all messages in chronological order with sources and metadata.

    **Example Response:**
    ```json
    {
        "session_id": "abc123",
        "persona": "citoyen",
        "language": "fr",
        "created_at": "2026-01-10T10:00:00",
        "updated_at": "2026-01-10T10:05:00",
        "messages": [
            {
                "id": 1,
                "role": "user",
                "content": "Comment créer une entreprise?",
                "sources": null,
                "confidence": null,
                "created_at": "2026-01-10T10:00:00"
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "Pour créer une entreprise...",
                "sources": [...],
                "confidence": 0.85,
                "created_at": "2026-01-10T10:00:05"
            }
        ]
    }
    ```
    """
    assert isinstance(session_id, str) and len(session_id) > 0, "session_id must be a non-empty string"

    try:
        conversation = await _conversation_accessible(
            db, session_id, user.id if user else None, avec_messages=True
        )

        # Get messages in chronological order
        msg_stmt = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at)
        )
        msg_result = await db.execute(msg_stmt)
        messages = msg_result.scalars().all()

        return ConversationResponse(
            session_id=conversation.session_id,
            persona=conversation.persona,
            language=conversation.language,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=[MessageResponse.model_validate(m) for m in messages],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching conversation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/conversations/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Delete conversation and all associated messages.

    Useful for:
    - User wants to start fresh
    - Privacy/GDPR compliance
    - Clear conversation context

    Returns 204 No Content on success.
    """
    assert isinstance(session_id, str) and len(session_id) > 0, "session_id must be a non-empty string"

    try:
        conversation = await _conversation_accessible(
            db, session_id, user.id if user else None, avec_messages=False
        )

        await db.delete(conversation)
        await db.commit()

        logger.info(f"🗑️ Deleted conversation: {session_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting conversation: {e}")
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(rag_service: RAGService = Depends(get_rag_service)) -> dict:
    """
    Check health of RAG system components.

    Tests:
    - LLM service availability (Gemini API)
    - Database connectivity
    - SearchService availability

    **Example Response:**
    ```json
    {
        "status": "healthy",
        "llm": "healthy",
        "database": "connected",
        "search_service": "healthy"
    }
    ```
    """
    health_status = {
        "status": "healthy",
        "llm": "not_configured",
        "database": "unknown",
        "search_service": "unknown",
    }

    # Test LLM (Gemini - TODO: implement)
    if rag_service.llm is not None:
        try:
            llm_health = await rag_service.llm.health_check()
            etat_llm = llm_health.get("status", "unknown")
            health_status["llm"] = etat_llm
            if etat_llm == "quota_exhausted":
                # Un quota epuise se resorbe seul : le dire, et le distinguer
                # d'une panne. La raison porte le delai conseille.
                health_status["status"] = "degraded"
                health_status["llm_reason"] = llm_health.get("reason", "")
            elif etat_llm != "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["llm"] = f"error: {str(e)}"
            health_status["status"] = "unhealthy"
    else:
        health_status["llm"] = "not_configured"
        health_status["status"] = "degraded"

    # Test database
    try:
        await rag_service.db.execute(select(1))
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Test SearchService
    try:
        # Simple test query
        from app.schemas.search import SearchRequest

        # C'est l'ABSENCE d'exception qui fait la sonde : la reponse n'a pas a
        # etre liee a un nom.
        await rag_service.search_service.search(
            SearchRequest(query="test", mode="text", limit=1)
        )
        health_status["search_service"] = "healthy"
    except Exception as e:
        health_status["search_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    return health_status
