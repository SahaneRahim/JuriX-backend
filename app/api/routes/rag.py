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

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.database import get_db
from app.models.conversation import Conversation, Message
from app.schemas.rag import (
    ConversationResponse,
    MessageResponse,
    RAGRequest,
    RAGResponse,
)
from app.services.rag_service import RAGService, RAGServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_rag_service(db: AsyncSession = Depends(get_db)) -> RAGService:
    """Dependency injection for RAGService."""
    return RAGService(db)


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
    "/conversations/{session_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
)
async def get_conversation(
    session_id: str, db: AsyncSession = Depends(get_db)
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
        stmt = (
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .options(joinedload(Conversation.messages))
        )
        result = await db.execute(stmt)
        # .unique() obligatoire apres joinedload sur une collection, sinon
        # InvalidRequestError -> cet endpoint renvoyait systematiquement 500.
        conversation = result.unique().scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Conversation {session_id} not found"
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
async def delete_conversation(session_id: str, db: AsyncSession = Depends(get_db)):
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
        stmt = select(Conversation).where(Conversation.session_id == session_id)
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Conversation {session_id} not found"
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
            health_status["llm"] = llm_health.get("status", "unknown")
            if llm_health.get("status") != "healthy":
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

        test_response = await rag_service.search_service.search(
            SearchRequest(query="test", mode="text", limit=1)
        )
        health_status["search_service"] = "healthy"
    except Exception as e:
        health_status["search_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    return health_status
