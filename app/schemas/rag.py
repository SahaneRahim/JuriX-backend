"""
Pydantic schemas for RAG chatbot service.

Defines request/response models for RAG endpoints, citations, and conversation management.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RAGRequest(BaseModel):
    """Request schema for RAG ask endpoint."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="User question",
        examples=["Quelle est la responsabilité des dirigeants de société?"]
    )
    persona: str = Field(
        "citoyen",
        description="User persona (citoyen/avocat/entrepreneur/étudiant)"
    )
    language: str = Field(
        "fr",
        description="Response language (fr/en)"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity"
    )
    stream: bool = Field(
        False,
        description="Enable streaming response"
    )
    law_id: Optional[int] = Field(
        None,
        description="Current law ID being viewed (prioritizes this document in search)"
    )
    top_k: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )

    @field_validator("persona")
    @classmethod
    def validate_persona(cls, v: str) -> str:
        allowed = {"citoyen", "avocat", "entrepreneur", "étudiant"}
        if v not in allowed:
            raise ValueError(f"Persona must be one of: {', '.join(allowed)}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in {"fr", "en"}:
            raise ValueError("Language must be 'fr' or 'en'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quelle est la responsabilité des dirigeants de société?",
                "persona": "avocat",
                "language": "fr",
                "session_id": "abc123",
                "stream": False
            }
        }
    }


class Citation(BaseModel):
    """Source citation with structured data for frontend linking."""

    law_id: int = Field(..., description="Database ID of law")
    law_reference: str = Field(..., description="Law reference (e.g., LOI-2024-001)")
    law_title: str = Field(..., description="Law title")
    article_number: Optional[str] = Field(None, description="Article number if specific")
    excerpt: str = Field(..., max_length=300, description="Relevant excerpt")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "law_id": 156,
                "law_reference": "LOI-2024-001",
                "law_title": "Code OHADA",
                "article_number": "161",
                "excerpt": "Les dirigeants sont responsables...",
                "relevance_score": 0.92
            }
        }
    }


class RAGResponse(BaseModel):
    """Response schema for RAG ask endpoint."""

    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[Citation] = Field(default_factory=list, description="Source citations")
    session_id: str = Field(..., description="Session ID for continuation")
    retrieval_time_ms: int = Field(..., description="Time for document retrieval")
    generation_time_ms: int = Field(..., description="Time for answer generation")
    total_time_ms: int = Field(..., description="Total processing time")
    persona: str = Field(..., description="Persona used for response")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "La responsabilité des dirigeants de société...",
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
                "persona": "citoyen"
            }
        }
    }


class RAGStreamChunk(BaseModel):
    """Streaming response chunk for SSE."""

    chunk: str = Field(..., description="Response text chunk")
    done: bool = Field(False, description="Whether generation is complete")
    sources: Optional[List[Citation]] = Field(
        None,
        description="Sources (only in final chunk)"
    )
    confidence: Optional[float] = Field(
        None,
        description="Confidence (only in final chunk)"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID (only in final chunk)"
    )


class MessageResponse(BaseModel):
    """Single message in conversation."""

    id: int
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Citation]] = None
    confidence: Optional[float] = None
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class ConversationResponse(BaseModel):
    """Conversation history response."""

    session_id: str
    persona: str
    language: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]

    model_config = {
        "from_attributes": True
    }
