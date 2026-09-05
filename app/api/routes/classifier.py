"""
API Routes pour la classification d'un document dans un domaine juridique.

Endpoints:
- POST /api/v1/classifier/classify - Classe un document dans UN domaine
- GET  /api/v1/classifier/health   - Sante du service

`GET /api/v1/classifier/categories` a ete supprime : il renvoyait un
dictionnaire code en dur qui contredisait `GET /api/v1/categories`, lequel vient
de la base. Deux listes de categories concurrentes servies par la meme API,
c'etait la garantie qu'un client se fie a la mauvaise. Le front n'appelait pas
cet endpoint.

Usage:
    curl -X POST http://localhost:8000/api/v1/classifier/classify \
        -H "Content-Type: application/json" \
        -d '{"title": "Loi portant Code Minier", "text": "..."}'
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.law import Category
from app.services.legal_domain_classifier import (
    CANONICAL_DOMAINS,
    LegalDomainClassifier,
    get_legal_domain_classifier,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["classifier"],
    responses={500: {"description": "Erreur interne du service de classification"}},
)


# ==================== MODELS ====================


class ClassifyDocumentRequest(BaseModel):
    """Requete de classification."""

    title: str = Field(
        default="",
        max_length=1000,
        description="Titre du document — signal principal du classement",
    )
    text: str = Field(
        default="",
        max_length=200_000,
        description="Texte integral, utilise si le titre ne tranche pas",
    )
    doc_type: Optional[str] = Field(
        default=None, max_length=50, description="loi, decret, arrete..."
    )

    @field_validator("title", "text")
    @classmethod
    def strip_input(cls, value: str) -> str:
        return (value or "").strip()

    def has_signal(self) -> bool:
        return bool(self.title or self.text)


class ClassifyDocumentResponse(BaseModel):
    """Verdict de classement."""

    domain: str = Field(..., description="Domaine juridique retenu")
    category_id: Optional[int] = Field(
        None,
        description="Identifiant resolu depuis la table categories, nul si le domaine y manque",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    rule: str = Field(..., description="Regle qui a decide, pour diagnostic")
    source: str = Field(..., description="title, content ou doctype-default")
    runners_up: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float


# ==================== ENDPOINTS ====================


@router.post(
    "/classify",
    response_model=ClassifyDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Classer un document dans un domaine juridique",
    description=(
        "Rend UN domaine parmi les 14 domaines canoniques, plus l'identifiant "
        "de la ligne correspondante dans `categories`, resolu PAR LE NOM. "
        "Le titre decide en priorite ; le contenu n'est consulte que s'il ne "
        "tranche pas. Aucun appel reseau."
    ),
)
async def classify_document(
    request: ClassifyDocumentRequest,
    db: AsyncSession = Depends(get_db),
    classifier: LegalDomainClassifier = Depends(get_legal_domain_classifier),
) -> ClassifyDocumentResponse:
    """Classe un document et resout son identifiant de categorie."""
    if not request.has_signal():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Fournissez au moins un titre ou un texte",
        )

    start = time.perf_counter()
    try:
        result = classifier.classify(request.title, request.text, request.doc_type)
    except Exception as exc:  # pragma: no cover - le classifieur est pur
        logger.exception("Classification impossible")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de classification: {exc}",
        ) from exc

    rows = (await db.execute(select(Category.name, Category.id))).all()
    domain_map = {name.lower(): identifier for name, identifier in rows}
    category_id = domain_map.get(result.domain.lower())
    if category_id is None:
        logger.error(
            "Domaine %r absent de la table categories (%d lignes presentes)",
            result.domain, len(domain_map),
        )

    return ClassifyDocumentResponse(
        domain=result.domain,
        category_id=category_id,
        confidence=result.confidence,
        rule=result.rule,
        source=result.source,
        runners_up=[{"domain": d, "score": round(s, 4)} for d, s in result.runners_up],
        processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Sante du service de classification",
)
async def health_check(
    classifier: LegalDomainClassifier = Depends(get_legal_domain_classifier),
) -> Dict[str, Any]:
    """Rend l'etat du classifieur et le nombre de domaines qu'il connait."""
    report = classifier.health_check()
    report["canonical_domains"] = list(CANONICAL_DOMAINS)
    return report
