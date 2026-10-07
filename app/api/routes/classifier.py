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

`/classify` est RESERVE AUX ADMINISTRATEURS. Chaque appel coute une requete
Groq sur le quota du classement des lois (1 000 par jour) : publique, la route
permettait a n'importe qui de le vider, et de bloquer l'ingestion.

Usage:
    curl -X POST http://localhost:8000/api/v1/classifier/classify \
        -H "Authorization: Bearer <jeton administrateur>" \
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

from app.core.auth import get_current_admin_user
from app.core.database import get_db
from app.models.law import Category
from app.models.user import User
from app.services.legal_domain_classifier import (
    CANONICAL_DOMAINS,
    ClassementIndisponible,
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
        description="Texte integral ; seul un extrait (l'article premier) est envoye au modele",
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
    rule: str = Field(..., description="Modele qui a decide, pour diagnostic")
    source: str = Field(..., description="Toujours groq")
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
        "Classement par Groq (une requete) : reserve aux administrateurs, et "
        "503 avec Retry-After quand le modele ne peut pas repondre."
    ),
    responses={
        503: {"description": "Classement indisponible (quota ou panne de Groq)"},
    },
)
async def classify_document(
    request: ClassifyDocumentRequest,
    db: AsyncSession = Depends(get_db),
    classifier: LegalDomainClassifier = Depends(get_legal_domain_classifier),
    current_admin: User = Depends(get_current_admin_user),
) -> ClassifyDocumentResponse:
    """Classe un document et resout son identifiant de categorie."""
    if not request.has_signal():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Fournissez au moins un titre ou un texte",
        )

    start = time.perf_counter()
    try:
        # await : l'appel synchrone bloquait la boucle d'evenements, donc
        # toutes les autres requetes, le temps de l'appel a Groq.
        result = await classifier.classify_async(request.title, request.text, request.doc_type)
    except ClassementIndisponible as exc:
        logger.warning("Classement indisponible : %s", exc.raison)
        entetes = {}
        if exc.retry_after is not None:
            entetes["Retry-After"] = str(max(1, int(exc.retry_after + 0.999)))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classement indisponible pour le moment, réessayez plus tard.",
            headers=entetes or None,
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
