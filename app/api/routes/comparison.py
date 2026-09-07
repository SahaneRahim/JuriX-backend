"""
API route for the comparison mode.

Endpoint:
- POST /api/v1/compare - Compare two legal regimes, cell by cell, with sources

Author: JuriX Team
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services.comparison_service import (
    ComparisonError,
    ComparisonOverloadedError,
    ComparisonQuotaError,
    ComparisonService,
    NoContextError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_comparison_service(db: AsyncSession = Depends(get_db)) -> ComparisonService:
    """Injection du service, calquee sur get_rag_service."""
    return ComparisonService(db)


@router.post("", response_model=ComparisonResponse)
async def compare(
    request: ComparisonRequest,
    service: ComparisonService = Depends(get_comparison_service),
) -> ComparisonResponse:
    """
    Compare deux regimes juridiques sur des criteres imposes.

    Une recherche par sujet, et non une requete melangee : mesure sur le corpus,
    la requete unique rendait six articles d'un regime contre trois de l'autre,
    en ratant le bloc qui definissait le second.

    Chaque cellule porte ses articles source, texte integral compris, et une
    cellule sans source affiche l'absence plutot que de la combler.

    Raises:
        404: aucun texte trouve pour l'un des deux sujets
        429: quota de generation epuise
        503: service de generation sature
    """
    try:
        return await service.compare(
            subject_a=request.subject_a,
            subject_b=request.subject_b,
            language=request.language,
            criteria=request.criteria,
            law_id=request.law_id,
            top_k=request.top_k,
        )

    except NoContextError as e:
        logger.info(f"⚖️ Comparaison sans matiere : {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ComparisonQuotaError as e:
        logger.warning(f"⚠️ Quota epuise: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": "60"},
        )
    except ComparisonOverloadedError as e:
        logger.warning(f"⚠️ Generation saturee: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "10"},
        )
    except ComparisonError as e:
        logger.error(f"❌ Comparaison en echec: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    except Exception as e:
        # `detail` ne reprend PAS le message : la route est publique et une
        # erreur inattendue peut porter une cle.
        logger.error(f"❌ Erreur inattendue: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur",
        )
