"""
API Routes pour détection de langue.

Endpoints:
- POST /api/v1/language/detect - Détecte langue d'un texte
- GET /api/v1/language/health - Vérifie santé du service

Usage:
    curl -X POST http://localhost:8000/api/v1/language/detect \
        -H "Content-Type: application/json" \
        -d '{"text": "Article 1. La présente loi..."}'
"""

from typing import Dict, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator

from app.core.dependencies import get_language_detector
from app.services.language_detector import LanguageDetector, LanguageDetectionError

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    tags=["language"],
    responses={
        500: {"description": "Erreur interne du service de détection"}
    }
)


# ==================== MODELS ====================

class DetectLanguageRequest(BaseModel):
    """Requête de détection de langue."""

    text: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Texte à analyser (50-50000 caractères)",
        example="Article 1. La présente loi régit les conditions de création des sociétés commerciales au Cameroun."
    )

    min_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Seuil de confiance minimum (0-1)",
        example=0.80
    )

    @validator('text')
    def text_not_empty(cls, v):
        """Valide que le texte n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("Le texte ne peut pas être vide")
        return v.strip()

    # model_config / json_schema_extra et non `class Config: schema_extra` :
    # cette derniere est la forme Pydantic v1, ignoree en silence sous v2 —
    # l'exemple ne s'affichait donc jamais dans la documentation OpenAPI.
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Article 1. La présente loi régit les conditions de création des sociétés commerciales au Cameroun. Les dispositions du Code civil s'appliquent en matière de contrats.",
                "min_confidence": 0.80
            }
        }
    }


class DetectLanguageResponse(BaseModel):
    """Réponse de détection de langue."""

    language: str = Field(
        ...,
        description="Code langue détectée ('fr' ou 'en')",
        example="fr"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance (0-1)",
        example=0.98
    )

    method_votes: Dict[str, str] = Field(
        ...,
        description="Votes de chaque méthode de détection",
        example={
            "langdetect": "fr",
            "fasttext": "fr"
        }
    )

    consensus: bool = Field(
        ...,
        description="True si au moins 2/3 méthodes d'accord",
        example=True
    )

    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Temps de traitement en millisecondes",
        example=450
    )

    text_length: int = Field(
        ...,
        ge=0,
        description="Longueur du texte analysé",
        example=156
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "language": "fr",
                "confidence": 0.98,
                "method_votes": {
                    "langdetect": "fr",
                    "fasttext": "fr"
                },
                "consensus": True,
                "processing_time_ms": 450,
                "text_length": 156
            }
        }
    }


class HealthResponse(BaseModel):
    """Réponse du health check."""

    service: str = Field(..., example="LanguageDetector")
    status: str = Field(..., example="healthy")
    models: Dict[str, str] = Field(
        ...,
        example={
            "fasttext": "✅ OK"
        }
    )


# ==================== ENDPOINTS ====================

@router.post(
    "/detect",
    response_model=DetectLanguageResponse,
    status_code=status.HTTP_200_OK,
    summary="Détecte la langue d'un texte",
    description="""
    Détecte automatiquement la langue (français ou anglais) d'un texte juridique.

    **Méthode:** Ensemble langdetect + fastText avec vote majoritaire

    **Performance:** <1 seconde pour textes jusqu'à 5000 caractères

    **Précision:** >99% sur documents juridiques
    """,
    response_description="Langue détectée avec métadonnées de confiance"
)
async def detect_language(
    request: DetectLanguageRequest,
    detector: LanguageDetector = Depends(get_language_detector)
) -> DetectLanguageResponse:
    """
    Détecte la langue d'un texte.

    Args:
        request: Requête contenant le texte à analyser
        detector: Service de détection (injecté)

    Returns:
        Résultat de détection avec langue, confiance et votes

    Raises:
        HTTPException 400: Si texte invalide (trop court, vide)
        HTTPException 500: Si erreur interne de détection
    """
    assert request is not None, "DetectLanguageRequest must not be None"
    assert isinstance(request.text, str) and len(request.text) >= 50, "Text must be at least 50 characters"

    try:
        logger.info(f"📥 Requête détection langue (texte: {len(request.text)} chars)")

        # Appeler le service de détection
        result = detector.detect(
            text=request.text,
            min_confidence=request.min_confidence
        )

        logger.info(
            f"✅ Détection réussie: {result['language']} "
            f"({result['confidence']:.2%}, {result['processing_time_ms']}ms)"
        )

        return DetectLanguageResponse(**result)

    except ValueError as e:
        # Erreur de validation (texte trop court, etc.)
        logger.warning(f"⚠️  Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except LanguageDetectionError as e:
        # Erreur du service de détection
        logger.error(f"❌ Detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la détection: {str(e)}"
        )

    except Exception as e:
        # Erreur inattendue
        logger.exception(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Vérifie la santé du service",
    description="""
    Vérifie que tous les modèles NLP sont chargés et fonctionnels.

    **Modèles vérifiés:**
    - fastText (lid.176.bin)
    """,
    response_description="État de santé de chaque composant"
)
async def health_check(
    detector: LanguageDetector = Depends(get_language_detector)
) -> HealthResponse:
    """
    Vérifie l'état de santé du service de détection.

    Args:
        detector: Service de détection (injecté)

    Returns:
        État de santé avec status de chaque modèle

    Raises:
        HTTPException 503: Si un ou plusieurs modèles sont défaillants
    """
    try:
        health_status = detector.health_check()

        # Si status != "healthy", retourner 503 Service Unavailable
        if health_status["status"] != "healthy":
            logger.warning(f"⚠️  Service degradé: {health_status}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )

        logger.info("✅ Health check OK")
        return HealthResponse(**health_status)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.exception(f"❌ Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Erreur lors du health check: {str(e)}"
        )
