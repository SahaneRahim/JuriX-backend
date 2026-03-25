"""
API Routes pour catégorisation de documents juridiques.

Endpoints:
- POST /api/v1/classifier/classify - Classifie un document
- GET /api/v1/classifier/health - Vérifie santé du service
- GET /api/v1/classifier/categories - Liste toutes les catégories

Usage:
    curl -X POST http://localhost:8000/api/v1/classifier/classify \
        -H "Content-Type: application/json" \
        -d '{"text": "Article 1. La SARL est une société commerciale..."}'
"""

from typing import Dict, Any, List
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator

from app.core.dependencies import get_document_classifier
from app.services.document_classifier import DocumentClassifier, DocumentClassificationError

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/classifier",
    tags=["classifier"],
    responses={
        500: {"description": "Erreur interne du service de classification"}
    }
)


# ==================== MODELS ====================

class ClassifyDocumentRequest(BaseModel):
    """Requête de classification de document."""

    text: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Texte du document à classifier (50-50000 caractères)",
        example="Article 1. Les sociétés commerciales sont régies par l'acte uniforme OHADA. La SARL doit avoir un capital social minimum."
    )

    language: str = Field(
        default="fr",
        description="Langue du document ('fr' ou 'en')",
        example="fr"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Nombre de suggestions à retourner (1-12)",
        example=3
    )

    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Seuil de confiance minimum (0-1)",
        example=0.0
    )

    @validator('text')
    def text_not_empty(cls, v):
        """Valide que le texte n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("Le texte ne peut pas être vide")
        return v

    @validator('language')
    def language_valid(cls, v):
        """Valide que la langue est supportée."""
        if v not in ['fr', 'en']:
            raise ValueError("Langue doit être 'fr' ou 'en'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Article 1. La SARL est régie par l'acte uniforme OHADA relatif au droit des sociétés commerciales. Le capital social est divisé en parts sociales.",
                "language": "fr",
                "top_k": 3,
                "min_confidence": 0.0
            }
        }


class CategorySuggestion(BaseModel):
    """Suggestion de catégorie avec confiance."""

    category_id: int = Field(
        ...,
        ge=1,
        le=12,
        description="ID de la catégorie (1-12)",
        example=4
    )

    category_name: str = Field(
        ...,
        description="Nom de la catégorie",
        example="Droit Commercial OHADA"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance (0-1)",
        example=0.92
    )

    method: str = Field(
        ...,
        description="Méthode utilisée ('keyword', 'ml', 'hybrid')",
        example="keyword"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "category_id": 4,
                "category_name": "Droit Commercial OHADA",
                "confidence": 0.92,
                "method": "keyword"
            }
        }


class ClassifyDocumentResponse(BaseModel):
    """Réponse de classification."""

    suggestions: List[CategorySuggestion] = Field(
        ...,
        description="Liste des suggestions classées par confiance décroissante"
    )

    processing_time_ms: int = Field(
        ...,
        description="Temps de traitement en millisecondes",
        example=125
    )

    class Config:
        json_schema_extra = {
            "example": {
                "suggestions": [
                    {
                        "category_id": 4,
                        "category_name": "Droit Commercial OHADA",
                        "confidence": 0.92,
                        "method": "keyword"
                    },
                    {
                        "category_id": 12,
                        "category_name": "Droit des Affaires",
                        "confidence": 0.68,
                        "method": "keyword"
                    },
                    {
                        "category_id": 2,
                        "category_name": "Droit Civil",
                        "confidence": 0.45,
                        "method": "keyword"
                    }
                ],
                "processing_time_ms": 125
            }
        }


class HealthResponse(BaseModel):
    """Réponse du health check."""

    service: str = Field(..., example="DocumentClassifier")
    status: str = Field(..., example="healthy")
    mode: str = Field(..., example="keywords-only")
    ml_model_loaded: bool = Field(..., example=False)
    vectorizer_loaded: bool = Field(..., example=False)
    categories_count: int = Field(..., example=12)
    version: str = Field(..., example="1.0.0-mvp")


class Category(BaseModel):
    """Catégorie juridique."""

    id: int = Field(..., ge=1, le=12, example=4)
    name: str = Field(..., example="Droit Commercial OHADA")


class CategoriesResponse(BaseModel):
    """Liste des catégories disponibles."""

    categories: List[Category] = Field(
        ...,
        description="Liste des 12 catégories juridiques"
    )


# ==================== ENDPOINTS ====================

@router.post(
    "/classify",
    response_model=ClassifyDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Classifier un document juridique",
    description="""
Classifie un document juridique dans une ou plusieurs des 12 catégories.

**Méthodes:**
- Phase 1 (MVP): Keywords-only (70-75% précision)
- Phase 3: Hybrid 40% keywords + 60% ML (87% top-1, 96% top-3)

**Catégories disponibles:**
1. Droit Constitutionnel
2. Droit Civil
3. Droit Pénal
4. Droit Commercial OHADA
5. Droit du Travail
6. Droit Fiscal
7. Droit Administratif
8. Droit Foncier
9. Droit de la Famille
10. Droit de l'Environnement
11. Droit International
12. Droit des Affaires

**Performance:**
- Temps de réponse: <500ms (MVP), <2s (ML)
- Top-k suggestions classées par confiance décroissante
"""
)
async def classify_document(
    request: ClassifyDocumentRequest,
    classifier: DocumentClassifier = Depends(get_document_classifier)
) -> ClassifyDocumentResponse:
    """
    Classifie un document juridique.

    Args:
        request: Requête avec texte et paramètres
        classifier: Instance du classifier (injectée)

    Returns:
        Suggestions de catégories avec scores de confiance

    Raises:
        HTTPException 400: Texte invalide
        HTTPException 500: Erreur interne de classification
    """
    assert request is not None, "ClassifyDocumentRequest must not be None"
    assert isinstance(request.text, str) and len(request.text) >= 50, "Text must be at least 50 characters"

    try:
        import time
        start_time = time.time()

        # Classification
        results = classifier.classify(
            text=request.text,
            language=request.language,
            top_k=request.top_k,
            min_confidence=request.min_confidence
        )

        # Formatter réponse
        suggestions = []
        for category_id, confidence, method in results:
            category_name = classifier.get_category_name(category_id)
            suggestions.append(
                CategorySuggestion(
                    category_id=category_id,
                    category_name=category_name,
                    confidence=confidence,
                    method=method
                )
            )

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Classification réussie: {len(suggestions)} suggestions, "
            f"top={suggestions[0].category_name if suggestions else 'N/A'}, "
            f"temps={processing_time_ms}ms"
        )

        return ClassifyDocumentResponse(
            suggestions=suggestions,
            processing_time_ms=processing_time_ms
        )

    except ValueError as e:
        logger.warning(f"Requête invalide: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except DocumentClassificationError as e:
        logger.error(f"Erreur classification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de classification: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Erreur inattendue: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du service de classification"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Vérifier la santé du service",
    description="Vérifie l'état de santé du service de classification et les modèles chargés."
)
async def health_check(
    classifier: DocumentClassifier = Depends(get_document_classifier)
) -> HealthResponse:
    """
    Vérifie la santé du service de classification.

    Args:
        classifier: Instance du classifier (injectée)

    Returns:
        État du service et des modèles
    """
    try:
        health_status = classifier.health_check()

        logger.debug(f"Health check: {health_status}")

        return HealthResponse(**health_status)

    except Exception as e:
        logger.error(f"Erreur health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de vérifier la santé du service"
        )


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    status_code=status.HTTP_200_OK,
    summary="Lister les catégories",
    description="Retourne la liste des 12 catégories juridiques disponibles."
)
async def get_categories(
    classifier: DocumentClassifier = Depends(get_document_classifier)
) -> CategoriesResponse:
    """
    Retourne toutes les catégories juridiques disponibles.

    Args:
        classifier: Instance du classifier (injectée)

    Returns:
        Liste des catégories avec IDs et noms
    """
    try:
        all_categories = classifier.get_all_categories()

        categories = [
            Category(id=cat_id, name=name)
            for cat_id, name in sorted(all_categories.items())
        ]

        logger.debug(f"Retour de {len(categories)} catégories")

        return CategoriesResponse(categories=categories)

    except Exception as e:
        logger.error(f"Erreur récupération catégories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de récupérer les catégories"
        )
