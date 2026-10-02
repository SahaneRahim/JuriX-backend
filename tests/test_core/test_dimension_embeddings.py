"""
La dimension des embeddings : un accord a trois, et une garde au demarrage.

Trois declarations portent la meme dimension — la colonne (Article.embedding),
la configuration (settings.EMBEDDING_DIM) et le service. Si elles divergent,
rien n'echoue a l'import : l'ingestion casse a la premiere ecriture, la
recherche a la premiere question, et l'hybride avale cette erreur en retombant
sans bruit sur le plein texte. Ces tests tournent sans base.

Usage:
    pytest tests/test_core/test_dimension_embeddings.py -v
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.models.law import Article
from app.services.embedding_service import EmbeddingService


def test_colonne_configuration_et_service_s_accordent():
    colonne = Article.__table__.c.embedding.type.dim

    assert colonne == settings.EMBEDDING_DIM == EmbeddingService.EMBEDDING_DIM


@pytest.mark.parametrize("dimension", [3072, 1536, 512])
def test_une_autre_dimension_est_refusee_au_chargement(dimension):
    """
    Le cas vise : un .env d'avant la bascule, qui porte EMBEDDING_DIM=3072 et
    l'emporte sur le defaut.
    """
    with pytest.raises(ValidationError) as erreur:
        Settings(EMBEDDING_DIM=dimension)

    message = str(erreur.value)
    assert "vector(768)" in message
    # Le remede, pas seulement le refus : un refus sans remede se contourne.
    assert "Retirez EMBEDDING_DIM" in message


def test_la_dimension_du_schema_est_acceptee():
    assert Settings(EMBEDDING_DIM=768).EMBEDDING_DIM == 768
