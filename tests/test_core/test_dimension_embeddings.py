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
from pgvector.sqlalchemy import HALFVEC
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.models.law import Article
from app.services.embedding_service import EmbeddingService


def test_colonne_configuration_et_service_s_accordent():
    colonne = Article.__table__.c.embedding.type.dim

    assert colonne == settings.EMBEDDING_DIM == EmbeddingService.EMBEDDING_DIM


def test_la_colonne_est_en_halfvec():
    """Deux octets par composante : 2 Ko par article au lieu de 4."""
    assert isinstance(Article.__table__.c.embedding.type, HALFVEC)


@pytest.mark.parametrize("dimension", [3072, 1536, 768, 512])
def test_une_autre_dimension_est_refusee_au_chargement(dimension):
    """
    Le cas vise : un .env d'avant une bascule, qui porte EMBEDDING_DIM=768
    (EmbeddingGemma) ou 3072 et l'emporte sur le defaut.
    """
    with pytest.raises(ValidationError) as erreur:
        Settings(EMBEDDING_DIM=dimension)

    message = str(erreur.value)
    assert "halfvec(1024)" in message
    # Le remede, pas seulement le refus : un refus sans remede se contourne.
    assert "Retirez EMBEDDING_DIM" in message


def test_la_dimension_du_schema_est_acceptee():
    assert Settings(EMBEDDING_DIM=1024).EMBEDDING_DIM == 1024


@pytest.mark.parametrize("valeur", ["gemma", " Gemma "])
def test_le_fournisseur_gemma_est_refuse_avec_le_remede(valeur):
    """
    Un .env d'avant la bascule porte EMBEDDING_PROVIDER=gemma : ses vecteurs
    (768) ne tiennent plus dans la colonne. Refus lisible, avec le remede,
    plutot que l'erreur generique du Literal.
    """
    with pytest.raises(ValidationError) as erreur:
        Settings(EMBEDDING_PROVIDER=valeur)

    message = str(erreur.value)
    assert "EMBEDDING_PROVIDER=gemma n'existe plus" in message
    assert "VOYAGE_API_KEY" in message


@pytest.mark.parametrize("valeur, attendu", [("voyage", "voyage"), ("Gemini", "gemini")])
def test_les_fournisseurs_connus_sont_acceptes(valeur, attendu):
    assert Settings(EMBEDDING_PROVIDER=valeur).EMBEDDING_PROVIDER == attendu


@pytest.mark.parametrize("reglage", ["VOYAGE_TIMEOUT_QUESTION_S", "VOYAGE_TIMEOUT_DOCUMENT_S"])
def test_un_delai_voyage_nul_est_refuse(reglage):
    """A 0, l'echeance des questions disparaitrait alors que chaque appel echouerait."""
    with pytest.raises(ValidationError):
        Settings(**{reglage: 0})
