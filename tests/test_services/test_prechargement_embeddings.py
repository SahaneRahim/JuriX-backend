"""
Le prechargement du service d'embeddings, au demarrage de l'API.

Avant lui, une configuration fausse — cle absente ou refusee, modele inconnu —
ne se revelait qu'a la premiere question : la recherche hybride avalait
l'erreur et repondait en plein texte seul, sans rien signaler.

Le fournisseur est une API distante : un echec DEFINITIF ecarte le service, un
echec PASSAGER (reseau pas encore pret au redemarrage de la machine) le garde.

Les singletons de search_service sont affectes directement : la doublure de
tests/conftest.py restaure leur etat apres chaque test.

Usage:
    pytest tests/test_services/test_prechargement_embeddings.py -v
"""

from app.services import search_service
from app.services.embedding_service import (
    EmbeddingDefinitiveError,
    EmbeddingServiceError,
    suivi_semantique,
)


class _Sonde:
    def __init__(self, erreur):
        self.erreur = erreur

    def prechauffer(self):
        raise self.erreur


def test_un_service_operationnel_est_garde():
    # La doublure de conftest est installee ; son prechauffage ne fait rien.
    service = search_service._embedding_service_instance

    assert search_service.precharger_les_embeddings() is True
    assert search_service._embedding_service_instance is service


def test_un_refus_definitif_ecarte_le_service():
    """
    Une cle refusee le sera a chaque question : ecarte, la recherche
    semantique est coupee proprement et la sante le dit.
    """
    search_service._embedding_service_instance = _Sonde(
        EmbeddingDefinitiveError("Voyage HTTP 401 : Provided API key is invalid.")
    )

    assert search_service.precharger_les_embeddings() is False
    assert search_service._embedding_service_instance is None


def test_un_echec_passager_garde_le_service():
    """
    Reseau ou DNS pas encore prets au redemarrage de la VM, 5xx, 429 :
    l'ecarter couperait la recherche semantique jusqu'au prochain redemarrage,
    pour une panne d'une minute. Chaque question retentera.
    """
    sonde = _Sonde(EmbeddingServiceError("Voyage injoignable : ConnectError"))
    search_service._embedding_service_instance = sonde

    assert search_service.precharger_les_embeddings() is False
    assert search_service._embedding_service_instance is sonde
    # Garde, mais la sante le montre jusqu'au premier appel reussi.
    assert suivi_semantique.etat()["degrade"]


def test_une_sonde_reussie_est_notee():
    assert search_service.precharger_les_embeddings() is True
    assert suivi_semantique.etat()["sonde_demarrage"]["erreur"] is None


def test_un_service_impossible_a_construire_est_signale():
    search_service._embedding_service_instance = None
    search_service._singletons_initialized = True

    assert search_service.precharger_les_embeddings() is False
