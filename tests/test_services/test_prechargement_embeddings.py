"""
Le prechargement du service d'embeddings, au demarrage de l'API.

Avant lui, une installation incomplete — modele absent, fichier tronque — ne se
revelait qu'a la premiere question : la recherche hybride avalait l'erreur et
repondait en plein texte seul, sans rien signaler.

Les singletons de search_service sont affectes directement : la doublure de
tests/conftest.py restaure leur etat apres chaque test.

Usage:
    pytest tests/test_services/test_prechargement_embeddings.py -v
"""

from app.services import search_service


def test_un_service_operationnel_est_garde():
    # La doublure de conftest est installee ; son prechauffage ne fait rien.
    service = search_service._embedding_service_instance

    assert search_service.precharger_les_embeddings() is True
    assert search_service._embedding_service_instance is service


def test_un_prechauffage_en_echec_ecarte_le_service():
    """
    Garde, un service casse retenterait de charger 300 Mo a chaque question,
    pour echouer pareil. Ecarte, la recherche semantique est coupee
    proprement et la sante le dit.
    """

    class _Casse:
        def prechauffer(self):
            raise RuntimeError("fichier du modele tronque")

    search_service._embedding_service_instance = _Casse()

    assert search_service.precharger_les_embeddings() is False
    assert search_service._embedding_service_instance is None


def test_un_service_impossible_a_construire_est_signale():
    search_service._embedding_service_instance = None
    search_service._singletons_initialized = True

    assert search_service.precharger_les_embeddings() is False
