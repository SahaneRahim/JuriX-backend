"""
Tests du classement en domaine juridique via Groq / Qwen.

Chaque assertion porte sur le domaine canonique retourné et les invariants du système.

Usage:
    pytest tests/test_services/test_legal_domain_classifier.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.legal_domain_classifier import (
    ADMINISTRATIF,
    AFFAIRES,
    CANONICAL_DOMAINS,
    CIVIL,
    CONSTITUTIONNEL,
    EDUCATION,
    ENVIRONNEMENT,
    FAMILLE,
    FINANCES,
    FONCIER,
    FONCTION_PUBLIQUE,
    INTERNATIONAL,
    PENAL,
    SANTE,
    TRAVAIL,
    LegalDomainClassifier,
    get_legal_domain_classifier,
)


@pytest.fixture
def classifier() -> LegalDomainClassifier:
    return LegalDomainClassifier()


class TestPublicSurface:
    def test_canonical_domains_count_is_14(self):
        assert len(CANONICAL_DOMAINS) == 14
        assert len(set(CANONICAL_DOMAINS)) == 14

    def test_no_document_type_among_domains(self):
        for doc_type in ["loi", "décret", "arrêté", "ordonnance", "circulaire", "décision"]:
            assert not any(
                doc_type in domain.lower() or f"{doc_type}s" in domain.lower()
                for domain in CANONICAL_DOMAINS
            )

    def test_new_domains_present(self):
        assert SANTE in CANONICAL_DOMAINS
        assert EDUCATION in CANONICAL_DOMAINS
        assert "Santé Publique et Sécurité Sanitaire" in CANONICAL_DOMAINS
        assert "Éducation, Recherche, Culture et Médias" in CANONICAL_DOMAINS

    def test_result_is_frozen(self, classifier):
        result = classifier.classify("Décret portant nomination")
        with pytest.raises(Exception):
            result.domain = "autre"


class TestEdgeCases:
    def test_empty_arguments_fall_to_default(self, classifier):
        res = classifier.classify("", "", "")
        assert res.domain in CANONICAL_DOMAINS
        assert res.source == "default"

    def test_none_arguments(self, classifier):
        res = classifier.classify(None, None, None)
        assert res.domain in CANONICAL_DOMAINS
        assert res.source == "default"

    def test_shared_instance(self):
        assert get_legal_domain_classifier() is get_legal_domain_classifier()

    def test_health_check(self, classifier):
        report = classifier.health_check()
        assert report["status"] == "healthy"
        assert report["domains"] == 14
        assert report["mode"] == "groq_qwen"


class TestClassificationMock:
    def test_mock_groq_success(self, classifier):
        mock_data = {
            "categorie": "Santé Publique et Sécurité Sanitaire",
            "confiance": 0.95,
            "categories_secondaires": ["Droit Administratif"],
            "justification": "Médecine traditionnelle et santé publique.",
        }
        with patch("app.services.groq_service.GroqService.classify_legal_domain_sync", return_value=mock_data):
            res = classifier.classify("Loi portant exercice de la médecine traditionnelle")
            assert res.domain == "Santé Publique et Sécurité Sanitaire"
            assert res.confidence == 0.95
            assert res.source == "groq"
            assert len(res.runners_up) == 1
            assert res.runners_up[0][0] == "Droit Administratif"

    def test_mock_groq_empty_fallback(self, classifier):
        with patch("app.services.groq_service.GroqService.classify_legal_domain_sync", return_value=None):
            res = classifier.classify("Titre quelconque")
            assert res.domain in CANONICAL_DOMAINS
            assert res.source == "default"


class TestLiveGroqClassification:
    """Tests réels avec l'API Groq configurée."""

    def test_live_health_check(self, classifier):
        assert classifier.health_check()["status"] == "healthy"

    def test_live_classify_nomination(self, classifier):
        res = classifier.classify("Décret N°2023/120 portant nomination d'un Préfet")
        assert res.domain == FONCTION_PUBLIQUE
        assert res.confidence >= 0.80

    def test_live_classify_finances(self, classifier):
        res = classifier.classify("Loi de finances de la République du Cameroun pour l'exercice 2024")
        assert res.domain == FINANCES
        assert res.confidence >= 0.90
