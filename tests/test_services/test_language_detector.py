"""
Tests unitaires pour le service LanguageDetector.

Ces tests vérifient:
- Détection français
- Détection anglais
- Validation texte court
- Validation texte vide
- Détection texte bilingue (langue dominante)

Usage:
    pytest backend/tests/test_services/test_language_detector.py -v
    pytest backend/tests/test_services/test_language_detector.py -v --cov=backend/app/services/language_detector
"""

import pytest
from app.services.language_detector import LanguageDetector, LanguageDetectionError


@pytest.fixture(scope="module")
def detector():
    """
    Fixture: Instance partagée de LanguageDetector.

    Utilise scope="module" pour éviter de recharger les modèles
    entre chaque test (économie de temps et mémoire).
    """
    return LanguageDetector()


class TestLanguageDetectorFrench:
    """Tests de détection de documents français."""

    def test_detect_french_legal_text(self, detector):
        """Test détection d'un document juridique français."""
        text = """
        Article 1. La présente loi régit les conditions de création
        des sociétés commerciales au Cameroun. Les dispositions du
        Code civil camerounais s'appliquent en matière de contrats
        et d'obligations contractuelles.
        Article 2. Les sociétés commerciales sont soumises aux règles
        du droit OHADA en ce qui concerne leur organisation et leur
        fonctionnement. La responsabilité civile et pénale des dirigeants
        est engagée conformément aux dispositions légales en vigueur.
        """

        result = detector.detect(text)

        # Vérifications principales
        assert result["language"] == "fr", "Langue détectée devrait être français"
        assert result["confidence"] > 0.80, f"Confiance {result['confidence']:.2%} < 80%"
        assert result["consensus"] is True, "Devrait avoir consensus (2/3 ou 3/3)"

        # Vérifier qu'au moins 2 méthodes ont voté (fasttext peut échouer avec numpy 2.0)
        assert len(result["method_votes"]) >= 2, "Au moins 2 méthodes doivent voter"
        assert "langdetect" in result["method_votes"], "langdetect doit toujours voter"

        # Vérifier temps de traitement
        assert result["processing_time_ms"] > 0
        assert result["processing_time_ms"] < 2000, "Détection trop lente (>2s)"

        # Vérifier longueur texte
        assert result["text_length"] > 0

    def test_detect_french_short_legal_text(self, detector):
        """Test détection d'un texte français court mais valide."""
        text = """
        Article 1. La présente loi régit le droit commercial.
        Article 2. Les sociétés sont soumises au droit OHADA.
        """

        result = detector.detect(text)

        assert result["language"] == "fr"
        assert result["confidence"] > 0.80
        assert result["consensus"] is True


class TestLanguageDetectorEnglish:
    """Tests de détection de documents anglais."""

    def test_detect_english_legal_text(self, detector):
        """Test détection d'un document juridique anglais."""
        text = """
        Section 1. This Act governs the conditions for creating
        commercial companies in Cameroon. The provisions of the
        Cameroonian Civil Code apply to contracts and contractual
        obligations.
        Section 2. Commercial companies are subject to OHADA law
        with regard to their organization and operation. The civil
        and criminal liability of managers is engaged in accordance
        with the legal provisions in force.
        """

        result = detector.detect(text)

        # Vérifications principales
        assert result["language"] == "en", "Langue détectée devrait être anglais"
        assert result["confidence"] > 0.75, f"Confiance {result['confidence']:.2%} < 75%"
        assert result["consensus"] is True, "Devrait avoir consensus"

        # Vérifier qu'au moins 2 méthodes ont voté (fasttext peut échouer avec numpy 2.0)
        assert len(result["method_votes"]) >= 2, "Au moins 2 méthodes doivent voter"
        assert "langdetect" in result["method_votes"], "langdetect doit toujours voter"

    def test_detect_english_short_legal_text(self, detector):
        """Test détection d'un texte anglais court mais valide."""
        text = """
        Section 1. This law governs commercial law in Cameroon.
        Section 2. Companies are subject to OHADA regulations.
        """

        result = detector.detect(text)

        assert result["language"] == "en"
        assert result["confidence"] > 0.80


class TestLanguageDetectorValidation:
    """Tests de validation des inputs."""

    def test_text_too_short_raises_error(self, detector):
        """Test qu'un texte trop court lève ValueError."""
        short_text = "Article 1. Court."

        with pytest.raises(ValueError, match="Texte trop court"):
            detector.detect(short_text)

    def test_empty_text_raises_error(self, detector):
        """Test qu'un texte vide lève ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            detector.detect("")

    def test_whitespace_only_text_raises_error(self, detector):
        """Test qu'un texte contenant uniquement des espaces lève ValueError."""
        whitespace_text = "   \n\t   "

        with pytest.raises(ValueError, match="ne peut pas être vide"):
            detector.detect(whitespace_text)

    def test_minimum_length_threshold(self, detector):
        """Test du seuil de longueur minimum configurable."""
        # Texte de 75 caractères
        text = "Article 1. Ce texte fait exactement soixante-quinze caractères précis."

        # Devrait passer avec min_length=50 (défaut)
        result = detector.detect(text, min_length=50)
        assert result["language"] in ["fr", "en"]

        # Devrait échouer avec min_length=100
        with pytest.raises(ValueError, match="Texte trop court"):
            detector.detect(text, min_length=100)


class TestLanguageDetectorBilingual:
    """Tests sur textes bilingues (FR + EN)."""

    def test_bilingual_text_french_dominant(self, detector):
        """Test texte mixte avec français dominant."""
        text = """
        Article 1. La présente loi régit les sociétés commerciales au Cameroun.
        Article 2. Les obligations contractuelles sont définies par le Code civil.
        Article 3. La responsabilité civile s'applique conformément à la loi.
        Article 4. Les dispositions du droit OHADA sont applicables.
        Section 5. This section is written in English for reference.
        Article 6. Les dispositions finales entrent en vigueur immédiatement.
        Article 7. Le présent texte abroge toutes dispositions contraires.
        """

        result = detector.detect(text)

        # Français devrait être détecté comme dominant
        assert result["language"] == "fr", "Français devrait être langue dominante"

        # Confiance peut être plus basse pour texte mixte
        assert result["confidence"] > 0.70, "Confiance devrait rester > 70%"

    def test_bilingual_text_english_dominant(self, detector):
        """Test texte mixte avec anglais dominant."""
        text = """
        Section 1. This law governs commercial companies in Cameroon.
        Section 2. Contractual obligations are defined by the Civil Code.
        Section 3. Civil liability applies in accordance with the law.
        Section 4. The provisions of OHADA law are applicable.
        Article 5. Cet article est écrit en français pour référence.
        Section 6. The final provisions enter into force immediately.
        Section 7. This text repeals all contrary provisions.
        """

        result = detector.detect(text)

        # Anglais devrait être détecté comme dominant
        assert result["language"] == "en", "Anglais devrait être langue dominante"
        assert result["confidence"] > 0.70


class TestLanguageDetectorPerformance:
    """Tests de performance."""

    def test_detection_time_under_1_second(self, detector):
        """Test que la détection prend <1s pour texte de taille normale."""
        # Texte de ~500 mots (taille typique d'un article de loi)
        text = ("Article 1. La présente loi régit les conditions de création " +
                "des sociétés commerciales au Cameroun. " * 50)

        result = detector.detect(text)

        # Devrait être < 1000ms
        assert result["processing_time_ms"] < 1000, (
            f"Détection trop lente: {result['processing_time_ms']}ms (>1s)"
        )

    def test_large_text_sampling(self, detector):
        """Test que les textes longs sont échantillonnés (performance)."""
        # Texte très long (>5000 caractères)
        long_text = ("Article 1. La présente loi régit le droit commercial. " * 200)

        result = detector.detect(long_text)

        # Devrait quand même être rapide grâce au sampling
        assert result["processing_time_ms"] < 1500, "Échantillonnage inefficace"
        assert result["language"] == "fr"


class TestLanguageDetectorConsensus:
    """Tests sur la logique de consensus."""

    def test_perfect_consensus_high_confidence(self, detector):
        """Test qu'un texte clair donne un consensus parfait."""
        # Texte très clairement français
        text = """
        Article 1. La République du Cameroun garantit l'égalité devant la loi.
        Article 2. Les citoyens camerounais jouissent des droits fondamentaux.
        Article 3. La Constitution est la loi suprême de la République.
        Article 4. Les juridictions nationales assurent le respect du droit.
        """

        result = detector.detect(text)

        assert result["consensus"] is True
        assert result["confidence"] > 0.80, "Texte clair devrait avoir >80% confiance"

        # Vérifier que les 3 méthodes ont voté FR
        votes = list(result["method_votes"].values())
        assert votes.count("fr") >= 2, "Au moins 2/3 méthodes devraient voter FR"


class TestLanguageDetectorHealthCheck:
    """Tests du health check."""

    def test_health_check_returns_status(self, detector):
        """Test que le health check retourne un statut."""
        health = detector.health_check()

        assert "service" in health
        assert "status" in health
        assert "models" in health

        assert health["service"] == "LanguageDetector"
        assert health["status"] in ["healthy", "degraded"]

    def test_health_check_models_status(self, detector):
        """Test que le health check vérifie chaque modèle."""
        health = detector.health_check()

        models = health["models"]

        # Vérifier présence des 3 modèles
        assert "spacy_fr" in models
        assert "spacy_en" in models
        assert "fasttext" in models

        # Si service healthy, tous modèles devraient être OK
        if health["status"] == "healthy":
            assert "✅" in models["spacy_fr"]
            assert "✅" in models["spacy_en"]
            assert "✅" in models["fasttext"]


class TestLanguageDetectorEdgeCases:
    """Tests des cas limites."""

    def test_text_with_numbers_and_symbols(self, detector):
        """Test texte avec beaucoup de nombres et symboles."""
        text = """
        Article 1234. Les sociétés à responsabilité limitée (SARL) doivent avoir
        un capital minimum de 1.000.000 FCFA (§2). Les statuts doivent mentionner:
        a) La dénomination sociale; b) Le siège social; c) L'objet social (art. 5°).
        Le taux d'imposition est fixé à 30% (trente pour cent).
        """

        result = detector.detect(text)

        # Devrait quand même détecter correctement
        assert result["language"] == "fr"
        assert result["confidence"] > 0.70

    def test_text_with_legal_acronyms(self, detector):
        """Test texte avec acronymes juridiques."""
        text = """
        Article 1. La SARL, la SA, la SNC et la SCS sont régies par l'OHADA.
        Les statuts de la SARL doivent être déposés au RCCM (Registre du Commerce).
        Le DG (Directeur Général) représente la société. Le CA (Conseil d'Administration)
        délibère conformément au Code CIMA et aux règlements CEMAC en vigueur.
        """

        result = detector.detect(text)

        assert result["language"] == "fr"
        # Confiance peut être un peu plus basse avec beaucoup d'acronymes
        assert result["confidence"] > 0.65

    def test_text_exactly_minimum_length(self, detector):
        """Test texte ayant exactement la longueur minimum."""
        # Créer un texte de exactement 50 caractères (padding avec x)
        text = "Article 1. Contenu juridique de 50 caractères.xxxx"
        assert len(text) == 50

        result = detector.detect(text, min_length=50)

        # Devrait passer sans erreur
        assert result["language"] in ["fr", "en"]


# ==================== TESTS D'INTÉGRATION ====================

class TestLanguageDetectorIntegration:
    """Tests d'intégration avec l'API."""

    @pytest.mark.asyncio
    async def test_api_integration_would_work(self, detector):
        """
        Test simulant l'utilisation via l'API.

        Note: Ce n'est pas un vrai test API (pas de client HTTP),
        mais simule le flux d'utilisation.
        """
        # Simuler une requête API
        request_text = """
        Article 1. La présente loi régit les conditions de création
        des sociétés commerciales au Cameroun conformément au droit OHADA.
        """

        # Appeler le service (comme le ferait l'endpoint)
        result = detector.detect(request_text, min_confidence=0.80)

        # Vérifier format de réponse compatible avec DetectLanguageResponse
        assert "language" in result
        assert "confidence" in result
        assert "method_votes" in result
        assert "consensus" in result
        assert "processing_time_ms" in result
        assert "text_length" in result

        # Vérifier types
        assert isinstance(result["language"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["method_votes"], dict)
        assert isinstance(result["consensus"], bool)
        assert isinstance(result["processing_time_ms"], int)
        assert isinstance(result["text_length"], int)
