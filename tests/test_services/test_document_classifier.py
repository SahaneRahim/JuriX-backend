"""
Tests unitaires pour DocumentClassifier.

Tests Phase 1 (MVP Keywords-Only):
- Classification par catégorie (12 tests)
- Validation inputs (4 tests)
- Edge cases (3 tests)
- Performance (2 tests)
- Health check (2 tests)

Total: 23 tests pour Phase 1
Phase 3 ajoutera: 10+ tests ML
"""

import pytest
import time
from typing import List, Tuple

from app.services.document_classifier import DocumentClassifier, DocumentClassificationError


# ==================== FIXTURES ====================

@pytest.fixture
def classifier():
    """Instance partagée du classifier pour tous les tests."""
    return DocumentClassifier()


# ==================== TESTS PAR CATÉGORIE ====================

class TestCategoryClassification:
    """Tests de classification pour chaque catégorie juridique."""

    def test_classify_droit_constitutionnel(self, classifier):
        """Test catégorie 1: Droit Constitutionnel."""
        text = """
        Article 1. La Constitution du Cameroun établit les pouvoirs du Président de la République.
        Le Parlement comprend l'Assemblée Nationale et le Sénat. Les institutions démocratiques
        sont fondées sur le pouvoir exécutif et le pouvoir législatif. Un référendum peut être
        organisé pour modifier la Constitution.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        # La catégorie 1 doit être dans le top-3
        category_ids = [r[0] for r in result]
        assert 1 in category_ids
        # Vérifier méthode
        assert result[0][2] in ['keyword', 'hybrid']

    def test_classify_droit_civil(self, classifier):
        """Test catégorie 2: Droit Civil."""
        text = """
        Article 1. Le Code civil régit les contrats et les obligations entre débiteur et créancier.
        La responsabilité civile s'applique en cas de dommages. Les délais de prescription sont
        fixés par la loi. Les obligations contractuelles doivent être respectées.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 2 in category_ids

    def test_classify_droit_penal(self, classifier):
        """Test catégorie 3: Droit Pénal."""
        text = """
        Article 1. Le Code pénal définit les crimes et délits passibles d'emprisonnement.
        Toute infraction entraîne une peine. Le tribunal correctionnel juge les délits.
        Les procédures criminelles sont régies par la loi pénale.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 3 in category_ids

    def test_classify_droit_commercial_ohada(self, classifier):
        """Test catégorie 4: Droit Commercial OHADA."""
        text = """
        Article 1. L'acte uniforme OHADA régit les sociétés commerciales au Cameroun.
        La SARL doit avoir un capital social minimum. Les parts sociales sont réparties
        entre les associés. Les actionnaires élisent les dirigeants de la SA.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 4 in category_ids
        # Cette catégorie devrait avoir une bonne confiance
        top_category = result[0]
        if top_category[0] == 4:
            assert top_category[1] > 0.60

    def test_classify_droit_travail(self, classifier):
        """Test catégorie 5: Droit du Travail."""
        text = """
        Article 1. Le Code du travail régit les relations entre salarié et employeur.
        Le licenciement doit respecter la procédure légale. Le contrat de travail définit
        les conditions d'emploi. Les congés payés et le salaire sont garantis par la loi.
        L'inspection du travail veille au respect des droits.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 5 in category_ids

    def test_classify_droit_fiscal(self, classifier):
        """Test catégorie 6: Droit Fiscal."""
        text = """
        Article 1. La fiscalité camerounaise comprend l'impôt sur les sociétés (IS),
        la taxe sur la valeur ajoutée (TVA), et l'impôt sur le revenu des personnes
        physiques (IRPP). Les contribuables doivent déclarer leurs revenus.
        L'administration fiscale assure le recouvrement des impôts et taxes.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 6 in category_ids

    def test_classify_droit_administratif(self, classifier):
        """Test catégorie 7: Droit Administratif."""
        text = """
        Article 1. Le service public est assuré par la fonction publique.
        Les agents publics sont régis par le statut de la fonction publique.
        Le contentieux administratif relève de la juridiction administrative.
        L'autorité administrative exerce le pouvoir réglementaire.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 7 in category_ids

    def test_classify_droit_foncier(self, classifier):
        """Test catégorie 8: Droit Foncier."""
        text = """
        Article 1. Le titre foncier confère la propriété foncière d'un terrain.
        L'immatriculation au cadastre est obligatoire. Le domaine national et
        les domaines privés sont régis par la loi foncière. Le lotissement
        doit être autorisé par l'administration.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 8 in category_ids

    def test_classify_droit_famille(self, classifier):
        """Test catégorie 9: Droit de la Famille."""
        text = """
        Article 1. Le mariage et le divorce sont régis par le droit de la famille.
        La succession est ouverte au décès. L'autorité parentale appartient aux parents.
        La filiation établit le lien de parenté. Le conjoint et les héritiers ont
        des droits successoraux. La tutelle protège les mineurs.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 9 in category_ids

    def test_classify_droit_environnement(self, classifier):
        """Test catégorie 10: Droit de l'Environnement."""
        text = """
        Article 1. La protection de l'environnement est un objectif national.
        La pollution de l'air et de l'eau est interdite. Les ressources naturelles
        doivent être préservées. Le développement durable guide les politiques
        écologiques. La faune et la flore sont protégées.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 10 in category_ids

    def test_classify_droit_international(self, classifier):
        """Test catégorie 11: Droit International."""
        text = """
        Article 1. Le traité international doit être ratifié par le Parlement.
        Les conventions des Nations Unies s'appliquent au Cameroun. Les accords
        bilatéraux renforcent la coopération. La souveraineté nationale est
        respectée dans les relations internationales.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 11 in category_ids

    def test_classify_droit_affaires(self, classifier):
        """Test catégorie 12: Droit des Affaires."""
        text = """
        Article 1. Le contrat commercial régit les relations d'affaires.
        La franchise et le bail commercial sont des contrats spéciaux.
        Le droit de la concurrence protège le marché. Le registre du commerce
        enregistre les commerçants et sociétés.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 1
        category_ids = [r[0] for r in result]
        assert 12 in category_ids


# ==================== TESTS VALIDATION INPUTS ====================

class TestInputValidation:
    """Tests de validation des paramètres d'entrée."""

    def test_empty_text_raises_error(self, classifier):
        """Test que texte vide lève ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            classifier.classify("")

    def test_whitespace_only_text_raises_error(self, classifier):
        """Test que texte avec espaces seulement lève ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            classifier.classify("   \n\t   ")

    def test_text_too_short_raises_error(self, classifier):
        """Test que texte trop court lève ValueError."""
        short_text = "Article 1. Court."  # <50 chars

        with pytest.raises(ValueError, match="Texte trop court"):
            classifier.classify(short_text)

    def test_text_too_long_raises_error(self, classifier):
        """Test que texte trop long lève ValueError."""
        long_text = "A" * 51000  # >50000 chars

        with pytest.raises(ValueError, match="Texte trop long"):
            classifier.classify(long_text)

    def test_top_k_invalid_raises_error(self, classifier):
        """Test que top_k invalide lève ValueError."""
        text = "A" * 100

        with pytest.raises(ValueError, match="top_k"):
            classifier.classify(text, top_k=0)

        with pytest.raises(ValueError, match="top_k"):
            classifier.classify(text, top_k=13)


# ==================== TESTS EDGE CASES ====================

class TestEdgeCases:
    """Tests des cas limites."""

    def test_top_k_1_returns_single_result(self, classifier):
        """Test que top_k=1 retourne 1 seul résultat."""
        text = """
        Article 1. La SARL est une société commerciale régie par l'acte uniforme OHADA.
        """

        result = classifier.classify(text, top_k=1)

        assert len(result) == 1
        cat_id, confidence, method = result[0]
        assert 1 <= cat_id <= 12
        assert 0 <= confidence <= 1
        assert method in ['keyword', 'hybrid']

    def test_min_confidence_filters_results(self, classifier):
        """Test que min_confidence filtre les résultats."""
        text = """
        Article 1. La SARL est une société commerciale régie par l'acte uniforme OHADA.
        """

        # Sans filtre
        result_all = classifier.classify(text, top_k=12, min_confidence=0.0)

        # Avec filtre élevé
        result_filtered = classifier.classify(text, top_k=12, min_confidence=0.80)

        # Devrait filtrer certains résultats
        assert len(result_filtered) <= len(result_all)

        # Tous les résultats filtrés doivent avoir confiance >= 0.80
        for cat_id, confidence, method in result_filtered:
            assert confidence >= 0.80

    def test_ambiguous_text_multiple_categories(self, classifier):
        """Test texte ambigu qui peut correspondre à plusieurs catégories."""
        text = """
        Article 1. Les sociétés commerciales sont soumises à l'impôt sur les sociétés.
        La fiscalité des entreprises est régie par le Code général des impôts.
        Les contribuables doivent respecter leurs obligations contractuelles.
        """

        result = classifier.classify(text, top_k=3)

        assert len(result) >= 2
        # Devrait avoir Droit Fiscal (6) et/ou Droit Commercial (4)
        category_ids = [r[0] for r in result]
        assert 6 in category_ids or 4 in category_ids


# ==================== TESTS PERFORMANCE ====================

class TestPerformance:
    """Tests de performance."""

    def test_classification_time_under_500ms(self, classifier):
        """Test que classification prend <500ms (MVP)."""
        text = """
        Article 1. L'acte uniforme OHADA régit les sociétés commerciales.
        """ * 10  # Texte moyen

        start = time.time()
        result = classifier.classify(text, top_k=3)
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Classification trop lente: {elapsed:.3f}s > 0.5s"
        assert len(result) >= 1

    def test_classification_returns_sorted_by_confidence(self, classifier):
        """Test que résultats sont triés par confiance décroissante."""
        text = """
        Article 1. L'acte uniforme OHADA régit les sociétés commerciales au Cameroun.
        La SARL doit avoir un capital social minimum. Les parts sociales sont
        réparties entre les associés.
        """

        result = classifier.classify(text, top_k=5)

        # Vérifier ordre décroissant
        confidences = [r[1] for r in result]
        assert confidences == sorted(confidences, reverse=True)


# ==================== TESTS HEALTH CHECK ====================

class TestHealthCheck:
    """Tests du health check."""

    def test_health_check_returns_status(self, classifier):
        """Test que health_check retourne un statut."""
        health = classifier.health_check()

        assert health["service"] == "DocumentClassifier"
        assert health["status"] == "healthy"
        assert health["mode"] in ["keywords-only", "hybrid"]
        assert isinstance(health["ml_model_loaded"], bool)
        assert isinstance(health["vectorizer_loaded"], bool)
        assert health["categories_count"] == 12
        assert "version" in health

    def test_health_check_keywords_only_mode(self, classifier):
        """Test health check en mode keywords-only (MVP)."""
        health = classifier.health_check()

        # Phase 1: devrait être en mode keywords-only
        if health["mode"] == "keywords-only":
            assert health["ml_model_loaded"] is False
            assert health["vectorizer_loaded"] is False


# ==================== TESTS UTILITAIRES ====================

class TestUtilityMethods:
    """Tests des méthodes utilitaires."""

    def test_get_category_name_valid(self, classifier):
        """Test récupération nom de catégorie valide."""
        name = classifier.get_category_name(4)
        assert name == "Droit Commercial OHADA"

    def test_get_category_name_invalid(self, classifier):
        """Test récupération nom de catégorie invalide."""
        name = classifier.get_category_name(99)
        assert name is None

    def test_get_all_categories(self, classifier):
        """Test récupération de toutes les catégories."""
        categories = classifier.get_all_categories()

        assert isinstance(categories, dict)
        assert len(categories) == 12
        assert 1 in categories
        assert 12 in categories
        assert categories[1] == "Droit Constitutionnel"
        assert categories[12] == "Droit des Affaires"


# ==================== TESTS PREPROCESSING ====================

class TestPreprocessing:
    """Tests du preprocessing."""

    def test_preprocessing_lowercase(self, classifier):
        """Test que preprocessing convertit en minuscules."""
        text_upper = "ARTICLE 1. LA SARL EST UNE SOCIÉTÉ."
        text_lower = "article 1. la sarl est une société."

        result_upper = classifier.classify(text_upper * 2, top_k=1)
        result_lower = classifier.classify(text_lower * 2, top_k=1)

        # Devrait donner même résultat
        assert result_upper[0][0] == result_lower[0][0]

    def test_preprocessing_removes_article_numbers(self, classifier):
        """Test que preprocessing supprime les numéros d'articles."""
        text_with_numbers = """
        Article 1. La SARL est régie par l'OHADA.
        Art. 2. Le capital social est divisé.
        Article 3. Les parts sociales sont nominatives.
        """

        # Devrait quand même classifier correctement
        result = classifier.classify(text_with_numbers, top_k=1)
        assert len(result) == 1


# ==================== TESTS ML MODE ====================

class TestMLMode:
    """Tests spécifiques au mode ML (hybrid)."""

    def test_ml_models_loaded(self, classifier):
        """Test que les modèles ML sont chargés."""
        assert classifier.ml_model is not None, "ML model should be loaded"
        assert classifier.vectorizer is not None, "Vectorizer should be loaded"

    def test_classify_returns_hybrid_method(self, classifier):
        """Test que la méthode retournée est 'hybrid' quand ML activé."""
        text = """
        Article 1. La constitution établit les pouvoirs du parlement.
        Le président de la république exerce le pouvoir exécutif.
        L'assemblée nationale vote les lois.
        """

        result = classifier.classify(text, top_k=1)

        assert len(result) > 0
        category_id, confidence, method = result[0]
        assert method == 'hybrid', f"Expected 'hybrid' method, got '{method}'"

    def test_hybrid_vs_keyword_comparison(self, classifier):
        """Test comparaison hybrid vs keyword sur même texte."""
        text = """
        Article 1. La SARL et la SA sont régies par l'acte uniforme OHADA.
        Le capital social est divisé en parts sociales pour la SARL.
        Les associés et actionnaires disposent de droits spécifiques.
        """ * 2

        # Mode hybrid
        result_hybrid = classifier.classify(text, top_k=3)

        # Mode keyword (désactiver ML temporairement)
        ml_backup = classifier.ml_model
        vec_backup = classifier.vectorizer
        classifier.ml_model = None
        classifier.vectorizer = None

        result_keyword = classifier.classify(text, top_k=3)

        # Restaurer
        classifier.ml_model = ml_backup
        classifier.vectorizer = vec_backup

        # Vérifier que les deux modes détectent catégorie 4 (Droit Commercial OHADA)
        hybrid_categories = [r[0] for r in result_hybrid]
        keyword_categories = [r[0] for r in result_keyword]

        assert 4 in hybrid_categories, "Hybrid should detect category 4"
        assert 4 in keyword_categories, "Keyword should detect category 4"

    def test_ml_confidence_scores_valid(self, classifier):
        """Test que les scores ML sont dans [0, 1]."""
        text = """
        Article 1. Le salarié et l'employeur sont liés par un contrat de travail.
        Le licenciement doit respecter la procédure du code du travail.
        Le salaire et les congés sont des droits du salarié.
        """

        result = classifier.classify(text, top_k=3)

        for category_id, confidence, method in result:
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"
            assert method == 'hybrid'

    def test_ml_top3_probabilities_sum_valid(self, classifier):
        """Test que les probabilités top-3 sont cohérentes."""
        text = """
        Article 1. L'impôt sur les sociétés (IS) et la TVA sont régis par le code général des impôts.
        Le contribuable doit déclarer ses revenus à l'administration fiscale.
        Le recouvrement des taxes est effectué par le service des impôts.
        """

        result = classifier.classify(text, top_k=3)

        # Vérifier qu'on a 3 résultats
        assert len(result) == 3

        # Les confidences doivent être décroissantes
        confidences = [conf for _, conf, _ in result]
        assert confidences == sorted(confidences, reverse=True), "Confidences should be in descending order"

    def test_ml_performance_under_2s(self, classifier):
        """Test que le temps de réponse ML est <2s."""
        import time

        text = """
        Article 1. La présente loi fixe les règles relatives à la protection de l'environnement.
        La pollution de l'air, de l'eau et du sol est interdite.
        Le développement durable et la protection de la faune et la flore sont prioritaires.
        Les ressources naturelles doivent être préservées pour les générations futures.
        """ * 10  # Texte plus long

        start = time.time()
        result = classifier.classify(text, top_k=3)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"ML classification took {elapsed:.2f}s (>2s threshold)"
        assert len(result) == 3

    def test_ml_consistent_results(self, classifier):
        """Test que ML donne résultats cohérents sur appels multiples."""
        text = """
        Article 1. Le mariage et le divorce sont régis par le code de la famille.
        La succession et la filiation sont déterminées par la loi.
        L'autorité parentale appartient aux deux conjoints.
        """

        # Classifier 3 fois
        result1 = classifier.classify(text, top_k=1)
        result2 = classifier.classify(text, top_k=1)
        result3 = classifier.classify(text, top_k=1)

        # Devrait donner même catégorie
        assert result1[0][0] == result2[0][0] == result3[0][0]

        # Confidences devraient être très proches (tolérance numérique)
        conf1, conf2, conf3 = result1[0][1], result2[0][1], result3[0][1]
        assert abs(conf1 - conf2) < 0.001
        assert abs(conf2 - conf3) < 0.001
