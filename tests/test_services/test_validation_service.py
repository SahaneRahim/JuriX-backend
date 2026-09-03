"""
Tests unitaires pour le service ValidationService.

Ces tests vérifient:
- Prévention injection SQL
- Prévention XSS
- Rate limiting
- Validation formats (email, URL)
- Validation longueur et patterns

Usage:
    pytest backend/tests/test_services/test_validation_service.py -v
    pytest backend/tests/test_services/test_validation_service.py -v --cov=backend/app/services/validation_service
"""

import time

import pytest

from app.services.validation_service import ValidationService


@pytest.fixture
def validation_service():
    """Fixture: ValidationService instance."""
    return ValidationService()


# ==================== SQL INJECTION PREVENTION TESTS ====================


class TestSQLInjectionPrevention:
    """Tests de prévention injection SQL."""

    def test_detect_sql_injection_drop(self, validation_service):
        """Test détection DROP TABLE."""
        is_safe, reason = validation_service.validate_sql_input("'; DROP TABLE users; --")
        
        assert is_safe is False
        assert "DROP" in reason

    def test_detect_sql_injection_union(self, validation_service):
        """Test détection UNION SELECT."""
        is_safe, reason = validation_service.validate_sql_input("1' UNION SELECT * FROM passwords")
        
        assert is_safe is False
        assert "UNION" in reason or "SELECT" in reason

    def test_safe_sql_input(self, validation_service):
        """Test qu'une entrée sûre passe."""
        is_safe, reason = validation_service.validate_sql_input("John Doe")
        
        assert is_safe is True
        assert reason is None

    def test_sanitize_sql_input(self, validation_service):
        """Test sanitization SQL."""
        dangerous = "John'; DROP TABLE users; --"
        sanitized = validation_service.sanitize_sql_input(dangerous)
        
        assert "DROP" not in sanitized
        assert "--" not in sanitized
        assert ";" not in sanitized
        # "TABLE" n'est PAS retire, et ne doit pas l'etre : SQL_DANGEROUS_KEYWORDS
        # sert aussi a validate_sql_input, qui rejette toute entree contenant un
        # mot de la liste. Y ajouter TABLE ferait rejeter "table des matieres" —
        # et UNION y figure deja, ce qui rejette "union europeenne". Isole du
        # verbe, le mot ne porte aucune charge.


# ==================== XSS PREVENTION TESTS ====================


class TestXSSPrevention:
    """Tests de prévention XSS."""

    def test_detect_script_tag(self, validation_service):
        """Test détection balise script."""
        has_xss, pattern = validation_service.check_xss("<script>alert('XSS')</script>")
        
        assert has_xss is True
        assert "script" in pattern.lower()

    def test_detect_event_handler(self, validation_service):
        """Test détection event handlers."""
        has_xss, pattern = validation_service.check_xss("<img src=x onerror='alert(1)'>")
        
        assert has_xss is True
        assert "event" in pattern.lower()

    def test_sanitize_html_removes_script(self, validation_service):
        """Test que sanitize_html retire les scripts."""
        dirty = "<p>Safe content</p><script>alert('XSS')</script>"
        clean = validation_service.sanitize_html(dirty)
        
        assert "<p>Safe content</p>" in clean
        assert "<script>" not in clean
        assert "alert" not in clean

    def test_sanitize_html_preserves_safe_tags(self, validation_service):
        """Test que sanitize_html préserve les tags sûrs."""
        safe = "<p>Paragraph</p><strong>Bold</strong><em>Italic</em>"
        clean = validation_service.sanitize_html(safe)
        
        assert "<p>" in clean
        assert "<strong>" in clean
        assert "<em>" in clean


# ==================== RATE LIMITING TESTS ====================


class TestRateLimiting:
    """Tests de rate limiting."""

    def test_rate_limit_enforced(self, validation_service):
        """Test que le rate limit est appliqué."""
        key = "test_user_1"
        limit = 5
        
        # Faire 5 requêtes (devrait passer)
        for i in range(limit):
            is_allowed, reason = validation_service.check_rate_limit(key, limit=limit, window_seconds=60)
            assert is_allowed is True, f"Request {i+1} should be allowed"
        
        # 6ème requête (devrait échouer)
        is_allowed, reason = validation_service.check_rate_limit(key, limit=limit, window_seconds=60)
        assert is_allowed is False
        assert "exceeded" in reason.lower()

    def test_rate_limit_reset(self, validation_service):
        """Test que le reset fonctionne."""
        key = "test_user_2"
        
        # Remplir le rate limit
        for _ in range(5):
            validation_service.check_rate_limit(key, limit=5, window_seconds=60)
        
        # Reset
        validation_service.reset_rate_limit(key)
        
        # Devrait pouvoir refaire des requêtes
        is_allowed, reason = validation_service.check_rate_limit(key, limit=5, window_seconds=60)
        assert is_allowed is True

    def test_different_keys_independent(self, validation_service):
        """Test que différentes clés sont indépendantes."""
        key1 = "user_1"
        key2 = "user_2"
        
        # Remplir rate limit pour key1
        for _ in range(5):
            validation_service.check_rate_limit(key1, limit=5, window_seconds=60)
        
        # key2 devrait toujours fonctionner
        is_allowed, reason = validation_service.check_rate_limit(key2, limit=5, window_seconds=60)
        assert is_allowed is True


# ==================== FORMAT VALIDATION TESTS ====================


class TestFormatValidation:
    """Tests de validation de formats."""

    def test_email_validation_valid(self, validation_service):
        """Test validation email valide."""
        is_valid, reason = validation_service.validate_email("user@example.com")
        
        assert is_valid is True
        assert reason is None

    def test_email_validation_invalid(self, validation_service):
        """Test validation email invalide."""
        is_valid, reason = validation_service.validate_email("invalid-email")
        
        assert is_valid is False
        assert reason is not None

    def test_url_validation_valid(self, validation_service):
        """Test validation URL valide."""
        is_valid, reason = validation_service.validate_url("https://example.com")
        
        assert is_valid is True
        assert reason is None

    def test_url_validation_javascript_blocked(self, validation_service):
        """Test que javascript: est bloqué."""
        is_valid, reason = validation_service.validate_url("javascript:alert('XSS')")
        
        assert is_valid is False
        assert "scheme" in reason.lower()


# ==================== GENERAL VALIDATION TESTS ====================


class TestGeneralValidation:
    """Tests de validation générale."""

    def test_length_validation_valid(self, validation_service):
        """Test validation longueur valide."""
        is_valid, reason = validation_service.validate_length("Hello", min_length=3, max_length=10)
        
        assert is_valid is True
        assert reason is None

    def test_length_validation_too_short(self, validation_service):
        """Test validation trop court."""
        is_valid, reason = validation_service.validate_length("Hi", min_length=5)
        
        assert is_valid is False
        assert "short" in reason.lower()

    def test_length_validation_too_long(self, validation_service):
        """Test validation trop long."""
        is_valid, reason = validation_service.validate_length("Very long text", max_length=5)
        
        assert is_valid is False
        assert "long" in reason.lower()

    def test_pattern_validation(self, validation_service):
        """Test validation pattern."""
        is_valid, reason = validation_service.validate_pattern("ABC123", r"^[A-Z]{3}\d{3}$", "code")
        
        assert is_valid is True
        assert reason is None


# ==================== COMBINED VALIDATION TESTS ====================


class TestCombinedValidation:
    """Tests de validation combinée."""

    def test_validate_input_safe(self, validation_service):
        """Test validation combinée sûre."""
        is_valid, errors = validation_service.validate_input(
            "Hello World",
            check_sql=True,
            check_xss=True,
            min_length=5,
            max_length=20
        )
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_input_sql_injection(self, validation_service):
        """Test validation combinée détecte SQL."""
        is_valid, errors = validation_service.validate_input(
            "'; DROP TABLE users;",
            check_sql=True
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("SQL" in error for error in errors)

    def test_validate_input_xss(self, validation_service):
        """Test validation combinée détecte XSS."""
        is_valid, errors = validation_service.validate_input(
            "<script>alert('XSS')</script>",
            check_xss=True
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("XSS" in error for error in errors)


# ==================== UTILITY TESTS ====================


class TestUtilities:
    """Tests des utilitaires."""

    def test_sanitize_filename(self, validation_service):
        """Test sanitization nom de fichier."""
        dangerous = "../../etc/passwd"
        safe = validation_service.sanitize_filename(dangerous)
        
        assert ".." not in safe
        assert "/" not in safe
        assert "\\" not in safe

    def test_normalize_text(self, validation_service):
        """Test normalisation texte."""
        messy = "Hello    World\n\n\nTest"
        clean = validation_service.normalize_text(messy)
        
        assert "    " not in clean  # No multiple spaces
        assert "\n\n\n" not in clean  # No triple newlines


# ==================== HEALTH CHECK TESTS ====================


class TestHealthCheck:
    """Tests du health check."""

    def test_health_check_returns_status(self, validation_service):
        """Test que le health check retourne un statut."""
        health = validation_service.health_check()
        
        assert "service" in health
        assert "status" in health
        assert "features" in health
        
        assert health["service"] == "ValidationService"
        assert health["status"] == "healthy"

    def test_health_check_features(self, validation_service):
        """Test que toutes les features sont listées."""
        health = validation_service.health_check()
        
        features = health["features"]
        assert features["sql_injection_prevention"] is True
        assert features["xss_prevention"] is True
        assert features["rate_limiting"] is True
        assert features["format_validation"] is True
