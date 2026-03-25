"""
Service de validation et sécurisation des données d'entrée.

Ce service gère:
- Validation schéma (Pydantic)
- Prévention injection SQL
- Prévention XSS
- Rate limiting

Objectif: Sécuriser toutes les entrées utilisateur.

Usage:
    service = ValidationService()
    is_safe, reason = service.validate_sql_input("user input")
    cleaned_html = service.sanitize_html("<p>content</p>")
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from app.utils.validation_utils import (
    check_length_limits,
    contains_xss,
    is_safe_sql_input,
    normalize_whitespace,
    sanitize_filename,
    sanitize_html,
    sanitize_sql_input,
    validate_email,
    validate_pattern,
    validate_url,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception levée lors d'erreurs de validation."""

    pass


class RateLimitExceeded(Exception):
    """Exception levée quand rate limit dépassé."""

    pass


class ValidationService:
    """
    Service de validation et sécurisation.

    Fonctionnalités:
    - Validation SQL injection
    - Sanitization XSS
    - Rate limiting
    - Validation formats (email, URL)

    Attributes:
        rate_limit_data: Données de rate limiting
        rate_limit_lock: Lock pour thread safety
    """

    def __init__(self):
        """Initialise le service de validation."""
        logger.info("🚀 Initialisation du ValidationService...")

        # Rate limiting storage (in-memory)
        # Format: {key: [(timestamp, count), ...]}
        self.rate_limit_data: Dict[str, list] = defaultdict(list)
        self.rate_limit_lock = Lock()

        logger.info("✅ ValidationService initialisé")

    # ==================== SQL INJECTION PREVENTION ====================

    def validate_sql_input(
        self, input_str: str, strict: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        NOTE: Not recursive — delegates to imported ``validate_sql_input`` from ``app.utils.validation_utils``.

        Valide une entrée contre injection SQL.

        Args:
            input_str: Entrée à valider
            strict: Mode strict (défaut: True)

        Returns:
            Tuple (is_safe, reason)

        Example:
            >>> service.validate_sql_input("John Doe")
            (True, None)
            >>> service.validate_sql_input("'; DROP TABLE users;")
            (False, "Contains SQL keyword: DROP")
        """
        return is_safe_sql_input(input_str, strict=strict)

    def sanitize_sql_input(self, input_str: str) -> str:
        """
        NOTE: Not recursive — delegates to imported ``sanitize_sql_input`` from ``app.utils.validation_utils``.

        Sanitize une entrée SQL.

        Note: Toujours utiliser des requêtes paramétrées en priorité.

        Args:
            input_str: Entrée à sanitizer

        Returns:
            Entrée sanitizée

        Example:
            >>> service.sanitize_sql_input("John'; DROP TABLE")
            "John DROP TABLE"
        """
        return sanitize_sql_input(input_str)

    # ==================== XSS PREVENTION ====================

    def check_xss(self, input_str: str) -> Tuple[bool, Optional[str]]:
        """
        NOTE: Not recursive — delegates to imported ``check_xss`` from ``app.utils.validation_utils``.

        Vérifie si l'entrée contient du XSS.

        Args:
            input_str: Entrée à vérifier

        Returns:
            Tuple (contains_xss, pattern_found)

        Example:
            >>> service.check_xss("<script>alert('XSS')</script>")
            (True, "script tag")
        """
        return contains_xss(input_str)

    def sanitize_html(
        self, html: str, allowed_tags: Optional[list] = None
    ) -> str:
        """
        NOTE: Not recursive — delegates to imported ``sanitize_html`` from ``app.utils.validation_utils``.

        Sanitize HTML pour prévenir XSS.

        Args:
            html: HTML à sanitizer
            allowed_tags: Tags autorisés (défaut: p, br, strong, em, u, a, ul, ol, li)

        Returns:
            HTML sanitizé

        Example:
            >>> service.sanitize_html("<p>Safe</p><script>alert('XSS')</script>")
            "<p>Safe</p>"
        """
        return sanitize_html(html, allowed_tags=allowed_tags)

    # ==================== RATE LIMITING ====================

    def check_rate_limit(
        self, key: str, limit: int = 100, window_seconds: int = 60
    ) -> Tuple[bool, Optional[str]]:
        """
        Vérifie le rate limit pour une clé.

        Utilise sliding window algorithm.

        Args:
            key: Clé unique (ex: IP, user_id)
            limit: Nombre max de requêtes
            window_seconds: Fenêtre en secondes

        Returns:
            Tuple (is_allowed, reason)

        Raises:
            RateLimitExceeded: Si limite dépassée

        Example:
            >>> service.check_rate_limit("192.168.1.1", limit=10, window_seconds=60)
            (True, None)
        """
        with self.rate_limit_lock:
            current_time = time.time()
            window_start = current_time - window_seconds

            # Nettoyer les anciennes entrées
            self.rate_limit_data[key] = [
                timestamp
                for timestamp in self.rate_limit_data[key]
                if timestamp > window_start
            ]

            # Compter les requêtes dans la fenêtre
            request_count = len(self.rate_limit_data[key])

            if request_count >= limit:
                return False, f"Rate limit exceeded: {request_count}/{limit} in {window_seconds}s"

            # Ajouter la requête actuelle
            self.rate_limit_data[key].append(current_time)

            return True, None

    def reset_rate_limit(self, key: str) -> None:
        """
        Reset le rate limit pour une clé.

        Args:
            key: Clé à reset

        Example:
            >>> service.reset_rate_limit("192.168.1.1")
        """
        with self.rate_limit_lock:
            if key in self.rate_limit_data:
                del self.rate_limit_data[key]
                logger.debug(f"Rate limit reset pour: {key}")

    def get_rate_limit_stats(self, key: str, window_seconds: int = 60) -> Dict[str, Any]:
        """
        Récupère les stats de rate limit.

        Args:
            key: Clé à vérifier
            window_seconds: Fenêtre en secondes

        Returns:
            Dictionnaire avec stats

        Example:
            >>> service.get_rate_limit_stats("192.168.1.1")
            {'requests': 5, 'window_seconds': 60, 'oldest_request': 1234567890}
        """
        with self.rate_limit_lock:
            current_time = time.time()
            window_start = current_time - window_seconds

            # Filtrer les requêtes dans la fenêtre
            requests_in_window = [
                timestamp
                for timestamp in self.rate_limit_data.get(key, [])
                if timestamp > window_start
            ]

            return {
                "requests": len(requests_in_window),
                "window_seconds": window_seconds,
                "oldest_request": min(requests_in_window) if requests_in_window else None,
                "newest_request": max(requests_in_window) if requests_in_window else None,
            }

    # ==================== FORMAT VALIDATION ====================

    def validate_email(self, email: str) -> Tuple[bool, Optional[str]]:
        """
        NOTE: Not recursive — delegates to imported ``validate_email`` from ``app.utils.validation_utils``.

        Valide un email.

        Args:
            email: Email à valider

        Returns:
            Tuple (is_valid, reason)

        Example:
            >>> service.validate_email("user@example.com")
            (True, None)
        """
        if validate_email(email):
            return True, None
        else:
            return False, "Invalid email format"

    def validate_url(
        self, url: str, allowed_schemes: Optional[list] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        NOTE: Not recursive — delegates to imported ``validate_url`` from ``app.utils.validation_utils``.

        Valide une URL.

        Args:
            url: URL à valider
            allowed_schemes: Schémas autorisés (défaut: http, https)

        Returns:
            Tuple (is_valid, reason)

        Example:
            >>> service.validate_url("https://example.com")
            (True, None)
        """
        return validate_url(url, allowed_schemes=allowed_schemes)

    # ==================== GENERAL VALIDATION ====================

    def validate_length(
        self,
        input_str: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Valide la longueur d'une chaîne.

        Args:
            input_str: Chaîne à valider
            min_length: Longueur minimale
            max_length: Longueur maximale

        Returns:
            Tuple (is_valid, reason)

        Example:
            >>> service.validate_length("Hello", min_length=3, max_length=10)
            (True, None)
        """
        return check_length_limits(input_str, min_length, max_length)

    def validate_pattern(
        self, input_str: str, pattern: str, pattern_name: str = "pattern"
    ) -> Tuple[bool, Optional[str]]:
        """
        Valide contre un pattern regex.

        Args:
            input_str: Chaîne à valider
            pattern: Pattern regex
            pattern_name: Nom du pattern

        Returns:
            Tuple (is_valid, reason)

        Example:
            >>> service.validate_pattern("ABC123", r"^[A-Z]{3}\\d{3}$", "code")
            (True, None)
        """
        return validate_pattern(input_str, pattern, pattern_name)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize un nom de fichier.

        Args:
            filename: Nom de fichier

        Returns:
            Nom sanitizé

        Example:
            >>> service.sanitize_filename("../../etc/passwd")
            "etc_passwd"
        """
        return sanitize_filename(filename)

    def normalize_text(self, text: str) -> str:
        """
        Normalise le texte (whitespace).

        Args:
            text: Texte à normaliser

        Returns:
            Texte normalisé

        Example:
            >>> service.normalize_text("Hello    World")
            "Hello World"
        """
        return normalize_whitespace(text)

    # ==================== COMBINED VALIDATION ====================

    def validate_input(
        self,
        input_str: str,
        check_sql: bool = True,
        check_xss: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[bool, list]:
        """
        Validation combinée d'une entrée.

        Args:
            input_str: Entrée à valider
            check_sql: Vérifier SQL injection
            check_xss: Vérifier XSS
            min_length: Longueur minimale
            max_length: Longueur maximale

        Returns:
            Tuple (is_valid, errors)

        Example:
            >>> service.validate_input("Hello World", min_length=5, max_length=20)
            (True, [])
        """
        errors = []

        # Check SQL injection
        if check_sql:
            is_safe, reason = self.validate_sql_input(input_str)
            if not is_safe:
                errors.append(f"SQL injection detected: {reason}")

        # Check XSS
        if check_xss:
            has_xss, pattern = self.check_xss(input_str)
            if has_xss:
                errors.append(f"XSS detected: {pattern}")

        # Check length
        if min_length is not None or max_length is not None:
            is_valid_length, reason = self.validate_length(
                input_str, min_length, max_length
            )
            if not is_valid_length:
                errors.append(f"Length validation failed: {reason}")

        return len(errors) == 0, errors

    # ==================== HEALTH CHECK ====================

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du service.

        Returns:
            Dictionnaire avec statut
        """
        with self.rate_limit_lock:
            total_keys = len(self.rate_limit_data)
            total_requests = sum(len(v) for v in self.rate_limit_data.values())

        return {
            "service": "ValidationService",
            "status": "healthy",
            "rate_limiting": {
                "active_keys": total_keys,
                "total_requests_tracked": total_requests,
            },
            "features": {
                "sql_injection_prevention": True,
                "xss_prevention": True,
                "rate_limiting": True,
                "format_validation": True,
            },
        }
