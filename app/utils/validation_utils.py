"""
Utility functions for input validation and sanitization.

Provides helpers for:
- SQL injection prevention
- XSS prevention
- Input sanitization
- Pattern validation
- Length validation

Author: JuriX Development Team
Date: 2026-01-11
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse


# SQL injection dangerous keywords
SQL_DANGEROUS_KEYWORDS = [
    "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "EXEC", "EXECUTE", "UNION", "DECLARE", "CAST", "CONVERT",
    "--", "/*", "*/", "xp_", "sp_", "INFORMATION_SCHEMA"
]

# XSS dangerous patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",  # Event handlers like onclick=
    r"<iframe",
    r"<object",
    r"<embed",
]


def is_safe_sql_input(input_str: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Check if input is safe from SQL injection.

    Args:
        input_str: Input string to check
        strict: If True, apply strict checking

    Returns:
        Tuple of (is_safe, reason)

    Example:
        >>> is_safe_sql_input("John Doe")
        (True, None)
        >>> is_safe_sql_input("'; DROP TABLE users; --")
        (False, "Contains SQL keyword: DROP")
    """
    if not input_str:
        return True, None

    # Check for dangerous SQL keywords
    input_upper = input_str.upper()
    for keyword in SQL_DANGEROUS_KEYWORDS:
        if keyword in input_upper:
            return False, f"Contains SQL keyword: {keyword}"

    # Check for SQL comment patterns
    if "--" in input_str or "/*" in input_str or "*/" in input_str:
        return False, "Contains SQL comment pattern"

    # Check for multiple statements (semicolon)
    if strict and ";" in input_str:
        return False, "Contains statement separator (;)"

    # Check for quote escaping attempts
    if strict and ("\\'" in input_str or '\\"' in input_str):
        return False, "Contains quote escaping"

    return True, None


def sanitize_sql_input(input_str: str) -> str:
    """
    Sanitize input for SQL queries.

    Note: This is a defense-in-depth measure.
    Always use parameterized queries as primary defense.

    Args:
        input_str: Input to sanitize

    Returns:
        Sanitized string

    Example:
        >>> sanitize_sql_input("John'; DROP TABLE")
        "John DROP TABLE"
    """
    if not input_str:
        return ""

    # Remove SQL keywords
    sanitized = input_str
    for keyword in SQL_DANGEROUS_KEYWORDS:
        sanitized = re.sub(
            rf"\b{keyword}\b",
            "",
            sanitized,
            flags=re.IGNORECASE
        )

    # Remove SQL comment patterns
    sanitized = sanitized.replace("--", "")
    sanitized = sanitized.replace("/*", "")
    sanitized = sanitized.replace("*/", "")

    # Remove semicolons
    sanitized = sanitized.replace(";", "")

    # Remove extra whitespace
    sanitized = " ".join(sanitized.split())

    return sanitized.strip()


def contains_xss(input_str: str) -> Tuple[bool, Optional[str]]:
    """
    Check if input contains XSS patterns.

    Args:
        input_str: Input string to check

    Returns:
        Tuple of (contains_xss, pattern_found)

    Example:
        >>> contains_xss("<script>alert('XSS')</script>")
        (True, "script tag")
        >>> contains_xss("Hello World")
        (False, None)
    """
    if not input_str:
        return False, None

    input_lower = input_str.lower()

    # Check for script tags
    if "<script" in input_lower:
        return True, "script tag"

    # Check for javascript: protocol
    if "javascript:" in input_lower:
        return True, "javascript: protocol"

    # Check for event handlers
    if re.search(r"on\w+\s*=", input_lower):
        return True, "event handler"

    # Check for iframe
    if "<iframe" in input_lower:
        return True, "iframe tag"

    # Check for object/embed
    if "<object" in input_lower or "<embed" in input_lower:
        return True, "object/embed tag"

    return False, None


def sanitize_html(html: str, allowed_tags: Optional[list] = None) -> str:
    """
    Sanitize HTML to prevent XSS.

    Args:
        html: HTML string to sanitize
        allowed_tags: List of allowed HTML tags (default: basic formatting)

    Returns:
        Sanitized HTML

    Example:
        >>> sanitize_html("<p>Safe</p><script>alert('XSS')</script>")
        "<p>Safe</p>"
    """
    if not html:
        return ""

    try:
        import bleach

        if allowed_tags is None:
            allowed_tags = ["p", "br", "strong", "em", "u", "a", "ul", "ol", "li"]

        allowed_attrs = {
            "a": ["href", "title"],
        }

        cleaned = bleach.clean(
            html,
            tags=allowed_tags,
            attributes=allowed_attrs,
            strip=True
        )

        return cleaned

    except ImportError:
        # Fallback: strip all HTML tags
        return re.sub(r"<[^>]+>", "", html)


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        True if valid email format

    Example:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid-email")
        False
    """
    if not email:
        return False

    # RFC 5322 simplified pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_url(url: str, allowed_schemes: Optional[list] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate URL format and scheme.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed schemes (default: http, https)

    Returns:
        Tuple of (is_valid, reason)

    Example:
        >>> validate_url("https://example.com")
        (True, None)
        >>> validate_url("javascript:alert('XSS')")
        (False, "Invalid scheme: javascript")
    """
    if not url:
        return False, "Empty URL"

    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]

    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in allowed_schemes:
            return False, f"Invalid scheme: {parsed.scheme}"

        # Check netloc (domain)
        if not parsed.netloc:
            return False, "Missing domain"

        return True, None

    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def check_length_limits(
    input_str: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if string length is within limits.

    Args:
        input_str: String to check
        min_length: Minimum length (optional)
        max_length: Maximum length (optional)

    Returns:
        Tuple of (is_valid, reason)

    Example:
        >>> check_length_limits("Hello", min_length=3, max_length=10)
        (True, None)
        >>> check_length_limits("Hi", min_length=3)
        (False, "Too short: 2 < 3")
    """
    length = len(input_str)

    if min_length is not None and length < min_length:
        return False, f"Too short: {length} < {min_length}"

    if max_length is not None and length > max_length:
        return False, f"Too long: {length} > {max_length}"

    return True, None


def validate_pattern(input_str: str, pattern: str, pattern_name: str = "pattern") -> Tuple[bool, Optional[str]]:
    """
    Validate string against regex pattern.

    Args:
        input_str: String to validate
        pattern: Regex pattern
        pattern_name: Name of pattern for error message

    Returns:
        Tuple of (is_valid, reason)

    Example:
        >>> validate_pattern("ABC123", r"^[A-Z]{3}\\d{3}$", "code")
        (True, None)
        >>> validate_pattern("abc123", r"^[A-Z]{3}\\d{3}$", "code")
        (False, "Does not match code pattern")
    """
    if not re.match(pattern, input_str):
        return False, f"Does not match {pattern_name} pattern"

    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename

    Example:
        >>> sanitize_filename("../../etc/passwd")
        "etc_passwd"
        >>> sanitize_filename("file<script>.txt")
        "file_script_.txt"
    """
    if not filename:
        return "unnamed"

    # Remove path separators
    sanitized = filename.replace("/", "_").replace("\\", "_")

    # Remove dangerous characters
    sanitized = re.sub(r"[<>:\"|?*]", "_", sanitized)

    # Remove leading dots (hidden files)
    sanitized = sanitized.lstrip(".")

    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    return sanitized or "unnamed"


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Text to normalize

    Returns:
        Normalized text

    Example:
        >>> normalize_whitespace("Hello    World\\n\\n\\nTest")
        "Hello World\\nTest"
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with single newline
    text = re.sub(r"\n+", "\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text
