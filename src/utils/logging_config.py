"""
PURPOSE: Poore project ke liye EK JAGAH se logging configure karna, taake har
         file mein alag-alag logging.basicConfig() na likhna pade, aur format
         consistent rahe (Section 36).
"""

import logging
from pathlib import Path

DEFAULT_SENSITIVE_KEYS = [
    "password",
    "api_key",
    "card_number",
    "cvv",
    "ssn",
    "account_number",
    "token",
]


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """Root logger ko ek consistent format ke sath configure karta hai."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=(
            [logging.FileHandler(log_file), logging.StreamHandler()]
            if log_file is not None
            else None
        ),
    )


def sanitize_for_logging(
    data: dict, sensitive_keys: list[str] | None = None
) -> dict:
    """Sensitive values ko redact kar ke loggable copy return karta hai.

    Note:
        data ki COPY parse karta hai (original modify NAHI hota).
    """
    keys = sensitive_keys if sensitive_keys is not None else DEFAULT_SENSITIVE_KEYS
    sensitive_lower = {key.lower() for key in keys}

    sanitized = dict(data)
    for key, value in sanitized.items():
        if key.lower() in sensitive_lower:
            sanitized[key] = "***REDACTED***"
    return sanitized