"""
PURPOSE: sanitize_for_logging() ki redaction behavior (Section 36) aur
         setup_logging() ker overall smoke test.
"""

import pytest

from src.utils.logging_config import sanitize_for_logging, setup_logging


def test_sanitize_redacts_sensitive_keys():
    data = {"password": "secret123", "amount": 250.0}

    sanitized = sanitize_for_logging(data)

    assert sanitized["password"] == "***REDACTED***"
    assert sanitized["amount"] == 250.0


def test_sanitize_is_case_insensitive():
    data = {"Password": "secret123"}

    sanitized = sanitize_for_logging(data)

    assert sanitized["Password"] == "***REDACTED***"


def test_sanitize_does_not_mutate_original_dict():
    data = {"password": "secret123", "amount": 250.0}

    sanitize_for_logging(data)

    assert data["password"] == "secret123"
    assert data["amount"] == 250.0


def test_setup_logging_runs_without_error():
    setup_logging()  # koi exception NAHI aani chahiye