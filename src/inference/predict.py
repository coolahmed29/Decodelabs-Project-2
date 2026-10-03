"""
PURPOSE: Ek naye/unseen transaction (single record ya batch) ko saved production
         pipeline (Phase 12 se) ke through predict karna, STRICT input validation
         ke sath (Section 24). Ye woh module hai jo "real world" se aane wale
         data ko handle karega — isliye ye SABSE defensive/paranoid file honi
         chahiye pure project mein.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.selection import load_model

logger = logging.getLogger(__name__)

MAX_FIELD_LENGTH = 1000


class InferenceService:
    """Saved production pipeline ke through predictions, strict validation ke sath."""

    def __init__(
        self,
        model_path: Path,
        required_fields: list[str],
        categorical_fields: list[str] | None = None,
        numerical_fields: list[str] | None = None,
        non_negative_fields: list[str] | None = None,
        threshold: float = 0.5,
        model_version: str = "unknown",
    ):
        self.model = load_model(model_path)
        self.model_path = Path(model_path)
        self.required_fields = required_fields
        self.categorical_fields = categorical_fields or []
        self.numerical_fields = numerical_fields or []
        self.non_negative_fields = non_negative_fields or []
        self.threshold = threshold
        self.model_version = model_version

        logger.info("Model loaded from %s", self.model_path)

    def validate_input(self, transaction: dict) -> None:
        """Section 24 ki sab checks apply karta hai; koi exception nahi -> valid."""
        if not isinstance(transaction, dict):
            raise ValueError("Transaction must be a dictionary")

        missing_fields = [f for f in self.required_fields if f not in transaction]
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

        for field in self.numerical_fields:
            value = transaction[field]
            if value is None:
                raise ValueError(f"Field '{field}' cannot be None")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Invalid data type for numeric field '{field}': "
                    f"expected int/float, got {type(value).__name__}"
                )
            if np.isnan(value) or np.isinf(value):
                raise ValueError(
                    f"Field '{field}' cannot be NaN or infinite"
                )
            if field in self.non_negative_fields and value < 0:
                raise ValueError("Transaction amount cannot be negative")

        for field in self.categorical_fields:
            value = transaction[field]
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid data type for categorical field '{field}': "
                    f"expected string, got {type(value).__name__}"
                )

        oversize_fields = [
            f
            for f, value in transaction.items()
            if isinstance(value, str) and len(value) > MAX_FIELD_LENGTH
        ]
        if oversize_fields:
            raise ValueError(
                "Oversized field value for fields: "
                + ", ".join(oversize_fields)
            )

    def predict_single(self, transaction: dict) -> dict:
        """Single transaction ke liye prediction + Section 23 ka exact response format."""
        self.validate_input(transaction)

        df = pd.DataFrame([transaction])
        fraud_probability = self.model.predict_proba(df)[:, 1][0]
        is_fraud = bool(fraud_probability >= self.threshold)

        result = {
            "fraud_probability": float(fraud_probability),
            "is_fraud": is_fraud,
            "model_version": self.model_version,
        }

        logger.info("Prediction made, is_fraud=%s", is_fraud)
        return result

    def predict_batch(self, transactions: list[dict]) -> list[dict]:
        """Har record ka prediction; ek bad record pure batch ko fail nahi karta."""
        results = []
        for i, transaction in enumerate(transactions):
            try:
                results.append(self.predict_single(transaction))
            except Exception as exc:
                results.append({"error": str(exc), "index": i})
        return results