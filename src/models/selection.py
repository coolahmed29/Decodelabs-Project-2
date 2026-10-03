"""
PURPOSE: Trained Logistic Regression aur Random Forest models ke evaluation
         results compare karke ek "production candidate" select karna
         (Section 21 ka model-level version — Phase 11 threshold-level tha,
         ye MODEL-level hai), phir final chuna hua pipeline joblib se disk
         par save karna (Section 22).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compare_models(model_results: dict[str, dict]) -> pd.DataFrame:
    """Model evaluation results ko ek DataFrame mein convert karta hai.

    Input format: {"logistic_regression": {"precision": ..., "recall": ...,
    "roc_auc": ...}, "random_forest": {...}}
    """
    results_df = pd.DataFrame.from_dict(model_results, orient="index")
    results_df = results_df.reset_index().rename(columns={"index": "model"})
    return results_df


def select_production_model(
    model_results: dict[str, dict], min_recall: float, min_precision: float
) -> str:
    """Model-level production candidate select karta hai (Section 21 ka exact rule).

    Rule:
        1. recall >= min_recall AND precision >= min_precision wale candidates
        2. Koi candidate qualify na kare -> fallback: HIGHEST RECALL wala model
           (fraud detection mein missed fraud costly hai, isliye recall priority)
        3. Candidates hain -> unmein se HIGHEST roc_auc wala model

    Returns:
        Model ka NAAM (string) — actual pipeline object nahi; caller apne
        paas se object nikalega is naam ke against.
    """
    candidates = {
        name: metrics
        for name, metrics in model_results.items()
        if metrics["recall"] >= min_recall and metrics["precision"] >= min_precision
    }

    if not candidates:
        logger.warning(
            "No model meets both min recall and precision requirements — falling "
            "back to highest recall model"
        )
        return max(model_results, key=lambda name: model_results[name]["recall"])

    return max(candidates, key=lambda name: candidates[name]["roc_auc"])


def save_model(pipeline, model_name: str, version: str, model_dir: Path) -> Path:
    """Pipeline ko joblib se {model_name}_{version}.joblib ke naam par save karta hai.

    Returns:
        full_path (saved file ka Path).
    """
    filename = f"{model_name}_{version}.joblib"
    full_path = model_dir / filename

    joblib.dump(pipeline, full_path)
    logger.info("Model saved to %s", full_path)

    return full_path


def load_model(model_path: Path):
    """Saved pipeline ko joblib se load karke return karta hai.

    Raises:
        FileNotFoundError: Agar model file maujood nahi hai.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return joblib.load(model_path)


def verify_serialization_integrity(
    pipeline, sample_X: pd.DataFrame, temp_save_path: Path
) -> bool:
    """Serialization round-trip check: save -> load -> predict match (Section 35).

    Original pipeline aur loaded pipeline dono se predict_proba(sample_X)
    nikalo; agar np.allclose match ho to True. Mismatch hona NAHI chahiye.
    """
    original_output = pipeline.predict_proba(sample_X)

    joblib.dump(pipeline, temp_save_path)
    loaded_pipeline = joblib.load(temp_save_path)
    loaded_output = loaded_pipeline.predict_proba(sample_X)

    return bool(np.allclose(original_output, loaded_output))