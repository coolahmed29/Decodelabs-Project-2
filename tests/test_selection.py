"""
PURPOSE: Model comparison, selection rule, aur serialization round-trip
         (save -> load -> predict match) ko test karna.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE, SMOTE_SAMPLING_STRATEGY
from src.data.splitter import split_train_test
from src.features.preprocessing import build_preprocessor
from src.models.selection import (
    compare_models,
    load_model,
    save_model,
    select_production_model,
    verify_serialization_integrity,
)
from src.models.train import train_logistic_regression

NUMERICAL_FEATURES = ["amount", "age"]
CATEGORICAL_FEATURES = ["channel"]
TARGET = "is_fraud"

CHANNELS = ["online", "in_store", "mobile"]


@pytest.fixture
def preprocessor():
    return build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


@pytest.fixture
def fitted_pipeline_and_sample():
    """Chhota real fitted pipeline + eske sample X (predict ke liye)."""
    df = pd.DataFrame(
        {
            "amount": [100.0 + i for i in range(200)],
            "age": [20.0 + i % 60 for i in range(200)],
            "channel": [CHANNELS[i % len(CHANNELS)] for i in range(200)],
            TARGET: [0] * 180 + [1] * 20,
        }
    )
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, _ = split_train_test(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = train_logistic_regression(
        X_train,
        y_train,
        build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES),
        random_state=RANDOM_STATE,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    )
    return pipeline, X_test


# ---------- compare_models ----------
def test_compare_models_returns_dataframe_with_model_column():
    model_results = {
        "logistic_regression": {"precision": 0.7, "recall": 0.6, "roc_auc": 0.85},
        "random_forest": {"precision": 0.8, "recall": 0.7, "roc_auc": 0.9},
    }

    df = compare_models(model_results)

    assert isinstance(df, pd.DataFrame)
    assert "model" in df.columns
    assert set(df["model"]) == {"logistic_regression", "random_forest"}


# ---------- select_production_model ----------
def test_selects_model_with_highest_roc_auc_among_valid_candidates():
    model_results = {
        "logistic_regression": {"precision": 0.7, "recall": 0.6, "roc_auc": 0.85},
        "random_forest": {"precision": 0.8, "recall": 0.7, "roc_auc": 0.95},
    }

    selected = select_production_model(model_results, min_recall=0.5, min_precision=0.5)

    assert selected == "random_forest"


def test_fallback_to_highest_recall_when_none_qualify():
    model_results = {
        "logistic_regression": {"precision": 0.7, "recall": 0.6, "roc_auc": 0.85},
        "random_forest": {"precision": 0.8, "recall": 0.7, "roc_auc": 0.9},
    }

    selected = select_production_model(model_results, min_recall=0.99, min_precision=0.99)

    assert selected == "random_forest"  # highest recall (0.7 > 0.6)


def test_single_model_meeting_criteria_is_selected():
    # RF ka roc_auc zyada hai, lekin LR hi min_recall satisfy karta hai.
    model_results = {
        "logistic_regression": {"precision": 0.5, "recall": 0.6, "roc_auc": 0.99},
        "random_forest": {"precision": 0.8, "recall": 0.3, "roc_auc": 1.0},
    }

    selected = select_production_model(model_results, min_recall=0.5, min_precision=0.5)

    assert selected == "logistic_regression"


# ---------- save_model / load_model ----------
def test_save_model_creates_file_with_correct_name(tmp_path):
    model_name = "test_model"
    version = "v1"

    saved_path = save_model({"a": 1}, model_name, version, tmp_path)

    assert saved_path.exists()
    assert saved_path.name == f"{model_name}_{version}.joblib"


def test_load_model_raises_error_for_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        load_model(tmp_nonexistent_path())


def tmp_nonexistent_path():
    import tempfile

    return "/nonexistent/path/model_v1.joblib"


def test_save_then_load_returns_equivalent_object(fitted_pipeline_and_sample, tmp_path):
    pipeline, X_test = fitted_pipeline_and_sample

    original_proba = pipeline.predict_proba(X_test)
    saved_path = save_model(pipeline, "logistic_regression", "v1", tmp_path)
    loaded_pipeline = load_model(saved_path)
    loaded_proba = loaded_pipeline.predict_proba(X_test)

    assert np.allclose(original_proba, loaded_proba)


# ---------- verify_serialization_integrity ----------
def test_verify_serialization_integrity_returns_true_for_valid_pipeline(
    fitted_pipeline_and_sample, tmp_path
):
    pipeline, X_test = fitted_pipeline_and_sample

    result = verify_serialization_integrity(pipeline, X_test, tmp_path / "check.joblib")

    assert result is True