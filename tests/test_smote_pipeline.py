"""
PURPOSE: SMOTE integration ko strictly test karna — especially ke test set
         KABHI resample na ho, aur few-minority-sample edge case sahi handle ho.
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, SMOTE_SAMPLING_STRATEGY
from src.features.preprocessing import build_preprocessor
from src.models.pipelines import (
    build_full_pipeline,
    get_safe_k_neighbors,
    verify_no_test_leakage,
)

NUMERICAL_FEATURES = ["amount", "age"]
CATEGORICAL_FEATURES = ["channel"]
TARGET = "is_fraud"

CHANNELS = ["online", "in_store", "mobile"]


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """100 rows, 90 class-0 / 10 class-1, 2 numerical + 1 categorical column."""
    return pd.DataFrame(
        {
            "amount": [100.0 + i for i in range(100)],
            "age": [20.0 + i % 40 for i in range(100)],
            "channel": [CHANNELS[i % len(CHANNELS)] for i in range(100)],
            TARGET: [0] * 90 + [1] * 10,
        }
    )


@pytest.fixture
def preprocessor():
    return build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


@pytest.fixture
def classifier():
    return RandomForestClassifier(
        n_estimators=20, max_depth=4, random_state=RANDOM_STATE
    )


@pytest.fixture
def full_pipeline(preprocessor, classifier):
    return build_full_pipeline(
        preprocessor=preprocessor,
        classifier=classifier,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )


@pytest.fixture
def split_data(imbalanced_df):
    X = imbalanced_df.drop(columns=[TARGET])
    y = imbalanced_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------- build_full_pipeline structure ----------
def test_pipeline_has_correct_steps_in_order(full_pipeline):
    step_names = [name for name, _ in full_pipeline.steps]

    assert step_names == ["preprocessor", "smote", "classifier"]


def test_pipeline_is_imblearn_type_not_sklearn(full_pipeline):
    # imblearn ka Pipeline sklearn Pipeline ka subclass hai, isliye isinstance
    # se farq nahi pata chalta — exact type check karo. Plain sklearn Pipeline
    # ka type SklearnPipeline hota, ImbPipeline ka ImbPipeline.
    assert type(full_pipeline) is ImbPipeline
    assert type(full_pipeline) is not SklearnPipeline


# ---------- verify_no_test_leakage ----------
def test_verify_no_test_leakage_passes_for_correct_pipeline(full_pipeline):
    verify_no_test_leakage(full_pipeline)


def test_verify_no_test_leakage_fails_if_smote_missing(preprocessor, classifier):
    bad_pipeline = ImbPipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    with pytest.raises(AssertionError):
        verify_no_test_leakage(bad_pipeline)


# ---------- get_safe_k_neighbors ----------
def test_safe_k_neighbors_normal_case():
    y_train = pd.Series([0] * 80 + [1] * 20)

    assert get_safe_k_neighbors(y_train, desired_k=5) == 5


def test_safe_k_neighbors_caps_when_minority_is_small():
    y_train = pd.Series([0] * 97 + [1] * 3)

    assert get_safe_k_neighbors(y_train, desired_k=5) == 2


def test_safe_k_neighbors_raises_when_minority_too_small():
    y_train = pd.Series([0] * 99 + [1] * 1)

    with pytest.raises(ValueError, match="fewer than 2 minority class samples"):
        get_safe_k_neighbors(y_train, desired_k=5)


# ---------- END-TO-END fit behavior (SABSE IMPORTANT) ----------
def test_pipeline_fit_transforms_training_data_only(full_pipeline, split_data):
    X_train, X_test, y_train, _ = split_data

    full_pipeline.fit(X_train, y_train)
    assert "smote" in full_pipeline.named_steps

    predictions = full_pipeline.predict(X_test)

    assert len(predictions) == len(X_test)


def test_class_balance_after_fit_is_more_balanced(full_pipeline, split_data):
    X_train, _, y_train, _ = split_data

    full_pipeline.fit(X_train, y_train)

    preprocessed_X_train = full_pipeline.named_steps["preprocessor"].transform(X_train)
    _, y_resampled = full_pipeline.named_steps["smote"].fit_resample(
        preprocessed_X_train, y_train
    )

    original_ratio = (y_train == 1).mean()
    resampled_ratio = (y_resampled == 1).mean()

    assert resampled_ratio > original_ratio

    # sampling_strategy = minority/majority ratio (median of SMOTE); final
    # class balance close to current strategy par verify karo.
    majority_count = (y_resampled == 0).sum()
    minority_count = (y_resampled == 1).sum()
    assert abs(minority_count / majority_count - SMOTE_SAMPLING_STRATEGY) < 0.05


# ---------- reproducibility ----------
def test_pipeline_with_same_random_state_gives_same_results(
    preprocessor, classifier, split_data
):
    X_train, _, y_train, _ = split_data

    pipeline_1 = build_full_pipeline(
        preprocessor=preprocessor,
        classifier=classifier,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )
    pipeline_2 = build_full_pipeline(
        preprocessor=preprocessor,
        classifier=classifier,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )

    pipeline_1.fit(X_train, y_train)
    pipeline_2.fit(X_train, y_train)

    proba_1 = pipeline_1.predict_proba(X_train)
    proba_2 = pipeline_2.predict_proba(X_train)

    assert np.allclose(proba_1, proba_2)