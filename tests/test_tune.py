"""
PURPOSE: tune_logistic_regression() aur tune_random_forest() ko verify karna —
         especially ke TEST SET kahin bhi is process mein pass NAHI kiya ja raha,
         aur best_params_/best_estimator_ properly milte hain.
"""

import inspect

import pandas as pd
import pytest

from src.config import RANDOM_STATE
from src.data.splitter import split_train_test
from src.features.preprocessing import build_preprocessor
from src.models.tune import tune_logistic_regression, tune_random_forest

NUMERICAL_FEATURES = ["amount", "age"]
CATEGORICAL_FEATURES = ["channel"]
TARGET = "is_fraud"

CHANNELS = ["online", "in_store", "mobile"]

CV_FOLDS = 3
N_ITER = 3

LR_PARAM_SPACE = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
RF_PARAM_SPACE = {"classifier__n_estimators": [100, 200, 300, 500]}


@pytest.fixture
def preprocessor():
    return build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


@pytest.fixture
def split_data():
    """400 rows, 360 class-0 / 40 class-1 — cv_folds=3 mein bhi kafi minority."""
    df = pd.DataFrame(
        {
            "amount": [100.0 + i for i in range(400)],
            "age": [20.0 + i % 80 for i in range(400)],
            "channel": [CHANNELS[i % len(CHANNELS)] for i in range(400)],
            TARGET: [0] * 360 + [1] * 40,
        }
    )
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return split_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE)


# ---------- tune_logistic_regression ----------
def test_tune_lr_returns_fitted_search_object(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search = tune_logistic_regression(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert search.best_params_


def test_tune_lr_best_estimator_is_full_pipeline(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search = tune_logistic_regression(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    step_names = [name for name, _ in search.best_estimator_.steps]
    assert step_names == ["preprocessor", "smote", "classifier"]


def test_tune_lr_best_params_are_within_search_space(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search = tune_logistic_regression(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    assert search.best_params_["classifier__C"] in LR_PARAM_SPACE["classifier__C"]


def test_tune_lr_can_predict_after_tuning(split_data, preprocessor):
    X_train, X_test, y_train, _ = split_data

    search = tune_logistic_regression(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    predictions = search.best_estimator_.predict(X_test)

    assert len(predictions) == len(X_test)


# ---------- tune_random_forest ----------
def test_tune_rf_returns_fitted_search_object(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search = tune_random_forest(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert search.best_params_


def test_tune_rf_best_estimator_is_full_pipeline(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search = tune_random_forest(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    step_names = [name for name, _ in search.best_estimator_.steps]
    assert step_names == ["preprocessor", "smote", "classifier"]


def test_tune_rf_best_params_are_within_search_space(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search = tune_random_forest(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    assert (
        search.best_params_["classifier__n_estimators"]
        in RF_PARAM_SPACE["classifier__n_estimators"]
    )


def test_tune_rf_can_predict_after_tuning(split_data, preprocessor):
    X_train, X_test, y_train, _ = split_data

    search = tune_random_forest(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=N_ITER,
    )

    predictions = search.best_estimator_.predict(X_test)

    assert len(predictions) == len(X_test)


# ---------- CRITICAL: no test-set leakage during tuning ----------
def test_tuning_never_receives_test_data():
    for func in (tune_logistic_regression, tune_random_forest):
        parameters = inspect.signature(func).parameters
        assert "X_test" not in parameters
        assert "y_test" not in parameters
        assert "test" not in [name.lower() for name in parameters]


# ---------- reproducibility ----------
def test_tuning_is_reproducible_with_same_random_state(split_data, preprocessor):
    X_train, _, y_train, _ = split_data

    search_1 = tune_logistic_regression(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=4,
    )
    search_2 = tune_logistic_regression(
        X_train, y_train, preprocessor, random_state=RANDOM_STATE, cv_folds=CV_FOLDS,
        n_iter=4,
    )

    assert search_1.best_params_ == search_2.best_params_