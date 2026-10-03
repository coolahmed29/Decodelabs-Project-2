"""
PURPOSE: train_logistic_regression() aur get_predictions() ko synthetic imbalanced
         data par test karna — fit hona, predictions ki shape/range sahi hona,
         aur reproducibility.
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE, SMOTE_SAMPLING_STRATEGY
from src.features.preprocessing import build_preprocessor
from src.data.splitter import split_train_test
from src.models.train import get_predictions, train_logistic_regression

NUMERICAL_FEATURES = ["amount", "age"]
CATEGORICAL_FEATURES = ["channel"]
TARGET = "is_fraud"

CHANNELS = ["online", "in_store", "mobile"]


@pytest.fixture
def preprocessor():
    return build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


@pytest.fixture
def split_data():
    """200 rows, 180 class-0 / 20 class-1 — minority >= 10, SMOTE k=5 ke liye kafi."""
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
    return split_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE)


def _train(split_data, preprocessor):
    X_train, _, y_train, _ = split_data
    return train_logistic_regression(
        X_train,
        y_train,
        preprocessor,
        random_state=RANDOM_STATE,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    )


def test_train_logistic_regression_returns_fitted_pipeline(split_data, preprocessor):
    pipeline = _train(split_data, preprocessor)

    assert isinstance(pipeline, ImbPipeline)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_pipeline_can_predict_on_test_set(split_data, preprocessor):
    pipeline = _train(split_data, preprocessor)
    _, X_test, _, _ = split_data

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)


def test_predictions_are_binary(split_data, preprocessor):
    pipeline = _train(split_data, preprocessor)
    _, X_test, _, _ = split_data

    y_pred, _ = get_predictions(pipeline, X_test)

    assert np.isin(y_pred, [0, 1]).all()


def test_probabilities_are_in_valid_range(split_data, preprocessor):
    pipeline = _train(split_data, preprocessor)
    _, X_test, _, _ = split_data

    _, y_proba = get_predictions(pipeline, X_test)

    assert (y_proba >= 0.0).all()
    assert (y_proba <= 1.0).all()


def test_probability_and_prediction_are_consistent(split_data, preprocessor):
    pipeline = _train(split_data, preprocessor)
    _, X_test, _, _ = split_data

    y_pred, y_proba = get_predictions(pipeline, X_test)

    # Default predict() threshold 0.5 hai — probability aur prediction align honi chahiye.
    assert ((y_proba > 0.5) == (y_pred == 1)).all()


def test_training_is_reproducible(split_data, preprocessor):
    X_train, X_test, y_train, _ = split_data

    pipeline_1 = train_logistic_regression(
        X_train,
        y_train,
        preprocessor,
        random_state=RANDOM_STATE,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    )
    pipeline_2 = train_logistic_regression(
        X_train,
        y_train,
        preprocessor,
        random_state=RANDOM_STATE,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    )

    proba_1 = pipeline_1.predict_proba(X_test)
    proba_2 = pipeline_2.predict_proba(X_test)

    assert np.allclose(proba_1, proba_2)


def test_train_raises_clear_error_with_insufficient_minority_samples(preprocessor):
    # Sirf 1 fraud sample — get_safe_k_neighbors() train-level par bhi fail kare.
    X_train = pd.DataFrame(
        {
            "amount": [100.0 + i for i in range(20)],
            "age": [25.0 + i for i in range(20)],
            "channel": [CHANNELS[i % len(CHANNELS)] for i in range(20)],
        }
    )
    y_train = pd.Series([0] * 19 + [1])

    with pytest.raises(ValueError, match="fewer than 2 minority class samples"):
        train_logistic_regression(
            X_train,
            y_train,
            preprocessor,
            random_state=RANDOM_STATE,
            smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        )