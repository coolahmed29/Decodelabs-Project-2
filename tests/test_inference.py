"""
PURPOSE: InferenceService ke saare validation edge cases (Section 24 ki
         COMPLETE list) aur end-to-end predict flow test karna.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE, SMOTE_SAMPLING_STRATEGY
from src.data.splitter import split_train_test
from src.features.preprocessing import build_preprocessor
from src.inference.predict import InferenceService
from src.models.selection import save_model
from src.models.train import train_logistic_regression

NUMERICAL_FEATURES = ["amount", "age"]
CATEGORICAL_FEATURES = ["channel"]
TARGET = "is_fraud"
REQUIRED_FIELDS = ["amount", "age", "channel"]
CHANNELS = ["online", "in_store", "mobile"]

VALID_TRANSACTION = {"amount": 250.0, "age": 30.0, "channel": "online"}


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    """Real trained pipeline save karo aur InferenceService usi path se load karo."""
    tmp_dir = tmp_path_factory.mktemp("inference_models")

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
    X_train, _, y_train, _ = split_train_test(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = train_logistic_regression(
        X_train,
        y_train,
        build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES),
        random_state=RANDOM_STATE,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    )
    model_path = save_model(pipeline, "logistic_regression", "v1", tmp_dir)

    return InferenceService(
        model_path=model_path,
        required_fields=REQUIRED_FIELDS,
        categorical_fields=CATEGORICAL_FEATURES,
        numerical_fields=NUMERICAL_FEATURES,
        non_negative_fields=["amount"],
        threshold=0.5,
        model_version="v1",
    )


# ---------- validate_input: required fields ----------
def test_missing_required_field_raises_error(service):
    transaction = {"age": 30.0, "channel": "online"}

    with pytest.raises(Exception) as exc_info:
        service.validate_input(transaction)

    assert "amount" in str(exc_info.value)


# ---------- validate_input: numerical fields ----------
def test_negative_amount_rejected(service):
    transaction = dict(VALID_TRANSACTION, amount=-100.0)

    with pytest.raises(Exception):
        service.validate_input(transaction)


def test_nan_numeric_value_rejected(service):
    transaction = dict(VALID_TRANSACTION, amount=float("nan"))

    with pytest.raises(Exception):
        service.validate_input(transaction)


def test_infinite_numeric_value_rejected(service):
    transaction = dict(VALID_TRANSACTION, amount=float("inf"))

    with pytest.raises(Exception):
        service.validate_input(transaction)


def test_wrong_type_numeric_field_rejected(service):
    transaction = dict(VALID_TRANSACTION, amount="not_a_number")

    with pytest.raises(Exception):
        service.validate_input(transaction)


def test_valid_numeric_value_passes(service):
    transaction = dict(VALID_TRANSACTION, amount=250.0)

    service.validate_input(transaction)  # koi exception NAHI aani chahiye


# ---------- validate_input: categorical fields ----------
def test_wrong_type_categorical_field_rejected(service):
    transaction = dict(VALID_TRANSACTION, channel=12345)

    with pytest.raises(Exception):
        service.validate_input(transaction)


def test_unknown_but_valid_string_category_does_not_get_rejected_at_validation(
    service,
):
    transaction = dict(VALID_TRANSACTION, channel="completely_new_category")

    service.validate_input(transaction)  # TYPE check hi yahan hona chahiye


# ---------- validate_input: payload size ----------
def test_oversized_field_value_rejected(service):
    transaction = dict(VALID_TRANSACTION, channel="x" * 2000)

    with pytest.raises(Exception):
        service.validate_input(transaction)


# ---------- predict_single: end-to-end ----------
def test_predict_single_returns_correct_response_format(service):
    result = service.predict_single(VALID_TRANSACTION)

    assert set(result.keys()) == {"fraud_probability", "is_fraud", "model_version"}


def test_predict_single_probability_in_valid_range(service):
    result = service.predict_single(VALID_TRANSACTION)

    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_single_is_fraud_matches_threshold_logic(service):
    result = service.predict_single(VALID_TRANSACTION)

    expected = result["fraud_probability"] >= 0.5
    assert result["is_fraud"] == expected


def test_predict_single_raises_for_invalid_input(service):
    transaction = {"amount": 250.0, "age": 30.0}  # "channel" missing

    with pytest.raises(Exception):
        service.predict_single(transaction)


# ---------- predict_batch ----------
def test_predict_batch_returns_list_same_length_as_input(service):
    transactions = [
        dict(VALID_TRANSACTION, amount=150.0),
        dict(VALID_TRANSACTION, amount=350.0),
        dict(VALID_TRANSACTION, amount=800.0),
    ]

    results = service.predict_batch(transactions)

    assert len(results) == 3


def test_predict_batch_handles_mixed_valid_and_invalid_records(service):
    transactions = [
        dict(VALID_TRANSACTION, amount=150.0),
        {"amount": 250.0, "age": 30.0},  # invalid: "channel" missing
        dict(VALID_TRANSACTION, amount=350.0),
    ]

    results = service.predict_batch(transactions)

    assert len(results) == 3
    successes = [r for r in results if "fraud_probability" in r]
    errors = [r for r in results if "error" in r]
    assert len(successes) == 2
    assert len(errors) == 1
    assert errors[0]["index"] == 1


# ---------- unseen category end-to-end ----------
def test_predict_single_with_unseen_category_does_not_crash(service):
    transaction = dict(VALID_TRANSACTION, channel="brand_new_category")

    result = service.predict_single(transaction)

    assert "fraud_probability" in result