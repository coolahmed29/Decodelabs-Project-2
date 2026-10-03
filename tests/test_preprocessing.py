"""
PURPOSE: build_preprocessor aur infer_feature_types dono ko synthetic data
         par test karna — especially missing-value handling, unseen categories,
         aur output shape correctness.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import build_preprocessor, infer_feature_types

TARGET = "is_fraud"
NUMERICAL_FEATURES = ["amount", "age"]
CATEGORICAL_FEATURES = ["channel", "card_type"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "amount": [100.0, 200.0, np.nan, 400.0],
            "age": [30.0, np.nan, 42.0, 28.0],
            "channel": ["online", "in_store", "online", np.nan],
            "card_type": ["visa", "mastercard", "visa", "visa"],
            TARGET: [0, 1, 0, 1],
        }
    )


@pytest.fixture
def preprocessor():
    return build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


def _to_dense(output):
    return output.toarray() if hasattr(output, "toarray") else np.asarray(output)


# ---------- infer_feature_types ----------
def test_infer_feature_types_excludes_target(sample_df):
    numerical, categorical = infer_feature_types(sample_df, TARGET)

    assert TARGET not in numerical
    assert TARGET not in categorical


def test_infer_feature_types_excludes_id_columns(sample_df):
    numerical, categorical = infer_feature_types(sample_df, TARGET, id_columns=["transaction_id"])

    assert "transaction_id" not in numerical
    assert "transaction_id" not in categorical


def test_infer_feature_types_correct_classification(sample_df):
    numerical, categorical = infer_feature_types(sample_df, TARGET)

    assert "amount" in numerical
    assert "age" in numerical
    assert "channel" in categorical
    assert "card_type" in categorical


# ---------- build_preprocessor: fit/transform basic behavior ----------
def test_preprocessor_fit_transform_no_errors(preprocessor, sample_df):
    output = preprocessor.fit_transform(sample_df[FEATURE_COLUMNS])

    assert output is not None


def test_preprocessor_output_has_no_nan(preprocessor, sample_df):
    output = _to_dense(preprocessor.fit_transform(sample_df[FEATURE_COLUMNS]))

    assert not np.isnan(output).any()


def test_preprocessor_output_row_count_matches_input(preprocessor, sample_df):
    output = preprocessor.fit_transform(sample_df[FEATURE_COLUMNS])

    assert output.shape[0] == len(sample_df)


# ---------- missing value handling ----------
def test_numerical_missing_values_imputed(preprocessor):
    df = pd.DataFrame(
        {
            "amount": [100.0, np.nan, 300.0],
            "age": [20.0, 30.0, np.nan],
            "channel": ["online", "in_store", "online"],
            "card_type": ["visa", "mastercard", "visa"],
        }
    )

    output = _to_dense(preprocessor.fit_transform(df[FEATURE_COLUMNS]))

    numerical_part = output[:, : len(NUMERICAL_FEATURES)]
    assert not np.isnan(numerical_part).any()


def test_categorical_missing_values_imputed(preprocessor):
    df = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
            "age": [20.0, 30.0, 40.0],
            "channel": ["online", np.nan, "online"],
            "card_type": ["visa", np.nan, "visa"],
        }
    )

    output = _to_dense(preprocessor.fit_transform(df[FEATURE_COLUMNS]))

    assert not np.isnan(output).any()


# ---------- unseen category handling (CRITICAL - Section 28) ----------
def test_unknown_category_at_transform_time_does_not_crash(preprocessor):
    train = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
            "age": [20.0, 30.0, 40.0],
            "channel": ["online", "in_store", "online"],
            "card_type": ["visa", "mastercard", "visa"],
        }
    )
    preprocessor.fit(train[FEATURE_COLUMNS])

    new_data = pd.DataFrame(
        {
            "amount": [50.0],
            "age": [25.0],
            "channel": ["unknown_type"],  # training data mein nahi thi
            "card_type": ["visa"],
        }
    )

    output = _to_dense(preprocessor.transform(new_data[FEATURE_COLUMNS]))

    assert output.shape[0] == 1
    assert not np.isnan(output).any()


# ---------- train-only fitting discipline ----------
def test_preprocessor_is_not_fit_on_test_data_separately(preprocessor):
    train = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
            "age": [20.0, 30.0, 40.0],
            "channel": ["online", "in_store", "online"],
            "card_type": ["visa", "mastercard", "visa"],
        }
    )
    test = pd.DataFrame(
        {
            "amount": [50.0, np.nan],
            "age": [25.0, 60.0],
            "channel": ["online", "unknown_channel"],
            "card_type": ["visa", "amex_unseen"],
        }
    )
    preprocessor.fit(train[FEATURE_COLUMNS])

    train_out = _to_dense(preprocessor.transform(train[FEATURE_COLUMNS]))
    test_out = _to_dense(preprocessor.transform(test[FEATURE_COLUMNS]))

    assert train_out.shape[1] == test_out.shape[1]
    assert not np.isnan(test_out).any()