"""
PURPOSE: DataValidator ke har check ko ALAG SE aur validate() orchestrator ko
         COMBINED scenario mein test karna. Har test ke liye chhota synthetic
         pd.DataFrame banao — koi real dataset load nahi karna.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import DataValidator

REQUIRED_COLUMNS = ["amount", "is_fraud"]
TARGET = "is_fraud"


def _validator() -> DataValidator:
    return DataValidator(required_columns=REQUIRED_COLUMNS, target_column=TARGET)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"amount": [10.0, 20.0, 30.0], TARGET: [0, 1, 0]}
    )


# ---------- check_not_empty ----------
def test_empty_dataframe_raises_error():
    df = pd.DataFrame({"amount": [], TARGET: []})

    with pytest.raises(ValueError):
        _validator().check_not_empty(df)


# ---------- check_required_columns ----------
def test_missing_required_column_raises_error():
    df = pd.DataFrame({TARGET: [0, 1, 0]})  # "amount" missing

    with pytest.raises(ValueError) as exc_info:
        _validator().check_required_columns(df)

    assert "amount" in str(exc_info.value)


def test_all_required_columns_present_passes():
    _validator().check_required_columns(_valid_df())


# ---------- check_target_binary ----------
def test_target_column_missing_raises_error():
    df = pd.DataFrame({"amount": [10.0, 20.0]})

    with pytest.raises(ValueError, match="Target column not found"):
        _validator().check_target_binary(df)


def test_target_with_non_binary_values_raises_error():
    df = pd.DataFrame({"amount": [10.0, 20.0, 30.0], TARGET: [0, 1, 2]})

    with pytest.raises(ValueError):
        _validator().check_target_binary(df)


def test_target_with_nan_raises_error():
    df = pd.DataFrame({"amount": [10.0, 20.0], TARGET: [0, np.nan]})

    with pytest.raises(ValueError):
        _validator().check_target_binary(df)


def test_zero_fraud_samples_raises_error():
    df = pd.DataFrame({"amount": [10.0, 20.0], TARGET: [0, 0]})

    with pytest.raises(ValueError, match="No fraud samples found"):
        _validator().check_target_binary(df)


def test_valid_binary_target_passes():
    _validator().check_target_binary(_valid_df())


# ---------- check_duplicates ----------
def test_duplicates_detected_and_logged(caplog):
    df = pd.DataFrame({"amount": [10.0, 10.0, 20.0], TARGET: [0, 0, 1]})

    count = _validator().check_duplicates(df)

    assert count == 1
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_no_duplicates_returns_zero():
    count = _validator().check_duplicates(_valid_df())

    assert count == 0


# ---------- check_missing_values ----------
def test_missing_values_percentage_calculated_correctly():
    df = pd.DataFrame({"amount": [10.0, np.nan, 30.0], TARGET: [0, 1, 0]})

    result = _validator().check_missing_values(df)

    assert result["amount"] == pytest.approx(100.0 / 3)
    assert result[TARGET] == 0.0


def test_fully_missing_column_logs_warning(caplog):
    df = pd.DataFrame(
        {"amount": [10.0, np.nan], "other": [np.nan, np.nan], TARGET: [0, 1]}
    )

    result = _validator().check_missing_values(df)

    assert result["other"] == 100.0
    assert any(record.levelname == "WARNING" for record in caplog.records)


# ---------- check_impossible_numeric_values ----------
def test_infinite_value_raises_error():
    df = pd.DataFrame({"amount": [10.0, np.inf], TARGET: [0, 1]})

    with pytest.raises(ValueError) as exc_info:
        _validator().check_impossible_numeric_values(df)

    assert "amount" in str(exc_info.value)


def test_negative_infinite_value_raises_error():
    df = pd.DataFrame({"amount": [10.0, -np.inf], TARGET: [0, 1]})

    with pytest.raises(ValueError) as exc_info:
        _validator().check_impossible_numeric_values(df)

    assert "amount" in str(exc_info.value)


# ---------- validate() orchestrator ----------
def test_validate_passes_for_clean_dataset():
    _validator().validate(_valid_df())


def test_validate_stops_at_first_failure():
    # Do problems: "amount" column missing AND non-binary target [0, 2, 0]
    # check_required_columns pehle chalta hai, isliye "amount" wali error aaye.
    df = pd.DataFrame({TARGET: [0, 2, 0]})

    with pytest.raises(ValueError) as exc_info:
        _validator().validate(df)

    assert "amount" in str(exc_info.value)