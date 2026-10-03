"""
PURPOSE: split_features_target, split_train_test, aur get_split_summary
         teeno functions ko synthetic data ke sath test karna — especially
         stratification aur reproducibility verify karna.
"""

import pandas as pd
import pytest

from src.data.splitter import (
    get_split_summary,
    split_features_target,
    split_train_test,
)

TARGET = "is_fraud"


def _imbalanced_df() -> pd.DataFrame:
    """100 rows, 90 zeros + 10 ones — deterministic, predictable counts."""
    return pd.DataFrame(
        {
            "amount": list(range(100)),
            TARGET: [0] * 90 + [1] * 10,
        }
    )


def _split_data():
    df = _imbalanced_df()
    X, y = split_features_target(df, TARGET)
    return X, y


# ---------- split_features_target ----------
def test_split_features_target_removes_target_from_X():
    X, _ = split_features_target(_imbalanced_df(), TARGET)

    assert TARGET not in X.columns


def test_split_features_target_y_matches_target_column():
    df = _imbalanced_df()
    _, y = split_features_target(df, TARGET)

    assert y.equals(df[TARGET])


def test_split_features_target_row_count_preserved():
    df = _imbalanced_df()
    X, y = split_features_target(df, TARGET)

    assert len(X) == len(y) == len(df)


# ---------- split_train_test ----------
def test_split_preserves_total_row_count():
    X, y = _split_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_split_respects_test_size():
    X, y = _split_data()
    X_train, X_test, _, _ = split_train_test(X, y, test_size=0.2)

    assert len(X_test) == pytest.approx(20, abs=1)


def test_split_is_stratified():
    X, y = _split_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)

    original_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(train_ratio - original_ratio) < 0.05
    assert abs(test_ratio - original_ratio) < 0.05


def test_split_is_reproducible():
    X, y = _split_data()
    X_train_1, X_test_1, y_train_1, y_test_1 = split_train_test(X, y, random_state=42)
    X_train_2, X_test_2, y_train_2, y_test_2 = split_train_test(X, y, random_state=42)

    assert X_train_1.index.equals(X_train_2.index)
    assert X_test_1.index.equals(X_test_2.index)
    assert y_train_1.equals(y_train_2)
    assert y_test_1.equals(y_test_2)


def test_split_with_different_random_state_gives_different_result():
    X, y = _split_data()
    X_train_1, _, _, _ = split_train_test(X, y, random_state=1)
    X_train_2, _, _, _ = split_train_test(X, y, random_state=2)

    assert not X_train_1.index.equals(X_train_2.index)


# ---------- get_split_summary ----------
def test_get_split_summary_counts_correct():
    y_train = pd.Series([0, 0, 1, 0, 1])  # total 5, fraud 2
    y_test = pd.Series([0, 0, 0, 1])  # total 4, fraud 1

    summary = get_split_summary(y_train, y_test)

    assert summary["train"] == {"total": 5, "fraud_count": 2, "fraud_pct": 40.0}
    assert summary["test"] == {"total": 4, "fraud_count": 1, "fraud_pct": 25.0}


def test_get_split_summary_percentage_calculation():
    y_train = pd.Series([0, 0, 1, 0])  # 1/4
    y_test = pd.Series([1, 1, 0, 0, 0])  # 2/5

    summary = get_split_summary(y_train, y_test)

    assert summary["train"]["fraud_pct"] == pytest.approx(25.0)
    assert summary["test"]["fraud_pct"] == pytest.approx(40.0)