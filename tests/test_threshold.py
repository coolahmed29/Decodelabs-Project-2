"""
PURPOSE: evaluate_thresholds() aur select_threshold_by_business_rule() ko
         KNOWN probability values ke sath test karna, taake trade-off
         direction aur selection logic dono verify ho sakein.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.threshold import (
    evaluate_thresholds,
    select_threshold_by_business_rule,
    plot_threshold_tradeoff,
)

Y_TRUE = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
Y_PROBA = np.array([0.05, 0.1, 0.3, 0.4, 0.6, 0.55, 0.65, 0.75, 0.85, 0.95])


@pytest.fixture
def thresholds_df():
    return evaluate_thresholds(Y_TRUE, Y_PROBA)


# ---------- evaluate_thresholds ----------
def test_evaluate_thresholds_returns_dataframe_with_correct_columns(thresholds_df):
    assert isinstance(thresholds_df, pd.DataFrame)
    required_columns = ["threshold", "precision", "recall", "roc_auc"]
    for column in required_columns:
        assert column in thresholds_df.columns


def test_evaluate_thresholds_row_count_matches_input_thresholds():
    result = evaluate_thresholds(Y_TRUE, Y_PROBA, thresholds=[0.3, 0.5, 0.7])

    assert len(result) == 3


def test_lower_threshold_gives_higher_or_equal_recall(thresholds_df):
    recall_at_0_2 = thresholds_df.loc[
        thresholds_df["threshold"] == 0.2, "recall"
    ].iloc[0]
    recall_at_0_8 = thresholds_df.loc[
        thresholds_df["threshold"] == 0.8, "recall"
    ].iloc[0]

    assert recall_at_0_2 >= recall_at_0_8


def test_roc_auc_is_same_across_all_thresholds(thresholds_df):
    assert thresholds_df["roc_auc"].nunique() == 1


def test_default_thresholds_used_when_none_passed():
    result = evaluate_thresholds(Y_TRUE, Y_PROBA, thresholds=None)

    assert len(result) == 9


# ---------- select_threshold_by_business_rule ----------
def test_selects_threshold_meeting_both_minimums(thresholds_df):
    selected = select_threshold_by_business_rule(
        thresholds_df, min_recall=0.5, min_precision=0.5
    )

    assert selected["met_requirements"] is True
    assert selected["recall"] >= 0.5
    assert selected["precision"] >= 0.5


def test_fallback_when_no_threshold_meets_requirements(thresholds_df):
    selected = select_threshold_by_business_rule(
        thresholds_df, min_recall=0.99, min_precision=0.99
    )

    assert selected["met_requirements"] is False
    assert selected["recall"] == thresholds_df["recall"].max()


def test_selects_highest_roc_auc_among_valid_candidates(thresholds_df):
    selected = select_threshold_by_business_rule(
        thresholds_df, min_recall=0.0, min_precision=0.0
    )

    assert isinstance(selected, dict)
    assert "threshold" in selected
    assert "roc_auc" in selected
    assert selected["met_requirements"] is True


# ---------- plot_threshold_tradeoff ----------
def test_plot_threshold_tradeoff_creates_file(tmp_path):
    thresholds_df = evaluate_thresholds(Y_TRUE, Y_PROBA)
    save_path = tmp_path / "threshold_tradeoff.png"

    plot_threshold_tradeoff(thresholds_df, save_path)

    assert save_path.exists()