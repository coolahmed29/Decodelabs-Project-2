"""
PURPOSE: Model ke predict_proba() output par DIFFERENT thresholds test karna,
         taake fixed 0.5 cutoff blindly use karne ki bajaye business-driven
         threshold select ki ja sake (Section 19). Fraud detection mein
         False Negative (missed fraud) ka cost aksar False Positive
         (extra review) se zyada hota hai — isliye default 0.5 hamesha
         optimal nahi hota.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/non-interactive backend — savefig-only plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float] | None = None
) -> pd.DataFrame:
    """Har threshold par metrics calculate karke sorted DataFrame return karta hai.

    roc_auc HAR row mein same rahega (threshold-independent metric hai) — ise
    sirf reference ke liye har row mein dikhaya ja raha hai, comparison ke liye.

    Returns:
        pd.DataFrame: columns = threshold, precision, recall, roc_auc, tp, fp, fn, tn.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    rows = []
    for threshold in thresholds:
        y_pred_at_threshold = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred_at_threshold, y_proba)
        rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
            }
        )

    threshold_df = pd.DataFrame(rows)
    return threshold_df.sort_values("threshold").reset_index(drop=True)


def select_threshold_by_business_rule(
    threshold_df: pd.DataFrame, min_recall: float, min_precision: float
) -> dict:
    """Business minimums ko satisfy karne wala best threshold select karta hai.

    Section 21 ke threshold-level version:
        - Filter: recall >= min_recall AND precision >= min_precision
        - Candidates empty -> fallback: HIGHEST recall wali row,
          "met_requirements": False
        - Candidates maujood -> HIGHEST roc_auc wali row,
          "met_requirements": True

    Returns:
        dict: {"threshold", "precision", "recall", "roc_auc", "met_requirements"}
    """
    filtered = threshold_df[
        (threshold_df["recall"] >= min_recall)
        & (threshold_df["precision"] >= min_precision)
    ]

    if filtered.empty:
        logger.warning(
            "No threshold meets both minimum recall and precision requirements"
        )
        best = threshold_df.loc[threshold_df["recall"].idxmax()]
        met_requirements = False
    else:
        best = filtered.loc[filtered["roc_auc"].idxmax()]
        met_requirements = True

    return {
        "threshold": float(best["threshold"]),
        "precision": best["precision"],
        "recall": best["recall"],
        "roc_auc": best["roc_auc"],
        "met_requirements": met_requirements,
    }


def plot_threshold_tradeoff(threshold_df: pd.DataFrame, save_path: Path) -> None:
    """Threshold vs precision/recall ka line plot banata hai aur save karta hai."""
    plt.figure()
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.title("Precision-Recall Tradeoff by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(save_path)
    plt.close()