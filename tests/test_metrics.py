"""
PURPOSE: calculate_metrics() ko KNOWN/manually-verifiable values ke sath test
         karna — taake confirm ho sake formulas sahi implement hue hain
         (Section 15-17 ke exact definitions ke mutabiq).
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_metrics,
    get_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


# ---------- calculate_metrics: known values ----------
def test_perfect_predictions_give_precision_recall_1():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_all_wrong_predictions_give_zero_precision_recall():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_manually_calculated_precision_recall():
    # TP=2, FP=1, FN=1, TN=2 -> Precision = Recall = 2/3
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_confusion_matrix_values_correct():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2


# ---------- edge case: single-class predictions (Section 28) ----------
def test_model_predicts_only_one_class_does_not_crash():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_roc_auc_with_single_class_in_y_true_returns_none(caplog):
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["roc_auc"] is None
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_roc_auc_calculated_correctly_for_known_case():
    # Perfect ranking separation -> AUC sirf 1.0 ho sakta hai is combo ke liye.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    assert metrics["roc_auc"] == pytest.approx(1.0)


# ---------- get_classification_report ----------
def test_classification_report_returns_string():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    report = get_classification_report(y_true, y_pred)

    assert isinstance(report, str)
    assert "precision" in report.lower()
    assert "recall" in report.lower()


# ---------- plotting functions (smoke tests only) ----------
def test_plot_confusion_matrix_creates_file(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    save_path = tmp_path / "cm.png"

    plot_confusion_matrix(y_true, y_pred, save_path, model_name="test_model")

    assert save_path.exists()


def test_plot_roc_curve_creates_file(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])
    save_path = tmp_path / "roc.png"

    plot_roc_curve(y_true, y_proba, save_path, model_name="test_model")

    assert save_path.exists()