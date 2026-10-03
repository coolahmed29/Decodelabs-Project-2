"""
PURPOSE: Model ke predictions ko Precision, Recall, ROC-AUC, Confusion Matrix,
         aur ROC Curve mein convert karna — Section 15-20 ke mutabiq. Accuracy
         CALCULATE ki ja sakti hai reference ke liye, lekin ise kahin bhi
         "primary metric" ke tor par treat NAHI karna (Section 18).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/non-interactive backend — savefig-only plots ke liye
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    """Precision, Recall, ROC-AUC, Confusion Matrix aur Accuracy ka dict return karta hai.

    zero_division=0 CRITICAL hai (Section 28): agar classifier sirf ek class
    predict kare to precision/recall crash ki jagah 0 return karte hain.
    ROC-AUC single-class y_true pe ValueError de sakta hai — us edge case ko
    try/except se None set karke log karo, crash mat hone dena.

    Returns:
        dict: {"precision", "recall", "roc_auc", "accuracy", "tp", "tn", "fp", "fn"}
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = None

    # Naye sklearn versions single-class case mein error ki jagah nan dete hain
    if roc_auc is not None and np.isnan(roc_auc):
        logger.warning("ROC-AUC cannot be computed — only one class present in y_true")
        roc_auc = None

    # labels=[0, 1] CRITICAL: sirf ek class present ho to confusion_matrix 1x1
    # matrix return karta hai (Section 28 edge case) — labels pass karne se
    # hamesha 2x2 matrix milti hai.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)  # REFERENCE ONLY, primary metric nahi

    return {
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """sklearn ka classification_report string return karta hai (direct print/log ke liye)."""
    return classification_report(y_true, y_pred, zero_division=0)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, model_name: str
) -> None:
    """Confusion matrix ka heatmap plot banata hai aur save_path par save karta hai.

    plt.close() zaroori hai (memory leak avoid karne ke liye jab multiple
    models loop mein plot honge).
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Path, model_name: str
) -> None:
    """ROC curve plot banata hai (diagonal baseline ke sath) aur save_path par save karta hai."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
    plt.title(f"ROC Curve — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()