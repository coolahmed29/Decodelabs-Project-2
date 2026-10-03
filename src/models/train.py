"""
PURPOSE: build_full_pipeline() (pipelines.py se) ko use karke Logistic Regression
         aur Random Forest ke TRAINING functions banana — jo preprocessing.py ka
         preprocessor, pipelines.py ka SMOTE-wrapped pipeline, aur classifier
         combine karke ek FIT hua pipeline return kare. Hyperparameter TUNING abhi
         nahi (Phase 9) — sirf baseline models train karna hai taake pura flow
         end-to-end verify ho sake.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.pipelines import build_full_pipeline, get_safe_k_neighbors

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    random_state: int,
    smote_sampling_strategy: float,
):
    """Preprocessor + SMOTE + LogisticRegression ka fitted pipeline return karta hai.

    Steps:
        1. get_safe_k_neighbors(y_train) -> SMOTE ke liye safe k_neighbors
           (Section 28 edge case — very few fraud samples).
        2. LogisticRegression(random_state, max_iter=1000, solver="lbfgs")
           class_weight=None (SMOTE pehle se balance kar rahi hai).
        3. build_full_pipeline(...) -> fit -> log -> return.

    Returns:
        Fitted ImbPipeline (steps: preprocessor -> smote -> classifier).
    """
    safe_k = get_safe_k_neighbors(y_train)

    classifier = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight=None,
        solver="lbfgs",
    )

    pipeline = build_full_pipeline(
        preprocessor=preprocessor,
        classifier=classifier,
        smote_sampling_strategy=smote_sampling_strategy,
        random_state=random_state,
        k_neighbors=safe_k,
    )

    pipeline.fit(X_train, y_train)

    logger.info("Logistic Regression training complete — X_train shape: %s", X_train.shape)

    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    random_state: int,
    smote_sampling_strategy: float,
    n_jobs: int = -1,
):
    """Preprocessor + SMOTE + RandomForestClassifier ka fitted pipeline return karta hai.

    Random Forest tree-based hai — scaling ki zaroorat nahi hoti, lekin SAME
    preprocessor reuse karna hai (StandardScaler harmful nahi, bas thoda
    unnecessary — consistency/code-reuse ke liye).

    Steps:
        1. get_safe_k_neighbors(y_train) — SAME helper jo LR mein use hua tha.
        2. RandomForestClassifier(n_estimators=200, random_state,
           class_weight=None, n_jobs=n_jobs) — class_weight=None kyunke SMOTE
           already balance kar rahi hai.
        3. build_full_pipeline(...) -> fit -> log -> return.

    Returns:
        Fitted ImbPipeline (steps: preprocessor -> smote -> classifier).
    """
    safe_k = get_safe_k_neighbors(y_train)

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight=None,
        n_jobs=n_jobs,
    )

    pipeline = build_full_pipeline(
        preprocessor=preprocessor,
        classifier=classifier,
        smote_sampling_strategy=smote_sampling_strategy,
        random_state=random_state,
        k_neighbors=safe_k,
    )

    pipeline.fit(X_train, y_train)

    logger.info("Random Forest training complete — X_train shape: %s", X_train.shape)

    return pipeline


def get_predictions(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predictions aur fraud-class probabilities return karta hai.

    Dono models (LR aur RF) isi helper ko reuse karte hain — alag banane ki
    koi zaroorat nahi.

    Returns:
        (y_pred, y_proba) — y_proba fraud class (label "1") ki probability hai.
    """
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    return y_pred, y_proba