"""
PURPOSE: Logistic Regression aur Random Forest dono ke liye hyperparameters ko
         StratifiedKFold cross-validation ke through tune karna — SIRF training
         data par (Section 14: "Do not tune on the test set"). Test set is
         POORI file mein kabhi bhi touch nahi hoga.

CRITICAL RULE: RandomizedSearchCV/GridSearchCV ko humesha wahi FULL PIPELINE
         (preprocessor + SMOTE + classifier) diya jayega jo pipelines.py mein
         bana hai — isolated classifier nahi. Isse CV ke har fold mein
         preprocessing aur SMOTE dono sahi tareeqe se sirf us fold ki training
         portion par apply honge (Section 4 ka core purpose yehi hai).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import SMOTE_SAMPLING_STRATEGY
from src.models.pipelines import build_full_pipeline, get_safe_k_neighbors

logger = logging.getLogger(__name__)


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    random_state: int,
    cv_folds: int,
    n_iter: int = 20,
) -> RandomizedSearchCV:
    """Logistic Regression ke hyperparameters RandomizedSearchCV se tune karta hai.

    SIRF training data par — test set kabhi bhi is process mein pass nahi hota.
    Full pipeline (preprocessor + SMOTE + classifier) CV ko diya jata hai, taake
    har fold mein preprocessing/SMOTE sirf fold ki training portion par apply ho.

    Returns:
        Fitted RandomizedSearchCV (best_estimator_ available hoga, refit=True).
    """
    safe_k = get_safe_k_neighbors(y_train)

    base_pipeline = build_full_pipeline(
        preprocessor=preprocessor,
        classifier=LogisticRegression(random_state=random_state, max_iter=1000),
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=random_state,
        k_neighbors=safe_k,
    )

    param_distributions = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    logger.info(
        "LR tuning complete — best_params: %s, best_score: %.4f",
        search.best_params_,
        search.best_score_,
    )

    return search


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    random_state: int,
    cv_folds: int,
    n_iter: int = 20,
) -> RandomizedSearchCV:
    """Random Forest ke hyperparameters RandomizedSearchCV se tune karta hai.

    Same structure jaisa LR tuning — bas RF classifier aur uski param grid ke saath.

    Returns:
        Fitted RandomizedSearchCV (best_estimator_ available hoga, refit=True).
    """
    safe_k = get_safe_k_neighbors(y_train)

    base_pipeline = build_full_pipeline(
        preprocessor=preprocessor,
        classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=random_state,
        k_neighbors=safe_k,
    )

    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    logger.info(
        "RF tuning complete — best_params: %s, best_score: %.4f",
        search.best_params_,
        search.best_score_,
    )

    return search