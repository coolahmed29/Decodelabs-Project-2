"""
PURPOSE: Numerical aur categorical features ko ALAG SE preprocess karna (imputation +
         scaling/encoding), sklearn ka ColumnTransformer use karke. CRITICAL RULE
         (Section 10): ye transformer sirf TRAINING DATA pe .fit() hoga — test data
         par sirf .transform() call hoga. Is file mein koi bhi jagah aisi na ho jahan
         poore dataset (train+test combined) pe fit ho raha ho.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numerical_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Numerical + categorical pipelines ka ColumnTransformer banata hai.

    - Numerical: SimpleImputer(median) -> StandardScaler()
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
      <-- handle_unknown="ignore" CRITICAL hai (Section 28: "Unknown categorical
      values") — inference time par unseen category aaye to crash nahi hona chahiye.
    - remainder="drop": koi aur column silently pass-through na ho, taake
      accidental leakage/unexpected columns na aayein.

    Returns:
        ColumnTransformer (fit abhi nahi hota — caller training data pe fit karega).
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def infer_feature_types(
    df: pd.DataFrame,
    target_column: str,
    id_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Dtypes se numerical aur categorical feature lists infer karta hai.

    target_column aur id_columns (agar diye hon, e.g. "transaction_id") dono
    lists se exclude hote hain.

    Returns:
        (numerical_features, categorical_features)

    NOTE: Convenience/helper hai — schema pehle se pata ho to isko bypass karke
    manually bhi lists build_preprocessor() ko pass ki ja sakti hain.
    """
    exclude = {target_column}
    if id_columns:
        exclude.update(id_columns)

    remaining = [col for col in df.columns if col not in exclude]

    numerical_features = [
        col for col in remaining if col in df.select_dtypes(include=[np.number]).columns
    ]
    categorical_features = [
        col
        for col in remaining
        if col in df.select_dtypes(include=["object", "category"]).columns
    ]

    return list(numerical_features), list(categorical_features)