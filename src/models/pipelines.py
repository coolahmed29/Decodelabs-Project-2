"""
PURPOSE: Preprocessing + SMOTE + Classifier ko EK SINGLE imblearn Pipeline mein
         combine karna. Ye CRITICAL hai (Section 4 aur Section 11): agar SMOTE
         pipeline ke bahar apply hui, ya agar sklearn ka normal Pipeline use kiya
         (imblearn ka nahi), to cross-validation ke folds bhi ghalat tareeqe se
         resample honge aur DATA LEAKAGE ho jayega. Is file mein KOI BHI JAGAH
         "sklearn.pipeline.Pipeline" use NAHI honi chahiye jahan SMOTE step ho —
         hamesha "imblearn.pipeline.Pipeline" use karna.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def build_full_pipeline(
    preprocessor,
    classifier,
    smote_sampling_strategy: float,
    random_state: int,
    k_neighbors: int = 5,
) -> ImbPipeline:
    """Preprocessing + SMOTE + Classifier ko ek single imblearn Pipeline mein jorta hai.

    Order MATTERS: preprocessor -> smote -> classifier. ImbPipeline zaroori hai
    (sklearn ka Pipeline resampling steps ko silently ignore kar deta hai).

    Returns:
        ImbPipeline object (fit abhi nahi — caller sirf TRAINING data par fit karega).
    """
    smote = SMOTE(
        sampling_strategy=smote_sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", classifier),
        ]
    )
    return pipeline


def get_safe_k_neighbors(y_train: pd.Series, desired_k: int = 5) -> int:
    """SMOTE ke liye safe k_neighbors compute karta hai (Section 28 — few fraud samples).

    SMOTE ka rule: k_neighbors < minority_class_count hona chahiye, warna
    "Expected n_neighbors <= n_samples" wali error aati hai.

    Raises:
        ValueError: Agar minority class count <= 1 — SMOTE 1/0 sample ke sath
            kaam hi nahi kar sakta.

    Returns:
        safe_k = min(desired_k, minority_class_count - 1)
    """
    minority_class_count = int(y_train.value_counts().min())

    if minority_class_count <= 1:
        raise ValueError(
            "Cannot apply SMOTE: fewer than 2 minority class samples available"
        )

    safe_k = min(desired_k, minority_class_count - 1)
    return safe_k


def verify_no_test_leakage(pipeline: ImbPipeline) -> None:
    """Defensive helper: pipeline ke SMOTE step ki presence/type verify karta hai.

    Runtime mein kuch fix nahi karta, sirf assert karta hai ke:
      1. "smote" step maujood hai
      2. "smote" step ka type imblearn SMOTE hai

    Raises:
        AssertionError: Agar SMOTE step missing hai ya galat type ka hai.
    """
    step_names = [name for name, _ in pipeline.steps]
    smote_step = pipeline.named_steps.get("smote")

    if "smote" not in step_names or not isinstance(smote_step, SMOTE):
        raise AssertionError(
            "Pipeline is misconfigured — SMOTE step missing or wrong type"
        )