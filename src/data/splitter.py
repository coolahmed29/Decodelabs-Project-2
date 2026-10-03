"""
PURPOSE: Raw (validated) dataset ko features (X) aur target (y) mein separate karna,
         aur STRATIFIED train/test split karna — taake fraud ka rare ratio train aur
         test dono sets mein preserve rahe (Section 9). Ye woh critical checkpoint
         hai jiske baad "Training Set" aur "Test Set" hamesha alag rehte hain —
         is point ke baad koi bhi fitting/transformation sirf training set pe honi
         chahiye (Section 4 — Data Leakage Prevention).

FUNCTION-BASED module hai (class ki zaroorat nahi — pure/stateless operations).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset ko features (X) aur target (y) mein separate karta hai.

    NOTE: Is function ko validation ke BAAD hi call karna — ye khud koi validation
    nahi karta, sirf separation karta hai. Dono copies hain, original df untouched.
    """
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split karta hai (sklearn.train_test_split).

    stratify=y CRITICAL hai — ye missing hua to rare fraud ratio test set mein
    distort ho sakta hai. Defaults config (TEST_SIZE, RANDOM_STATE) se aate hain.

    Returns:
        X_train, X_test, y_train, y_test (isi order mein, sklearn ke default order se).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def get_split_summary(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Har set (train/test) ke total, fraud_count aur fraud_pct ka dict return karta hai.

    Caller (train.py, agle phases) isko log karega taake har run pe fraud ratio
    verify ho sake (Section 9 ka core purpose).
    """
    train_total = len(y_train)
    train_fraud = int((y_train == 1).sum())
    test_total = len(y_test)
    test_fraud = int((y_test == 1).sum())

    return {
        "train": {
            "total": train_total,
            "fraud_count": train_fraud,
            "fraud_pct": train_fraud / train_total * 100 if train_total else 0.0,
        },
        "test": {
            "total": test_total,
            "fraud_count": test_fraud,
            "fraud_pct": test_fraud / test_total * 100 if test_total else 0.0,
        },
    }