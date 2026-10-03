"""
PURPOSE: Loader ke baad, dataset ke CONTENT ko strictly validate karna — schema,
         target column ki correctness, aur data-quality issues (duplicates, missing
         values, impossible numeric values) detect karna. Loader sirf file uthata hai,
         ye module decide karta hai ke data "usable" hai ya nahi. Section 7 aur
         Section 28 (Edge Cases) dono is file pe apply hote hain.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Dataset ke content ko strictly validate karta hai (schema, target, quality).

    STATELESS hai: same instance multiple DataFrames pe reuse ho sakta hai
    (kisi bhi method mein df ko attribute mein store nahi karta).
    """

    def __init__(self, required_columns: list[str], target_column: str):
        """required_columns aur target_column store karta hai.

        Args:
            required_columns: Un columns ki list jo dataset mein must hain.
            target_column: Target/label column ka naam.
        """
        self.required_columns = list(required_columns)
        self.target_column = target_column

    def validate(self, df: pd.DataFrame) -> None:
        """Saare checks order mein chalti hain — koi fail ho to turant raise.

        Fail-fast: pura report accumulate nahi karta, pehli error pe ruk jata hai.
        Duplicates aur missing-values wale checks sirf log karte hain (raise nahi).
        """
        self.check_not_empty(df)
        self.check_required_columns(df)
        self.check_target_binary(df)
        self.check_duplicates(df)
        self.check_missing_values(df)
        self.check_impossible_numeric_values(df)

    def check_not_empty(self, df: pd.DataFrame) -> None:
        """0-row dataset -> ValueError raise karta hai (Section 28: empty -> fail fast)."""
        if df.shape[0] == 0:
            raise ValueError("Dataset is empty — validation cannot proceed")

    def check_required_columns(self, df: pd.DataFrame) -> None:
        """Koi required column missing ho to raise; missing names error mein include hain."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s): {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    def check_target_binary(self, df: pd.DataFrame) -> None:
        """Target correctness: exists, sirf {0, 1} (bina NaN), aur kom az kam ek fraud."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column not found: '{self.target_column}'")

        values = df[self.target_column].to_numpy()

        if pd.isna(values).any():
            raise ValueError(
                f"Target column '{self.target_column}' contains NaN values — not allowed"
            )

        unexpected = [v for v in pd.unique(values) if v not in {0, 1}]
        if unexpected:
            raise ValueError(
                f"Unexpected values in target column '{self.target_column}': {unexpected}. "
                f"Allowed values: {{0, 1}}"
            )

        fraud_count = int((df[self.target_column] == 1).sum())
        if fraud_count == 0:
            raise ValueError("No fraud samples found — cannot train")

    def check_duplicates(self, df: pd.DataFrame) -> int:
        """Duplicate rows count karta hai, warning log karta hai, count return karta hai.

        Yahan raise nahi karta — duplicate ka decision baad mein EDA mein hoga
        (Section 28: "Investigate duplicates before training").
        """
        count = int(df.duplicated().sum())
        if count > 0:
            logger.warning("Found %d duplicate row(s) in dataset", count)
        return count

    def check_missing_values(self, df: pd.DataFrame) -> dict:
        """Har column ka missing % {column: percent} return karta hai.

        100% missing columns (Section 28: "All values missing") alag se warning
        se highlight hote hain — ye drop karne ke candidate hain.
        """
        percentages = (df.isnull().mean() * 100).to_dict()
        fully_missing = [col for col, pct in percentages.items() if pct == 100.0]
        if fully_missing:
            logger.warning(
                "Column(s) with 100%% missing values (drop candidates): %s", fully_missing
            )
        return percentages

    def check_impossible_numeric_values(self, df: pd.DataFrame) -> None:
        """Numeric columns mein +inf/-inf hone pe raise karta hai, column name ke saath.

        Negative-value check yahan nahi hai — wo business-rule specific hai
        aur dataset schema pe depend karta hai; abhi sirf inf/-inf tak seemit.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if np.isinf(df[column]).any():
                raise ValueError(
                    f"Column '{column}' contains infinite values (+inf/-inf) — impossible "
                    f"numeric data"
                )