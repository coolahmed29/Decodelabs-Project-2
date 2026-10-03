"""
PURPOSE: Raw transaction dataset ko disk se load karna aur load-time par hi basic
         structural checks (existence, shape, class distribution) log karna.
         Ye Data Validation (agla feature) se ALAG hai — loader sirf "file uthao aur
         DataFrame do", detailed content validation (validation.py) mein hogi.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN

logger = logging.getLogger(__name__)

# Supported file formats: extension -> pandas reader
SUPPORTED_EXTENSIONS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
}


class DataLoader:
    """Disk se CSV/Parquet dataset load karta hai with basic structural checks."""

    def __init__(self, file_path: str | Path = RAW_DATA_PATH / "transactions.csv"):
        """file_path ko Path object mein convert karke store karta hai.

        Args:
            file_path: Dataset file ka path (CSV ya Parquet).
                Default config ke RAW_DATA_PATH (data/raw/) pe point karta hai.
        """
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """File load karta hai, structural checks karta hai, DataFrame return karta hai.

        Steps:
            1. File existence check -> FileNotFoundError
            2. Extension check (.csv / .parquet) -> ValueError (unsupported)
            3. Empty dataset check (0 rows) -> ValueError
            4. Shape aur target class distribution log karo
               (target column maujood nahi to sirf warning - error nahi)

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: Agar file maujood nahi hai.
            ValueError: Agar format unsupported hai ya dataset empty hai.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

        extension = self.file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{extension}'. "
                f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        reader = SUPPORTED_EXTENSIONS[extension]
        df = reader(self.file_path)

        if df.shape[0] == 0:
            raise ValueError(f"Dataset is empty: {self.file_path}")

        logger.info("Loaded dataset from %s with shape %s", self.file_path, df.shape)

        if TARGET_COLUMN in df.columns:
            class_distribution = df[TARGET_COLUMN].value_counts().to_dict()
            logger.info(
                "Target column '%s' class distribution: %s",
                TARGET_COLUMN,
                class_distribution,
            )
        else:
            logger.warning(
                "Target column '%s' not found in dataset %s (only warning, validation ka kaam)",
                TARGET_COLUMN,
                self.file_path,
            )

        return df

    def get_shape_summary(self, df: pd.DataFrame) -> dict:
        """Simple helper: rows, columns aur column_names ki dict return karta hai."""
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
        }