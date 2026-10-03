"""
PURPOSE: Central configuration for the whole pipeline — random seed, file paths,
         model/SMOTE defaults, and thresholds. Koi bhi hardcoded value (magic number/path)
         kisi aur file mein nahi honi chahiye; sab yahan se import hoga.
"""

import logging
from pathlib import Path

# Project root: src/config.py -> parent = src, parent.parent = project root.
# Saare paths isi base se relative banenge, taake kahin se run karo to kaam kare.
BASE_DIR = Path(__file__).resolve().parent.parent


# ---------- Reproducibility ----------
# Fixed seed: train/test split, SMOTE, RandomForest, CV, aur hyperparameter search
# sab isi ek value ko consistently use karenge (Section 29 - Reproducibility).
RANDOM_STATE = 42

# ---------- Data paths ----------
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORTS_METRICS_DIR = BASE_DIR / "reports" / "metrics"
REPORTS_FIGURES_DIR = BASE_DIR / "reports" / "figures"

# ---------- Target column ----------
# Target/label column ka naam (dataset ke hisaab se change hoga, abhi placeholder).
TARGET_COLUMN = "is_fraud"

# ---------- Train/test split ----------
TEST_SIZE = 0.20

# ---------- SMOTE ----------
# Configurable ratio (Section 11 ke mutabiq - blindly 50/50 nahi, 0.5 default).
SMOTE_SAMPLING_STRATEGY = 0.5

# ---------- Cross-validation ----------
CV_FOLDS = 5

# ---------- Logging ----------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_TO_CONSOLE = True
LOG_TO_FILE = True
LOG_FILE = BASE_DIR / "pipeline.log"


def ensure_directories() -> None:
    """MODEL_DIR / REPORTS_* dirs exist ensure karta hai; nahi to create kar deta hai.

    Taake baad mein "directory not found" errors na aayein.
    """
    for directory in (MODEL_DIR, REPORTS_METRICS_DIR, REPORTS_FIGURES_DIR, LOG_FILE.parent):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Logging config initialize: level, format, aur console/file handlers."""
    handlers = []
    if LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler())
    if LOG_TO_FILE:
        handlers.append(logging.FileHandler(LOG_FILE))
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=handlers)


# Import waqt hi dirs aur logging ready kar do - puri pipeline yahan se
# consistent state pe depend karti hai.
ensure_directories()
setup_logging()