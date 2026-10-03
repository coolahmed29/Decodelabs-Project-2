"""
PURPOSE: config.py ke saare constants/directories verify karna.
         Jab tak ye saare tests pass na ho jayein, agle feature pe move nahi karna.
"""

from pathlib import Path

import pytest

from src import config


def test_random_state_is_int():
    assert isinstance(config.RANDOM_STATE, int)


def test_paths_are_path_objects():
    path_constants = [
        config.RAW_DATA_PATH,
        config.PROCESSED_DATA_PATH,
        config.MODEL_DIR,
        config.REPORTS_METRICS_DIR,
        config.REPORTS_FIGURES_DIR,
    ]
    for constant in path_constants:
        assert isinstance(constant, Path), f"{constant} is not a Path instance"


def test_smote_ratio_in_valid_range():
    assert isinstance(config.SMOTE_SAMPLING_STRATEGY, (int, float))
    assert 0 < config.SMOTE_SAMPLING_STRATEGY <= 1


def test_directories_get_created(monkeypatch, tmp_path):
    tmp_model_dir = tmp_path / "models"
    tmp_metrics_dir = tmp_path / "reports" / "metrics"
    tmp_figures_dir = tmp_path / "reports" / "figures"

    monkeypatch.setattr(config, "MODEL_DIR", tmp_model_dir)
    monkeypatch.setattr(config, "REPORTS_METRICS_DIR", tmp_metrics_dir)
    monkeypatch.setattr(config, "REPORTS_FIGURES_DIR", tmp_figures_dir)

    config.ensure_directories()

    assert tmp_model_dir.exists()
    assert tmp_metrics_dir.exists()
    assert tmp_figures_dir.exists()