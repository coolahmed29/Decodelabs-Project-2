"""
PURPOSE: DataLoader ke saare success aur failure paths test karna, bina real
         large dataset ke — chhote temporary/mock CSV files use karke.
"""

import pandas as pd
import pytest

from src.data.loader import DataLoader

VALID_CSV = "transaction_id,amount,is_fraud\n1,100.0,0\n2,250.5,1\n3,75.0,0\n"


def test_load_valid_csv_returns_dataframe(tmp_path):
    file_path = tmp_path / "transactions.csv"
    file_path.write_text(VALID_CSV, encoding="utf-8")

    df = DataLoader(file_path).load()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3


def test_load_nonexistent_file_raises_error(tmp_path):
    fake_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        DataLoader(fake_path).load()


def test_load_empty_csv_raises_error(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("transaction_id,amount,is_fraud\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset is empty"):
        DataLoader(file_path).load()


def test_load_unsupported_extension_raises_error(tmp_path):
    file_path = tmp_path / "data.json"
    file_path.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        DataLoader(file_path).load()


def test_get_shape_summary_returns_correct_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    summary = DataLoader("unused.csv").get_shape_summary(df)

    assert summary == {"rows": 3, "columns": 2, "column_names": ["a", "b"]}


def test_missing_target_column_logs_warning_not_error(tmp_path, caplog):
    file_path = tmp_path / "no_target.csv"
    file_path.write_text("transaction_id,amount\n1,100.0\n2,250.5\n", encoding="utf-8")

    df = DataLoader(file_path).load()

    assert isinstance(df, pd.DataFrame)
    assert "is_fraud" not in df.columns
    assert any(
        record.levelname == "WARNING" and "is_fraud" in record.getMessage()
        for record in caplog.records
    )