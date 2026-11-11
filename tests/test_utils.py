"""Tests for src/utils.py: CSV and model read/write helpers.

Unit tests check the CSV round trip and fail-fast reads; the integration test checks a saved model reloads.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.utils import load_csv, load_model, save_csv, save_model


def _make_df() -> pd.DataFrame:
    """Tiny trip-like table: a text ID, a text column and two numeric columns."""
    data = {
        "trip_id": [
            "2024-01-0000001",
            "2024-01-0000002",
            "2024-01-0000003",
            "2024-01-0000004",
        ],
        "pu_borough": ["Manhattan", "Queens", "Manhattan", "Bronx"],
        "straight_km": [1.5, 12.0, 3.2, 8.4],
        "duration_min": [8.0, 35.0, 14.0, 27.0],
    }
    return pd.DataFrame(data)


def test_unit_save_then_load_csv_returns_same_data(tmp_path):
    """save_csv creates missing folders and writes no index; load_csv reads the same table back."""
    df = _make_df()
    filepath = tmp_path / "nested" / "folder" / "trips.csv"

    save_csv(df, filepath)
    loaded = load_csv(filepath)

    assert filepath.is_file()
    assert list(loaded.columns) == [
        "trip_id",
        "pu_borough",
        "straight_km",
        "duration_min",
    ]
    pd.testing.assert_frame_equal(loaded, df)


def test_unit_load_csv_missing_file_raises(tmp_path):
    """load_csv fails fast with FileNotFoundError and names the missing path."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="does_not_exist.csv"):
        load_csv(missing)


def test_unit_load_model_missing_file_raises(tmp_path):
    """load_model fails fast with FileNotFoundError and names the missing path."""
    missing = tmp_path / "no_model.joblib"

    with pytest.raises(FileNotFoundError, match="no_model.joblib"):
        load_model(missing)


def test_integration_save_then_load_model_gives_same_predictions(tmp_path):
    """A model saved with save_model and reloaded with load_model predicts exactly the same values."""
    df = _make_df()
    X = df[["straight_km"]]
    y = df["duration_min"]
    model = LinearRegression().fit(X, y)
    filepath = tmp_path / "models" / "model.joblib"

    save_model(model, filepath)
    reloaded = load_model(filepath)

    assert filepath.is_file()
    np.testing.assert_allclose(reloaded.predict(X), model.predict(X))
