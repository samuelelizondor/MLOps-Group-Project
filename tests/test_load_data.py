"""Tests for src/load_data.py: read the raw trips CSV and fail fast on bad inputs.

Unit tests check a valid file loads unchanged and each bad input stops the run;
the integration test reads a file written by utils.save_csv.
"""

import pandas as pd
import pytest

from src.load_data import load_raw_data


def _make_raw_df() -> pd.DataFrame:
    """Three raw trips with TLC column names (a subset), one with a missing passenger_count."""
    data = {
        "trip_id": ["2024-01-0000001", "2024-01-0000002", "2024-01-0000003"],
        "tpep_pickup_datetime": [
            "2024-01-27 18:20:13",
            "2024-01-12 18:11:47",
            "2024-01-19 11:58:03",
        ],
        "tpep_dropoff_datetime": [
            "2024-01-27 18:51:57",
            "2024-01-12 18:21:04",
            "2024-01-19 12:47:59",
        ],
        "passenger_count": [1.0, 1.0, None],
        "trip_distance": [10.17, 1.15, 6.14],
        "PULocationID": [79, 100, 166],
        "DOLocationID": [119, 234, 186],
    }
    return pd.DataFrame(data)


def test_unit_load_raw_data_returns_file_unchanged(tmp_path):
    """A valid CSV loads with the expected shape and values; missing values stay missing."""
    df = _make_raw_df()
    filepath = tmp_path / "raw.csv"
    df.to_csv(filepath, index=False)

    loaded = load_raw_data(filepath)

    assert loaded.shape == (3, 7)
    assert loaded["passenger_count"].isna().sum() == 1
    pd.testing.assert_frame_equal(loaded, df)


def test_unit_load_raw_data_bad_inputs_raise(tmp_path):
    """Each bad input stops the run with its own error: missing file, folder, 0-byte file, header only."""
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        load_raw_data(tmp_path / "missing.csv")

    with pytest.raises(IsADirectoryError):
        load_raw_data(tmp_path)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_raw_data(empty)

    header_only = tmp_path / "header_only.csv"
    header_only.write_text("trip_id,PULocationID,DOLocationID\n")
    with pytest.raises(ValueError, match="no rows"):
        load_raw_data(header_only)


def test_integration_load_raw_data_reads_file_written_by_save_csv(tmp_path):
    """A file written by utils.save_csv (the pipeline's writer) loads back unchanged."""
    from src.utils import save_csv

    df = _make_raw_df()
    filepath = tmp_path / "data" / "raw" / "trips.csv"
    save_csv(df, filepath)

    loaded = load_raw_data(filepath)

    pd.testing.assert_frame_equal(loaded, df)
