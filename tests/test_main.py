"""Tests for src/main.py: split_data, the quality gates, and the whole pipeline on tiny data.

Unit tests check the time-ordered split and the gate statuses; the pipeline test runs main() on
synthetic CSVs in tmp_path and checks the three artifacts and the predictions file.
"""

import numpy as np
import pandas as pd
import pytest

from src import main as pipeline
from src.main import check_quality_gates, split_data


def _make_trips(rows: int, month: str, seed: int) -> pd.DataFrame:
    """Synthetic raw TLC trips in known zones; trip time grows with meter distance, plus noise."""
    rng = np.random.default_rng(seed)
    zones = np.array([79, 119, 100, 234, 132])
    start = pd.Timestamp(f"{month}-01")
    pickup = start + pd.to_timedelta(rng.integers(0, 27 * 24 * 60, rows), unit="min")
    miles = rng.uniform(0.5, 8.0, rows)
    minutes = 4.0 + 3.0 * miles + rng.uniform(0.0, 6.0, rows)
    dropoff = pickup + pd.to_timedelta(minutes, unit="min")
    data = {
        "trip_id": [f"{month}-{i:07d}" for i in range(1, rows + 1)],
        "tpep_pickup_datetime": pickup.strftime("%Y-%m-%d %H:%M:%S"),
        "tpep_dropoff_datetime": dropoff.strftime("%Y-%m-%d %H:%M:%S"),
        "passenger_count": rng.integers(1, 4, rows).astype(float),
        "trip_distance": miles.round(2),
        "PULocationID": rng.choice(zones, rows),
        "DOLocationID": rng.choice(zones, rows),
        "fare_amount": (3.0 + 2.5 * miles).round(2),
    }
    return pd.DataFrame(data)


def _write_inputs(root) -> pd.DataFrame:
    """Write raw trips, the zone table and new bookings where SETTINGS expects them under root."""
    from tests.test_clean_data import _make_zone_table

    booking_columns = ["trip_id", "tpep_pickup_datetime", "passenger_count"]
    booking_columns += ["PULocationID", "DOLocationID"]
    bookings = _make_trips(15, "2024-02", seed=2)[booking_columns]
    inputs = {
        "raw_data": _make_trips(120, "2024-01", seed=1),
        "zone_table": _make_zone_table(),
        "inference_data": bookings,
    }
    for key, frame in inputs.items():
        target = root / pipeline.SETTINGS["paths"][key]
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    return bookings


def test_unit_split_data_is_time_ordered_with_expected_sizes():
    """100 rows, 20% test and 20% validation of the whole set: 60/20/20, in order, no overlap."""
    X = pd.DataFrame({"row": range(100)})
    y = pd.Series(range(100))

    X_train, X_val, X_test, _, _, y_test = split_data(
        X,
        y,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        stratify=False,
        shuffle=False,
    )

    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert list(X_train["row"]) == list(range(60))
    assert list(X_val["row"]) == list(range(60, 80))
    assert list(X_test["row"]) == list(range(80, 100))
    assert list(y_test) == list(range(80, 100))
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_data(X, y, 0.5, 0.5, 42, False)
    with pytest.raises(ValueError, match="stratify needs shuffle"):
        split_data(X, y, 0.2, 0.2, 42, True, shuffle=False)


def test_unit_quality_gates_report_met_not_met_and_not_set():
    """Each gate reports met or not met (boundaries count as met); bad gates raise."""
    metrics = {
        "mae": 4.10,
        "on_time_share": 0.57,
        "bias": -0.86,
        "big_miss_share": 0.05,
    }
    gates = [
        {"metric_key": "mae", "comparison": "<=", "threshold": 4.10},
        {"metric_key": "on_time_share", "comparison": ">=", "threshold": 0.577},
        {"metric_key": "bias", "comparison": "between", "threshold": [-1.0, 2.0]},
        {"metric_key": "big_miss_share", "comparison": "<=", "threshold": None},
    ]

    assert check_quality_gates(metrics, gates) == {
        "mae": "met",
        "on_time_share": "not met",
        "bias": "met",
        "big_miss_share": "not set",
    }
    unknown_metric = [{"metric_key": "r2", "comparison": ">=", "threshold": 0.5}]
    with pytest.raises(ValueError, match="r2"):
        check_quality_gates(metrics, unknown_metric)
    unknown_comparison = [{"metric_key": "mae", "comparison": "<", "threshold": 4.0}]
    with pytest.raises(ValueError, match="Unknown comparison"):
        check_quality_gates(metrics, unknown_comparison)


def test_integration_main_runs_end_to_end_on_tiny_data(tmp_path, monkeypatch, caplog):
    """main() writes processed data, model and predictions; one quote per booking, ID first."""
    from src.utils import load_model

    bookings = _write_inputs(tmp_path)
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    pipeline.main()

    settings = pipeline.SETTINGS
    paths = {key: tmp_path / value for key, value in settings["paths"].items()}
    id_column = settings["problem"]["id_column"]
    eta_column = settings["output"]["prediction_column"]
    assert paths["processed_data"].is_file()
    predictions = pd.read_csv(paths["predictions"])
    assert len(predictions) == len(bookings)
    assert predictions.columns[0] == id_column
    assert predictions.columns[-1] == eta_column
    assert list(predictions[id_column]) == list(bookings["trip_id"])
    assert predictions[eta_column].min() >= settings["cleaning"]["min_minutes"]
    assert hasattr(load_model(paths["model"]), "predict")
    log_text = paths["log_file"].read_text(encoding="utf-8")
    assert " | INFO | src.utils | Loading CSV: " in log_text
    assert "not for release" in caplog.text


def test_integration_main_exits_1_when_zone_table_is_missing(
    tmp_path, monkeypatch, caplog
):
    """Bad inputs stop the run before training: a 'Pipeline stopped' error in the log, exit 1."""
    _write_inputs(tmp_path)
    (tmp_path / pipeline.SETTINGS["paths"]["zone_table"]).unlink()
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    with pytest.raises(SystemExit) as stopped:
        pipeline.main()

    assert stopped.value.code == 1
    assert any(
        record.levelname == "ERROR" and "Pipeline stopped" in record.getMessage()
        for record in caplog.records
    )
    assert not (tmp_path / pipeline.SETTINGS["paths"]["model"]).exists()


def test_integration_entry_point_calls_main_and_exits_1_without_data(
    tmp_path, monkeypatch
):
    """Running main.py as a script goes through its __main__ guard; missing data exits 1."""
    import runpy

    import yaml

    from src import config

    settings = config.load_config(config.CONFIG_PATH)
    settings["paths"] = {
        key: str(tmp_path / value) for key, value in settings["paths"].items()
    }
    test_config = tmp_path / "config.yaml"
    test_config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", test_config)

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(pipeline.__file__, run_name="__main__")

    assert stopped.value.code == 1
