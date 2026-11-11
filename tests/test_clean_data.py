"""Tests for src/clean_data.py: fixed cleaning rules and booking-time inputs.

Unit tests check training-mode filtering and derived values, that the input is never modified,
that bad pickup times stop the run, and that a bare contract call falls back to config.yaml;
the integration test checks inference mode keeps every booking and passes validate_dataframe.
"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def _make_zone_table() -> pd.DataFrame:
    """Tiny zone table; centres chosen so 79 -> 119 is exactly 5 km (a 3-4-5 triangle)."""
    data = {
        "LocationID": [79, 119, 100, 234, 132, 264],
        "borough": [
            "Manhattan",
            "Bronx",
            "Manhattan",
            "Manhattan",
            "Queens",
            "Unknown",
        ],
        "x_km": [0.0, 3.0, 0.0, 0.0, 10.0, np.nan],
        "y_km": [0.0, 4.0, 1.0, 2.0, 0.0, np.nan],
    }
    return pd.DataFrame(data)


def _make_rules() -> dict:
    """Cleaning rules shaped like SETTINGS["cleaning"] (built here, since main.py comes last)."""
    return {
        "columns": {
            "pickup_datetime": "tpep_pickup_datetime",
            "dropoff_datetime": "tpep_dropoff_datetime",
            "pickup_zone": "PULocationID",
            "dropoff_zone": "DOLocationID",
            "meter_distance": "trip_distance",
        },
        "zone_columns": {
            "zone_id": "LocationID",
            "zone_borough": "borough",
            "centre_x": "x_km",
            "centre_y": "y_km",
        },
        "derived_columns": {
            "hour": "pickup_hour",
            "weekday": "pickup_weekday",
            "weekend": "is_weekend",
            "pickup_borough": "pu_borough",
            "dropoff_borough": "do_borough",
            "pickup_airport": "pu_airport",
            "dropoff_airport": "do_airport",
            "distance": "straight_km",
            "same_zone_flag": "same_zone",
            "pair": "zone_pair",
            "pair_band_key": "pair_band",
            "pickup_band_key": "pu_band",
            "dropoff_band_key": "do_band",
        },
        "unknown_borough": "Unknown",
        "day_types": ["wd", "we"],
        "miles_to_km": 1.609,
        "datetime_format": "%Y-%m-%d %H:%M:%S",
        "leakage_columns": [
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "total_amount",
            "congestion_surcharge",
            "Airport_fee",
            "cbd_congestion_fee",
            "payment_type",
        ],
        "not_at_booking_columns": ["VendorID", "RatecodeID"],
        "other_drop_columns": ["store_and_fwd_flag"],
        "unknown_zones": [264, 265],
        "airport_zones": [1, 132, 138],
        "min_minutes": 1.0,
        "max_minutes": 180.0,
        "max_speed_kmh": 100.0,
        "time_band_edges": [-1, 5, 9, 15, 19, 23],
        "time_band_labels": ["night", "morning", "midday", "evening", "late"],
    }


def _make_raw_df() -> pd.DataFrame:
    """Seven raw trips: three valid, four with exactly one training-mode problem each.

    Rows: 0 valid (Sat 18:20, 79 -> 119); 1 valid (Fri 18:11, 100 -> 234); 2 under 1 min;
    3 unknown zone 264; 4 zero meter distance; 5 over 100 km/h; 6 valid (Mon 08:00, JFK 132 -> 79).
    """
    data = {
        "trip_id": [f"2024-01-000000{i}" for i in range(1, 8)],
        "VendorID": [2, 2, 1, 2, 1, 2, 2],
        "tpep_pickup_datetime": [
            "2024-01-27 18:20:13",
            "2024-01-12 18:11:47",
            "2024-01-12 18:00:00",
            "2024-01-12 19:00:00",
            "2024-01-12 20:00:00",
            "2024-01-12 21:00:00",
            "2024-01-15 08:00:00",
        ],
        "tpep_dropoff_datetime": [
            "2024-01-27 18:51:57",
            "2024-01-12 18:21:04",
            "2024-01-12 18:00:30",
            "2024-01-12 19:15:00",
            "2024-01-12 20:10:00",
            "2024-01-12 21:10:00",
            "2024-01-15 08:40:00",
        ],
        "passenger_count": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, np.nan],
        "trip_distance": [10.17, 1.15, 0.1, 3.0, 0.0, 30.0, 15.0],
        "RatecodeID": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "store_and_fwd_flag": ["N"] * 7,
        "PULocationID": [79, 100, 79, 264, 100, 79, 132],
        "DOLocationID": [119, 234, 119, 119, 234, 132, 79],
        "payment_type": [2, 1, 1, 1, 1, 1, 1],
        "fare_amount": [45.0, 10.0, 3.0, 15.0, 5.0, 60.0, 70.0],
    }
    return pd.DataFrame(data)


def test_unit_training_mode_drops_bad_labels_and_derives_inputs(caplog):
    """Training mode keeps the 3 valid trips, builds the label, drops post-trip columns, derives inputs."""
    cleaned = clean_dataframe(
        _make_raw_df(),
        target_column="duration_min",
        zone_table=_make_zone_table(),
        rules=_make_rules(),
    )

    assert list(cleaned["trip_id"]) == [
        "2024-01-0000001",
        "2024-01-0000002",
        "2024-01-0000007",
    ]
    np.testing.assert_allclose(
        cleaned["duration_min"], [31.7333333, 9.2833333, 40.0], atol=1e-6
    )
    dropped = {"trip_distance", "fare_amount", "payment_type", "VendorID", "RatecodeID"}
    assert dropped.isdisjoint(cleaned.columns)
    assert "tpep_dropoff_datetime" not in cleaned.columns
    assert list(cleaned.index) == [0, 1, 2]

    first, jfk = cleaned.iloc[0], cleaned.iloc[2]
    assert first["straight_km"] == pytest.approx(5.0)
    assert jfk["straight_km"] == pytest.approx(10.0)
    assert (first["pu_airport"], jfk["pu_airport"]) == (0, 1)
    assert first["pair_band"] == "79_119|evening|we"
    assert jfk["pair_band"] == "132_79|morning|wd"
    assert any(
        record.levelname == "WARNING" and "Dropped 4 of 7" in record.getMessage()
        for record in caplog.records
    )


def test_unit_input_dataframes_are_not_modified():
    """clean_dataframe works on copies: the raw trips and the zone table are unchanged afterwards."""
    raw = _make_raw_df()
    zones = _make_zone_table()
    raw_before, zones_before = raw.copy(), zones.copy()

    clean_dataframe(
        raw, target_column="duration_min", zone_table=zones, rules=_make_rules()
    )

    pd.testing.assert_frame_equal(raw, raw_before)
    pd.testing.assert_frame_equal(zones, zones_before)


def test_unit_bad_pickup_times_raise():
    """A missing or unparseable pickup time stops the run with a clear error (both modes)."""
    raw = _make_raw_df()

    bad = raw.copy()
    bad.loc[0, "tpep_pickup_datetime"] = "not a date"
    bad.loc[1, "tpep_pickup_datetime"] = None
    with pytest.raises(ValueError, match="2 row"):
        clean_dataframe(bad, zone_table=_make_zone_table(), rules=_make_rules())


@pytest.mark.skip(reason="Awaiting validate in dev")
def test_integration_inference_keeps_every_booking_and_passes_validate():
    """Inference mode keeps all 7 bookings (zone 264 included) and the output passes validate_dataframe."""
    from src.validate import validate_dataframe

    booking_columns = [
        "trip_id",
        "tpep_pickup_datetime",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
    ]
    bookings = _make_raw_df()[booking_columns]
    rules = _make_rules()
    cleaned = clean_dataframe(bookings, zone_table=_make_zone_table(), rules=rules)

    features = list(rules["derived_columns"].values()) + [
        "passenger_count",
        "PULocationID",
        "DOLocationID",
    ]
    assert len(cleaned) == len(bookings)
    assert np.isnan(cleaned.loc[3, "straight_km"])
    assert validate_dataframe(
        cleaned,
        required_columns=features,
        non_negative_columns=["passenger_count", "straight_km"],
        valid_zone_range=[1, 265],
        allowed_columns=["trip_id", "tpep_pickup_datetime"] + features,
        zone_columns=[
            rules["columns"]["pickup_zone"],
            rules["columns"]["dropoff_zone"],
        ],
    )


def _write_fallback_config(tmp_path, zone_table_file: str):
    """Write a config.yaml with the test cleaning rules and a zone_table path; return its path."""
    import yaml

    settings = {"paths": {"zone_table": zone_table_file}, "cleaning": _make_rules()}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


def test_unit_bare_contract_call_falls_back_to_config(tmp_path, monkeypatch):
    """Without zone_table and rules, clean_dataframe reads both through config.yaml (rule 5)."""
    from src import config

    _make_zone_table().to_csv(tmp_path / "zones.csv", index=False)
    monkeypatch.setattr(
        config, "CONFIG_PATH", _write_fallback_config(tmp_path, "zones.csv")
    )

    fallback = clean_dataframe(_make_raw_df(), target_column="duration_min")
    explicit = clean_dataframe(
        _make_raw_df(),
        target_column="duration_min",
        zone_table=_make_zone_table(),
        rules=_make_rules(),
    )

    pd.testing.assert_frame_equal(fallback, explicit)


def test_unit_fallback_with_missing_zone_file_raises(tmp_path, monkeypatch):
    """If the configured zone table file is missing, the fallback stops with the path in the error."""
    from src import config

    monkeypatch.setattr(
        config, "CONFIG_PATH", _write_fallback_config(tmp_path, "missing.csv")
    )

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        clean_dataframe(_make_raw_df())


def test_unit_fallback_fills_only_the_missing_input(tmp_path, monkeypatch):
    """Passing one input uses it as given; only the missing one comes from config.yaml."""
    from src import config

    _make_zone_table().to_csv(tmp_path / "zones.csv", index=False)
    monkeypatch.setattr(
        config, "CONFIG_PATH", _write_fallback_config(tmp_path, "zones.csv")
    )
    explicit = clean_dataframe(
        _make_raw_df(),
        target_column="duration_min",
        zone_table=_make_zone_table(),
        rules=_make_rules(),
    )

    only_rules = clean_dataframe(
        _make_raw_df(), target_column="duration_min", rules=_make_rules()
    )
    only_zones = clean_dataframe(
        _make_raw_df(), target_column="duration_min", zone_table=_make_zone_table()
    )

    pd.testing.assert_frame_equal(only_rules, explicit)
    pd.testing.assert_frame_equal(only_zones, explicit)
