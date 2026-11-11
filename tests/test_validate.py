"""Tests for src/validate.py: validate_dataframe returns True or raises ValueError, never fixes.

Unit tests check valid data (missing feature values included) passes and each broken rule raises;
the integration test checks the output of clean_dataframe passes with the brief's rules.
"""

import numpy as np
import pandas as pd
import pytest

from src.validate import validate_dataframe


def _make_clean_df() -> pd.DataFrame:
    """Three cleaned trips (a column subset); missing passenger_count and straight_km are allowed."""
    data = {
        "trip_id": ["2024-01-0000001", "2024-01-0000002", "2024-01-0000003"],
        "tpep_pickup_datetime": [
            "2024-01-27 18:20:13",
            "2024-01-12 18:11:47",
            "2024-01-15 08:00:00",
        ],
        "duration_min": [31.7, 9.3, 40.0],
        "passenger_count": [1.0, np.nan, 2.0],
        "PULocationID": [79, 100, 264],
        "DOLocationID": [119, 234, 79],
        "straight_km": [5.0, 1.0, np.nan],
        "pu_borough": ["Manhattan", "Manhattan", "Unknown"],
    }
    return pd.DataFrame(data)


def _make_rules() -> dict:
    """Keyword arguments shaped as main.py will pass them (SETTINGS["validation"] plus columns)."""
    features = [
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "straight_km",
        "pu_borough",
    ]
    return {
        "required_columns": features,
        "target_column": "duration_min",
        "non_negative_columns": ["passenger_count", "straight_km"],
        "valid_zone_range": [1, 265],
        "target_range": [1.0, 180.0],
        "allowed_columns": ["trip_id", "tpep_pickup_datetime", "duration_min"]
        + features,
        "zone_columns": ["PULocationID", "DOLocationID"],
    }


def test_unit_valid_data_returns_true_and_changes_nothing():
    """Valid data returns True, even with missing passenger_count and straight_km values."""
    df = _make_clean_df()
    before = df.copy()

    assert validate_dataframe(df, **_make_rules()) is True
    pd.testing.assert_frame_equal(df, before)


def test_unit_each_broken_rule_raises():
    """Each broken rule stops the run with its own message; one rule is broken per case."""
    rules = _make_rules()
    good = _make_clean_df()

    with pytest.raises(ValueError, match="no rows"):
        validate_dataframe(good.iloc[0:0], **rules)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(good.drop(columns=["straight_km"]), **rules)
    with pytest.raises(ValueError, match="outside the data contract"):
        validate_dataframe(good.assign(rider_phone="555-0100"), **rules)
    with pytest.raises(ValueError, match="PULocationID outside 1-265"):
        validate_dataframe(good.assign(PULocationID=[999, 100, 264]), **rules)
    with pytest.raises(ValueError, match="needs zone_columns"):
        validate_dataframe(good, **{**rules, "zone_columns": None})
    with pytest.raises(ValueError, match="not_a_column"):
        validate_dataframe(good, **{**rules, "non_negative_columns": ["not_a_column"]})
    with pytest.raises(ValueError, match="negative passenger_count"):
        validate_dataframe(good.assign(passenger_count=[-1.0, np.nan, 2.0]), **rules)
    with pytest.raises(ValueError, match="duration_min not found"):
        validate_dataframe(good.drop(columns=["duration_min"]), **rules)
    with pytest.raises(ValueError, match="no duration_min"):
        validate_dataframe(good.assign(duration_min=[np.nan, 9.3, 40.0]), **rules)
    with pytest.raises(ValueError, match="duration_min outside 1.0-180.0"):
        validate_dataframe(good.assign(duration_min=[500.0, 9.3, 40.0]), **rules)


def test_unit_unset_rules_are_skipped():
    """Without allow-list, zone range or target range, valid data still passes (rules are optional)."""
    rules = {
        **_make_rules(),
        "allowed_columns": None,
        "valid_zone_range": None,
        "target_range": None,
    }

    assert validate_dataframe(_make_clean_df(), **rules) is True


def test_integration_clean_dataframe_output_passes_validation():
    """Training-mode output of clean_dataframe passes with arguments built as main.py will build them."""
    from src.clean_data import clean_dataframe
    from tests.test_clean_data import _make_raw_df, _make_zone_table
    from tests.test_clean_data import _make_rules as _make_cleaning_rules

    cleaning = _make_cleaning_rules()
    cleaned = clean_dataframe(
        _make_raw_df(),
        target_column="duration_min",
        zone_table=_make_zone_table(),
        rules=cleaning,
    )
    columns = cleaning["columns"]
    zones = [columns["pickup_zone"], columns["dropoff_zone"]]
    features = list(cleaning["derived_columns"].values()) + ["passenger_count"] + zones

    assert validate_dataframe(
        cleaned,
        required_columns=features,
        target_column="duration_min",
        non_negative_columns=["passenger_count", "straight_km"],
        valid_zone_range=[1, 265],
        target_range=[cleaning["min_minutes"], cleaning["max_minutes"]],
        allowed_columns=["trip_id", columns["pickup_datetime"], "duration_min"]
        + features,
        zone_columns=zones,
    )
