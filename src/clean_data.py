"""Clean step: fixed cleaning rules plus booking-time inputs derived with the zone table.

Inputs: raw trips, zone table and SETTINGS["cleaning"] (from main.py; bare calls: config.yaml).
Outputs: the cleaned DataFrame; training mode adds the label and drops untrusted rows.
Used by: main.py, before validate.py; the same derivations serve training and inference.
Why: no post-trip leakage, no bad labels, no training-serving skew; inference keeps every row.
"""

import logging

import numpy as np
import pandas as pd

from src import config, utils

logger = logging.getLogger(__name__)


def _resolve_inputs(zone_table: pd.DataFrame | None, rules: dict | None) -> tuple:
    """Fall back to config.yaml when the zone table or the rules are not passed (contract calls)."""
    if zone_table is not None and rules is not None:
        return zone_table, rules
    settings = config.load_config(
        config.CONFIG_PATH, required_sections=("paths", "cleaning")
    )
    if rules is None:
        logger.info(
            "No cleaning rules passed: using the cleaning section of %s",
            config.CONFIG_PATH,
        )
        rules = settings["cleaning"]
    if zone_table is None:
        zone_path = config.CONFIG_PATH.parent / settings["paths"]["zone_table"]
        logger.info("No zone table passed: loading %s", zone_path)
        zone_table = utils.load_csv(zone_path)
    return zone_table, rules


def _parse_pickup_times(df: pd.DataFrame, rules: dict) -> pd.Series:
    """Parse pickup times; stop the run if any is missing or cannot be read as a date-time."""
    column = rules["columns"]["pickup_datetime"]
    pickup = pd.to_datetime(
        df[column], format=rules["datetime_format"], errors="coerce"
    )
    bad = int(pickup.isna().sum())
    if bad:
        raise ValueError(
            f"{bad} row(s) have a missing or unparseable {column}; fix the source data"
        )
    return pickup


def _filter_training_rows(
    df: pd.DataFrame, pickup: pd.Series, target_column: str, rules: dict
) -> pd.DataFrame:
    """Training only: build the label in minutes and keep rows whose label can be trusted."""
    columns = rules["columns"]
    dropoff = pd.to_datetime(
        df[columns["dropoff_datetime"]],
        format=rules["datetime_format"],
        errors="coerce",
    )
    minutes = (dropoff - pickup).dt.total_seconds() / 60
    distance = df[columns["meter_distance"]]
    speed_kmh = distance * rules["miles_to_km"] / (minutes / 60)
    zones = df[[columns["pickup_zone"], columns["dropoff_zone"]]]
    keep = (
        minutes.between(rules["min_minutes"], rules["max_minutes"])
        & ~zones.isin(rules["unknown_zones"]).any(axis=1)
        & (distance > 0)
        & (speed_kmh <= rules["max_speed_kmh"])
    )
    df[target_column] = minutes
    return df[keep]


def _drop_unused_columns(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Drop post-trip, dispatch-time and unused columns that are present (both modes)."""
    unused = (
        rules["leakage_columns"]
        + rules["not_at_booking_columns"]
        + rules["other_drop_columns"]
    )
    return df.drop(columns=[column for column in unused if column in df.columns])


def _add_time_features(
    df: pd.DataFrame, pickup: pd.Series, rules: dict
) -> pd.DataFrame:
    """Add pickup hour, weekday (Monday = 0) and a weekend flag from the parsed pickup times."""
    derived = rules["derived_columns"]
    pickup = pickup.loc[df.index]  # training mode may have dropped rows
    df[derived["hour"]] = pickup.dt.hour
    df[derived["weekday"]] = pickup.dt.day_of_week
    df[derived["weekend"]] = (pickup.dt.day_of_week >= 5).astype(int)  # 5 = Saturday
    return df


def _add_zone_features(
    df: pd.DataFrame, zone_table: pd.DataFrame, rules: dict
) -> pd.DataFrame:
    """Add boroughs, airport flags, straight-line distance, same-zone flag and zone pair."""
    columns, derived, zone_cols = (
        rules["columns"],
        rules["derived_columns"],
        rules["zone_columns"],
    )
    zone_info = zone_table.set_index(zone_cols["zone_id"])
    centres = {}
    for side in ("pickup", "dropoff"):
        zone = df[columns[f"{side}_zone"]]
        borough = zone.map(zone_info[zone_cols["zone_borough"]])
        df[derived[f"{side}_borough"]] = borough.fillna(rules["unknown_borough"])
        df[derived[f"{side}_airport"]] = zone.isin(rules["airport_zones"]).astype(int)
        centres[side] = [
            zone.map(zone_info[zone_cols[axis]]) for axis in ("centre_x", "centre_y")
        ]
    df[derived["distance"]] = np.hypot(
        centres["pickup"][0] - centres["dropoff"][0],
        centres["pickup"][1] - centres["dropoff"][1],
    )
    pickup_zone, dropoff_zone = df[columns["pickup_zone"]], df[columns["dropoff_zone"]]
    df[derived["same_zone_flag"]] = (pickup_zone == dropoff_zone).astype(int)
    df[derived["pair"]] = pickup_zone.astype(str) + "_" + dropoff_zone.astype(str)
    return df


def _add_route_keys(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Add route x time-of-day keys, e.g. "161_237|morning|wd", for the target encoder."""
    columns, derived = rules["columns"], rules["derived_columns"]
    band = pd.cut(
        df[derived["hour"]],
        bins=rules["time_band_edges"],
        labels=rules["time_band_labels"],
    ).astype(str)
    weekday_label, weekend_label = rules["day_types"]
    day_type = np.where(df[derived["weekend"]] == 1, weekend_label, weekday_label)
    when = band + "|" + day_type
    df[derived["pair_band_key"]] = df[derived["pair"]] + "|" + when
    df[derived["pickup_band_key"]] = df[columns["pickup_zone"]].astype(str) + "|" + when
    df[derived["dropoff_band_key"]] = (
        df[columns["dropoff_zone"]].astype(str) + "|" + when
    )
    return df


def clean_dataframe(
    df_raw: pd.DataFrame,
    target_column: str | None = None,
    zone_table: pd.DataFrame | None = None,
    rules: dict | None = None,
) -> pd.DataFrame:
    """Apply fixed cleaning rules and derive booking-time inputs.

    Inputs: df_raw, raw trips or bookings; target_column, given in training mode, None for
    inference; zone_table, the zone table; rules, SETTINGS["cleaning"] (both from config.yaml
    if None).
    Outputs: a new cleaned DataFrame with a fresh index; training mode adds target_column.
    Why: one function for both modes, so training and inference derive inputs identically;
    inference never drops a booking.
    """
    mode = "training" if target_column is not None else "inference"
    logger.info("Cleaning %s rows (%s mode)", len(df_raw), mode)
    zone_table, rules = _resolve_inputs(zone_table, rules)
    df = df_raw.copy()  # never modify the caller's DataFrame
    pickup = _parse_pickup_times(df, rules)
    if target_column is not None:
        df = _filter_training_rows(df, pickup, target_column, rules)
        dropped = len(df_raw) - len(df)
        if dropped:
            logger.warning(
                "Dropped %s of %s training rows (cleaning rules)", dropped, len(df_raw)
            )
    df = _drop_unused_columns(df, rules)
    df = _add_time_features(df, pickup, rules)
    df = _add_zone_features(df, zone_table, rules)
    df = _add_route_keys(df, rules)
    return df.reset_index(drop=True)
