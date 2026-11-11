"""Validate step: check cleaned data against the agreed rules; return True or raise ValueError.

Inputs: the cleaned DataFrame, required and allowed columns, SETTINGS["validation"] rules.
Outputs: True when every rule holds; otherwise a ValueError naming the broken rule.
Used by: main.py, after clean_data.py (training and inference), before the split or predict.
Why: bad data stops the run before training or quoting; validation never repairs anything.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _check_columns(
    df: pd.DataFrame, required_columns: list[str], allowed_columns: list[str] | None
) -> None:
    """Stop the run on an empty table, missing required columns or columns outside the contract."""
    if df.empty:
        raise ValueError("Data has no rows; check the input file")
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if allowed_columns is not None:
        extra = [column for column in df.columns if column not in allowed_columns]
        if extra:
            raise ValueError(
                f"Columns outside the data contract: {extra}; remove them before the pipeline"
            )


def _check_zone_range(
    df: pd.DataFrame, zone_columns: list[str] | None, valid_zone_range: list[int] | None
) -> None:
    """Stop the run if any zone ID falls outside the zone map (app and map out of sync)."""
    if valid_zone_range is None:
        return
    if not zone_columns:
        raise ValueError(
            "valid_zone_range needs zone_columns; main.py passes them from SETTINGS"
        )
    low, high = valid_zone_range
    for column in zone_columns:
        bad = ~df[column].between(low, high)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} row(s) have a {column} outside {low}-{high}; update zone table"
            )


def _check_non_negative(
    df: pd.DataFrame, non_negative_columns: list[str] | None
) -> None:
    """Stop the run on negative values; missing values pass and are imputed later by the recipe."""
    for column in non_negative_columns or []:
        if column not in df.columns:
            raise ValueError(
                f"Non-negative rule names {column}, which is not in the data"
            )
        negative = df[column] < 0
        if negative.any():
            raise ValueError(
                f"{int(negative.sum())} row(s) have a negative {column}; fix the source data"
            )


def _check_target(
    df: pd.DataFrame, target_column: str | None, target_range: list[float] | None
) -> None:
    """Training only: stop the run if the label is missing or outside its allowed range."""
    if target_column is None:
        return
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column} not found; check SETTINGS['problem']"
        )
    missing = df[target_column].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} row(s) have no {target_column} value")
    if target_range is not None:
        low, high = target_range
        bad = ~df[target_column].between(low, high)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} row(s) have a {target_column} outside {low}-{high}"
            )


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    target_column: str | None = None,
    non_negative_columns: list[str] | None = None,
    valid_zone_range: list[int] | None = None,
    target_range: list[float] | None = None,
    allowed_columns: list[str] | None = None,
    zone_columns: list[str] | None = None,
) -> bool:
    """Return True if the data meets every rule, otherwise raise ValueError.

    Inputs: df, cleaned data; required_columns; target_column (training only); rule values from
    SETTINGS["validation"]; allowed_columns, the data contract; zone_columns, the zone ID columns.
    Outputs: True when every rule holds.
    Why: bad data stops the run here with a message naming the rule; nothing is ever repaired.
    """
    logger.info(
        "Validating %s rows against %s required columns", len(df), len(required_columns)
    )
    _check_columns(df, required_columns, allowed_columns)
    _check_zone_range(df, zone_columns, valid_zone_range)
    _check_non_negative(df, non_negative_columns)
    _check_target(df, target_column, target_range)
    return True
