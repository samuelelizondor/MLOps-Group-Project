"""Orchestrator: run the whole pipeline with one command, python -m src.main.

Inputs: config.yaml (read through src/config.py); raw trips, zone table, bookings.
Outputs: data/processed/clean.csv, models/model.joblib, reports/predictions.csv.
Flow: load, clean, validate, split by time, train, evaluate and gate on validation, test audit,
save the model, then clean, validate and quote the new bookings.
Why: one entry point; main.py only calls other modules and owns the order and the split.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import (
    clean_data,
    config,
    evaluate,
    features,
    infer,
    load_data,
    train,
    utils,
    validate,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SECTIONS = (
    "paths",
    "problem",
    "split",
    "cleaning",
    "features",
    "recipe",
    "model",
    "validation",
    "business",
    "quality_gates",
    "test_gate",
    "output",
    "logging",
)
# Every setting lives in config.yaml; the paths inside it are resolved at run time.
SETTINGS = config.load_config(config.CONFIG_PATH, required_sections=CONFIG_SECTIONS)


def _path(key: str) -> Path:
    """Resolve SETTINGS["paths"][key] against PROJECT_ROOT at run time (tests patch the root)."""
    return PROJECT_ROOT / SETTINGS["paths"][key]


def _feature_columns() -> list[str]:
    """All model inputs, in group order: numeric, categorical, target-encoded."""
    groups = SETTINGS["features"]
    return groups["numeric"] + groups["categorical"] + groups["target_encoded"]


def configure_logging() -> None:
    """Configure logging once per run: terminal and log file, one format, level from SETTINGS.

    Replaces only the handlers it added before, so a second call never duplicates lines
    and pytest's own capture handler stays in place.
    """
    root = logging.getLogger()
    names = ("pipeline-terminal", "pipeline-file")
    for handler in [old for old in root.handlers if old.get_name() in names]:
        root.removeHandler(handler)
        handler.close()
    log_path = _path("log_file")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(SETTINGS["logging"]["format"])
    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    )
    for name, handler in zip(names, handlers):
        handler.set_name(name)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(SETTINGS["logging"]["level"])


def _validation_arguments(training: bool) -> dict:
    """Arguments for validate_dataframe; each column name and range comes from SETTINGS once."""
    problem, cleaning = SETTINGS["problem"], SETTINGS["cleaning"]
    feature_columns = _feature_columns()
    target = problem["target_column"] if training else None
    contract = [problem["id_column"], problem["time_column"]] + (
        [target] if training else []
    )
    zones = cleaning["columns"]
    return {
        "required_columns": feature_columns,
        "target_column": target,
        "non_negative_columns": SETTINGS["validation"]["non_negative_columns"],
        "valid_zone_range": SETTINGS["validation"]["valid_zone_range"],
        "target_range": (
            [cleaning["min_minutes"], cleaning["max_minutes"]] if training else None
        ),
        "allowed_columns": contract + feature_columns,
        "zone_columns": [zones["pickup_zone"], zones["dropoff_zone"]],
    }


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
    shuffle: bool = True,
) -> tuple:
    """Split into train, validation and test; sizes are shares of the whole dataset.

    Inputs: X, y; test_size and val_size as shares of all rows; random_state; stratify
    (classification); shuffle=False keeps row order, so a time-sorted X gives a time split.
    Outputs: X_train, X_val, X_test, y_train, y_val, y_test.
    Why: split before anything is fitted; production always predicts the future.
    """
    logger.info(
        "Splitting %s rows: test %.0f%%, validation %.0f%%",
        len(X),
        test_size * 100,
        val_size * 100,
    )
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError(
            "test_size and val_size must be in (0, 1) and sum to less than 1"
        )
    if stratify and not shuffle:
        raise ValueError(
            "stratify needs shuffle=True; a time-ordered split cannot be stratified"
        )
    random_state = random_state if shuffle else None  # no randomness without shuffling
    x_rest, x_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y if stratify else None,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest,
        y_rest,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        shuffle=shuffle,
        stratify=y_rest if stratify else None,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def check_quality_gate(metrics: dict[str, float], gate: dict) -> str:
    """Return "met", "not met" or "not set" for one gate; a missing threshold never passes."""
    key, comparison, threshold = (
        gate["metric_key"],
        gate["comparison"],
        gate["threshold"],
    )
    if key not in metrics:
        raise ValueError(f"Gate metric '{key}' not in the metrics: {sorted(metrics)}")
    if threshold is None:
        return "not set"
    value = metrics[key]
    if comparison == ">=":
        return "met" if value >= threshold else "not met"
    if comparison == "<=":
        return "met" if value <= threshold else "not met"
    if comparison == "between":
        low, high = threshold
        return "met" if low <= value <= high else "not met"
    raise ValueError(f"Unknown comparison '{comparison}'; use '>=', '<=' or 'between'")


def check_quality_gates(metrics: dict[str, float], gates: list[dict]) -> dict[str, str]:
    """Check every gate; returns {metric_key: "met" | "not met" | "not set"}."""
    return {gate["metric_key"]: check_quality_gate(metrics, gate) for gate in gates}


def _prepare_training_data(zone_table: pd.DataFrame) -> pd.DataFrame:
    """Load and clean the raw trips (training mode), validate them, save the processed data."""
    raw = load_data.load_raw_data(_path("raw_data"))
    cleaned = clean_data.clean_dataframe(
        raw,
        target_column=SETTINGS["problem"]["target_column"],
        zone_table=zone_table,
        rules=SETTINGS["cleaning"],
    )
    validate.validate_dataframe(cleaned, **_validation_arguments(training=True))
    utils.save_csv(cleaned, _path("processed_data"))
    return cleaned


def _split_by_time(cleaned: pd.DataFrame) -> tuple:
    """Sort trips by pickup time, separate inputs from the target, split before any fitting."""
    problem = SETTINGS["problem"]
    pickup = pd.to_datetime(
        cleaned[problem["time_column"]], format=SETTINGS["cleaning"]["datetime_format"]
    )
    ordered = cleaned.iloc[pickup.argsort(kind="stable").to_numpy()].reset_index(
        drop=True
    )
    return split_data(
        ordered[_feature_columns()],
        ordered[problem["target_column"]],
        stratify=problem["problem_type"] == "classification",
        **SETTINGS["split"],
    )


def _fit_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Build the unfitted recipe and train the model on the training split only."""
    groups, seed = SETTINGS["features"], SETTINGS["split"]["random_state"]
    recipe = features.get_feature_preprocessor(
        groups["numeric"],
        groups["categorical"],
        groups["target_encoded"],
        random_state=seed,
        target_encoder_folds=SETTINGS["recipe"]["target_encoder_folds"],
    )
    return train.train_model(
        x_train,
        y_train,
        recipe,
        SETTINGS["problem"]["problem_type"],
        SETTINGS["model"],
        seed,
    )


def _report_gates(model, splits: tuple) -> None:
    """Gates on validation (met / NOT met, never hidden), then the test audit, reported only."""
    _, x_val, x_test, _, y_val, y_test = splits
    problem_type, business = SETTINGS["problem"]["problem_type"], SETTINGS["business"]
    val_metrics = evaluate.evaluate_model(model, x_val, y_val, problem_type, **business)
    status = check_quality_gates(val_metrics, SETTINGS["quality_gates"])
    for key, result in status.items():
        level = logging.INFO if result == "met" else logging.WARNING
        logger.log(level, "Gate %s: %.3f -> %s", key, val_metrics[key], result.upper())
    all_met = all(result == "met" for result in status.values())
    logger.log(
        logging.INFO if all_met else logging.WARNING,
        "Validation gates: %s",
        "all met" if all_met else "NOT met: not for release",
    )
    _report_test_audit(model, x_test, y_test, val_metrics["mae"])


def _report_test_audit(
    model, x_test: pd.DataFrame, y_test: pd.Series, val_mae: float
) -> None:
    """Test audit, reported only: test mae must stay within the agreed ratio of validation mae."""
    problem_type, business = SETTINGS["problem"]["problem_type"], SETTINGS["business"]
    test_metrics = evaluate.evaluate_model(
        model, x_test, y_test, problem_type, **business
    )
    ratio = test_metrics["mae"] / val_mae
    limit = SETTINGS["test_gate"]["max_test_to_val_mae"]
    met = ratio <= limit
    logger.log(
        logging.INFO if met else logging.WARNING,
        "Test audit (reported only): test/validation mae %.2f, limit %s: %s",
        ratio,
        limit,
        "met" if met else "NOT met",
    )


def _quote_bookings(model, zone_table: pd.DataFrame) -> pd.DataFrame:
    """Clean, validate and quote the new bookings; save them with the trip ID first."""
    problem, output = SETTINGS["problem"], SETTINGS["output"]
    bookings = load_data.load_raw_data(_path("inference_data"))
    cleaned = clean_data.clean_dataframe(
        bookings, zone_table=zone_table, rules=SETTINGS["cleaning"]
    )
    validate.validate_dataframe(cleaned, **_validation_arguments(training=False))
    regression = problem["problem_type"] == "regression"
    floor = SETTINGS["cleaning"]["min_minutes"] if regression else None
    scored = infer.run_inference(
        model, cleaned[_feature_columns()], min_prediction=floor
    )
    quotes = cleaned[[problem["id_column"]] + output["context_columns"]].copy()
    quotes[output["prediction_column"]] = scored["prediction"].round(output["decimals"])
    utils.save_csv(quotes, _path("predictions"))
    return quotes


def run() -> None:
    """Run every step in order; main.py only calls other modules and owns the order."""
    logger.info("RideQuote pipeline: start")
    zone_table = utils.load_csv(_path("zone_table"))
    cleaned = _prepare_training_data(zone_table)
    splits = _split_by_time(cleaned)
    model = _fit_model(splits[0], splits[3])
    _report_gates(model, splits)
    utils.save_model(model, _path("model"))
    _quote_bookings(model, zone_table)
    for key in ("processed_data", "model", "predictions"):
        logger.info("Saved %s: %s", key, _path(key))


def main() -> None:
    """Entry point: configure logging and run; bad data or config logs an error and exits 1."""
    configure_logging()
    logger.info("Using config %s", config.CONFIG_PATH)
    try:
        run()
    except (ValueError, OSError, KeyError) as exc:
        logger.exception("Pipeline stopped: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
