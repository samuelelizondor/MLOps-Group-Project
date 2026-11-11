"""Generic file helpers: read and write CSV files and the fitted model.

Inputs: file paths (from SETTINGS["paths"] via main.py); DataFrames or fitted models to save.
Outputs: DataFrames read from CSV, models loaded with joblib, files written to disk.
Used by: load_data.py, main.py and the inference tests.
Why: one copy of each read/write rule (create parent folders, no index, fail fast on missing files).
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Inputs: filepath, path to an existing CSV file.
    Outputs: the file's contents as a DataFrame.
    Why: fail fast with a clear message instead of a pandas error deep in the pipeline.
    """
    filepath = Path(filepath)
    logger.info("Loading CSV: %s", filepath)
    if not filepath.is_file():
        raise FileNotFoundError(
            f"CSV not found: {filepath}. Check the path in SETTINGS['paths']"
        )
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Write a DataFrame to CSV without the index, creating parent folders.

    Inputs: df, the table to save; filepath, where to write it.
    Outputs: none; the CSV file is written to disk.
    Why: index=False keeps the columns unchanged, so a reloaded file passes the data-contract check.
    """
    filepath = Path(filepath)
    logger.info("Saving CSV: %s (%s rows)", filepath, len(df))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """Save a fitted model with joblib, creating parent folders.

    Inputs: model, a fitted estimator or Pipeline; filepath, where to write it.
    Outputs: none; the model file is written to disk.
    Why: the saved Pipeline carries its fitted preprocessing, so inference reuses that recipe.
    """
    filepath = Path(filepath)
    logger.info("Saving model: %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """Load a model saved with save_model.

    Inputs: filepath, path to a joblib file written by save_model.
    Outputs: the fitted model, ready to predict.
    Why: fail fast with a clear message when the model was never trained or the path is wrong.
    """
    filepath = Path(filepath)
    logger.info("Loading model: %s", filepath)
    if not filepath.is_file():
        raise FileNotFoundError(
            f"Model not found: {filepath}. Train it first with python -m src.main"
        )
    return joblib.load(filepath)
