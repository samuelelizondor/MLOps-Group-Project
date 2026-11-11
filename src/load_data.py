"""Load step: read the raw trips CSV into a DataFrame, exactly as it is on disk.

Inputs: path to the raw file (SETTINGS["paths"]["raw_data"] or ["inference_data"], via main.py).
Outputs: the raw DataFrame, untouched.
Used by: main.py; the result goes to clean_data.py.
Why: one read-only entry point that fails fast on a missing, folder or empty file.
"""

import logging
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Read the raw trips CSV; fail fast if it is missing, a folder or empty.

    Inputs: raw_data_path, path to the raw CSV file.
    Outputs: the raw DataFrame, exactly as stored on disk.
    Why: a bad path or an empty export stops the run here, with a message saying what to fix.
    """
    raw_data_path = Path(raw_data_path)
    logger.info("Loading raw data: %s", raw_data_path)
    if raw_data_path.is_dir():
        raise IsADirectoryError(
            f"Expected a CSV file, got a folder: {raw_data_path}. Check SETTINGS['paths']"
        )
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_data_path}. Check SETTINGS['paths']"
        )
    if raw_data_path.stat().st_size == 0:
        raise ValueError(
            f"Raw data file is empty: {raw_data_path}. Re-export the source data"
        )
    df = utils.load_csv(raw_data_path)
    if df.empty:
        raise ValueError(
            f"Raw data has a header but no rows: {raw_data_path}. Re-export the source data"
        )
    return df
