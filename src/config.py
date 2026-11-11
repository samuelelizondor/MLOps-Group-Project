"""Config step: read config.yaml, the single source of truth for every non-secret setting.

Inputs: the path to config.yaml and the sections the caller needs (from main.py).
Outputs: the settings as a plain dict, with YAML types kept (numbers, lists, nested sections).
Used by: main.py at import (SETTINGS); later the API reads the same file.
Why: settings change without code edits, and a broken file stops the run with a clear message.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# The one place that knows where config.yaml lives (repo root); main.py and clean_data.py use it.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(config_path: Path, required_sections: tuple[str, ...] = ()) -> dict:
    """Read config.yaml with yaml.safe_load; stop the run if the file or a section is missing.

    Inputs: config_path, the YAML file; required_sections, top-level sections the caller reads.
    Outputs: the settings as a plain dict.
    Why: a missing or broken config fails here, naming the file and what to fix.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    if not isinstance(settings, dict):
        raise ValueError(
            f"{config_path} must be a mapping of sections (key: value), not empty or a list"
        )
    missing = [section for section in required_sections if section not in settings]
    if missing:
        raise ValueError(f"{config_path} is missing sections: {missing}")
    logger.info("Loaded config from %s", config_path)
    return settings
