"""Tests for src/config.py: load_config reads config.yaml and fails fast with clear messages.

Unit tests check a valid file loads, and that a missing file, a non-mapping file or missing
sections raise; the integration test loads the repo's config.yaml with every section main.py needs.
"""

import pytest

from src.config import load_config


def _write_config(tmp_path, text: str):
    """Write YAML text to tmp_path/config.yaml and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unit_valid_config_loads_as_plain_dict(tmp_path):
    """A valid file loads as a dict, with YAML types kept: numbers, lists, nested sections."""
    path = _write_config(
        tmp_path,
        "paths:\n  raw_data: data/raw/trips.csv\n"
        "split:\n  test_size: 0.15\n  random_state: 42\n"
        "quality_gates:\n"
        "  - {metric_key: bias, comparison: between, threshold: [-1.0, 2.0]}\n",
    )

    settings = load_config(path, required_sections=("paths", "split", "quality_gates"))

    assert settings["paths"]["raw_data"] == "data/raw/trips.csv"
    assert settings["split"] == {"test_size": 0.15, "random_state": 42}
    assert settings["quality_gates"][0]["threshold"] == [-1.0, 2.0]


def test_unit_bad_config_files_raise(tmp_path):
    """Missing file, empty or non-mapping file, or missing sections stop with clear errors."""
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write_config(tmp_path, ""))
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write_config(tmp_path, "- just\n- a list\n"))
    path = _write_config(tmp_path, "paths:\n  raw_data: x.csv\n")
    with pytest.raises(ValueError, match=r"config\.yaml.*problem.*split"):
        load_config(path, required_sections=("paths", "problem", "split"))


def test_integration_repo_config_loads_with_every_section_main_needs():
    """The repo's config.yaml loads with main.py's sections; quoted YAML values arrive intact."""
    from src import config
    from src import main as pipeline

    settings = load_config(
        config.CONFIG_PATH, required_sections=pipeline.CONFIG_SECTIONS
    )

    comparisons = {gate["comparison"] for gate in settings["quality_gates"]}
    assert comparisons == {">=", "<=", "between"}
    assert settings["cleaning"]["datetime_format"] == "%Y-%m-%d %H:%M:%S"
    assert settings["logging"]["format"].startswith("%(asctime)s | %(levelname)s")
    assert settings == pipeline.SETTINGS
