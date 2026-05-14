"""Tests for YAML config validator."""

import pytest
import yaml
from pathlib import Path

from src.ktc_framework.runner.config_validator import load_config, ConfigError


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "experiment.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


VALID_CONFIG = {
    "data_plugin": "KTCLoader",
    "mesh_path": "data/Mesh/",
    "levels": [1, 2, 7],
    "samples": ["A", "B", "C"],
    "methods": ["simple_baseline"],
    "dataset_root": "data/EvaluationData_full/",
    "output_dir": "outputs/",
}


def test_valid_config_loads(tmp_path):
    p = write_yaml(tmp_path, VALID_CONFIG)
    config = load_config(p)
    assert config["data_plugin"] == "KTCLoader"
    assert config["levels"] == [1, 2, 7]


def test_missing_required_field(tmp_path):
    bad = {k: v for k, v in VALID_CONFIG.items() if k != "mesh_path"}
    p = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError, match="mesh_path"):
        load_config(p)


def test_invalid_level(tmp_path):
    bad = {**VALID_CONFIG, "levels": [1, 8]}
    p = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError, match="Invalid levels"):
        load_config(p)


def test_invalid_sample(tmp_path):
    bad = {**VALID_CONFIG, "samples": ["A", "D"]}
    p = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError, match="Invalid samples"):
        load_config(p)


def test_empty_methods(tmp_path):
    bad = {**VALID_CONFIG, "methods": []}
    p = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError, match="methods"):
        load_config(p)


def test_file_not_found():
    with pytest.raises(ConfigError, match="not found"):
        load_config("nonexistent.yaml")


def test_empty_mesh_path(tmp_path):
    bad = {**VALID_CONFIG, "mesh_path": ""}
    p = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError, match="mesh_path"):
        load_config(p)
