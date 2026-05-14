"""YAML config validator — catches mistakes before the experiment run starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VALID_LEVELS = set(range(1, 8))
VALID_SAMPLES = {"A", "B", "C"}
REQUIRED_FIELDS = {"data_plugin", "mesh_path", "levels", "samples", "methods", "dataset_root", "output_dir"}


class ConfigError(Exception):
    pass


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate experiment.yaml. Raises ConfigError on any problem."""
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if path.suffix not in (".yaml", ".yml"):
        raise ConfigError(f"Config file must be .yaml or .yml, got: {path.suffix}")

    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Could not parse YAML: {e}") from e

    if not isinstance(config, dict):
        raise ConfigError("Config file must be a YAML mapping (key: value pairs).")

    _check_required_fields(config)
    _check_levels(config)
    _check_samples(config)
    _check_methods(config)
    _check_mesh_path(config)

    return config


def _check_required_fields(config: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(config.keys())
    if missing:
        raise ConfigError(f"Missing required fields: {sorted(missing)}")


def _check_levels(config: dict[str, Any]) -> None:
    levels = config.get("levels", [])
    if not isinstance(levels, list) or len(levels) == 0:
        raise ConfigError("'levels' must be a non-empty list.")
    invalid = [l for l in levels if l not in VALID_LEVELS]
    if invalid:
        raise ConfigError(f"Invalid levels {invalid}. Must be integers between 1 and 7.")


def _check_samples(config: dict[str, Any]) -> None:
    samples = config.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        raise ConfigError("'samples' must be a non-empty list.")
    invalid = [s for s in samples if s not in VALID_SAMPLES]
    if invalid:
        raise ConfigError(f"Invalid samples {invalid}. Must be one of: A, B, C.")


def _check_methods(config: dict[str, Any]) -> None:
    methods = config.get("methods", [])
    if not isinstance(methods, list) or len(methods) == 0:
        raise ConfigError("'methods' must be a non-empty list.")
    for m in methods:
        if not isinstance(m, str) or not m.strip():
            raise ConfigError(f"Each method must be a non-empty string, got: {m!r}")


def _check_mesh_path(config: dict[str, Any]) -> None:
    mesh_path = config.get("mesh_path", "")
    if not mesh_path or not isinstance(mesh_path, str):
        raise ConfigError("'mesh_path' must be a non-empty string path.")
