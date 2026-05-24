"""YAML config validator — catches mistakes before the experiment run starts.

Usage
-----
    from src.ktc_framework.runner.config_validator import load_config, ConfigError

    try:
        config = load_config("configs/experiment.yaml")
    except ConfigError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: list[str] = [
    "data_plugin",
    "mesh_path",
    "levels",
    "samples",
    "methods",
    "dataset_root",
    "output_dir",
]

VALID_LEVELS:  set[int] = set(range(1, 8))          # 1–7 inclusive
VALID_SAMPLES: set[str] = {"A", "B", "C", "1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when a YAML config fails validation.

    Every instance carries a human-readable message that names the exact
    field and value that caused the failure.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate an experiment YAML config file.

    Validation steps (in order):
    1.  File extension must be ``.yaml`` or ``.yml``.
    2.  File must be parseable by ``yaml.safe_load``.
    3.  All required fields must be present (checked one at a time).
    4.  ``levels``      — non-empty list of integers in 1–7.
    5.  ``samples``     — non-empty list, each in ``{A, B, C, 1, 2, 3, 4}``.
    6.  ``methods``     — non-empty list of non-empty strings.
    7.  ``mesh_path``   — non-empty string; path must exist (file or directory).
    8.  ``KTC_DATASET_ROOT`` env var overrides ``dataset_root`` if set.
    9.  ``dataset_root``— directory must exist (checked after env-var override).
    10. ``ref.mat``     — soft check: warn if absent, do not raise.
    11. ``output_dir``  — created with ``os.makedirs`` if it does not exist.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file (absolute or relative).

    Returns
    -------
    dict[str, Any]
        Fully validated configuration dict, ready for use by ``BatchRunner``.

    Raises
    ------
    ConfigError
        On any validation failure.  The message names the specific field and
        the offending value so the user knows exactly what to fix.
    """
    # ── 1. Extension check ─────────────────────────────────────────────────
    _, ext = os.path.splitext(str(config_path))
    if ext.lower() not in (".yaml", ".yml"):
        raise ConfigError(
            f"Config error: file must have a .yaml or .yml extension, "
            f"got '{ext}' (path: {config_path})"
        )

    if not os.path.isfile(str(config_path)):
        raise ConfigError(f"Config error: file not found: {config_path}")

    # ── 2. Parse YAML ──────────────────────────────────────────────────────
    with open(str(config_path), "r", encoding="utf-8") as fh:
        try:
            config: Any = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Config error: could not parse YAML in '{config_path}': {exc}"
            ) from exc

    if not isinstance(config, dict):
        raise ConfigError(
            f"Config error: top-level YAML must be a mapping (got {type(config).__name__})"
        )

    # ── 3. Required fields — one at a time for specific error messages ─────
    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ConfigError(
                f"Config error: required field '{field}' is missing from {config_path}"
            )

    # ── 4. Validate levels ─────────────────────────────────────────────────
    levels = config["levels"]
    if not isinstance(levels, list) or len(levels) == 0:
        raise ConfigError(
            f"Config error: 'levels' must be a non-empty list, got {levels!r}"
        )
    bad_levels = [lv for lv in levels if not isinstance(lv, int) or lv not in VALID_LEVELS]
    if bad_levels:
        raise ConfigError(
            f"Config error: levels must be integers between 1 and 7, "
            f"got {bad_levels!r} in {levels!r}"
        )

    # ── 5. Validate samples ────────────────────────────────────────────────
    samples = config["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        raise ConfigError(
            f"Config error: 'samples' must be a non-empty list, got {samples!r}"
        )
    bad_samples = [s for s in samples if str(s) not in VALID_SAMPLES]
    if bad_samples:
        raise ConfigError(
            f"Config error: samples must be in {sorted(VALID_SAMPLES)}, "
            f"got {bad_samples!r} in {samples!r}"
        )

    # ── 6. Validate methods ────────────────────────────────────────────────
    methods = config["methods"]
    if not isinstance(methods, list) or len(methods) == 0:
        raise ConfigError(
            f"Config error: 'methods' must be a non-empty list, got {methods!r}"
        )
    for m in methods:
        if not isinstance(m, str) or not m.strip():
            raise ConfigError(
                f"Config error: each method must be a non-empty string, "
                f"got {m!r} in methods list {methods!r}"
            )

    # ── 7. Validate mesh_path ──────────────────────────────────────────────
    mesh_path: str = config["mesh_path"]
    if not isinstance(mesh_path, str) or not mesh_path.strip():
        raise ConfigError(
            f"Config error: 'mesh_path' must be a non-empty string, got {mesh_path!r}"
        )
    # Accept a direct .mat file OR a directory that contains Mesh_sparse.mat
    if not os.path.isfile(mesh_path) and not os.path.isdir(mesh_path):
        raise ConfigError(
            f"Config error: mesh_path does not exist: '{mesh_path}'. "
            f"Expected either a .mat file or a directory containing Mesh_sparse.mat."
        )

    # ── 8. Environment-variable override for dataset_root ─────────────────
    env_root = os.environ.get("KTC_DATASET_ROOT")
    if env_root:
        config["dataset_root"] = env_root

    # ── 9. Validate dataset_root ───────────────────────────────────────────
    dataset_root: str = config["dataset_root"]
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ConfigError(
            f"Config error: 'dataset_root' must be a non-empty string, "
            f"got {dataset_root!r}"
        )
    if not os.path.isdir(dataset_root):
        raise ConfigError(
            f"Config error: dataset_root directory does not exist: '{dataset_root}'. "
            f"Set KTC_DATASET_ROOT env var or fix 'dataset_root' in the config."
        )

    # ── 10. Soft check for ref.mat ─────────────────────────────────────────
    ref_mat = os.path.join(dataset_root, "ref.mat")
    if not os.path.isfile(ref_mat):
        warnings.warn(
            f"ref.mat not found at '{ref_mat}'. "
            f"BackProjection and GaussNewton will fall back to mean-subtraction "
            f"for difference imaging — reconstruction quality will be reduced.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── 11. Create output_dir if needed ────────────────────────────────────
    output_dir: str = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    return config
