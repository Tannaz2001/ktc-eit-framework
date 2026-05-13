"""Validation helpers for missing or incomplete dataset samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class FillResult:
    values: np.ndarray
    estimated_count: int
    missing_ratio: float
    is_reliable: bool
    strategy_used: str
    reason: Optional[str] = None


def validate_sample_file(path: Path | str, min_size_bytes: int = 16) -> ValidationResult:
    """
    Basic safety checks before processing a sample.

    A file is considered incomplete when it exists but is very small.
    """
    sample_path = Path(path)

    if not sample_path.exists():
        return ValidationResult(False, "missing_file")
    if not sample_path.is_file():
        return ValidationResult(False, "not_a_file")

    size = sample_path.stat().st_size
    if size < min_size_bytes:
        return ValidationResult(False, f"incomplete_file_size<{min_size_bytes}")

    return ValidationResult(True, None)


FillStrategy = Literal["zero", "mean", "median", "forward_fill"]


def fill_missing_values(array: np.ndarray, strategy: FillStrategy = "mean") -> np.ndarray:
    """
    Fill NaN values in an array without modifying the input in-place.

    Strategies:
    - zero: replace NaN with 0
    - mean: replace NaN with global array mean (ignoring NaN)
    - median: replace NaN with global array median (ignoring NaN)
    - forward_fill: 1D forward fill, then backward fill for leading NaNs
    """
    valid_strategies = {"zero", "mean", "median", "forward_fill"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown fill strategy: {strategy}")

    values = np.array(array, dtype=float, copy=True)
    nan_mask = np.isnan(values)
    if not np.any(nan_mask):
        return values

    if strategy == "zero":
        values[nan_mask] = 0.0
        return values

    if strategy == "mean":
        replacement = 0.0 if np.all(nan_mask) else float(np.nanmean(values))
        values[nan_mask] = replacement
        return values

    if strategy == "median":
        replacement = 0.0 if np.all(nan_mask) else float(np.nanmedian(values))
        values[nan_mask] = replacement
        return values

    if strategy == "forward_fill":
        flat = values.reshape(-1)
        for idx in range(1, flat.size):
            if np.isnan(flat[idx]):
                flat[idx] = flat[idx - 1]
        for idx in range(flat.size - 2, -1, -1):
            if np.isnan(flat[idx]):
                flat[idx] = flat[idx + 1]
        flat[np.isnan(flat)] = 0.0
        return flat.reshape(values.shape)

    raise ValueError(f"Unknown fill strategy: {strategy}")


