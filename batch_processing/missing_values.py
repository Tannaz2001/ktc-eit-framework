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


def fill_missing_values_gp(
    array: np.ndarray,
    *,
    max_missing_ratio: float = 0.35,
    length_scale: float = 3.0,
    noise: float = 1e-6,
) -> FillResult:
    """
    Fill missing values with a Gaussian Process-style interpolation.

    The method uses an RBF kernel over flattened measurement indices and predicts
    missing entries from observed entries. If too much data is missing, the sample
    is marked as unreliable and returned unchanged.
    """
    values = np.array(array, dtype=float, copy=True)
    flat = values.reshape(-1)
    nan_mask = np.isnan(flat)
    missing_count = int(np.sum(nan_mask))
    total_count = int(flat.size)
    missing_ratio = (missing_count / total_count) if total_count else 0.0

    if missing_count == 0:
        return FillResult(
            values=values,
            estimated_count=0,
            missing_ratio=0.0,
            is_reliable=True,
            strategy_used="none",
        )

    if missing_ratio > max_missing_ratio:
        return FillResult(
            values=values,
            estimated_count=0,
            missing_ratio=missing_ratio,
            is_reliable=False,
            strategy_used="gp",
            reason=f"too_many_missing_values>{max_missing_ratio:.2f}",
        )

    observed_idx = np.where(~nan_mask)[0].astype(float)
    missing_idx = np.where(nan_mask)[0].astype(float)
    observed_values = flat[~nan_mask]

    if observed_idx.size == 0:
        return FillResult(
            values=np.zeros_like(values),
            estimated_count=missing_count,
            missing_ratio=missing_ratio,
            is_reliable=False,
            strategy_used="gp",
            reason="all_values_missing",
        )

    def rbf_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        distances = (x1[:, None] - x2[None, :]) ** 2
        return np.exp(-0.5 * distances / (length_scale**2))

    k_xx = rbf_kernel(observed_idx, observed_idx) + noise * np.eye(observed_idx.size)
    k_mx = rbf_kernel(missing_idx, observed_idx)

    alpha = np.linalg.solve(k_xx, observed_values)
    predicted = k_mx @ alpha
    flat[nan_mask] = predicted

    return FillResult(
        values=flat.reshape(values.shape),
        estimated_count=missing_count,
        missing_ratio=missing_ratio,
        is_reliable=True,
        strategy_used="gp",
    )


def fill_missing_values_with_fallback(
    array: np.ndarray,
    *,
    max_missing_ratio: float = 0.60,
    fallback_strategy: FillStrategy = "median",
) -> FillResult:
    """
    Fill missing values robustly for batch processing.

    Policy:
    - Try GP first when missing ratio is acceptable.
    - If GP is marked unreliable, apply deterministic fallback imputation
      instead of skipping the sample.
    """
    gp_result = fill_missing_values_gp(array, max_missing_ratio=max_missing_ratio)
    if gp_result.is_reliable:
        return gp_result

    fallback_values = fill_missing_values(array, strategy=fallback_strategy)
    nan_mask = np.isnan(np.asarray(array, dtype=float))
    missing_count = int(np.sum(nan_mask))
    total_count = int(np.asarray(array).size)
    missing_ratio = (missing_count / total_count) if total_count else 0.0

    return FillResult(
        values=fallback_values,
        estimated_count=missing_count,
        missing_ratio=missing_ratio,
        is_reliable=True,
        strategy_used=f"fallback_{fallback_strategy}",
        reason=f"gp_unreliable:{gp_result.reason}",
    )
