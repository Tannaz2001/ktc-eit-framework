from typing import NamedTuple
import numpy as np


class DataBatch(NamedTuple):
    """A single batch of EIT measurement data."""

    voltages: np.ndarray
    """Measured voltage readings, shape (n_samples, n_electrodes)."""

    injection_patterns: np.ndarray
    """Current injection patterns used during measurement, shape (n_patterns, n_electrodes)."""

    ground_truth: np.ndarray
    """Ground-truth conductivity map or label array, shape (n_samples, *spatial_dims)."""

    level: int
    """Difficulty or resolution level of this batch (e.g. mesh refinement level)."""

    sample_id: str
    """Unique identifier for this batch or sample."""
