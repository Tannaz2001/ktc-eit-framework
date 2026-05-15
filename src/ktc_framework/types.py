from typing import NamedTuple
import numpy as np


class DataBatch(NamedTuple):
    """A single EIT measurement sample."""

    voltages: np.ndarray
    """Flat voltage vector, shape (2356,) for KTC dataset."""

    injection_patterns: np.ndarray
    """Adjacent-pair injection matrix, shape (32, 76)."""

    ground_truth: np.ndarray
    """Segmentation labels {0, 1, 2}, shape (256, 256)."""

    level: int
    """Difficulty level (1–7 in KTC dataset)."""

    sample_id: str
    """Unique identifier for this sample."""
