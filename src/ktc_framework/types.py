"""Shared data contract for the KTC EIT benchmarking framework.

Every data loader (KTCDataPlugin, MockDataPlugin, TrainingDataPlugin) returns
a DataBatch, and every reconstruction method (BackProjection, GaussNewton)
consumes one. Keeping a single immutable type here enforces a consistent
interface across all modules without any coupling between them.
"""

from typing import Any, NamedTuple

import numpy as np


class DataBatch(NamedTuple):
    """Immutable container for one EIT measurement sample."""

    voltages: np.ndarray           # shape (N,)      float32 — boundary voltage measurements
    injection_patterns: np.ndarray # shape (32, 76)  float32 — adjacent-pair injection matrix
    ground_truth: np.ndarray       # shape (256, 256) uint8  — 0=water, 1=resistive, 2=conductive
    level: int                     # int 1–7                 — difficulty level
    sample_id: str                 # str e.g. 'level1_A'     — unique sample identifier
    mesh: Any = None               # dict from Mesh_sparse.mat via scipy.io.loadmat, or None
    reference_voltages: Any = None # shape (N,) float32      — empty-tank voltages from ref.mat
    measurement_patterns: Any = None # shape (31, 32) float32 — measurement-pattern matrix from Mpat in ref.mat