# src/ktc_framework/utils/mock_mesh.py
"""Utility for creating a dummy DataBatch with a real pyEIT mesh.

Used in tests and CLI sanity checks where real .mat files are not available.
The duplicate DataBatch namedtuple that used to live here has been removed —
the canonical DataBatch (with mesh field) now lives in src/ktc_framework/types.py.
"""

from __future__ import annotations

import numpy as np

from src.ktc_framework.types import DataBatch

_N_ELECTRODES = 32
_N_INJ_COLS   = 76
_N_VOLTAGES   = 2356   # 76 injections × 31 pairs


def make_adjacent_protocol(n_electrodes: int = 32, n_patterns: int = _N_INJ_COLS) -> np.ndarray:
    """Build an adjacent-pair injection protocol matrix (n_electrodes × n_patterns)."""
    patterns = np.zeros((n_electrodes, n_patterns))
    for j in range(n_patterns):
        patterns[j % n_electrodes, j] =  1.0
        patterns[(j + 1) % n_electrodes, j] = -1.0
    return patterns


def make_dummy_batch_with_mesh() -> DataBatch:
    """Create a dummy DataBatch with a valid pyEIT mesh attached.

    Uses pyeit.mesh.create() so BackProjection and GaussNewton
    can run their real algorithms instead of the random fallback.
    Returns a DataBatch with mesh=None if pyeit is not installed.
    """
    voltages           = np.random.randn(_N_VOLTAGES).astype(np.float64)
    injection_patterns = make_adjacent_protocol(_N_ELECTRODES)
    ground_truth       = np.zeros((256, 256), dtype=int)

    mesh_obj = None
    try:
        from pyeit.mesh import create as mesh_create  # type: ignore[import]
        mesh_obj = mesh_create(n_el=_N_ELECTRODES, h0=0.1)
    except ImportError:
        pass  # pyeit not installed — mesh stays None, methods use fallback

    return DataBatch(
        voltages=voltages,
        injection_patterns=injection_patterns,
        ground_truth=ground_truth,
        level=1,
        sample_id="dummy",
        mesh=mesh_obj,
    )
