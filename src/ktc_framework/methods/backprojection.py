"""Back-projection reconstruction for the KTC 2023 EIT framework.

Method
------
Linear back-projection on top of KTC's own forward operator:

    deltareco = J^T * (Uel - Uelref)

where J is the Jacobian computed by KTC's EITFEM (data/KTCScoring/KTCFwd.py)
at homogeneous sigma=1 S/m, contact impedance 1e-6.  Difference imaging
requires an empty-tank reference: we use the MEASURED Uelref from ref.mat
(carried on ``batch.reference_voltages``), NOT an FEM-simulated reference.
The simulated reference disagrees with Uelref by ~22% RMS even on the
correct mesh -- using the measured one removes the bias.

The previous implementation used pyEIT's BP solver on a synthetic
386-node unit-circle mesh; it could not reproduce KTC's CEM forward
operator (quadratic triangles, contact impedance, vincl-masked
measurements) and the back-projection landed the inclusion in the wrong
place with a 35% systematic offset.  See trace_phase1.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
import warnings

import numpy as np

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.eit_utils import (
    N_MEAS_TOTAL,
    adaptive_segment,
    build_ktc_jacobian,
    build_vincl,
    load_ktc_mesh,
    rasterize,
)
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch


_DEFAULT_MESH = "Codes_Matlab/Mesh_sparse.mat"
_DEFAULT_MPAT_PATH = "Codes_Matlab/TrainingData/ref.mat"


# Module-level caches shared across plugin instantiations.  The runner
# constructs a fresh BackProjection per (method, level, sample), so an
# instance cache rebuilds the Jacobian every sample (~25 s wasted each).
_MESH_CACHE: dict[str, dict] = {}
_JAC_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _inj_key(injection_patterns: np.ndarray) -> str:
    arr = np.ascontiguousarray(injection_patterns, dtype=np.float64)
    return hashlib.md5(arr.tobytes()).hexdigest()


def _load_mesh_cached(mesh_path: str) -> dict:
    if mesh_path not in _MESH_CACHE:
        _MESH_CACHE[mesh_path] = load_ktc_mesh(mesh_path)
    return _MESH_CACHE[mesh_path]


def _default_mpat() -> np.ndarray:
    """Fallback Mpat: 31 adjacent differential pairs (col j: +1 at j, -1 at j+1)."""
    mpat = np.zeros((32, 31), dtype=np.float64)
    for j in range(31):
        mpat[j, j] = 1.0
        mpat[j + 1, j] = -1.0
    return mpat


@register
class BackProjection(MethodPlugin):
    """KTC linear back-projection: deltareco = J^T (Uel - Uelref)."""

    def __init__(self, mesh_path: str = _DEFAULT_MESH) -> None:
        self._mesh_path = mesh_path
        self._mesh: Optional[dict] = None
        try:
            self._mesh = _load_mesh_cached(mesh_path)
        except Exception:
            self._mesh = None

    def _ensure_mesh(self) -> dict:
        if self._mesh is None:
            self._mesh = _load_mesh_cached(self._mesh_path)
        return self._mesh

    def _get_jacobian(
        self,
        level: int,
        injection_patterns: np.ndarray,
        measurement_patterns: np.ndarray,
        reference_voltages: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute J + vincl for this level, or return the cached pair."""
        cache_key = (self._mesh_path, int(level), _inj_key(injection_patterns))
        cached = _JAC_CACHE.get(cache_key)
        if cached is not None:
            return cached

        mesh = self._ensure_mesh()
        vincl = build_vincl(level, injection_patterns)
        J, _inv_gamma, _solver, _Usim = build_ktc_jacobian(
            mesh,
            injection_patterns,
            measurement_patterns,
            vincl,
            reference_voltages=reference_voltages,
        )
        entry = (J, vincl)
        _JAC_CACHE[cache_key] = entry
        return entry

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        try:
            mesh = self._ensure_mesh()
        except Exception as exc:
            warnings.warn(
                f"BackProjection: mesh unavailable ({exc}); returning zeros.",
                RuntimeWarning, stacklevel=2,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        # Measurement pattern: carried on the batch when the loader could
        # find ref.mat; otherwise fall back to the canonical KTC pattern.
        mpat = (
            np.asarray(batch.measurement_patterns, dtype=np.float64)
            if getattr(batch, "measurement_patterns", None) is not None
            else _default_mpat()
        )

        # Reference voltage: measured Uelref from ref.mat.  If the loader
        # never populated it, fall back to subtracting the mean -- this
        # removes the DC offset but is strictly worse than a real Uelref.
        v_ref = batch.reference_voltages
        if v_ref is None:
            v_ref = np.full(N_MEAS_TOTAL, float(np.asarray(batch.voltages).mean()))
        v_ref = np.asarray(v_ref, dtype=np.float64).ravel()

        J, vincl = self._get_jacobian(
            batch.level, batch.injection_patterns, mpat, v_ref
        )

        v1 = np.asarray(batch.voltages, dtype=np.float64).ravel()
        # dv on the full 2356 grid, then restricted to the valid measurements
        # for this level (matches J's row count by construction).
        dv_full = v1 - v_ref
        dv = dv_full[vincl]

        # Pure back-projection: deltareco_node = J^T @ dv.
        deltareco = J.T @ dv

        grid = rasterize(deltareco, mesh)
        labels = adaptive_segment(grid)
        self.validate_output(labels)
        return labels
