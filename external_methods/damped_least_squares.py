"""External runtime method: damped least-squares EIT reconstruction.

This file intentionally lives outside ``src/ktc_framework/methods`` so it
tests the runtime method adapter path. It is a real inverse reconstruction
method, separate from BackProjection and GaussNewton.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.sparse.linalg import lsqr

from src.ktc_framework.methods.eit_utils import (
    N_MEAS_TOTAL,
    adaptive_segment,
    build_ktc_jacobian,
    build_vincl,
    load_ktc_mesh,
    rasterize,
)
from src.ktc_framework.registry import register_method
from src.ktc_framework.types import DataBatch


_DEFAULT_MESH = "Codes_Matlab/Mesh_sparse.mat"
_MESH_CACHE: dict[str, dict] = {}
_JACOBIAN_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _default_mpat() -> np.ndarray:
    """Fallback measurement pattern: 31 adjacent differential pairs."""
    mpat = np.zeros((32, 31), dtype=np.float64)
    for j in range(31):
        mpat[j, j] = 1.0
        mpat[j + 1, j] = -1.0
    return mpat


def _array_key(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.md5(arr.tobytes()).hexdigest()


def _get_mesh(mesh_path: str) -> dict:
    if mesh_path not in _MESH_CACHE:
        _MESH_CACHE[mesh_path] = load_ktc_mesh(mesh_path)
    return _MESH_CACHE[mesh_path]


@register_method
class DampedLeastSquaresReconstruction:
    """One-step damped least-squares reconstruction using KTC's Jacobian."""

    def __init__(
        self,
        mesh_path: str = _DEFAULT_MESH,
        damping: float = 0.15,
        max_iterations: int = 80,
    ) -> None:
        self.mesh_path = mesh_path
        self.damping = damping
        self.max_iterations = max_iterations

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        mesh = _get_mesh(self.mesh_path)
        injection_patterns = np.asarray(batch.injection_patterns, dtype=np.float64)
        measurement_patterns = (
            np.asarray(batch.measurement_patterns, dtype=np.float64)
            if batch.measurement_patterns is not None
            else _default_mpat()
        )

        voltages = np.asarray(batch.voltages, dtype=np.float64).ravel()
        reference = batch.reference_voltages
        if reference is None or np.asarray(reference).size != N_MEAS_TOTAL:
            reference = np.full(N_MEAS_TOTAL, float(voltages.mean()))
        reference = np.asarray(reference, dtype=np.float64).ravel()

        vincl = build_vincl(batch.level, injection_patterns)
        cache_key = (
            self.mesh_path,
            int(batch.level),
            _array_key(injection_patterns),
            _array_key(measurement_patterns),
        )
        if cache_key not in _JACOBIAN_CACHE:
            J, _inv_gamma_n, _solver, _usim = build_ktc_jacobian(
                mesh,
                injection_patterns,
                measurement_patterns,
                vincl,
                reference_voltages=reference,
            )
            _JACOBIAN_CACHE[cache_key] = (J, vincl)

        J, vincl = _JACOBIAN_CACHE[cache_key]
        dv = voltages - reference
        if J.shape[0] != dv.size:
            dv = dv[vincl]

        # Solve min ||J*x - dv||_2^2 + damping^2 ||x||_2^2.
        result = lsqr(J, dv, damp=self.damping, iter_lim=self.max_iterations)
        delta_sigma = result[0]

        grid = rasterize(delta_sigma, mesh)
        return adaptive_segment(grid)
