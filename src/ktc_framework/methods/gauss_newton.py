"""Gauss-Newton (JAC) reconstruction method using pyEIT.

Algorithm
---------
1. Build a ``PyEITMesh`` from ``batch.mesh`` (Mesh_sparse.mat struct).
2. Create a standard 32-electrode adjacent-pair protocol via pyEIT.
3. Slice ``batch.voltages`` and ``batch.reference_voltages`` to the protocol
   measurement count (``protocol.n_meas_tot``).
   *Fallback*: if ``reference_voltages`` is ``None``, use mean-subtraction
   (``v0 = mean(v1) * ones``).
4. Run ``JAC.setup(p=0.2, lamb=1e-2, method='kotre')`` then
   ``JAC.solve(v1, v0, n_iter=1)`` → element-based conductivity change vector.
5. Interpolate from mesh element centroids to 256×256 float grid via griddata.
6. Segment with double-Otsu → labels ``{0, 1, 2}``.
7. Validate and return ``(256, 256)`` int array.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.methods.segment import segment
from src.ktc_framework.types import DataBatch
from src.ktc_framework.utils.pyeit_utils import build_pyeit_mesh, interpolate_to_grid

# ── optional pyEIT imports ─────────────────────────────────────────────────
try:
    from pyeit.eit.jac import JAC                          # type: ignore[import]
    from pyeit.eit.protocol import create as proto_create  # type: ignore[import]
    _PYEIT_AVAILABLE = True
except ImportError:
    _PYEIT_AVAILABLE = False


def _prepare_voltages(
    v1_raw: np.ndarray,
    v0_raw: Optional[np.ndarray],
    n_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice/pad voltage vectors to *n_total* samples.

    Parameters
    ----------
    v1_raw:
        Measurement voltages, shape ``(2356,)`` from KTC dataset.
    v0_raw:
        Reference (all-water) voltages, or ``None``.
    n_total:
        Number of measurements the pyEIT protocol expects.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(v1, v0)`` both shape ``(n_total,)`` float64.
    """
    v1 = v1_raw.ravel().astype(np.float64)

    if v0_raw is not None:
        v0 = v0_raw.ravel().astype(np.float64)
    else:
        warnings.warn(
            "GaussNewton: reference_voltages is None — "
            "using mean-subtraction fallback (v0 = mean(v1)).",
            RuntimeWarning,
            stacklevel=3,
        )
        v0 = np.full(len(v1), fill_value=float(v1.mean()))

    for name, arr in (("v1", v1), ("v0", v0)):
        if len(arr) < n_total:
            warnings.warn(
                f"GaussNewton: {name} length {len(arr)} < "
                f"protocol n_meas_tot {n_total} — zero-padding.",
                RuntimeWarning,
                stacklevel=3,
            )

    v1 = v1[:n_total] if len(v1) >= n_total else np.pad(v1, (0, n_total - len(v1)))
    v0 = v0[:n_total] if len(v0) >= n_total else np.pad(v0, (0, n_total - len(v0)))
    return v1, v0


@register
class GaussNewton(MethodPlugin):
    """Gauss-Newton (linearised Jacobian) EIT reconstruction via pyEIT.

    Uses Tikhonov regularisation with one Jacobian iteration (``n_iter=1``).
    Falls back to a segmented random-noise image if pyEIT is not installed
    or if no valid mesh is present in the batch.
    """

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Reconstruct a (256, 256) segmentation label array.

        Parameters
        ----------
        batch:
            A ``DataBatch`` with ``voltages``, and optionally
            ``reference_voltages`` and ``mesh`` populated by ``BatchRunner``.

        Returns
        -------
        np.ndarray
            Shape ``(256, 256)``, dtype int, labels
            ``{0=water, 1=resistive, 2=conductive}``.
        """
        # ── 1. Guard: pyEIT availability ──────────────────────────────────
        if not _PYEIT_AVAILABLE:
            warnings.warn(
                "GaussNewton: pyeit not installed — using random fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._random_fallback()

        # ── 2. Build PyEITMesh from batch.mesh ────────────────────────────
        mesh_obj = build_pyeit_mesh(batch.mesh)
        if mesh_obj is None:
            warnings.warn(
                "GaussNewton: no valid mesh in batch — using random fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._random_fallback()

        # ── 3. Protocol ───────────────────────────────────────────────────
        try:
            protocol   = proto_create(32, dist_exc=1, step_meas=1, parser_meas="std")
            n_meas_tot = protocol.n_meas_tot   # 928 for standard 32-el adjacent
        except Exception as exc:
            warnings.warn(
                f"GaussNewton: protocol creation failed ({exc}) — random fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._random_fallback()

        # ── 4. Align voltages to protocol size ────────────────────────────
        v1, v0 = _prepare_voltages(
            batch.voltages,
            batch.reference_voltages,
            n_meas_tot,
        )

        # ── 5. Solve (one Jacobian iteration) ─────────────────────────────
        try:
            jac = JAC(mesh_obj, protocol=protocol)
            jac.setup(p=0.2, lamb=1e-2, method="kotre")
            ds = jac.solve(v1, v0, normalize=False)   # (n_elements,) element-based
        except Exception as exc:
            warnings.warn(
                f"GaussNewton: JAC.solve failed ({exc}) — random fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._random_fallback()

        # ── 6. Interpolate mesh → 256×256 pixel grid ──────────────────────
        sigma_map = interpolate_to_grid(ds, mesh_obj)

        # ── 7. Segment → labels {0, 1, 2} ────────────────────────────────
        labels = segment(sigma_map)
        self.validate_output(labels)
        return labels

    # ── private helpers ───────────────────────────────────────────────────
    @staticmethod
    def _random_fallback() -> np.ndarray:
        """Segment a random sigma map — used when the real pipeline cannot run."""
        return segment(np.random.rand(256, 256))
