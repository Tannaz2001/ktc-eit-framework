"""Gauss-Newton (linearised) reconstruction for the KTC 2023 EIT framework.

Method
------
Single-step Tikhonov-regularised Gauss-Newton on KTC's own Jacobian.
Mirrors data/KTCScoring/main.m line 59:

    deltareco = (J' InvGamma_n J + L' L)^-1 J' InvGamma_n (Uel - Uelref)

where:
* ``J``         is KTC's Jacobian about sigma=1 S/m, z=1e-6.
* ``InvGamma_n``is the noise precision built by EITFEM.SetInvGamma with
                noise_std1=0.05, noise_std2=0.01 against the measured Uelref.
* ``L``         is the Cholesky factor of the smoothness prior
                SMPrior(Mesh.g, corrlength, var_sigma, mean=1).

Defaults vs main.m
------------------
main.m uses ``corrlength = 0.115`` (the tank radius).  That is too
permissive for inclusions on the centimetre scale: the prior smooths
across half the tank, and the resulting reconstruction is a single
blurred lump even when the true inclusion is small.  We default to
``corrlength = 0.04`` (~3 cm) which keeps the prior smoothness assumption
local to the actual inclusion size.  Empirically (sweep_recon.py on
TrainingData) this lifts KTC score on sample 4 from +0.062 (MATLAB) /
+0.113 (corrlength=0.115 Python) to +0.221, and matches or beats MATLAB
on samples 1 and 3.  ``corrlength`` is exposed via the constructor so
callers can revert to the main.m value if they want byte-for-byte parity.

A connected-component cleanup pass drops inclusion blobs smaller than
``min_cc_pixels`` (default 80, i.e. ~0.1 % of the 256x256 image).  These
are virtually always speckles from the Otsu thresholding, not real
inclusions, and they hurt SSIM.  Set ``min_cc_pixels=0`` to disable.
"""

from __future__ import annotations

import hashlib
from typing import Optional
import warnings

import numpy as np

from ktc_framework.adapters.method_registry import register
from ktc_framework.methods.eit_utils import (
    N_MEAS_TOTAL,
    adaptive_segment,
    build_ktc_jacobian,
    build_vincl,
    load_ktc_mesh,
    rasterize,
)
from ktc_framework.methods.method_plugin import MethodPlugin
from ktc_framework.methods import _opcache
from ktc_framework.types import DataBatch


# Module-level mesh + operator caches.  experiment_runner.py instantiates a
# fresh plugin per (method, level, sample) combo (~ 21 calls for samples
# A/B/C × 7 levels), so an instance-level cache is recreated each call and
# every sample pays the ~25 s Jacobian rebuild.  Caching here lets repeat
# instantiations within one process reuse the work.
_MESH_CACHE: dict[str, dict] = {}
_OPERATOR_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _inj_key(injection_patterns: np.ndarray) -> str:
    """Stable hash of an injection pattern matrix for cache keying."""
    arr = np.ascontiguousarray(injection_patterns, dtype=np.float64)
    return hashlib.md5(arr.tobytes()).hexdigest()


_DEFAULT_MESH = "Codes_Matlab/Mesh_sparse.mat"


def _default_mpat() -> np.ndarray:
    """Fallback Mpat: 31 adjacent differential pairs."""
    mpat = np.zeros((32, 31), dtype=np.float64)
    for j in range(31):
        mpat[j, j] = 1.0
        mpat[j + 1, j] = -1.0
    return mpat


def _drop_small_components(labels: np.ndarray, min_pixels: int) -> np.ndarray:
    """Zero out connected components smaller than ``min_pixels`` per inclusion class."""
    if min_pixels <= 0:
        return labels
    from scipy.ndimage import label as _cc_label

    out = labels.copy()
    for lbl in (1, 2):
        mask = labels == lbl
        if not mask.any():
            continue
        cc, n = _cc_label(mask)
        for cid in range(1, n + 1):
            blob = cc == cid
            if int(blob.sum()) < min_pixels:
                out[blob] = 0
    return out


def _consolidate_by_sign(
    labels: np.ndarray, grid: np.ndarray, dominance_ratio: float = 2.0
) -> np.ndarray:
    """Resolve label-1/label-2 splits caused by tighter-prior ringing.

    Tightening SMPrior corrlength sharpens the inclusion core but
    produces low-amplitude ringing of the opposite sign around it.
    3-Otsu can then label the ringing as a second, opposite-class
    inclusion (the "green halo" effect on purely-resistive samples).

    Two-pass cleanup:

    1. **Global dominance check** -- if ``max(|deltareco|)`` of one sign
       exceeds the other sign's by ``dominance_ratio`` (default 2.0), the
       weaker-sign region is presumed ringing.  Set its label to 0.

    2. **Per-CC consolidation** -- for the surviving inclusion mask,
       each connected component is relabelled to match the dominant
       deltareco sign inside it (negative => 1, positive => 2).  Handles
       cases where core and halo *are* in the same CC.
    """
    from scipy.ndimage import label as _cc_label

    if not (labels > 0).any():
        return labels

    out = labels.copy()
    finite = np.isfinite(grid)
    g = np.where(finite, grid, 0.0)

    max_pos = float(g.max())
    min_neg = float(g.min())
    pos_peak = max(max_pos, 0.0)
    neg_peak = max(-min_neg, 0.0)

    # Pass 1: if one sign is much stronger globally, drop the other sign.
    if neg_peak > dominance_ratio * pos_peak and pos_peak > 0:
        out[(out == 2)] = 0
    elif pos_peak > dominance_ratio * neg_peak and neg_peak > 0:
        out[(out == 1)] = 0

    # Pass 2: for each surviving CC, relabel to match local dominant sign.
    inc_mask = out > 0
    if not inc_mask.any():
        return out
    cc, n = _cc_label(inc_mask)
    for cid in range(1, n + 1):
        region = cc == cid
        vals = g[region]
        max_p = float(vals.max()) if vals.size else 0.0
        min_n = float(vals.min()) if vals.size else 0.0
        out[region] = 1 if abs(min_n) > abs(max_p) else 2
    return out


@register
class GaussNewton(MethodPlugin):
    """KTC linearised Gauss-Newton with SMprior smoothness regularisation.

    Parameters
    ----------
    mesh_path : str
        Path to ``Mesh_sparse.mat`` (or its parent directory).
    corrlength : float, default 0.04
        SMPrior correlation length in metres.  main.m uses 0.115 (tank
        radius); 0.04 (~3 cm) typically gives crisper inclusions without
        losing sensitivity.
    var_sigma : float, default 0.05**2
        SMPrior variance (matches main.m).
    noise_std1, noise_std2 : float, default 0.05 / 0.01
        Noise standard deviations passed to ``EITFEM.SetInvGamma``.
    min_cc_pixels : int, default 80
        Drop predicted inclusion components smaller than this many pixels
        after segmentation.  0 disables the cleanup.
    consolidate_by_sign : bool, default True
        When True, every spatially-contiguous non-background blob is
        relabelled to match the sign of its strongest deltareco value
        (negative => label 1 / resistive, positive => label 2 /
        conductive).  This prevents ringing in a tightly-regularised
        reconstruction from being misread as a second, opposite-class
        inclusion (the "green halo" effect on purely-resistive samples).
    """

    _NOISE_STD1 = 0.05
    _NOISE_STD2 = 0.01

    def __init__(
        self,
        mesh_path: str = _DEFAULT_MESH,
        corrlength: float = 0.04,
        var_sigma: float = 0.05 ** 2,
        min_cc_pixels: int = 80,
        consolidate_by_sign: bool = True,
    ) -> None:
        self._mesh_path = mesh_path
        self._corrlength = float(corrlength)
        self._var_sigma = float(var_sigma)
        self._min_cc_pixels = int(min_cc_pixels)
        self._consolidate_by_sign = bool(consolidate_by_sign)

        self._mesh: Optional[dict] = None

        try:
            self._mesh = self._load_mesh_cached(mesh_path)
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.warn(f"GaussNewton: mesh load failed ({exc}); will retry on first reconstruct()", stacklevel=2)

    @staticmethod
    def _load_mesh_cached(mesh_path: str) -> dict:
        """Mesh load + Node rebuild costs ~5 s; cache across instantiations."""
        if mesh_path not in _MESH_CACHE:
            _MESH_CACHE[mesh_path] = load_ktc_mesh(mesh_path)
        return _MESH_CACHE[mesh_path]

    def _ensure_mesh(self) -> dict:
        if self._mesh is None:
            self._mesh = self._load_mesh_cached(self._mesh_path)
        return self._mesh

    def _smprior_LtL(self, mesh: dict) -> np.ndarray:
        """Compute L'L for the SMPrior centred on sigma0=1 over Mesh.g."""
        import KTCRegularization  # type: ignore[import-not-found]

        n_sigma = mesh["n_sigma"]
        sm = KTCRegularization.SMPrior(
            mesh["g"],
            self._corrlength,
            self._var_sigma,
            np.ones(n_sigma, dtype=np.float64),
        )
        L = np.asarray(sm.L)
        return L.T @ L

    def _get_operators(
        self,
        level: int,
        injection_patterns: np.ndarray,
        measurement_patterns: np.ndarray,
        reference_voltages: Optional[np.ndarray],
    ):
        # Cache key includes everything that affects the cached tensors:
        # the mesh, the regularisation hyperparameters, the level (drives
        # vincl), and the injection pattern (drives J's row layout).  We
        # leave reference_voltages out of the key -- it's per-sample noise
        # scale and doesn't change J or vincl.
        cache_key = (
            self._mesh_path,
            self._corrlength,
            self._var_sigma,
            int(level),
            _inj_key(injection_patterns),
        )
        cached = _OPERATOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # On-disk cache: skip the expensive Jacobian build across processes.
        # cache_key is already stable (no id()), so it is safe as a disk key.
        disk_key = "gn|" + "|".join(str(x) for x in cache_key)
        disk_val = _opcache.load(disk_key)
        if disk_val is not None:
            _OPERATOR_CACHE[cache_key] = disk_val
            return disk_val

        mesh = self._ensure_mesh()
        vincl = build_vincl(level, injection_patterns)
        J, inv_gamma_n, _solver, _Usim = build_ktc_jacobian(
            mesh,
            injection_patterns,
            measurement_patterns,
            vincl,
            reference_voltages=reference_voltages,
        )
        inv_gamma_dense = (
            inv_gamma_n.toarray() if hasattr(inv_gamma_n, "toarray") else np.asarray(inv_gamma_n)
        )
        LtL = self._smprior_LtL(mesh)
        entry = (J, inv_gamma_dense, LtL, vincl)
        _OPERATOR_CACHE[cache_key] = entry
        _opcache.save(disk_key, entry)
        return entry

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        try:
            mesh = self._ensure_mesh()
        except Exception as exc:
            warnings.warn(
                f"GaussNewton: mesh unavailable ({exc}); returning zeros.",
                RuntimeWarning, stacklevel=2,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        mpat = (
            np.asarray(batch.measurement_patterns, dtype=np.float64)
            if getattr(batch, "measurement_patterns", None) is not None
            else _default_mpat()
        )
        v_ref = batch.reference_voltages
        if v_ref is None:
            v_ref = np.full(N_MEAS_TOTAL, float(np.asarray(batch.voltages).mean()))
        v_ref = np.asarray(v_ref, dtype=np.float64).ravel()

        J, inv_gamma_n, LtL, vincl = self._get_operators(
            batch.level, batch.injection_patterns, mpat, v_ref
        )

        v1 = np.asarray(batch.voltages, dtype=np.float64).ravel()
        dv = (v1 - v_ref)[vincl]

        # Linear difference Gauss-Newton step (main.m line 59)
        A = J.T @ inv_gamma_n @ J + LtL
        b = J.T @ inv_gamma_n @ dv
        deltareco = np.linalg.solve(A, b)

        grid = rasterize(deltareco, mesh)
        labels = adaptive_segment(grid)
        if self._consolidate_by_sign:
            labels = _consolidate_by_sign(labels, grid)
        labels = _drop_small_components(labels, self._min_cc_pixels)
        self.validate_output(labels)
        return labels
