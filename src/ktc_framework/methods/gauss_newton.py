"""GaussNewton reconstruction plugin for the KTC 2023 EIT framework.

Overview
--------
The linearised Gauss-Newton method solves the EIT inverse problem by
minimising the regularised least-squares objective:

    min_Δσ  ‖J Δσ − Δv‖²  +  λ ‖R Δσ‖²

where
  * J  is the Jacobian (sensitivity matrix), shape (n_meas, n_elements)
  * Δv = v_measured − v_ref  is the voltage difference vector
  * λ  is the regularisation parameter
  * R  is a regularisation operator (Kotre's method used here)

The closed-form solution for one linearised step is:

    Δσ  =  −(J^T J + λ R)^{−1} J^T Δv
         =  −H Δv

This is identical in structure to back-projection, but with a
physics-informed H that incorporates the full Jacobian and regularisation
— producing sharper, more accurate reconstructions than the simple
1/distance sensitivity approximation used by BackProjection.

Regularisation — Kotre's method
--------------------------------
``method='kotre'`` applies Kotre's diagonal weighting to J^T J before
adding λ I, emphasising elements that contribute strongly to measurements
(near the electrodes) and down-weighting deep, poorly-determined elements.
``p=0.5`` is the balance exponent; ``lamb=0.01`` is the regularisation
strength.

Normalisation (``jac_normalized=True``)
----------------------------------------
The Jacobian is normalised by the forward voltage ``v0 = v_ref`` column-
wise.  This makes the reconstruction dimensionless and more robust to
measurement scale differences between samples and difficulty levels.

Lazy initialisation
-------------------
Identical to BackProjection: the framework calls ``GaussNewton()`` with no
arguments.  The JAC solver is built on the first ``reconstruct()`` call
and cached.  Building the Jacobian (a full FEM solve per excitation) is
the most expensive step — roughly 0.5 s for the 1 602-node KTC mesh.

Pipeline (per sample)
---------------------
1. Parse ``batch.mesh`` → ``PyEITMesh``  (or use the one loaded in init).
2. Build ``PyEITProtocol`` from ``batch.injection_patterns``.
3. Compute FEM reference voltages ``v_ref`` (σ = 1 S/m).
4. Build ``jac.JAC`` and call ``.setup(p=0.5, lamb=0.01,
   method='kotre', perm=1.0, jac_normalized=True)``.
5. ``ds = jac.solve(v1, v_ref, normalize=True)`` → (n_elements,) Δσ.
6. ``rasterize(ds, mesh_obj)``  → 256×256 float image.
7. ``segment(img)``             → discrete labels {0, 1, 2}.
8. Validate and return.

Fallback
--------
Any failure returns a zero-filled (256, 256) uint8 array so the
benchmarking loop is never interrupted.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.io

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.backprojection import _build_pyeit_mesh
from src.ktc_framework.methods.eit_utils import (
    build_ktc_protocol,
    compute_v_ref,
    rasterize,
)
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.methods.segment import segment
from src.ktc_framework.types import DataBatch

try:
    from pyeit.eit.jac import JAC as _JAC
    from pyeit.mesh.wrapper import PyEITMesh
    _PYEIT_AVAILABLE = True
except ImportError:
    _PYEIT_AVAILABLE = False
    warnings.warn(
        "pyeit is not installed — GaussNewton will always return zeros.  "
        "Install it with:  pip install pyeit",
        ImportWarning,
        stacklevel=2,
    )


# ────────────────────────────────────────────────────────────────────────────
# GaussNewton plugin
# ────────────────────────────────────────────────────────────────────────────

@register
class GaussNewton(MethodPlugin):
    """Linearised Gauss-Newton EIT reconstruction for the KTC 2023 dataset.

    Uses pyEIT's ``jac.JAC`` solver with Kotre regularisation.  The Jacobian
    is computed once during solver setup and then applied as a fixed linear
    operator for each sample.

    Parameters
    ----------
    mesh_path : str, optional
        Path to ``Mesh_sparse.mat`` **or** the directory containing it.
        Defaults to ``"Codes_Matlab/Mesh_sparse.mat"``.

        The framework calls ``GaussNewton()`` with no arguments, so the
        default covers the standard KTC repo layout.

    Notes
    -----
    The JAC solver is built lazily on the first ``reconstruct()`` call.
    Subsequent calls reuse the cached solver (same mesh + protocol).
    Building the Jacobian is O(n_exc × FEM solve) ≈ 0.5 s for the
    1 602-node KTC mesh — acceptable for offline benchmarking.
    """

    # ── Regularisation hyper-parameters (class-level constants) ─────────────
    #
    # These match the default KTC reconstruction setup recommended by the
    # challenge organisers and commonly used in pyEIT examples.
    #
    #   p    = 0.5   — Kotre exponent; 0 → pure Tikhonov, 1 → full Kotre
    #   lamb = 0.01  — regularisation strength; higher → smoother but blurrier
    #   method = 'kotre' — diagonal weighting of J^T J
    #   perm = 1.0   — background conductivity for Jacobian computation (S/m)
    #   jac_normalized = True — normalise J column-wise by v_ref

    _JAC_P      = 0.5
    _JAC_LAMB   = 0.01
    _JAC_METHOD = "kotre"
    _JAC_PERM   = 1.0
    _JAC_NORM   = True

    def __init__(self, mesh_path: str = "Codes_Matlab/Mesh_sparse.mat") -> None:
        # ── Store mesh path for lazy loading ───────────────────────────────
        self._mesh_path: str = mesh_path

        # ── Cached solver state (all None until first reconstruct) ─────────
        self._mesh_obj  = None   # PyEITMesh — parsed from mat_struct
        self._protocol  = None   # PyEITProtocol — built from injection_patterns
        self._v_ref     = None   # np.ndarray (2356,) — FEM reference voltages
        self._jac       = None   # jac.JAC — the solver instance

        # ── Eagerly try to load the mesh ───────────────────────────────────
        # Best-effort: failure defers to batch.mesh at reconstruct time.
        # We load the mesh here (not the solver) because the solver also
        # needs batch.injection_patterns, which isn't available yet.
        try:
            self._mesh_obj = self._load_mesh_from_path(mesh_path)
        except Exception as exc:
            warnings.warn(
                f"GaussNewton.__init__: could not load mesh from "
                f"'{mesh_path}' ({exc!r}).  "
                f"Mesh will be taken from batch.mesh at reconstruct time.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Private: mesh loading ────────────────────────────────────────────────

    @staticmethod
    def _load_mesh_from_path(mesh_path: str) -> "PyEITMesh":
        """Load ``Mesh_sparse.mat`` from *mesh_path* and return a PyEITMesh.

        Accepts a direct path to the ``.mat`` file or a directory containing
        it.

        Parameters
        ----------
        mesh_path : str

        Returns
        -------
        PyEITMesh

        Raises
        ------
        FileNotFoundError
            If the .mat file cannot be found.
        KeyError
            If the file does not contain the ``'Mesh'`` key.
        """
        from pathlib import Path

        p = Path(mesh_path)
        mat_file = p / "Mesh_sparse.mat" if p.is_dir() else p

        if not mat_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mat_file}")

        mat = scipy.io.loadmat(
            str(mat_file), squeeze_me=True, struct_as_record=False
        )
        if "Mesh" not in mat:
            raise KeyError(f"'Mesh' key not found in {mat_file}")

        return _build_pyeit_mesh(mat["Mesh"])

    # ── Private: lazy solver initialisation ──────────────────────────────────

    def _ensure_solver(self, batch: DataBatch) -> None:
        """Build the JAC solver if it has not been built yet.

        Called at the top of every ``reconstruct()`` call.  Returns
        immediately after the first successful build (cached state).

        Build steps
        -----------
        1. **PyEITMesh** — from ``self._mesh_obj`` (loaded in ``__init__``)
           or by converting ``batch.mesh`` (raw mat_struct from the loader).
        2. **PyEITProtocol** — from ``batch.injection_patterns`` via
           ``build_ktc_protocol`` (76 patterns, 31 adjacent meas pairs).
        3. **v_ref** — FEM forward simulation at σ = 1 S/m via
           ``compute_v_ref``, shape (2 356,).
        4. **JAC solver** — ``jac.JAC(mesh, protocol).setup(...)``
           computes the full Jacobian J (shape 2 356 × 3 073) and the
           regularised back-projection matrix H = (J^T J + λR)^{-1} J^T.

        Parameters
        ----------
        batch : DataBatch
            Provides ``injection_patterns`` and optionally ``mesh``.

        Raises
        ------
        RuntimeError
            If pyEIT is not installed.
        ValueError
            If no mesh is available from either source.
        """
        # Fast path: already initialised
        if self._jac is not None:
            return

        # ── Guard: pyeit must be installed ─────────────────────────────────
        if not _PYEIT_AVAILABLE:
            raise RuntimeError(
                "pyeit is not installed — cannot build GaussNewton solver."
            )

        # ── Step 1: get PyEITMesh ──────────────────────────────────────────
        if self._mesh_obj is None:
            if batch.mesh is None:
                raise ValueError(
                    "GaussNewton: no mesh available.  "
                    "Either provide a valid mesh_path or ensure the data "
                    "loader populates batch.mesh."
                )
            # Convert the raw scipy mat_struct to a PyEITMesh object
            self._mesh_obj = _build_pyeit_mesh(batch.mesh)

        # ── Step 2: build the KTC measurement protocol ─────────────────────
        # build_ktc_protocol reads the 76 injection pairs from the (32×76)
        # Inj matrix and constructs the fixed 31-adjacent-pair meas pattern.
        # Returns PyEITProtocol with n_exc=76, n_meas=31, n_meas_tot=2356.
        self._protocol = build_ktc_protocol(batch.injection_patterns)

        # ── Step 3: compute FEM reference voltages ─────────────────────────
        # Runs EITForward with homogeneous σ = 1.0 S/m.
        # Returns flat (2 356,) array matching the KTC voltage vector shape.
        self._v_ref = compute_v_ref(self._mesh_obj, self._protocol)

        # ── Step 4: build and setup the JAC solver ─────────────────────────
        # JAC(mesh, protocol) wires up the FEM forward model.
        # .setup() computes:
        #   - J : Jacobian, shape (n_meas_tot, n_elements) = (2356, 3073)
        #         dV[m] / dσ[e] — sensitivity of measurement m to element e
        #   - H : regularised back-projection matrix, shape (3073, 2356)
        #         H = −(J^T J + λ R)^{-1} J^T  (Kotre weighting)
        # After setup, jac.solve(v1, v0) computes ds = H @ (v1 − v0).
        #
        # Hyper-parameters:
        #   p=0.5      : Kotre exponent — balances uniform vs depth-weighted reg
        #   lamb=0.01  : regularisation strength
        #   method='kotre': diagonal weighting before adding λ I
        #   perm=1.0   : background conductivity for J computation
        #   jac_normalized=True: normalise J column-wise by v_ref for
        #                        scale-invariant reconstruction
        self._jac = _JAC(self._mesh_obj, self._protocol)
        self._jac.setup(
            p=self._JAC_P,
            lamb=self._JAC_LAMB,
            method=self._JAC_METHOD,
            perm=self._JAC_PERM,
            jac_normalized=self._JAC_NORM,
        )

    # ── Public: reconstruct ───────────────────────────────────────────────────

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Reconstruct a (256, 256) segmentation map from one KTC sample.

        Parameters
        ----------
        batch : DataBatch
            Single EIT measurement sample with fields:

            * ``voltages``           — (2 356,) float32 boundary voltages
            * ``injection_patterns`` — (32, 76) float32 KTC Inj matrix
            * ``mesh``               — scipy mat_struct (may be None if
                                        ``__init__`` loaded from disk)

        Returns
        -------
        np.ndarray
            Shape (256, 256), dtype uint8, values in {0, 1, 2}.

            * 0 = water / background
            * 1 = resistive inclusion
            * 2 = conductive inclusion

            Returns a zero-filled array on any failure (never raises).

        Pipeline
        --------
        1. ``_ensure_solver(batch)`` — build JAC solver if not yet done.
        2. ``jac.solve(v1, v_ref, normalize=True)``
             One linearised Gauss-Newton step:
             ds = −H @ ((v1 − v_ref) / v_ref)  shape (3 073,).
        3. ``rasterize(ds, mesh_obj)``
             Scatter ds at element centroids → 256×256 float image.
        4. ``segment(img)``
             Double-Otsu thresholding → uint8 labels {0, 1, 2}.
        5. ``validate_output(labels)`` — assert shape and label values.

        Notes
        -----
        Unlike BackProjection (per-node ds), JAC returns per-**element** ds
        because H has shape (n_elements, n_meas_tot).  ``rasterize`` uses
        element centroids as scatter points for this case.
        """
        try:
            # ── Step 1: ensure solver is ready ─────────────────────────────
            self._ensure_solver(batch)

            # ── Step 2: one linearised Gauss-Newton step ───────────────────
            # jac.solve(v1, v0, normalize=True) computes:
            #   dv = (v1 - v0) / v0      (element-wise normalised difference)
            #   ds = -H @ dv             (regularised back-projection)
            # v1 = measured voltages (2 356,)
            # v0 = FEM reference voltages (2 356,) at σ = 1.0 S/m
            # ds shape: (n_elements,) = (3 073,)  — per-element Δσ
            v1 = batch.voltages.ravel().astype(np.float64)  # (2356,)
            ds = self._jac.solve(v1, self._v_ref, normalize=True)

            # ── Step 3: rasterise to 256×256 pixel grid ────────────────────
            # Negate ds before rasterising: pyEIT's JAC computes ds = -H @ dv,
            # so resistive inclusions produce NEGATIVE ds (like BP).
            # segment() assigns label 1 to HIGH values, so flip the sign so
            # resistive regions become the positive peak in the image.
            # .real: JAC can return complex ds; real part = conductivity change.
            sigma_map = rasterize(-ds.real, self._mesh_obj)  # (256,256) float32

            # ── Step 4: segment into discrete labels ────────────────────────
            # segment() applies double-Otsu thresholding:
            #   label 0 → background / water (low values after negation)
            #   label 1 → resistive inclusion (high values after negation)
            #   label 2 → conductive inclusion (highest values after negation)
            labels = segment(sigma_map)   # (256, 256) int
            labels = labels.astype(np.uint8)

            # ── Step 5: validate and return ─────────────────────────────────
            # Raises ValueError if shape != (256, 256) or labels ∉ {0, 1, 2}
            self.validate_output(labels)
            return labels

        except Exception as exc:
            # ── Fallback: never crash the benchmarking loop ─────────────────
            warnings.warn(
                f"GaussNewton.reconstruct failed for sample "
                f"'{batch.sample_id}' (level {batch.level}): {exc!r}.  "
                f"Returning zero segmentation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((256, 256), dtype=np.uint8)
