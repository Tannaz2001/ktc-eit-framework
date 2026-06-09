"""BackProjection reconstruction plugin for the KTC 2023 EIT framework.

Overview
--------
Back-projection (BP) is the simplest linear EIT reconstruction.  It
projects the measured voltage difference ``Δv = v_measured − v_ref``
back through a sensitivity matrix ``H`` to produce a per-element
conductivity change image:

    Δσ_elem  ≈  −H @ Δv

The class wraps pyEIT's ``bp.BP`` solver and plugs into the KTC
benchmarking framework through the ``MethodPlugin`` interface.

Lazy initialisation
-------------------
The framework instantiates every method with **no constructor arguments**
(``BackProjection()``).  The pyEIT forward model and BP solver require
the mesh geometry AND the injection protocol, and the injection protocol
is carried inside each ``DataBatch``.  To keep ``__init__`` lightweight
and the class framework-compatible, the solver is built on the **first**
call to ``reconstruct()`` and then cached for all subsequent calls.

Pipeline (per sample)
---------------------
1. Parse ``batch.mesh`` (scipy mat_struct) → ``PyEITMesh``.
2. Extract the 76 injection pairs from ``batch.injection_patterns``
   → ``PyEITProtocol`` via ``build_ktc_protocol``.
3. Compute FEM reference voltages for homogeneous σ = 1 S/m
   → ``v_ref``  via ``compute_v_ref``  (2 356-element vector).
4. Build and setup ``bp.BP`` solver (once; cached as ``self._bp``).
5. Solve: ``ds = bp.solve(batch.voltages, v_ref, normalize=True)``
   → per-element Δσ, shape (n_elements,).
6. Rasterise ``ds`` → 256×256 float image via ``rasterize``.
7. Segment image → discrete labels {0, 1, 2} via ``segment``.
8. Validate and return.

Fallback
--------
Any failure during steps 1-7 (pyEIT not installed, mesh missing,
numerical error) is caught and logged.  A zero-filled (256, 256) uint8
array is returned so the benchmarking loop is never interrupted.

Dependencies
------------
pyeit, numpy, scipy, scikit-image  — all required by the project.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.io
from pyeit.mesh.wrapper import PyEITMesh

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.eit_utils import (
    build_ktc_protocol,
    compute_v_ref,
    rasterize,
)
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.methods.segment import segment
from src.ktc_framework.types import DataBatch

try:
    from pyeit.eit.bp import BP as _BP  # noqa: N814
    _PYEIT_AVAILABLE = True
except ImportError:
    _PYEIT_AVAILABLE = False
    warnings.warn(
        "pyeit is not installed — BackProjection will always return zeros.  "
        "Install it with:  pip install pyeit",
        ImportWarning,
        stacklevel=2,
    )


# ────────────────────────────────────────────────────────────────────────────
# Module-level mesh helper
# ────────────────────────────────────────────────────────────────────────────

def _build_pyeit_mesh(mesh_struct) -> PyEITMesh:
    """Convert a KTC scipy mat_struct (from Mesh_sparse.mat) to PyEITMesh.

    The KTC mesh is stored in MATLAB format with 1-based node indices.
    We convert to 0-based and locate the representative node for each
    electrode by finding the nearest mesh node to the midpoint of the
    first boundary edge in each electrode's ``elfaces`` entry.

    Parameters
    ----------
    mesh_struct :
        scipy mat_struct (``squeeze_me=True, struct_as_record=False``),
        i.e. the value of ``mat['Mesh']``.  Expected fields:

        * ``g``       — (1 602, 2) float  node xy-coordinates
        * ``H``       — (3 073, 3) int    triangle connectivity (1-indexed)
        * ``elfaces`` — (32,)  object array; each entry is a (K, 2) uint8
                        array of boundary-edge node pairs (1-indexed)

    Returns
    -------
    PyEITMesh
        Ready for use with ``EITForward`` and ``bp.BP``.

    Raises
    ------
    AttributeError
        If expected fields are missing from ``mesh_struct``.
    """
    # ── Nodes: (1 602, 2) float64, already in metres / normalised coords ───
    nodes = np.asarray(mesh_struct.g, dtype=np.float64)         # (N, 2)

    # ── Elements: convert from 1-indexed MATLAB to 0-indexed numpy ─────────
    elements = np.asarray(mesh_struct.H, dtype=np.int32) - 1   # (M, 3)

    # ── Electrode positions: one representative node per electrode ──────────
    # elfaces[i] is a (K, 2) array of boundary-edge node pairs (1-indexed).
    # We take the midpoint of the *first* edge and find the nearest node.
    elfaces = mesh_struct.elfaces
    el_pos_list: list[int] = []

    for i in range(len(elfaces)):
        edge = np.asarray(elfaces[i], dtype=np.int32)  # (K, 2) or (2,)

        if edge.ndim == 1:
            # Rare: edge stored as flat [node_a, node_b]
            na, nb = int(edge[0]) - 1, int(edge[1]) - 1
        else:
            # Normal case: first row of the (K, 2) edge list
            na, nb = int(edge[0, 0]) - 1, int(edge[0, 1]) - 1

        # Midpoint of the boundary edge in physical space
        midpoint = (nodes[na] + nodes[nb]) * 0.5          # (2,)

        # Nearest mesh node to that midpoint
        dists   = np.linalg.norm(nodes - midpoint, axis=1)
        nearest = int(np.argmin(dists))
        el_pos_list.append(nearest)

    el_pos = np.array(el_pos_list, dtype=np.int32)         # (32,)

    # ── Assemble PyEITMesh ──────────────────────────────────────────────────
    # perm=1.0 sets the default background conductivity to 1 S/m.
    # This is overridden when we call solve_eit(perm=...) in compute_v_ref.
    return PyEITMesh(node=nodes, element=elements, perm=1.0, el_pos=el_pos)


# ────────────────────────────────────────────────────────────────────────────
# BackProjection plugin
# ────────────────────────────────────────────────────────────────────────────

@register
class BackProjection(MethodPlugin):
    """Linear back-projection EIT reconstruction for the KTC 2023 dataset.

    Parameters
    ----------
    mesh_path : str, optional
        Path to ``Mesh_sparse.mat`` **or** the directory that contains it.
        Defaults to ``"Codes_Matlab/Mesh_sparse.mat"``.

        The framework instantiates plugins with no arguments
        (``BackProjection()``), so the default must point to the standard
        KTC layout.  Override by passing the path explicitly when testing
        outside the framework:

            >>> bp = BackProjection("path/to/Mesh_sparse.mat")

    Notes
    -----
    The pyEIT solver (``bp.BP``) is built lazily on the first call to
    ``reconstruct()`` and cached for all subsequent calls.  This avoids
    running an expensive FEM forward pass during import.
    """

    def __init__(self, mesh_path: str = "Codes_Matlab/Mesh_sparse.mat") -> None:
        # ── Store mesh path for lazy loading ───────────────────────────────
        self._mesh_path: str = mesh_path

        # ── Cached solver state (built on first reconstruct call) ──────────
        self._mesh_obj:  Optional[PyEITMesh]   = None  # pyEIT mesh object
        self._protocol                          = None  # PyEITProtocol
        self._v_ref:     Optional[np.ndarray]  = None  # (2356,) reference voltages
        self._bp                                = None  # bp.BP solver instance

        # ── Eagerly try to load the mesh so __init__ does real work ────────
        # This is best-effort: failure is silently deferred to reconstruct().
        try:
            self._mesh_obj = self._load_mesh_from_path(mesh_path)
        except Exception as exc:
            warnings.warn(
                f"BackProjection.__init__: could not load mesh from "
                f"'{mesh_path}' ({exc!r}).  "
                f"Mesh will be taken from batch.mesh at reconstruct time.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Private: mesh loading ────────────────────────────────────────────────

    @staticmethod
    def _load_mesh_from_path(mesh_path: str) -> PyEITMesh:
        """Load Mesh_sparse.mat from *mesh_path* and return a PyEITMesh.

        Parameters
        ----------
        mesh_path : str
            Either a direct ``.mat`` file path or a directory that contains
            ``Mesh_sparse.mat``.

        Returns
        -------
        PyEITMesh

        Raises
        ------
        FileNotFoundError
            If the .mat file cannot be located.
        KeyError
            If the loaded file does not contain the ``'Mesh'`` key.
        """
        import os
        from pathlib import Path

        p = Path(mesh_path)
        # If a directory is given, look for Mesh_sparse.mat inside it
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
        """Build the BP solver if it hasn't been built yet.

        Called at the top of every ``reconstruct()`` call.  After the first
        successful call the solver is cached and this method returns
        immediately.

        Steps
        -----
        1. Obtain a ``PyEITMesh`` — from ``self._mesh_obj`` (set in
           ``__init__``) or by converting ``batch.mesh`` (the raw mat_struct
           injected by the framework's data loader).
        2. Build a ``PyEITProtocol`` from ``batch.injection_patterns``
           (shape 32×76) via ``build_ktc_protocol``.
        3. Compute FEM reference voltages for homogeneous σ = 1 S/m via
           ``compute_v_ref`` — shape (2 356,).
        4. Instantiate ``pyeit.eit.bp.BP`` and call ``.setup(weight='none')``.

        Parameters
        ----------
        batch : DataBatch
            Current sample — provides ``injection_patterns`` and
            optionally ``mesh`` (used if ``__init__`` could not load the
            mesh from disk).

        Raises
        ------
        RuntimeError
            If pyEIT is not installed.
        ValueError
            If neither ``self._mesh_obj`` nor ``batch.mesh`` is available.
        """
        # Fast path: already initialised
        if self._bp is not None:
            return

        # ── Guard: pyeit must be installed ─────────────────────────────────
        if not _PYEIT_AVAILABLE:
            raise RuntimeError(
                "pyeit is not installed — cannot build BP solver."
            )

        # ── Step 1: get PyEITMesh ──────────────────────────────────────────
        if self._mesh_obj is None:
            # Fall back to the raw mat_struct provided in DataBatch.mesh
            if batch.mesh is None:
                raise ValueError(
                    "BackProjection: no mesh available.  "
                    "Either provide a valid mesh_path or ensure the data "
                    "loader populates batch.mesh."
                )
            self._mesh_obj = _build_pyeit_mesh(batch.mesh)

        # ── Step 2: build the KTC protocol (76 injections, 31 meas each) ──
        # build_ktc_protocol extracts source/sink pairs from the (32×76) Inj
        # matrix and builds the fixed 31-adjacent-pair measurement pattern.
        self._protocol = build_ktc_protocol(batch.injection_patterns)

        # ── Step 3: compute FEM reference voltages ─────────────────────────
        # compute_v_ref runs EITForward with σ = 1.0 S/m everywhere to
        # simulate the empty-tank baseline measurement vector (2 356,).
        self._v_ref = compute_v_ref(self._mesh_obj, self._protocol)

        # ── Step 4: instantiate and setup BP solver ────────────────────────
        # BP(mesh, protocol) stores the forward model for smearing H.
        # setup(weight='none') builds the back-projection matrix H:
        #   H shape: (n_elements, n_meas_tot) = (3073, 2356)
        # After setup, bp.solve(v1, v0) computes:
        #   ds = −H @ (v1 − v0)
        self._bp = _BP(self._mesh_obj, self._protocol)
        self._bp.setup(weight="none")

    # ── Public: reconstruct ───────────────────────────────────────────────────

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Reconstruct a (256, 256) segmentation map from one KTC sample.

        Parameters
        ----------
        batch : DataBatch
            Single EIT measurement sample with fields:

            * ``voltages``          — (2 356,) float32 boundary voltages
            * ``injection_patterns``— (32, 76) float32 KTC Inj matrix
            * ``mesh``              — scipy mat_struct (may be None if
                                       ``__init__`` loaded from disk)
            * ``reference_voltages``— (2 356,) float32 from ref.mat (unused;
                                       we use the FEM-computed ``v_ref``)

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
        1. ``_ensure_solver(batch)`` — build BP solver if not yet done.
        2. ``bp.solve(v1, v_ref, normalize=True)`` — per-element Δσ (3 073,).
        3. ``rasterize(ds, mesh_obj)`` — interpolate to 256×256 float image.
        4. ``segment(img)`` — double-Otsu thresholding → labels {0, 1, 2}.
        5. ``validate_output(labels)`` — asserts shape and label values.
        """
        try:
            # ── Step 1: ensure solver is ready ─────────────────────────────
            self._ensure_solver(batch)

            # ── Step 2: back-project Δv → per-element conductivity change ──
            # bp.solve(v1, v0, normalize=True) computes:
            #   dv = (v1 - v0) / v0        (normalised difference)
            #   ds = -H @ dv               (back-projection)
            # v1 = batch voltages (2 356,), v0 = FEM reference (2 356,)
            v1 = batch.voltages.ravel().astype(np.float64)  # (2356,)
            ds = self._bp.solve(v1, self._v_ref, normalize=True)
            # ds shape: (n_elements,) = (3073,)  — per-element Δσ

            # ── Step 3: rasterise to 256×256 grid ──────────────────────────
            # rasterize() interpolates node/centroid values onto the mesh
            # domain and applies a circular tank mask (outside → 0.0).
            # Negate ds before rasterising: pyEIT's BP computes ds = -H @ dv,
            # so resistive inclusions (higher measured voltage → positive dv)
            # produce NEGATIVE ds.  segment() assigns label 1 to HIGH values,
            # so we flip the sign to make resistive regions the positive peak.
            sigma_map = rasterize(-ds, self._mesh_obj)  # (256, 256) float32

            # ── Step 4: segment into discrete labels ────────────────────────
            # segment() applies double-Otsu thresholding:
            #   label 0 → background / water (low values after negation)
            #   label 1 → resistive inclusion (high values after negation)
            #   label 2 → conductive inclusion (highest values after negation)
            labels = segment(sigma_map)                  # (256, 256) int/uint8
            labels = labels.astype(np.uint8)

            # ── Step 5: validate and return ─────────────────────────────────
            # validate_output() raises ValueError if shape != (256, 256) or
            # labels contain values outside {0, 1, 2}.
            self.validate_output(labels)
            return labels

        except Exception as exc:
            # ── Fallback: return zeros — never crash the benchmarking loop ──
            warnings.warn(
                f"BackProjection.reconstruct failed for sample "
                f"'{batch.sample_id}' (level {batch.level}): {exc!r}.  "
                f"Returning zero segmentation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((256, 256), dtype=np.uint8)
