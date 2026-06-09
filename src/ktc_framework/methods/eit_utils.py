"""KTC EIT solver utilities.

This module wires the KTC 2023 reference forward/Jacobian solver
(``data/KTCScoring/KTCFwd.py``) into the benchmarking framework.  pyEIT
is no longer used: KTC's voltages were recorded under a complete-electrode
model with quadratic-triangle FEM, contact impedance, and an Mincl-masked
measurement pattern -- none of which pyEIT's stock solvers reproduce.

Public functions
----------------
* ``load_ktc_mesh(mesh_path)`` -- read ``Mesh_sparse.mat`` and return a
  dict containing the rebuilt ``Mesh`` (1st-order, used for sigma DOFs)
  and ``Mesh2`` (2nd-order, used for FEM forward).  The rebuild fixes two
  artefacts of the saved .mat: 1-indexed topologies become 0-indexed, and
  ``Node.Coordinate`` (saved on the unit circle) is realigned with the
  metric ``g`` coordinates (~0.115 m tank radius).
* ``build_vincl(level, n_inj=76, n_meas=31, n_electrodes=32)`` -- build
  the level-dependent measurement-validity mask exactly as main.m does:
  for ``categoryNbr=k``, drop all measurements at electrodes 1..2(k-1) and
  drop any injection that uses those electrodes.
* ``build_ktc_jacobian(mesh, injection_patterns, measurement_patterns,
  vincl)`` -- instantiate KTC's ``EITFEM`` and compute the Jacobian about
  homogeneous sigma=1 S/m with contact impedance 1e-6.  Returns
  ``(J, InvGamma_n_diag, solver)`` so callers can build their own
  regulariser.
* ``rasterize(values, mesh)`` -- linear-barycentric interpolation of a
  per-node value array (sigma DOFs) onto a 256x256 image, masked to the
  tank circle.  No normalisation -- the segmenter handles that.
* ``adaptive_segment(grid)`` -- 3-class Otsu plus the bg-is-largest-class
  re-assignment used in main.m line 75-96 to map deltareco signs to
  ``{0=water, 1=resistive, 2=conductive}``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io
from scipy.interpolate import griddata


_KTC_SCORING_DIR = Path(__file__).resolve().parents[3] / "data" / "KTCScoring"
if str(_KTC_SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(_KTC_SCORING_DIR))


N_ELECTRODES = 32
N_INJECTIONS = 76
N_MEAS_PER_INJ = 31
N_MEAS_TOTAL = N_INJECTIONS * N_MEAS_PER_INJ  # 2356


def as_flat(x) -> np.ndarray:
    """Return a 1-D float64 ndarray regardless of np.matrix or trailing dims.

    KTCFwd.SolveForward returns the measured voltages as an ``np.matrix``
    of shape ``(N, 1)`` because it uses ``np.matrix`` for internal
    operations.  Without an explicit ``.ravel()`` after ``np.asarray``,
    arithmetic like ``Uel - Usim`` silently broadcasts ``(N,) - (N, 1)``
    to ``(N, N)`` -- a 5.5M-cell tensor that "works" but produces nonsense.
    Use this helper at every boundary where KTCFwd output meets the rest
    of the pipeline.
    """
    return np.asarray(x, dtype=np.float64).ravel()


def prepare_ktc_mesh_inplace(mat: dict) -> tuple:
    """In-place rewrite of a scipy-loaded Mesh_sparse.mat for KTCFwd.

    Three fixes in one shot:
      1. Convert ``Mesh.H`` and ``Mesh2.H`` from 1-indexed (MATLAB) to
         0-indexed (Python).
      2. Convert each ``Element.Topology`` to 0-indexed; convert
         ``Element.Electrode`` from a NumPy 2.x-hostile 2-element object
         ndarray to a plain Python list ``[elec_idx, edge_nodes]`` (or
         ``[]`` when the element has no electrode).
      3. Rebuild every ``Node.Coordinate`` from the matching row of
         ``g`` so the Jacobian (which reads ``Node.Coordinate``) and the
         forward solve (which reads ``g``) operate at the same metric
         scale.  The saved .mat stores ``Node.Coordinate`` on the unit
         circle but ``g`` in metres -- silent bug for any caller that
         touches both.

    Mutates ``mat['Mesh']`` and ``mat['Mesh2']``.  Returns the same two
    structs as a tuple for convenience.

    Examples
    --------
    >>> mat = scipy.io.loadmat('Mesh_sparse.mat',
    ...                       squeeze_me=True, struct_as_record=False)
    >>> mesh, mesh2 = prepare_ktc_mesh_inplace(mat)
    >>> solver = EITFEM(mesh2, Inj, Mpat, vincl)
    """
    mesh = mat["Mesh"]
    mesh2 = mat["Mesh2"]

    # H: 1-indexed -> 0-indexed
    mesh.H = np.asarray(mesh.H, dtype=np.int64) - 1
    mesh2.H = np.asarray(mesh2.H, dtype=np.int64) - 1

    for target in (mesh, mesh2):
        # Topology + Electrode normalisation
        for elem in target.Element:
            elem.Topology = np.asarray(elem.Topology, dtype=np.int64) - 1
            if isinstance(elem.Electrode, np.ndarray) and elem.Electrode.size > 0:
                elec_idx = int(elem.Electrode[0]) - 1
                edge_nodes = np.asarray(elem.Electrode[1], dtype=np.int64) - 1
                elem.Electrode = [elec_idx, edge_nodes]
            else:
                elem.Electrode = []
        # Node.Coordinate := matching row of g, ElementConnection 0-indexed
        g = np.asarray(target.g, dtype=np.float64)
        for k, node in enumerate(target.Node):
            node.Coordinate = g[k].copy()
            ec = np.asarray(node.ElementConnection, dtype=np.int64)
            node.ElementConnection = (ec - 1) if ec.size > 0 else ec

    return mesh, mesh2


def _rebuild_nodes(H_one_indexed: np.ndarray, g_metres: np.ndarray):
    """Return an object-array of NODE(Coordinate, ElementConnection).

    Mirrors ``KTCMeshing.MakeNode2dSmallFast`` but takes the (already
    loaded) 1-indexed connectivity and metric coordinates -- so the
    rebuilt ``Coordinate`` matches ``g`` instead of the unit-circle
    values that were serialised into Mesh_sparse.mat.
    """
    import KTCMeshing  # available because data/KTCScoring is on sys.path

    n_nodes = g_metres.shape[0]
    H = np.asarray(H_one_indexed, dtype=np.int64) - 1
    max_per_node = 16  # generous; grow if needed
    econn = np.zeros((n_nodes, max_per_node + 1), dtype=np.int64)
    econn[:, 0] = 1
    rowlen = H.shape[1]
    for k in range(H.shape[0]):
        ids = H[k, :]
        idlen = econn[ids, 0]
        if idlen.max() >= econn.shape[1]:
            grown = np.zeros((n_nodes, econn.shape[1] + 10), dtype=np.int64)
            grown[:, :econn.shape[1]] = econn
            econn = grown
        econn[ids, 0] = idlen + 1
        for col in range(rowlen):
            econn[ids[col], idlen[col]] = k

    nodes = np.empty(n_nodes, dtype=object)
    for k in range(n_nodes):
        elen = int(econn[k, 0])
        nodes[k] = KTCMeshing.NODE(
            g_metres[k].copy(),
            econn[k, 1:elen].astype(np.uint32),
        )
    return nodes


def _rebuild_elements(H_one_indexed: np.ndarray, original_elements):
    """Return an object-array of ELEMENT(Topology[0-indexed], Electrode).

    The Electrode field is normalised to a Python list so the truthiness
    check ``if Element.Electrode:`` in KTCFwd.SolveForward works under
    NumPy 2.x (an empty ndarray raises VisibleDeprecationWarning /
    ValueError when treated as a bool).
    """
    import KTCMeshing

    H = np.asarray(H_one_indexed, dtype=np.int64) - 1
    out = np.empty(H.shape[0], dtype=object)
    for i in range(H.shape[0]):
        topo = H[i, :].copy()
        orig_elec = original_elements[i].Electrode
        if hasattr(orig_elec, "shape") and orig_elec.size == 0:
            elec: list = []
        else:
            elec_idx = int(orig_elec[0]) - 1
            edge_nodes = np.asarray(orig_elec[1], dtype=np.int64) - 1
            elec = [elec_idx, edge_nodes]
        out[i] = KTCMeshing.ELEMENT(topo, elec)
    return out


def load_ktc_mesh(mesh_path: str | Path) -> dict:
    """Load Mesh_sparse.mat and return a fully-prepared mesh dict.

    Returned dict has:
    * ``Mesh``   -- 1st-order mesh, KTCMeshing.Mesh-compatible
    * ``Mesh2``  -- 2nd-order mesh, KTCMeshing.Mesh-compatible
    * ``g``      -- (n_nodes_1, 2) float64 metric node coordinates (== Mesh.g)
    * ``H``      -- (n_elements, 3) int64 0-indexed connectivity (== Mesh.H)
    * ``n_sigma``-- int, number of sigma DOFs (= n_nodes_1 = 1602 for KTC)
    """
    import KTCMeshing

    p = Path(mesh_path)
    mat_file = p / "Mesh_sparse.mat" if p.is_dir() else p
    if not mat_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mat_file}")

    mat = scipy.io.loadmat(
        str(mat_file), squeeze_me=True, struct_as_record=False
    )
    if "Mesh" not in mat or "Mesh2" not in mat:
        raise KeyError(
            f"{mat_file} must contain both 'Mesh' and 'Mesh2'. "
            f"Got: {[k for k in mat if not k.startswith('_')]}"
        )

    raw_mesh = mat["Mesh"]
    raw_mesh2 = mat["Mesh2"]

    # Rebuild Element + Node arrays with consistent 0-indexed Topology and
    # metric Coordinate fields, then assemble two new Mesh containers.
    mesh = KTCMeshing.Mesh(
        H=np.asarray(raw_mesh.H, dtype=np.int64) - 1,
        g=np.asarray(raw_mesh.g, dtype=np.float64),
        elfaces=[np.asarray(e, dtype=np.int64) - 1 for e in raw_mesh.elfaces],
        Node=_rebuild_nodes(raw_mesh.H, np.asarray(raw_mesh.g, dtype=np.float64)),
        Element=_rebuild_elements(raw_mesh.H, raw_mesh.Element),
    )
    mesh2 = KTCMeshing.Mesh(
        H=np.asarray(raw_mesh2.H, dtype=np.int64) - 1,
        g=np.asarray(raw_mesh2.g, dtype=np.float64),
        elfaces=[np.asarray(e, dtype=np.int64) - 1 for e in raw_mesh2.elfaces],
        Node=_rebuild_nodes(raw_mesh2.H, np.asarray(raw_mesh2.g, dtype=np.float64)),
        Element=_rebuild_elements(raw_mesh2.H, raw_mesh2.Element),
    )

    return {
        "Mesh":    mesh,
        "Mesh2":   mesh2,
        "g":       mesh.g,
        "H":       mesh.H,
        "n_sigma": mesh.g.shape[0],
    }


def build_vincl(
    level: int,
    injection_patterns: np.ndarray,
    n_meas: int = N_MEAS_PER_INJ,
) -> np.ndarray:
    """Build the (n_meas*n_inj,) bool mask of valid measurements for a level.

    Mirrors main.m lines 10-18 exactly: for difficulty ``categoryNbr=level``,
    let rmind = [0..2*(level-1)-1] be the indices of the first 2*(level-1)
    electrodes.  Then:
      * if any of those electrodes is part of an injection pair, drop the
        entire 31-measurement column for that injection.
      * within every remaining injection, drop the measurements that
        differential pair (rmind[k], rmind[k]+1) -- i.e. measurements
        recorded at the removed electrodes.

    The mask is flattened column-major to match KTCFwd's
    ``self.mincl.shape = (n_meas, n_inj)`` reshape.
    """
    n_inj = injection_patterns.shape[1]
    vincl = np.ones((n_meas, n_inj), dtype=bool)
    n_remove = 2 * (level - 1)
    if n_remove > 0:
        rmind = np.arange(n_remove)
        inj_uses_removed = np.any(injection_patterns[rmind, :] != 0, axis=0)
        vincl[:, inj_uses_removed] = False
        vincl[rmind, :] = False
    # Column-major flatten matches MATLAB's vincl(:) and KTCFwd's reshape.
    return vincl.flatten(order="F")


def build_ktc_jacobian(
    mesh: dict,
    injection_patterns: np.ndarray,
    measurement_patterns: np.ndarray,
    vincl: np.ndarray,
    reference_voltages: Optional[np.ndarray] = None,
    sigma0_value: float = 1.0,
    z_value: float = 1e-6,
):
    """Build KTC's Jacobian and noise-precision matrix.

    Parameters
    ----------
    mesh : dict
        Output of :func:`load_ktc_mesh`.
    injection_patterns : np.ndarray, shape (32, 76)
        KTC ``Injref`` matrix.
    measurement_patterns : np.ndarray, shape (32, 31)
        KTC ``Mpat`` matrix.
    vincl : np.ndarray, shape (2356,) bool
        Output of :func:`build_vincl`.
    reference_voltages : np.ndarray | None, shape (2356,)
        Empty-tank ``Uelref``.  Used to set the noise precision matrix as
        per main.m line 41 (``SetInvGamma(0.05, 0.01, Uelref)``).  If
        None, a Uel-scaled identity is used as a fallback.
    sigma0_value, z_value : float
        Linearisation point (sigma0=1 S/m, z=1e-6 ohm).

    Returns
    -------
    J : (n_valid_meas, n_sigma) float64 Jacobian.
    inv_gamma_n : sparse (n_valid_meas, n_valid_meas) noise precision.
    solver : the EITFEM instance (kept so callers can reuse e.g.
             SetInvGamma with a different noise model).
    Usim : (n_valid_meas,) homogeneous-sigma forward solution.
    """
    import KTCFwd  # type: ignore[import-not-found]

    inj = np.asarray(injection_patterns, dtype=np.float64)
    mpat = np.asarray(measurement_patterns, dtype=np.float64)
    vincl = np.asarray(vincl, dtype=bool)

    if inj.shape != (N_ELECTRODES, N_INJECTIONS):
        raise ValueError(
            f"injection_patterns must be ({N_ELECTRODES}, {N_INJECTIONS}); "
            f"got {inj.shape}"
        )
    if mpat.shape != (N_ELECTRODES, N_MEAS_PER_INJ):
        raise ValueError(
            f"measurement_patterns must be ({N_ELECTRODES}, {N_MEAS_PER_INJ}); "
            f"got {mpat.shape}"
        )
    if vincl.size != N_MEAS_TOTAL:
        raise ValueError(
            f"vincl must have size {N_MEAS_TOTAL}; got {vincl.size}"
        )

    solver = KTCFwd.EITFEM(mesh["Mesh2"], inj, mpat, vincl)

    n_sigma = mesh["n_sigma"]
    sigma0 = sigma0_value * np.ones(n_sigma, dtype=np.float64)
    z = z_value * np.ones(N_ELECTRODES, dtype=np.float64)

    # SolveForward returns an np.matrix of shape (N, 1) -- as_flat collapses
    # to a 1-D ndarray so downstream arithmetic doesn't accidentally
    # broadcast (N,) - (N, 1) into an (N, N) matrix.
    Usim = as_flat(solver.SolveForward(sigma0, z))
    J = np.asarray(solver.Jacobian(sigma0, z))

    # Noise precision -- main.m line 41 uses the FULL (2356,) Uelref as the
    # measurement-scale source, so InvGamma_n is initially (2356, 2356).
    # For levels >= 2, J has fewer rows than 2356 because EITFEM applies the
    # vincl mask in SolveForward / Jacobian.  main.m line 59 selects
    # InvGamma_n(vincl, vincl) at the solve step; we do the same here so
    # the returned matrix has rows/cols aligned with J's row count.
    noise_scale = (
        np.asarray(reference_voltages, dtype=np.float64).ravel()
        if reference_voltages is not None and np.asarray(reference_voltages).size == N_MEAS_TOTAL
        else Usim
    )
    solver.SetInvGamma(0.05, 0.01, noise_scale)

    inv_gamma_full = solver.InvGamma_n
    inv_gamma_dense = (
        inv_gamma_full.toarray()
        if hasattr(inv_gamma_full, "toarray")
        else np.asarray(inv_gamma_full)
    )
    if inv_gamma_dense.shape[0] == vincl.size and J.shape[0] != vincl.size:
        # vincl-mask the noise precision so it lines up with J's row count.
        inv_gamma_dense = inv_gamma_dense[vincl, :][:, vincl]

    return J, inv_gamma_dense, solver, Usim


def rasterize(
    values: np.ndarray,
    mesh: dict,
    grid_size: int = 256,
) -> np.ndarray:
    """Linearly interpolate per-node ``values`` onto a (256, 256) image.

    The image spans the bounding box of ``mesh['g']`` exactly; pixels
    outside the inscribed circle (radius = half the x-extent, centred on
    the bounding-box midpoint) are zeroed.  No normalisation is applied
    -- callers segment from the raw conductivity-change values so the
    sign convention (resistive < 0, conductive > 0) is preserved.

    Parameters
    ----------
    values : np.ndarray, shape (n_sigma,)
        Reconstruction at the 1st-order mesh nodes.
    mesh : dict
        Output of :func:`load_ktc_mesh`.
    grid_size : int, default 256

    Returns
    -------
    np.ndarray, shape (grid_size, grid_size), float32.
    """
    g = mesh["g"]
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vals.size != g.shape[0]:
        raise ValueError(
            f"rasterize: values has size {vals.size}, mesh has {g.shape[0]} nodes"
        )

    x_min, x_max = g[:, 0].min(), g[:, 0].max()
    y_min, y_max = g[:, 1].min(), g[:, 1].max()
    xi = np.linspace(x_min, x_max, grid_size)
    # Reverse y so row 0 (top of image when displayed with imshow) maps to
    # y_max (mathematical "up").  Without this, mesh +y lands at the bottom
    # of the output and reconstructions are vertically flipped relative to
    # the ground-truth label arrays and MATLAB's reference output.  The
    # symptom is "mixed-inclusion samples look like one centred blob" --
    # actually the two inclusions are present but in vertically-swapped
    # locations, so the eye reads them as overlapping in the middle.
    yi = np.linspace(y_max, y_min, grid_size)
    gx, gy = np.meshgrid(xi, yi)

    grid = griddata(g, vals, (gx, gy), method="linear", fill_value=0.0)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    radius = (x_max - x_min) * 0.5
    outside = (gx - cx) ** 2 + (gy - cy) ** 2 > radius ** 2
    grid[outside] = 0.0

    return grid.astype(np.float32)


def adaptive_segment(grid: np.ndarray) -> np.ndarray:
    """Three-class Otsu with main.m's bg-is-largest-class re-assignment.

    Mirrors main.m lines 75-96.  Returns a uint8 array shaped like
    ``grid`` with labels in ``{0=water, 1=resistive, 2=conductive}``.

    The mapping depends on which of the three Otsu classes holds the
    most pixels -- that class is assumed to be the tank background.
    For deltareco where resistive inclusions are negative and conductive
    inclusions are positive, the background sits in the middle class and
    the low/high classes get labels 1 and 2 respectively.
    """
    from skimage.filters import threshold_multiotsu

    finite = np.isfinite(grid)
    safe = np.where(finite, grid, 0.0).astype(np.float64)

    # threshold_multiotsu needs at least 3 distinct values; otherwise
    # default to background-only output.
    if np.unique(safe).size < 3:
        return np.zeros(safe.shape, dtype=np.uint8)

    try:
        t1, t2 = threshold_multiotsu(safe, classes=3)
    except ValueError:
        return np.zeros(safe.shape, dtype=np.uint8)

    low = safe < t1
    mid = (safe >= t1) & (safe <= t2)
    hi = safe > t2

    counts = (int(low.sum()), int(mid.sum()), int(hi.sum()))
    bg_class = int(np.argmax(counts))  # 0=low, 1=mid, 2=hi

    out = np.zeros(safe.shape, dtype=np.uint8)
    if bg_class == 0:
        # bg is the low class -- treat mid and hi as conductive (label 2).
        # This branch fires when the background sits near the bottom of
        # the histogram, e.g. for predominantly-conductive scenes.
        out[mid] = 2
        out[hi] = 2
    elif bg_class == 1:
        # bg is the middle class -- the natural case for linear-difference
        # deltareco where resistive < 0 < conductive.
        out[low] = 1
        out[hi] = 2
    else:
        # bg is the high class -- both other classes are resistive.
        out[low] = 1
        out[mid] = 1

    return out
