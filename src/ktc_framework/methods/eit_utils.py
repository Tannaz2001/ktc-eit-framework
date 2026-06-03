"""Shared EIT utilities for the KTC 2023 framework.

Three public functions used by BackProjection, GaussNewton, and any
future reconstruction method that needs to work with the KTC dataset.

KTC 2023 measurement protocol recap
------------------------------------
* 32 electrodes on a circular tank (unit circle, centred at origin).
* 76 injection patterns  — each column of the (32×76) Inj matrix has
  exactly one +1 (current source) and one -1 (current sink).
* 31 voltage measurements per injection — fixed set of 31 adjacent
  differential pairs (el_0–el_1), (el_1–el_2), …, (el_30–el_31).
* Total: 76 × 31 = 2 356 scalar measurements per data sample.

Dependencies
------------
numpy, scipy, pyeit  (all already required by the project).

Authors
-------
KTC EIT framework team.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import griddata

# pyEIT protocol building blocks
from pyeit.eit.protocol import PyEITProtocol

# pyEIT forward solver (Complete Electrode Model FEM)
from pyeit.eit.fem import EITForward


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

#: Number of electrodes in the KTC 2023 tank.
N_ELECTRODES: int = 32

#: Number of injection patterns in a single KTC data sample.
N_INJECTIONS: int = 76

#: Number of voltage measurements per injection pattern.
#: KTC uses the 31 adjacent differential pairs (0-1),(1-2),…,(30-31).
N_MEAS_PER_INJ: int = 31

#: Total number of voltage measurements per sample (76 × 31 = 2 356).
N_MEAS_TOTAL: int = N_INJECTIONS * N_MEAS_PER_INJ  # 2 356


# ────────────────────────────────────────────────────────────────────────────
# Function 1 — build_ktc_protocol
# ────────────────────────────────────────────────────────────────────────────

def build_ktc_protocol(injection_patterns: np.ndarray) -> PyEITProtocol:
    """Build a pyEIT protocol object that encodes the KTC measurement setup.

    The KTC Inj matrix has shape (32, 76): each *column* describes one
    injection pattern.  pyEIT's ``ex_mat`` convention is the *transpose*:
    shape (76, 2) with [source_electrode, sink_electrode] per row.

    The 31 voltage measurements per injection are the fixed adjacent pairs:

        (0, 1), (1, 2), (2, 3), …, (30, 31)

    These are the *same* 31 pairs for every injection pattern (the KTC
    protocol does not skip the driving electrodes in its measurement list).

    The returned ``PyEITProtocol`` is compatible with ``pyeit.eit.bp.BP``
    and ``pyeit.eit.jac.JAC``:

    * ``protocol.n_exc``     → 76
    * ``protocol.n_meas``    → 31
    * ``protocol.n_meas_tot``→ 2 356  (matches the KTC voltage vector length)

    Parameters
    ----------
    injection_patterns : np.ndarray
        Shape **(32, 76)** — the KTC Inj matrix loaded from the .mat file.
        Each column has exactly one +1 (source) and one -1 (sink).

    Returns
    -------
    PyEITProtocol
        Fully constructed protocol object ready for use with pyEIT solvers.

    Raises
    ------
    ValueError
        If fewer than one valid injection pair can be extracted from
        ``injection_patterns`` (e.g. the matrix is all zeros).

    Examples
    --------
    >>> protocol = build_ktc_protocol(batch.injection_patterns)
    >>> bp_solver = BP(mesh_obj, protocol)
    >>> bp_solver.setup()
    """

    # ------------------------------------------------------------------
    # Step A — Extract ex_mat (76, 2) from the KTC Inj matrix (32, 76).
    # ------------------------------------------------------------------
    # The Inj matrix stores injection patterns column-wise:
    #   col j has +1 at the source electrode index and -1 at the sink.
    # We iterate over columns to collect [source_idx, sink_idx] pairs.

    inj = np.asarray(injection_patterns, dtype=np.float64)  # (32, 76)

    if inj.ndim != 2 or inj.shape[0] != N_ELECTRODES:
        raise ValueError(
            f"injection_patterns must have shape ({N_ELECTRODES}, {N_INJECTIONS}), "
            f"got {inj.shape}"
        )

    ex_list: list[list[int]] = []
    for col in range(inj.shape[1]):           # iterate over 76 patterns
        col_vec = inj[:, col]
        # Find the electrode carrying positive current (+1)
        src_idx = np.where(col_vec > 0.5)[0]
        # Find the electrode carrying negative current (-1 / sink)
        snk_idx = np.where(col_vec < -0.5)[0]

        if src_idx.size > 0 and snk_idx.size > 0:
            # Both electrodes found — store 0-indexed pair
            ex_list.append([int(src_idx[0]), int(snk_idx[0])])
        else:
            # Warn but skip malformed columns (shouldn't happen with real KTC data)
            warnings.warn(
                f"build_ktc_protocol: column {col} of injection_patterns has no "
                f"valid +1/-1 pair — skipping.",
                RuntimeWarning,
                stacklevel=2,
            )

    if not ex_list:
        raise ValueError(
            "build_ktc_protocol: no valid injection pairs found in injection_patterns."
        )

    # ex_mat shape: (n_valid_inj, 2) — typically (76, 2)
    ex_mat = np.array(ex_list, dtype=np.int32)
    n_inj = ex_mat.shape[0]  # should be 76

    if n_inj != N_INJECTIONS:
        warnings.warn(
            f"build_ktc_protocol: expected {N_INJECTIONS} injection patterns, "
            f"got {n_inj}.  Proceeding with {n_inj}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Step B — Build meas_mat (n_inj, 31, 2).
    # ------------------------------------------------------------------
    # KTC measures 31 adjacent differential voltages for every injection.
    # The pairs are fixed: (el_0 − el_1), (el_1 − el_2), …, (el_30 − el_31).
    # pyEIT's subtract_row() interprets meas_mat[i] as (n_meas, 2) where:
    #   column 0 = "positive" electrode (N)
    #   column 1 = "negative" / reference electrode (M)
    # giving  v_diff = v[N] - v[M]  for each row.

    # Build the 31 adjacent pairs once — same for every injection
    meas_a = np.arange(N_MEAS_PER_INJ, dtype=np.int32)         # [0, 1, …, 30]
    meas_b = meas_a + 1                                          # [1, 2, …, 31]
    single_meas = np.stack([meas_a, meas_b], axis=1)            # (31, 2)

    # Tile across all injections: (n_inj, 31, 2)
    # np.tile repeats along the first axis, giving the same 31 pairs
    # for every one of the 76 injection patterns.
    meas_mat = np.tile(single_meas, (n_inj, 1, 1))              # (76, 31, 2)

    # ------------------------------------------------------------------
    # Step C — Build keep_ba boolean mask.
    # ------------------------------------------------------------------
    # keep_ba marks which (injection, measurement) combinations are *valid*
    # and should be retained in the data vector.  For KTC we keep all
    # 76 × 31 = 2 356 measurements, so the mask is entirely True.
    # Shape: (n_inj * N_ELECTRODES,) following pyEIT's internal convention.
    # (pyEIT constructs keep_ba over the full n_el measurement space before
    #  trimming — we keep the full block True here.)
    keep_ba = np.ones(n_inj * N_ELECTRODES, dtype=bool)

    return PyEITProtocol(ex_mat, meas_mat, keep_ba)


# ────────────────────────────────────────────────────────────────────────────
# Function 2 — compute_v_ref
# ────────────────────────────────────────────────────────────────────────────

def compute_v_ref(mesh_obj, protocol_obj: PyEITProtocol) -> np.ndarray:
    """Compute reference voltages for a *homogeneous* empty tank.

    Runs the pyEIT FEM forward solver with uniform conductivity σ = 1.0
    S/m over all mesh elements.  This simulates the voltage measurements
    that would be recorded if the tank were filled with perfectly uniform
    saline — the "empty tank" baseline used for difference imaging.

    Difference imaging subtracts this reference from the actual
    measurements to isolate the conductivity perturbation caused by an
    inclusion:

        Δv = v_measured − v_ref

    Parameters
    ----------
    mesh_obj :
        A ``pyeit.mesh.wrapper.PyEITMesh`` (or compatible object) with:

        * ``.node``    — shape **(n_nodes, 2)** float, node coordinates.
        * ``.element`` — shape **(n_elements, 3)** int, triangle connectivity.
        * ``.el_pos``  — shape **(32,)** int, electrode-to-node mapping.
        * ``.perm``    — shape **(n_elements,)** float, default conductivities
          (will be overridden by the σ = 1.0 homogeneous value).

    protocol_obj : PyEITProtocol
        Protocol object returned by :func:`build_ktc_protocol`.
        Must have ``n_exc = 76`` and ``n_meas = 31``.

    Returns
    -------
    np.ndarray
        Flat float64 voltage array of length **2 356** (76 × 31),
        matching the shape of a single KTC sample's voltage vector.

    Notes
    -----
    * ``EITForward.solve_eit(perm)`` iterates over every excitation in
      ``protocol_obj.ex_mat``, solves the FEM linear system, then applies
      ``subtract_row`` with ``protocol_obj.meas_mat[i]`` to build the
      differential measurements — exactly replicating the KTC acquisition.
    * Using σ = 1.0 is equivalent to setting the background conductivity;
      the absolute value does not matter for difference imaging because it
      cancels in Δv = v_meas − v_ref (assuming the same background
      conductivity in both).

    Examples
    --------
    >>> protocol  = build_ktc_protocol(batch.injection_patterns)
    >>> v_ref     = compute_v_ref(mesh_obj, protocol)
    >>> delta_v   = batch.voltages.ravel() - v_ref   # difference voltages
    """

    # ------------------------------------------------------------------
    # Build the FEM forward solver.
    # EITForward takes (mesh, protocol) and assembles the stiffness
    # matrix and electrode boundary conditions internally.
    # ------------------------------------------------------------------
    fwd = EITForward(mesh_obj, protocol_obj)

    # ------------------------------------------------------------------
    # Number of finite elements (triangles in 2-D mesh).
    # mesh_obj.element has shape (n_elements, 3).
    # ------------------------------------------------------------------
    n_elements: int = mesh_obj.element.shape[0]

    # ------------------------------------------------------------------
    # Run the forward simulation with homogeneous conductivity σ = 1.0.
    # solve_eit(perm) accepts a scalar or (n_elements,) array.
    # It returns a flat vector of shape (n_exc * n_meas,) = (2 356,).
    # ------------------------------------------------------------------
    homogeneous_perm = np.ones(n_elements, dtype=np.float64)
    v_ref: np.ndarray = fwd.solve_eit(perm=homogeneous_perm)

    # ------------------------------------------------------------------
    # Sanity check: warn if the output length doesn't match expectation.
    # ------------------------------------------------------------------
    expected_len = protocol_obj.n_meas_tot  # should be 2 356
    if v_ref.size != expected_len:
        warnings.warn(
            f"compute_v_ref: expected {expected_len} voltage values, "
            f"got {v_ref.size}.  Check protocol and mesh compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )

    return v_ref.ravel().astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Function 3 — rasterize
# ────────────────────────────────────────────────────────────────────────────

def rasterize(ds: np.ndarray, mesh_obj) -> np.ndarray:
    """Interpolate per-element conductivity values onto a 256×256 pixel grid.

    Converts the unstructured finite-element result (one scalar per
    triangular element) into a regular pixel image suitable for
    segmentation and scoring.

    Algorithm
    ---------
    1. Compute element centroids as the mean of the three corner nodes.
    2. Use ``scipy.interpolate.griddata`` (linear barycentric) to
       interpolate ``ds`` from centroids onto a uniform 256×256 grid
       spanning [−1, +1] × [−1, +1] (the KTC tank domain).
    3. Mask pixels outside the unit circle (x² + y² > 1) by setting
       them to 0.0 — they are outside the physical tank.

    Parameters
    ----------
    ds : np.ndarray
        Conductivity change values produced by ``BP.solve()`` or
        ``JAC.solve()`` in pyEIT.  Accepted shapes:

        * **(n_elements,)** — per-element values (e.g. from JAC).
          Scatter points are element centroids.
        * **(n_nodes,)** — per-node values (e.g. from BP, which uses a
          node-based smear matrix H of shape (n_nodes, n_meas_tot)).
          Scatter points are the node coordinates directly.

    mesh_obj :
        pyEIT mesh object with:

        * ``.node``    — shape **(n_nodes, 2)** float, xy coordinates in
          the same coordinate system as the tank (unit circle).
        * ``.element`` — shape **(n_elements, 3)** int, 0-indexed triangle
          connectivity (node indices for each triangle's three corners).

    Returns
    -------
    np.ndarray
        Shape **(256, 256)** float32.  Values inside the tank are
        interpolated conductivity changes; values outside are 0.0.

    Notes
    -----
    * ``griddata`` uses linear interpolation (barycentric coordinates
      within each triangle of the Delaunay triangulation of the centroids).
      Pixels that fall outside the convex hull of the centroids receive
      the ``fill_value=0.0`` (these are near the boundary anyway and are
      subsequently masked by the circle mask).
    * The output dtype is **float32** to match downstream segmentation
      functions (``threshold_otsu`` and ``_segment_adaptive``).

    Examples
    --------
    >>> protocol  = build_ktc_protocol(batch.injection_patterns)
    >>> v_ref     = compute_v_ref(mesh_obj, protocol)
    >>> bp_solver = BP(mesh_obj, protocol)
    >>> bp_solver.setup()
    >>> ds        = bp_solver.solve(batch.voltages.ravel(), v_ref)
    >>> img       = rasterize(ds, mesh_obj)   # (256, 256) float32
    """

    # ------------------------------------------------------------------
    # Step 1 — Element centroids.
    # mesh_obj.node  : (n_nodes, 2)   — x, y coordinates of each node
    # mesh_obj.element: (n_elements, 3) — indices of three corner nodes
    # Indexing node by element gives (n_elements, 3, 2); mean over axis=1
    # gives the centroid of each triangle.
    # ------------------------------------------------------------------
    # PyEITMesh stores nodes as (n_nodes, 3) even for 2-D meshes — it appends
    # a z = 0 column.  We only need the first two (x, y) columns for griddata.
    nodes    = np.asarray(mesh_obj.node,    dtype=np.float64)[:, :2]  # (n_nodes, 2)
    elements = np.asarray(mesh_obj.element, dtype=np.int32)           # (n_elements, 3)

    ds_flat = np.asarray(ds, dtype=np.float64).ravel()

    n_nodes    = nodes.shape[0]
    n_elements = elements.shape[0]

    if ds_flat.shape[0] == n_elements:
        # Per-element values — use element centroids as scatter points.
        # Centroid = arithmetic mean of the three corner positions.
        scatter_pts = nodes[elements].mean(axis=1)             # (n_elements, 2)
    elif ds_flat.shape[0] == n_nodes:
        # Per-node values — pyEIT's BP solver returns (n_nodes,) because
        # its back-projection matrix H has shape (n_nodes, n_meas_tot).
        # Use node coordinates directly as scatter points.
        scatter_pts = nodes                                    # (n_nodes, 2)
    else:
        raise ValueError(
            f"rasterize: ds length {ds_flat.shape[0]} matches neither "
            f"n_elements ({n_elements}) nor n_nodes ({n_nodes})."
        )

    # ------------------------------------------------------------------
    # Step 2 — Build the 256×256 pixel grid over the actual mesh extent.
    #
    # IMPORTANT: the KTC mesh is in physical units (metres).  Mesh_sparse.mat
    # stores nodes in the range [-0.115, 0.115] m (tank radius ≈ 0.115 m),
    # NOT the normalised [-1, 1] range.  Hardcoding [-1, 1] would make the
    # mesh nodes occupy only ~1% of the grid, filling the rest with
    # fill_value=0 from griddata — producing an almost entirely zero image.
    #
    # Fix: derive the grid bounds from the actual scatter-point extent and
    # add a small margin so boundary nodes are never clipped.
    # ------------------------------------------------------------------
    grid_size = 256

    margin = 0.02 * (scatter_pts[:, 0].max() - scatter_pts[:, 0].min())
    x_min = scatter_pts[:, 0].min() - margin
    x_max = scatter_pts[:, 0].max() + margin
    y_min = scatter_pts[:, 1].min() - margin
    y_max = scatter_pts[:, 1].max() + margin

    xi = np.linspace(x_min, x_max, grid_size)   # x pixel centres
    yi = np.linspace(y_min, y_max, grid_size)   # y pixel centres
    gx, gy = np.meshgrid(xi, yi)                 # each (256, 256)

    # ------------------------------------------------------------------
    # Step 3 — Scattered-data interpolation.
    # griddata(points, values, (xi, yi), method='linear')
    #   points : (n_pts, 2) scatter coordinates (centroids or nodes)
    #   values : (n_pts,)   ds values at each scatter point
    #   result : (256, 256) interpolated image
    # Pixels outside the convex hull of scatter points get fill_value=0.0.
    # ------------------------------------------------------------------
    grid = griddata(
        scatter_pts,         # scatter points (centroids or nodes)
        ds_flat,             # scatter values
        (gx, gy),            # query grid
        method="linear",
        fill_value=0.0,
    )                        # (256, 256) float64

    # ------------------------------------------------------------------
    # Step 4 — Circular mask: zero out pixels outside the tank boundary.
    # The tank is a circle of radius r_tank centred at the origin.
    # We estimate r_tank from the maximum node distance from centre.
    # ------------------------------------------------------------------
    r_tank = np.sqrt((scatter_pts[:, 0] ** 2 + scatter_pts[:, 1] ** 2).max())
    outside_tank = (gx ** 2 + gy ** 2) > r_tank ** 2
    grid[outside_tank] = 0.0

    return grid.astype(np.float32)           # (256, 256) float32
