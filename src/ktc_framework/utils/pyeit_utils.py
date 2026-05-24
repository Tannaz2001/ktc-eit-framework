"""Shared pyEIT utilities.

Provides two helpers used by BackProjection and GaussNewton:

    build_pyeit_mesh  — converts a scipy.io mat_struct (Mesh_sparse.mat)
                        or an already-built PyEITMesh into a PyEITMesh.

    interpolate_to_grid — maps a node-based or element-based reconstruction
                          vector onto a 256×256 NumPy float64 array using
                          scipy linear griddata.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.interpolate import griddata


def build_pyeit_mesh(mesh_data) -> Optional[object]:
    """Build a ``PyEITMesh`` from *mesh_data*.

    Parameters
    ----------
    mesh_data:
        One of:
        * ``None``            → returns ``None``.
        * ``PyEITMesh``       → returned unchanged.
        * scipy mat_struct    → parsed using ``.g``, ``.H``, ``.elfaces``.
        * ``dict``            → parsed using keys ``'g'``, ``'H'``, ``'elfaces'``.

    Returns
    -------
    PyEITMesh | None
        A valid pyEIT mesh, or ``None`` if construction fails.
    """
    if mesh_data is None:
        return None

    try:
        from pyeit.mesh.wrapper import PyEITMesh  # type: ignore[import]
    except ImportError:
        warnings.warn("pyeit not installed — cannot build mesh.", RuntimeWarning, stacklevel=2)
        return None

    # Already the right type
    if isinstance(mesh_data, PyEITMesh):
        return mesh_data

    # ── extract arrays from mat_struct or dict ─────────────────────────────
    try:
        if hasattr(mesh_data, "g"):          # scipy mat_struct
            g       = mesh_data.g            # (n_nodes, 2)  float64
            H       = mesh_data.H            # (n_elem,  3)  uint16 — 1-indexed
            elfaces = mesh_data.elfaces      # (32,)  each entry shape (k, 2)
        elif isinstance(mesh_data, dict):    # plain dict
            g       = mesh_data["g"]
            H       = mesh_data["H"]
            elfaces = mesh_data["elfaces"]
        else:
            warnings.warn(
                f"Unrecognised mesh type {type(mesh_data).__name__} — skipping.",
                RuntimeWarning, stacklevel=2,
            )
            return None
    except (KeyError, AttributeError) as exc:
        warnings.warn(f"Mesh missing required field: {exc}", RuntimeWarning, stacklevel=2)
        return None

    # Convert to 0-indexed int32 triangles
    tri = H.astype(np.int32) - 1            # MATLAB is 1-indexed

    # One representative boundary node per electrode
    # elfaces[i] is shape (n_segs, 2) — take the first node of the first segment
    try:
        el_pos = np.array(
            [int(elfaces[i][0, 0]) - 1 for i in range(len(elfaces))],
            dtype=np.int32,
        )
    except Exception as exc:
        warnings.warn(f"Could not extract electrode positions: {exc}", RuntimeWarning, stacklevel=2)
        return None

    # Homogeneous unit conductivity background
    perm = np.ones(tri.shape[0], dtype=np.float64)

    try:
        mesh_obj = PyEITMesh(
            node=g,
            element=tri,
            perm=perm,
            el_pos=el_pos,
            ref_node=0,
        )
        return mesh_obj
    except Exception as exc:
        warnings.warn(f"PyEITMesh construction failed: {exc}", RuntimeWarning, stacklevel=2)
        return None


def interpolate_to_grid(ds: np.ndarray, mesh_obj) -> np.ndarray:
    """Interpolate a reconstruction vector onto a 256×256 float64 grid.

    Parameters
    ----------
    ds:
        1-D reconstruction vector, either:
        * node-based  — length ``n_nodes``  (BackProjection output)
        * element-based — length ``n_elements`` (GaussNewton output)
    mesh_obj:
        A ``PyEITMesh`` instance.

    Returns
    -------
    np.ndarray
        Shape ``(256, 256)``, float64.  Pixels outside the mesh convex hull
        are filled with ``0.0``.
    """
    pts = mesh_obj.node[:, :2]          # (n_nodes, 2)
    tri = mesh_obj.element              # (n_elem,  3)

    n_nodes = pts.shape[0]
    n_elem  = tri.shape[0]

    # ── choose interpolation coordinates ──────────────────────────────────
    if ds.shape[0] == n_nodes:
        coords = pts
        values = ds.ravel()
    elif ds.shape[0] == n_elem:
        # Element centroids
        coords = pts[tri].mean(axis=1)  # (n_elem, 2)
        values = ds.ravel()
    else:
        warnings.warn(
            f"ds length {ds.shape[0]} matches neither nodes ({n_nodes}) "
            f"nor elements ({n_elem}). Returning zero grid.",
            RuntimeWarning, stacklevel=2,
        )
        return np.zeros((256, 256), dtype=np.float64)

    # ── build regular 256×256 grid over the mesh bounding box ─────────────
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()

    xi = np.linspace(xmin, xmax, 256)
    yi = np.linspace(ymin, ymax, 256)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    grid = griddata(
        coords, values,
        (xi_grid, yi_grid),
        method="linear",
        fill_value=0.0,
    )
    return grid.astype(np.float64)
