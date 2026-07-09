"""KTC protocol utilities for calibrated reconstruction."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.interpolate import griddata
from skimage.filters import threshold_multiotsu


def pattern_pairs(pattern: np.ndarray) -> np.ndarray:
    """Return electrode source and sink pairs from 32 x N pattern matrix."""
    p = np.asarray(pattern, dtype=float)

    if p.ndim != 2:
        return np.zeros((0, 2), dtype=int)

    if p.shape[0] != 32 and p.shape[1] == 32:
        p = p.T

    pairs = []

    for j in range(p.shape[1]):
        col = p[:, j]
        pos = np.where(col > 0)[0]
        neg = np.where(col < 0)[0]

        if pos.size and neg.size:
            pairs.append([int(pos[0]), int(neg[0])])

    return np.asarray(pairs, dtype=int)


def ktc_vincl(level: int, inj: np.ndarray) -> np.ndarray:
    """Apply KTC level masking like MATLAB vincl logic."""
    inj = np.asarray(inj, dtype=float)

    if inj.shape[0] != 32 and inj.shape[1] == 32:
        inj = inj.T

    n_inj = inj.shape[1]
    n_meas = 31

    vincl = np.ones((n_meas, n_inj), dtype=bool)

    rmind = np.arange(max(0, 2 * (int(level) - 1)), dtype=int)

    if rmind.size:
        for ii in range(n_inj):
            if np.any(np.abs(inj[rmind, ii]) > 1e-12):
                vincl[:, ii] = False

            valid_rmind = rmind[rmind < n_meas]
            vincl[valid_rmind, ii] = False

    return vincl.T.reshape(-1)


def reshape_ktc_vector(v: np.ndarray, n_inj: int = 76, n_meas: int = 31) -> np.ndarray:
    """Reshape KTC voltage vector to 76 x 31."""
    flat = np.asarray(v, dtype=float).ravel()
    total = n_inj * n_meas

    if flat.size < total:
        flat = np.pad(flat, (0, total - flat.size))

    return flat[:total].reshape(n_inj, n_meas)


def mesh_arrays(mesh_data):
    """Extract nodes and elements from Mesh_sparse.mat."""
    nodes = getattr(mesh_data, "g", None)
    elems = getattr(mesh_data, "H", None)

    if nodes is None or elems is None:
        raise ValueError("Mesh must contain g and H fields")

    nodes = np.asarray(nodes, dtype=float)
    elems = np.asarray(elems, dtype=int)

    if elems.min() >= 1:
        elems = elems - 1

    return nodes, elems


def electrode_nodes(mesh_data, nodes: np.ndarray) -> np.ndarray:
    """Find one representative mesh node for each electrode."""
    elfaces = getattr(mesh_data, "elfaces", None)

    if elfaces is None:
        warnings.warn(
            "Mesh has no elfaces; using boundary fallback.",
            RuntimeWarning,
            stacklevel=2,
        )

        theta = np.arctan2(nodes[:, 1], nodes[:, 0])
        radius = np.linalg.norm(nodes, axis=1)
        boundary = np.where(radius > np.quantile(radius, 0.98))[0]

        targets = np.linspace(np.pi, -np.pi, 32, endpoint=False)

        return np.array([
            boundary[np.argmin(np.abs(np.angle(np.exp(1j * (theta[boundary] - t)))))]
            for t in targets
        ])

    out = []

    for i in range(len(elfaces)):
        edge_nodes = np.asarray(elfaces[i], dtype=int).ravel() - 1
        edge_nodes = edge_nodes[(edge_nodes >= 0) & (edge_nodes < len(nodes))]

        if edge_nodes.size == 0:
            continue

        centre = nodes[np.unique(edge_nodes)].mean(axis=0)
        out.append(int(np.argmin(np.linalg.norm(nodes - centre, axis=1))))

    if len(out) != 32:
        raise ValueError(f"Expected 32 electrodes, got {len(out)}")

    return np.asarray(out, dtype=int)


def point_gradient(points: np.ndarray, electrodes_xy: np.ndarray, a: int, b: int) -> np.ndarray:
    """Approximate point-electrode potential gradient."""
    pa = electrodes_xy[a]
    pb = electrodes_xy[b]

    ra = points - pa
    rb = points - pb

    eps = 1e-3

    return (
        ra / (np.sum(ra * ra, axis=1)[:, None] + eps)
        - rb / (np.sum(rb * rb, axis=1)[:, None] + eps)
    )


def build_sensitivity(mesh_data, inj: np.ndarray, mpat: Optional[np.ndarray]):
    """Build KTC-aligned sensitivity matrix.

    Output order is:
    76 injections x 31 voltage measurements = 2356 rows.
    """
    nodes, elems = mesh_arrays(mesh_data)
    centroids = nodes[elems].mean(axis=1)

    el_nodes = electrode_nodes(mesh_data, nodes)
    electrodes_xy = nodes[el_nodes]

    inj_pairs = pattern_pairs(inj)

    if inj_pairs.shape[0] == 0:
        raise ValueError("No valid Inj current pairs found")

    if mpat is None:
        meas_pairs = np.column_stack([np.arange(31), np.arange(1, 32)]).astype(int)
    else:
        meas_pairs = pattern_pairs(mpat)

        if meas_pairs.shape[0] == 0:
            meas_pairs = np.column_stack([np.arange(31), np.arange(1, 32)]).astype(int)

    drive = np.stack([
        point_gradient(centroids, electrodes_xy, a, b)
        for a, b in inj_pairs
    ])

    meas = np.stack([
        point_gradient(centroids, electrodes_xy, a, b)
        for a, b in meas_pairs
    ])

    sensitivity = -np.einsum("ime,jme->ijm", drive, meas)

    J = sensitivity.reshape(
        inj_pairs.shape[0] * meas_pairs.shape[0],
        centroids.shape[0],
    )

    J = J / (np.linalg.norm(J, axis=0, keepdims=True) + 1e-12)

    return J.astype(np.float64), nodes, elems, centroids


def prepare_difference(v1: np.ndarray, v0: Optional[np.ndarray], inj: np.ndarray, level: int):
    """Prepare normalized voltage difference and KTC vincl mask."""
    y1 = reshape_ktc_vector(v1).reshape(-1)

    if v0 is None:
        warnings.warn(
            "No reference voltages; using mean subtraction fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        y0 = np.full_like(y1, float(np.mean(y1)))
    else:
        y0 = reshape_ktc_vector(v0).reshape(-1)

    y = (y1 - y0) / (np.abs(y0) + 1e-6)

    y = y - np.median(y)
    y = y / (np.std(y) + 1e-12)

    mask = ktc_vincl(level, inj)

    return y.astype(np.float64), mask


def interpolate_elements_to_ktc_grid(
    nodes: np.ndarray,
    elems: np.ndarray,
    values: np.ndarray,
    size: int = 256,
) -> np.ndarray:
    """Interpolate element values to KTC 256 x 256 grid."""
    centroids = nodes[elems].mean(axis=1)

    radius = (nodes[:, 0].max() - nodes[:, 0].min()) / 2.0
    pixwidth = 2.0 * radius / size

    pix = np.linspace(
        -radius + pixwidth / 2.0,
        radius - pixwidth / 2.0,
        size,
    )

    X, Y = np.meshgrid(pix, pix)

    grid = griddata(
        centroids,
        np.asarray(values, dtype=float).ravel(),
        (X, Y),
        method="linear",
        fill_value=0.0,
    )

    return np.flipud(grid).astype(np.float64)


def segment_ktc(img: np.ndarray) -> np.ndarray:
    """Convert reconstruction image to labels 0, 1, 2."""
    x = np.nan_to_num(
        np.asarray(img, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if float(x.max() - x.min()) < 1e-12:
        return np.zeros(x.shape, dtype=int)

    try:
        t1, t2 = threshold_multiotsu(x.ravel(), classes=3)
    except (ValueError, RuntimeError):
        return np.zeros(x.shape, dtype=int)

    low = x < t1
    mid = (x >= t1) & (x <= t2)
    high = x > t2

    bg = int(np.argmax([low.sum(), mid.sum(), high.sum()]))

    out = np.zeros(x.shape, dtype=int)

    if bg == 0:
        out[mid | high] = 2
    elif bg == 1:
        out[low] = 1
        out[high] = 2
    else:
        out[low | mid] = 1

    return out