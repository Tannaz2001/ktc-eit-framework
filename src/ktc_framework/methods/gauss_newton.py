"""KTC-calibrated Gauss-Newton style reconstruction.

Final update:
- Uses real Inj/Injref and Mpat from the DataBatch.
- Keeps only finite voltage differences, because KTC levels 2-7 may contain NaN values.
- Prevents NaN from entering sigma_elem, sigma_map, and segmentation.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.interpolate import griddata
from skimage.filters import threshold_multiotsu

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch


def _has_key(obj, key: str) -> bool:
    if hasattr(obj, key):
        return True
    try:
        return key in obj
    except TypeError:
        return False


def _get_key(obj, key: str):
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        if key in obj:
            return obj[key]
    except TypeError:
        pass
    return None


def _pattern_pairs(pattern: Optional[np.ndarray]) -> np.ndarray:
    if pattern is None:
        return np.zeros((0, 2), dtype=int)

    p = np.asarray(pattern, dtype=float)

    if p.ndim != 2:
        return np.zeros((0, 2), dtype=int)

    if p.shape[0] != 32 and p.shape[1] == 32:
        p = p.T

    pairs = []

    for col_i in range(p.shape[1]):
        col = p[:, col_i]
        pos = np.where(col > 0)[0]
        neg = np.where(col < 0)[0]

        if pos.size > 0 and neg.size > 0:
            pairs.append([int(pos[0]), int(neg[0])])

    return np.asarray(pairs, dtype=int)


def _reshape_ktc_vector(v: np.ndarray, n_inj: int, n_meas: int) -> np.ndarray:
    flat = np.asarray(v, dtype=float).ravel()
    total = n_inj * n_meas

    if flat.size < total:
        padded = np.full(total, np.nan, dtype=float)
        padded[: flat.size] = flat
        flat = padded

    return flat[:total].reshape(n_inj, n_meas)


def _segment_ktc(sigma: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(
        np.asarray(sigma, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if float(np.max(x) - np.min(x)) < 1e-12:
        return np.zeros(x.shape, dtype=int)

    try:
        t1, t2 = threshold_multiotsu(x.ravel(), classes=3)

        low = x < t1
        mid = (x >= t1) & (x <= t2)
        high = x > t2

        counts = [int(low.sum()), int(mid.sum()), int(high.sum())]
        bg = int(np.argmax(counts))

        labels = np.zeros(x.shape, dtype=int)

        if bg == 0:
            labels[mid] = 1
            labels[high] = 2
        elif bg == 1:
            labels[low] = 1
            labels[high] = 2
        else:
            labels[low] = 1
            labels[mid] = 2

        return labels.astype(int)

    except Exception:
        abs_map = np.abs(x)
        threshold = np.percentile(abs_map, 92)

        labels = np.zeros(x.shape, dtype=int)
        labels[abs_map > threshold] = 1

        return labels.astype(int)


@register
class GaussNewton(MethodPlugin):
    """Dataset-aligned regularized Gauss-Newton reconstruction for KTC EIT."""

    _NODE_KEYS = ["g", "Node", "node", "p", "pts"]
    _ELEMENT_KEYS = ["H", "Element", "element", "t", "tri"]

    def __init__(self, lamb: float = 100.0) -> None:
        self.lamb = float(lamb)

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        if batch.mesh is None:
            warnings.warn(
                "GaussNewton: no mesh available; returning zeros.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((256, 256), dtype=int)

        try:
            nodes, elements = self._parse_mesh(batch.mesh)
            electrode_nodes = self._electrode_positions(batch.mesh, nodes)

            inj_pairs = _pattern_pairs(batch.injection_patterns)
            meas_pairs = _pattern_pairs(getattr(batch, "measurement_patterns", None))

            if inj_pairs.shape[0] == 0:
                raise ValueError("No valid Inj current pairs found.")

            if meas_pairs.shape[0] == 0:
                warnings.warn(
                    "GaussNewton: no valid Mpat found; using adjacent voltage pairs.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                meas_pairs = np.column_stack(
                    [np.arange(31), np.arange(1, 32)]
                ).astype(int)

            n_inj = inj_pairs.shape[0]
            n_meas = meas_pairs.shape[0]

            J = self._build_sensitivity(
                nodes,
                elements,
                electrode_nodes,
                inj_pairs,
                meas_pairs,
            )

            y, mask = self._prepare_difference(
                batch.voltages,
                batch.reference_voltages,
                n_inj,
                n_meas,
            )

            if not mask.any():
                warnings.warn(
                    f"GaussNewton: no finite voltage differences for "
                    f"level={batch.level}, sample={batch.sample_id}; returning zeros.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return np.zeros((256, 256), dtype=int)

            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            Jm = J[mask]
            ym = y[mask]

            Jm = Jm / (np.linalg.norm(Jm, axis=1, keepdims=True) + 1e-12)

            A = Jm @ Jm.T
            A.flat[:: A.shape[0] + 1] += self.lamb

            coeff = np.linalg.solve(A, ym)
            sigma_elem = Jm.T @ coeff

            sigma_elem = np.nan_to_num(
                sigma_elem,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if np.std(sigma_elem) > 1e-12:
                sigma_elem = sigma_elem - np.median(sigma_elem)
                sigma_elem = sigma_elem / (np.std(sigma_elem) + 1e-12)

            sigma_map = self._interpolate_to_grid(
                nodes,
                elements,
                sigma_elem,
                size=256,
            )

            sigma_map = np.nan_to_num(
                sigma_map,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            print(
                f"[DEBUG] {self.__class__.__name__} "
                f"level={batch.level} sample={batch.sample_id} "
                f"sigma min={sigma_map.min():.6f}, "
                f"max={sigma_map.max():.6f}, "
                f"std={sigma_map.std():.6f}"
            )

            labels = _segment_ktc(sigma_map)
            self.validate_output(labels)

            return labels

        except Exception as exc:
            warnings.warn(
                f"GaussNewton failed after KTC alignment ({exc!r}); returning zeros.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((256, 256), dtype=int)

    def _parse_mesh(self, mesh_data) -> tuple[np.ndarray, np.ndarray]:
        node_key = next((key for key in self._NODE_KEYS if _has_key(mesh_data, key)), None)
        elem_key = next((key for key in self._ELEMENT_KEYS if _has_key(mesh_data, key)), None)

        if node_key is None or elem_key is None:
            raise ValueError("Mesh missing node or element arrays.")

        nodes = np.asarray(_get_key(mesh_data, node_key), dtype=np.float64)
        elements = np.asarray(_get_key(mesh_data, elem_key), dtype=np.int32)

        if elements.min() >= 1:
            elements = elements - 1

        return nodes, elements

    @staticmethod
    def _prepare_difference(
        voltages: np.ndarray,
        reference_voltages: Optional[np.ndarray],
        n_inj: int,
        n_meas: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        v1 = _reshape_ktc_vector(voltages, n_inj, n_meas).reshape(-1)

        if reference_voltages is None:
            warnings.warn(
                "GaussNewton: no reference voltages; using mean subtraction.",
                RuntimeWarning,
                stacklevel=2,
            )
            v0 = np.full_like(v1, float(np.nanmean(v1)))
        else:
            v0 = _reshape_ktc_vector(reference_voltages, n_inj, n_meas).reshape(-1)

        y = (v1 - v0) / (np.abs(v0) + 1e-6)

        mask = np.isfinite(y)

        if mask.any():
            y_finite = y[mask]
            y[mask] = y_finite - np.median(y_finite)
            y[mask] = y[mask] / (np.std(y[mask]) + 1e-12)

        return y.astype(np.float64), mask

    @staticmethod
    def _build_sensitivity(
        nodes: np.ndarray,
        elements: np.ndarray,
        electrode_nodes: np.ndarray,
        inj_pairs: np.ndarray,
        meas_pairs: np.ndarray,
    ) -> np.ndarray:
        centroids = np.mean(nodes[elements], axis=1)
        electrode_xy = nodes[electrode_nodes]

        drive = np.stack(
            [
                GaussNewton._point_gradient(centroids, electrode_xy, a, b)
                for a, b in inj_pairs
            ]
        )

        meas = np.stack(
            [
                GaussNewton._point_gradient(centroids, electrode_xy, a, b)
                for a, b in meas_pairs
            ]
        )

        sensitivity = -np.einsum("ime,jme->ijm", drive, meas)

        J = sensitivity.reshape(
            inj_pairs.shape[0] * meas_pairs.shape[0],
            centroids.shape[0],
        )

        J = J / (np.linalg.norm(J, axis=0, keepdims=True) + 1e-12)

        return J.astype(np.float64)

    @staticmethod
    def _point_gradient(
        points: np.ndarray,
        electrodes_xy: np.ndarray,
        a: int,
        b: int,
    ) -> np.ndarray:
        pa = electrodes_xy[int(a)]
        pb = electrodes_xy[int(b)]

        ra = points - pa
        rb = points - pb

        eps = 1e-3

        return (
            ra / (np.sum(ra * ra, axis=1)[:, None] + eps)
            - rb / (np.sum(rb * rb, axis=1)[:, None] + eps)
        )

    @staticmethod
    def _electrode_positions(mesh_data, nodes: np.ndarray) -> np.ndarray:
        n_nodes = nodes.shape[0]

        for key in ["elfaces", "ElFaces", "el_faces", "elFaces"]:
            elfaces = _get_key(mesh_data, key)

            if elfaces is None:
                continue

            try:
                electrode_node_list = []

                for i in range(len(elfaces)):
                    edge = np.asarray(elfaces[i], dtype=np.int32)
                    edge_nodes = edge.ravel() - 1
                    edge_nodes = edge_nodes[
                        (edge_nodes >= 0) & (edge_nodes < n_nodes)
                    ]

                    if edge_nodes.size == 0:
                        continue

                    centre = nodes[np.unique(edge_nodes)].mean(axis=0)
                    nearest = int(np.argmin(np.linalg.norm(nodes - centre, axis=1)))
                    electrode_node_list.append(nearest)

                electrode_nodes = np.asarray(electrode_node_list, dtype=np.int32)

                if electrode_nodes.shape[0] == 32:
                    return electrode_nodes

            except Exception as exc:
                warnings.warn(
                    f"GaussNewton: elfaces parse failed ({exc!r}); "
                    "using fallback electrode positions.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                break

        return np.round(np.linspace(0, n_nodes - 1, 32)).astype(np.int32)

    @staticmethod
    def _interpolate_to_grid(
        nodes: np.ndarray,
        elements: np.ndarray,
        values: np.ndarray,
        size: int = 256,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).ravel()
        centroids = np.mean(nodes[elements], axis=1)

        if values.shape[0] == nodes.shape[0]:
            values = np.mean(values[elements], axis=1)
        elif values.shape[0] != elements.shape[0]:
            warnings.warn(
                f"_interpolate_to_grid: value length {values.shape[0]} does not "
                f"match nodes {nodes.shape[0]} or elements {elements.shape[0]}. "
                "Returning zero grid.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((size, size), dtype=np.float64)

        radius = (nodes[:, 0].max() - nodes[:, 0].min()) / 2.0
        pixwidth = 2.0 * radius / size
        pix = np.linspace(-radius + pixwidth / 2.0, radius - pixwidth / 2.0, size)

        gx, gy = np.meshgrid(pix, pix)

        grid = griddata(
            centroids,
            values,
            (gx, gy),
            method="linear",
            fill_value=0.0,
        )

        return np.flipud(grid).astype(np.float64)