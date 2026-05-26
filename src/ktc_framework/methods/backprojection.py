"""BackProjection reconstruction method for EIT -- KTC 2023 dataset.

Three bugs fixed vs the original implementation
-------------------------------------------------
Bug 1 -- Protocol mismatch
    Old code fed delta_v[:928] into pyEIT's adjacent-pair BP solver.
    The KTC dataset has 76 injection patterns x 31 measurements = 2356
    voltages, which do NOT correspond to pyEIT's 32-pattern adjacent-pair
    ordering.  Feeding the wrong measurements to the wrong patterns produces
    physically meaningless (visually random) reconstructions.

    Fix: extract the actual 76 injection pairs from batch.injection_patterns
    (shape 32 x 76), reshape delta_v to (76, 31), and run a direct
    back-projection that accumulates per-element sensitivity-weighted
    conductivity change for every (injection, measurement) pair.
    The sensitivity is approximated as 1 / (d_inj * d_meas) -- inverse
    product of element-to-injection-midpoint and element-to-measurement-
    midpoint distances -- which is a standard BP approximation.

Bug 2 -- Electrode node mapping
    Old code did elfaces[i][0, 0] - 1, which always returned node 0
    because it read the column index, not the node value.

    elfaces[i] is a (K, 2) uint8 array of boundary-edge node pairs
    (1-indexed, MATLAB convention).  Each pair [a, b] is one boundary edge.

    Fix: take the midpoint of the first boundary edge in node-space, then
    find the nearest mesh node to that midpoint.  This gives the correct
    representative node for each of the 32 electrodes.

Bug 3 -- Forced three-class segmentation
    Old segment() always applied double-Otsu regardless of whether a
    conductive inclusion was actually present, creating a spurious third
    class from reconstruction noise.

    Fix: _segment_adaptive() checks the range of the sigma map (returns
    zeros for flat maps) and only assigns the conductive class (2) when
    the values above the first Otsu threshold have a clear second peak
    that is well-separated from the threshold.

Algorithm (corrected)
---------------------
1. Difference imaging  : delta_v = voltages - reference_voltages
                         (mean fallback when ref absent).
2. Mesh parsing        : key-safe attribute / dict access for mat_struct.
3. Electrode positions : midpoint-of-boundary-edge -> nearest node (Bug 2).
4. Build ex_mat        : 76 KTC injection pairs from batch.injection_patterns
                         instead of pyEIT's default adjacent-pair (Bug 1).
5. Direct BP           : vectorised sensitivity accumulation (Bug 1).
6. Grid interpolation  : element-centroid griddata -> 256x256 float grid.
7. Adaptive segment    : flat map -> zeros; suppress conductive class when
                         no clear second peak detected (Bug 3).
8. Validate & return   : (256, 256) int array.  Never raises.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from skimage.filters import threshold_otsu

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _has_key(obj, key: str) -> bool:
    """True if *obj* exposes *key* as an attribute or dict entry.

    Works transparently with scipy mat_struct (attribute access) and
    plain dict (key access).
    """
    if hasattr(obj, key):
        return True
    try:
        return key in obj
    except TypeError:
        return False


def _get_key(obj, key: str) -> Optional[np.ndarray]:
    """Retrieve *key* from *obj* via attribute or dict access.

    Returns None if the key is absent.
    """
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        if key in obj:
            return obj[key]
    except TypeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Adaptive segmentation  (Bug 3 fix)
# ---------------------------------------------------------------------------

def _segment_adaptive(sigma: np.ndarray) -> np.ndarray:
    """Convert a float conductivity map to discrete labels {0, 1, 2}.

    Bug 3 fix: the old double-Otsu always created three classes even when
    the ground truth has only two (e.g. Level 1 -- one resistive blob, no
    conductive).  The conductive class is now only assigned when a clear
    second peak exists above the first Otsu threshold.

    Parameters
    ----------
    sigma : np.ndarray
        Shape (256, 256) float conductivity map.

    Returns
    -------
    np.ndarray
        Shape (256, 256) int, labels in {0, 1, 2}.
        0 = water/background, 1 = resistive, 2 = conductive.
    """
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # Bug 3 fix: return zeros immediately for flat / degenerate maps
    sigma_range = float(sigma.max() - sigma.min())
    if sigma_range < 1e-6:
        return np.zeros(sigma.shape, dtype=int)

    # First Otsu threshold: background vs inclusions
    t1 = threshold_otsu(sigma)
    labels = np.zeros_like(sigma, dtype=int)
    labels[sigma > t1] = 1          # tentatively resistive

    # Bug 3 fix: only promote to conductive (class 2) when the region above
    # t1 has a clear second peak well-separated from t1.
    above = sigma[sigma > t1]
    if above.size > 10:
        t2 = threshold_otsu(above)
        # Accept conductive class only if the upper-tier peak is
        # at least 5 % of the full range above the second threshold.
        if (above.max() - t2) > 0.05 * sigma_range:
            labels[sigma > t2] = 2  # conductive

    return labels


# ---------------------------------------------------------------------------
# BackProjection
# ---------------------------------------------------------------------------

@register
class BackProjection(MethodPlugin):
    """Back-projection EIT reconstruction for the KTC 2023 dataset.

    Uses the actual KTC 76-pattern injection protocol derived from
    ``batch.injection_patterns`` (32 x 76 matrix) and correctly maps
    electrode positions from the finite-element mesh elfaces field.

    Falls back to a reproducibly-seeded random segmentation only when
    the mesh is absent or completely unparseable.
    """

    # Candidate key names in Mesh_sparse.mat (mat_struct or dict)
    _NODE_KEYS    = ["g", "Node", "node", "p", "pts"]
    _ELEMENT_KEYS = ["H", "Element", "element", "t", "tri"]
    _ELFACE_KEYS  = ["elfaces", "ElFaces", "el_faces", "elFaces"]

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Return a (256, 256) segmentation label array.

        Labels: 0 = water, 1 = resistive, 2 = conductive.

        Parameters
        ----------
        batch : DataBatch
            Populated by BatchRunner.  ``mesh`` and ``reference_voltages``
            drive physics-based reconstruction when present.

        Returns
        -------
        np.ndarray
            Shape (256, 256), dtype int, values in {0, 1, 2}.
        """
        # ── Step 1: Difference imaging ─────────────────────────────────────
        if batch.reference_voltages is not None:
            delta_v = (batch.voltages - batch.reference_voltages).ravel().astype(np.float64)
        else:
            warnings.warn(
                "BackProjection: no reference voltages -- "
                "falling back to mean subtraction. Quality will be poor.",
                RuntimeWarning, stacklevel=2,
            )
            delta_v = (batch.voltages - np.mean(batch.voltages)).ravel().astype(np.float64)

        # ── Step 2: Parse mesh ─────────────────────────────────────────────
        if batch.mesh is None:
            return self._random_fallback(batch)

        node_key = next((k for k in self._NODE_KEYS    if _has_key(batch.mesh, k)), None)
        elem_key = next((k for k in self._ELEMENT_KEYS if _has_key(batch.mesh, k)), None)

        if node_key is None or elem_key is None:
            warnings.warn(
                "BackProjection: mesh missing node/element arrays.",
                RuntimeWarning, stacklevel=2,
            )
            return self._random_fallback(batch)

        nodes    = np.asarray(_get_key(batch.mesh, node_key),    dtype=np.float64)  # (N, 2)
        elements = np.asarray(_get_key(batch.mesh, elem_key),    dtype=np.int32)    # (M, 3)

        # MATLAB uses 1-based indexing; convert to 0-based for numpy
        if elements.min() >= 1:
            elements = elements - 1

        # ── Step 3: Electrode positions  (Bug 2 fix) ──────────────────────
        el_pos = self._electrode_positions(batch.mesh, nodes)

        # ── Step 4: Build ex_mat from KTC Inj matrix  (Bug 1 fix) ─────────
        if batch.injection_patterns is not None:
            ex_mat = self._build_ex_mat(batch.injection_patterns)
        else:
            ex_mat = np.zeros((0, 2), dtype=int)

        try:
            if ex_mat.shape[0] == 0:
                raise ValueError("No valid injection patterns extracted.")

            # ── Step 5: Direct back-projection  (Bug 1 fix) ───────────────
            sigma_elem = self._direct_backproject(
                nodes, elements, el_pos, ex_mat, delta_v,
                n_meas_per_inj=31,
            )

            # ── Step 6: Interpolate to 256x256 pixel grid ─────────────────
            sigma_map = self._interpolate_to_grid(nodes, elements, sigma_elem, size=256)

            # ── Step 7: Adaptive segmentation  (Bug 3 fix) ────────────────
            labels = _segment_adaptive(sigma_map)
            self.validate_output(labels)
            return labels

        except Exception as exc:
            warnings.warn(
                f"BackProjection failed ({exc!r}) -- using random fallback.",
                RuntimeWarning, stacklevel=2,
            )
            return self._random_fallback(batch)

    # ── private: build injection matrix  (Bug 1) ────────────────────────────

    @staticmethod
    def _build_ex_mat(injection_patterns: np.ndarray) -> np.ndarray:
        """Extract (n_inj, 2) source/sink pairs from the KTC Inj matrix.

        Bug 1 fix: the KTC injection matrix has shape (32, 76).  Each column
        encodes one injection pattern: +1 marks the source electrode and -1
        marks the sink electrode.  We extract these pairs to drive the BP
        directly, rather than assuming pyEIT's adjacent-pair ordering.

        Parameters
        ----------
        injection_patterns : np.ndarray
            Shape (32, 76) -- KTC injection matrix from the .mat file.

        Returns
        -------
        np.ndarray
            Shape (n_valid, 2) int, 0-indexed [source, sink] electrode pairs.
        """
        inj = np.asarray(injection_patterns, dtype=np.float64)  # (32, 76)
        ex_list = []
        for col_i in range(inj.shape[1]):          # iterate over 76 patterns
            col = inj[:, col_i]
            src = np.where(col > 0.5)[0]           # electrode with +1 current
            dst = np.where(col < -0.5)[0]          # electrode with -1 current
            if src.size > 0 and dst.size > 0:
                ex_list.append([int(src[0]), int(dst[0])])
        if not ex_list:
            return np.zeros((0, 2), dtype=int)
        return np.array(ex_list, dtype=int)        # (76, 2)

    # ── private: direct back-projection  (Bug 1) ────────────────────────────

    @staticmethod
    def _direct_backproject(
        nodes:          np.ndarray,
        elements:       np.ndarray,
        el_pos:         np.ndarray,
        ex_mat:         np.ndarray,
        delta_v:        np.ndarray,
        n_meas_per_inj: int = 31,
    ) -> np.ndarray:
        """Direct back-projection using the actual KTC measurement protocol.

        Bug 1 fix: the KTC dataset delivers 76 injections x 31 measurements
        = 2356 voltages.  The old code sliced delta_v[:928] and fed it to
        pyEIT's adjacent-pair solver -- mixing up injection patterns and
        measurement pairs entirely.

        This implementation reshapes delta_v to (76, 31) and accumulates,
        for every (injection i, measurement j) pair:

            sigma_elem += delta_v[i, j] * sensitivity(element, i, j)

        where sensitivity is approximated as the inverse product of element-
        centroid distances to the injection dipole midpoint and the
        measurement dipole midpoint.

        KTC measurement convention (inferred from the 31-per-injection count
        and 32-electrode setup): measurements are the 31 adjacent voltage
        pairs (el_0, el_1), (el_1, el_2), ..., (el_30, el_31) -- the same
        fixed set for every injection pattern.

        The inner double-sum is fully vectorised:
            sigma (M,) = sensitivity (M, 76*31) @ dv_mat.ravel() (76*31,)

        Memory: ~3073 x 76 x 31 x 8 bytes ~ 58 MB -- acceptable.

        Parameters
        ----------
        nodes, elements :
            Mesh geometry (0-indexed).
        el_pos :
            (32,) 0-indexed electrode node indices.
        ex_mat :
            (76, 2) 0-indexed [source, sink] pairs.
        delta_v :
            (2356,) difference voltage vector.
        n_meas_per_inj :
            Measurements per injection pattern (31 for KTC).

        Returns
        -------
        np.ndarray
            Per-element conductivity change, shape (M,).
        """
        n_inj   = ex_mat.shape[0]
        n_total = n_inj * n_meas_per_inj          # 76 x 31 = 2356

        # Reshape delta_v into (n_inj, n_meas_per_inj) -- Bug 1 fix
        dv_mat = delta_v[:n_total].reshape(n_inj, n_meas_per_inj)  # (76, 31)

        # Element centroids: shape (M, 2)
        centroids = np.mean(nodes[elements], axis=1)

        # Electrode positions in node-space: shape (32, 2)
        elec_pos = nodes[el_pos]

        # ── Injection midpoints: (76, 2) ───────────────────────────────────
        # midpoint of each [source, sink] electrode pair
        inj_mids = (elec_pos[ex_mat[:, 0]] + elec_pos[ex_mat[:, 1]]) * 0.5

        # Distance from each element centroid to each injection midpoint
        # centroids (M, 1, 2) - inj_mids (1, 76, 2) -> (M, 76)
        d_inj = np.linalg.norm(
            centroids[:, np.newaxis, :] - inj_mids[np.newaxis, :, :], axis=2
        ) + 1e-8                                  # (M, 76)

        # ── Measurement midpoints: (31, 2) ─────────────────────────────────
        # Fixed set: adjacent pairs (0,1), (1,2), ..., (30, 31)
        meas_a    = np.arange(n_meas_per_inj, dtype=int)   # [0 .. 30]
        meas_b    = meas_a + 1                              # [1 .. 31]
        meas_mids = (elec_pos[meas_a] + elec_pos[meas_b]) * 0.5  # (31, 2)

        # Distance from each centroid to each measurement midpoint
        # centroids (M, 1, 2) - meas_mids (1, 31, 2) -> (M, 31)
        d_meas = np.linalg.norm(
            centroids[:, np.newaxis, :] - meas_mids[np.newaxis, :, :], axis=2
        ) + 1e-8                                  # (M, 31)

        # ── Sensitivity tensor: (M, 76, 31) ───────────────────────────────
        # sens[e, i, j] = 1 / (d_inj[e, i] * d_meas[e, j])
        sens = 1.0 / (
            d_inj[:, :, np.newaxis] * d_meas[:, np.newaxis, :]
        )                                         # (M, 76, 31)

        # ── Accumulate: sigma_elem = sens @ dv_mat (vectorised) ───────────
        # Flatten last two dims: (M, 76*31) @ (76*31,) -> (M,)
        sigma_elem = sens.reshape(centroids.shape[0], -1) @ dv_mat.ravel()

        return sigma_elem                          # (M,)

    # ── private: electrode positions  (Bug 2) ───────────────────────────────

    @staticmethod
    def _electrode_positions(mesh_data, nodes: np.ndarray) -> np.ndarray:
        """Return (32,) int32 array of 0-indexed electrode node indices.

        Bug 2 fix: the old code did elfaces[i][0, 0] - 1, which always
        returned node 0 because [0, 0] reads the first column index (=1),
        giving 1 - 1 = 0 for every electrode.

        elfaces[i] is a (K, 2) uint8 array of boundary-edge node pairs
        (1-indexed, MATLAB convention).  Example for electrode 0:
            [[1, 66], [66, 2]]  -- two boundary edges sharing node 66

        Fix: take the midpoint of the first boundary edge in physical space
        and find the nearest mesh node.  This correctly identifies the
        representative node for each electrode.

        Parameters
        ----------
        mesh_data :
            scipy mat_struct or dict containing the elfaces field.
        nodes :
            (N, 2) float64 node coordinates.

        Returns
        -------
        np.ndarray
            Shape (32,) int32, 0-indexed electrode node indices.
        """
        n_nodes = nodes.shape[0]

        for key in ["elfaces", "ElFaces", "el_faces", "elFaces"]:
            elfaces = _get_key(mesh_data, key)
            if elfaces is None:
                continue
            try:
                el_pos_list = []
                for i in range(len(elfaces)):
                    # elfaces[i]: (K, 2) -- K boundary edges, 1-indexed nodes
                    edge = np.asarray(elfaces[i], dtype=np.int32)

                    if edge.ndim == 1 and edge.size >= 2:
                        # Single edge stored as flat array [node_a, node_b]
                        na, nb = int(edge[0]) - 1, int(edge[1]) - 1
                    elif edge.ndim == 2 and edge.shape[1] >= 2:
                        # Multiple edges -- take the first row
                        na, nb = int(edge[0, 0]) - 1, int(edge[0, 1]) - 1
                    else:
                        raise ValueError(
                            f"Unexpected elfaces[{i}] shape {edge.shape}"
                        )

                    # Bug 2 fix: compute midpoint of this boundary edge
                    midpoint = (nodes[na] + nodes[nb]) * 0.5   # (2,)
                    # Find the mesh node nearest to that midpoint
                    dists   = np.linalg.norm(nodes - midpoint, axis=1)
                    nearest = int(np.argmin(dists))
                    el_pos_list.append(nearest)

                el_pos = np.array(el_pos_list, dtype=np.int32)
                if el_pos.shape[0] == 32:
                    return el_pos
            except Exception as exc:
                warnings.warn(
                    f"BackProjection: elfaces parse failed ({exc!r})"
                    f" -- using equally-spaced electrode fallback.",
                    RuntimeWarning, stacklevel=3,
                )
                break

        # Fallback: equally-spaced node indices around the mesh boundary
        return np.round(np.linspace(0, n_nodes - 1, 32)).astype(np.int32)

    # ── private: grid interpolation ─────────────────────────────────────────

    @staticmethod
    def _interpolate_to_grid(
        nodes:    np.ndarray,
        elements: np.ndarray,
        values:   np.ndarray,
        size:     int = 256,
    ) -> np.ndarray:
        """Interpolate per-element values onto a *size* x *size* pixel grid.

        Uses scipy.interpolate.griddata (linear) with element centroids as
        scatter coordinates.  Pixels outside the mesh convex hull are 0.0.

        Parameters
        ----------
        nodes :
            (N, 2) float64 node coordinates.
        elements :
            (M, 3) int32 0-indexed element connectivity.
        values :
            1-D float array, either per-element (M,) or per-node (N,).
        size :
            Grid edge length in pixels (default 256).

        Returns
        -------
        np.ndarray
            Shape (size, size) float64.
        """
        from scipy.interpolate import griddata

        values    = values.ravel().astype(np.float64)
        centroids = np.mean(nodes[elements], axis=1)   # (M, 2)

        # If per-node values, average down to per-element
        if values.shape[0] == nodes.shape[0]:
            values = np.mean(values[elements], axis=1)
        elif values.shape[0] != elements.shape[0]:
            warnings.warn(
                f"_interpolate_to_grid: length {values.shape[0]} matches "
                f"neither nodes ({nodes.shape[0]}) nor elements "
                f"({elements.shape[0]}). Returning zero grid.",
                RuntimeWarning, stacklevel=2,
            )
            return np.zeros((size, size), dtype=np.float64)

        xi      = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), size)
        yi      = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), size)
        gx, gy  = np.meshgrid(xi, yi)

        grid = griddata(
            centroids, values, (gx, gy), method="linear", fill_value=0.0
        )
        return grid.astype(np.float64)

    # ── private: random fallback ─────────────────────────────────────────────

    def _random_fallback(self, batch: DataBatch) -> np.ndarray:
        """Reproducibly-seeded fallback when mesh is absent or unreadable."""
        warnings.warn(
            "BackProjection: no valid mesh -- returning seeded mock reconstruction.",
            RuntimeWarning, stacklevel=2,
        )
        rng       = np.random.RandomState(hash(batch.sample_id) % 2**31)
        sigma_map = rng.randn(256, 256)
        labels    = _segment_adaptive(sigma_map)
        self.validate_output(labels)
        return labels
