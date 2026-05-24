"""BackProjection reconstruction method for EIT.

Algorithm
---------
1. Difference imaging  : delta_v = voltages - reference_voltages (mean fallback).
2. Key-safe mesh access: try multiple attribute / dict-key names so the method
   works with scipy mat_struct (Mesh_sparse.mat) and plain Python dicts alike.
3. Build PyEITMesh + 32-electrode adjacent protocol, run BP.solve.
4. Interpolate via element-centroid griddata → 256×256 float grid.
5. Double-Otsu segment → integer labels {0, 1, 2}.
6. Validate and return (256, 256) int array.  Never raises — always falls back.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.methods.segment import segment
from src.ktc_framework.types import DataBatch


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _has_key(obj, key: str) -> bool:
    """Return True if *obj* exposes *key* as an attribute or dict entry.

    Handles both scipy mat_struct (attribute access) and plain ``dict``
    (key access) so callers can write ``if _has_key(mesh, 'g'):`` and have
    it work regardless of what ``BatchRunner._load_mesh`` returned.
    """
    if hasattr(obj, key):
        return True
    try:
        return key in obj
    except TypeError:
        return False


def _get_key(obj, key: str) -> Optional[np.ndarray]:
    """Retrieve *key* from *obj* via attribute or dict access.

    Returns ``None`` if the key is absent.
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
# BackProjection
# ---------------------------------------------------------------------------

@register
class BackProjection(MethodPlugin):
    """Back-projection EIT reconstruction via pyEIT.

    Uses ``batch.mesh`` (loaded from ``Mesh_sparse.mat`` by ``BatchRunner``)
    to run a real BP reconstruction.  Falls back to a reproducibly-seeded
    random segmentation when pyEIT is unavailable or the mesh cannot be parsed
    — so the pipeline **never** raises an unhandled exception.
    """

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Return a ``(256, 256)`` segmentation label array.

        Labels: 0 = water (background), 1 = resistive, 2 = conductive.

        Parameters
        ----------
        batch:
            ``DataBatch`` populated by ``BatchRunner``.  The optional
            ``reference_voltages`` and ``mesh`` fields drive real physics-based
            reconstruction when present.

        Returns
        -------
        np.ndarray
            Shape ``(256, 256)``, dtype ``int``, values in ``{0, 1, 2}``.
        """
        # ── Step 1: Difference imaging ─────────────────────────────────────
        if batch.reference_voltages is not None:
            delta_v = batch.voltages - batch.reference_voltages
        else:
            warnings.warn(
                "No reference voltages available. "
                "Falling back to mean subtraction. Results will be poor.",
                RuntimeWarning,
                stacklevel=2,
            )
            delta_v = batch.voltages - np.mean(batch.voltages)

        delta_v = delta_v.ravel().astype(np.float64)

        # ── Step 2: Setup pyEIT mesh and solver ────────────────────────────
        if batch.mesh is not None:
            # Extract mesh data from the scipy loadmat struct / dict.
            # Common key names in Mesh_sparse.mat: 'g' or 'Node' for node
            # coords, 'H' or 'Element' for connectivity.
            # Try multiple possible key names for robustness.
            node_key = None
            for k in ["g", "Node", "node", "p", "pts"]:
                if _has_key(batch.mesh, k):
                    node_key = k
                    break

            element_key = None
            for k in ["H", "Element", "element", "t", "tri"]:
                if _has_key(batch.mesh, k):
                    element_key = k
                    break

            if node_key and element_key:
                nodes    = np.asarray(_get_key(batch.mesh, node_key),    dtype=np.float64)
                elements = np.asarray(_get_key(batch.mesh, element_key), dtype=np.int32)

                # Attempt to create pyEIT mesh and run BP
                try:
                    from pyeit.mesh.wrapper import PyEITMesh          # type: ignore[import]
                    from pyeit.eit.bp import BP                        # type: ignore[import]
                    from pyeit.eit.protocol import create as proto_create  # type: ignore[import]

                    # PyEITMesh expects 0-indexed elements; MATLAB is 1-indexed
                    if elements.min() >= 1:
                        elements = elements - 1

                    n_elements = elements.shape[0]
                    n_nodes    = nodes.shape[0]

                    # PyEITMesh requires perm (conductivity) and el_pos
                    # (electrode node indices); derive both from the mesh.
                    perm   = np.ones(n_elements, dtype=np.float64)
                    el_pos = self._electrode_positions(batch.mesh, n_nodes)

                    mesh_obj = PyEITMesh(
                        node    =nodes,
                        element =elements,
                        perm    =perm,
                        el_pos  =el_pos,
                        ref_node=0,
                    )

                    # Standard 32-electrode adjacent-pair protocol
                    # (matches the KTC dataset excitation pattern)
                    protocol   = proto_create(
                        32, dist_exc=1, step_meas=1, parser_meas="std"
                    )
                    n_meas_tot = protocol.n_meas_tot  # 928 for adjacent-pair

                    # Slice / pad delta_v to match protocol measurement count
                    if len(delta_v) >= n_meas_tot:
                        dv = delta_v[:n_meas_tot]
                    else:
                        warnings.warn(
                            f"BackProjection: delta_v length {len(delta_v)} "
                            f"< n_meas_tot {n_meas_tot} — zero-padding.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        dv = np.pad(delta_v, (0, n_meas_tot - len(delta_v)))

                    # v0 is the zero background — so BP receives delta_v as-is
                    v0 = np.zeros(n_meas_tot, dtype=np.float64)

                    bp = BP(mesh_obj, protocol=protocol)
                    bp.setup(weight="none")
                    # ds is conductivity change per node or element
                    ds = bp.solve(dv, v0, normalize=False)

                    # Interpolate ds onto a 256×256 pixel grid
                    sigma_map = self._interpolate_to_grid(nodes, elements, ds, size=256)
                    labels    = segment(sigma_map)
                    self.validate_output(labels)
                    return labels

                except Exception as e:
                    warnings.warn(
                        f"pyEIT BP failed: {e}. Falling back to simple interpolation.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # ── Fallback: seeded random reconstruction ─────────────────────────
        # Use a deterministic seed so repeated calls on the same sample return
        # the same output (useful for debugging / reproducibility checks).
        warnings.warn(
            "No valid mesh available. Returning mock reconstruction.",
            RuntimeWarning,
            stacklevel=2,
        )
        rng       = np.random.RandomState(hash(batch.sample_id) % 2**31)
        sigma_map = rng.randn(256, 256)
        labels    = segment(sigma_map)
        self.validate_output(labels)
        return labels

    # ── private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _electrode_positions(mesh_data, n_nodes: int) -> np.ndarray:
        """Return a ``(32,)`` int32 array of 0-indexed electrode node indices.

        Reads ``elfaces`` from *mesh_data* (present in ``Mesh_sparse.mat``).
        Each ``elfaces[i]`` is shape ``(n_segs, 2)``; the first node of the
        first segment is used as the representative boundary node.

        Falls back to 32 equally-spaced node indices if ``elfaces`` is absent
        or cannot be parsed.
        """
        for key in ["elfaces", "ElFaces", "el_faces", "elFaces"]:
            elfaces = _get_key(mesh_data, key)
            if elfaces is not None:
                try:
                    el_pos = np.array(
                        [int(elfaces[i][0, 0]) - 1 for i in range(len(elfaces))],
                        dtype=np.int32,
                    )
                    if el_pos.shape[0] == 32:
                        return el_pos
                except Exception as exc:
                    warnings.warn(
                        f"BackProjection: elfaces parse failed ({exc})"
                        f" — using equally-spaced electrode fallback.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    break  # skip remaining elfaces key variants

        # Equally-spaced fallback across all node indices
        return np.round(np.linspace(0, n_nodes - 1, 32)).astype(np.int32)

    @staticmethod
    def _interpolate_to_grid(
        nodes:    np.ndarray,
        elements: np.ndarray,
        values:   np.ndarray,
        size:     int = 256,
    ) -> np.ndarray:
        """Interpolate per-element or per-node *values* onto a *size*×*size* grid.

        Uses ``scipy.interpolate.griddata`` with element centroids as the
        scatter coordinates.  Pixels outside the mesh convex hull are ``0.0``.

        Parameters
        ----------
        nodes:
            ``(n_nodes, 2)`` float64 — mesh node ``(x, y)`` coordinates.
        elements:
            ``(n_elements, 3)`` int32 — 0-indexed triangle connectivity.
        values:
            1-D float array.  Accepts either per-element length (uses element
            centroids directly) or per-node length (averages node values to
            element centroids so the same helper works for BP and JAC output).
        size:
            Grid edge length in pixels (default 256).

        Returns
        -------
        np.ndarray
            Shape ``(size, size)`` float64.
        """
        from scipy.interpolate import griddata

        values = values.ravel().astype(np.float64)

        # Compute centroids of each triangle: shape (n_elements, 2)
        centroids = np.mean(nodes[elements], axis=1)  # (M, 2)

        # If values are node-based (e.g. BP output), average to element level
        if values.shape[0] == nodes.shape[0]:
            values = np.mean(values[elements], axis=1)  # (n_elements,)
        elif values.shape[0] != elements.shape[0]:
            warnings.warn(
                f"BackProjection._interpolate_to_grid: values length "
                f"{values.shape[0]} matches neither nodes ({nodes.shape[0]}) "
                f"nor elements ({elements.shape[0]}). Returning zero grid.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((size, size), dtype=np.float64)

        # Create a regular size×size grid spanning the mesh bounding box
        xi     = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), size)
        yi     = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), size)
        grid_x, grid_y = np.meshgrid(xi, yi)

        # Interpolate scatter (centroid, value) pairs onto the regular grid
        sigma_grid = griddata(
            centroids,
            values,
            (grid_x, grid_y),
            method="linear",
            fill_value=0.0,
        )
        return sigma_grid.astype(np.float64)
