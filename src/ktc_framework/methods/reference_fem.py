"""KTC linear difference reconstruction using the official solver modules.

This adapter keeps the framework API intact while using the professor-provided
FEM/Jacobian/smoothness-prior reconstruction pipeline from data/KTCScoring.
Inputs and outputs remain MATLAB-style arrays: a DataBatch with .mat-loaded
voltages/protocol/mesh in, and a 256x256 label image out.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
import warnings

import numpy as np

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch


_ROOT = Path(__file__).resolve().parents[3]
_KTC_SCORING_DIR = _ROOT / "data" / "KTCScoring"
if str(_KTC_SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(_KTC_SCORING_DIR))

try:
    import KTCAux  # type: ignore[import]
    import KTCFwd  # type: ignore[import]
    import KTCMeshing  # type: ignore[import]
    import KTCRegularization  # type: ignore[import]
    import KTCScoring  # type: ignore[import]
except Exception as exc:  # pragma: no cover - exercised by runtime availability
    KTCAux = KTCFwd = KTCMeshing = KTCRegularization = KTCScoring = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _as_array(value, dtype=None) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _mat_field(obj, key: str):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj[key]
    raise AttributeError(key)


def _convert_node(node):
    coord = _as_array(_mat_field(node, "Coordinate"), dtype=float).ravel()
    elems = _as_array(_mat_field(node, "ElementConnection"), dtype=int).ravel()
    if elems.size and elems.min() >= 1:
        elems = elems - 1
    return KTCMeshing.NODE(coord, elems)


def _convert_element(element):
    topology = _as_array(_mat_field(element, "Topology"), dtype=int).ravel()
    if topology.size and topology.min() >= 1:
        topology = topology - 1

    electrode = []
    try:
        raw = _mat_field(element, "Electrode")
        flat = _as_array(raw, dtype=object).ravel()
        if flat.size:
            electrode_id = int(_as_array(flat[0]).ravel()[0])
            nodes = (
                _as_array(flat[1], dtype=int).ravel()
                if flat.size > 1
                else np.array([], dtype=int)
            )
            if electrode_id >= 1:
                electrode_id -= 1
            if nodes.size and nodes.min() >= 1:
                nodes = nodes - 1
            electrode = [electrode_id, nodes]
    except Exception:
        electrode = []

    return KTCMeshing.ELEMENT(topology, electrode)


def _convert_mesh(mesh_struct):
    """Convert scipy-loaded MATLAB Mesh/Mesh2 structs to KTCMeshing.Mesh."""
    h = _as_array(_mat_field(mesh_struct, "H"), dtype=np.uint32)
    if h.size and h.min() >= 1:
        h = h - 1

    g = _as_array(_mat_field(mesh_struct, "g"), dtype=float)

    elfaces = []
    for face in np.ravel(_mat_field(mesh_struct, "elfaces")):
        arr = _as_array(face, dtype=np.uint32)
        if arr.size and arr.min() >= 1:
            arr = arr - 1
        elfaces.append(arr)

    nodes = [_convert_node(n) for n in np.ravel(_mat_field(mesh_struct, "Node"))]
    elements = [
        _convert_element(e) for e in np.ravel(_mat_field(mesh_struct, "Element"))
    ]

    return KTCMeshing.Mesh(h, g, elfaces, nodes, elements)


def _reference_mask(level: int, inj: np.ndarray) -> np.ndarray:
    """KTC measurement inclusion mask with shape (31, 76)."""
    inj = np.asarray(inj)
    if inj.shape[0] != 32 and inj.shape[1] == 32:
        inj = inj.T

    vincl = np.ones((31, inj.shape[1]), dtype=bool)
    rmind = np.arange(0, 2 * (int(level) - 1), dtype=int)

    for col in range(inj.shape[1]):
        for row in rmind:
            if row < inj.shape[0] and inj[row, col]:
                vincl[:, col] = False
            if row < vincl.shape[0]:
                vincl[row, :] = False

    return vincl


def _segment_reference(image: np.ndarray) -> np.ndarray:
    """Segment a conductivity-change image using the KTC Otsu2 logic."""
    level, x = KTCScoring.Otsu2(np.asarray(image).ravel(), 256, 7)

    labels = np.zeros_like(image, dtype=np.uint8)
    low = image < x[level[0]]
    mid = (image >= x[level[0]]) & (image <= x[level[1]])
    high = image > x[level[1]]

    counts = [int(low.sum()), int(mid.sum()), int(high.sum())]
    background = int(np.argmax(counts))

    if background == 0:
        labels[mid | high] = 2
    elif background == 1:
        labels[low] = 1
        labels[high] = 2
    else:
        labels[low | mid] = 1

    return labels


@lru_cache(maxsize=16)
def _prior_for_mesh(mesh_key: tuple[int, int, int, float, float]):
    mesh, _mesh2, _inj_shape, _mpat_shape = _REFERENCE_CACHE[mesh_key]
    sigma0 = np.ones((len(mesh.g), 1))
    prior = KTCRegularization.SMPrior(
        mesh.g,
        mesh_key[3],
        mesh_key[4],
        sigma0,
    )
    return sigma0, prior


_REFERENCE_CACHE: dict[tuple[int, int, int, float, float], tuple[object, object, tuple, tuple]] = {}

# Caches the expensive, per-sample-invariant operators: the converted Mesh,
# the Jacobian, and the full noise-precision matrix InvGamma_n.  These depend
# only on (mesh, injection pattern, measurement pattern, level, reference
# voltages, noise params) -- all constant across the A/B/C samples of a level.
# Without this, every sample paid a ~10 s SolveForward + Jacobian rebuild.
# Mirrors the _OPERATOR_CACHE approach already used in gauss_newton.py.
_OPERATOR_CACHE: dict[tuple, tuple[object, object, object, object]] = {}


def _ref_key(reference: np.ndarray) -> str:
    """Stable hash of the reference voltages for cache keying."""
    import hashlib

    arr = np.ascontiguousarray(np.asarray(reference, dtype=np.float64))
    return hashlib.md5(arr.tobytes()).hexdigest()


@register
class LinearDifferenceReconstruction(MethodPlugin):
    """Official KTC linear difference inverse reconstruction.

    This mirrors Codes_Matlab/main.m:
    solve((J' * InvGamma * J + L' * L), J' * InvGamma * (Uel - Uelref)),
    then interpolation to the pixel grid and Otsu2 segmentation.
    """

    def __init__(
        self,
        noise_std1: float = 0.05,
        noise_std2: float = 0.01,
        corrlength: float = 0.115,
        var_sigma: float = 0.05**2,
    ) -> None:
        self.noise_std1 = float(noise_std1)
        self.noise_std2 = float(noise_std2)
        self.corrlength = float(corrlength)
        self.var_sigma = float(var_sigma)

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                f"Could not import KTCScoring modules from {_KTC_SCORING_DIR}: "
                f"{_IMPORT_ERROR}"
            )

        if batch.mesh is None:
            warnings.warn(
                "LinearDifferenceReconstruction: no mesh available; returning zeros."
            )
            return np.zeros((256, 256), dtype=np.uint8)

        reference = getattr(batch, "reference_voltages", None)
        if reference is None:
            warnings.warn(
                "LinearDifferenceReconstruction: no reference voltages; returning zeros."
            )
            return np.zeros((256, 256), dtype=np.uint8)

        try:
            mesh_struct = batch.mesh
            mesh2_struct = getattr(batch, "mesh2", None)
            if isinstance(mesh_struct, dict):
                mesh2_struct = mesh_struct.get("Mesh2", mesh2_struct)
                mesh_struct = mesh_struct.get("Mesh", mesh_struct)
            elif hasattr(mesh_struct, "Mesh") and hasattr(mesh_struct, "Mesh2"):
                mesh2_struct = mesh_struct.Mesh2
                mesh_struct = mesh_struct.Mesh

            if mesh2_struct is None:
                mesh2_struct = getattr(batch.mesh, "Mesh2", None)
            if mesh2_struct is None:
                raise ValueError(
                    "LinearDifferenceReconstruction requires both Mesh and Mesh2 structs."
                )

            inj = np.asarray(batch.injection_patterns, dtype=float)
            mpat = np.asarray(batch.measurement_patterns, dtype=float)
            ref = np.asarray(reference, dtype=float).reshape(-1, 1)
            voltages = np.asarray(batch.voltages, dtype=float).reshape(-1, 1)

            vincl = _reference_mask(batch.level, inj)
            flat_mask = vincl.T.reshape(-1)

            finite = np.isfinite(voltages.ravel()) & np.isfinite(ref.ravel())
            flat_mask = flat_mask & finite[: flat_mask.size]

            if not flat_mask.any():
                warnings.warn(
                    "LinearDifferenceReconstruction: no finite included measurements; "
                    "returning zeros."
                )
                return np.zeros((256, 256), dtype=np.uint8)

            # ── operator cache: mesh conversion + prior + forward solve + Jacobian ──
            # All of these are identical for every sample at the same level, so
            # build them once and reuse.  The per-sample voltages only enter via
            # delta_u below (cheap matrix-vector ops).  This is what makes the
            # A/B/C samples of a level near-instant after the first one.
            op_key = (
                id(batch.mesh),
                inj.shape[1],
                mpat.shape[1],
                int(batch.level),
                self.corrlength,
                self.var_sigma,
                self.noise_std1,
                self.noise_std2,
                _ref_key(ref),
            )
            cached_ops = _OPERATOR_CACHE.get(op_key)
            if cached_ops is None:
                mesh = _convert_mesh(mesh_struct)
                mesh2 = _convert_mesh(mesh2_struct)

                prior_key = (
                    id(batch.mesh),
                    inj.shape[1],
                    mpat.shape[1],
                    self.corrlength,
                    self.var_sigma,
                )
                _REFERENCE_CACHE.setdefault(prior_key, (mesh, mesh2, inj.shape, mpat.shape))
                sigma0, prior = _prior_for_mesh(prior_key)

                solver = KTCFwd.EITFEM(mesh2, inj, mpat, vincl)
                solver.SetInvGamma(self.noise_std1, self.noise_std2, ref)
                solver.SolveForward(sigma0.copy(), 1e-6 * np.ones((32, 1)))
                jacobian = solver.Jacobian(sigma0.copy(), 1e-6 * np.ones((32, 1)))
                inv_gamma_full = solver.InvGamma_n

                _OPERATOR_CACHE[op_key] = (mesh, jacobian, inv_gamma_full, prior)
            else:
                mesh, jacobian, inv_gamma_full, prior = cached_ops

            delta_u = voltages - ref
            gamma = inv_gamma_full[np.ix_(flat_mask, flat_mask)]
            lhs = jacobian.T @ gamma @ jacobian + prior.L.T @ prior.L
            rhs = jacobian.T @ gamma @ delta_u[flat_mask]
            delta_reco = np.linalg.solve(lhs, rhs)

            image = KTCAux.interpolateRecoToPixGrid(delta_reco, mesh)
            labels = _segment_reference(image)
            self.validate_output(labels)
            return labels

        except Exception as exc:
            warnings.warn(
                f"LinearDifferenceReconstruction failed ({exc!r}); returning zeros."
            )
            return np.zeros((256, 256), dtype=np.uint8)


@register
class RegularizedFEMReconstruction(LinearDifferenceReconstruction):
    """Backward-compatible alias for the same inverse reconstruction."""


@register
class ReferenceFEM(LinearDifferenceReconstruction):
    """Backward-compatible alias for older configs."""
