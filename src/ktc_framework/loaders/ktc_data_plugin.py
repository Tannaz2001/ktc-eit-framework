"""KTCDataPlugin — loads KTC 2023 evaluation-dataset samples for BatchRunner.

Actual dataset layout (Zenodo download, unzipped):
    <dataset_root>/evaluation_datasets/level{N}/data{1|2|3}.mat
    <dataset_root>/GroundTruths/level_{N}/{1|2|3}_true.mat

The plugin auto-detects alternate folder names at ``__init__`` time so it
works whether the caller uses the Zenodo layout or a re-arranged copy:
    Evaluation data : ``evaluation_datasets/``  OR  ``EvaluationData/``
    Ground truths   : ``GroundTruths/``          OR  ``groundtruths/``

``mesh`` and ``reference_voltages`` are always ``None`` on return —
``BatchRunner`` fills them from the shared resources it loaded at startup.
"""

from __future__ import annotations

import os
from os import path
import warnings
from typing import Optional

import numpy as np
import scipy.io

from src.ktc_framework.registry import PluginRegistry
from src.ktc_framework.types import DataBatch

try:
    import h5py as _h5py
except ImportError:
    _h5py = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Candidate folder names tried in priority order during path discovery
_EVAL_CANDIDATES: list[str] = ["evaluation_datasets", "EvaluationData"]
_GT_CANDIDATES:   list[str] = ["GroundTruths", "groundtruths", "GroundTruth"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_folder(root: str, candidates: list[str]) -> Optional[str]:
    """Return the first existing subfolder of *root* from *candidates*.

    Parameters
    ----------
    root:
        Absolute or relative path to the dataset root directory.
    candidates:
        Folder names to try, in priority order.

    Returns
    -------
    str | None
        The matching folder name, or ``None`` if none are found.
    """
    for name in candidates:
        if os.path.isdir(os.path.join(root, name)):
            return name
    return None


def _load_mat(path: str) -> dict:
    """Load a ``.mat`` file and return its contents as a plain Python dict.

    Supports both MATLAB v5 (``scipy.io.loadmat``) and v7.3 / HDF5
    (``h5py``).  The h5py path is taken when ``scipy`` raises
    ``NotImplementedError`` (its signal that the file is HDF5-based).

    Parameters
    ----------
    path:
        Absolute path to the ``.mat`` file.

    Returns
    -------
    dict
        Variable name → array mapping.

    Raises
    ------
    ImportError
        If the file is v7.3 and ``h5py`` is not installed.
    """
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if _h5py is None:
            raise ImportError(
                "h5py is required to load MATLAB v7.3 (.mat) files. "
                "Run:  pip install h5py"
            )
        result: dict = {}
        with _h5py.File(path, "r") as f:
            for key in f.keys():
                result[key] = np.array(f[key])
        return result


# ---------------------------------------------------------------------------
# KTCDataPlugin
# ---------------------------------------------------------------------------

@PluginRegistry.register("KTCDataPlugin")
class KTCDataPlugin:
    """Loads KTC 2023 evaluation samples by difficulty level and sample letter.

    ``BatchRunner`` instantiates this class as:

    .. code-block:: python

        plugin = KTCDataPlugin(dataset_root)   # plain string

    A config dict is also accepted for direct use:

    .. code-block:: python

        plugin = KTCDataPlugin({"dataset_root": "EvaluationData"})

    Parameters
    ----------
    dataset_root_or_config:
        Path to the folder that contains ``evaluation_datasets/`` and
        ``GroundTruths/``; **or** a config dict with key ``'dataset_root'``.
    """

    def __init__(self, dataset_root_or_config: str | dict = "") -> None:

        # ── 1. Resolve dataset_root from string or config dict ─────────────
        if isinstance(dataset_root_or_config, dict):
            self.dataset_root: str = str(
                dataset_root_or_config.get("dataset_root", "")
            )
        else:
            self.dataset_root = str(dataset_root_or_config)

        # ── 2. Sample-letter → file-number mapping ─────────────────────────
        # Supports A/B/C letter convention and numeric 1/2/3/4 directly.
        self._sample_map: dict[str, str] = {
            "A": "1", "B": "2", "C": "3",
            "1": "1", "2": "2", "3": "3", "4": "4",
        }

        # ── 3. Auto-detect actual folder names on disk ─────────────────────
        self._eval_folder: Optional[str] = _find_folder(
            self.dataset_root, _EVAL_CANDIDATES
        )
        self._gt_folder: Optional[str] = _find_folder(
            self.dataset_root, _GT_CANDIDATES
        )

        # ── 4. Log resolved paths (always shown so the user can verify) ────
        if self._eval_folder:
            print(
                f"[KTCDataPlugin] Evaluation data : "
                f"{os.path.join(self.dataset_root, self._eval_folder)}"
            )
        else:
            warnings.warn(
                f"KTCDataPlugin: no evaluation-data folder found under "
                f"'{self.dataset_root}'. Tried: {_EVAL_CANDIDATES}. "
                f"load_sample() will raise FileNotFoundError.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self._gt_folder:
            print(
                f"[KTCDataPlugin] Ground truths   : "
                f"{os.path.join(self.dataset_root, self._gt_folder)}"
            )
        else:
            warnings.warn(
                f"KTCDataPlugin: no ground-truth folder found under "
                f"'{self.dataset_root}'. Tried: {_GT_CANDIDATES}. "
                f"Ground truth will default to zeros(256,256).",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── public interface ────────────────────────────────────────────────────

    def load_sample(self, level: int, sample: str) -> DataBatch:
        """Load one KTC measurement sample.

        Parameters
        ----------
        level:
            Difficulty level, 1–7.
        sample:
            Sample identifier — ``'A'``, ``'B'``, ``'C'`` (or ``'1'``–``'3'``).

        Returns
        -------
        DataBatch
            Fully populated batch.  ``mesh`` and ``reference_voltages`` are
            ``None``; ``BatchRunner`` attaches the shared resources later.

        Raises
        ------
        FileNotFoundError
            If the voltage ``.mat`` file does not exist on disk.
        ValueError
            If the loaded file does not contain the expected ``'Uel'`` key.
        """
        num = self._sample_map.get(sample.upper(), sample)

        # Use discovered folder names, or fall back to first candidate so the
        # FileNotFoundError message contains a meaningful path.
        eval_folder = self._eval_folder or _EVAL_CANDIDATES[0]
        gt_folder   = self._gt_folder   or _GT_CANDIDATES[0]

        # Construct paths with os.path for cross-platform compatibility
        data_path = os.path.join(
            self.dataset_root, eval_folder, f"level{level}", f"data{num}.mat"
        )
        gt_path = os.path.join(
            self.dataset_root, gt_folder, f"level_{level}", f"{num}_true.mat"
        )

        print(f"[KTCDataPlugin] Loading voltages : {data_path}")
        print(f"[KTCDataPlugin] Loading GT       : {gt_path}")

        # Voltage file is mandatory — raise immediately if missing
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Voltage data not found: {data_path}\n"
                f"  dataset_root = '{self.dataset_root}'\n"
                f"  eval_folder  = '{eval_folder}'\n"
                f"  level={level}, sample='{sample}' → num='{num}'"
            )

        voltages, injection, mpat, ref_voltages = self._load_data(data_path)
        ground_truth        = self._load_gt(gt_path)

        return DataBatch(
            voltages           = voltages,
            injection_patterns = injection,
            ground_truth       = ground_truth,
            level              = level,
            sample_id          = f"level{level}_{sample}",
            mesh               = None,   # filled by BatchRunner._run_one
            reference_voltages = ref_voltages,   # per-level ref.mat from evaluation data
            measurement_patterns=mpat,
        )

    # ── private helpers ─────────────────────────────────────────────────────

    def _load_data(
        self, path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract Uel, Inj, and Mpat from KTC data file.

        Evaluation data files usually contain Uel only.
        If Inj or Mpat are missing, they are loaded from ref.mat.
        """
        mat = _load_mat(path)

        # Uel: mandatory voltage measurement vector
        if "Uel" not in mat:
            raise ValueError(
                f"Key 'Uel' not found in {path}. "
                f"Available keys: {[k for k in mat if not k.startswith('_')]}"
            )

        voltages = np.asarray(mat["Uel"], dtype=np.float32).ravel()

        protocol = None

        # Inj: load from data file if available, otherwise from ref.mat
        if "Inj" in mat:
            injection = np.asarray(mat["Inj"], dtype=np.float32)
        else:
            protocol = self._load_protocol_from_ref(path)
            injection = protocol["Inj"]

        if injection.ndim == 2 and injection.shape == (76, 32):
            injection = injection.T

        if injection.shape != (32, 76):
            raise ValueError(
                f"Invalid Inj shape in {path}: {injection.shape}. "
                f"Expected (32, 76)."
            )

        # Mpat: load from data file if available, otherwise from ref.mat
        if "Mpat" in mat:
            mpat = np.asarray(mat["Mpat"], dtype=np.float32)
        else:
            if protocol is None:
                protocol = self._load_protocol_from_ref(path)

            mpat = protocol["Mpat"]

        if mpat.ndim == 2 and mpat.shape == (31, 32):
            mpat = mpat.T

        if mpat.shape != (32, 31):
            raise ValueError(
                f"Invalid Mpat shape in {path}: {mpat.shape}. "
                f"Expected (32, 31)."
            )

        ref_voltages = None
        try:
            protocol_for_ref = protocol or self._load_protocol_from_ref(path)
            ref_voltages = protocol_for_ref.get("Uelref")
        except Exception:
            ref_voltages = None

        return voltages, injection, mpat, ref_voltages

    def _load_gt(self, path: str) -> np.ndarray:
        """Load the ground-truth segmentation mask.

        Parameters
        ----------
        path:
            Absolute path to the ground-truth ``.mat`` file.

        Returns
        -------
        np.ndarray
            Shape ``(256, 256)`` uint8, labels ``{0, 1, 2}``.

        Raises
        ------
        FileNotFoundError
            If the GT file does not exist on disk.
        ValueError
            If the file is unreadable, missing the expected key, or has the
            wrong spatial shape.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Ground-truth file not found: {path}\n"
                f"  Check that GroundTruths/ is present under dataset_root "
                f"and that the level/sample numbering matches."
            )

        try:
            mat = _load_mat(path)
        except Exception as exc:
            raise ValueError(
                f"Could not load ground truth from {path}: {exc}"
            ) from exc

        # Try common key names for the truth / segmentation array
        truth_array: Optional[np.ndarray] = None
        for key in ["truth", "Truth", "gt", "GT", "labels", "mask"]:
            if key in mat:
                truth_array = mat[key]
                break

        if truth_array is None:
            visible_keys = [k for k in mat if not k.startswith("_")]
            raise ValueError(
                f"No recognised ground-truth key in {path}. "
                f"Keys found: {visible_keys}. Expected one of: truth, Truth, gt, GT, labels, mask."
            )

        gt = np.asarray(truth_array)

        # Drop any leading size-1 dimensions (e.g. h5py shape (1,256,256))
        if gt.ndim > 2:
            gt = gt.squeeze()

        # Validate spatial shape
        if gt.shape != (256, 256):
            raise ValueError(
                f"Ground-truth shape {gt.shape} != (256, 256) in {path}. "
                f"Ensure the GT file matches the KTC dataset format."
            )

        return gt.astype(np.uint8)

    def _default_mpat(self) -> np.ndarray:
        """Create default adjacent voltage measurement pattern with shape (32, 31)."""
        mpat = np.zeros((32, 31), dtype=np.float32)

        for i in range(31):
            mpat[i, i] = 1.0
            mpat[i + 1, i] = -1.0
        return mpat
    
    def _load_protocol_from_ref(self, data_path: str) -> dict[str, np.ndarray]:
        """Load Inj/Injref and Mpat from ref.mat."""

        from pathlib import Path

        data_path = Path(data_path)

        candidates = [
            data_path.parent / "ref.mat",
            data_path.parent.parent / "ref.mat",
            data_path.parent.parent.parent / "ref.mat",
            Path("EvaluationData") / "ref.mat",
            Path("Codes_Matlab") / "TrainingData" / "ref.mat",
        ]

        for ref_path in candidates:
            if not ref_path.exists():
                continue

            ref_mat = _load_mat(str(ref_path))

            if "Injref" in ref_mat:
                injection = np.asarray(ref_mat["Injref"], dtype=np.float32)
            elif "Inj" in ref_mat:
                injection = np.asarray(ref_mat["Inj"], dtype=np.float32)
            else:
                continue

            if "Mpat" not in ref_mat:
                continue

            mpat = np.asarray(ref_mat["Mpat"], dtype=np.float32)

            if injection.ndim == 2 and injection.shape == (76, 32):
                injection = injection.T

            if mpat.ndim == 2 and mpat.shape == (31, 32):
                mpat = mpat.T

            if injection.shape == (32, 76) and mpat.shape == (32, 31):
                result = {
                    "Inj": injection,
                    "Mpat": mpat,
                }
                if "Uelref" in ref_mat:
                    result["Uelref"] = np.asarray(ref_mat["Uelref"], dtype=np.float32).ravel()
                elif "Uel" in ref_mat:
                    result["Uelref"] = np.asarray(ref_mat["Uel"], dtype=np.float32).ravel()
                return result

        raise FileNotFoundError(
            "Could not load Inj/Injref and Mpat from ref.mat. Tried: "
            + ", ".join(str(p) for p in candidates)
        )