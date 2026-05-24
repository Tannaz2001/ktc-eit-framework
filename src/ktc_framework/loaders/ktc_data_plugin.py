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
import warnings
from typing import Optional

import numpy as np
import scipy.io

from src.ktc_framework.loaders.ktc_loader import PluginRegistry
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

        voltages, injection = self._load_data(data_path)
        ground_truth        = self._load_gt(gt_path)

        return DataBatch(
            voltages           = voltages,
            injection_patterns = injection,
            ground_truth       = ground_truth,
            level              = level,
            sample_id          = f"level{level}_{sample}",
            mesh               = None,   # filled by BatchRunner._run_one
            reference_voltages = None,   # filled by BatchRunner._run_one
        )

    # ── private helpers ─────────────────────────────────────────────────────

    def _load_data(
        self, path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``Uel`` (voltages) and ``Inj`` (injection patterns).

        Parameters
        ----------
        path:
            Absolute path to the data ``.mat`` file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            * ``voltages``   — shape ``(2356,)`` float32, flattened.
            * ``injection``  — shape ``(32, 76)`` float32.

        Raises
        ------
        ValueError
            If ``'Uel'`` is not present in the loaded file.
        """
        mat = _load_mat(path)

        # ── Uel: mandatory voltage measurement vector ──────────────────────
        if "Uel" not in mat:
            raise ValueError(
                f"Key 'Uel' not found in {path}. "
                f"Available keys: {[k for k in mat if not k.startswith('_')]}"
            )
        voltages = np.asarray(mat["Uel"], dtype=np.float32).ravel()   # (2356,)

        # ── Inj: injection-pattern matrix — optional ───────────────────────
        if "Inj" in mat:
            injection = np.asarray(mat["Inj"], dtype=np.float32)
            # h5py loads HDF5 datasets transposed relative to scipy
            if injection.ndim == 2 and injection.shape == (76, 32):
                injection = injection.T                                 # → (32, 76)
        else:
            warnings.warn(
                f"'Inj' key not found in {path} — "
                f"using default zeros(32, 76) placeholder.",
                RuntimeWarning,
                stacklevel=3,
            )
            injection = np.zeros((32, 76), dtype=np.float32)

        return voltages, injection

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
            Returns zeros if the file is absent or unreadable.
        """
        if not os.path.exists(path):
            warnings.warn(
                f"Ground-truth file not found: {path} — returning zeros(256,256).",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        try:
            mat = _load_mat(path)
        except Exception as exc:
            warnings.warn(
                f"Could not load ground truth from {path}: {exc} "
                f"— returning zeros(256,256).",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        # Try common key names for the truth / segmentation array
        truth_array: Optional[np.ndarray] = None
        for key in ["truth", "Truth", "gt", "GT", "labels", "mask"]:
            if key in mat:
                truth_array = mat[key]
                break

        if truth_array is None:
            visible_keys = [k for k in mat if not k.startswith("_")]
            warnings.warn(
                f"No recognised ground-truth key in {path}. "
                f"Keys found: {visible_keys} — returning zeros(256,256).",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        gt = np.asarray(truth_array)

        # Drop any leading size-1 dimensions (e.g. h5py shape (1,256,256))
        if gt.ndim > 2:
            gt = gt.squeeze()

        # Validate spatial shape
        if gt.shape != (256, 256):
            warnings.warn(
                f"Ground-truth shape {gt.shape} != (256, 256) in {path} "
                f"— returning zeros(256,256).",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        return gt.astype(np.uint8)
