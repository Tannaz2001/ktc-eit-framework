"""KTCDataPlugin — loads real KTC 2023 evaluation dataset for BatchRunner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from src.ktc_framework.loaders.ktc_loader import PluginRegistry
from src.ktc_framework.types import DataBatch

try:
    import h5py as _h5py
except ImportError:
    _h5py = None  # type: ignore[assignment]

# Sample letter → file number (KTC dataset: A=1, B=2, C=3)
_SAMPLE_MAP = {"A": "1", "B": "2", "C": "3"}


@PluginRegistry.register("KTCDataPlugin")
class KTCDataPlugin:
    """
    Loads KTC 2023 evaluation data from the official dataset structure.

    Dataset layout (under dataset_root):
      evaluation_datasets/level{N}/data{1|2|3}.mat  — voltages + injection
      GroundTruths/level_{N}/{1|2|3}_true.mat        — ground truth (256x256)
    """

    def __init__(self, dataset_root: str = "") -> None:
        self.dataset_root = Path(dataset_root)

    def load_sample(self, level: int, sample: str) -> DataBatch:
        """Load one KTC sample by level and sample letter (A/B/C)."""
        num = _SAMPLE_MAP.get(sample.upper(), sample)

        data_path = self.dataset_root / "evaluation_datasets" / f"level{level}" / f"data{num}.mat"
        gt_path = self.dataset_root / "GroundTruths" / f"level_{level}" / f"{num}_true.mat"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        voltages, injection = self._load_data(data_path)
        ground_truth = self._load_gt(gt_path)

        return DataBatch(
            voltages=voltages,
            injection_patterns=injection,
            ground_truth=ground_truth,
            level=level,
            sample_id=f"level{level}_{sample}",
        )

    def _load_data(self, path: Path):
        try:
            mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
            voltages = np.asarray(mat["Uel"], dtype=np.float32).ravel()
            injection = np.asarray(mat["Inj"], dtype=np.float32)
        except NotImplementedError:
            if _h5py is None:
                raise ImportError("h5py is required for .mat v7.3 files. Run: pip install h5py")
            with _h5py.File(str(path), "r") as f:
                voltages = np.array(f["Uel"], dtype=np.float32).ravel()
                injection = np.array(f["Inj"], dtype=np.float32).T
        return voltages, injection

    def _load_gt(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((256, 256), dtype=np.float32)
        try:
            mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
            return np.asarray(mat["truth"], dtype=np.float32)
        except NotImplementedError:
            if _h5py is None:
                raise ImportError("h5py is required for .mat v7.3 files. Run: pip install h5py")
            with _h5py.File(str(path), "r") as f:
                return np.array(f["truth"], dtype=np.float32).T
