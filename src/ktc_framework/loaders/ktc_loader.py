"""
ktc_loader.py
-------------
Provides:
  - PluginRegistry  : lightweight name → class registry with decorator support
  - KTCValidator    : static validation helpers for KTC DataBatch objects
  - KTCLoader       : loads KTC .mat files (v5 and v7.3) and returns DataBatch
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List

import numpy as np
import scipy.io

from src.ktc_framework.types import DataBatch

try:
    import h5py as _h5py
except ImportError:
    _h5py = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """Maps string names to plugin classes."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(plugin_cls: type) -> type:
            cls._registry[name] = plugin_cls
            return plugin_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise KeyError(f"Plugin '{name}' not found. Available: {available}")
        return cls._registry[name]


# ---------------------------------------------------------------------------
# KTCValidator
# ---------------------------------------------------------------------------

class KTCValidator:
    VOLTAGE_SHAPE: tuple = (2356,)
    GT_SHAPE: tuple = (256, 256)
    VALID_LABELS: set = {0, 1, 2}

    @staticmethod
    def validate(batch: DataBatch) -> None:
        if batch.voltages.shape != KTCValidator.VOLTAGE_SHAPE:
            raise ValueError(
                f"voltages shape mismatch: expected {KTCValidator.VOLTAGE_SHAPE}, "
                f"got {batch.voltages.shape}"
            )
        gt_spatial = batch.ground_truth.shape[-2:]
        if gt_spatial != KTCValidator.GT_SHAPE:
            raise ValueError(
                f"ground_truth shape mismatch: expected {KTCValidator.GT_SHAPE}, "
                f"got {gt_spatial}"
            )
        unique_labels = set(np.unique(batch.ground_truth).astype(int).tolist())
        invalid = unique_labels - KTCValidator.VALID_LABELS
        if invalid:
            raise ValueError(f"ground_truth contains invalid labels: {sorted(invalid)}")


# ---------------------------------------------------------------------------
# KTCLoader
# ---------------------------------------------------------------------------

@PluginRegistry.register('ktc_loader')
class KTCLoader:
    """Loads KTC EIT dataset samples from .mat files."""

    def __init__(self, data_dir: str | Path, level: int = 1) -> None:
        self.data_dir = Path(data_dir)
        self.level = level

    def load(self, filename: str) -> DataBatch:
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            mat = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
            return self._parse_mat_v5(mat, filename)
        except NotImplementedError:
            return self._parse_mat_v73(filepath, filename)

    def load_sample(self, level: int, sample: str) -> DataBatch:
        return self.load(f"level{level}_{sample}.mat")

    def list_samples(self) -> List[str]:
        pattern = f"level{self.level}_*.mat"
        return sorted(
            e.name for e in self.data_dir.iterdir()
            if e.is_file() and fnmatch.fnmatch(e.name, pattern)
        )

    def _parse_mat_v5(self, mat: dict, filename: str) -> DataBatch:
        voltages = np.asarray(mat['Uel'], dtype=np.float32)
        injection = np.asarray(mat['Inj'], dtype=np.float32)
        ground_truth = np.asarray(mat['truth'], dtype=np.float32)
        level = int(mat.get('level', self.level))
        return DataBatch(
            voltages=voltages,
            injection_patterns=injection,
            ground_truth=ground_truth,
            level=level,
            sample_id=Path(filename).stem,
        )

    def _parse_mat_v73(self, filepath: Path, filename: str) -> DataBatch:
        if _h5py is None:
            raise ImportError("h5py is required to load MATLAB v7.3 files. pip install h5py")
        with _h5py.File(str(filepath), 'r') as f:
            voltages = np.array(f['Uel'], dtype=np.float32).T
            injection = np.array(f['Inj'], dtype=np.float32).T
            ground_truth = np.array(f['truth'], dtype=np.float32).T
            level = int(np.array(f['level']).squeeze()) if 'level' in f else self.level
        return DataBatch(
            voltages=voltages,
            injection_patterns=injection,
            ground_truth=ground_truth,
            level=level,
            sample_id=filepath.stem,
        )
