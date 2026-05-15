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
import os
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
    """Maps string names to plugin classes.

    Usage
    -----
    @PluginRegistry.register('my_plugin')
    class MyPlugin:
        ...

    cls = PluginRegistry.get('my_plugin')
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that stores the decorated class under *name*."""
        def decorator(plugin_cls: type) -> type:
            cls._registry[name] = plugin_cls
            return plugin_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Return the class registered under *name*.

        Raises
        ------
        KeyError
            If *name* is not registered; the message lists available names.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise KeyError(
                f"Plugin '{name}' not found. Available plugins: {available}"
            )
        return cls._registry[name]


# ---------------------------------------------------------------------------
# KTCValidator
# ---------------------------------------------------------------------------

class KTCValidator:
    """Static validation rules for KTC DataBatch objects.

    Constants
    ---------
    VOLTAGE_SHAPE : expected flat shape (2356,) = 76 injections × 31 voltage pairs.
    GT_SHAPE      : expected (height, width) spatial shape for ground_truth.
    VALID_LABELS  : allowed integer label values in ground_truth.
    """

    # 76 injection patterns × 31 differential voltage pairs = 2356
    VOLTAGE_SHAPE: tuple = (2356,)
    GT_SHAPE: tuple = (256, 256)
    VALID_LABELS: set = {0, 1, 2}

    @staticmethod
    def validate(batch: DataBatch) -> None:
        """Validate a DataBatch against KTC dataset constraints.

        Checks
        ------
        1. voltages.shape == VOLTAGE_SHAPE
        2. ground_truth.shape[-2:] == GT_SHAPE  (allows leading batch dim)
        3. All values in ground_truth are in VALID_LABELS

        Raises
        ------
        ValueError
            Descriptive message for the first constraint that is violated.
        """
        if batch.voltages.shape != KTCValidator.VOLTAGE_SHAPE:
            raise ValueError(
                f"voltages shape mismatch: expected {KTCValidator.VOLTAGE_SHAPE}, "
                f"got {batch.voltages.shape}"
            )

        gt_spatial = batch.ground_truth.shape[-2:]
        if gt_spatial != KTCValidator.GT_SHAPE:
            raise ValueError(
                f"ground_truth spatial shape mismatch: expected {KTCValidator.GT_SHAPE}, "
                f"got {gt_spatial} (full shape: {batch.ground_truth.shape})"
            )

        unique_labels = set(np.unique(batch.ground_truth).astype(int).tolist())
        invalid = unique_labels - KTCValidator.VALID_LABELS
        if invalid:
            raise ValueError(
                f"ground_truth contains invalid label(s): {sorted(invalid)}. "
                f"Valid labels are {sorted(KTCValidator.VALID_LABELS)}."
            )


# ---------------------------------------------------------------------------
# KTCLoader
# ---------------------------------------------------------------------------

@PluginRegistry.register('ktc_loader')
class KTCLoader:
    """Loads KTC EIT dataset samples from .mat files.

    Supports both MATLAB v5 (scipy.io.loadmat) and v7.3 / HDF5 (h5py) files.
    Every loaded sample is validated with KTCValidator before being returned.

    Parameters
    ----------
    data_dir : str | Path
        Root directory that contains the .mat files.
    level : int
        Difficulty level used to filter filenames (matches ``level{level}_*.mat``).
    """

    _FILENAME_PATTERN = "level{level}_*.mat"

    def __init__(self, data_dir: str | Path, level: int = 1) -> None:
        self.data_dir = Path(data_dir)
        self.level = level

    def load(self, filename: str) -> DataBatch:
        """Load a single .mat file and return a validated DataBatch."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            mat = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
            batch = self._parse_mat_v5(mat, filename)
        except NotImplementedError:
            batch = self._parse_mat_v73(filepath, filename)
        KTCValidator.validate(batch)
        return batch

    def load_sample(self, level: int, sample: str) -> DataBatch:
        """Load a sample by level number and sample letter (A/B/C).

        Constructs the canonical KTC filename ``level{level}_{sample}.mat``
        and delegates to :meth:`load`.
        """
        filename = f"level{level}_{sample}.mat"
        return self.load(filename)

    def list_samples(self) -> List[str]:
        """Return a sorted list of .mat filenames matching the level pattern."""
        pattern = self._FILENAME_PATTERN.format(level=self.level)
        return sorted(
            e.name for e in self.data_dir.iterdir()
            if e.is_file() and fnmatch.fnmatch(e.name, pattern)
        )

    def _parse_mat_v5(self, mat: dict, filename: str) -> DataBatch:
        """Extract arrays from a scipy.io.loadmat result dict."""
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
        """Extract arrays from an HDF5-based .mat v7.3 file using h5py."""
        if _h5py is None:
            raise ImportError("h5py is required to load MATLAB v7.3 files. Run: pip install h5py")
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
