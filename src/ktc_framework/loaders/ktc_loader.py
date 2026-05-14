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

import h5py
import numpy as np
import scipy.io

from src.ktc_framework.types import DataBatch


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
    VOLTAGE_SHAPE : expected (n_measurements, n_electrodes) for voltages.
    GT_SHAPE      : expected (height, width) spatial shape for ground_truth.
    VALID_LABELS  : allowed integer label values in ground_truth.
    """

    VOLTAGE_SHAPE: tuple[int, int] = (76, 30)
    GT_SHAPE: tuple[int, int] = (256, 256)
    VALID_LABELS: set[int] = {0, 1, 2}

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
        # --- voltage shape ---
        if batch.voltages.shape != KTCValidator.VOLTAGE_SHAPE:
            raise ValueError(
                f"voltages shape mismatch: expected {KTCValidator.VOLTAGE_SHAPE}, "
                f"got {batch.voltages.shape}"
            )

        # --- ground-truth spatial shape (last two dims) ---
        gt_spatial = batch.ground_truth.shape[-2:]
        if gt_spatial != KTCValidator.GT_SHAPE:
            raise ValueError(
                f"ground_truth spatial shape mismatch: expected {KTCValidator.GT_SHAPE}, "
                f"got {gt_spatial} (full shape: {batch.ground_truth.shape})"
            )

        # --- label values ---
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

    # Pattern template for filename discovery.
    _FILENAME_PATTERN = "level{level}_*.mat"

    def __init__(self, data_dir: str | Path, level: int = 1) -> None:
        self.data_dir = Path(data_dir)
        self.level = level

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, filename: str) -> DataBatch:
        """Load a single .mat file and return a validated DataBatch.

        Tries scipy.io.loadmat first; falls back to h5py for v7.3 (HDF5)
        files that loadmat cannot parse.

        Parameters
        ----------
        filename : str
            Name of the .mat file (relative to *data_dir*).

        Returns
        -------
        DataBatch
            Validated batch containing the data from *filename*.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the loaded data fails KTCValidator checks.
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Attempt v5 load; fall back to v7.3 / HDF5.
        try:
            mat = scipy.io.loadmat(
                str(filepath),
                squeeze_me=True,
                struct_as_record=False,
            )
            batch = self._parse_mat_v5(mat, filename)
        except NotImplementedError:
            # scipy raises NotImplementedError for HDF5 (.mat v7.3) files.
            batch = self._parse_mat_v73(filepath, filename)

        # Validate before handing data to the caller.
        KTCValidator.validate(batch)
        return batch

    def list_samples(self) -> List[str]:
        """Return a sorted list of .mat filenames matching the level pattern.

        Scans *data_dir* (non-recursively) for files matching
        ``level{self.level}_*.mat``.

        Returns
        -------
        List[str]
            Sorted filenames (not full paths).
        """
        pattern = self._FILENAME_PATTERN.format(level=self.level)
        matches = [
            entry.name
            for entry in self.data_dir.iterdir()
            if entry.is_file() and fnmatch.fnmatch(entry.name, pattern)
        ]
        return sorted(matches)

    # ------------------------------------------------------------------
    # Private parsing helpers
    # ------------------------------------------------------------------

    def _parse_mat_v5(self, mat: dict, filename: str) -> DataBatch:
        """Extract arrays from a scipy.io.loadmat result dict."""
        # Key names follow the KTC dataset convention.
        voltages = np.asarray(mat['Uel'], dtype=np.float32)          # (76, 30)
        injection = np.asarray(mat['Injref'], dtype=np.float32)       # (n_patterns, n_electrodes)
        ground_truth = np.asarray(mat['truth'], dtype=np.float32)     # (256, 256)
        level = int(mat.get('level', self.level))
        sample_id = Path(filename).stem

        return DataBatch(
            voltages=voltages,
            injection_patterns=injection,
            ground_truth=ground_truth,
            level=level,
            sample_id=sample_id,
        )

    def _parse_mat_v73(self, filepath: Path, filename: str) -> DataBatch:
        """Extract arrays from an HDF5-based .mat v7.3 file using h5py."""
        with h5py.File(str(filepath), 'r') as f:
            # h5py stores arrays in C order; transpose to match MATLAB layout.
            voltages = np.array(f['Uel'], dtype=np.float32).T          # (76, 30)
            injection = np.array(f['Injref'], dtype=np.float32).T      # (n_patterns, n_electrodes)
            ground_truth = np.array(f['truth'], dtype=np.float32).T    # (256, 256)
            level = int(np.array(f['level']).squeeze()) if 'level' in f else self.level

        sample_id = filepath.stem

        return DataBatch(
            voltages=voltages,
            injection_patterns=injection,
            ground_truth=ground_truth,
            level=level,
            sample_id=sample_id,
        )
