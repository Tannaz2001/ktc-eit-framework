"""
training_data_plugin.py
-----------------------
Loads the KTC training data from the local Codes_Matlab/ layout:

    Codes_Matlab/TrainingData/data1.mat  →  voltages + injection_patterns
    Codes_Matlab/GroundTruths/true1.mat  →  ground_truth

Samples are numbered 1–4.  The runner maps them via the config:
    samples: ["1", "2", "3", "4"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from src.ktc_framework.loaders.ktc_loader import PluginRegistry
from src.ktc_framework.types import DataBatch


@PluginRegistry.register('TrainingDataPlugin')
class TrainingDataPlugin:
    """Loads KTC training samples from Codes_Matlab/ folder layout.

    Parameters
    ----------
    dataset_root : str
        Path to the folder that contains TrainingData/ and GroundTruths/.
        Typically ``Codes_Matlab/``.
    """

    def __init__(self, dataset_root: str = "Codes_Matlab") -> None:
        self.root = Path(dataset_root)

    def load_sample(self, level: int, sample: str) -> DataBatch:
        """Load one training sample.

        Parameters
        ----------
        level : int
            Ignored — all training samples are the same difficulty.
        sample : str
            Sample number as a string: "1", "2", "3", or "4".
        """
        data_path = self.root / "TrainingData" / f"data{sample}.mat"
        truth_path = self.root / "GroundTruths" / f"true{sample}.mat"

        if not data_path.exists():
            raise FileNotFoundError(f"Voltage file not found: {data_path}")
        if not truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {truth_path}")

        data_mat = scipy.io.loadmat(str(data_path), squeeze_me=True)
        truth_mat = scipy.io.loadmat(str(truth_path), squeeze_me=True)

        voltages = np.asarray(data_mat['Uel'], dtype=np.float64)          # (2356,)
        injection = np.asarray(data_mat['Inj'], dtype=np.float64)         # (32, 76)
        ground_truth = np.asarray(truth_mat['truth'], dtype=np.uint8)     # (256, 256)

        return DataBatch(
            voltages=voltages,
            injection_patterns=injection,
            ground_truth=ground_truth,
            level=level,
            sample_id=f"training_{sample}",
        )
