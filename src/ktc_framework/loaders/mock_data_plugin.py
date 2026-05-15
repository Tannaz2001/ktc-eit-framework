"""
mock_data_plugin.py
-------------------
Generates synthetic EIT DataBatch objects whose array shapes exactly match
the real KTC 2023 dataset:

    voltages           (2356,)    float64   — 76 injections × 31 voltage pairs
    injection_patterns (32, 76)   float64   — adjacent-pair current protocol
    ground_truth       (256, 256) uint8     — labels {0, 1, 2}

RNG is seeded from sample_id so the same sample_id always produces identical
data — essential for deterministic unit tests.
"""

from __future__ import annotations

from typing import Generator, Optional

import numpy as np

from src.ktc_framework.loaders.ktc_loader import PluginRegistry
from src.ktc_framework.types import DataBatch

# KTC-realistic class weights: background dominates at ~90 %
_LABEL_PROBS = [0.90, 0.05, 0.05]

# Injection protocol shape — fixed by the KTC dataset
_N_ELECTRODES = 32
_N_INJ_COLS = 76
_N_VOLTAGES = 2356  # 76 injection patterns × 31 voltage pairs


@PluginRegistry.register('MockDataPlugin')
class MockDataPlugin:
    """Synthetic DataBatch generator with KTC-correct array shapes.

    Parameters
    ----------
    dataset_root : str
        Accepted for interface compatibility with KTCLoader; unused here.
    """

    def __init__(self, dataset_root: str = "") -> None:
        self.dataset_root = dataset_root

    def get_batch(
        self,
        n_samples: int = 1,
        level: int = 1,
        sample_id: Optional[str] = None,
    ) -> DataBatch:
        """Return one synthetic DataBatch with KTC-correct shapes.

        RNG is seeded from *sample_id* so identical inputs always return
        identical data.
        """
        sid = sample_id if sample_id is not None else "mock-0000"
        seed = hash(sid) % (2 ** 32)
        rng = np.random.default_rng(seed)

        voltages = rng.standard_normal(_N_VOLTAGES).astype(np.float64)
        injection_patterns = self._make_injection_patterns()
        gt_flat = rng.choice([0, 1, 2], size=(256 * 256), p=_LABEL_PROBS)
        ground_truth = gt_flat.reshape(256, 256).astype(np.uint8)

        return DataBatch(
            voltages=voltages,
            injection_patterns=injection_patterns,
            ground_truth=ground_truth,
            level=level,
            sample_id=sid,
        )

    def load_sample(self, level: int, sample: str) -> DataBatch:
        """Unified interface matching KTCLoader.load_sample().

        Constructs a deterministic sample_id from *level* and *sample* so
        BatchRunner can call this method on either plugin transparently.
        """
        return self.get_batch(level=level, sample_id=f"mock_level{level}_{sample}")

    def iter_batches(
        self,
        n_batches: int,
        n_samples: int = 1,
        level: int = 1,
    ) -> Generator[DataBatch, None, None]:
        """Yield *n_batches* synthetic DataBatch objects."""
        for i in range(n_batches):
            yield self.get_batch(
                n_samples=n_samples,
                level=level,
                sample_id=f"mock-{i:04d}",
            )

    @staticmethod
    def _make_injection_patterns() -> np.ndarray:
        """Build the KTC adjacent-pair injection matrix (32 × 76)."""
        patterns = np.zeros((_N_ELECTRODES, _N_INJ_COLS), dtype=np.float64)
        for i in range(_N_ELECTRODES):
            source = i % _N_INJ_COLS
            sink = (i + 1) % _N_INJ_COLS
            patterns[i, source] = 1.0
            patterns[i, sink] = -1.0
        return patterns
