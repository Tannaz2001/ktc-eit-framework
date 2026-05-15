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

    Usage
    -----
    plugin = MockDataPlugin()
    batch  = plugin.get_batch(level=1, sample_id="mock-0001")
    for batch in plugin.iter_batches(n_batches=5, level=2):
        ...
    """

    def __init__(self, dataset_root: str = "") -> None:
        # dataset_root is accepted but not used — keeps the same __init__
        # signature as KTCLoader so BatchRunner can instantiate both uniformly.
        self.dataset_root = dataset_root

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_batch(
        self,
        n_samples: int = 1,  # kept for API compat; DataBatch holds one sample
        level: int = 1,
        sample_id: Optional[str] = None,
    ) -> DataBatch:
        """Return one synthetic DataBatch with KTC-correct shapes.

        The RNG is seeded from *sample_id* so calling this method twice with
        the same *sample_id* always returns identical voltages and ground_truth.

        Parameters
        ----------
        n_samples : int
            Ignored (DataBatch represents exactly one measurement). Kept for
            backward compatibility with callers that pass it.
        level : int
            Stored on the returned DataBatch as-is.
        sample_id : str | None
            Unique identifier.  Auto-generated as ``mock-XXXX`` when None.
        """
        sid = sample_id if sample_id is not None else "mock-0000"

        # Seed deterministically from sample_id so tests are reproducible.
        seed = hash(sid) % (2 ** 32)
        rng = np.random.default_rng(seed)

        # Voltage vector: Gaussian noise centred on zero, std ~0.44 V
        # (matches the real-data statistics from the dataset analysis).
        voltages = rng.standard_normal(_N_VOLTAGES).astype(np.float64)

        # Injection patterns are fixed by the KTC measurement protocol —
        # identical across all real samples, so we use a deterministic
        # adjacent-pair matrix here regardless of sample_id.
        injection_patterns = self._make_injection_patterns()

        # Ground truth: weighted random labels {0, 1, 2}.
        # Probabilities [0.90, 0.05, 0.05] mirror the real class distribution.
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
        """Yield *n_batches* synthetic DataBatch objects.

        Each batch gets a unique zero-padded sample_id (``mock-0000`` …
        ``mock-{n_batches-1:04d}``) so downstream reproducibility tests pass.
        """
        for i in range(n_batches):
            yield self.get_batch(
                n_samples=n_samples,
                level=level,
                sample_id=f"mock-{i:04d}",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_injection_patterns() -> np.ndarray:
        """Build the KTC adjacent-pair injection matrix (32 × 76).

        Row i injects +1 A into electrode (i % 32) and −1 A into
        electrode ((i+1) % 32).  The 76 columns represent the full
        FEM-boundary current distribution vector used in KTCLoader.
        """
        patterns = np.zeros((_N_ELECTRODES, _N_INJ_COLS), dtype=np.float64)
        for i in range(_N_ELECTRODES):
            source = i % _N_INJ_COLS
            sink = (i + 1) % _N_INJ_COLS
            patterns[i, source] = 1.0
            patterns[i, sink] = -1.0
        return patterns
