from __future__ import annotations

import uuid
from typing import Iterator, Optional

import numpy as np

from src.ktc_framework.types import DataBatch
from src.ktc_framework.adapters.method_registry import register


@register
class MockDataPlugin:
    """Generates synthetic EIT DataBatch objects for testing and development.

    Usage
    -----
    plugin = MockDataPlugin(n_electrodes=16, spatial_dims=(32, 32), seed=42)
    for batch in plugin.iter_batches(n_batches=10, level=1):
        ...
    """

    def __init__(
        self,
        n_electrodes: int = 16,
        n_patterns: int = 13,
        spatial_dims: tuple[int, ...] = (32, 32),
        seed: Optional[int] = None,
    ) -> None:
        self.n_electrodes = n_electrodes
        self.n_patterns = n_patterns
        self.spatial_dims = spatial_dims
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_batch(self, n_samples: int = 1, level: int = 1, sample_id: Optional[str] = None) -> DataBatch:
        """Return a single synthetic DataBatch."""
        voltages = self._rng.standard_normal((n_samples, self.n_electrodes)).astype(np.float32)
        injection_patterns = self._make_injection_patterns()
        ground_truth = self._rng.uniform(0.1, 2.0, size=(n_samples, *self.spatial_dims)).astype(np.float32)
        sid = sample_id if sample_id is not None else str(uuid.uuid4())
        return DataBatch(
            voltages=voltages,
            injection_patterns=injection_patterns,
            ground_truth=ground_truth,
            level=level,
            sample_id=sid,
        )

    def iter_batches(
        self,
        n_batches: int,
        n_samples: int = 1,
        level: int = 1,
    ) -> Iterator[DataBatch]:
        """Yield *n_batches* synthetic DataBatch objects."""
        for i in range(n_batches):
            yield self.get_batch(n_samples=n_samples, level=level, sample_id=f"mock-{i:04d}")

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the internal RNG (pass None for a random seed)."""
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_injection_patterns(self) -> np.ndarray:
        """Build an adjacent-pair injection pattern matrix (n_patterns × n_electrodes)."""
        patterns = np.zeros((self.n_patterns, self.n_electrodes), dtype=np.float32)
        for i in range(self.n_patterns):
            source = i % self.n_electrodes
            sink = (i + 1) % self.n_electrodes
            patterns[i, source] = 1.0
            patterns[i, sink] = -1.0
        return patterns
