"""MockDataPlugin — returns synthetic voltage data for testing the pipeline."""

from __future__ import annotations

import numpy as np
from src.ktc_framework.adapters.method_registry import register


@register
class MockDataPlugin:
    """
    Fake data loader that returns random voltage arrays.
    Used to test the full pipeline before the real KTC loader is ready.
    Input shape matches KTC Level 1: 76 injections x 30 measurements.
    """

    def load(self, level: int, sample: str) -> dict:
        rng = np.random.default_rng(seed=level + ord(sample))
        return {
            "voltages": rng.random((76, 30)).astype(np.float32),
            "level": level,
            "sample": sample,
        }
