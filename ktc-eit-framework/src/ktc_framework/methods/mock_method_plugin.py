"""MockMethodPlugin — returns a dummy 256x256 segmentation for testing."""

from __future__ import annotations

import numpy as np
from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.types import DataBatch


@register
class MockMethodPlugin:
    """
    Fake reconstruction method that returns a valid 256x256 segmentation.
    All pixels are labelled 0 (water) — wrong but structurally valid.
    Used to test scoring, visualization, and reporting before real methods exist.
    """

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        return np.zeros((256, 256), dtype=np.int32)
