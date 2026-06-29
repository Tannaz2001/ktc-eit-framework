"""Runtime adapter for reconstruction method plugins."""

from __future__ import annotations

import numpy as np

from src.ktc_framework.types import DataBatch


class MethodAdapter:
    """Normalize method execution and reconstruction output."""

    def __init__(self, method_instance: object) -> None:
        self.method = method_instance
        self.name = method_instance.__class__.__name__

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Run a method and return a validated 256x256 uint8 reconstruction."""
        if not hasattr(self.method, "reconstruct"):
            raise TypeError(f"{self.name} must implement reconstruct(batch)")

        result = self.method.reconstruct(batch)

        if isinstance(result, dict):
            result = result.get("reconstruction")

        if result is None:
            raise ValueError(f"{self.name} returned no reconstruction")

        reconstruction = np.asarray(result)
        if reconstruction.shape != (256, 256):
            raise ValueError(
                f"{self.name} must return shape (256, 256), "
                f"got {reconstruction.shape}"
            )

        if np.any(np.isnan(reconstruction)):
            raise ValueError(f"{self.name} returned reconstruction containing NaN values")

        if np.any(np.isinf(reconstruction)):
            raise ValueError(f"{self.name} returned reconstruction containing Inf values")

        return reconstruction.astype(np.uint8, copy=False)
