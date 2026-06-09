from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch
import numpy as np

@register
class GroundTruthOracle(MethodPlugin):
    """Registered GroundTruth Oracle that returns the exact ground truth."""

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        # Return exact ground truth for debugging or evaluation purposes
        gt = np.asarray(batch.ground_truth, dtype=int)

        if gt.shape != (256, 256):
            raise ValueError(f"Expected ground truth shape (256,256), got {gt.shape}")

        self.validate_output(gt)
        return gt