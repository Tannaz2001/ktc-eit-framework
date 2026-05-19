from abc import ABC, abstractmethod
import numpy as np

class MethodPlugin(ABC):
    """Abstract base class for reconstruction methods."""

    @abstractmethod
    def reconstruct(self, batch) -> np.ndarray:
        """Reconstruct a 256x256 segmentation map from DataBatch."""
        pass

    def validate_output(self, output):
        if output.shape != (256, 256):
            raise ValueError("Output must be 256x256")
        if not np.all(np.isin(output, [0, 1, 2])):
            raise ValueError("Output labels must be 0,1,2")