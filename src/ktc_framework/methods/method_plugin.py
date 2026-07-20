from abc import ABC, abstractmethod
import numpy as np

class MethodPlugin(ABC):
    """Abstract base class for reconstruction methods.

    Subclass this and implement ``reconstruct()``.  The framework calls
    ``validate_output()`` automatically via ``MethodAdapter`` after every
    reconstruction, so implementations do not need to validate themselves.
    """

    @abstractmethod
    def reconstruct(self, batch) -> np.ndarray:
        """Reconstruct a 256x256 segmentation map from EIT measurements.

        Parameters
        ----------
        batch : DataBatch
            Holds voltages, injection_patterns, ground_truth, level (1–7),
            sample_id, mesh, reference_voltages, measurement_patterns.

        Returns
        -------
        numpy.ndarray, shape (256, 256), dtype convertible to uint8
            Pixel-wise segmentation labels:
              0 = background (water/tank wall)
              1 = resistive inclusion (plastic)
              2 = conductive inclusion (metal)
        """
        pass

    def validate_output(self, output):
        if output.shape != (256, 256):
            raise ValueError("Output must be 256x256")
        if not np.all(np.isin(output, [0, 1, 2])):
            raise ValueError("Output labels must be 0,1,2")