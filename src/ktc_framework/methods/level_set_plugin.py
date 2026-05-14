import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import find_contours

class LevelSetPlugin:
    """
    Extracts interfaces from a 2D reconstruction using Otsu's thresholding 
    and contour finding.
    """

    def run(self, reconstruction: np.ndarray) -> dict:
        """
        Binarizes the input reconstruction and extracts interfaces.

        Args:
            reconstruction (np.ndarray): 2D array of shape (256, 256).

        Returns:
            dict: Contains 'contours' (list of contour arrays) and 'n_objects' (int).
        """
        if not isinstance(reconstruction, np.ndarray) or reconstruction.shape != (256, 256):
            raise ValueError("Reconstruction must be a numpy array of shape (256, 256).")

        # Check for uniform arrays where Otsu's method will fail
        if np.all(reconstruction == reconstruction[0, 0]):
            return {'contours': [], 'n_objects': 0}

        try:
            # Determine threshold and binarize
            thresh = threshold_otsu(reconstruction)
            binary = reconstruction > thresh
            
            # Find contours
            contours = find_contours(binary, level=0.5)
        except Exception:
            # Fallback if something goes wrong during thresholding/contour finding
            contours = []

        return {
            'contours': contours,
            'n_objects': len(contours)
        }
