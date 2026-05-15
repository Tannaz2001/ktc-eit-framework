import logging
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import find_contours

logger = logging.getLogger(__name__)

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

        logger.info("Starting LevelSetPlugin.run with reconstruction shape: %s", reconstruction.shape)

        # Check for uniform arrays where Otsu's method will fail
        if np.all(reconstruction == reconstruction[0, 0]):
            logger.warning("Uniform array detected; returning empty contours.")
            return {'contours': [], 'n_objects': 0}

        try:
            # Determine threshold and binarize
            thresh = threshold_otsu(reconstruction)
            logger.debug("Computed Otsu threshold: %s", thresh)
            
            binary = reconstruction > thresh
            
            # Find contours
            contours = find_contours(binary, level=0.5)
            logger.info("Extracted %d contours.", len(contours))
        except Exception as e:
            # Fallback if something goes wrong during thresholding/contour finding
            logger.error("Exception during contour extraction: %s", e)
            contours = []

        return {
            'contours': contours,
            'n_objects': len(contours)
        }
