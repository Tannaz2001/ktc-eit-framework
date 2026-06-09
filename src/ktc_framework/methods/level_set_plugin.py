"""LevelSetPlugin — post-processing contour extractor for EIT reconstructions.

This is a UTILITY, not a reconstruction method. It does not implement
MethodPlugin.reconstruct() and is not registered in the MethodRegistry.

Use it after reconstruction to extract object boundaries::

    from src.ktc_framework.methods.level_set_plugin import LevelSetPlugin
    result = LevelSetPlugin().run(reconstruction)
    # result = {'contours': [...], 'n_objects': N}
"""

import logging
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import find_contours

logger = logging.getLogger(__name__)


class LevelSetPlugin:
    """Extract interfaces from a 2D reconstruction using Otsu thresholding."""

    def run(self, reconstruction: np.ndarray) -> dict:
        """Binarize the reconstruction and extract boundary contours.

        Parameters
        ----------
        reconstruction : np.ndarray
            2D array of shape (256, 256).

        Returns
        -------
        dict
            ``{'contours': list[np.ndarray], 'n_objects': int}``
        """
        if not isinstance(reconstruction, np.ndarray) or reconstruction.shape != (256, 256):
            raise ValueError("Reconstruction must be a numpy array of shape (256, 256).")

        if np.all(reconstruction == reconstruction[0, 0]):
            logger.warning("Uniform array detected; returning empty contours.")
            return {"contours": [], "n_objects": 0}

        try:
            thresh = threshold_otsu(reconstruction)
            binary = reconstruction > thresh
            contours = find_contours(binary, level=0.5)
        except Exception as exc:
            logger.error("Contour extraction failed: %s", exc)
            contours = []

        return {"contours": contours, "n_objects": len(contours)}
