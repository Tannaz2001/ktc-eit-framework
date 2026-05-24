import numpy as np
from skimage.filters import threshold_otsu


def segment(sigma: np.ndarray) -> np.ndarray:
    """Convert a 256x256 float conductivity map into discrete labels.

    Labels: 0 = water (background), 1 = resistive, 2 = conductive.

    NaN and inf values (produced by griddata outside the mesh hull) are
    replaced with 0.0 before thresholding so Otsu never receives a
    non-finite histogram range.
    """
    if sigma.ndim != 2 or sigma.shape != (256, 256):
        raise ValueError("Input must be a 256x256 array")

    # Sanitise: replace NaN / ±inf with 0.0 (background value)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    t1 = threshold_otsu(sigma)
    t2 = threshold_otsu(sigma[sigma > t1]) if np.any(sigma > t1) else t1 + 1e-6

    labels = np.zeros_like(sigma, dtype=int)
    labels[sigma > t1] = 1
    labels[sigma > t2] = 2
    return labels