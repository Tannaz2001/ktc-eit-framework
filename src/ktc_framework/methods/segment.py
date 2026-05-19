import numpy as np
from skimage.filters import threshold_otsu

def segment(sigma: np.ndarray) -> np.ndarray:
    """Convert 256x256 float map into discrete labels {0=water,1=resistive,2=conductive}."""
    if sigma.ndim != 2 or sigma.shape != (256, 256):
        raise ValueError("Input must be 256x256 array")

    t1 = threshold_otsu(sigma)
    t2 = threshold_otsu(sigma[sigma > t1]) if np.any(sigma > t1) else t1 + 1e-6

    labels = np.zeros_like(sigma, dtype=int)
    labels[sigma > t1] = 1
    labels[sigma > t2] = 2
    return labels