"""
back_projection_plugin.py
--------------------------
Simple linear back-projection EIT reconstruction.

Algorithm
---------
1. Reshape the flat voltage vector (2356,) into a (76, 31) matrix where
   rows = injection patterns, columns = voltage measurements.
2. Sum across injections to get a 1-D sensitivity profile (31,).
3. Interpolate to a 256×256 image using radial basis interpolation from
   the 32-electrode ring geometry.
4. Apply Gaussian smoothing to reduce artefacts.
5. Threshold into three classes:
     0 = background  (within ±1 std of mean)
     1 = resistive   (significantly below mean — lower conductivity)
     2 = conductive  (significantly above mean — higher conductivity)

This is a simplified reconstruction — not a full FEM solver — but it
produces spatially meaningful output that varies with the input voltages
and can be scored against real ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.types import DataBatch

# Electrode positions on a unit circle (KTC: 32 electrodes, equally spaced)
_N_ELECTRODES = 32
_ANGLES = np.linspace(0, 2 * np.pi, _N_ELECTRODES, endpoint=False)
_ELEC_X = np.cos(_ANGLES)   # (32,)
_ELEC_Y = np.sin(_ANGLES)   # (32,)

# Image grid coordinates in [-1, 1]
_IMG_SIZE = 256
_gx, _gy = np.meshgrid(
    np.linspace(-1, 1, _IMG_SIZE),
    np.linspace(-1, 1, _IMG_SIZE),
)
_CIRCLE_MASK = (_gx ** 2 + _gy ** 2) <= 1.0  # pixels inside the electrode ring


@register
class BackProjectionPlugin:
    """Linear back-projection EIT reconstruction.

    Reconstructs a 256×256 segmentation map from the voltage vector by
    interpolating electrode-level sensitivity values onto the image grid
    and thresholding into background / resistive / conductive classes.

    Parameters
    ----------
    smooth_sigma : float
        Gaussian smoothing sigma applied after interpolation (pixels).
    threshold_std : float
        Number of standard deviations from the mean used to separate
        resistive (below) and conductive (above) from background.
    """

    def __init__(self, smooth_sigma: float = 8.0, threshold_std: float = 0.8) -> None:
        self.smooth_sigma = smooth_sigma
        self.threshold_std = threshold_std

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Return a (256, 256) uint8 segmentation with labels {0, 1, 2}.

        Parameters
        ----------
        batch : DataBatch
            Input containing voltages (2356,) and injection_patterns (32, 76).
        """
        voltages = batch.voltages.astype(np.float64)          # (2356,)
        inj = batch.injection_patterns.astype(np.float64)     # (32, 76)

        # --- Step 1: reshape voltages to (76 injections, 31 measurements) ---
        v_matrix = voltages.reshape(76, 31)                   # (76, 31)

        # --- Step 2: compute per-electrode sensitivity by back-projecting
        #     each injection pattern weighted by its voltage response ---
        # inj: (32 electrodes, 76 injections) → transpose to (76, 32)
        inj_t = inj.T                                         # (76, 32)

        # Mean voltage magnitude per injection
        v_magnitude = np.abs(v_matrix).mean(axis=1)           # (76,)

        # Weighted sum over injections → sensitivity per electrode (32,)
        elec_sensitivity = inj_t.T @ v_magnitude              # (32,)

        # Normalise to [-1, 1]
        s_min, s_max = elec_sensitivity.min(), elec_sensitivity.max()
        if s_max > s_min:
            elec_sensitivity = 2.0 * (elec_sensitivity - s_min) / (s_max - s_min) - 1.0
        else:
            elec_sensitivity = np.zeros(_N_ELECTRODES)

        # --- Step 3: interpolate electrode values onto 256×256 grid ---
        points = np.stack([_ELEC_X, _ELEC_Y], axis=1)        # (32, 2)
        image = griddata(
            points,
            elec_sensitivity,
            (_gx, _gy),
            method="cubic",
            fill_value=0.0,
        )                                                      # (256, 256)

        # Zero outside the electrode ring
        image[~_CIRCLE_MASK] = 0.0

        # --- Step 4: smooth ---
        image = gaussian_filter(image, sigma=self.smooth_sigma)

        # --- Step 5: threshold into {0, 1, 2} ---
        seg = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)
        inside = _CIRCLE_MASK
        mu = image[inside].mean()
        std = image[inside].std()

        if std > 0:
            seg[inside & (image < mu - self.threshold_std * std)] = 1   # resistive
            seg[inside & (image > mu + self.threshold_std * std)] = 2   # conductive

        return seg
