"""
ktc_score.py
------------
Python port of the official KTC 2023 MATLAB scoring functions:
  - KTCssim.m  →  _ktcssim(truth, reco, r)
  - scoringFunction.m  →  compute_ktc_score(pred, gt)

No external KTCScoring module required. Uses scipy.ndimage.gaussian_filter
with truncate=2.0 to match the MATLAB kernel window (ws = ceil(2*r)).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Internal SSIM helper  (port of KTCssim.m)
# ---------------------------------------------------------------------------

def _ktcssim(truth: np.ndarray, reco: np.ndarray, r: float = 80.0) -> float:
    """Gaussian-kernel SSIM score — direct port of KTCssim.m.

    Parameters
    ----------
    truth, reco : np.ndarray
        Binary float64 arrays of identical shape (0.0 / 1.0 values).
    r : float
        Gaussian radius in pixels.  MATLAB default is 80.

    Returns
    -------
    float
        Mean SSIM value across the image, range roughly [−1, 1].

    Notes
    -----
    MATLAB truncates the kernel at ws = ceil(2*r) pixels from the centre,
    equivalent to 2 standard deviations.  scipy.ndimage.gaussian_filter is
    used with truncate=2.0 to reproduce this exactly.

    The correction array (gaussian filter of an all-ones image) handles
    boundary effects the same way MATLAB's conv2(...,'same')/correction does.
    """
    c1, c2 = 1e-4, 9e-4

    t = truth.astype(np.float64)
    r_arr = reco.astype(np.float64)

    def _smooth(arr: np.ndarray) -> np.ndarray:
        return gaussian_filter(arr, sigma=r, mode="constant", cval=0.0, truncate=2.0)

    correction = _smooth(np.ones_like(t))

    # Local means
    mu_t = _smooth(t) / correction
    mu_r = _smooth(r_arr) / correction

    mu_t2 = mu_t ** 2
    mu_r2 = mu_r ** 2
    mu_tr = mu_t * mu_r

    # Local variances / covariance
    sigma_t2 = _smooth(t ** 2) / correction - mu_t2
    sigma_r2 = _smooth(r_arr ** 2) / correction - mu_r2
    sigma_tr = _smooth(t * r_arr) / correction - mu_tr

    num = (2.0 * mu_tr + c1) * (2.0 * sigma_tr + c2)
    den = (mu_t2 + mu_r2 + c1) * (sigma_t2 + sigma_r2 + c2)

    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Public scoring entry point  (port of scoringFunction.m)
# ---------------------------------------------------------------------------

def compute_ktc_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Official KTC 2023 score — Python port of scoringFunction.m.

    Computes per-class Gaussian-SSIM for conductive (label 2) and resistive
    (label 1) binary masks, then averages them:
        score = 0.5 * (ssim_conductive + ssim_resistive)

    Parameters
    ----------
    pred, gt : np.ndarray
        256×256 integer arrays with labels {0, 1, 2}.

    Returns
    -------
    float
        Score in roughly [0, 1].  Returns 0.0 if pred is not 256×256.
    """
    if pred.shape != (256, 256):
        return 0.0
    if gt.shape != (256, 256):
        raise ValueError(f"gt must be (256, 256), got {gt.shape}")

    # Conductive inclusion  (label 2)
    score_c = _ktcssim((gt == 2).astype(np.float64), (pred == 2).astype(np.float64))

    # Resistive inclusion  (label 1)
    score_d = _ktcssim((gt == 1).astype(np.float64), (pred == 1).astype(np.float64))

    return 0.5 * (score_c + score_d)


# ---------------------------------------------------------------------------
# Per-class Dice and IoU
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Dice score for a single class label."""
    pred_mask = pred == label
    gt_mask = gt == label
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def iou(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """IoU score for a single class label."""
    pred_mask = pred == label
    gt_mask = gt == label
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    denom = tp + fp + fn
    return (tp / denom) if denom > 0 else 0.0


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute all metrics for one sample."""
    return {
        "ktc_score":       compute_ktc_score(pred, gt),
        "dice_resistive":  dice(pred, gt, label=1),
        "dice_conductive": dice(pred, gt, label=2),
        "iou_resistive":   iou(pred, gt, label=1),
        "iou_conductive":  iou(pred, gt, label=2),
    }
