"""Scoring module — pure-Python port of KTCssim.m / scoringFunction.m + HD95."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def _ktcssim(truth: np.ndarray, reco: np.ndarray, r: float = 80.0) -> float:
    """Gaussian-kernel SSIM matching the MATLAB KTCssim implementation."""
    c1, c2 = 1e-4, 9e-4
    t = truth.astype(np.float64)
    ra = reco.astype(np.float64)

    def _s(a: np.ndarray) -> np.ndarray:
        return gaussian_filter(a, sigma=r, mode="constant", cval=0.0, truncate=2.0)

    correction = _s(np.ones_like(t))
    mu_t = _s(t) / correction
    mu_r = _s(ra) / correction
    mu_t2, mu_r2, mu_tr = mu_t ** 2, mu_r ** 2, mu_t * mu_r
    sigma_t2 = _s(t ** 2) / correction - mu_t2
    sigma_r2 = _s(ra ** 2) / correction - mu_r2
    sigma_tr = _s(t * ra) / correction - mu_tr
    num = (2.0 * mu_tr + c1) * (2.0 * sigma_tr + c2)
    den = (mu_t2 + mu_r2 + c1) * (sigma_t2 + sigma_r2 + c2)
    return float(np.mean(num / den))


def compute_ktc_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """KTC 2023 official score: mean of per-class SSIM for resistive and conductive."""
    if pred.shape != (256, 256):
        return 0.0
    score_c = _ktcssim((gt == 2).astype(np.float64), (pred == 2).astype(np.float64))
    score_d = _ktcssim((gt == 1).astype(np.float64), (pred == 1).astype(np.float64))
    return 0.5 * (score_c + score_d)


def dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Dice score for a single class label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def iou(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """IoU score for a single class label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    denom = tp + fp + fn
    return (tp / denom) if denom > 0 else 0.0


def hd95(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """95th-percentile Hausdorff Distance for a single class label (pixels).

    Uses distance transforms for efficiency — O(N) vs O(N²) for brute force.
    Returns 0.0 if either mask is empty (no surface to measure).

    Lower is better. A score of 0.0 means perfect boundary overlap.
    """
    pred_mask = (pred == label)
    gt_mask = (gt == label)

    # If either mask is empty there is no surface — return 0 to avoid inf
    if not pred_mask.any() or not gt_mask.any():
        return 0.0

    # Distance from every gt pixel to nearest pred boundary pixel, and vice versa
    dist_pred_to_gt = distance_transform_edt(~pred_mask)   # dist of each px to pred surface
    dist_gt_to_pred = distance_transform_edt(~gt_mask)     # dist of each px to gt surface

    # Hausdorff: directed distances at surface pixels only
    hd_pred = dist_gt_to_pred[pred_mask]   # how far pred boundary is from gt
    hd_gt = dist_pred_to_gt[gt_mask]       # how far gt boundary is from pred

    all_distances = np.concatenate([hd_pred, hd_gt])
    return float(np.percentile(all_distances, 95))


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Compute all metrics for one sample.
    Dice and IoU are computed separately for resistive (1) and conductive (2).
    """
    return {
        "ktc_score": compute_ktc_score(pred, gt),
        "dice_resistive": dice(pred, gt, label=1),
        "dice_conductive": dice(pred, gt, label=2),
        "iou_resistive": iou(pred, gt, label=1),
        "iou_conductive": iou(pred, gt, label=2),
    }
