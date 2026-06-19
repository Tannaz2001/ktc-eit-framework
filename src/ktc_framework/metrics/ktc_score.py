"""Scoring module — pure-Python port of KTCssim.m / scoringFunction.m."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


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


# ---------------------------------------------------------------------------
# KTC score — two flavours, clearly separated
# ---------------------------------------------------------------------------

def compute_ktc_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """KTC 2023 official score — normalized against the all-water baseline.

    Formula (matches scoringFunction.m):
        score = 0.5 * (norm_ssim_resistive + norm_ssim_conductive)

    where for each class:
        norm_ssim = (SSIM(pred, gt) - SSIM(zeros, gt)) / (1 - SSIM(zeros, gt))

    This ensures:
      - All-water (MockPlugin) baseline always scores exactly 0.0
      - Perfect prediction scores 1.0
      - Predictions worse than all-water score negative (e.g. -0.010)

    Edge case: when a class is absent from both pred and gt the denominator
    collapses to 0; that class contributes 0.0 to keep the all-water
    baseline anchored at 0.0.
    """
    if pred.shape != (256, 256):
        return 0.0

    zeros = np.zeros((256, 256), dtype=np.float64)

    gt_res  = (gt == 1).astype(np.float64)
    gt_cond = (gt == 2).astype(np.float64)
    pr_res  = (pred == 1).astype(np.float64)
    pr_cond = (pred == 2).astype(np.float64)

    ssim_res  = _ktcssim(gt_res,  pr_res)
    ssim_cond = _ktcssim(gt_cond, pr_cond)

    base_res  = _ktcssim(gt_res,  zeros)
    base_cond = _ktcssim(gt_cond, zeros)

    denom_res  = 1.0 - base_res
    denom_cond = 1.0 - base_cond

    norm_res  = (ssim_res  - base_res)  / denom_res  if denom_res  > 1e-9 else 0.0
    norm_cond = (ssim_cond - base_cond) / denom_cond if denom_cond > 1e-9 else 0.0

    return round(0.5 * (norm_res + norm_cond), 6)


def compute_ktc_score_raw(pred: np.ndarray, gt: np.ndarray) -> float:
    """Raw (un-normalized) KTC SSIM: 0.5 * (ssim_resistive + ssim_conductive).

    Does NOT subtract the all-water baseline. Useful for debugging and
    cross-checking against the reference MATLAB scoringFunction.m.

    Edge case: when a class is absent in both pred and gt, that class
    contributes 0.5 rather than 1.0 so trivially-empty classes do not
    inflate the combined score.
    """
    if pred.shape != (256, 256):
        return 0.0

    gt_res  = (gt == 1).astype(np.float64)
    gt_cond = (gt == 2).astype(np.float64)
    pr_res  = (pred == 1).astype(np.float64)
    pr_cond = (pred == 2).astype(np.float64)

    if not np.any(gt_res) and not np.any(pr_res):
        ssim_res = 0.5
    else:
        ssim_res = _ktcssim(gt_res, pr_res)

    if not np.any(gt_cond) and not np.any(pr_cond):
        ssim_cond = 0.5
    else:
        ssim_cond = _ktcssim(gt_cond, pr_cond)

    return round(0.5 * (ssim_res + ssim_cond), 6)


# ---------------------------------------------------------------------------
# Dice and IoU — diagnostic metrics (not used in composite ranking)
# ---------------------------------------------------------------------------

def _dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = pred == label
    g = gt == label
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    tp = int((p & g).sum())
    return float(2 * tp / (int(p.sum()) + int(g.sum())))


def _iou(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = pred == label
    g = gt == label
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    intersection = int((p & g).sum())
    union = int((p | g).sum())
    return float(intersection / union)


def dice_resistive(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for resistive class (label=1)."""
    return _dice(pred, gt, 1)


def dice_conductive(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for conductive class (label=2)."""
    return _dice(pred, gt, 2)


def iou_resistive(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union for resistive class (label=1)."""
    return _iou(pred, gt, 1)


def iou_conductive(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union for conductive class (label=2)."""
    return _iou(pred, gt, 2)


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute all five metrics for one sample."""
    return {
        "ktc_score":       compute_ktc_score(pred, gt),
        "dice_resistive":  dice_resistive(pred, gt),
        "dice_conductive": dice_conductive(pred, gt),
        "iou_resistive":   iou_resistive(pred, gt),
        "iou_conductive":  iou_conductive(pred, gt),
    }
