"""Scoring module — wraps KTCScoring.scoringFunction and per-class Dice/IoU."""

from __future__ import annotations

import numpy as np


def compute_ktc_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Official KTC score via KTCScoring.scoringFunction.
    pred and gt are 256x256 arrays with labels 0, 1, 2.
    Returns score between 0 and 1.
    """
    try:
        from KTCScoring import scoringFunction  # provided with KTC dataset
        return float(scoringFunction(pred, gt))
    except ImportError:
        raise ImportError(
            "KTCScoring module not found. "
            "Make sure the KTC dataset scoring code is on your PYTHONPATH."
        )


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
