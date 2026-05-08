from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim


class MaskMetrics:
    """
    Optional metrics wrapper for Syeda's evaluation module.
    The adapter can run without this, but the batch/evaluation team can use it.
    """

    def compute(self, prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
        prediction = np.asarray(prediction, dtype=np.uint8)
        ground_truth = np.asarray(ground_truth, dtype=np.uint8)
        return {
            "ssim": self.ssim_score(prediction, ground_truth),
            "iou": self.mean_iou(prediction, ground_truth),
            "dice": self.mean_dice(prediction, ground_truth),
            "accuracy": self.pixel_accuracy(prediction, ground_truth),
        }

    def ssim_score(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return float(ssim(ground_truth, prediction, data_range=2))

    def mean_iou(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        values = [self._iou_for_label(prediction, ground_truth, label) for label in (0, 1, 2)]
        return float(np.mean(values))

    def mean_dice(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        values = [self._dice_for_label(prediction, ground_truth, label) for label in (0, 1, 2)]
        return float(np.mean(values))

    def pixel_accuracy(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return float(np.mean(prediction == ground_truth))

    def _iou_for_label(self, prediction: np.ndarray, ground_truth: np.ndarray, label: int) -> float:
        pred = prediction == label
        truth = ground_truth == label
        union = np.logical_or(pred, truth).sum()
        if union == 0:
            return 1.0
        intersection = np.logical_and(pred, truth).sum()
        return float(intersection / union)

    def _dice_for_label(self, prediction: np.ndarray, ground_truth: np.ndarray, label: int) -> float:
        pred = prediction == label
        truth = ground_truth == label
        denominator = pred.sum() + truth.sum()
        if denominator == 0:
            return 1.0
        numerator = 2 * np.logical_and(pred, truth).sum()
        return float(numerator / denominator)
