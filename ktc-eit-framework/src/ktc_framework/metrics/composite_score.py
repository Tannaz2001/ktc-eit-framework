"""CompositeScore — combines all metrics into a weighted score and letter grade."""

from __future__ import annotations


METRIC_WEIGHTS: dict[str, float] = {
    "ktc_score": 0.40,
    "dice_resistive": 0.20,
    "dice_conductive": 0.20,
    "iou_resistive": 0.10,
    "iou_conductive": 0.10,
}


def composite_score(metrics: dict[str, float]) -> float:
    """
    Weighted combination of all metrics normalised to 0–100.
    Weights must sum to 1.0.
    """
    score = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        value = metrics.get(metric, 0.0)
        score += value * weight * 100
    return round(score, 2)


def letter_grade(score: float) -> str:
    """Return a letter grade for a composite score out of 100."""
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def compute_composite(metrics: dict[str, float]) -> dict[str, float | str]:
    """Return composite score and letter grade for a metrics dict."""
    score = composite_score(metrics)
    return {
        "composite_score": score,
        "grade": letter_grade(score),
    }
