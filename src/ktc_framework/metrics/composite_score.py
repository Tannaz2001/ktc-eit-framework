"""CompositeScore — combines all metrics into a weighted score and letter grade."""

from __future__ import annotations


def composite_score(metrics: dict[str, float]) -> float:
    """Return KTC score as the composite (only metric)."""
    return round(metrics.get("ktc_score", 0.0) * 100, 2)


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
