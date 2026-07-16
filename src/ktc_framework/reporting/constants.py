"""Shared display constants for both the Streamlit dashboard and HTML reports.

Centralising these here breaks the previous duplication where dashboard/scoring.py
and html_report.py each defined their own colour palette and grade thresholds,
causing the same method to appear in different colours depending on which output
you were looking at.
"""
from __future__ import annotations

# Fixed, per-method colours from the Okabe-Ito colourblind-safe palette.
# Keys are the canonical internal method names used in scores.json / per_run_metrics.json.
# Any method not listed gets _METHOD_COLOR_FALLBACK (a neutral grey).
METHOD_COLORS: dict[str, str] = {
    "CompetitionCNN": "#D55E00",
    "BackProjection": "#009E73",
    "LinearDifferenceReconstruction": "#CC79A7",
    "KTC2023_CUQI2_main": "#0072B2",
    "KTC2023_CUQI1": "#56B4E9",
}
METHOD_COLOR_FALLBACK = "#64748B"

# Canonical metric list: (display_label, json_key).
# Use short labels in dense tables; long labels in tooltips / sidebar.
METRIC_SPECS: list[tuple[str, str]] = [
    ("KTC Score", "ktc_score"),
    ("Dice Resistive", "dice_resistive"),
    ("Dice Conductive", "dice_conductive"),
    ("IoU Resistive", "iou_resistive"),
    ("IoU Conductive", "iou_conductive"),
]
METRIC_SHORT_LABELS: dict[str, str] = {
    "ktc_score": "KTC",
    "dice_resistive": "Dice R",
    "dice_conductive": "Dice C",
    "iou_resistive": "IoU R",
    "iou_conductive": "IoU C",
}


def get_method_color(name: str) -> str:
    """Return the fixed display colour for *name*, falling back to grey."""
    return METHOD_COLORS.get(name, METHOD_COLOR_FALLBACK)


def letter_grade(ktc_score: float) -> str:
    """Map a raw KTC score (0–1) to an A/B/C/D letter grade.

    Thresholds: A ≥ 0.60 · B ≥ 0.30 · C ≥ 0.10 · D < 0.10
    """
    if ktc_score >= 0.60:
        return "A"
    if ktc_score >= 0.30:
        return "B"
    if ktc_score >= 0.10:
        return "C"
    return "D"
