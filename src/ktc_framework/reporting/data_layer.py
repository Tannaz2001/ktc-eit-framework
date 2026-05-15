"""Streamlit data layer — reads scores.json and returns DataFrames for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_scores(scores_path: str | Path = "outputs/scores.json") -> pd.DataFrame:
    """
    Read scores.json and return a flat DataFrame.
    One row per (method, level, sample) combination.
    """
    path = Path(scores_path)
    if not path.exists():
        return pd.DataFrame()

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for r in data:
        row = {
            "method": r["method"],
            "level": r["level"],
            "sample": r["sample"],
            "runtime_ms": r.get("runtime_ms", 0.0),
            "composite_score": r.get("composite_score", 0.0),
            "grade": r.get("grade", "D"),
            "degradation_slope": r.get("degradation_slope", 0.0),
            "git_sha": r.get("git_sha", "unknown"),
        }
        for metric, value in r.get("metrics", {}).items():
            row[metric] = value
        rows.append(row)

    return pd.DataFrame(rows)


def get_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Return a leaderboard ranked by average composite score per method."""
    if df.empty:
        return pd.DataFrame()

    leaderboard = (
        df.groupby("method")
        .agg(
            avg_composite=("composite_score", "mean"),
            avg_ktc_score=("ktc_score", "mean"),
            avg_runtime_ms=("runtime_ms", "mean"),
            total_runs=("method", "count"),
        )
        .reset_index()
        .sort_values("avg_composite", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.index += 1
    return leaderboard


def get_degradation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return average KTC score per method per level for the degradation curve chart."""
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["method", "level"])
        .agg(avg_ktc_score=("ktc_score", "mean"))
        .reset_index()
        .sort_values(["method", "level"])
    )


def get_per_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return all metrics averaged per method and level."""
    if df.empty:
        return pd.DataFrame()

    metric_cols = [c for c in df.columns if c in (
        "ktc_score", "dice_resistive", "dice_conductive",
        "iou_resistive", "iou_conductive", "composite_score", "runtime_ms"
    )]

    return (
        df.groupby(["method", "level"])[metric_cols]
        .mean()
        .round(4)
        .reset_index()
        .sort_values(["method", "level"])
    )


def get_worst_samples(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return the n worst performing samples per method by composite score."""
    if df.empty:
        return pd.DataFrame()

    return (
        df.sort_values("composite_score", ascending=True)
        .groupby("method")
        .head(n)
        .reset_index(drop=True)
    )


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return top-level summary stats for the dashboard header cards."""
    if df.empty:
        return {
            "total_runs": 0,
            "total_methods": 0,
            "total_levels": 0,
            "best_method": "N/A",
            "best_score": 0.0,
        }

    leaderboard = get_leaderboard(df)
    return {
        "total_runs": len(df),
        "total_methods": df["method"].nunique(),
        "total_levels": df["level"].nunique(),
        "best_method": leaderboard.iloc[0]["method"] if not leaderboard.empty else "N/A",
        "best_score": round(leaderboard.iloc[0]["avg_composite"], 2) if not leaderboard.empty else 0.0,
    }
