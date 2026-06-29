"""Streamlit data layer — single source of truth for reading run outputs.

The dashboard (app.py), the bridge (example_usage.py), and tests all load
benchmark results through this module so the JSON contract lives in exactly
one place:

* ``scores``  — ``{method: {metric_name: averaged_value}}``
* ``per_run`` — ``{method: {"L{level}_{sample}": {ktc_score, composite_score,
                  grade, runtime_ms, level, sample, gt_missing, *paths}}}``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Run discovery + loading (used by app.py)
# ---------------------------------------------------------------------------

def _run_has_data(run_dir: Path) -> bool:
    """Return True only for run folders with non-empty scores and per-run data."""
    try:
        scores_path = run_dir / "scores.json"
        per_run_path = run_dir / "per_run_metrics.json"
        if not scores_path.exists() or not per_run_path.exists():
            return False
        with scores_path.open(encoding="utf-8") as f:
            scores = json.load(f)
        with per_run_path.open(encoding="utf-8") as f:
            per_run = json.load(f)
        total_runs = sum(len(v) for v in per_run.values()) if isinstance(per_run, dict) else 0
        return bool(scores) and bool(per_run) and total_runs > 0
    except Exception:
        return False


def find_latest_run(runs_root: str | Path = "outputs") -> Path:
    """Return the active run folder: latest.txt pointer, newest run_*, or flat root."""
    runs_root = Path(runs_root)
    pointer = runs_root / "latest.txt"
    if pointer.exists():
        latest = Path(pointer.read_text().strip())
        if latest.exists() and _run_has_data(latest):
            return latest
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True)
    for run_dir in run_dirs:
        if _run_has_data(run_dir):
            return run_dir
    return runs_root


def load_merged_run_data(runs_root: str | Path = "outputs") -> tuple[dict, dict]:
    """Merge scores from every completed run directory, newest data per method wins.

    This lets a single-method run (e.g. after uploading a new plugin) slot into
    the existing leaderboard without re-running all methods.
    """
    runs_root = Path(runs_root)
    merged_scores: dict = {}
    merged_per_run: dict = {}

    # Newest run first so the first occurrence per method is the most recent.
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True)
    for run_dir in run_dirs:
        if not _run_has_data(run_dir):
            continue
        scores, per_run = load_run_data(run_dir)
        for method, metrics in scores.items():
            if method not in merged_scores:
                merged_scores[method] = metrics
                if method in per_run:
                    merged_per_run[method] = per_run[method]

    # Fall back to flat outputs/ layout (pre-run_* era).
    if not merged_scores and _run_has_data(runs_root):
        merged_scores, merged_per_run = load_run_data(runs_root)

    return merged_scores, merged_per_run


def load_run_data(run_dir: str | Path) -> tuple[dict, dict]:
    """Load (scores, per_run) from a run folder, with flat-outputs fallbacks.

    Normalises the legacy flat-list scores.json (BatchRunner's raw output)
    into the per-method-averages dict shape the dashboard consumes.
    """
    run_dir = Path(run_dir)

    # A run folder is self-contained: never fall back to another folder's
    # files, or the dashboard would silently mix two different experiments.
    # (Flat outputs/ layouts still work — find_latest_run returns outputs/
    # itself when no run folders exist.)
    scores: dict = {}
    for candidate in [run_dir / "dashboard_scores.json", run_dir / "scores.json"]:
        if candidate.exists():
            with candidate.open(encoding="utf-8") as f:
                scores = json.load(f)
            break

    per_run: dict = {}
    per_run_path = run_dir / "per_run_metrics.json"
    if per_run_path.exists():
        with per_run_path.open(encoding="utf-8") as f:
            per_run = json.load(f)

    # Legacy bridge format (pre-rewrite example_usage.py):
    #   {method: {"samples": {sid: score}, "mean_ktc": x}}
    if isinstance(scores, dict) and scores and all(
            isinstance(v, dict) and ("mean_ktc" in v or "samples" in v)
            for v in scores.values()):
        legacy = scores
        scores = {m: {"ktc_score": float(v.get("mean_ktc", 0.0))}
                  for m, v in legacy.items()}
        if not per_run:
            per_run = {}
            for m, v in legacy.items():
                per_run[m] = {}
                for sid, score in (v.get("samples") or {}).items():
                    comp = round(float(score) * 100, 2)
                    per_run[m][f"L1_{sid}"] = {
                        "ktc_score": float(score),
                        "composite_score": comp,
                        "grade": ("A" if comp >= 80 else "B" if comp >= 60
                                  else "C" if comp >= 40 else "D"),
                        "runtime_ms": 0.0,
                        "level": 1,
                        "sample": str(sid),
                        "gt_missing": False,
                    }

    if isinstance(scores, list):
        flat_runs = scores
        buckets: dict = {}
        for run in flat_runs:
            buckets.setdefault(run["method"], []).append(run)

        scores = {}
        for method, rows in buckets.items():
            metric_names = sorted({m for row in rows for m in row.get("metrics", {})})
            scores[method] = {
                metric: float(np.mean([row.get("metrics", {}).get(metric, 0.0) for row in rows]))
                for metric in metric_names
            }

        if not per_run:
            per_run = {}
            for method, rows in buckets.items():
                per_run[method] = {}
                for row in rows:
                    key = f"L{row['level']}_{row['sample']}"
                    per_run[method][key] = {
                        **row.get("metrics", {}),
                        "composite_score": row.get("composite_score", 0.0),
                        "grade": row.get("grade", "D"),
                        "runtime_ms": row.get("runtime_ms", 0.0),
                        "level": row["level"],
                        "sample": row["sample"],
                        "gt_missing": row.get("gt_missing", False),
                        "hull": row.get("hull", {}),
                    }

    # Defence in depth: every view assumes scores values are numbers —
    # drop anything else so a dirty/legacy file can degrade, not crash.
    if isinstance(scores, dict):
        scores = {
            m: {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            for m, metrics in scores.items() if isinstance(metrics, dict)
        }

    return scores, per_run


def create_method_mapping(scores: dict, per_run: dict) -> dict[str, str]:
    """Map score display names to per_run keys.

    Both files are written by the same BatchRunner save, so names normally
    match exactly — identity first. Fuzzy matching remains only as a
    fallback for hand-edited or legacy files.
    """
    mapping: dict[str, str] = {}
    for display_name in scores.keys():
        if display_name in per_run:
            mapping[display_name] = display_name
            continue
        display_lower = display_name.lower()
        for internal_key in per_run.keys():
            ik_lower = internal_key.lower()
            if (ik_lower in display_lower or display_lower in ik_lower
                    or ik_lower.replace('_', ' ') in display_lower
                    or display_lower.replace(' ', '_') == ik_lower):
                mapping[display_name] = internal_key
                break
    return mapping


def filter_by_level(entries: dict, lvl_min: int, lvl_max: int) -> dict:
    """Filter one method's per_run entries by their 'level' field."""
    return {
        key: e for key, e in entries.items()
        if lvl_min <= int(e.get("level", 1)) <= lvl_max
    }


def count_gt_missing(per_run: dict) -> int:
    """Number of runs scored against an all-zero (missing) ground truth."""
    return sum(
        1 for entries in per_run.values()
        for e in entries.values() if e.get("gt_missing")
    )


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
        "ktc_score", "composite_score", "runtime_ms"
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
