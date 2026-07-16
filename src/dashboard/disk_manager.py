"""Disk space monitoring and automated cleanup."""

import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


def get_disk_usage() -> Dict[str, float]:
    """Get disk usage statistics for outputs directory."""
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        return {"used_gb": 0, "total_gb": 0, "percent": 0}

    try:
        total, used, free = shutil.disk_usage(outputs_dir)
        return {
            "used_gb": round(used / (1024**3), 2),
            "total_gb": round(total / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent": round((used / total) * 100, 1),
        }
    except OSError:
        return {"used_gb": 0, "total_gb": 0, "percent": 0, "error": "Cannot read disk stats"}


def check_disk_threshold(threshold_percent: float = 85.0) -> Tuple[bool, Dict]:
    """
    Check if disk usage exceeds threshold.

    Returns:
        (is_critical: bool, stats: dict)
    """
    stats = get_disk_usage()

    if "error" in stats:
        return False, stats

    is_critical = stats["percent"] >= threshold_percent

    return is_critical, stats


def list_runs_by_size() -> List[Tuple[str, float]]:
    """List all runs sorted by size (largest first)."""
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        return []

    runs = []
    for run_dir in outputs_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        size_bytes = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024**3)
        runs.append((run_dir.name, size_gb))

    return sorted(runs, key=lambda x: x[1], reverse=True)


def get_run_metadata(run_dir: Path) -> Dict:
    """Extract metadata from a run directory."""
    scores_file = run_dir / "scores.json"
    per_run_file = run_dir / "per_run_metrics.json"

    metadata = {
        "run_name": run_dir.name,
        "created_at": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
        "size_gb": sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / (1024**3),
    }

    if scores_file.exists():
        try:
            scores = json.loads(scores_file.read_text(encoding="utf-8"))
            metadata["method_count"] = len(scores)
        except (json.JSONDecodeError, OSError):
            pass

    return metadata


def cleanup_old_runs(
    keep_days: int = 30,
    keep_count: int = 10,
    target_free_gb: float = 50.0,
) -> Dict:
    """
    Intelligently delete old runs based on:
    1. Age (keep last N days)
    2. Count (always keep last N runs)
    3. Free space (until free space > target_free_gb)

    Returns:
        summary of deletions
    """
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        return {"status": "outputs/ does not exist", "deleted": 0}

    runs = sorted(
        [d for d in outputs_dir.glob("run_*") if d.is_dir()],
        key=lambda d: d.stat().st_ctime,
        reverse=True,  # newest first
    )

    if len(runs) <= keep_count:
        return {
            "status": f"Have {len(runs)} runs, keeping {keep_count} — no cleanup needed",
            "deleted": 0,
        }

    cutoff_date = datetime.now() - timedelta(days=keep_days)
    deleted_count = 0
    deleted_gb = 0.0

    for i, run_dir in enumerate(runs):
        # Keep at least keep_count runs
        if i < keep_count:
            continue

        run_age = datetime.fromtimestamp(run_dir.stat().st_ctime)

        # Delete if older than keep_days OR if we need free space
        stats = get_disk_usage()
        need_space = stats["free_gb"] < target_free_gb

        if run_age < cutoff_date or need_space:
            run_size_gb = sum(
                f.stat().st_size for f in run_dir.rglob("*") if f.is_file()
            ) / (1024**3)

            try:
                shutil.rmtree(run_dir)
                deleted_count += 1
                deleted_gb += run_size_gb
                print(f"Deleted {run_dir.name} ({run_size_gb:.2f} GB)")
            except OSError as e:
                print(f"Failed to delete {run_dir.name}: {e}")

        # Check if we have enough space now
        stats = get_disk_usage()
        if stats["free_gb"] >= target_free_gb and run_age >= cutoff_date:
            break

    return {
        "status": "cleanup complete",
        "deleted_runs": deleted_count,
        "freed_gb": round(deleted_gb, 2),
        "disk_stats_after": get_disk_usage(),
    }


def get_disk_report() -> Dict:
    """Get comprehensive disk usage report."""
    stats = get_disk_usage()
    runs = list_runs_by_size()[:5]  # top 5 largest

    return {
        "disk_usage": stats,
        "top_5_largest_runs": runs,
        "total_runs": len(list((Path("outputs") / "run_*").glob("*"))),
        "max_disk_threshold_pct": int(os.getenv("MAX_DISK_USAGE_PCT", "85")),
        "warning": "⚠️  Disk usage critical!" if stats.get("percent", 0) >= 85 else None,
    }
