"""One-shot script: compute hull analysis for all existing mat_predictions.

Reads the current per_run_metrics.json, loads each .mat reconstruction,
runs HullAnalyzer to extract + compare hulls against ground truth, and writes
the enriched per_run_metrics.json back. Safe to re-run (idempotent).

Usage:
    python compute_hull_data.py
"""
from __future__ import annotations
import json
import numpy as np
import scipy.io
from pathlib import Path
from src.ktc_framework.plugins.hull_plugin import HullAnalyzer, compute_hull_record
from src.ktc_framework.reporting.data_layer import find_latest_run

SAMPLE_TO_NUM = {"A": "1", "B": "2", "C": "3"}

def load_gt(dataset_root: str, level: int, sample: str) -> np.ndarray | None:
    """Load ground truth segmentation for a given level/sample."""
    gt_dir = Path(dataset_root) / "GroundTruths"
    snum = SAMPLE_TO_NUM.get(sample, sample)
    candidates = [
        gt_dir / f"level_{level}" / f"{snum}_true.mat",
        gt_dir / f"level{level}" / f"{snum}_true.mat",
        gt_dir / f"level_{level}" / f"true{level}{sample}.mat",
    ]
    for path in candidates:
        if path.exists():
            mat = scipy.io.loadmat(str(path), squeeze_me=True)
            for key in ["truth", "Segmentation", "gt", "seg"]:
                if key in mat:
                    arr = np.asarray(mat[key], dtype=np.uint8)
                    if arr.shape == (256, 256):
                        return arr
    return None


def main():
    run_dir = find_latest_run()
    pr_path = run_dir / "per_run_metrics.json"
    if not pr_path.exists():
        pr_path = Path("outputs") / "per_run_metrics.json"
    print(f"Loading {pr_path}")

    with pr_path.open(encoding="utf-8") as f:
        per_run = json.load(f)

    mat_root = Path("outputs") / "mat_predictions"
    dataset_root = "EvaluationData"
    analyzer = HullAnalyzer()
    updated = 0

    for method, entries in per_run.items():
        for key, entry in entries.items():
            level = entry.get("level", 1)
            sample = entry.get("sample", "A")

            mat_path = mat_root / method / f"level_{level}" / f"sample_{sample}.mat"
            if not mat_path.exists():
                alt = entry.get("mat_path", "")
                if alt and Path(alt).exists():
                    mat_path = Path(alt)
                else:
                    continue

            try:
                mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
                pred = np.asarray(mat["reconstruction"], dtype=np.uint8)
            except Exception as e:
                print(f"  Skip {method}/{key}: {e}")
                continue

            if pred.shape != (256, 256):
                continue

            gt = load_gt(dataset_root, level, sample)

            try:
                entry["hull"] = compute_hull_record(pred, gt, analyzer)
                updated += 1
            except Exception as e:
                print(f"  Hull failed {method}/{key}: {e}")

    # Write back
    with pr_path.open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2)

    # Also update the copy in the run folder if different
    run_copy = run_dir / "per_run_metrics.json"
    if run_copy != pr_path and run_copy.exists():
        with run_copy.open("w", encoding="utf-8") as f:
            json.dump(per_run, f, indent=2)

    print(f"Done — enriched {updated} entries with hull data.")
    print(f"Saved to {pr_path}")


if __name__ == "__main__":
    main()
