"""
prepare_dashboard.py  —  Bridge between BatchRunner and Streamlit dashboard.

BatchRunner (run.py) saves outputs/scores.json as a flat list:
  [ {method, level, sample, metrics: {ktc_score}}, ... ]

The Streamlit dashboard (app.py) reads from a timestamped run folder pointed
to by outputs/latest.txt.

This script:
  1. Reads outputs/scores.json (flat list from BatchRunner)
  2. Creates a new timestamped run folder  outputs/run_YYYYMMDD_HHMMSS/
  3. Writes scores.json (averaged per method) into that folder
  4. Writes per_run_metrics.json (per-run breakdown) into that folder
  5. Updates outputs/latest.txt so the dashboard picks up the new data

Usage:
    python prepare_dashboard.py
    python prepare_dashboard.py --input outputs/scores.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BatchRunner output to Streamlit dashboard format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/scores.json"),
        help="Flat-list scores.json produced by run.py (default: outputs/scores.json)",
    )
    args = parser.parse_args()

    # ── Load runner output ────────────────────────────────────────────
    if not args.input.exists():
        print(f"[ERROR] {args.input} not found.")
        print("  Run the benchmark first:")
        print("    python run.py --config configs/ktc_all_methods.yaml")
        raise SystemExit(1)

    with open(args.input, encoding="utf-8") as f:
        runs = json.load(f)

    if not isinstance(runs, list) or len(runs) == 0:
        print(f"[ERROR] {args.input} is not a list or is empty.")
        raise SystemExit(1)

    print(f"[OK] Loaded {len(runs)} runs from {args.input}")

    # ── Create timestamped run folder ─────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Build averaged scores per method (for leaderboard & radar) ────
    method_buckets: dict[str, list] = defaultdict(list)
    for run in runs:
        method_buckets[run["method"]].append(run["metrics"])

    scores: dict[str, dict] = {}
    for method, metric_list in method_buckets.items():
        n = len(metric_list)
        averaged: dict[str, float] = {}
        for key in metric_list[0].keys():
            try:
                averaged[key] = round(sum(m[key] for m in metric_list) / n, 6)
            except (TypeError, KeyError):
                pass
        scores[method] = averaged

    # Write scores.json into run folder
    scores_out = run_dir / "scores.json"
    with open(scores_out, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"[OK] Averaged scores → {scores_out}")
    for method, m in scores.items():
        print(f"     {method}: KTC={m.get('ktc_score', 0):.4f}")

    # ── Build per-run metrics (for degradation curve & comparison) ────
    per_run: dict[str, dict] = defaultdict(dict)
    for run in runs:
        key = f"L{run['level']}_{run['sample']}"
        per_run[run["method"]][key] = {
            **run["metrics"],
            "composite_score": run.get("composite_score", 0.0),
            "grade":           run.get("grade", "D"),
            "runtime_ms":      run.get("runtime_ms", 0.0),
            "level":           run["level"],
            "sample":          run["sample"],
        }

    per_run_out = run_dir / "per_run_metrics.json"
    with open(per_run_out, "w", encoding="utf-8") as f:
        json.dump(dict(per_run), f, indent=2)
    print(f"[OK] Per-run metrics  → {per_run_out}")
    for method, runs_dict in per_run.items():
        print(f"     {method}: {len(runs_dict)} individual runs")

    # ── Update latest.txt so dashboard finds this run ─────────────────
    latest_txt = Path("outputs") / "latest.txt"
    latest_txt.write_text(str(run_dir))
    print(f"[OK] latest.txt       → {run_dir}")

    print()
    print("[OK] Dashboard is ready. Run:")
    print("       python -m streamlit run app.py")
    print(f"      (showing run: {run_id})")


if __name__ == "__main__":
    main()
