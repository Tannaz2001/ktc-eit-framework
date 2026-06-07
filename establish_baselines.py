#!/usr/bin/env python3
"""
Establish ground truth baseline scores from training data using BatchRunner.

Uses the proper framework architecture:
- BatchRunner loads mesh ONCE at startup
- Attaches shared mesh + reference voltages to every batch
- Runs all methods through the standard pipeline
- Computes KTC scores via the metrics registry

Usage:
    python3 establish_baselines.py

Output:
    outputs/training/scores.json — per-run scores (used by test harness)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ktc_framework.runner.config_validator import load_config
from src.ktc_framework.runner.experiment_runner import BatchRunner


def main() -> None:
    config_path = Path("configs/training_experiment.yaml")

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        raise SystemExit(1)

    config = load_config(config_path)
    config["dataset_root"] = config.get("dataset_root", "Codes_Matlab")

    output_dir = Path(config.get("output_dir", "outputs/training/"))

    print("=" * 70)
    print("ESTABLISHING BASELINES FROM TRAINING DATA")
    print("=" * 70)
    print(f"  Config:   {config_path}")
    print(f"  Data:     {config['dataset_root']}")
    print(f"  Mesh:     {config.get('mesh_path', 'N/A')}")
    print(f"  Levels:   {config['levels']}")
    print(f"  Samples:  {config['samples']}")
    print(f"  Methods:  {config['methods']}")
    print(f"  Output:   {output_dir}")
    print()

    runner = BatchRunner(config=config, output_dir=output_dir)
    results = runner.run()

    print(f"\n{'='*70}")
    print(f"BASELINE RESULTS: {len(results)} runs completed")
    print(f"{'='*70}")

    if not results:
        print("No results produced. Check mesh_path and dataset_root.")
        raise SystemExit(1)

    for r in results:
        m = r["metrics"]
        print(
            f"  {r['method']:30s}  sample={r['sample']}  "
            f"ktc={m['ktc_score']:+.6f}  "
            f"runtime={r['runtime_ms']:.0f}ms"
        )

    print(f"\nScores saved to: {output_dir / 'scores.json'}")


if __name__ == "__main__":
    main()
