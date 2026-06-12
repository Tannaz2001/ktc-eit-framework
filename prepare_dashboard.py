"""
prepare_dashboard.py — snapshot existing BatchRunner output for the dashboard.

This is a thin wrapper: all run-folder creation and latest.txt handling lives
in example_usage.project_to_dashboard (the backend -> frontend bridge), so
there is exactly one producer of dashboard run folders.

Use this when you already ran `python run.py --config ...` and only want to
publish the results to the dashboard without re-running the benchmark.

Usage:
    python prepare_dashboard.py
    python prepare_dashboard.py --output-dir outputs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from example_usage import project_to_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish existing BatchRunner output to the Streamlit dashboard."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="BatchRunner output folder containing dashboard_scores.json / "
             "per_run_metrics.json (default: outputs/)",
    )
    args = parser.parse_args()

    run_dir = project_to_dashboard(args.output_dir)

    print()
    print("[OK] Dashboard is ready. Run:")
    print("       python -m streamlit run app.py")
    print(f"      (showing run: {run_dir.name})")


if __name__ == "__main__":
    main()
