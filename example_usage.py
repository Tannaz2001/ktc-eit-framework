"""
example_usage.py — backend -> frontend bridge.

This script does NOT load data or run reconstructions itself — BatchRunner
and the data plugins already own that job. It:

  1. Runs the full benchmark from a YAML config (default:
     configs/ktc_all_methods.yaml, which includes the runtime-adapter
     method DampedLeastSquaresReconstruction via method_plugin_paths).
  2. Picks up the scores BatchRunner wrote (dashboard_scores.json,
     per_run_metrics.json) and snapshots them into a timestamped
     outputs/run_YYYYMMDD_HHMMSS/ folder.
  3. Arranges the panel/overlay PNGs into the layout app.py expects
     (reconstructions/level_N/sample_S/<method>.png, error_overlays/).
  4. Points outputs/latest.txt at the new run folder (written LAST, so a
     crashed run never hijacks the dashboard).
  5. Launches the Streamlit dashboard (app.py).

Run:
    python example_usage.py
    python example_usage.py --config configs/ktc_all_methods.yaml
    python example_usage.py --no-app          # prepare data, don't launch UI
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ktc_framework.runner.config_validator import load_config, ConfigError
from src.ktc_framework.runner.experiment_runner import BatchRunner


def run_benchmark(config_path: Path) -> dict:
    """Validate the config and run the full BatchRunner pipeline."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)

    print(f"[1/3] Running benchmark: {config_path}")
    print(f"      methods = {config['methods']}")
    runner = BatchRunner(config=config, output_dir=Path(config["output_dir"]))
    runner.run()
    return config


def project_to_dashboard(output_dir: str | Path, runs_root: str | Path = "outputs") -> Path:
    """Snapshot BatchRunner output into a run folder app.py can read.

    latest.txt is updated only after non-empty scores/per-run files and images
    have been validated and copied. Empty/crashed runs must never become latest.
    """
    output_dir = Path(output_dir)
    runs_root = Path(runs_root)

    print("[2/3] Validating benchmark outputs before dashboard projection")

    dashboard_scores = output_dir / "dashboard_scores.json"
    if not dashboard_scores.exists():
        print(f"[ERROR] {dashboard_scores} not found - did the benchmark run?")
        raise SystemExit(1)
    with dashboard_scores.open(encoding="utf-8") as f:
        dashboard_data = json.load(f)

    per_run_path = output_dir / "per_run_metrics.json"
    if not per_run_path.exists():
        print(f"[ERROR] {per_run_path} not found - benchmark may have crashed mid-run.")
        raise SystemExit(1)
    with per_run_path.open(encoding="utf-8") as f:
        per_run = json.load(f)

    total_runs = sum(len(v) for v in per_run.values()) if isinstance(per_run, dict) else 0
    if not dashboard_data or not per_run or total_runs == 0:
        print("[ERROR] Benchmark produced no scored reconstructions.")
        print("        latest.txt was NOT changed; previous completed dashboard data is preserved.")
        raise SystemExit(1)

    run_dir = runs_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] Projecting scores to dashboard folder: {run_dir}")

    shutil.copyfile(dashboard_scores, run_dir / "scores.json")
    shutil.copyfile(per_run_path, run_dir / "per_run_metrics.json")

    recon_dir = run_dir / "reconstructions"
    overlay_dir = run_dir / "error_overlays"
    copied = 0
    for method, runs in per_run.items():
        for entry in runs.values():
            level, sample = entry.get("level"), entry.get("sample")

            png = Path(entry.get("png_path", ""))
            if png.exists():
                dest = recon_dir / f"level_{level}" / f"sample_{sample}"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(png, dest / f"{method}.png")
                copied += 1

            overlay = Path(entry.get("overlay_path", ""))
            if int(level) == 1 and overlay.exists():
                overlay_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(overlay, overlay_dir / f"{method}_sample_{sample}.png")
                copied += 1

    print(f"      {copied} images arranged for the dashboard")

    src_figures = output_dir / "figures"
    if src_figures.exists():
        dst_figures = run_dir / "figures"
        shutil.copytree(src_figures, dst_figures, dirs_exist_ok=True)
        print(f"      figures copied for reports: {len(list(dst_figures.glob('*.png')))} PNGs")

    (runs_root / "latest.txt").write_text(str(run_dir.resolve()))
    print(f"      {runs_root / 'latest.txt'} -> {run_dir}")
    return run_dir

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the benchmark and project results into the dashboard."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ktc_all_methods.yaml"),
        help="Experiment YAML (default: configs/ktc_all_methods.yaml)",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Prepare dashboard data but do not launch Streamlit.",
    )
    args = parser.parse_args()

    config = run_benchmark(args.config)
    run_dir = project_to_dashboard(config["output_dir"])

    if args.no_app:
        print(f"[3/3] Done. Launch the dashboard with: streamlit run app.py")
        print(f"      (it will show {run_dir})")
        return

    print("[3/3] Launching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
