"""CLI entrypoint — run an EIT benchmark experiment from a config file.

Usage:
    python run.py --config configs/training_experiment.yaml
    python run.py --config configs/experiment.yaml

All data paths are auto-detected. Just drop your data folders in the
project root and run. Override with KTC_DATASET_ROOT env var if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ktc_framework.runner.config_validator import load_config, ConfigError
from src.ktc_framework.runner.experiment_runner import BatchRunner
from example_usage import project_to_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EIT benchmark experiment from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config (e.g. configs/training_experiment.yaml)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    output_dir = Path(config["output_dir"])
    runner = BatchRunner(config=config, output_dir=output_dir)

    print(f"[OK] Config loaded: {args.config}")
    print(f"     Data:    {config['dataset_root']}")
    print(f"     Mesh:    {config['mesh_path']}")
    print(f"     Levels:  {config['levels']}")
    print(f"     Samples: {config['samples']}")
    print(f"     Methods: {config['methods']}")
    print(f"     Output:  {config['output_dir']}")
    print()

    print("[...] Running experiment...")
    results = runner.run()
    dashboard_run = project_to_dashboard(output_dir)

    print(f"[OK] Done. {len(results)} runs completed.")
    print(f"     Results saved to: {output_dir / 'scores.json'}")
    print(f"     Dashboard run:    {dashboard_run}")


if __name__ == "__main__":
    main()
