"""CLI entrypoint — run an EIT benchmark experiment from a config file."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

from src.ktc_framework.runner.config_validator import load_config, ConfigError
from src.ktc_framework.runner.experiment_runner import BatchRunner
from src.ktc_framework.utils.mock_mesh import make_dummy_batch_with_mesh  # always valid mesh

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EIT benchmark experiment from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config (e.g. configs/experiment.yaml)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"[ERROR] Config validation failed: {e}")
        raise SystemExit(1)
    # -----------------------------
    # Dataset root: env var overrides config value; config value is the fallback
    dataset_path = os.environ.get("KTC_DATASET_ROOT") or config.get("dataset_root", "")
    if dataset_path:
        print(f"[INFO] Using dataset path: {dataset_path}")
        config["dataset_root"] = dataset_path
    else:
        print("[WARN] KTC_DATASET_ROOT not set and no dataset_root in config — data loading may fail.")

    # mesh_path is always relative to the project root, NOT dataset_root.
    # (Mesh_sparse.mat lives in Codes_Matlab/, independent of the dataset location.)

    # -----------------------------
    print(f"[OK] Config loaded: {args.config}")
    print(f"     Levels : {config['levels']}")
    print(f"     Samples: {config['samples']}")
    print(f"     Methods: {config['methods']}")
    print(f"     Mesh   : {config['mesh_path']}")
    print()

    if 'mesh_path' in config:
        print(f"     Mesh   : {config['mesh_path']}")
    print()

    output_dir = Path(config.get("output_dir", "outputs/"))
    runner = BatchRunner(config=config, output_dir=output_dir)

    print("[...] Running experiment...")
    results = runner.run()  # no 'batch=' argument

    print(f"[OK] Done. {len(results)} runs completed.")
    print(f"     Results saved to: {output_dir / 'scores.json'}")


if __name__ == "__main__":
    main()
