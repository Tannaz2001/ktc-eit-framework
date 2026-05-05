"""CLI entrypoint for level-based EIT batch experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import ExperimentConfig, ExperimentPipeline


def _parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run EIT reconstruction batch experiment by dataset level.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to Kuopio dataset root directory.")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        required=True,
        help="Selected challenge levels (1-7). Example: --levels 1 7",
    )
    parser.add_argument("--method", type=str, default="back_projection", help="Reconstruction method adapter name.")
    parser.add_argument(
        "--data-format",
        type=str,
        default="auto",
        choices=["auto", "numpy", "csv", "json", "matlab"],
        help="Input data format used to filter correct files from dataset.",
    )
    parser.add_argument(
        "--per-level-concurrency",
        type=int,
        default=1,
        help="Concurrent samples per level (default 1 for memory-safe streaming).",
    )
    parser.add_argument(
        "--verbose-missing",
        action="store_true",
        help="Print per-sample missing-value fill details and index preview.",
    )
    
    args = parser.parse_args()
    
    return ExperimentConfig(
        dataset_root=args.dataset_root,
        levels=args.levels,
        method=args.method,
        data_format=args.data_format,
        per_level_concurrency=args.per_level_concurrency,
        verbose_missing=args.verbose_missing,
    )


def _main() -> None:
    config = _parse_args()
    pipeline = ExperimentPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    _main()
