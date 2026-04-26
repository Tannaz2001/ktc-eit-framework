"""CLI entrypoint for level-based EIT batch experiments."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from batch_processing.async_runner import AsyncLevelRunner
from batch_processing.level_selector import (
    SampleFile,
    classify_sample_level,
    get_extensions_for_format,
    normalize_levels,
)
from batch_processing.missing_values import fill_missing_values_with_fallback
from batch_processing.sample_loader import ensure_standardized_sample


async def run_reconstruction_method(
    sample: SampleFile | np.ndarray | dict[str, Any],
    method_name: str,
    *,
    verbose_missing: bool = False,
) -> None:
    """
    Placeholder method adapter integration point.

    In the full framework this is where you'd:
    - load measurements for `sample`
    - call selected reconstruction method
    - pass output to evaluation/visualization
    """
    standardized = ensure_standardized_sample(sample)
    missing_mask = np.isnan(np.asarray(standardized.measurements, dtype=float))
    missing_indices = np.flatnonzero(missing_mask)
    fill_result = fill_missing_values_with_fallback(
        standardized.measurements,
        max_missing_ratio=0.60,
        fallback_strategy="median",
    )
    clean_measurements = fill_result.values

    if verbose_missing:
        preview = ",".join(map(str, missing_indices[:12]))
        preview = preview if preview else "-"
        suffix = "..." if missing_indices.size > 12 else ""
        print(
            f"[missing] sample={standardized.sample_id} level={standardized.level} "
            f"before={missing_indices.size} filled={fill_result.estimated_count} "
            f"ratio={fill_result.missing_ratio:.3f} strategy={fill_result.strategy_used} "
            f"indices=[{preview}{suffix}]"
        )

    # When input came from file selection, validate level agreement.
    if isinstance(sample, SampleFile):
        detected_level = classify_sample_level(sample.path)
        if detected_level != sample.level:
            raise ValueError(
                f"Level mismatch for sample {sample.sample_id}: "
                f"selector={sample.level}, classifier={detected_level}"
            )

    _ = clean_measurements
    _ = fill_result
    await asyncio.sleep(0.001)
    _ = method_name


def _parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    levels = normalize_levels(args.levels)
    selected_extensions = get_extensions_for_format(args.data_format)

    runner = AsyncLevelRunner(
        process_sample=lambda sample: run_reconstruction_method(
            sample,
            args.method,
            verbose_missing=args.verbose_missing,
        ),
        per_level_concurrency=args.per_level_concurrency,
    )
    reports = await runner.run_levels(args.dataset_root, levels, extensions=selected_extensions)

    print("\nBatch execution completed.\n")
    print(f"Data format: {args.data_format} | extensions: {selected_extensions}")
    print(f"Parallel levels launched: {levels}")
    for level in levels:
        report = reports[level]
        print(
            f"Level {level}: processed={report.processed}, skipped={report.skipped}, "
            f"errors={report.errors}, duration={report.duration_seconds:.2f}s"
        )


if __name__ == "__main__":
    asyncio.run(_main())
