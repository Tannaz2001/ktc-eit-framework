"""Parallel level-based batch execution for EIT experiments using ProcessPoolExecutor."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

from batch_processing.exceptions import SampleSkipError
from batch_processing.level_selector import SampleFile, iter_samples_for_levels, normalize_levels
from batch_processing.missing_values import validate_sample_file

ProcessSample = Callable[[SampleFile], Any]

@dataclass(slots=True)
class LevelRunReport:
    level: int
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    skipped_details: list[str] = field(default_factory=list)
    error_details: list[str] = field(default_factory=list)


def _process_one_sample(sample: SampleFile, process_fn: ProcessSample) -> tuple[str, str | None, str | None]:
    """
    Worker function for processing a single sample.
    """
    validation = validate_sample_file(sample.path)
    if not validation.is_valid:
        return 'skipped', f"{sample.path}: {validation.reason}", None
    
    try:
        process_fn(sample)
        return 'processed', None, None
    except SampleSkipError as exc:
        return 'skipped', f"{sample.path}: {exc}", None
    except Exception as exc:
        return 'error', None, f"{sample.path}: {exc}"


class ParallelLevelRunner:
    """
    Runs selected dataset levels in parallel using ProcessPoolExecutor.
    - True parallelism across CPU cores.
    - Bypasses the GIL for numerical work.
    """

    def __init__(
        self,
        process_sample: ProcessSample,
        *,
        per_level_concurrency: int = 1,
    ) -> None:
        self.process_sample = process_sample
        # we repurpose per_level_concurrency to max_workers across the pool
        self.max_workers = max(1, per_level_concurrency)

    def run_levels(
        self,
        dataset_root: Path | str,
        levels: Sequence[int],
        *,
        extensions: Iterable[str] | None = None,
    ) -> dict[int, LevelRunReport]:
        selected_levels = normalize_levels(levels)
        reports = {level: LevelRunReport(level=level) for level in selected_levels}
        started = {level: perf_counter() for level in selected_levels}

        # Gather all samples for the target levels
        samples = list(iter_samples_for_levels(dataset_root, selected_levels, extensions=extensions or ()))
        
        if not samples:
            for level in selected_levels:
                reports[level].duration_seconds = perf_counter() - started[level]
            return reports

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Map futures back to their original samples so we know which level it belonged to
            futures = {
                executor.submit(_process_one_sample, sample, self.process_sample): sample 
                for sample in samples
            }
            
            for future in concurrent.futures.as_completed(futures):
                sample = futures[future]
                report = reports[sample.level]
                try:
                    status, skip_detail, err_detail = future.result()
                    if status == 'processed':
                        report.processed += 1
                    elif status == 'skipped':
                        report.skipped += 1
                        if skip_detail:
                            report.skipped_details.append(skip_detail)
                    elif status == 'error':
                        report.errors += 1
                        if err_detail:
                            report.error_details.append(err_detail)
                except Exception as exc:
                    report.errors += 1
                    report.error_details.append(f"Worker crashed: {exc}")

        for level in selected_levels:
            reports[level].duration_seconds = perf_counter() - started[level]
        
        return reports
