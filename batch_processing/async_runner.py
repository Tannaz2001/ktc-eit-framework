"""Asynchronous level-based batch execution for EIT experiments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Iterable, Sequence

from batch_processing.exceptions import SampleSkipError
from batch_processing.level_selector import SampleFile, iter_samples_for_levels, normalize_levels
from batch_processing.missing_values import validate_sample_file

ProcessSample = Callable[[SampleFile], Awaitable[Any]]


@dataclass(slots=True)
class LevelRunReport:
    level: int
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    skipped_details: list[str] = field(default_factory=list)
    error_details: list[str] = field(default_factory=list)


class AsyncLevelRunner:
    """
    Runs selected dataset levels in parallel using asyncio.

    - Level tasks run concurrently.
    - Samples are streamed from disk one by one.
    - Invalid samples are skipped safely with reasons.
    """

    def __init__(
        self,
        process_sample: ProcessSample,
        *,
        per_level_concurrency: int = 1,
    ) -> None:
        self.process_sample = process_sample
        self.per_level_concurrency = max(1, per_level_concurrency)

    async def run_levels(
        self,
        dataset_root: Path | str,
        levels: Sequence[int],
        *,
        extensions: Iterable[str] | None = None,
    ) -> dict[int, LevelRunReport]:
        selected_levels = normalize_levels(levels)
        tasks = [
            asyncio.create_task(self._run_single_level(dataset_root, level, extensions=extensions))
            for level in selected_levels
        ]
        reports = await asyncio.gather(*tasks)
        return {report.level: report for report in reports}

    async def _run_single_level(
        self,
        dataset_root: Path | str,
        level: int,
        *,
        extensions: Iterable[str] | None = None,
    ) -> LevelRunReport:
        report = LevelRunReport(level=level)
        started = perf_counter()
        sem = asyncio.Semaphore(self.per_level_concurrency)
        running_tasks: set[asyncio.Task[None]] = set()

        async def _process_one(sample: SampleFile) -> None:
            async with sem:
                validation = validate_sample_file(sample.path)
                if not validation.is_valid:
                    report.skipped += 1
                    report.skipped_details.append(f"{sample.path}: {validation.reason}")
                    return
                try:
                    await self.process_sample(sample)
                    report.processed += 1
                except SampleSkipError as exc:
                    report.skipped += 1
                    report.skipped_details.append(f"{sample.path}: {exc}")
                except Exception as exc:  # defensive path for batch robustness
                    report.errors += 1
                    report.error_details.append(f"{sample.path}: {exc}")

        for sample in iter_samples_for_levels(dataset_root, [level], extensions=extensions or ()):
            task = asyncio.create_task(_process_one(sample))
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)

            if len(running_tasks) >= self.per_level_concurrency * 2:
                await asyncio.gather(*running_tasks)

        if running_tasks:
            await asyncio.gather(*running_tasks)

        report.duration_seconds = perf_counter() - started
        return report
