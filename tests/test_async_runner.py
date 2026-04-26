import asyncio
from pathlib import Path
from time import perf_counter

from batch_processing.async_runner import AsyncLevelRunner
from batch_processing.level_selector import SampleFile


def _make_valid_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"1234567890abcdefghijklmnop")


def test_levels_1_and_7_run_in_parallel(tmp_path: Path) -> None:
    _make_valid_sample(tmp_path / "level_1" / "sample_a.npy")
    _make_valid_sample(tmp_path / "level_7" / "sample_c.npy")

    async def process_sample(sample: SampleFile) -> None:
        _ = sample
        await asyncio.sleep(0.2)

    runner = AsyncLevelRunner(process_sample=process_sample, per_level_concurrency=1)

    start = perf_counter()
    reports = asyncio.run(runner.run_levels(tmp_path, [1, 7], extensions=(".npy",)))
    elapsed = perf_counter() - start

    assert reports[1].processed == 1
    assert reports[7].processed == 1
    # Parallel target: should be much closer to 0.2s than 0.4s.
    assert elapsed < 0.38


def test_invalid_files_are_skipped_and_reported(tmp_path: Path) -> None:
    valid = tmp_path / "level_1" / "ok.npy"
    tiny = tmp_path / "level_1" / "bad.npy"
    _make_valid_sample(valid)
    tiny.parent.mkdir(parents=True, exist_ok=True)
    tiny.write_bytes(b"12")

    async def process_sample(sample: SampleFile) -> None:
        _ = sample
        await asyncio.sleep(0)

    runner = AsyncLevelRunner(process_sample=process_sample, per_level_concurrency=1)
    reports = asyncio.run(runner.run_levels(tmp_path, [1], extensions=(".npy",)))

    assert reports[1].processed == 1
    assert reports[1].skipped == 1
    assert reports[1].errors == 0
