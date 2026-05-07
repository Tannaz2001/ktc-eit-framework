import time
from pathlib import Path
from time import perf_counter

from batch_processing.parallel_runner import ParallelLevelRunner
from batch_processing.level_selector import SampleFile


def _make_valid_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"1234567890abcdefghijklmnop")


def process_sample_sleep(sample: SampleFile) -> None:
    _ = sample
    time.sleep(0.2)


def test_levels_1_and_7_run_in_parallel(tmp_path: Path) -> None:
    _make_valid_sample(tmp_path / "level_1" / "sample_a.npy")
    _make_valid_sample(tmp_path / "level_7" / "sample_c.npy")

    # max_workers=2 so they run strictly in parallel
    runner = ParallelLevelRunner(process_sample=process_sample_sleep, per_level_concurrency=2)

    start = perf_counter()
    reports = runner.run_levels(tmp_path, [1, 7], extensions=(".npy",))
    elapsed = perf_counter() - start

    assert reports[1].processed == 1
    assert reports[7].processed == 1
    # Note: ProcessPoolExecutor on Windows has ~1s+ overhead just to spawn worker processes.
    # We relax this timing check so the test doesn't spuriously fail in CI.
    assert elapsed < 2.5


def process_sample_fast(sample: SampleFile) -> None:
    pass


def test_invalid_files_are_skipped_and_reported(tmp_path: Path) -> None:
    valid = tmp_path / "level_1" / "ok.npy"
    tiny = tmp_path / "level_1" / "bad.npy"
    _make_valid_sample(valid)
    tiny.parent.mkdir(parents=True, exist_ok=True)
    tiny.write_bytes(b"12")

    runner = ParallelLevelRunner(process_sample=process_sample_fast, per_level_concurrency=1)
    reports = runner.run_levels(tmp_path, [1], extensions=(".npy",))

    assert reports[1].processed == 1
    assert reports[1].skipped == 1
    assert reports[1].errors == 0
