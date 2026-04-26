from pathlib import Path

import pytest

from batch_processing.level_selector import (
    classify_sample_level,
    get_extensions_for_format,
    iter_samples_for_levels,
)


def test_classify_sample_level_detects_expected_level() -> None:
    assert classify_sample_level(Path("dataset/level_1/sample_a.npy")) == 1
    assert classify_sample_level(Path("dataset/lvl7/sample_b.csv")) == 7


def test_classify_sample_level_raises_for_unknown_pattern() -> None:
    with pytest.raises(ValueError):
        classify_sample_level(Path("dataset/unknown/sample.npy"))


def test_get_extensions_for_format() -> None:
    assert get_extensions_for_format("numpy") == (".npz", ".npy")
    assert get_extensions_for_format("csv") == (".csv",)


def test_iter_samples_for_levels_filters_by_level_and_extension(tmp_path: Path) -> None:
    sample_1 = tmp_path / "level_1" / "a.npy"
    sample_7 = tmp_path / "level_7" / "b.csv"
    ignored = tmp_path / "level_1" / "notes.txt"

    sample_1.parent.mkdir(parents=True, exist_ok=True)
    sample_7.parent.mkdir(parents=True, exist_ok=True)
    sample_1.write_bytes(b"1234567890abcdefghijklmnop")
    sample_7.write_text("v1,v2\n1,2\n", encoding="utf-8")
    ignored.write_text("ignore", encoding="utf-8")

    samples = list(iter_samples_for_levels(tmp_path, levels=[1, 7], extensions=(".npy",)))
    assert len(samples) == 1
    assert samples[0].level == 1
    assert samples[0].path.name == "a.npy"
