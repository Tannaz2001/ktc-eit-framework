from pathlib import Path

import numpy as np
import pytest

from batch_processing.missing_values import (
    fill_missing_values,
    fill_missing_values_gp,
    fill_missing_values_with_fallback,
    validate_sample_file,
)


def test_fill_missing_values_mean_strategy() -> None:
    values = np.array([1.0, np.nan, 3.0], dtype=float)
    output = fill_missing_values(values, strategy="mean")
    assert np.allclose(output, np.array([1.0, 2.0, 3.0]))


def test_fill_missing_values_all_nan_falls_back_to_zero() -> None:
    values = np.array([np.nan, np.nan], dtype=float)
    output = fill_missing_values(values, strategy="mean")
    assert np.allclose(output, np.array([0.0, 0.0]))


def test_fill_missing_values_forward_fill() -> None:
    values = np.array([np.nan, 2.0, np.nan, 4.0], dtype=float)
    output = fill_missing_values(values, strategy="forward_fill")
    assert np.allclose(output, np.array([2.0, 2.0, 2.0, 4.0]))


def test_fill_missing_values_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        fill_missing_values(np.array([1.0]), strategy="bad_strategy")  # type: ignore[arg-type]


def test_validate_sample_file_states(tmp_path: Path) -> None:
    missing = validate_sample_file(tmp_path / "missing.npy")
    assert not missing.is_valid
    assert missing.reason == "missing_file"

    tiny = tmp_path / "level_1" / "tiny.npy"
    tiny.parent.mkdir(parents=True, exist_ok=True)
    tiny.write_bytes(b"123")
    invalid = validate_sample_file(tiny)
    assert not invalid.is_valid

    valid = tmp_path / "level_1" / "valid.npy"
    valid.write_bytes(b"1234567890abcdefghijkl")
    ok = validate_sample_file(valid)
    assert ok.is_valid


def test_fill_missing_values_gp_fills_small_missing_ratio() -> None:
    values = np.array([1.0, np.nan, 2.0, 2.5, np.nan], dtype=float)
    result = fill_missing_values_gp(values, max_missing_ratio=0.6)
    assert result.is_reliable
    assert result.estimated_count == 2
    assert not np.isnan(result.values).any()


def test_fill_missing_values_gp_marks_unreliable_when_too_many_missing() -> None:
    values = np.array([np.nan, np.nan, np.nan, 2.0], dtype=float)
    result = fill_missing_values_gp(values, max_missing_ratio=0.5)
    assert not result.is_reliable
    assert result.reason is not None


def test_fill_missing_values_with_fallback_processes_high_missing_ratio() -> None:
    values = np.array([np.nan, np.nan, np.nan, 2.0], dtype=float)
    result = fill_missing_values_with_fallback(values, max_missing_ratio=0.5, fallback_strategy="median")
    assert result.is_reliable
    assert result.strategy_used == "fallback_median"
    assert not np.isnan(result.values).any()
