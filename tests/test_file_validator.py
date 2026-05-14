from pathlib import Path

from src.ktc_framework.loaders.file_validator import validate_sample_file


def test_missing_file(tmp_path: Path) -> None:
    result = validate_sample_file(tmp_path / "missing.npy")
    assert not result.is_valid
    assert result.reason == "missing_file"


def test_file_too_small(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.npy"
    tiny.write_bytes(b"123")
    result = validate_sample_file(tiny)
    assert not result.is_valid


def test_valid_file(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npy"
    valid.write_bytes(b"1234567890abcdefghijkl")
    result = validate_sample_file(valid)
    assert result.is_valid
    assert result.reason is None
