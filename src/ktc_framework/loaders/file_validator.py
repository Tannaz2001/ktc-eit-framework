"""Validation helpers for dataset samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None


def validate_sample_file(path: Path | str, min_size_bytes: int = 16) -> ValidationResult:
    """Basic safety checks before processing a sample."""
    sample_path = Path(path)

    if not sample_path.exists():
        return ValidationResult(False, "missing_file")
    if not sample_path.is_file():
        return ValidationResult(False, "not_a_file")

    size = sample_path.stat().st_size
    if size < min_size_bytes:
        return ValidationResult(False, f"incomplete_file_size<{min_size_bytes}")

    return ValidationResult(True, None)
