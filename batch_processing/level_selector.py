"""Utilities for selecting dataset samples by difficulty level."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Generator, Iterable, Optional, Sequence

LEVEL_MIN = 1
LEVEL_MAX = 7

_LEVEL_PATTERNS = [
    re.compile(r"(?:^|[_\-/\s])level[_\s-]*([1-7])(?:$|[_\-/\s.])", re.IGNORECASE),
    re.compile(r"(?:^|[_\-/\s])lvl[_\s-]*([1-7])(?:$|[_\-/\s.])", re.IGNORECASE),
    re.compile(r"(?:^|[_\-/\s])l([1-7])(?:$|[_\-/\s.])", re.IGNORECASE),
]

DEFAULT_EXTENSIONS = (".npz", ".npy", ".csv", ".json", ".mat")
DATASET_FORMAT_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "auto": DEFAULT_EXTENSIONS,
    "numpy": (".npz", ".npy"),
    "csv": (".csv",),
    "json": (".json",),
    "matlab": (".mat",),
}


@dataclass(frozen=True, slots=True)
class SampleFile:
    """Represents a single dataset sample and its detected level."""

    level: int
    path: Path
    sample_id: str


def normalize_levels(levels: Sequence[int]) -> list[int]:
    """Validate and normalize requested levels."""
    unique_levels = sorted(set(levels))
    invalid = [lvl for lvl in unique_levels if lvl < LEVEL_MIN or lvl > LEVEL_MAX]
    if invalid:
        raise ValueError(f"Invalid levels {invalid}; expected values from {LEVEL_MIN} to {LEVEL_MAX}.")
    return unique_levels


def detect_level_from_path(path: Path) -> Optional[int]:
    """Try to detect challenge level from any path segment."""
    normalized = str(path).replace("\\", "/")
    for pattern in _LEVEL_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return int(match.group(1))
    return None


def classify_sample_level(sample_path: Path | str) -> int:
    """
    Classify a sample into level 1..7 based on file path tokens.
    """
    level = detect_level_from_path(Path(sample_path))
    if level is None:
        raise ValueError(f"Could not classify level from path: {sample_path}")
    return level


def get_extensions_for_format(dataset_format: str) -> tuple[str, ...]:
    """Resolve expected file extensions for a given dataset format."""
    key = dataset_format.strip().lower()
    if key not in DATASET_FORMAT_EXTENSIONS:
        supported = ", ".join(sorted(DATASET_FORMAT_EXTENSIONS))
        raise ValueError(f"Unsupported dataset format '{dataset_format}'. Supported: {supported}.")
    return DATASET_FORMAT_EXTENSIONS[key]


def iter_samples_for_levels(
    dataset_root: Path | str,
    levels: Sequence[int],
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> Generator[SampleFile, None, None]:
    """
    Stream sample files from disk and yield only requested levels.

    This generator avoids loading file contents and yields one path at a time.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    selected_levels = set(normalize_levels(levels))
    extension_set = {ext.lower() for ext in extensions}

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if extension_set and file_path.suffix.lower() not in extension_set:
            continue
        level = detect_level_from_path(file_path)
        if level is None or level not in selected_levels:
            continue
        sample_id = file_path.stem
        yield SampleFile(level=level, path=file_path, sample_id=sample_id)
