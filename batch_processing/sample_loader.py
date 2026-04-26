"""Self-contained standardized sample loading for common dataset formats."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from batch_processing.exceptions import SampleSkipError
from batch_processing.level_selector import SampleFile

_PREFERRED_MEASUREMENT_KEYS = (
    "Uel",
    "meas",
    "measurements",
    "voltages",
    "U",
    "data",
)


@dataclass(slots=True)
class StandardizedSample:
    sample_id: str
    level: int
    source_path: Path
    measurements: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_standardized_sample(
    source: SampleFile | StandardizedSample | np.ndarray | dict[str, Any],
    *,
    default_level: int = 1,
    default_sample_id: str = "external_sample",
) -> StandardizedSample:
    """
    Normalize external payload/array input into StandardizedSample.

    Supported inputs:
    - SampleFile: loaded from disk with `load_standardized_sample`
    - StandardizedSample: passed through
    - np.ndarray: used directly as measurements
    - dict payload: expects one measurement key such as `measurements`, `voltages`, `Uel`, `meas`, `data`
      and optional `sample_id`, `level`, `source_path`, `metadata`.
    """
    if isinstance(source, StandardizedSample):
        return source

    if isinstance(source, SampleFile):
        return load_standardized_sample(source)

    if isinstance(source, np.ndarray):
        return StandardizedSample(
            sample_id=default_sample_id,
            level=default_level,
            source_path=Path(f"<{default_sample_id}>"),
            measurements=np.asarray(source, dtype=float),
            metadata={"source_type": "ndarray"},
        )

    if isinstance(source, dict):
        measurement_value = None
        for key in _PREFERRED_MEASUREMENT_KEYS:
            if key in source:
                measurement_value = source[key]
                break
        if measurement_value is None:
            raise SampleSkipError("payload_missing_measurements")

        sample_id = str(source.get("sample_id", default_sample_id))
        level = int(source.get("level", default_level))
        source_path_raw = source.get("source_path", f"<{sample_id}>")
        metadata = source.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        return StandardizedSample(
            sample_id=sample_id,
            level=level,
            source_path=Path(str(source_path_raw)),
            measurements=np.asarray(measurement_value, dtype=float),
            metadata=metadata,
        )

    raise SampleSkipError(f"unsupported_payload_type:{type(source).__name__}")


def load_standardized_sample(sample: SampleFile, *, allow_reference: bool = False) -> StandardizedSample:
    """
    Load a sample into a standardized payload without relying on external loader modules.
    """
    source = sample.path

    if source.stem.lower() == "ref" and not allow_reference:
        raise SampleSkipError("reference_file_excluded")

    suffix = source.suffix.lower()
    if suffix == ".mat":
        measurements = _load_mat(source)
    elif suffix == ".npy":
        measurements = np.load(source)
    elif suffix == ".npz":
        measurements = _load_npz(source)
    elif suffix == ".csv":
        measurements = _load_csv(source)
    elif suffix == ".json":
        measurements = _load_json(source)
    else:
        raise SampleSkipError(f"unsupported_file_extension:{suffix}")

    values = np.asarray(measurements, dtype=float)
    if values.size == 0:
        raise SampleSkipError("empty_measurement_array")

    return StandardizedSample(
        sample_id=sample.sample_id,
        level=sample.level,
        source_path=source,
        measurements=values,
        metadata={"extension": suffix},
    )


def _load_npz(path: Path) -> np.ndarray:
    archive = np.load(path)
    if not archive.files:
        raise SampleSkipError("npz_has_no_arrays")

    for key in _PREFERRED_MEASUREMENT_KEYS:
        if key in archive.files:
            return np.asarray(archive[key])
    return np.asarray(archive[archive.files[0]])


def _load_csv(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.genfromtxt(path, delimiter=",", skip_header=1)


def _load_json(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return np.asarray(data)

    if isinstance(data, dict):
        for key in _PREFERRED_MEASUREMENT_KEYS:
            if key in data:
                return np.asarray(data[key])
        for value in data.values():
            if isinstance(value, (list, tuple)):
                return np.asarray(value)

    raise SampleSkipError("json_does_not_contain_measurements")


def _load_mat(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat  # imported lazily to keep module lightweight
    except ImportError as exc:
        raise SampleSkipError("scipy_required_for_mat_files") from exc

    mat_data = loadmat(path)
    payload = {
        key: value
        for key, value in mat_data.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.size > 0
    }
    if not payload:
        raise SampleSkipError("mat_has_no_numeric_arrays")

    for key in _PREFERRED_MEASUREMENT_KEYS:
        if key in payload:
            return np.asarray(payload[key]).squeeze()

    # fallback: first largest numeric array
    key = max(payload.keys(), key=lambda name: payload[name].size)
    return np.asarray(payload[key]).squeeze()

