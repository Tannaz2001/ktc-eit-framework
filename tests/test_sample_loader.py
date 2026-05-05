from pathlib import Path

import numpy as np
import pytest

from batch_processing.exceptions import SampleSkipError
from batch_processing.level_selector import SampleFile
from batch_processing.sample_loader import ensure_standardized_sample, load_standardized_sample


def test_load_standardized_sample_from_npy(tmp_path: Path) -> None:
    sample_path = tmp_path / "level_1" / "data1.npy"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sample_path, np.array([1.0, 2.0, 3.0]))

    sample = SampleFile(level=1, path=sample_path, sample_id="data1")
    loaded = load_standardized_sample(sample)

    assert loaded.level == 1
    assert loaded.sample_id == "data1"
    assert np.allclose(loaded.v_meas, np.array([1.0, 2.0, 3.0]))


def test_load_standardized_sample_skips_reference_by_default(tmp_path: Path) -> None:
    ref_path = tmp_path / "level_1" / "ref.npy"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ref_path, np.array([1.0]))

    sample = SampleFile(level=1, path=ref_path, sample_id="ref")
    with pytest.raises(SampleSkipError):
        load_standardized_sample(sample)


def test_ensure_standardized_sample_from_array() -> None:
    payload = np.array([1.0, 2.0, 3.0], dtype=float)
    standardized = ensure_standardized_sample(payload, default_level=3, default_sample_id="x1")
    assert standardized.level == 3
    assert standardized.sample_id == "x1"
    assert np.allclose(standardized.v_meas, payload)


def test_ensure_standardized_sample_from_payload_dict() -> None:
    payload = {
        "sample_id": "external_1",
        "level": 7,
        "measurements": [1.0, 2.0, 3.0],
        "metadata": {"source": "teammate_loader"},
    }
    standardized = ensure_standardized_sample(payload)
    assert standardized.level == 7
    assert standardized.sample_id == "external_1"
    assert np.allclose(standardized.v_meas, np.array([1.0, 2.0, 3.0]))
