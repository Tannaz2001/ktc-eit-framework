from __future__ import annotations

from pathlib import Path

import numpy as np

from ktc_framework.config_schema import GreitConfig
from ktc_framework.contracts import StandardizedKtcInput
from ktc_framework.methods.greit import GreitAdapter


def build_test_adapter(tmp_path: Path) -> GreitAdapter:
    matrix = np.zeros((64 * 64, 2356), dtype=np.float64)
    matrix[100, 0] = 1.0
    matrix[200, 1] = -1.0

    model_path = tmp_path / "greit_test_model.npz"
    np.savez(model_path, reconstruction_matrix=matrix)

    config = GreitConfig(
        model_path=model_path,
        greit_image_shape=(64, 64),
        output_image_size=(256, 256),
        expected_input_size=2356,
        segmentation_strategy="threshold",
        threshold=0.05,
        fail_on_invalid_input=True,
    )

    return GreitAdapter(config)


def test_greit_returns_256_mask_with_valid_labels(tmp_path: Path):
    adapter = build_test_adapter(tmp_path)

    vector = np.zeros(2356)
    vector[0] = 1.0
    vector[1] = 1.0

    sample = StandardizedKtcInput(
        sample_id="sample_001",
        level=4,
        delta_v=vector,
        active_measurement_mask=np.ones(2356, dtype=bool),
    )

    result = adapter.run(sample)

    assert result.success is True
    assert result.segmentation_mask.shape == (256, 256)
    assert set(np.unique(result.segmentation_mask)).issubset({0, 1, 2})


def test_standardized_input_rejects_wrong_vector_length():
    try:
        StandardizedKtcInput(sample_id="bad", level=1, delta_v=np.zeros(10))
        assert False, "Expected validation error"
    except ValueError:
        assert True
