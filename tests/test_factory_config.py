from __future__ import annotations

from pathlib import Path

import yaml

from ktc_framework.config_loader import PydanticAdapterConfigLoader


def test_pydantic_config_loader_validates_adapter_config(tmp_path: Path):
    config_path = tmp_path / "greit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "greit": {
                    "method_name": "GREIT",
                    "model_path": "models/greit_fixed_2356.npz",
                    "expected_input_size": 2356,
                    "greit_image_shape": [64, 64],
                    "output_image_size": [256, 256],
                    "positive_class": "conductive",
                    "segmentation_strategy": "threshold",
                    "threshold": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )

    config = PydanticAdapterConfigLoader().load(config_path)

    assert config.greit.method_name == "GREIT"
    assert config.greit.expected_input_size == 2356
