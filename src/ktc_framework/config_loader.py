from __future__ import annotations

from pathlib import Path

import yaml

from ktc_framework.config_schema import AdapterConfig


class PydanticAdapterConfigLoader:
    """
    Loads the GREIT adapter config and validates it with Pydantic.
    This is only for the method adapter, not for the dataset loader.
    """

    def load(self, config_path: str | Path) -> AdapterConfig:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
        return AdapterConfig.model_validate(raw_config)
