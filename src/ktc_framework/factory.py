from __future__ import annotations

from pathlib import Path

from ktc_framework.config_loader import PydanticAdapterConfigLoader
from ktc_framework.methods.greit import GreitAdapter


class GreitAdapterFactory:
    """
    Small factory used by the batch engine to create the GREIT adapter.
    """

    def from_yaml(self, config_path: str | Path) -> GreitAdapter:
        config = PydanticAdapterConfigLoader().load(config_path)
        return GreitAdapter(config.greit)
