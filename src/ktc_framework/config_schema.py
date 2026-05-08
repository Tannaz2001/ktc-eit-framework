from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GreitConfig(BaseModel):
    method_name: str = "GREIT"
    model_path: Path
    reconstruction_matrix_key: str = "reconstruction_matrix"

    expected_input_size: int = Field(default=2356, ge=1)
    greit_image_shape: tuple[int, int] = (64, 64)
    output_image_size: tuple[int, int] = (256, 256)

    positive_class: Literal["conductive", "resistive"] = "conductive"
    segmentation_strategy: Literal["threshold", "otsu"] = "threshold"
    threshold: float = Field(default=0.20, gt=0.0)
    clip_percentile: float = Field(default=99.0, gt=0.0, le=100.0)

    zero_inactive_measurements: bool = True
    fail_on_invalid_input: bool = True

    @field_validator("greit_image_shape", "output_image_size")
    @classmethod
    def validate_image_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("Image shape must contain exactly two values.")
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("Image dimensions must be positive.")
        return value


class AdapterConfig(BaseModel):
    greit: GreitConfig
