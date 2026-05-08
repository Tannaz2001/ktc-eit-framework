from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StandardizedKtcInput(BaseModel):
    """
    Input contract received from the data-loader / batch-engine team.

    This adapter does NOT read .mat/.npy/.csv files.
    It expects the data pipeline to already provide a stabilized 2356-length
    difference-voltage vector.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    level: int = Field(ge=1, le=7)
    delta_v: Any
    active_measurement_mask: Any | None = None
    ground_truth: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("delta_v")
    @classmethod
    def validate_delta_v(cls, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != 2356:
            raise ValueError(
                f"delta_v must be a stabilized 2356-length vector. Got {array.size}."
            )
        return array

    @field_validator("active_measurement_mask")
    @classmethod
    def validate_active_measurement_mask(cls, value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        mask = np.asarray(value).astype(bool).reshape(-1)
        if mask.size != 2356:
            raise ValueError(
                f"active_measurement_mask must have 2356 entries. Got {mask.size}."
            )
        return mask

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth(cls, value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        ground_truth = np.asarray(value)
        if ground_truth.shape != (256, 256):
            raise ValueError(
                f"ground_truth must have shape (256, 256). Got {ground_truth.shape}."
            )
        labels = set(np.unique(ground_truth).astype(int).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"ground_truth contains invalid labels: {labels}.")
        return ground_truth.astype(np.uint8)


class AdapterResult(BaseModel):
    """
    Standard output contract returned to the batch engine, metrics module,
    visualization module, and GUI.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method_name: str
    sample_id: str
    level: int = Field(ge=1, le=7)
    raw_image: Any
    normalized_image: Any
    segmentation_mask: Any
    runtime_seconds: float = Field(ge=0.0)
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output_contract(self) -> "AdapterResult":
        mask = np.asarray(self.segmentation_mask)
        if mask.shape != (256, 256):
            raise ValueError(f"segmentation_mask must be (256, 256). Got {mask.shape}.")
        labels = set(np.unique(mask).astype(int).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"segmentation_mask contains invalid labels: {labels}.")
        self.segmentation_mask = mask.astype(np.uint8)
        self.raw_image = np.asarray(self.raw_image)
        self.normalized_image = np.asarray(self.normalized_image, dtype=np.float64)
        return self


class ReconstructionAdapter(ABC):
    """
    Strict adapter contract required by the project proposal.
    """

    name: str

    @abstractmethod
    def preprocess(self, sample: StandardizedKtcInput) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, preprocessed_input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, raw_reconstruction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def run(self, sample: StandardizedKtcInput) -> AdapterResult:
        raise NotImplementedError
