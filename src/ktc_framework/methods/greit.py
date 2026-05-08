from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom

from ktc_framework.config_schema import GreitConfig
from ktc_framework.contracts import AdapterResult, ReconstructionAdapter, StandardizedKtcInput
from ktc_framework.segmentation import GreitSegmentationPostProcessor


@dataclass(frozen=True)
class GreitModel:
    matrix: np.ndarray
    image_shape: tuple[int, int]
    expected_input_size: int


class GreitModelLoader:
    """
    Loads a GREIT reconstruction matrix.

    The model must match the stabilized adapter input contract:
    matrix shape = GREIT pixels × 2356.
    """

    def __init__(self, config: GreitConfig):
        self.config = config

    def load(self) -> GreitModel:
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"GREIT model file not found: {self.config.model_path}"
            )

        model_file = np.load(self.config.model_path, allow_pickle=True)
        matrix_key = self.config.reconstruction_matrix_key

        if matrix_key not in model_file.files:
            raise ValueError(
                f"GREIT model must contain key '{matrix_key}'. "
                f"Available keys: {model_file.files}"
            )

        matrix = np.asarray(model_file[matrix_key], dtype=np.float64)
        expected_pixels = self.config.greit_image_shape[0] * self.config.greit_image_shape[1]

        if matrix.ndim != 2:
            raise ValueError("GREIT reconstruction matrix must be 2D.")

        if matrix.shape != (expected_pixels, self.config.expected_input_size):
            raise ValueError(
                "GREIT matrix shape mismatch. "
                f"Expected {(expected_pixels, self.config.expected_input_size)}, got {matrix.shape}."
            )

        return GreitModel(
            matrix=matrix,
            image_shape=self.config.greit_image_shape,
            expected_input_size=self.config.expected_input_size,
        )


class GreitInputPreprocessor:
    """
    Adapter-side preprocessing only.

    It does not load files and does not calculate delta voltage from raw voltage files.
    It only prepares the standardized vector received from the colleague-owned
    data loader / batch engine.
    """

    def __init__(self, config: GreitConfig):
        self.config = config

    def transform(self, sample: StandardizedKtcInput) -> np.ndarray:
        vector = np.asarray(sample.delta_v, dtype=np.float64).reshape(-1)

        if self.config.zero_inactive_measurements and sample.active_measurement_mask is not None:
            vector = self._zero_inactive_values(vector, sample.active_measurement_mask)

        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        self._validate(vector)
        return vector

    def _zero_inactive_values(self, vector: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cleaned = vector.copy()
        cleaned[~mask.astype(bool)] = 0.0
        return cleaned

    def _validate(self, vector: np.ndarray) -> None:
        if vector.size != self.config.expected_input_size:
            raise ValueError(
                f"GREIT expected {self.config.expected_input_size} values, got {vector.size}."
            )


class GreitReconstructionEngine:
    """
    Core GREIT mathematical wrapper.
    """

    def __init__(self, model: GreitModel):
        self.model = model

    def reconstruct(self, vector: np.ndarray) -> np.ndarray:
        flat_image = self.model.matrix @ vector
        flat_image = np.real(flat_image)
        return flat_image.reshape(self.model.image_shape)


class GreitImagePostProcessor:
    """
    Resizes, normalizes, and quantizes the GREIT image.
    """

    def __init__(self, config: GreitConfig):
        self.config = config
        self.segmenter = GreitSegmentationPostProcessor(config)

    def resize(self, image: np.ndarray) -> np.ndarray:
        target_height, target_width = self.config.output_image_size
        source_height, source_width = image.shape
        return zoom(
            image,
            zoom=(target_height / source_height, target_width / source_width),
            order=1,
        )

    def normalize(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float64)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        clip_value = np.percentile(np.abs(image), self.config.clip_percentile)
        if clip_value <= 1e-12:
            return np.zeros_like(image, dtype=np.float64)

        return np.clip(image, -clip_value, clip_value) / clip_value

    def to_segmentation(self, image: np.ndarray) -> np.ndarray:
        resized = self.resize(image)
        normalized = self.normalize(resized)
        return self.segmenter.to_mask(normalized)


class GreitAdapter(ReconstructionAdapter):
    """
    Final GREIT adapter used by the batch engine.

    External colleague-owned modules should call:
    result = adapter.run(standardized_sample)
    """

    name = "GREIT"

    def __init__(self, config: GreitConfig):
        self.config = config
        self.model = GreitModelLoader(config).load()
        self.input_preprocessor = GreitInputPreprocessor(config)
        self.reconstruction_engine = GreitReconstructionEngine(self.model)
        self.image_postprocessor = GreitImagePostProcessor(config)

    def preprocess(self, sample: StandardizedKtcInput) -> np.ndarray:
        return self.input_preprocessor.transform(sample)

    def reconstruct(self, preprocessed_input: np.ndarray) -> np.ndarray:
        return self.reconstruction_engine.reconstruct(preprocessed_input)

    def postprocess(self, raw_reconstruction: np.ndarray) -> np.ndarray:
        return self.image_postprocessor.to_segmentation(raw_reconstruction)

    def run(self, sample: StandardizedKtcInput) -> AdapterResult:
        start_time = time.perf_counter()

        try:
            vector = self.preprocess(sample)
            raw_image = self.reconstruct(vector)
            normalized_image = self.image_postprocessor.normalize(
                self.image_postprocessor.resize(raw_image)
            )
            segmentation_mask = self.postprocess(raw_image)
            runtime_seconds = time.perf_counter() - start_time

            return AdapterResult(
                method_name=self.config.method_name,
                sample_id=sample.sample_id,
                level=sample.level,
                raw_image=raw_image,
                normalized_image=normalized_image,
                segmentation_mask=segmentation_mask,
                runtime_seconds=runtime_seconds,
                success=True,
                metadata={
                    "input_contract": "standardized_delta_v_2356",
                    "adapter_only": True,
                    "greit_image_shape": self.config.greit_image_shape,
                    "output_image_size": self.config.output_image_size,
                    "active_measurements": self._count_active(sample),
                },
            )

        except Exception as error:
            if self.config.fail_on_invalid_input:
                raise

            runtime_seconds = time.perf_counter() - start_time
            return AdapterResult(
                method_name=self.config.method_name,
                sample_id=sample.sample_id,
                level=sample.level,
                raw_image=np.zeros(self.config.greit_image_shape),
                normalized_image=np.zeros(self.config.output_image_size),
                segmentation_mask=np.zeros(self.config.output_image_size, dtype=np.uint8),
                runtime_seconds=runtime_seconds,
                success=False,
                error_message=str(error),
                metadata={"adapter_only": True},
            )

    def _count_active(self, sample: StandardizedKtcInput) -> int:
        if sample.active_measurement_mask is None:
            return self.config.expected_input_size
        return int(np.asarray(sample.active_measurement_mask).astype(bool).sum())
