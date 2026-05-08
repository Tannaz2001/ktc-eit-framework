from __future__ import annotations

import numpy as np
from skimage.filters import threshold_multiotsu

from ktc_framework.config_schema import GreitConfig


class GreitSegmentationPostProcessor:
    """
    Converts a continuous GREIT image into the required KTC class mask.

    Output labels:
    0 = water
    1 = resistive inclusion
    2 = conductive inclusion
    """

    def __init__(self, config: GreitConfig):
        self.config = config

    def to_mask(self, normalized_image: np.ndarray) -> np.ndarray:
        if self.config.segmentation_strategy == "otsu":
            return self._otsu_mask(normalized_image)
        return self._threshold_mask(normalized_image)

    def _threshold_mask(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape, dtype=np.uint8)
        threshold = self.config.threshold

        negative_region = image <= -threshold
        positive_region = image >= threshold

        if self.config.positive_class == "conductive":
            mask[negative_region] = 1
            mask[positive_region] = 2
        else:
            mask[negative_region] = 2
            mask[positive_region] = 1

        return self._validate(mask)

    def _otsu_mask(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape, dtype=np.uint8)

        if np.allclose(image, image.flat[0]):
            return mask

        low_threshold, high_threshold = threshold_multiotsu(image, classes=3)

        negative_region = image <= low_threshold
        positive_region = image >= high_threshold

        if self.config.positive_class == "conductive":
            mask[negative_region] = 1
            mask[positive_region] = 2
        else:
            mask[negative_region] = 2
            mask[positive_region] = 1

        return self._validate(mask)

    def _validate(self, mask: np.ndarray) -> np.ndarray:
        labels = set(np.unique(mask).astype(int).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"Invalid segmentation labels found: {labels}")
        if mask.shape != self.config.output_image_size:
            raise ValueError(
                f"Segmentation shape must be {self.config.output_image_size}. Got {mask.shape}."
            )
        return mask.astype(np.uint8)
