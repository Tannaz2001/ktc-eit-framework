import logging
import numpy as np
from models import EITSample, ProcessedSample

logger = logging.getLogger(__name__)


class FeatureTransformer:
    def __init__(self):
        self.global_std: float | None = None
        self.global_mean: float | None = None
        self._is_fitted: bool = False

    def fit(self, reference_samples: list[EITSample]) -> None:
        if not reference_samples:
            logger.warning("No reference samples provided to FeatureTransformer.fit()")
            return

        logger.info("FeatureTransformer fitting on %d reference samples.", len(reference_samples))
        all_delta_v = []
        for sample in reference_samples:
            delta_v = sample.v_meas - sample.v_ref
            all_delta_v.append(delta_v)

        stacked = np.concatenate(all_delta_v)
        self.global_std = float(np.std(stacked))
        self.global_mean = float(np.mean(stacked))
        self._is_fitted = True

    def process(self, sample: EITSample) -> ProcessedSample:
        logger.debug("FeatureTransformer processing sample: %s", sample.sample_id)

        delta_v = sample.v_meas - sample.v_ref
        delta_v = self._normalize(delta_v)

        return ProcessedSample(
            sample_id=sample.sample_id,
            level=sample.level,
            delta_v=delta_v,
            metadata=sample.metadata,
        )

    def _normalize(self, delta_v: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureTransformer must be fitted before calling process()"
            )
        if self.global_std and self.global_std != 0:
            return (delta_v - self.global_mean) / self.global_std
        return delta_v - self.global_mean
