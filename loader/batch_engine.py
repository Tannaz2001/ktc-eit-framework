import numpy as np
from models import EITSample, ProcessedSample

class FeatureTransformer:
    def __init__(self, use_gp_interpolation: bool = True):
        self.use_gp = use_gp_interpolation
        self.global_std: float | None = None
        self.global_mean: float | None = None
        self._is_fitted: bool = False

    def fit(self, reference_samples: list[EITSample]) -> None:
        """
        Computes global statistics on a reference set BEFORE parallel processing.
        This prevents data leakage and ensures consistent scaling.
        """
        if not reference_samples:
            return

        all_delta_v = []
        for sample in reference_samples:
            v_clean, _ = self._apply_gp(sample.v_meas)
            delta_v = v_clean - sample.v_ref
            all_delta_v.append(delta_v)
            
        stacked = np.concatenate(all_delta_v)
        self.global_std = float(np.std(stacked))
        self.global_mean = float(np.mean(stacked))
        self._is_fitted = True

    def process(self, sample: EITSample) -> ProcessedSample:
        """Applies all feature engineering steps sequentially."""
        
        # Step 1: Missing Data Imputation (Level 7 Guard)
        v_clean, mask = self._apply_gp(sample.v_meas)
        
        # Step 2: Difference Imaging Feature ($\Delta V$)
        # This cancels out systematic hardware errors.
        delta_v = v_clean - sample.v_ref
        
        # Step 3: Normalization (Optional, based on solver requirements)
        delta_v = self._normalize(delta_v)
        
        return ProcessedSample(
            sample_id=sample.sample_id,
            level=sample.level,
            delta_v=delta_v,
            gp_mask=mask,
            metadata=sample.metadata
        )

    def _apply_gp(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Placeholder for your Gaussian Process logic."""
        mask = np.isnan(v)
        if not mask.any() or not self.use_gp:
            return v, mask
            
        v_clean = np.copy(v)
        # TODO: Insert your actual GP interpolation here
        v_clean[mask] = 0.0 # Temporary fallback
        return v_clean, mask

    def _normalize(self, delta_v: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("FeatureTransformer must be fitted on a reference set before calling process()")
        
        # Standardize using global statistics
        if self.global_std and self.global_std != 0:
            return (delta_v - self.global_mean) / self.global_std
        return delta_v - self.global_mean