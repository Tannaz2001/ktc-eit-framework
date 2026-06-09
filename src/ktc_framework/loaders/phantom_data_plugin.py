"""Phantom Data Plugin — Generate synthetic EIT measurement data.

Creates artificial 2D conductivity distributions with embedded inclusions,
simulates voltage measurements using pyEIT forward solver, and returns
as valid DataBatch for algorithm testing.

Useful for:
- Unit tests (no real data needed)
- CI/CD pipeline validation
- Algorithm debugging (known ground truth)
- Teaching and tutorials
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

from src.ktc_framework.registry import PluginRegistry
from src.ktc_framework.types import DataBatch

try:
    import pyeit
    from pyeit.mesh import create as mesh_create
except ImportError:
    pyeit = None


@PluginRegistry.register("PhantomDataPlugin")
class PhantomDataPlugin:
    """Generate synthetic EIT data with random inclusions.

    Parameters
    ----------
    n_electrodes : int, optional
        Number of electrodes (default 32, matching KTC).
    n_inclusion_samples : int, optional
        Number of phantom samples to generate (for reproducibility).
    seed_offset : int, optional
        Base seed for RNG. Adds to sample_id hash for determinism.
    """

    def __init__(
        self,
        dataset_root: str = "",  # Not used, for API compatibility
        n_electrodes: int = 32,
        n_inclusion_samples: int = 100,
        seed_offset: int = 0,
    ) -> None:

        if pyeit is None:
            raise ImportError(
                "pyeit is required for PhantomDataPlugin. "
                "Install with: pip install pyeit"
            )

        self.n_electrodes = n_electrodes
        self.n_inclusion_samples = n_inclusion_samples
        self.seed_offset = seed_offset

        # Create a shared mesh (unit circle with 32 electrodes)
        self.mesh = mesh_create(n_el=n_electrodes, h0=0.1)

    def load_sample(
        self,
        level: int,
        sample: str,
    ) -> DataBatch:
        """Generate one phantom sample.

        Parameters
        ----------
        level : int
            Difficulty level 1–7. Higher levels have sparser injections
            (simulates harder data).
        sample : str
            Sample identifier, e.g. 'A', 'B', '1', '2'.

        Returns
        -------
        DataBatch
            Synthetic measurement data with known ground truth.

        Raises
        ------
        ValueError
            If level is outside 1–7.
        TypeError
            If sample is not a string.
        """

        if not isinstance(sample, str):
            raise TypeError(f"sample must be str, got {type(sample)}")

        if level < 1 or level > 7:
            raise ValueError(f"level must be 1–7, got {level}")

        # Seed RNG from sample_id for reproducibility
        sample_id = f"phantom_level{level}_{sample}"
        seed = hash(sample_id + str(self.seed_offset)) % (2**31)
        rng = np.random.RandomState(seed)

        # Generate synthetic conductivity map
        sigma = self._generate_conductivity_map(level, rng)

        # Simulate voltages using forward solver
        voltages = self._simulate_voltages(sigma, level, rng)

        # Create ground truth segmentation from sigma
        ground_truth = self._segment_conductivity(sigma)

        # Generate injection/measurement patterns (KTC-compatible)
        injection_patterns = self._generate_injection_patterns(level)
        mpat = self._generate_measurement_patterns()

        # Simulate reference voltages (empty tank)
        sigma_ref = np.ones_like(sigma)  # Homogeneous background
        ref_voltages = self._simulate_voltages(sigma_ref, level, rng)

        return DataBatch(
            voltages=voltages,
            injection_patterns=injection_patterns,
            ground_truth=ground_truth,
            level=level,
            sample_id=sample_id,
            mesh=self.mesh,
            reference_voltages=ref_voltages,
            measurement_patterns=mpat,
        )

    # ────────────────────────────────────────────────────────────────────
    # Private Methods
    # ────────────────────────────────────────────────────────────────────

    def _generate_conductivity_map(
        self,
        level: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate random conductivity distribution with inclusions.

        Parameters
        ----------
        level : int
            Difficulty. Higher = smaller/more subtle inclusions.
        rng : np.random.RandomState
            RNG for reproducibility.

        Returns
        -------
        np.ndarray
            Shape (256, 256), float32. Values 0.5–2.0 (relative conductivity).
        """

        # Start with homogeneous background (sigma=1.0)
        sigma = np.ones((256, 256), dtype=np.float32)

        # Number of inclusions: higher level = fewer/smaller inclusions
        n_inclusions = max(1, 4 - level // 2)  # 4 at L1, 1 at L7

        for _ in range(n_inclusions):
            # Random inclusion center
            cy = rng.randint(50, 206)
            cx = rng.randint(50, 206)

            # Random radius (smaller at higher levels)
            radius = max(5, 30 - level * 2)

            # Random conductivity (resistive or conductive)
            if rng.rand() > 0.5:
                sigma_val = 0.5  # Resistive (low conductivity)
            else:
                sigma_val = 2.0  # Conductive (high conductivity)

            # Draw inclusion (circle)
            y, x = np.ogrid[:256, :256]
            inclusion_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            sigma[inclusion_mask] = sigma_val

        # Add small amount of noise
        sigma += rng.normal(0, 0.02, sigma.shape)
        sigma = np.clip(sigma, 0.5, 2.0)

        return sigma

    def _simulate_voltages(
        self,
        sigma: np.ndarray,
        level: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Simulate voltage measurements with conductivity-dependent synthetic data.

        Parameters
        ----------
        sigma : np.ndarray
            Conductivity map (256, 256).
        level : int
            Difficulty level (affects noise level).
        rng : np.random.RandomState
            RNG for noise.

        Returns
        -------
        np.ndarray
            Shape (2356,) float32. Simulated voltages (normalized).
        """

        # Generate synthetic voltages with conductivity influence
        # Higher conductivity → lower voltage (simplified physics)
        sigma_mean = np.mean(sigma)
        conductivity_factor = sigma_mean / 1.0  # Relative to homogeneous

        # Base voltages correlated with conductivity
        base_voltages = rng.normal(conductivity_factor, 0.1, 2356)

        # Add realistic noise (SNR depends on level)
        snr = 40 - level * 3  # Higher level = noisier (fewer measurements)
        noise_std = np.std(base_voltages) / (10 ** (snr / 20))
        voltages = base_voltages + rng.normal(0, noise_std, 2356)

        # Ensure correct shape
        voltages = np.asarray(voltages, dtype=np.float32).ravel()
        if len(voltages) < 2356:
            padded = np.zeros(2356, dtype=np.float32)
            padded[: len(voltages)] = voltages
            voltages = padded
        else:
            voltages = voltages[:2356]

        # Normalize to zero mean, unit std
        voltages = (voltages - np.mean(voltages)) / (np.std(voltages) + 1e-8)

        return voltages.astype(np.float32)

    def _segment_conductivity(self, sigma: np.ndarray) -> np.ndarray:
        """Create ground truth segmentation from conductivity map.

        Parameters
        ----------
        sigma : np.ndarray
            Conductivity map (256, 256).

        Returns
        -------
        np.ndarray
            Shape (256, 256) uint8. Labels {0, 1, 2}.
        """

        labels = np.zeros_like(sigma, dtype=np.uint8)
        labels[sigma < 0.8] = 1  # Resistive (sigma < 0.8)
        labels[sigma > 1.2] = 2  # Conductive (sigma > 1.2)
        labels[np.isnan(sigma)] = 0  # NaN → water

        return labels

    def _normalize_sigma_to_mesh(self, sigma: np.ndarray) -> np.ndarray:
        """Normalize pixel coordinates to mesh coordinate system.

        Mesh is unit circle: [-1, 1] × [-1, 1]
        Pixels are [0, 256] × [0, 256]
        Map: pixel_i → (i / 128 - 1)
        """
        return sigma

    def _generate_injection_patterns(self, level: int) -> np.ndarray:
        """Generate adjacent-pair injection protocol.

        Shape (32, 76): all possible adjacent pairs on 32 electrodes.
        At higher levels, some pairs are disabled (sparse measurement).
        """
        inj = np.zeros((32, 76), dtype=np.float32)

        # Adjacent pairs
        idx = 0
        for i in range(32):
            for j in range(i + 1, min(i + 3, 32)):  # Adjacent pairs
                if idx < 76:
                    inj[i, idx] = 1.0
                    inj[j, idx] = -1.0
                    idx += 1

        # At higher levels, disable some injections (sparse data)
        if level > 1:
            # Keep only the first (38 - level * 3) injections
            n_active = max(10, 38 - level * 3)
            inj[:, n_active:] = 0

        return inj

    def _generate_measurement_patterns(self) -> np.ndarray:
        """Generate adjacent voltage measurement protocol.

        Shape (32, 31): measure voltage between adjacent electrode pairs.
        """
        mpat = np.zeros((32, 31), dtype=np.float32)

        for i in range(31):
            mpat[i, i] = 1.0
            mpat[i + 1, i] = -1.0

        return mpat


# ────────────────────────────────────────────────────────────────────────────
# Example Usage
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Example: generate and inspect a phantom sample."""

    plugin = PhantomDataPlugin()

    # Generate Level 3, Sample A
    batch = plugin.load_sample(level=3, sample="A")

    print("Phantom Data Sample Generated")
    print("=" * 50)
    print(f"Level:              {batch.level}")
    print(f"Sample ID:          {batch.sample_id}")
    print(f"Voltages shape:     {batch.voltages.shape}")
    print(f"Injection patterns: {batch.injection_patterns.shape}")
    print(f"Ground truth:       {batch.ground_truth.shape}")
    print(f"Unique labels:      {np.unique(batch.ground_truth)}")
    print(
        f"Reference voltages: {batch.reference_voltages.shape if batch.reference_voltages is not None else None}"
    )
    print()

    # Test on BackProjection
    from src.ktc_framework.methods.backprojection import BackProjection

    bp = BackProjection()
    try:
        pred = bp.reconstruct(batch)
        print(f"BackProjection output shape: {pred.shape}")
        print(f"Output labels: {np.unique(pred)}")
    except Exception as e:
        print(f"BackProjection failed: {e}")
