"""
test_mock_plugin.py
-------------------
Verifies that MockDataPlugin produces DataBatch objects whose shapes, dtypes,
and label values exactly match the KTC 2023 dataset contract.

No KTC dataset files are required — all tests use synthetic data.
"""

import numpy as np
import pytest

from src.ktc_framework.loaders.mock_data_plugin import MockDataPlugin
from src.ktc_framework.types import DataBatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plugin() -> MockDataPlugin:
    """A fresh MockDataPlugin instance for each test."""
    return MockDataPlugin()


@pytest.fixture
def batch(plugin: MockDataPlugin) -> DataBatch:
    """A single DataBatch produced by the default sample_id."""
    return plugin.get_batch(level=1, sample_id="mock-0001")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_get_batch_returns_databatch(self, batch: DataBatch) -> None:
        """get_batch() must return a DataBatch NamedTuple."""
        assert isinstance(batch, DataBatch)

    def test_load_sample_returns_databatch(self, plugin: MockDataPlugin) -> None:
        """load_sample() must return a DataBatch NamedTuple."""
        result = plugin.load_sample(level=2, sample="A")
        assert isinstance(result, DataBatch)

    def test_iter_batches_yields_databatch(self, plugin: MockDataPlugin) -> None:
        """iter_batches() must yield DataBatch objects."""
        for batch in plugin.iter_batches(n_batches=3, level=1):
            assert isinstance(batch, DataBatch)


# ---------------------------------------------------------------------------
# voltages
# ---------------------------------------------------------------------------

class TestVoltages:
    def test_shape(self, batch: DataBatch) -> None:
        """voltages must be a flat 1-D vector of length 2356."""
        assert batch.voltages.shape == (2356,)

    def test_dtype(self, batch: DataBatch) -> None:
        """voltages must be float64 to match real KTC data."""
        assert batch.voltages.dtype == np.float64

    def test_not_all_zeros(self, batch: DataBatch) -> None:
        """Voltage vector must contain non-zero values."""
        assert not np.all(batch.voltages == 0.0)


# ---------------------------------------------------------------------------
# injection_patterns
# ---------------------------------------------------------------------------

class TestInjectionPatterns:
    def test_shape(self, batch: DataBatch) -> None:
        """injection_patterns must be (32, 76) — 32 electrodes × 76 FEM cols."""
        assert batch.injection_patterns.shape == (32, 76)

    def test_dtype(self, batch: DataBatch) -> None:
        """injection_patterns must be float64."""
        assert batch.injection_patterns.dtype == np.float64

    def test_adjacent_pair_values(self, batch: DataBatch) -> None:
        """Each row must contain exactly one +1.0 and one −1.0, rest zeros."""
        for row in batch.injection_patterns:
            assert int((row == 1.0).sum()) == 1
            assert int((row == -1.0).sum()) == 1
            assert int((row == 0.0).sum()) == 74  # 76 - 2

    def test_identical_across_samples(self, plugin: MockDataPlugin) -> None:
        """Injection patterns are fixed by the KTC protocol — same for all samples."""
        b1 = plugin.get_batch(level=1, sample_id="mock-0001")
        b2 = plugin.get_batch(level=1, sample_id="mock-0042")
        np.testing.assert_array_equal(b1.injection_patterns, b2.injection_patterns)


# ---------------------------------------------------------------------------
# ground_truth
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_shape(self, batch: DataBatch) -> None:
        """ground_truth must be (256, 256)."""
        assert batch.ground_truth.shape == (256, 256)

    def test_dtype(self, batch: DataBatch) -> None:
        """ground_truth must be uint8."""
        assert batch.ground_truth.dtype == np.uint8

    def test_valid_labels(self, batch: DataBatch) -> None:
        """All pixel values must be in {0, 1, 2}."""
        unique = set(np.unique(batch.ground_truth).tolist())
        assert unique.issubset({0, 1, 2})

    def test_background_dominates(self, batch: DataBatch) -> None:
        """Background (label 0) must occupy the majority of pixels."""
        bg_fraction = (batch.ground_truth == 0).mean()
        assert bg_fraction > 0.7  # realistic threshold; mock targets ~90 %


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_level_stored(self, plugin: MockDataPlugin) -> None:
        """level must be stored exactly as passed."""
        for lv in [1, 3, 7]:
            b = plugin.get_batch(level=lv, sample_id="mock-0001")
            assert b.level == lv

    def test_sample_id_stored(self, plugin: MockDataPlugin) -> None:
        """sample_id must be stored exactly as passed."""
        b = plugin.get_batch(level=1, sample_id="my-custom-id")
        assert b.sample_id == "my-custom-id"

    def test_auto_sample_id(self, plugin: MockDataPlugin) -> None:
        """When sample_id is None a non-empty string must be assigned."""
        b = plugin.get_batch(level=1, sample_id=None)
        assert isinstance(b.sample_id, str) and len(b.sample_id) > 0


# ---------------------------------------------------------------------------
# Reproducibility  (seeded RNG from sample_id)
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_sample_id_identical_voltages(self, plugin: MockDataPlugin) -> None:
        """The same sample_id must always return byte-identical voltages."""
        b1 = plugin.get_batch(level=1, sample_id="mock-0001")
        b2 = plugin.get_batch(level=1, sample_id="mock-0001")
        np.testing.assert_array_equal(b1.voltages, b2.voltages)

    def test_same_sample_id_identical_ground_truth(self, plugin: MockDataPlugin) -> None:
        """The same sample_id must always return byte-identical ground_truth."""
        b1 = plugin.get_batch(level=1, sample_id="mock-0001")
        b2 = plugin.get_batch(level=1, sample_id="mock-0001")
        np.testing.assert_array_equal(b1.ground_truth, b2.ground_truth)

    def test_different_sample_ids_differ(self, plugin: MockDataPlugin) -> None:
        """Different sample_ids must produce different voltages."""
        b1 = plugin.get_batch(level=1, sample_id="mock-0001")
        b2 = plugin.get_batch(level=1, sample_id="mock-0002")
        assert not np.array_equal(b1.voltages, b2.voltages)

    def test_level_does_not_affect_voltages(self, plugin: MockDataPlugin) -> None:
        """Changing level must not change voltages for the same sample_id
        (level is metadata; RNG is seeded only from sample_id)."""
        b1 = plugin.get_batch(level=1, sample_id="mock-0001")
        b2 = plugin.get_batch(level=5, sample_id="mock-0001")
        np.testing.assert_array_equal(b1.voltages, b2.voltages)


# ---------------------------------------------------------------------------
# iter_batches
# ---------------------------------------------------------------------------

class TestIterBatches:
    def test_yields_correct_count(self, plugin: MockDataPlugin) -> None:
        """iter_batches(n) must yield exactly n batches."""
        batches = list(plugin.iter_batches(n_batches=5, level=1))
        assert len(batches) == 5

    def test_unique_sample_ids(self, plugin: MockDataPlugin) -> None:
        """Each batch yielded by iter_batches must have a unique sample_id."""
        ids = [b.sample_id for b in plugin.iter_batches(n_batches=4, level=1)]
        assert len(set(ids)) == 4

    def test_all_correct_level(self, plugin: MockDataPlugin) -> None:
        """Every batch in iter_batches must carry the requested level."""
        for b in plugin.iter_batches(n_batches=3, level=4):
            assert b.level == 4
