"""Unit tests for Hull Plugin geometry analysis."""

import numpy as np
import pytest
from src.ktc_framework.methods.hull_plugin import HullPlugin, HullResult


class TestHullPluginBasic:
    """Test basic hull computation on simple shapes."""

    def test_circle_center_and_area(self):
        """Test hull on a simple circle."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]
        circle = (y - 128) ** 2 + (x - 128) ** 2 <= 30**2
        pred[circle] = 2

        result = HullPlugin.analyze(pred)

        # Circle centered at (128, 128)
        assert result.conductive_center is not None
        assert abs(result.conductive_center[0] - 128) < 5
        assert abs(result.conductive_center[1] - 128) < 5

        # Area should be ~π*30² ≈ 2827
        assert result.conductive_area is not None
        assert 2700 < result.conductive_area < 2900

        # Resistive should be None
        assert result.resistive_center is None

    def test_degenerate_single_pixel(self):
        """Test with fewer than 3 pixels (should return None)."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        pred[100, 100] = 2

        result = HullPlugin.analyze(pred)

        assert result.conductive_center is None
        assert result.conductive_area is None
        assert result.conductive_perimeter is None

    def test_two_circles(self):
        """Test with both resistive and conductive regions."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]

        # Resistive circle at (100, 100)
        res_circle = (y - 100) ** 2 + (x - 100) ** 2 <= 25**2
        pred[res_circle] = 1

        # Conductive circle at (200, 200)
        cond_circle = (y - 200) ** 2 + (x - 200) ** 2 <= 30**2
        pred[cond_circle] = 2

        result = HullPlugin.analyze(pred)

        # Check resistive
        assert result.resistive_center is not None
        assert abs(result.resistive_center[0] - 100) < 5
        assert abs(result.resistive_center[1] - 100) < 5
        assert result.resistive_area is not None

        # Check conductive
        assert result.conductive_center is not None
        assert abs(result.conductive_center[0] - 200) < 5
        assert abs(result.conductive_center[1] - 200) < 5
        assert result.conductive_area is not None


class TestHullPluginValidation:
    """Test input validation."""

    def test_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        pred = np.zeros((300, 300), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected shape"):
            HullPlugin.analyze(pred)

    def test_wrong_dtype(self):
        """Test that wrong dtype raises ValueError."""
        pred = np.zeros((256, 256), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected dtype"):
            HullPlugin.analyze(pred)


class TestHullPluginComparison:
    """Test comparison between prediction and ground truth."""

    def test_perfect_match(self):
        """Test comparison when prediction matches ground truth."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]
        circle = (y - 128) ** 2 + (x - 128) ** 2 <= 30**2
        pred[circle] = 2

        # Same for ground truth
        gt = pred.copy()

        result_pred = HullPlugin.analyze(pred)
        result_gt = HullPlugin.analyze(gt)
        errors = HullPlugin.compare_hulls(result_pred, result_gt)

        # Errors should be near zero
        assert errors["conductive_center_error"] is not None
        assert errors["conductive_center_error"] < 1.0
        assert errors["conductive_area_error"] is not None
        assert abs(errors["conductive_area_error"]) < 10.0

    def test_offset_circles(self):
        """Test comparison with offset circles."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        gt = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]

        # Prediction: circle at (128, 128)
        circle_pred = (y - 128) ** 2 + (x - 128) ** 2 <= 30**2
        pred[circle_pred] = 2

        # Ground truth: circle at (138, 128) (10 pixels offset)
        circle_gt = (y - 138) ** 2 + (x - 128) ** 2 <= 30**2
        gt[circle_gt] = 2

        result_pred = HullPlugin.analyze(pred)
        result_gt = HullPlugin.analyze(gt)
        errors = HullPlugin.compare_hulls(result_pred, result_gt)

        # Center error should be ~10 pixels
        assert errors["conductive_center_error"] is not None
        assert 8.0 < errors["conductive_center_error"] < 12.0

    def test_missing_region(self):
        """Test comparison when one region is missing."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        gt = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]

        # Prediction has conductive
        circle = (y - 128) ** 2 + (x - 128) ** 2 <= 30**2
        pred[circle] = 2

        # GT has no conductive
        result_pred = HullPlugin.analyze(pred)
        result_gt = HullPlugin.analyze(gt)
        errors = HullPlugin.compare_hulls(result_pred, result_gt)

        # Errors should be None where GT is missing
        assert errors["conductive_center_error"] is None
        assert errors["conductive_area_error"] is None


class TestHullPluginMetadata:
    """Test metadata fields in HullResult."""

    def test_pixel_counts(self):
        """Test that pixel counts are correct."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]

        # Add 100 pixels of resistive
        pred[50:60, 50:60] = 1
        # Add 200 pixels of conductive
        pred[100:120, 100:110] = 2

        result = HullPlugin.analyze(pred)

        assert result.num_pixels_resistive == 100
        assert result.num_pixels_conductive == 200
        assert result.prediction_shape == (256, 256)

    def test_hull_vertices_shape(self):
        """Test that hull vertices have correct shape."""
        pred = np.zeros((256, 256), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]
        circle = (y - 128) ** 2 + (x - 128) ** 2 <= 30**2
        pred[circle] = 2

        result = HullPlugin.analyze(pred)

        # Hull should exist and have (N, 2) shape
        assert result.conductive_hull is not None
        assert result.conductive_hull.ndim == 2
        assert result.conductive_hull.shape[1] == 2
        # Should have at least 3 vertices
        assert result.conductive_hull.shape[0] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
