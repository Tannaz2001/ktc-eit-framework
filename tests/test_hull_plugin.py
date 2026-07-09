"""Unit tests for Hull Plugin.

Comprehensive test coverage for:
- Hull extraction (circle, empty, single pixel, two pixels)
- Hull comparison (perfect, missed, false positive, shifted)
- Qualitative aggregation

Tests follow specification for HullAnalyzer and HullDescriptor classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk

from ktc_framework.plugins.hull_plugin import HullAnalyzer, HullDescriptor
from ktc_framework.metrics.qualitative_metrics import (
    aggregate_qualitative,
    compute_qualitative_sample,
)


class TestHullExtraction:
    """Test convex hull extraction from segmentations."""

    @pytest.fixture
    def hull_analyzer(self):
        return HullAnalyzer()

    def test_circle_detection(self, hull_analyzer):
        """Create filled circle, verify area, centroid, vertex count."""
        segmentation = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        segmentation[rr, cc] = 1

        hull = hull_analyzer.extract(segmentation, class_id=1)

        assert not hull.empty, "Circle should not be empty"
        assert hull.vertex_count >= 8, "Circle should have many vertices"
        assert abs(hull.centroid[0] - 128) < 2, "Centroid row should be ~128"
        assert abs(hull.centroid[1] - 128) < 2, "Centroid col should be ~128"

        # Expected area ≈ π * 40² ≈ 5027 (within 5% tolerance)
        expected_area = np.pi * (40**2)
        assert abs(hull.area_px - expected_area) / expected_area < 0.05

    def test_empty_region(self, hull_analyzer):
        """Create all-zero segmentation, verify empty descriptor."""
        segmentation = np.zeros((256, 256), dtype=np.uint8)
        hull = hull_analyzer.extract(segmentation, class_id=1)

        assert hull.empty, "All-zero array should produce empty descriptor"
        assert hull.area_px == 0
        assert hull.vertex_count == 0

    def test_single_pixel(self, hull_analyzer):
        """Single pixel cannot form hull."""
        segmentation = np.zeros((256, 256), dtype=np.uint8)
        segmentation[128, 128] = 1

        hull = hull_analyzer.extract(segmentation, class_id=1)

        assert hull.empty, "Single pixel should be empty"

    def test_two_pixels(self, hull_analyzer):
        """Two pixels cannot form hull."""
        segmentation = np.zeros((256, 256), dtype=np.uint8)
        segmentation[128, 128] = 1
        segmentation[129, 128] = 1

        hull = hull_analyzer.extract(segmentation, class_id=1)

        assert hull.empty, "Two pixels should be empty"

    def test_collinear_pixels(self, hull_analyzer):
        """Collinear points are degenerate and should be rejected."""
        segmentation = np.zeros((256, 256), dtype=np.uint8)
        # Draw a line of pixels
        segmentation[128, 100:150] = 1

        hull = hull_analyzer.extract(segmentation, class_id=1)

        assert hull.empty, "Collinear points should be empty (degenerate hull)"


class TestHullComparison:
    """Test hull comparison logic."""

    @pytest.fixture
    def hull_analyzer(self):
        return HullAnalyzer()

    def test_perfect_detection(self, hull_analyzer):
        """Identical GT and pred circles should have IoU ≈ 1.0."""
        segmentation = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        segmentation[rr, cc] = 1

        gt_hull = hull_analyzer.extract(segmentation, class_id=1)
        pred_hull = hull_analyzer.extract(segmentation, class_id=1)

        comparison = hull_analyzer.compare(pred_hull, gt_hull)

        assert comparison["detected"], "Perfect detection should be detected"
        assert comparison["hull_iou"] > 0.95, f"IoU should be ~1.0, got {comparison['hull_iou']}"
        assert comparison["centroid_distance_px"] < 1, "Centroid distance should be ~0"
        assert abs(comparison["area_ratio"] - 1.0) < 0.05

    def test_missed_detection(self, hull_analyzer):
        """GT has circle, pred is all zeros."""
        gt_seg = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        gt_seg[rr, cc] = 1

        pred_seg = np.zeros((256, 256), dtype=np.uint8)

        gt_hull = hull_analyzer.extract(gt_seg, class_id=1)
        pred_hull = hull_analyzer.extract(pred_seg, class_id=1)

        comparison = hull_analyzer.compare(pred_hull, gt_hull)

        assert not comparison["detected"], "Missed detection should not be detected"
        assert comparison["hull_iou"] == 0.0
        assert comparison["detection_reason"] == "missed"

    def test_false_positive(self, hull_analyzer):
        """GT is all zeros, pred has circle."""
        gt_seg = np.zeros((256, 256), dtype=np.uint8)

        pred_seg = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        pred_seg[rr, cc] = 1

        gt_hull = hull_analyzer.extract(gt_seg, class_id=1)
        pred_hull = hull_analyzer.extract(pred_seg, class_id=1)

        comparison = hull_analyzer.compare(pred_hull, gt_hull)

        assert not comparison["detected"]
        assert comparison["detection_reason"] == "false_positive"

    def test_shifted_detection(self, hull_analyzer):
        """GT circle at (128,128), pred at (148,148) (shifted 20px)."""
        gt_seg = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        gt_seg[rr, cc] = 1

        pred_seg = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((148, 148), 40, shape=(256, 256))
        pred_seg[rr, cc] = 1

        gt_hull = hull_analyzer.extract(gt_seg, class_id=1)
        pred_hull = hull_analyzer.extract(pred_seg, class_id=1)

        comparison = hull_analyzer.compare(pred_hull, gt_hull)

        # Shifted circles have partial overlap but not full
        assert 0 < comparison["hull_iou"] < 1.0, "IoU should be partial"

        # Expected centroid distance: sqrt(20² + 20²) ≈ 28.3
        expected_dist = np.sqrt(20**2 + 20**2)
        assert (
            abs(comparison["centroid_distance_px"] - expected_dist) < 5
        ), f"Centroid dist should be ~{expected_dist}, got {comparison['centroid_distance_px']}"

        # If hull_iou >= 0.3, should be detected
        if comparison["hull_iou"] >= 0.3:
            assert comparison["detected"]


class TestQualitativeAggregation:
    """Test qualitative metrics aggregation."""

    def test_aggregate_basic(self):
        """Create mock sample results and aggregate."""
        # 5 samples: 3 with resistive_detected=True, 2 with False, 1 with false_positive
        samples = [
            {
                "resistive_in_gt": True,
                "resistive_detected": True,
                "resistive_hull_iou": 0.7,
                "resistive_centroid_dist": 5.0,
                "conductive_in_gt": False,
                "conductive_detected": False,
                "conductive_hull_iou": 0.0,
                "conductive_centroid_dist": 0.0,
                "false_positive_resistive": False,
                "false_positive_conductive": False,
            },
            {
                "resistive_in_gt": True,
                "resistive_detected": True,
                "resistive_hull_iou": 0.65,
                "resistive_centroid_dist": 8.0,
                "conductive_in_gt": False,
                "conductive_detected": False,
                "conductive_hull_iou": 0.0,
                "conductive_centroid_dist": 0.0,
                "false_positive_resistive": False,
                "false_positive_conductive": False,
            },
            {
                "resistive_in_gt": True,
                "resistive_detected": True,
                "resistive_hull_iou": 0.72,
                "resistive_centroid_dist": 4.5,
                "conductive_in_gt": False,
                "conductive_detected": False,
                "conductive_hull_iou": 0.0,
                "conductive_centroid_dist": 0.0,
                "false_positive_resistive": False,
                "false_positive_conductive": False,
            },
            {
                "resistive_in_gt": True,
                "resistive_detected": False,
                "resistive_hull_iou": 0.15,
                "resistive_centroid_dist": 50.0,
                "conductive_in_gt": False,
                "conductive_detected": False,
                "conductive_hull_iou": 0.0,
                "conductive_centroid_dist": 0.0,
                "false_positive_resistive": False,
                "false_positive_conductive": False,
            },
            {
                "resistive_in_gt": True,
                "resistive_detected": False,
                "resistive_hull_iou": 0.2,
                "resistive_centroid_dist": 45.0,
                "conductive_in_gt": False,
                "conductive_detected": False,
                "conductive_hull_iou": 0.0,
                "conductive_centroid_dist": 0.0,
                "false_positive_resistive": True,
                "false_positive_conductive": False,
            },
        ]

        agg = aggregate_qualitative(samples)

        assert agg["total_samples"] == 5
        assert agg["resistive_gt_count"] == 5
        assert agg["resistive_detected_count"] == 3
        assert agg["resistive_detected_str"] == "3/5"
        assert agg["resistive_detected_pct"] == 60.0
        assert agg["false_positive_count"] == 1

    def test_aggregate_empty(self):
        """Empty sample list should return empty dict."""
        agg = aggregate_qualitative([])
        assert agg == {}


class TestEndToEnd:
    """End-to-end test: create segmentations, compute qualitative metrics."""

    def test_compute_qualitative_sample_basic(self):
        """Create GT and pred, compute sample qualitative flags."""
        hull_analyzer = HullAnalyzer()

        gt = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        gt[rr, cc] = 1

        pred = np.zeros((256, 256), dtype=np.uint8)
        rr, cc = disk((128, 128), 40, shape=(256, 256))
        pred[rr, cc] = 1

        qual = compute_qualitative_sample(pred, gt, hull_analyzer)

        assert qual["resistive_in_gt"]
        assert qual["resistive_in_pred"]
        assert qual["resistive_detected"]
        assert qual["resistive_hull_iou"] > 0.95
        assert not qual["false_positive_resistive"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
