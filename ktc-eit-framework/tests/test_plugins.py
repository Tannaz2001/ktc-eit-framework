"""
test_plugins.py
---------------
Shape and output validation for LevelSetPlugin and HullPlugin.
Uses purely synthetic arrays — no KTC dataset required.
"""

import numpy as np
import pytest

from src.ktc_framework.methods.level_set_plugin import LevelSetPlugin
from src.ktc_framework.methods.hull_plugin import HullPlugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _circle_array(radius: int = 60) -> np.ndarray:
    """256×256 float array with a filled circle (value 1.0) on a zero background."""
    arr = np.zeros((256, 256), dtype=np.float32)
    cy, cx = 128, 128
    y, x = np.ogrid[:256, :256]
    arr[(y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2] = 1.0
    return arr


def _block_array(label: int = 1) -> np.ndarray:
    """256×256 uint8 segmentation with two rectangular blobs of *label*."""
    arr = np.zeros((256, 256), dtype=np.uint8)
    arr[40:80, 40:100] = label    # block A
    arr[150:200, 160:220] = label  # block B
    return arr


# ---------------------------------------------------------------------------
# LevelSetPlugin
# ---------------------------------------------------------------------------

class TestLevelSetPlugin:
    def setup_method(self):
        self.plugin = LevelSetPlugin()

    def test_returns_dict_with_expected_keys(self):
        result = self.plugin.run(_circle_array())
        assert isinstance(result, dict)
        assert "contours" in result
        assert "n_objects" in result

    def test_detects_circle_contour(self):
        result = self.plugin.run(_circle_array())
        # A filled circle should produce at least one closed contour
        assert result["n_objects"] >= 1

    def test_n_objects_matches_contours_length(self):
        result = self.plugin.run(_circle_array())
        assert result["n_objects"] == len(result["contours"])

    def test_each_contour_is_ndarray(self):
        result = self.plugin.run(_circle_array())
        for contour in result["contours"]:
            assert isinstance(contour, np.ndarray)
            assert contour.ndim == 2  # (N, 2) — row/col coordinates

    def test_uniform_array_returns_empty(self):
        uniform = np.ones((256, 256), dtype=np.float32)
        result = self.plugin.run(uniform)
        assert result["n_objects"] == 0
        assert result["contours"] == []

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            self.plugin.run(np.zeros((128, 128), dtype=np.float32))

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError):
            self.plugin.run([[0.0] * 256] * 256)


# ---------------------------------------------------------------------------
# HullPlugin
# ---------------------------------------------------------------------------

class TestHullPlugin:
    def setup_method(self):
        self.plugin = HullPlugin()

    def test_returns_list(self):
        result = self.plugin.run(_block_array(label=1), target_label=1)
        assert isinstance(result, list)

    def test_detects_two_blocks(self):
        result = self.plugin.run(_block_array(label=1), target_label=1)
        assert len(result) == 2

    def test_feature_dict_keys(self):
        result = self.plugin.run(_block_array(label=1), target_label=1)
        for region in result:
            assert "centroid" in region
            assert "area" in region
            assert "bbox" in region
            assert "convex_area" in region

    def test_area_positive(self):
        result = self.plugin.run(_block_array(label=1), target_label=1)
        for region in result:
            assert region["area"] > 0

    def test_centroid_within_image(self):
        result = self.plugin.run(_block_array(label=1), target_label=1)
        for region in result:
            cy, cx = region["centroid"]
            assert 0 <= cy < 256
            assert 0 <= cx < 256

    def test_absent_label_returns_empty(self):
        # label=2 not present in a label=1 block array
        result = self.plugin.run(_block_array(label=1), target_label=2)
        assert result == []

    def test_background_label_zero(self):
        # label 0 fills most of the image; should return one large region
        result = self.plugin.run(_block_array(label=1), target_label=0)
        assert len(result) >= 1
        total_area = sum(r["area"] for r in result)
        assert total_area > 256 * 256 * 0.8  # background dominates

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            self.plugin.run(np.zeros((128, 128), dtype=np.uint8), target_label=1)

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            self.plugin.run(_block_array(), target_label=5)
