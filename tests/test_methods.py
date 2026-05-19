import sys
import os
import pytest
import numpy as np
from collections import namedtuple

# -----------------------------
# Ensure Python can find ktc_framework
# -----------------------------
sys.path.insert(0, os.path.abspath("src"))

# -----------------------------
# Import all plugins
# -----------------------------
from ktc_framework.methods.mock_method_plugin import MockMethodPlugin
from ktc_framework.methods.backprojection import BackProjection
from ktc_framework.methods.gauss_newton import GaussNewton
from ktc_framework.methods.level_set_plugin import LevelSetPlugin
from ktc_framework.methods.hull_plugin import HullPlugin
from ktc_framework.methods.segment import segment

# -----------------------------
# Dummy DataBatch for testing
# -----------------------------
DataBatch = namedtuple("DataBatch", ["voltages", "injection_patterns", "ground_truth", "mesh"])

def make_dummy_batch():
    voltages = np.random.randn(2356).astype(np.float64)        # 76 injections × 31 voltages
    injection_patterns = np.zeros((32, 76), dtype=np.float64)  # KTC adjacent-pair protocol
    for i in range(32):
        injection_patterns[i, i % 76] = 1.0
        injection_patterns[i, (i+1) % 76] = -1.0
    ground_truth = np.random.randint(0,3, size=(256,256), dtype=np.uint8)
    mesh = None  # placeholder for pyEIT mesh object
    return DataBatch(voltages, injection_patterns, ground_truth, mesh)

# -----------------------------
# Test Method Plugins
# -----------------------------
@pytest.mark.parametrize("plugin_class", [
    MockMethodPlugin,
    BackProjection,
    GaussNewton
])
def test_method_plugins(plugin_class):
    batch = make_dummy_batch()
    plugin = plugin_class()
    labels = plugin.reconstruct(batch)
    assert labels.shape == (256,256)
    assert set(np.unique(labels)).issubset({0,1,2})

# -----------------------------
# Test segment helper
# -----------------------------
def test_segment_function():
    sigma = np.linspace(0,1,256*256).reshape(256,256)
    labels = segment(sigma)
    assert labels.shape == (256,256)
    assert set(np.unique(labels)).issubset({0,1,2})

# -----------------------------
# Test LevelSetPlugin
# -----------------------------
def test_levelset_plugin():
    sigma = np.zeros((256,256))
    sigma[50:100,50:100] = 1.0
    plugin = LevelSetPlugin()
    result = plugin.run(sigma)
    assert "contours" in result
    assert "n_objects" in result
    assert isinstance(result["contours"], list)
    assert isinstance(result["n_objects"], int)

# -----------------------------
# Test HullPlugin
# -----------------------------
def test_hull_plugin():
    seg = np.zeros((256,256), dtype=int)
    seg[50:80,50:80] = 1
    seg[100:130,100:130] = 2
    plugin = HullPlugin()
    features1 = plugin.run(seg, target_label=1)
    features2 = plugin.run(seg, target_label=2)
    assert isinstance(features1, list)
    assert isinstance(features2, list)
    assert all("centroid" in f for f in features1)
    assert all("bbox" in f for f in features2)