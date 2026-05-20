import sys
import os
import pytest
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, BASE_DIR)

from ktc_framework.utils.mock_mesh import make_dummy_batch_with_mesh
from ktc_framework.methods.mock_method_plugin import MockMethodPlugin
from ktc_framework.methods.backprojection import BackProjection
from ktc_framework.methods.gauss_newton import GaussNewton
from ktc_framework.methods.level_set_plugin import LevelSetPlugin
from ktc_framework.methods.hull_plugin import HullPlugin
from ktc_framework.methods.segment import segment

@pytest.mark.parametrize("plugin_class", [
    MockMethodPlugin,
    BackProjection,
    GaussNewton
])
def test_method_plugins_with_mesh(plugin_class):
    batch = make_dummy_batch_with_mesh()
    plugin = plugin_class()
    labels = plugin.reconstruct(batch)
    assert labels.shape == (256, 256)
    assert set(np.unique(labels)).issubset({0, 1, 2})

def test_segment_function():
    sigma = np.linspace(0,1,256*256).reshape(256,256)
    labels = segment(sigma)
    assert labels.shape == (256,256)
    assert set(np.unique(labels)).issubset({0,1,2})

def test_levelset_plugin():
    sigma = np.zeros((256,256))
    sigma[50:100,50:100] = 1.0
    plugin = LevelSetPlugin()
    result = plugin.run(sigma)
    assert "contours" in result
    assert "n_objects" in result
    assert isinstance(result["contours"], list)
    assert isinstance(result["n_objects"], int)

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