import sys
import os
import pytest
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, BASE_DIR)


from ktc_framework.loaders.mock_data_plugin import MockDataPlugin
from ktc_framework.methods.backprojection import BackProjection
from ktc_framework.methods.level_set_plugin import LevelSetPlugin
from ktc_framework.plugins.hull_plugin import HullAnalyzer
from ktc_framework.methods.segment import segment

@pytest.mark.parametrize("plugin_class", [
    BackProjection
])
def test_method_plugins_with_mock_data(plugin_class):
    batch = MockDataPlugin("").load_sample(level=1, sample="A")
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
    seg = np.zeros((256, 256), dtype=np.uint8)
    seg[50:80, 50:80] = 1
    seg[100:130, 100:130] = 2
    analyzer = HullAnalyzer()
    resistive = analyzer.extract(seg, class_id=1)
    conductive = analyzer.extract(seg, class_id=2)
    assert not resistive.empty
    assert not conductive.empty
    assert resistive.centroid is not None
    assert conductive.centroid is not None
    assert resistive.area_px > 0
    assert conductive.area_px > 0
