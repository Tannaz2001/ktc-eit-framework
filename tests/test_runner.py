"""Tests for BatchRunner — verifies runner completes without errors for all
registered method and data plugin combinations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.ktc_framework.runner.experiment_runner import BatchRunner


# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

MOCK_CONFIG = {
    "data_plugin": "MockDataPlugin",
    "mesh_path": "data/Mesh/",
    "levels": [1, 2],
    "samples": ["A", "B"],
    "methods": ["MockMethodPlugin"],
    "dataset_root": "data/EvaluationData_full/",
    "output_dir": "outputs/",
}

MULTI_METHOD_CONFIG = {
    "data_plugin": "MockDataPlugin",
    "mesh_path": "data/Mesh/",
    "levels": [1],
    "samples": ["A"],
    "methods": ["MockMethodPlugin", "BackProjection", "GaussNewton"],
    "dataset_root": "data/EvaluationData_full/",
    "output_dir": "outputs/",
}

SINGLE_CONFIG = {
    "data_plugin": "MockDataPlugin",
    "mesh_path": "data/Mesh/",
    "levels": [1],
    "samples": ["A"],
    "methods": ["MockMethodPlugin"],
    "dataset_root": "data/EvaluationData_full/",
    "output_dir": "outputs/",
}


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_runner_returns_correct_number_of_results(tmp_path):
    """1 method × 2 levels × 2 samples = 4 results."""
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    assert len(results) == 4


def test_each_result_has_required_fields(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    required = {"method", "level", "sample", "metrics", "runtime_ms",
                "output_shape", "git_sha", "composite_score", "grade"}
    for r in results:
        assert required.issubset(r.keys()), f"Missing fields: {required - r.keys()}"


def test_metrics_has_required_keys(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    required_metrics = {"ktc_score", "dice_resistive", "dice_conductive",
                        "iou_resistive", "iou_conductive"}
    for r in results:
        assert required_metrics.issubset(r["metrics"].keys())


def test_output_shape_is_256x256(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert r["output_shape"] == [256, 256]


def test_runtime_is_positive(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert r["runtime_ms"] >= 0


# ---------------------------------------------------------------------------
# Composite score + grade
# ---------------------------------------------------------------------------

def test_composite_score_is_float(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert isinstance(r["composite_score"], float)


def test_grade_is_valid_letter(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert r["grade"] in {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Degradation slope
# ---------------------------------------------------------------------------

def test_degradation_slope_present_after_run(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert "degradation_slope" in r
        assert isinstance(r["degradation_slope"], float)


# ---------------------------------------------------------------------------
# JSON output files
# ---------------------------------------------------------------------------

def test_scores_json_is_saved(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    runner.run()
    assert (tmp_path / "scores.json").exists()


def test_scores_json_content_is_valid(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    runner.run()
    with (tmp_path / "scores.json").open() as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 4


def test_scores_nested_json_is_saved(tmp_path):
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    runner.run()
    assert (tmp_path / "scores_nested.json").exists()


def test_scores_json_strips_private_keys(tmp_path):
    """_gt and _pred arrays must not appear in scores.json."""
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=tmp_path)
    runner.run()
    with (tmp_path / "scores.json").open() as f:
        data = json.load(f)
    for entry in data:
        for key in entry:
            assert not key.startswith("_"), f"Private key '{key}' leaked into scores.json"


# ---------------------------------------------------------------------------
# Multiple methods
# ---------------------------------------------------------------------------

def test_all_three_methods_run_without_error(tmp_path):
    """BackProjection and GaussNewton must register and produce results."""
    runner = BatchRunner(config=MULTI_METHOD_CONFIG, output_dir=tmp_path)
    results = runner.run()
    methods_seen = {r["method"] for r in results}
    assert "MockMethodPlugin" in methods_seen
    assert "BackProjection" in methods_seen
    assert "GaussNewton" in methods_seen


def test_multi_method_result_count(tmp_path):
    """3 methods × 1 level × 1 sample = 3 results."""
    runner = BatchRunner(config=MULTI_METHOD_CONFIG, output_dir=tmp_path)
    results = runner.run()
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Graceful failure handling
# ---------------------------------------------------------------------------

def test_unknown_method_is_skipped(tmp_path):
    """Runner must not crash if a method name is not registered."""
    config = {**SINGLE_CONFIG, "methods": ["MockMethodPlugin", "NonExistentMethod"]}
    runner = BatchRunner(config=config, output_dir=tmp_path)
    results = runner.run()
    # Only MockMethodPlugin should produce results
    assert len(results) == 1
    assert results[0]["method"] == "MockMethodPlugin"


def test_unknown_data_plugin_falls_back(tmp_path):
    """Runner falls back to MockDataPlugin if plugin name is not registered."""
    config = {**SINGLE_CONFIG, "data_plugin": "UnknownPlugin"}
    runner = BatchRunner(config=config, output_dir=tmp_path)
    results = runner.run()
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def test_output_dir_is_created(tmp_path):
    out = tmp_path / "nested" / "output"
    runner = BatchRunner(config=MOCK_CONFIG, output_dir=out)
    runner.run()
    assert out.exists()
