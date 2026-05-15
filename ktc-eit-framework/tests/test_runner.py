"""Tests for BatchRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ktc_framework.runner.experiment_runner import BatchRunner


SAMPLE_CONFIG = {
    "data_plugin": "MockDataPlugin",
    "mesh_path": "data/Mesh/",
    "levels": [1, 2],
    "samples": ["A", "B"],
    "methods": ["MockMethodPlugin"],
    "dataset_root": "data/EvaluationData_full/",
    "output_dir": "outputs/",
}


def test_runner_returns_correct_number_of_results(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    results = runner.run()
    # 1 method x 2 levels x 2 samples = 4
    assert len(results) == 4


def test_each_result_has_required_fields(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert "method" in r
        assert "level" in r
        assert "sample" in r
        assert "metrics" in r
        assert "runtime_ms" in r
        assert "output_shape" in r
        assert "git_sha" in r


def test_metrics_has_required_keys(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        m = r["metrics"]
        assert "ktc_score" in m
        assert "dice_resistive" in m
        assert "dice_conductive" in m
        assert "iou_resistive" in m
        assert "iou_conductive" in m


def test_output_shape_is_256x256(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert r["output_shape"] == [256, 256]


def test_runtime_is_positive(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    results = runner.run()
    for r in results:
        assert r["runtime_ms"] >= 0


def test_scores_json_is_saved(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    runner.run()
    scores_file = tmp_path / "scores.json"
    assert scores_file.exists()


def test_scores_json_content_is_valid(tmp_path):
    runner = BatchRunner(config=SAMPLE_CONFIG, output_dir=tmp_path)
    runner.run()
    scores_file = tmp_path / "scores.json"
    with scores_file.open() as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 4
