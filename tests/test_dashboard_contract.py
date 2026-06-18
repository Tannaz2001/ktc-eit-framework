"""Regression guard for the backend -> frontend data contract.

Runs the full pipeline on synthetic data (BatchRunner + MockDataPlugin),
projects it through the example_usage.py bridge, then loads the result the
same way the dashboard does (reporting.data_layer) and asserts every view
gets non-empty, correctly-shaped, correctly-oriented data.

This is the test that would have caught:
  * the method-mapping failure that blanked four dashboard views,
  * the inverted KTC polarity (failure tab showing the best runs as worst),
  * schema drift between scores.json and per_run_metrics.json.

Run:  python -m pytest tests/test_dashboard_contract.py -v   (from project root)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ktc_framework.runner.experiment_runner import BatchRunner
from src.ktc_framework.reporting.data_layer import (
    create_method_mapping,
    count_gt_missing,
    filter_by_level,
    load_run_data,
)
from example_usage import project_to_dashboard

METHODS = ["BackProjection", "GaussNewton"]
LEVELS = [1, 2]
SAMPLES = ["A", "B"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run benchmark -> bridge -> dashboard loader once for all tests."""
    tmp = tmp_path_factory.mktemp("contract")
    bench_dir = tmp / "bench"
    config = {
        "data_plugin": "MockDataPlugin",
        "dataset_root": "",
        "mesh_path": "",
        "levels": LEVELS,
        "samples": SAMPLES,
        "methods": METHODS,
        "output_dir": str(bench_dir),
    }
    runner = BatchRunner(config=config, output_dir=bench_dir)
    results = runner.run()

    runs_root = tmp / "runs"
    run_dir = project_to_dashboard(bench_dir, runs_root=runs_root)
    scores, per_run = load_run_data(run_dir)
    return {
        "results": results,
        "runs_root": runs_root,
        "run_dir": run_dir,
        "scores": scores,
        "per_run": per_run,
    }


def test_benchmark_completes_all_runs(pipeline):
    assert len(pipeline["results"]) == len(METHODS) * len(LEVELS) * len(SAMPLES)


def test_scores_shape(pipeline):
    scores = pipeline["scores"]
    assert set(scores.keys()) == set(METHODS)
    for method, metrics in scores.items():
        assert isinstance(metrics.get("ktc_score"), float), (
            f"{method}: scores.json must hold metric-name -> float averages"
        )


def test_per_run_contract(pipeline):
    per_run = pipeline["per_run"]
    assert set(per_run.keys()) == set(METHODS)
    for method, entries in per_run.items():
        assert len(entries) == len(LEVELS) * len(SAMPLES)
        for key, e in entries.items():
            assert key == f"L{e['level']}_{e['sample']}", (
                f"{method}: per_run key {key} must encode the entry's own level/sample"
            )
            for field in ("ktc_score", "composite_score", "grade",
                          "runtime_ms", "level", "sample", "gt_missing"):
                assert field in e, f"{method}/{key}: missing field '{field}'"


def test_method_mapping_is_identity_for_runner_output(pipeline):
    mm = create_method_mapping(pipeline["scores"], pipeline["per_run"])
    for method in METHODS:
        assert mm.get(method) == method


def test_ktc_polarity_is_bounded_and_higher_is_better(pipeline):
    """KTC scores should stay in the expected dashboard range."""
    per_run = pipeline["per_run"]
    means = {
        method: np.mean([e["ktc_score"] for e in entries.values()])
        for method, entries in per_run.items()
    }
    assert all(0.0 <= score <= 1.0 for score in means.values())
    # "worst" selection used by the failure gallery: lowest score first
    for method, entries in per_run.items():
        worst = min(entries.values(), key=lambda e: e["ktc_score"])
        assert worst["ktc_score"] <= means[method]


def test_level_filter_uses_entry_level(pipeline):
    entries = pipeline["per_run"][METHODS[0]]
    only_l1 = filter_by_level(entries, 1, 1)
    assert only_l1, "level filter must not drop everything"
    assert all(e["level"] == 1 for e in only_l1.values())
    assert len(only_l1) == len(SAMPLES)


def test_gt_missing_flag_present_and_false(pipeline):
    assert count_gt_missing(pipeline["per_run"]) == 0


def test_latest_txt_points_at_complete_run(pipeline):
    pointer = pipeline["runs_root"] / "latest.txt"
    assert pointer.exists()
    target = Path(pointer.read_text().strip())
    assert target == pipeline["run_dir"]
    assert (target / "scores.json").exists()
    assert (target / "per_run_metrics.json").exists()


def test_legacy_run_folder_is_normalized(tmp_path):
    """Old example_usage.py wrote {method: {samples: {...}, mean_ktc: x}} —
    the loader must convert it instead of feeding dicts into the views."""
    import json
    legacy = {
        "back_projection": {"samples": {"1": 0.4, "2": 0.6}, "mean_ktc": 0.5},
        "mock_baseline":   {"samples": {"1": 0.0, "2": 0.0}, "mean_ktc": 0.0},
    }
    (tmp_path / "scores.json").write_text(json.dumps(legacy), encoding="utf-8")

    scores, per_run = load_run_data(tmp_path)

    assert scores["back_projection"]["ktc_score"] == pytest.approx(0.5)
    for metrics in scores.values():
        assert all(isinstance(v, float) for v in metrics.values())
    entry = per_run["back_projection"]["L1_1"]
    assert entry["ktc_score"] == pytest.approx(0.4)
    assert entry["level"] == 1 and entry["sample"] == "1"
    assert entry["grade"] == "C"  # composite 40 with backend thresholds


def test_images_arranged_for_dashboard(pipeline):
    """Bridge must lay PNGs out exactly where app.py looks for them."""
    run_dir = pipeline["run_dir"]
    for method in METHODS:
        for level in LEVELS:
            for sample in SAMPLES:
                expected = run_dir / "reconstructions" / f"level_{level}" / f"sample_{sample}" / f"{method}.png"
                assert expected.exists(), f"missing dashboard image: {expected}"
        # overlay fallback is sample-keyed (level 1 only, by design)
        for sample in SAMPLES:
            assert (run_dir / "error_overlays" / f"{method}_sample_{sample}.png").exists()
