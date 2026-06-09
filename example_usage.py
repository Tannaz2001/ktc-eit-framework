"""
example_usage.py — REAL-DATA visualization driver.

Pulls real KTC training samples through TrainingDataPlugin + method plugins,
then uses the framework's own visualization functions to produce panels,
charts, and overlays.  All output lands in a timestamped run folder under
outputs/ so every run is preserved.

Run:
    python example_usage.py

Requires:  Codes_Matlab/ with TrainingData/ and GroundTruths/ subdirectories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# -- Trigger registration of all method plugins via __init__.py
import src.ktc_framework.methods   # noqa: F401

from src.ktc_framework.loaders.training_data_plugin import TrainingDataPlugin
from src.ktc_framework.registry import get_method
from src.ktc_framework.metrics.ktc_score import compute_ktc_score
from src.ktc_framework.metrics.composite_score import composite_score, letter_grade
from src.ktc_framework.visualization.plot_results import (
    plot_panel,
    plot_comparison_panel,
    plot_error_overlay,
    plot_degradation_curve,
    plot_leaderboard,
    plot_confusion_matrix,
    plot_electrodes,
    save_method_panel,
)


# =========================================================
# CONFIG
# =========================================================

METHODS = {
    "MockMethodPlugin":    "mock_baseline",
    "BackProjection":      "back_projection",
    "GaussNewton":         "gauss_newton",
}

TRAINING_DATA_ROOT = "Codes_Matlab"
TRAINING_SAMPLES   = ["1", "2", "3", "4"]

RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUTS_DIR = Path("outputs") / f"run_{RUN_ID}"
REPORTS_DIR = Path("reports") / f"run_{RUN_ID}"

COMPARISON_DIR      = OUTPUTS_DIR / "comparison_panels"
ERROR_OVERLAY_DIR   = OUTPUTS_DIR / "error_overlays"
CHARTS_DIR          = OUTPUTS_DIR / "charts"
RECONSTRUCTIONS_DIR = OUTPUTS_DIR / "reconstructions"
VISUALIZATION_DIR   = OUTPUTS_DIR / "visualization"

for d in [OUTPUTS_DIR, REPORTS_DIR, COMPARISON_DIR, ERROR_OVERLAY_DIR,
          CHARTS_DIR, RECONSTRUCTIONS_DIR, VISUALIZATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Write a pointer so app.py always knows which run is latest
(Path("outputs") / "latest.txt").write_text(str(OUTPUTS_DIR))

print(f"Run ID: {RUN_ID}")
print(f"Output: {OUTPUTS_DIR.resolve()}")
print("=" * 60)


# =========================================================
# [1/7] Load real KTC training samples
# =========================================================

print("\n[1/7] Loading real KTC training samples from Codes_Matlab/...")

loader = TrainingDataPlugin(TRAINING_DATA_ROOT)
batches: dict[str, object] = {}

for sid in TRAINING_SAMPLES:
    batch = loader.load_sample(level=1, sample=sid)
    batches[sid] = batch
    g = batch.ground_truth
    print(
        f"   sample {sid}: voltages={batch.voltages.shape}  "
        f"GT labels={dict(zip(*np.unique(g, return_counts=True)))}"
    )
print(f"   {len(batches)} real samples loaded")


# =========================================================
# [2/7] Run reconstruction methods on every sample
# =========================================================

print("\n[2/7] Running reconstruction methods...")

# predictions[method_key][sample_id] = (256,256) uint8
predictions:    dict[str, dict[str, np.ndarray]] = {}
ktc_scores:     dict[str, dict[str, float]]      = {}

for method_name, method_key in METHODS.items():
    try:
        method_cls = get_method(method_name)
    except KeyError:
        print(f"   {method_name} not registered — skipping")
        continue

    method = method_cls()
    predictions[method_key]  = {}
    ktc_scores[method_key]   = {}

    for sid, batch in batches.items():
        pred = np.asarray(method.reconstruct(batch), dtype=np.uint8)
        gt   = np.asarray(batch.ground_truth,         dtype=np.uint8)
        predictions[method_key][sid] = pred
        ktc_scores[method_key][sid]  = compute_ktc_score(pred, gt)
        print(f"   {method_name:>22} on sample {sid}: "
              f"KTC={ktc_scores[method_key][sid]:.3f}")

print(f"   {len(predictions)} methods × {len(TRAINING_SAMPLES)} samples done")


# =========================================================
# [3/7] Multi-method comparison panels
# =========================================================

print("\n[3/7] Comparison panels...")

for sid, batch in batches.items():
    gt = np.asarray(batch.ground_truth, dtype=np.uint8)
    methods_dict = {
        key.replace("_", " ").title(): predictions[key][sid]
        for key in predictions
    }
    plot_comparison_panel(
        gt=gt,
        methods_dict=methods_dict,
        save_path=COMPARISON_DIR / f"sample_{sid}.png",
    )
print(f"   Saved to {COMPARISON_DIR}")


# =========================================================
# [4/7] Error overlays
# =========================================================

print("\n[4/7] Error overlays...")

for sid, batch in batches.items():
    gt = np.asarray(batch.ground_truth, dtype=np.uint8)
    for method_key in predictions:
        plot_error_overlay(
            pred=predictions[method_key][sid],
            gt=gt,
            save_path=ERROR_OVERLAY_DIR / f"{method_key}_sample_{sid}.png",
        )
print(f"   Saved to {ERROR_OVERLAY_DIR}")


# =========================================================
# [5/7] Method panels (runner-style structure)
# =========================================================

print("\n[5/7] Method panels...")

for sid in TRAINING_SAMPLES:
    gt = np.asarray(batches[sid].ground_truth, dtype=np.uint8)
    for method_key in predictions:
        save_method_panel(
            pred=predictions[method_key][sid],
            gt=gt,
            level=1,
            sample=sid,
            method=method_key,
            output_dir=RECONSTRUCTIONS_DIR,
        )
print(f"   Saved to {RECONSTRUCTIONS_DIR}")


# =========================================================
# [6/7] Degradation curve + leaderboard
# =========================================================

print("\n[6/7] Charts...")

# Build a results list that matches the framework contract so we can
# call the same plot_degradation_curve / plot_leaderboard as BatchRunner.
results = []
for method_key in predictions:
    for idx, sid in enumerate(TRAINING_SAMPLES, start=1):
        score = ktc_scores[method_key][sid]
        comp  = composite_score({"ktc_score": score})
        results.append({
            "method":          method_key,
            "level":           idx,           # treat each sample as a "level"
            "sample":          sid,
            "metrics":         {"ktc_score": score},
            "composite_score": comp,
            "grade":           letter_grade(comp),
        })

plot_degradation_curve(results, CHARTS_DIR)
plot_leaderboard(results, CHARTS_DIR)
print(f"   Saved to {CHARTS_DIR}")


# =========================================================
# [7/7] Confusion matrix + electrode layout
# =========================================================

print("\n[7/7] Confusion matrix + electrode layout...")

# Pool all back_projection predictions for the confusion matrix
if "back_projection" in predictions:
    bp_preds = np.concatenate([predictions["back_projection"][s].ravel() for s in TRAINING_SAMPLES])
    bp_gts   = np.concatenate([np.asarray(batches[s].ground_truth).ravel() for s in TRAINING_SAMPLES])
    plot_confusion_matrix(pred=bp_preds.reshape(-1, 1), gt=bp_gts.reshape(-1, 1),
                          save_path=CHARTS_DIR / "confusion_matrix.png")

plot_electrodes(save_path=VISUALIZATION_DIR / "electrodes.png")
plot_panel(
    pred=predictions.get("back_projection", {}).get("1", np.zeros((256, 256), dtype=np.uint8)),
    gt=np.asarray(batches["1"].ground_truth, dtype=np.uint8),
    save_path=VISUALIZATION_DIR / "panel_sample_1.png",
)
print(f"   Saved to {CHARTS_DIR} + {VISUALIZATION_DIR}")


# =========================================================
# Save scores
# =========================================================

summary = {
    method_key: {
        "samples": {sid: ktc_scores[method_key][sid] for sid in TRAINING_SAMPLES},
        "mean_ktc": float(np.mean(list(ktc_scores[method_key].values()))),
    }
    for method_key in predictions
}
with (OUTPUTS_DIR / "scores.json").open("w") as f:
    json.dump(summary, f, indent=2)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("DONE — all outputs from real training data")
print("=" * 60)
print(f"\nComparison panels : {COMPARISON_DIR}")
print(f"Error overlays    : {ERROR_OVERLAY_DIR}")
print(f"Charts            : {CHARTS_DIR}")
print(f"Method panels     : {RECONSTRUCTIONS_DIR}")
print(f"Scores            : {OUTPUTS_DIR / 'scores.json'}")
print("\nLaunch dashboard  : streamlit run app.py")
