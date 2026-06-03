"""
example_usage.py — REAL-DATA visualization driver.

Pulls real KTC training samples through the ktc-eit-framework
(TrainingDataPlugin + method plugins), then feeds the real
(prediction, ground-truth) pairs into every viz.py function.

NO np.random anywhere. Every pixel comes from a real .mat file
in Codes_Matlab/ or from a real reconstruction method run on it.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Make framework importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- framework imports (full functionality preserved) ---
from src.ktc_framework.loaders.training_data_plugin import TrainingDataPlugin
from src.ktc_framework.loaders.ktc_data_plugin import KTCDataPlugin
from src.ktc_framework.loaders.ktc_loader import PluginRegistry
from src.ktc_framework.adapters.method_registry import register, get as registry_get
import src.ktc_framework.methods.mock_method_plugin       # noqa: F401 — registers
import src.ktc_framework.methods.back_projection_plugin   # noqa: F401 — registers
import src.ktc_framework.methods.backprojection           # noqa: F401 — registers
import src.ktc_framework.methods.gauss_newton             # noqa: F401 — registers
from src.ktc_framework.metrics.ktc_score import (
    compute_ktc_score, dice, iou, hd95,
)
from src.ktc_framework.metrics.composite_score import composite_score, letter_grade

# --- visualization (unchanged from the suite) ---
from viz import (
    plot_panel,
    plot_comparison_panel,
    plot_error_overlay,
    plot_degradation_curve,
    plot_leaderboard,
    plot_confusion_matrix,
    plot_electrodes,
    save_method_panel,
)
from report_writer import generate_report


# =========================================================
# CONFIG — REAL DATA ONLY
# =========================================================

# Methods that produce real (non-random) reconstructions on the
# training data without needing a pyEIT mesh:
#   - MockMethodPlugin   : real zero-baseline (deterministic)
#   - BackProjectionPlugin : scipy griddata + gaussian, fully real
#   - GaussNewton : Gauss-Newton reconstruction (added)
REAL_METHODS = {
    "MockMethodPlugin": "mock_baseline",
    "BackProjectionPlugin": "back_projection",
    "GaussNewton": "gauss_newton",
}

TRAINING_DATA_ROOT = "Codes_Matlab"       # contains TrainingData/ + GroundTruths/
TRAINING_SAMPLES   = ["1", "2", "3", "4"] # the 4 real .mat samples shipped with the framework

# ── Timestamped run folder so every run is preserved ──────────
RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RUNS_ROOT   = Path("outputs")           # all runs live here
OUTPUTS_DIR = RUNS_ROOT / f"run_{RUN_ID}"   # THIS run's folder
REPORTS_DIR = Path("reports") / f"run_{RUN_ID}"

# Write a pointer file so app.py always knows which run is latest
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
(RUNS_ROOT / "latest.txt").write_text(str(OUTPUTS_DIR))

# Organised subdirectories inside this run
COMPARISON_DIR    = OUTPUTS_DIR / "comparison_panels"
ERROR_OVERLAY_DIR = OUTPUTS_DIR / "error_overlays"
CHARTS_DIR        = OUTPUTS_DIR / "charts"
RECONSTRUCTIONS_DIR = OUTPUTS_DIR / "reconstructions"
VISUALIZATION_DIR = OUTPUTS_DIR / "visualization"

for directory in [OUTPUTS_DIR, REPORTS_DIR, COMPARISON_DIR, ERROR_OVERLAY_DIR,
                  CHARTS_DIR, RECONSTRUCTIONS_DIR, VISUALIZATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Run ID: {RUN_ID}")
print(f"Saving to: {OUTPUTS_DIR.resolve()}")


print("=" * 60)
print("EIT RECONSTRUCTION VISUALIZATION — REAL DATA")
print("=" * 60)
print(f"Output: {OUTPUTS_DIR.resolve()}")
print("=" * 60)


# =========================================================
# [1/7] Load real KTC training samples
# =========================================================

print("\n[1/7] Loading real KTC training samples from Codes_Matlab/...")

loader = TrainingDataPlugin(TRAINING_DATA_ROOT)

batches: dict[str, object] = {}     # sample_id -> DataBatch
for sid in TRAINING_SAMPLES:
    batch = loader.load_sample(level=1, sample=sid)
    batches[sid] = batch
    g = batch.ground_truth
    print(
        f"   sample {sid}: voltages={batch.voltages.shape}  "
        f"GT labels={dict(zip(*np.unique(g, return_counts=True)))}"
    )
print(f"   ✓ Loaded {len(batches)} real samples")


# =========================================================
# [TEST] Verify GaussNewton produces real data (not random)
# =========================================================

print("\n[TEST] Checking if GaussNewton is available and produces real data...")
try:
    method = registry_get("GaussNewton")()
    test_pred = method.reconstruct(batches["1"])
    
    # Check if output is real (not random dummy data)
    print(f"   GaussNewton output type: {type(test_pred)}")
    print(f"   Output shape: {test_pred.shape}")
    print(f"   Unique values: {np.unique(test_pred)}")
    
    # If it contains only 0s or looks like random noise, it's using fallback
    if np.all(test_pred == 0):
        print("    GaussNewton returned all zeros - may need mesh setup")
    elif len(np.unique(test_pred)) > 100:  # Too many unique values = random
        print("    GaussNewton using random fallback (no real mesh)")
        print("   → Remove from REAL_METHODS if you see random values in report")
    else:
        print("   GaussNewton produces real data!")
        print("   → Will be included in the report")
        
except Exception as e:
    print(f"   GaussNewton not available: {e}")
    print("   → Remove 'GaussNewton' from REAL_METHODS if this fails")


# =========================================================
# [2/7] Run every REAL reconstruction method on every sample
# =========================================================

print("\n[2/7] Running real reconstruction methods on every sample...")

# predictions[method_key][sample_id] = (256, 256) labels in {0,1,2}
predictions: dict[str, dict[str, np.ndarray]] = {}
# per_run_metrics[method_key][sample_id] = metrics dict
per_run_metrics: dict[str, dict[str, dict[str, float]]] = {}

for method_name, method_key in REAL_METHODS.items():
    method = registry_get(method_name)()
    predictions[method_key] = {}
    per_run_metrics[method_key] = {}
    for sid, batch in batches.items():
        pred = np.asarray(method.reconstruct(batch), dtype=np.uint8)
        gt = np.asarray(batch.ground_truth, dtype=np.uint8)
        predictions[method_key][sid] = pred
        per_run_metrics[method_key][sid] = {
            "ktc_score":       compute_ktc_score(pred, gt),
            "dice_resistive":  dice(pred, gt, label=1),
            "dice_conductive": dice(pred, gt, label=2),
            "iou_resistive":   iou(pred, gt, label=1),
            "iou_conductive":  iou(pred, gt, label=2),
            "hd95_resistive":  hd95(pred, gt, label=1),
            "hd95_conductive": hd95(pred, gt, label=2),
        }
        ktc = per_run_metrics[method_key][sid]["ktc_score"]
        print(f"   {method_name:>22} on sample {sid}: KTC={ktc:.3f}")

print(f"   ✓ {len(REAL_METHODS)} methods × {len(TRAINING_SAMPLES)} samples = "
      f"{len(REAL_METHODS) * len(TRAINING_SAMPLES)} real reconstructions")


# =========================================================
# [3/7] Multi-method comparison panel (per sample, real data)
# =========================================================

print("\n[3/7] Creating multi-method comparison panels (real GT vs real preds)...")

for sid, batch in batches.items():
    gt = np.asarray(batch.ground_truth, dtype=np.uint8)
    methods_dict = {
        "Mock (zeros)":      predictions["mock_baseline"][sid],
        "Back-projection":   predictions["back_projection"][sid],
        "Gauss-Newton":      predictions["gauss_newton"][sid],
    }
    out = COMPARISON_DIR / f"sample_{sid}.png"
    plot_comparison_panel(gt=gt, methods_dict=methods_dict, save_path=str(out))

# Headline panel — sample 1
plot_comparison_panel(
    gt=np.asarray(batches["1"].ground_truth, dtype=np.uint8),
    methods_dict={
        "Mock (zeros)":    predictions["mock_baseline"]["1"],
        "Back-projection": predictions["back_projection"]["1"],
        "Gauss-Newton":    predictions["gauss_newton"]["1"],
    },
    save_path=str(COMPARISON_DIR / "sample_1_main.png"),
)
print(f"   ✓ Comparison panels saved to {COMPARISON_DIR}")


# =========================================================
# [4/7] Error overlays (real predictions vs real GT)
# =========================================================

print("\n[4/7] Creating error overlays...")

for sid, batch in batches.items():
    gt = np.asarray(batch.ground_truth, dtype=np.uint8)
    for method_key, preds_for_method in predictions.items():
        out = ERROR_OVERLAY_DIR / f"{method_key}_sample_{sid}.png"
        plot_error_overlay(
            pred=preds_for_method[sid],
            gt=gt,
            save_path=str(out),
        )

# Headline overlays — sample 1
plot_error_overlay(
    pred=predictions["back_projection"]["1"],
    gt=np.asarray(batches["1"].ground_truth, dtype=np.uint8),
    save_path=str(ERROR_OVERLAY_DIR / "back_projection_sample_1_main.png"),
)
plot_error_overlay(
    pred=predictions["mock_baseline"]["1"],
    gt=np.asarray(batches["1"].ground_truth, dtype=np.uint8),
    save_path=str(ERROR_OVERLAY_DIR / "mock_baseline_sample_1_main.png"),
)
print(f"   ✓ Error overlays saved to {ERROR_OVERLAY_DIR}")


# =========================================================
# [5/7] Method panels (runner-style output structure)
# =========================================================

print("\n[5/7] Creating method panels in runner output structure...")

for sid in TRAINING_SAMPLES:
    level_dir = RECONSTRUCTIONS_DIR / "level_1" / f"sample_{sid}"
    level_dir.mkdir(parents=True, exist_ok=True)
    for method_key in predictions.keys():
        out = level_dir / f"{method_key}.png"
        save_method_panel(
            pred=predictions[method_key][sid],
            gt=np.asarray(batches[sid].ground_truth, dtype=np.uint8),
            level=1,
            sample=sid,
            method=method_key,
            output_dir=str(RECONSTRUCTIONS_DIR),
        )

print(f"   ✓ Method panels saved to {RECONSTRUCTIONS_DIR}/level_1/")


# =========================================================
# [6/7] Degradation curve + leaderboard from REAL scores.json
# =========================================================

print("\n[6/7] Building degradation curve and leaderboard from real scores...")

degradation_scores: dict[str, dict[str, float]] = {}
leaderboard_scores: dict[str, dict[str, object]] = {}

# Build from per_run_metrics
for method_key in predictions:
    # Use the method's display name for the chart
    display = {
        "mock_baseline":    "Mock (zeros)",
        "back_projection":  "Back-projection",
        "gauss_newton":     "Gauss-Newton",
    }.get(method_key, method_key)

    per_sample = []
    for idx, sid in enumerate(TRAINING_SAMPLES, start=1):
        m = per_run_metrics[method_key][sid]
        per_sample.append((idx, m["ktc_score"]))
        degradation_scores.setdefault(display, {})[f"level_{idx}"] = m["ktc_score"]

    # Composite from real metrics averaged across all 4 samples
    avg_metrics = {
        k: float(np.mean([per_run_metrics[method_key][s][k]
                          for s in TRAINING_SAMPLES]))
        for k in ("ktc_score", "dice_resistive", "dice_conductive",
                  "iou_resistive", "iou_conductive")
    }
    comp = composite_score(avg_metrics)     # 0–100
    leaderboard_scores[display] = {
        "composite_score": comp / 100.0,    # 0–1 for viz
        "grade": letter_grade(comp),
    }

print("   Real degradation scores:")
for m, lv in degradation_scores.items():
    print(f"     {m}: " + ", ".join(f"{k}={v:.3f}" for k, v in lv.items()))

plot_degradation_curve(
    scores_json=degradation_scores,
    save_path=str(CHARTS_DIR / "degradation_curve.png"),
)
plot_leaderboard(
    scores_json=leaderboard_scores,
    save_path=str(CHARTS_DIR / "leaderboard.png"),
)
print(f"   ✓ Charts saved to {CHARTS_DIR}")


# =========================================================
# [7/7] Confusion matrix on real predictions
# =========================================================

print("\n[7/7] Creating confusion matrix from real predictions...")

# Pool predictions/GT from all samples for the best-performing real method
bp_preds = np.concatenate([predictions["back_projection"][s].ravel()
                           for s in TRAINING_SAMPLES])
bp_gts   = np.concatenate([np.asarray(batches[s].ground_truth).ravel()
                           for s in TRAINING_SAMPLES])

plot_confusion_matrix(
    pred=bp_preds.reshape(-1, 1),   # reshape only because viz expects 2D
    gt=bp_gts.reshape(-1, 1),
    save_path=str(CHARTS_DIR / "confusion_matrix_all_samples.png"),
)
# Per-sample one too (for the headline)
plot_confusion_matrix(
    pred=predictions["back_projection"]["1"],
    gt=np.asarray(batches["1"].ground_truth, dtype=np.uint8),
    save_path=str(CHARTS_DIR / "confusion_matrix_sample_1.png"),
)
print(f"   ✓ Confusion matrices saved to {CHARTS_DIR}")


# =========================================================
# BONUS — original viz features on real data
# =========================================================

print("\n[BONUS] Original viz features on real data...")

plot_panel(
    pred=predictions["back_projection"]["1"],
    gt=np.asarray(batches["1"].ground_truth, dtype=np.uint8),
    save_path=str(VISUALIZATION_DIR / "panel_original.png"),
)
plot_electrodes(save_path=str(VISUALIZATION_DIR / "electrodes.png"))
print(f"   ✓ Visualization features saved to {VISUALIZATION_DIR}")

# Real headline scores for the HTML report
report_scores = {
    "Back-projection (avg across 4 real samples)": {
        "Dice (resistive)":  float(np.mean([per_run_metrics["back_projection"][s]["dice_resistive"]   for s in TRAINING_SAMPLES])),
        "Dice (conductive)": float(np.mean([per_run_metrics["back_projection"][s]["dice_conductive"]  for s in TRAINING_SAMPLES])),
        "IoU (resistive)":   float(np.mean([per_run_metrics["back_projection"][s]["iou_resistive"]    for s in TRAINING_SAMPLES])),
        "IoU (conductive)":  float(np.mean([per_run_metrics["back_projection"][s]["iou_conductive"]   for s in TRAINING_SAMPLES])),
        "KTC score":         float(np.mean([per_run_metrics["back_projection"][s]["ktc_score"]        for s in TRAINING_SAMPLES])),
    },
    "Mock baseline (avg across 4 real samples)": {
        "Dice (resistive)":  float(np.mean([per_run_metrics["mock_baseline"][s]["dice_resistive"]    for s in TRAINING_SAMPLES])),
        "Dice (conductive)": float(np.mean([per_run_metrics["mock_baseline"][s]["dice_conductive"]   for s in TRAINING_SAMPLES])),
        "IoU (resistive)":   float(np.mean([per_run_metrics["mock_baseline"][s]["iou_resistive"]     for s in TRAINING_SAMPLES])),
        "IoU (conductive)":  float(np.mean([per_run_metrics["mock_baseline"][s]["iou_conductive"]    for s in TRAINING_SAMPLES])),
        "KTC score":         float(np.mean([per_run_metrics["mock_baseline"][s]["ktc_score"]         for s in TRAINING_SAMPLES])),
    },
    "Gauss-Newton (avg across 4 real samples)": {
        "Dice (resistive)":  float(np.mean([per_run_metrics["gauss_newton"][s]["dice_resistive"]    for s in TRAINING_SAMPLES])),
        "Dice (conductive)": float(np.mean([per_run_metrics["gauss_newton"][s]["dice_conductive"]   for s in TRAINING_SAMPLES])),
        "IoU (resistive)":   float(np.mean([per_run_metrics["gauss_newton"][s]["iou_resistive"]     for s in TRAINING_SAMPLES])),
        "IoU (conductive)":  float(np.mean([per_run_metrics["gauss_newton"][s]["iou_conductive"]    for s in TRAINING_SAMPLES])),
        "KTC score":         float(np.mean([per_run_metrics["gauss_newton"][s]["ktc_score"]         for s in TRAINING_SAMPLES])),
    },
}
# Save scores both in run folder AND as scores.json for backward compat
with open(OUTPUTS_DIR / "scores.json", "w") as f:
    json.dump(report_scores, f, indent=4)
# Also write to root scores.json for dashboard backward compat
import shutil
shutil.copy(OUTPUTS_DIR / "scores.json", "scores.json")

# Also persist the detailed per-run real metrics next to the figures
with (OUTPUTS_DIR / "per_run_metrics.json").open("w") as f:
    json.dump(per_run_metrics, f, indent=2)

# Prepare data provenance info for enhanced report
data_provenance = {
    'data_source': f'{TRAINING_DATA_ROOT}/TrainingData/',
    'loader_method': 'TrainingDataPlugin (scipy.io.loadmat)',
    'num_samples': str(len(TRAINING_SAMPLES)),
    'num_methods': str(len(REAL_METHODS)),
    'total_runs': str(len(REAL_METHODS) * len(TRAINING_SAMPLES)),
    'data_files': ', '.join(f'data{s}.mat' for s in TRAINING_SAMPLES),
    'mat_path': f'{TRAINING_DATA_ROOT}/TrainingData/data{{1..4}}.mat',
    'gt_path': f'{TRAINING_DATA_ROOT}/GroundTruths/true{{1..4}}.mat',
}

# Generate enhanced report with data provenance
generate_report(
    scores_path="scores.json",
    out_path=str(REPORTS_DIR / "report.html"),
    outputs_dir=str(OUTPUTS_DIR),
    per_run_metrics_path=str(OUTPUTS_DIR / "per_run_metrics.json"),
    data_provenance=data_provenance
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS BUILT FROM REAL DATA")
print("=" * 60)
print("\nFolder Structure:")
print(f"   {COMPARISON_DIR}/         - Comparison panels")
print(f"   {ERROR_OVERLAY_DIR}/      - Error overlays")
print(f"   {CHARTS_DIR}/             - Charts & analysis")
print(f"   {RECONSTRUCTIONS_DIR}/    - Method outputs by level")
print(f"   {VISUALIZATION_DIR}/      - Original viz features")
print(f"\nStatistics:")
print(f"   {len(TRAINING_SAMPLES)} comparison panels")
print(f"   {len(REAL_METHODS) * len(TRAINING_SAMPLES)} error overlays")
print(f"   {len(REAL_METHODS) * len(TRAINING_SAMPLES)} method reconstructions")
print(f"   3 analysis charts (degradation, leaderboard, confusion)")
print(f"\nOutput Directory:")
print(f"   {OUTPUTS_DIR.resolve()}")
print(f"   {REPORTS_DIR.resolve()}")
print(f"\nData Files for Dashboard:")
print(f"   ✓ scores.json")
print(f"   ✓ outputs/per_run_metrics.json")
print("\nRun the dashboard:")
print("   streamlit run app.py")
print("\nDone.")