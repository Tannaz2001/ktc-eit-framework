# EIT Reconstruction — Framework + Visualization (Real Data)

This folder is the full **`ktc-eit-framework`** (loaders, runner, method
adapters, metrics, reporting) wired up to the **`viz.py`** visualization
suite. Every figure is produced from **real KTC training data** loaded
through the framework — no `np.random` reconstructions anywhere.

---

## Folder layout

```
eit_visualization/
├── src/ktc_framework/        ← full framework, untouched
│   ├── adapters/             ← method_registry (register / get)
│   ├── loaders/              ← KTCDataPlugin, TrainingDataPlugin, MockDataPlugin
│   ├── methods/              ← MockMethodPlugin, BackProjectionPlugin, …
│   ├── metrics/              ← ktc_score (SSIM port), dice, iou, hd95, composite
│   ├── runner/               ← BatchRunner, config_validator
│   └── reporting/            ← data_layer (DataFrame helpers for dashboards)
├── Codes_Matlab/             ← REAL data shipped with the framework
│   ├── TrainingData/         ← data1.mat … data4.mat (real KTC voltage measurements)
│   └── GroundTruths/         ← true1.mat … true4.mat (real 256×256 segmentation masks)
├── data/KTCScoring/          ← original KTC scoring scripts
├── configs/
│   ├── training_experiment.yaml   ← uses TrainingDataPlugin + real methods
│   ├── ktc_all_methods.yaml       ← full evaluation config (needs KTC eval set)
│   ├── experiment.yaml
│   └── mock_experiment.yaml
├── viz.py                    ← visualization suite (unchanged)
├── example_usage.py          ← REAL-data driver — replaces every dummy with real loads
├── report_writer.py          ← minimal HTML report (gallery of every PNG + metrics)
├── run.py                    ← framework CLI (BatchRunner entrypoint)
├── requirements.txt
└── environment.yml
```

---

## Setup

```bash
pip install -r requirements.txt
# or:
pip install numpy scipy scikit-image matplotlib seaborn rich pyyaml pandas h5py
```

---

## Two ways to run

### 1) Visualization only (recommended quick start)

```bash
python example_usage.py
```

What it does, step by step — every number is from a real `.mat` file:

1. **Load** the 4 real KTC training samples via `TrainingDataPlugin`
2. **Reconstruct** each one with every real (non-random) method:
   - `MockMethodPlugin` — real deterministic zero baseline
   - `BackProjectionPlugin` — real scipy `griddata` + gaussian smoothing on the real voltages
3. **Score** every reconstruction via the framework's `compute_ktc_score`,
   `dice`, `iou`, `hd95`, and `composite_score`
4. **Visualize** with every function from `viz.py`:
   - `plot_comparison_panel` — GT | Mock | Back-projection, per sample
   - `plot_error_overlay`    — grey / red / orange pixel-level failure map
   - `save_method_panel`     — runner-style `outputs/level_X/sample_Y/method_Z.png`
   - `plot_degradation_curve` — KTC score across the 4 real samples
   - `plot_leaderboard`      — bar chart of real composite scores with letter grades
   - `plot_confusion_matrix` — 3×3 class confusion from real predictions vs real GT
   - `plot_panel` / `plot_electrodes` — original viz features on real data
5. **Write** `reports/report.html` — HTML gallery + metrics table

### 2) Framework BatchRunner first, then visualize

```bash
python run.py --config configs/training_experiment.yaml
python example_usage.py
```

When `outputs/scores.json` already exists, `example_usage.py` reads it
and builds the degradation curve and leaderboard from the runner's
output. Otherwise it computes metrics inline from its own real
reconstructions. Either way, every input is a real sample.

### 3) Full evaluation set (level 1–7 × A/B/C)

If you have the official KTC evaluation set on disk, point at it via
the env var and use the full config — the framework's
`KTCDataPlugin` and `BatchRunner` handle everything:

```bash
export KTC_DATASET_ROOT="/path/to/EvaluationData_full"
python run.py --config configs/ktc_all_methods.yaml
python example_usage.py
```

This produces a true 7-level degradation curve.

---

## Why only two methods?

The framework ships four reconstruction methods:

| Method                  | Status here                                                                 |
| ----------------------- | --------------------------------------------------------------------------- |
| `MockMethodPlugin`      | ✅ Real zero-baseline                                                       |
| `BackProjectionPlugin`  | ✅ Real scipy interpolation (no extra deps)                                  |
| `BackProjection`        | ⚠️ Needs **pyEIT + a mesh object** on the `DataBatch`; otherwise falls back to `np.random.rand` — excluded to keep this folder dummy-free |
| `GaussNewton`           | ⚠️ Same caveat as above — excluded                                          |

To re-enable the pyEIT-based methods, install `pyeit`, attach a real
`pyeit.mesh.PyEITMesh` to each `DataBatch`, then list them in the YAML
under `methods:` and rerun. The framework will pick them up
automatically — nothing else changes.

---

## Outputs

After running:

```
outputs/
├── scores.json                     ← from BatchRunner (real)
├── scores_nested.json              ← method → level → sample → metrics
├── per_run_metrics.json            ← per-sample metrics from example_usage.py
├── comparison_panel.png            ← GT vs methods (real)
├── comparison_panel_sample_{1..4}.png
├── error_overlay_*.png             ← per method per sample
├── degradation_curve.png
├── leaderboard.png
├── confusion_matrix.png
├── electrodes.png
├── panel_original.png
└── level_1/sample_{1..4}/{mock_baseline,back_projection}.png

reports/
└── report.html                     ← HTML gallery + real metrics table
```

---

## How "real data" is guaranteed

- `Codes_Matlab/TrainingData/data{1..4}.mat` is loaded by
  `TrainingDataPlugin._load_data` → `scipy.io.loadmat` → real KTC
  voltage measurements (shape `(2356,)`).
- `Codes_Matlab/GroundTruths/true{1..4}.mat` is loaded the same way →
  real segmentation labels with shape `(256, 256)` and values in `{0, 1, 2}`.
- `MockMethodPlugin.reconstruct` returns deterministic zeros — same
  output every run, no RNG.
- `BackProjectionPlugin.reconstruct` uses `scipy.interpolate.griddata`
  on the real voltage vector — same output every run, no RNG.
- `example_usage.py` contains zero calls to `np.random.*`.

Grep yourself:

```bash
grep -nE "np\.random|random\.randint|random\.rand" example_usage.py
# → no matches
```
