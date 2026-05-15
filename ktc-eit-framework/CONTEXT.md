# KTC EIT Framework — LLM Context Document

> **Purpose:** This file gives a new LLM complete context on the project, its dataset,
> every file that exists, what each one does, what is finished, what is not, and the
> known constraints that must be respected when continuing work.

---

## 1. What This Project Is

A **modular benchmarking pipeline** for the
[Kuopio Tomography Challenge 2023 (KTC 2023)](https://zenodo.org/record/10986692)
Electrical Impedance Tomography (EIT) dataset.

Given voltage measurements from an EIT tank, the framework:
1. Loads and validates `.mat` files through a unified `DataBatch` interface
2. Runs any registered reconstruction method through a `MethodPlugin` API
3. Scores results using per-class Dice, IoU, and the official KTC SSIM score
4. Exports `scores.json`, `scores_nested.json`, and an HTML leaderboard report

**Design principle:** the core pipeline never changes. Every loader, method, and
metric is a plugin — adding one requires one file and one decorator. No other files
change.

---

## 2. Repository Layout

```
ktc-eit-framework/
├── src/ktc_framework/
│   ├── types.py                  ← DataBatch NamedTuple (the data contract)
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── ktc_loader.py         ← KTCLoader + KTCValidator + PluginRegistry
│   │   ├── mock_data_plugin.py   ← MockDataPlugin (synthetic data for testing)
│   │   └── file_validator.py     ← file existence/type/size checks
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── level_set_plugin.py   ← LevelSetPlugin (Otsu + contour extraction)
│   │   ├── hull_plugin.py        ← HullPlugin (connected-component geometry)
│   │   └── mock_method_plugin.py ← MockMethodPlugin (returns all-zeros, for testing)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── ktc_score.py          ← dice(), iou(), compute_all_metrics()
│   │   ├── composite_score.py    ← composite_score(), letter_grade()
│   │   └── metric_registry.py    ← callable metric registry
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── experiment_runner.py  ← BatchRunner (YAML-driven pipeline loop)
│   │   └── config_validator.py   ← load_config() with full YAML validation
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── method_registry.py    ← @register decorator + get() + list_methods()
│   ├── reporting/
│   │   └── __init__.py           ← generate_report() → HTML leaderboard
│   └── visualization/
│       └── __init__.py           ← plot_sample(), plot_overlay(), save_panel()
├── tests/
│   └── test_plugins.py           ← 18 synthetic tests for LevelSet + Hull plugins
├── run.py                        ← CLI entry point (--config argument)
├── environment.yml               ← pinned conda env (ktc-eit, Python 3.11)
└── CONTEXT.md                    ← this file
```

---

## 3. The Dataset (KTC 2023)

Located in `Codes_Matlab/` (MATLAB source, not in the framework folder).

### 3.1 Files

| Path | Contents |
|------|----------|
| `TrainingData/data{1-4}.mat` | Voltage measurements per sample |
| `TrainingData/ref.mat` | Reference measurement (homogeneous tank, no inclusions) |
| `GroundTruths/true{1-4}.mat` | 256×256 segmentation ground truth |
| `Output/{1-4}.mat` | Pre-computed MATLAB reconstructions |
| `Mesh_dense.mat` | Fine FEM mesh: 7337 elements, 3766 nodes (linear), 14868 nodes (quadratic) |
| `Mesh_sparse.mat` | Coarse FEM mesh: 3073 elements, 1602 nodes (linear), 6276 nodes (quadratic) |

### 3.2 Array Keys and Shapes

**`data{i}.mat`** (per-sample measurement file):

| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| `Inj` | (32, 76) | float64 | Current injection patterns — **identical across all 4 samples** |
| `Mpat` | (32, 31) | int16 | Voltage measurement protocol — **identical across all 4 samples** |
| `Uel` | **(2356,)** | float64 | Voltage measurements — this is the signal, varies per sample |

**`ref.mat`** (homogeneous baseline):

| Key | Shape | Notes |
|-----|-------|-------|
| `Injref` | (32, 76) | Same as `Inj` in data files |
| `Mpat` | (32, 31) | Same as `Mpat` in data files |
| `Uelref` | (2356,) | Reference voltages for a tank with no inclusions |

> **Why 2356?** 76 injection patterns × 31 differential voltage pairs = 2356.
> The 32 rows in `Inj`/`Mpat` are the 32 physical electrodes.

**`true{i}.mat`** (ground truth):

| Key | Shape | dtype | Labels |
|-----|-------|-------|--------|
| `truth` | (256, 256) | uint8 | 0 = Background, 1 = Resistive, 2 = Conductive |

### 3.3 Label Distribution (all 4 training samples)

| Sample | Background (0) | Resistive (1) | Conductive (2) | Classes present |
|--------|---------------|---------------|----------------|-----------------|
| 1 | 89.18% | 6.40% | 4.42% | Both |
| 2 | 90.68% | 6.32% | 3.00% | Both |
| 3 | 92.76% | 0% | 7.24% | Conductive only |
| 4 | 96.68% | 3.32% | 0% | Resistive only |

**Severe class imbalance** — background dominates at 89–97%. Per-class metrics
(Dice, IoU) are mandatory; pixel accuracy is misleading.

### 3.4 Reconstruction Signal

The actual perturbation used by reconstruction algorithms is the **difference voltage**:
```
delta_Uel = Uel - Uelref   # range roughly ±0.25 V
```
Raw `Uel` is not the signal on its own.

### 3.5 Mesh Geometry

Both meshes cover a **circular 23 cm-diameter tank domain** (node coordinates
span −0.115 m to +0.115 m). Each `.mat` contains two sub-meshes:
- `Mesh` — linear triangles (H: N_elements × 3)
- `Mesh2` — quadratic triangles (H: N_elements × 6, more nodes)

---

## 4. The Data Contract — `DataBatch`

```python
# src/ktc_framework/types.py
class DataBatch(NamedTuple):
    voltages: np.ndarray          # shape (2356,)   — flat voltage vector
    injection_patterns: np.ndarray # shape (32, 76) — current injection matrix
    ground_truth: np.ndarray      # shape (256, 256) — integer label map
    level: int                    # difficulty level 1–7
    sample_id: str                # e.g. "data1", "mock-0001"
```

**This is frozen.** Every loader must produce a `DataBatch`. Every method receives one.
Changing it requires a team discussion.

---

## 5. Plugin Architecture

### 5.1 Method Registry (`adapters/method_registry.py`)

```python
from src.ktc_framework.adapters.method_registry import register

@register
class MyMethod:
    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        ...  # must return (256, 256) int array, labels in {0, 1, 2}
```

`register` uses `cls.__name__` as the key. `get("MyMethod")` looks it up.

### 5.2 Loader Registry (`loaders/ktc_loader.py` — `PluginRegistry`)

```python
@PluginRegistry.register('ktc_loader')
class KTCLoader:
    ...
```

A separate registry for data plugins, keyed by string name.

### 5.3 Metric Registry (`metrics/metric_registry.py`)

```python
register_metric("dice_resistive", lambda pred, gt: dice(pred, gt, label=1))
run_all_metrics(pred, gt)  # runs all registered metrics
```

---

## 6. File-by-File Reference

### `loaders/ktc_loader.py`

Three classes in one file:

**`PluginRegistry`**
- `_registry: dict[str, type]` class variable
- `@classmethod register(name)` — decorator, returns class unchanged
- `@classmethod get(name)` — raises `KeyError` listing available names if not found

**`KTCValidator`** (static class)
- `VOLTAGE_SHAPE = (2356,)` — flat 1D vector (76 injections × 31 pairs)
- `GT_SHAPE = (256, 256)`
- `VALID_LABELS = {0, 1, 2}`
- `validate(batch)` — checks voltage shape, GT spatial shape, label values; raises
  `ValueError` with a descriptive message on any failure

**`KTCLoader`** (registered as `'ktc_loader'`)
- `load(filename)` — scipy v5 first, h5py fallback for v7.3; calls `KTCValidator.validate()`
- `list_samples()` — scans `data_dir` for `level{level}_*.mat`
- `_parse_mat_v5(mat, filename)` — reads `Uel`, `Inj`, `truth` keys
- `_parse_mat_v73(filepath, filename)` — same via h5py with transpose for C-order

> **Critical:** data files use key `Inj`. Only `ref.mat` uses `Injref`. The loader
> correctly reads `Inj`.

---

### `loaders/mock_data_plugin.py` — `MockDataPlugin`

Generates synthetic `DataBatch` objects. Used by `BatchRunner` when no real dataset
is present.

| Method | Description |
|--------|-------------|
| `get_batch(n_samples, level, sample_id)` | Returns one `DataBatch` with random data |
| `iter_batches(n_batches, n_samples, level)` | Generator yielding N batches |
| `reset(seed)` | Re-seeds the internal RNG |

Injection patterns use adjacent-pair scheme (source +1, sink −1).

---

### `loaders/file_validator.py`

`ValidationResult` dataclass + `validate_sample_file()` — checks file existence,
`.mat` extension, and non-zero file size. Used as a pre-flight guard before loading.

---

### `methods/level_set_plugin.py` — `LevelSetPlugin`

```python
result = LevelSetPlugin().run(reconstruction_256x256)
# {'contours': [...], 'n_objects': int}
```

Steps: Otsu threshold → binary mask → `skimage.measure.find_contours`.
Handles uniform arrays gracefully (returns empty). Logs at INFO/DEBUG/ERROR.

---

### `methods/hull_plugin.py` — `HullPlugin`

```python
features = HullPlugin().run(segmentation_256x256, target_label=1)
# [{'centroid': ..., 'area': ..., 'bbox': ..., 'convex_area': ...}, ...]
```

Steps: binary mask for `target_label` → `skimage.measure.label` →
`skimage.measure.regionprops`. Returns one dict per connected component.
Raises `ValueError` for wrong shape or label not in `{0, 1, 2}`.

---

### `methods/mock_method_plugin.py` — `MockMethodPlugin`

Returns `np.zeros((256, 256), dtype=int)`. Used by `BatchRunner` for dry runs.

---

### `metrics/ktc_score.py`

| Function | Returns |
|----------|---------|
| `dice(pred, gt, label)` | Dice score for one class (float 0–1) |
| `iou(pred, gt, label)` | IoU for one class (float 0–1) |
| `compute_ktc_score(pred, gt)` | Calls `KTCScoring.scoringFunction` (external module) |
| `compute_all_metrics(pred, gt)` | Dict of all five metrics |

`compute_ktc_score` requires the external `KTCScoring` module from the KTC dataset.
It raises `ImportError` with a clear message if not found.

---

### `metrics/composite_score.py`

Weighted combination:

| Metric | Weight |
|--------|--------|
| `ktc_score` | 0.40 |
| `dice_resistive` | 0.20 |
| `dice_conductive` | 0.20 |
| `iou_resistive` | 0.10 |
| `iou_conductive` | 0.10 |

`composite_score(metrics)` → float 0–100.
`letter_grade(score)` → A (≥80) / B (≥60) / C (≥40) / D (<40).

---

### `runner/experiment_runner.py` — `BatchRunner`

Reads a YAML config dict and loops `methods × levels × samples`.

- Uses `MockDataPlugin` and `MockMethodPlugin` for now (real `KTCLoader` not yet
  wired into the runner loop — see Sprint 4 gaps below)
- Saves `outputs/scores.json` (flat list) and `outputs/scores_nested.json`
  (method → level → sample → metrics)
- Prints Rich tables: experiment summary + degradation slope per method
- Embeds Git SHA in every result row
- `_print_degradation()` uses `np.polyfit` to compute per-method score slope
  across difficulty levels

---

### `runner/config_validator.py`

`load_config(path)` validates:
- File exists and is `.yaml`/`.yml`
- All required fields present: `data_plugin`, `mesh_path`, `levels`, `samples`,
  `methods`, `dataset_root`, `output_dir`
- Levels are integers 1–7
- Samples are strings in `{'A', 'B', 'C'}`
- Methods are non-empty strings

Raises `ConfigError` (custom exception) on any violation.

---

### `adapters/method_registry.py`

Simple module-level dict registry:
- `@register` — stores class under `cls.__name__`
- `get(name)` — raises `KeyError` listing all registered names if not found
- `list_methods()` — returns sorted list of registered names

---

### `reporting/__init__.py`

`generate_report(scores_path, output_path=None) → Path`

Reads `scores.json` (flat list or nested dict), builds an HTML table, writes a
self-contained styled HTML file. Returns the resolved output path.

---

### `visualization/__init__.py`

| Function | Description |
|----------|-------------|
| `plot_sample(gt, pred, sample_id, show)` | 3-panel figure: GT / Prediction / Error map |
| `plot_overlay(gt, pred, sample_id, show)` | Single axis: prediction fill + GT contours |
| `save_panel(gt, pred, path, sample_id, dpi)` | Calls `plot_sample`, saves PNG, closes figure |

Colour convention: Background = blue, Resistive = red, Conductive = green.

---

### `tests/test_plugins.py`

18 pytest tests — no dataset required.

**`TestLevelSetPlugin`** (7 tests): circle fixture, uniform edge case, return type,
contour count, contour array shape, wrong-shape guard, wrong-type guard.

**`TestHullPlugin`** (9 tests): two-block fixture, feature keys, area positive,
centroid in bounds, absent label → empty list, background coverage, wrong-shape
guard, invalid-label guard.

Run with: `pytest tests/test_plugins.py -v`

---

### `run.py` (CLI)

```bash
python run.py --config path/to/experiment.yaml
```

Calls `load_config()` then `BatchRunner(config, output_dir).run()`. Exits with
code 1 on `ConfigError`.

---

## 7. Sprint Status

| Sprint | Status | Key deliverables |
|--------|--------|-----------------|
| 1 | ✅ Done | Repo setup, data dictionary, colour map |
| 2 | ✅ Done | `DataBatch`, abstract plugin base classes, `MockDataPlugin`, `MockMethodPlugin` |
| 3 | ✅ Done | `KTCLoader`, `KTCValidator`, `LevelSetPlugin`, `HullPlugin`, `file_validator`, method/metric/loader registries |
| 4 | ✅ Done | `BatchRunner`, `config_validator`, `run.py`, `composite_score`, `ktc_score`, `metric_registry` |
| 4 (gaps) | ⚠️ Partial | `BatchRunner` still uses `MockDataPlugin` — real `KTCLoader` not yet wired in; `KTCScoring` module not yet on PYTHONPATH |
| 5 | ⏳ Planned | `HD95`, per-class `MeanIoU`, `test_metrics.py` with known-answer validation |
| 6 | ⏳ Planned | Full 7-level × 3-sample benchmark run, `report.html`, git tag v0.1 |
| 7 | ⏳ Planned | `GaussNewtonUNet` learned post-processor |
| 8–9 | ⏳ Planned | Streamlit dashboard, weight editor, failure gallery |
| 10–12 | ⏳ Planned | Final report, demo, handover |

---

## 8. Known Constraints and Decisions

### Fixed shapes — do not change these without team agreement
- Voltage vector: `(2356,)` — flat, 76 injections × 31 voltage pairs
- Injection matrix: `(32, 76)` — 32 physical electrodes
- Ground truth: `(256, 256)` uint8
- All reconstruction outputs must be `(256, 256)` int, labels `{0, 1, 2}`

### Key naming — data files vs ref file
- `data{1-4}.mat` uses key `Inj` for injection patterns
- `ref.mat` uses key `Injref`
- The loader reads `Inj` (correct); do not revert to `Injref`

### Reconstruction input
- Always subtract reference: `delta_Uel = Uel - Uelref`
- Raw `Uel` alone is not meaningful for reconstruction

### Class imbalance
- Background is 89–97% of every image
- Never use pixel accuracy as a metric
- Use per-class Dice (label=1, label=2) and IoU (label=1, label=2)

### `KTCScoring` external dependency
- `compute_ktc_score()` in `ktc_score.py` calls `KTCScoring.scoringFunction`
- This module ships with the official KTC dataset and must be on `PYTHONPATH`
- Without it, only Dice/IoU are available; `ktc_score` defaults to 0.0 in the runner

### Mesh files
- `Mesh_dense` and `Mesh_sparse` each contain `Mesh` (linear) and `Mesh2` (quadratic)
- Node coordinates are in **metres** (−0.115 to +0.115)
- Element connectivity `H` uses **1-based indices** (MATLAB convention) — subtract 1
  before using in Python/NumPy

### Out of scope
- 3D reconstruction
- Hardware / EIT tank modification
- Desktop GUIs (use CLI + YAML + Streamlit only)
- Pixel greyscaling or pre-processing before metrics

---

## 9. Team

| Member | Role |
|--------|------|
| Sahil Khan | Data loading, validation, dataset analysis |
| Syeda Ulya Seerat | Method plugins, segmentation |
| Tannaz Inamdar | Batch runner, scoring, CLI |
| Areeba Masood | Visualisation, reporting |

---

## 10. What to Work on Next

Priority order for the next session:

1. **Wire `KTCLoader` into `BatchRunner`** — `_run_one` currently uses
   `MockDataPlugin`. Replace with `KTCLoader` reading from `dataset_root` in config.
2. **Add `HD95` metric** to `metrics/` and register it in the runner.
3. **Write `test_metrics.py`** — known-answer tests for Dice, IoU, and composite score.
4. **Add `PhantomLoader`** for the phantom dataset (Sprint 4 planned item).
5. **Add `BackProjection` and `GaussNewton` method plugins** (Sprint 4 planned).
