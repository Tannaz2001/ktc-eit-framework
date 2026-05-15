# KTC EIT Benchmarking Framework

> A modular, reproducible evaluation and visualisation pipeline for the **Kuopio Tomography Challenge 2023** Electrical Impedance Tomography dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Sprint](https://img.shields.io/badge/Sprint-3%20%E2%80%94%2015%20May%202026-orange?style=flat-square)](#sprint-progress)
[![KTC 2023](https://img.shields.io/badge/Dataset-KTC%202023-purple?style=flat-square)](https://zenodo.org/record/10986692)

---

## What This Project Does

This framework provides a **plug-and-play benchmarking pipeline** for EIT image reconstruction methods. Given voltage measurements and injection patterns from the KTC 2023 dataset, it:

1. Loads and validates KTC `.mat` files through a unified `DataBatch` interface
2. Runs any registered reconstruction method through a standardised `MethodPlugin` API
3. Scores results using official KTC SSIM + complementary metrics (Dice, IoU, HD95)
4. Exports a leaderboard HTML report and per-sample visual overlays

**The core never changes.** Methods, datasets, and metrics are all plugins — adding a new one requires one file and one decorator.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Sprint 3 — What Was Built Today](#sprint-3--what-was-built-today-15-may-2026)
- [Running the Benchmark](#running-the-benchmark)
- [Adding a New Method](#adding-a-new-method-in-5-minutes)
- [Adding a New Metric](#adding-a-new-metric)
- [Testing](#testing)
- [Team](#team)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/ktc-eit-framework.git
cd ktc-eit-framework

# 2. Create the reproducible environment (pinned dependencies)
conda env create -f environment.yml
conda activate eit-bench

# 3. Download the KTC 2023 dataset
#    Get it from: https://zenodo.org/record/10986692
#    Place .mat files in: data/ktc/

# 4. Run the full benchmark
python run.py --config experiments/ktc_all_methods.yaml

# 5. Open the results dashboard
streamlit run app.py
```

> **Reproducibility guarantee:** `conda env create -f environment.yml` plus one run command produces identical `scores.json` and `report.html` on any Linux or macOS machine. The Git SHA is embedded in every output log.

---

## Project Structure

```
ktc-eit-framework/
│
├── src/
│   └── ktc_framework/
│       ├── types.py                  ← DataBatch namedtuple (the central data contract)
│       │
│       ├── loaders/                  ← DataPlugin implementations
│       │   ├── __init__.py           ← auto-registers all loaders on import
│       │   ├── ktc_loader.py         ← KTCLoader: reads KTC .mat files  ✅ Sprint 3
│       │   └── mock_data_plugin.py   ← MockDataPlugin: random data for testing  ✅ Sprint 3
│       │
│       ├── methods/                  ← MethodPlugin implementations
│       │   ├── __init__.py           ← auto-registers all methods on import
│       │   ├── level_set_plugin.py   ← LevelSetPlugin: Otsu + contour extraction  ✅ Sprint 3
│       │   └── hull_plugin.py        ← HullPlugin: connected-component geometry  ✅ Sprint 3
│       │
│       ├── metrics/                  ← MetricPlugin implementations
│       │   ├── __init__.py
│       │   └── ...                   ← (Sprint 5)
│       │
│       ├── core/                     ← Immutable pipeline (do not edit after v0.1)
│       │   ├── registry.py           ← PluginRegistry + @registry.register decorator
│       │   ├── runner.py             ← BatchRunner: loops levels × samples × methods
│       │   ├── logger.py             ← JSON experiment logger with Git SHA
│       │   └── exporter.py           ← report.html + KTC-format PNG export
│       │
│       └── viz/                      ← Visualisation module
│           └── ...                   ← (Sprint 8–9)
│
├── experiments/
│   ├── ktc_all_methods.yaml          ← Full benchmark config (all 7 levels, all methods)
│   ├── ktc_baseline.yaml             ← Fast sanity-check run
│   └── mock_test.yaml                ← Mock data run (no dataset needed)
│
├── tests/
│   ├── test_loaders.py               ← DataPlugin contract tests
│   ├── test_plugins.py               ← MethodPlugin shape + output tests  ✅ Sprint 3
│   └── test_metrics.py               ← Known-answer metric validation (Sprint 5)
│
├── data/
│   └── ktc/                          ← Place KTC .mat files here (not committed)
│
├── outputs/                          ← Auto-generated: PNGs, scores.json, report.html
├── environment.yml                   ← Pinned conda environment
├── Dockerfile                        ← Containerised alternative
├── run.py                            ← CLI entry point
└── app.py                            ← Streamlit dashboard entry point
```

---

## Architecture

The framework follows a **three-ring plugin architecture** around an immutable core:

```
┌─────────────────────────────────────────────────────┐
│  YAML config layer — one file drives everything     │
│                                                     │
│  ┌──────────────┐                                   │
│  │  DataPlugin  │──┐                                │
│  │  (loaders/)  │  │                                │
│  └──────────────┘  │   ┌─────────────────────┐     │
│                    ├──▶│   Immutable Core     │     │
│  ┌──────────────┐  │   │   BatchRunner        │──▶ scores.json
│  │ MethodPlugin │──┘   │   PluginRegistry     │    report.html
│  │  (methods/)  │      │   ExperimentLogger   │    PNG panels
│  └──────────────┘  ┌──▶│                     │     │
│                    │   └─────────────────────┘     │
│  ┌──────────────┐  │                                │
│  │ MetricPlugin │──┘                                │
│  │  (metrics/)  │                                   │
│  └──────────────┘                                   │
└─────────────────────────────────────────────────────┘
```

### The three plugin contracts

| Plugin type | One method to implement | Input | Output |
|---|---|---|---|
| `DataPlugin` | `load(config) → DataBatch` | YAML config dict | `DataBatch(voltages, patterns, ground_truth)` |
| `MethodPlugin` | `reconstruct(batch) → np.ndarray` | `DataBatch` | `256×256` int array, labels `{0, 1, 2}` |
| `MetricPlugin` | `score(pred, gt) → float` | Two `256×256` arrays | Float `0–100`, normalised |

### The I/O contract

Every reconstruction method **must** return a `256×256` NumPy array of integer pixel labels:

| Label | Class | Meaning |
|---|---|---|
| `0` | Water | Background (homogeneous saline) |
| `1` | Resistive | Insulating inclusion |
| `2` | Conductive | Conducting inclusion |

Any method returning the wrong shape or label values is **rejected at registration time** by the validator in `MethodPlugin`.

---

## Sprint 3 — What Was Built Today (15 May 2026)

### 1. `types.py` moved to framework root

`DataBatch` is now importable from the top-level package:

```python
from src.ktc_framework.types import DataBatch
```

This makes `DataBatch` the **single source of truth** for the data contract across all loaders, methods, and tests. The interface is now frozen — changing it requires a team discussion.

---

### 2. `KTCLoader` — reads real KTC `.mat` files

**File:** `src/ktc_framework/loaders/ktc_loader.py`

```python
from src.ktc_framework.types import DataBatch
from src.ktc_framework.core.registry import registry

@registry.register
class KTCLoader(DataPlugin):
    """Loads a single KTC 2023 .mat file and returns a validated DataBatch."""

    def load(self, config: dict) -> DataBatch:
        # Opens .mat file, validates field shapes, returns DataBatch
        ...
```

Validates that:
- `voltages` has shape `(76, 30)` — one row per injection pattern
- `injection_patterns` has shape `(76, 2)` — electrode pair indices
- `ground_truth` contains only values `{0, 1, 2}`

Raises a descriptive error if any check fails.

---

### 3. `MockDataPlugin` — fake data for offline testing

**File:** `src/ktc_framework/loaders/mock_data_plugin.py`

Returns a `DataBatch` filled with random arrays at the correct shapes. Allows Tannaz and Areeba to build and test the runner and visualisation modules **without needing the real KTC dataset**.

---

### 4. `LevelSetPlugin` — Otsu binarisation + contour extraction

**File:** `src/ktc_framework/methods/level_set_plugin.py`

```python
@registry.register
class LevelSetPlugin(MethodPlugin):
    """
    Binarises a 256×256 reconstruction array using Otsu thresholding,
    then extracts physical interfaces using skimage.measure.find_contours.
    """

    def reconstruct(self, batch: DataBatch) -> dict:
        # 1. Apply skimage.filters.threshold_otsu → binary mask
        # 2. Run skimage.measure.find_contours → list of contour arrays
        # 3. Return {'contours': [...], 'n_objects': count}
        ...
```

**Logging:** `INFO` and `DEBUG` statements track array shapes, computed Otsu threshold, and number of contours identified.

---

### 5. `HullPlugin` — connected-component geometry extraction

**File:** `src/ktc_framework/methods/hull_plugin.py`

```python
@registry.register
class HullPlugin(MethodPlugin):
    """
    Given a 256×256 segmentation map and a target_label {0, 1, 2},
    extracts geometric properties of all connected components of that class.
    """

    def reconstruct(self, batch: DataBatch, target_label: int = 1) -> list:
        # 1. Filter mask to target_label pixels
        # 2. skimage.measure.label → connected components
        # 3. skimage.measure.regionprops → centroid, area, bbox, convex_area
        # 4. Return list of dicts, one per object
        ...
```

**Use case:** feeding centroid positions into the **centroid error metric** (Tier 4, Sprint 5).

---

### 6. `test_plugins.py` — synthetic shape validation

**File:** `tests/test_plugins.py`

Procedurally generates:
- A synthetic circle array (via `np.ogrid`) to test `LevelSetPlugin`
- A synthetic block array to test `HullPlugin`

Validates that both plugins execute without errors and produce output in the expected format. Run with:

```bash
pytest tests/test_plugins.py -v
```

---

### 7. Git setup resolved

Diagnosed `fatal: not a git repository` context issue. Repository remote set and all Sprint 3 files staged, committed, and pushed to the team fork.

---

## Running the Benchmark

### Full benchmark (all levels, all methods)

```bash
python run.py --config experiments/ktc_all_methods.yaml
```

Produces:
- `outputs/scores.json` — all metric results, Git SHA tagged
- `outputs/report.html` — self-contained leaderboard report
- `outputs/level_X/sample_Y/method_Z.png` — per-sample visual panels

### Quick sanity check (mock data, no dataset needed)

```bash
python run.py --config experiments/mock_test.yaml
```

### Open the interactive dashboard

```bash
streamlit run app.py
```

---

## Adding a New Method in 5 Minutes

1. Create a file in `src/ktc_framework/methods/`

```python
# src/ktc_framework/methods/my_method.py
import numpy as np
from src.ktc_framework.types import DataBatch
from src.ktc_framework.core.registry import registry
from src.ktc_framework.methods.base import MethodPlugin

@registry.register                          # ← this one line registers it
class MyMethod(MethodPlugin):
    """One-sentence description of your method."""

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        voltages = batch.voltages            # shape: (76, 30)
        patterns = batch.injection_patterns  # shape: (76, 2)

        # --- your reconstruction logic here ---
        result = np.zeros((256, 256), dtype=int)

        return result                        # must be 256×256, values in {0,1,2}
```

2. Add the method name to your YAML config

```yaml
methods:
  - MyMethod
```

3. Run — the framework discovers it automatically. No other files change.

```bash
python run.py --config experiments/ktc_all_methods.yaml
```

---

## Adding a New Metric

```python
# src/ktc_framework/metrics/my_metric.py
from src.ktc_framework.core.registry import registry
from src.ktc_framework.metrics.base import MetricPlugin

@registry.register
class MyMetric(MetricPlugin):
    tier = 5            # which tier (1–5) this metric belongs to
    weight = 0.02       # contribution to composite score
    higher_is_better = True

    def score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # pred and gt are both 256×256 int arrays with labels {0,1,2}
        return my_formula(pred, gt)   # return a float 0–100
```

The composite scorer and dashboard pick it up automatically on the next run.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_loaders.py -v     # DataPlugin contract tests
pytest tests/test_plugins.py -v     # MethodPlugin shape + output tests
pytest tests/test_metrics.py -v     # Known-answer metric validation
```

### What the tests check

| Test file | What it validates |
|---|---|
| `test_loaders.py` | All 5 KTC training targets load with correct shapes and no NaN values |
| `test_plugins.py` | Both Sprint 3 plugins run without errors on synthetic shape data |
| `test_metrics.py` | KTCScore matches published Table 2 values within 0.01 tolerance |

---

## Environment

```bash
conda env create -f environment.yml   # create
conda activate eit-bench              # activate
conda env export > environment.yml    # update after installing new packages
```

Key dependencies: `scipy`, `numpy`, `scikit-image`, `matplotlib`, `seaborn`, `rich`, `streamlit`, `pytest`, `pyeit`

---

## Sprint Progress

| Sprint | Weeks | Status | Key deliverable |
|---|---|---|---|
| 1 | W1 | ✅ Done | Shared data dictionary, repo setup, colour map |
| 2 | W2 | ✅ Done | `DataBatch`, abstract plugin classes, mock data |
| 3 | W3 | ✅ Done | `KTCLoader`, `LevelSetPlugin`, `HullPlugin`, `test_plugins.py` |
| 4 | W4 | 🔄 Next | `PhantomLoader`, `BackProjection`, `GaussNewton`, full pipeline run |
| 5 | W5 | ⏳ Planned | `KTCScore`, `DiceScore`, `MeanIoU`, `HD95`, composite scorer |
| 6 | W6 | ⏳ Planned | Full 7-level × 3-sample benchmark, `report.html`, Git tag v0.1 |
| 7 | W7 | ⏳ Planned | `GaussNewtonUNet` learned post-processor |
| 8–9 | W8–9 | ⏳ Planned | Streamlit dashboard, weight editor, failure gallery |
| 10–12 | W10–12 | ⏳ Planned | Final report, demo, handover |

---

## Team

| Member | Role | Sprint 3 contribution |
|---|---|---|
| **Sahil Khan** | Data loading + validation | `KTCLoader`, `MockDataPlugin`, field validation schema |
| **Syeda Ulya Seerat** | Method plugins + segmentation | `LevelSetPlugin`, `HullPlugin`, `test_plugins.py` |
| **Tannaz Inamdar** | Batch runner + scoring + CLI | `BatchRunner` wiring, `PluginRegistry`, YAML config |
| **Areeba Masood** | Visualisation + reporting | Plot functions, HTML report template |

---

## Scope Boundaries

This framework is **benchmarking infrastructure**, not a novel reconstruction algorithm. The following are explicitly out of scope:

| Out of scope | Why |
|---|---|
| 3D reconstruction | Physics and KTC data are strictly 2D |
| Building or modifying the EIT tank | Hardware is not part of this project |
| Inventing a new mathematical EIT solver | Baselines (Back-projection, Gauss-Newton) prove the pipeline |
| Desktop GUIs (PyQt, Tkinter) | CLI + YAML + Streamlit only |
| Pixel greyscaling or image preprocessing | Metrics operate on integer label arrays `{0,1,2}` directly |

---

## Reference

> Räsänen et al., *Kuopio Tomography Challenge 2023 — Electrical Impedance Tomography Competition and Open Dataset*, Applied Mathematics for Modern Challenges, 2(2), 93–118, 2024. DOI: [10.3934/ammc.2024009](https://doi.org/10.3934/ammc.2024009)
>
> KTC 2023 dataset: [Zenodo 10986692](https://zenodo.org/record/10986692)
