# KTC EIT Reconstruction Framework

A benchmarking and analysis platform for **Electrical Impedance Tomography (EIT)**
image-reconstruction methods, built on the **Kuopio Tomography Challenge (KTC) 2023**
dataset. It runs many reconstruction algorithms on the same data, scores them with the
official KTC metric, and lets you explore the results through an **interactive dashboard**
and a downloadable **explanatory report**.

> **New here / evaluating this project?** Jump to [Quick Start](#-quick-start) to get the
> dashboard running in a few commands, then read [What you'll see](#-what-youll-see).

---

## Contents
1. [Background — the problem](#1-background--the-problem)
2. [Quick Start](#quick-start)
3. [What you'll see](#what-youll-see)
4. [The data](#2-the-data)
5. [How scoring works](#3-how-scoring-works)
6. [Reconstruction methods & example results](#4-reconstruction-methods--example-results)
7. [The dashboard](#5-the-dashboard)
8. [The explanatory report](#6-the-explanatory-report)
9. [Project structure](#7-project-structure)
10. [Troubleshooting](#8-troubleshooting)
11. [About](#9-about)

---

## 1. Background — the problem

**Electrical Impedance Tomography (EIT)** reconstructs the electrical conductivity
*inside* an object from voltage measurements taken by electrodes around its **boundary**.
In the KTC 2023 challenge the object is a **32-electrode water tank** containing resistive
(plastic) and conductive (metal) inclusions.

Recovering the interior from boundary measurements is a classic **ill-posed inverse
problem**: small measurement errors can cause large reconstruction errors, so every method
must trade *sharpness* against *stability* through some form of regularization.

The KTC 2023 dataset defines **7 difficulty levels**. Each higher level **removes more
electrode data**, so the problem gets progressively harder. This framework runs each method
across all levels and samples, so its accuracy — and how gracefully it degrades — can be
measured and compared fairly.

---

## Quick Start

Requires **Python 3.10+**. From a terminal in the project root:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
#    Windows (PowerShell):
venv\Scripts\Activate.ps1
#    macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the framework itself (REQUIRED — makes `ktc_framework` importable)
pip install -e .

# 4. Run the benchmark (generates the scores the dashboard reads)
python run.py --config configs/ktc_all_methods.yaml

# 5. Launch the interactive dashboard
python -m streamlit run app.py
```

> **Note — Step 3 is required.** Without `pip install -e .` you will see
> `ModuleNotFoundError: No module named 'ktc_framework'`. See [Troubleshooting](#8-troubleshooting).

> **Note — Step 4 needs the dataset present** under `EvaluationData/` (see [The data](#2-the-data)).
> If it is missing, the benchmark reports `FileNotFoundError` for every sample.

Everyday use afterwards is just steps 4–5 (and re-runs of step 4 are fast thanks to caching).

---

## What you'll see

- **Dashboard** — a leaderboard of methods by KTC score, degradation curves across the 7
  difficulty levels, per-class (resistive/conductive) metric breakdowns, a geometric
  accuracy view, and reconstruction images next to the ground truth.
- **Export report** — a self-contained HTML document (openable in any browser, no internet
  needed) that *explains* the results: what each chart means, why each method scores the way
  it does, and a plain-language recommendation.

---

## 2. The data

The benchmark reads the KTC 2023 evaluation set from `EvaluationData/`, in this layout:

```
EvaluationData/
├── evaluation_datasets/
│   ├── level1/  … level7/          # voltage measurements: data1.mat, data2.mat, data3.mat, ref.mat
└── GroundTruths/
    ├── level_1/ … level_7/         # ground-truth masks: 1_true.mat, 2_true.mat, 3_true.mat
```

- `data1/2/3.mat` correspond to samples **A / B / C**.
- The ground truth is the same physical object at every level (higher levels only remove
  electrodes), so each level folder holds the same `*_true.mat` masks.
- These `.mat` files are **not stored in git** (they are large). If your checkout is missing
  them, obtain the KTC 2023 challenge data (see [About](#9-about)) and place it in the layout
  above. A copy of the challenge voltage data and ground truths also ships inside
  `external_methods/ktc2023_postprocessing-master/` and can be arranged into this structure.

Four real **training** samples are also included under `Codes_Matlab/TrainingData/` for quick
experiments via `configs/training_experiment.yaml`.

---

## 3. How scoring works

Quality is measured with the official **KTC score** — a structural-similarity (SSIM)-based
comparison to the ground truth, computed separately for the conductive and resistive regions
and averaged:

| Score | Meaning |
|------:|---------|
| **1.0** | perfect reconstruction |
| **0.0** | no better than guessing an empty tank |
| **< 0** | *worse* than an empty-tank guess |

Supporting metrics show *where* a method succeeds or fails:

- **Dice** / **IoU** — overlap accuracy, per class (resistive / conductive)
- **Hull analysis** — geometric accuracy: is each inclusion in the right position, size, shape?

Scores are also mapped to **letter grades (A–D)** for quick reading.

---

## 4. Reconstruction methods & example results

Physics-based methods run out of the box. Deep-learning / competition methods are included
as plugins; some need extra dependencies (see the table).

| Method | What it is | Extra deps |
|---|---|:--:|
| **BackProjection** | One-shot linear back-projection (transpose of the Jacobian). Fast; finds *where* an inclusion is but smears its shape. | — |
| **GaussNewton** | Single-step Tikhonov-regularized Gauss-Newton with a spatial smoothness prior. | — |
| **LinearDifferenceReconstruction** / **ReferenceFEM** / **RegularizedFEMReconstruction** | Linear difference imaging vs. an empty-tank reference, using the official KTC FEM/Jacobian pipeline. | — |
| **CompetitionCNN** | Trained CNN post-processor (KTC "ABC1" submission) that refines an initial reconstruction. | TensorFlow |
| **KTC2023_CUQI1/2**, **ml_inverse_method_2**, **ktc2023_\*** | KTC competition submissions, run as isolated subprocesses. | some need PyTorch |

**Example leaderboard** (full 7-level run; your numbers may vary):

| Rank | Method | Mean KTC |
|--:|---|--:|
| 1 | CompetitionCNN | **+0.62** |
| 2 | KTC2023_CUQI2 | +0.60 |
| 3 | KTC2023_CUQI1 / main | +0.55 |
| 4 | LinearDifference / ReferenceFEM / RegularizedFEM | +0.44 |
| 5 | GaussNewton | +0.09 |
| 6 | BackProjection | −0.02 |

You can add your own method as a plugin without touching the core — see
[PLUGINS.md](PLUGINS.md) and [EXTERNAL_METHODS.md](EXTERNAL_METHODS.md).

---

## 5. The dashboard

```bash
python -m streamlit run app.py
```

From the dashboard you can:

- Filter which **methods**, **difficulty levels**, and **samples** to compare.
- Read the **leaderboard**, **degradation curves**, **metric breakdown**, and **radar** profile.
- Inspect **reconstruction images** against the ground truth.
- **Run benchmarks** and add new methods from the sidebar.
- **Export** the explanatory HTML report.

> Tip: a full "Run all methods" is slow because the competition methods each launch a separate
> process per sample. For a quick run, deselect those and keep the physics methods — it finishes
> in about a minute. Re-runs are fast thanks to result caching (`outputs/.opcache/`).

---

## 6. The explanatory report

**Export HTML Report** (dashboard sidebar) produces a self-contained document designed to be
understood by a non-specialist. It contains:

1. Executive summary, recommendation, and project background.
2. Key statistics — the score distribution behind the headline numbers.
3. Leaderboard, degradation, and metric breakdown, with "how to read this chart" captions.
4. **Each method's definition, statistics, and the reason behind its score.**
5. Failure analysis, geometric accuracy, and reconstruction images.

Every number is computed from the selected run, so the narrative updates automatically as
methods, metrics, or filters change.

---

## 7. Project structure

```
ktc-eit-framework/
├── app.py                     # Streamlit dashboard (main entry point)
├── run.py                     # command-line benchmark runner
├── example_usage.py           # benchmark driver used by the dashboard's "Run" buttons
├── configs/                   # experiment configs (which methods/levels/samples to run)
├── src/ktc_framework/
│   ├── loaders/               # data plugins (KTC, training, mock, phantom)
│   ├── methods/               # reconstruction methods (BackProjection, GaussNewton, …)
│   ├── metrics/               # KTC score (SSIM), composite score, qualitative metrics
│   ├── runner/                # BatchRunner + config validation + result caching
│   ├── reporting/             # data layer + explanatory HTML report generator
│   ├── adapters/              # method registry + plugin detection/wrapping
│   └── plugins/               # hull (geometric) analysis
├── external_methods/          # drop-in external / competition method plugins
├── Codes_Matlab/              # real KTC training data (voltages + ground-truth masks)
├── EvaluationData/            # KTC 2023 evaluation set (see §2)
├── data/KTCScoring/           # official KTC scoring scripts
├── outputs/                   # benchmark results, figures, and the result cache
└── docs/                      # dashboard, plugin, and adapter documentation
```

More docs: [RUN_GUIDE.md](RUN_GUIDE.md) · [COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md) ·
[PLUGINS.md](PLUGINS.md) · [docs/README_DASHBOARD.md](docs/README_DASHBOARD.md)

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'ktc_framework'` | The package isn't installed in the active environment. | Run `pip install -e .` in the venv. |
| Benchmark fails with `FileNotFoundError: ... data1.mat` for every sample | The evaluation dataset is missing. | Place the KTC data under `EvaluationData/` as shown in [§2](#2-the-data). |
| Dashboard shows **"No data"** on the leaderboard | No scores have been generated yet. | Run `python run.py --config configs/ktc_all_methods.yaml` (or the sidebar "Run all methods"). |
| Some methods score **0.000** or a `torch` / `tensorflow` error appears | Those deep-learning methods need extra libraries. | Install them (`pip install torch`) or ignore — the run skips them and continues. |
| "Run all methods" is very slow | The competition methods launch a subprocess per sample. | Select only the physics methods for a fast run; re-runs are cached. |

---

## 9. About

Developed as a summer research project on EIT reconstruction benchmarking, using the
**Kuopio Tomography Challenge (KTC) 2023** dataset.

- **Dataset & challenge:** <https://www.fips.fi/KTC2023.php>
- **Team (4 members):**
  - Tannaz Inamdar
  - Areeba Masood
  - Sahil Khan
  - Syeda Ulya Seerat
