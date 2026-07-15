# External Method Integration Guide

How the KTC framework discovers, wraps, and executes external reconstruction methods — and how to add new ones.

---

## How the Framework Loads External Methods

When a benchmark runs, `BatchRunner` calls `registry.load_methods()`, which scans `external_methods/` for two things:

| What it finds | How it's loaded |
|---|---|
| A directory containing `method.yaml` | `subprocess_wrapper.py` — full manifest-driven pipeline |
| A directory containing `main.py` (no yaml) | `cli_plugin_wrapper.py` — raw KTC competition contract |

Both paths produce the same object: a `MethodPlugin` subclass whose `reconstruct(batch)` returns a `(256, 256)` `uint8` NumPy array with labels `{0: background, 1: resistive, 2: conductive}`.

---

## Path A — `method.yaml` Bundle

Preferred for ML submissions that ship as structured packages.

### What the framework does

1. Reads and validates `method.yaml` (see schema below).
2. Finds a Python interpreter that has `check_import` available — falls back to the active interpreter, checks `py -3.12` / `py -3.11` etc. via the Windows py launcher, or uses `$ENV_OVERRIDE` if set.
3. For each `reconstruct()` call:
   - Copies the sample's `data{N}.mat` + `ref.mat` into a **fresh temp directory** (so multi-sample scripts don't cross-contaminate).
   - Runs `python <entry_point> <input_dir> <output_dir> <level>` as a subprocess.
   - Reads back the first `.mat` file from `<output_dir>`.
   - Validates the `reconstruction` field: must be shape `(256, 256)`, dtype convertible to `uint8`, values only in `{0, 1, 2}`.
   - On any failure (bad exit code, missing `.mat`, wrong shape, timeout, exception): logs `[ERROR]` to `outputs/benchmark_log.txt` and returns `np.zeros((256, 256), uint8)`.

### `method.yaml` schema

```yaml
name: MyMethodName          # required; must be a valid Python identifier
description: "..."          # optional; shown in dashboard

runtime:
  python_versions: ["3.12", "3.11", "3.10"]   # checked in order via `py` launcher
  env_override: MY_METHOD_PYTHON               # $ENV_VAR → path to a specific interpreter
  check_import: torch                          # package that must be importable

solver:
  entry_point: main.py      # script to run (relative to bundle dir)
  working_dir: .            # cwd for the subprocess (relative to bundle dir)
  timeout: 900              # seconds; hard-kill after this
  args: ["input_dir", "output_dir", "level"]   # positional args passed to script

weights:                    # optional; if listed, validate_manifest() errors if files missing
  - postprocessing_model/version_01/model.pt

sample_map:                 # optional; maps sample letter → data file number
  A: "1"
  B: "2"
  C: "3"
```

### Output contract for `entry_point`

The script must write a `.mat` file into `output_dir` containing a field named `reconstruction`:

```python
import scipy.io, numpy as np
seg = np.zeros((256, 256), dtype=np.int32)   # fill with 0/1/2
scipy.io.savemat(output_dir + "/1.mat", {"reconstruction": seg})
```

---

## Path B — Raw KTC CLI Contract (no `method.yaml`)

Used for competition submissions that were never packaged as bundles.

### Contract

```
python main.py <inputFolder> <outputFolder> <categoryNbr>
```

- `inputFolder` contains `ref.mat` (empty-tank reference) and `data1.mat` (one sample measurement).
- `outputFolder` receives `1.mat` with field `reconstruction` — same `(256, 256)` shape and `{0, 1, 2}` label requirement as Path A.

### Auto-detection without `method.yaml`

When a zip is uploaded in the dashboard with no `method.yaml`, `entry_detector.py` classifies it:

- **`cli_script`**: has `if __name__ == "__main__":` + argparse/`sys.argv` → wrapped by `cli_plugin_wrapper.py`
- **`method_plugin`**: defines a class inheriting something named `*Plugin` or `*Method` with a `reconstruct()` method → imported in-process (no subprocess overhead)

---

## Dashboard Zip Upload — What It Can and Cannot Do

The sidebar **Upload method** widget accepts `.zip` files. It is designed for **method code only**, not large weight files.

### What it does

1. Writes the zip to `external_methods/<stem>.zip`.
2. Tries to extract as a bundle (`extract_bundle`) — succeeds if the zip contains `method.yaml` at its root or inside a single top-level directory.
3. If no `method.yaml` is found, falls back to `extract_archive` + auto-generated manifest.
4. Registers the method and adds it to the benchmark config.
5. On any exception: **deletes the entire extracted directory** and shows an error in the sidebar.

### Size limit

Streamlit defaults to a **200 MB** upload cap. `.streamlit/config.toml` now sets:

```toml
[server]
maxUploadSize = 2000   # 2 GB
```

Even so, uploading a multi-hundred-MB zip through a browser is slow and unreliable. **Large weight files should always be placed on disk manually** alongside the method code — see the layout reference at the bottom of this document.

### When to use zip upload vs. manual placement

| Scenario | Use |
|---|---|
| New method code (< 50 MB, no weights) | Zip upload ✓ |
| Method code + large weights (> 200 MB) | Code via zip, weights manually on disk |
| Re-uploading to add weights to existing method | Place weights on disk directly — do not re-upload |
| Weights are already on disk and method is registered | Just run the benchmark — no re-upload needed |

---

## Diagnosing Silent Zero Scores

All subprocess failures degrade to `np.zeros((256, 256))` rather than crashing the benchmark. Errors are logged at `WARNING`/`ERROR` level and written to `outputs/benchmark_log.txt` (routing added in `example_usage.py`).

### Recognising the symptom

```
KTC score = 0.0 for ALL runs  +  Dice alternates 0.0/1.0
→ method is outputting all-zeros (trivially matches absent class)
```

When the true segmentation has no conductive inclusions, `dice_conductive = 1.0` trivially (both prediction and ground-truth are empty). This makes dashboard averages look non-zero when every individual run actually failed.

### Diagnostic checklist

1. **Check the benchmark log** — `outputs/benchmark_log.txt` — for lines starting with `[ERROR]` or `[WARNING]`.
2. **Run the method manually** to capture the real stderr:
   ```
   python external_methods/<method>/main.py <input_dir> <output_dir> <level>
   ```
3. **Common causes**:

| Symptom in log | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named '...'` | Missing dependency | `pip install <package>` or set `env_override` |
| `FileNotFoundError: .../model.pt` | Weights not on disk | Download and place at the expected path |
| `FileNotFoundError: .../version_01/` | Path points to directory, not file | Fix `level_to_model_path` to include the filename |
| `subprocess exited 1`, no other detail | Script crash | Run manually, read stderr |
| `no .mat file in output dir` | Script ran but wrote nothing | Check the script's output format |
| `reconstruction shape ... != (256, 256)` | Wrong output dimensions | Adapt the script or add a reshape step |
| No `[ERROR]` at all + very fast runtime | Cached zeros from `.opcache/` | Delete `outputs/.opcache/<method>/` and re-run |
| `RuntimeError: Expected all tensors on same device` | Mixed CPU/GPU tensors | Ensure all tensors use the same `device` variable |

---

## `ktc2023_postprocessing_master` — Integration Record

This method is a PyTorch UNet that postprocesses a linearised EIT reconstruction into a 3-class segmentation. Below is a record of what was needed to make it work.

### Bugs found and fixed in `main.py`

**Bug 1 — `torch.load` received a directory path instead of a file path.**

```python
# Before (broken — torch.load cannot open a directory)
level_to_model_path = {
    1: "postprocessing_model/version_01/",
    ...
}

# After (fixed)
level_to_model_path = {
    1: "postprocessing_model/version_01/model.pt",
    ...
}
```

The comment in the original file even showed the correct path (`/model.pt`) — the author stripped the filename before submitting.

**Bug 2 — `level` tensor hardcoded to CUDA regardless of available device.**

```python
# Before (crashes on CPU-only machines; reco follows device but level doesn't)
reco = torch.from_numpy(sigma_reco).float().to(device).unsqueeze(0)
level = torch.tensor([level]).to("cuda")

# After (fixed)
reco = torch.from_numpy(sigma_reco).float().to(device).unsqueeze(0)
level = torch.tensor([level]).to(device)
```

### Missing Python dependency

`ml_collections` is imported in `configs/postprocessing_config.py` but is not in the base environment. Install once:

```
pip install ml-collections
```

### Required asset downloads

The zip upload **cannot** deliver these — place them on disk manually.

**Model weights** — download from: https://seafile.zfn.uni-bremen.de/d/faaf3799e6e247198a23/

```
external_methods/ktc2023_postprocessing-master/postprocessing_model/version_01/model.pt
```

**Precomputed matrices** — download from: https://seafile.zfn.uni-bremen.de/d/9108bc95b2e84cd285f8/

```
external_methods/ktc2023_postprocessing-master/data/
  jac_sparse.npy                    ← Jacobian (precomputed with FEniCS)
  mesh_neighbour_matrix_sparse.npy  ← Graph-Laplacian TV regulariser
  smoothnessR_sparse.npy            ← Smoothness prior matrix
  mesh_coordinates_sparse.npy       ← FEM mesh node coordinates
  cells_sparse.npy                  ← FEM mesh triangle connectivity
```

`LinearisedRecoFenics` loads all five at `__init__` time; if any is missing the subprocess crashes immediately and returns zeros.

### Why the zip upload didn't work

The user's zip contained all assets but exceeded Streamlit's default 200 MB upload cap. The upload was silently rejected; the exception handler then deleted the partially-extracted bundle. The fix was two-fold:

1. `maxUploadSize = 2000` added to `.streamlit/config.toml`.
2. Assets placed on disk directly — the zip upload path is not suited for weights files regardless of size.

### Optional: dedicated Python environment

The method was developed with Python 3.10 + CUDA 11.8. The `env_override` key in `method.yaml` allows pointing the framework at a different interpreter without changing the default environment:

```bash
conda env create -f external_methods/ktc2023_postprocessing-master/environment.yml
conda activate eit_env
python -c "import sys; print(sys.executable)"   # copy this path
```

```bash
# Set before starting Streamlit:
set KTC2023_POSTPROCESSING_MASTER_PYTHON=C:\path\to\eit_env\python.exe
streamlit run app.py
```

### Verification

After placing both asset groups on disk, run:

```
configs/runtime_ktc2023_postprocessing_master.yaml
```

Expected: non-zero KTC scores across all 21 runs, no `[ERROR]` lines in `benchmark_log.txt`.

---

## Adding a New External Method

### From a zip upload (dashboard)

Zip must contain `method.yaml` alongside the code. Weights should **not** be in the zip — place them on disk after the method is registered.

```
my_method.zip
├── method.yaml        ← required
├── main.py
└── src/               ← supporting modules
```

### From a directory (manual)

1. Place the method directory under `external_methods/`.
2. Add a `method.yaml` following the schema above.
3. Click **Scan external methods** in the dashboard sidebar — or restart Streamlit.

### Writing a compatible `main.py`

```python
import argparse
import numpy as np
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("output_dir")
parser.add_argument("level", type=int)
args = parser.parse_args()

data = scipy.io.loadmat(f"{args.input_dir}/data1.mat")
ref  = scipy.io.loadmat(f"{args.input_dir}/ref.mat")

# ... reconstruction logic ...
seg = np.zeros((256, 256), dtype=np.int32)  # 0=bg, 1=resistive, 2=conductive

scipy.io.savemat(f"{args.output_dir}/1.mat", {"reconstruction": seg})
```

---

## File Layout Reference

```
external_methods/
  <method_name>/
    method.yaml                          ← manifest (required for Path A)
    main.py                              ← entry point
    postprocessing_model/
      version_01/
        model.pt                         ← weights (place manually; NOT via zip upload)
    data/                                ← precomputed matrices (place manually)
      jac_sparse.npy
      mesh_neighbour_matrix_sparse.npy
      smoothnessR_sparse.npy
      mesh_coordinates_sparse.npy
      cells_sparse.npy
    src/                                 ← supporting source code
    configs/                             ← method-specific config
```
