# KTC EIT Benchmarking Framework — Complete Setup & Run Guide

## 1️⃣ INITIAL SETUP (First Time Only)

### Clone the repository
```bash
git clone https://github.com/Tannaz2001/ktc-eit-framework.git
cd ktc-eit-framework
```

### Create Python 3.14 virtual environment
```bash
python -m venv venv
```

### Activate virtual environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2️⃣ DATA SETUP

### Download KTC 2023 Dataset

The benchmark requires:
- **EvaluationData/** — 7 levels with A/B/C samples + GroundTruths
- **Codes_Matlab/Mesh_sparse.mat** — FEM mesh geometry

Place in project root:
```
ktc-eit-framework/
├── EvaluationData/
│   ├── level1/
│   │   ├── SampleA.mat
│   │   ├── SampleB.mat
│   │   └── SampleC.mat
│   ├── level2/ ... level7/
│   └── GroundTruths/
│       └── GTs_BINARY.mat
└── Codes_Matlab/
    └── Mesh_sparse.mat
```

### (Optional) Set up CompetitionCNN (ABC1 Submission)

The CNN runs as a separate subprocess and requires the ABC1 repo placed
**next to** (not inside) the framework root:

```
Desktop/
├── KTC_WORK_HIS/          ← framework (this repo)
└── KTC2023-ABC1/
    └── KTC2023_Python_A01+/
        ├── main_python.py
        ├── MiscCodes/
        │   └── solver.py  ← needs the bug fix below
        └── ...
```

**Required bug fix in `solver.py`** — without this the CNN predicts no
conductive inclusions and scores flat `+0.0000` across all levels.

Open `KTC2023-ABC1/KTC2023_Python_A01+/MiscCodes/solver.py` and find
the `otsu_segmentation` function (~line 199). Change two lines:

```python
# BEFORE (broken — both non-background classes get the same label)
case 0:
    deltareco_pixgrid_segmented[ind1] = 2
    deltareco_pixgrid_segmented[ind2] = 2   # ← wrong, should be 2
case 2:
    deltareco_pixgrid_segmented[ind0] = 1
    deltareco_pixgrid_segmented[ind1] = 1   # ← wrong, should be 2

# AFTER (fixed — resistive=1, conductive=2 correctly assigned)
case 0:
    deltareco_pixgrid_segmented[ind1] = 1   # ← changed: 2 → 1
    deltareco_pixgrid_segmented[ind2] = 2
case 2:
    deltareco_pixgrid_segmented[ind0] = 1
    deltareco_pixgrid_segmented[ind1] = 2   # ← changed: 1 → 2
```

**TensorFlow** is required to run the CNN (~500 MB, CPU-only on Windows):
```bash
pip install tensorflow
```
If TF is not installed, the framework will print a clear warning at startup
and skip the CNN — all other methods still run normally.

If your ABC1 repo is not in the default location, set the env var:
```bash
# Windows PowerShell
$env:ABC1_SUBMISSION_PATH = "C:\path\to\KTC2023_Python_A01+"

# Windows CMD
set ABC1_SUBMISSION_PATH=C:\path\to\KTC2023_Python_A01+
```

---

## 3️⃣ VERIFY SETUP

### Check that all files are in place
```bash
# Verify data exists
ls -la EvaluationData/level1/
ls -la Codes_Matlab/Mesh_sparse.mat

# Verify Python environment
python --version  # Should be 3.14.x
pip list | grep -E "numpy|scipy|scikit-image|streamlit"
```

### Run tests (optional but recommended)
```bash
pytest tests/ -v
# Expected: 12/12 hull plugin tests passing
```

---

## 4️⃣ CONFIGURE BENCHMARK

### If your data is not at the default path

The shared config assumes `EvaluationData/` is in the project root. If your
data is somewhere else, create a **machine-local config** (never committed):

```bash
# copy the shared config and rename it
cp configs/ktc_all_methods.yaml configs/local_ktc_all_methods.yaml
```

Then edit `configs/local_ktc_all_methods.yaml` and set your actual paths:
```yaml
dataset_root: C:/Users/yourname/data/EvaluationData
mesh_path:    C:/Users/yourname/data/Codes_Matlab/Mesh_sparse.mat
```

Run with your local config:
```bash
python run.py --config configs/local_ktc_all_methods.yaml
```

Files matching `configs/local_*.yaml` are git-ignored — your paths will never
accidentally get committed and break someone else's setup.

### Edit `configs/ktc_all_methods.yaml`

**Full benchmark (all 7 levels, all samples):**
```yaml
data_plugin: KTCDataPlugin
dataset_root: EvaluationData
mesh_path: Codes_Matlab/Mesh_sparse.mat

levels: [1, 2, 3, 4, 5, 6, 7]
samples: [A, B, C]

methods:
  - RegularizedFEMReconstruction
  - ReferenceFEM
  - LinearDifferenceReconstruction
  - BackProjection
  - GaussNewton
  - CompetitionCNN  # (optional)

method_plugin_paths:
  - external_methods

include_external_methods: true

output_dir: outputs/
```

Or **quick test** (1 level only):
```yaml
levels: [1]
samples: [A]
methods:
  - BackProjection
  - GaussNewton
```

---

## 5️⃣ RUN THE BENCHMARK

### Option A: Full Benchmark (All Methods × Levels × Samples)
```bash
python run.py --config configs/ktc_all_methods.yaml
```

**Expected Runtime:**
- ~5-10 minutes (quick test, 1 level)
- ~30-60 minutes (full benchmark, 7 levels)
- Depends on: dataset size, CPU cores, method complexity

**Output:**
```
outputs/
├── scores.json                 # All metrics per run
├── scores_nested.json          # Method → Level → Sample structure
├── per_run_metrics.json        # Dashboard-ready format
├── dashboard_scores.json       # Averaged per method
├── report.html                 # Summary report with tables
├── figures/                    # Comparison PNGs
├── images/                     # Per-run segmentation panels
├── mat_predictions/            # .mat files per method/level
└── overlays/                   # Error overlay visualizations
```

### Option B: Quick Test (Single Method, Single Level)
```bash
# Edit configs/ktc_all_methods.yaml to:
# levels: [1]
# samples: [A]
# methods:
#   - BackProjection

python run.py --config configs/ktc_all_methods.yaml
```

**Expected Runtime:** ~30 seconds

### Option C: Custom Configuration
```bash
# Create your own config
cat > configs/my_test.yaml << 'EOF'
data_plugin: KTCDataPlugin
dataset_root: EvaluationData
mesh_path: Codes_Matlab/Mesh_sparse.mat
levels: [1, 2, 3]
samples: [A, B]
methods:
  - GaussNewton
  - BackProjection
output_dir: outputs_my_test/
EOF

# Run with it
python run.py --config configs/my_test.yaml
```

---

## 6️⃣ VIEW RESULTS

### View HTML Report
```bash
# Open in browser
open outputs/report.html              # macOS
xdg-open outputs/report.html          # Linux
start outputs/report.html             # Windows (cmd)
```

Or use Python server:
```bash
cd outputs
python -m http.server 8000
# Visit: http://localhost:8000/report.html
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

**Browser opens automatically to:**
```
http://localhost:8501
```

**Features:**
- 7 analysis tabs (Leaderboard, Heatmap, Method Details, etc.)
- Dark mode toggle
- Real-time qualitative metrics (detection rates, hull IoU)
- Filter by level/method/sample
- Download raw data

### View JSON Scores
```bash
# Pretty-print scores
python -c "import json; print(json.dumps(json.load(open('outputs/scores.json')), indent=2))" | head -50

# Count total runs
python -c "import json; data=json.load(open('outputs/scores.json')); print(f'Total runs: {len(data)}')"

# Get per-method averages
python -c "import json; data=json.load(open('outputs/dashboard_scores.json')); 
for m, metrics in data.items(): print(f'{m}: KTC={metrics.get(\"ktc_score\", 0):.4f}')"
```

---

## 7️⃣ OPTIONAL: RUN TESTS

### Run all tests
```bash
pytest tests/ -v
```

### Run only hull plugin tests
```bash
pytest tests/test_hull_plugin.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src/ktc_framework --cov-report=html
open htmlcov/index.html
```

---

## 8️⃣ TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'src.ktc_framework'"
**Solution:**
```bash
# Make sure you're in project root
cd ktc-eit-framework
python run.py --config configs/ktc_all_methods.yaml
```

### Issue: "EvaluationData not found"
**Solution:**
```bash
# Check files are in place
ls -la EvaluationData/level1/
ls -la EvaluationData/GroundTruths/

# If missing, benchmark will still run but GT will be all zeros (mock mode)
```

### Issue: "[CompetitionCNN] TensorFlow not found" printed at startup
This is a warning, not a crash. The benchmark still runs — CNN just returns
zeros and is excluded from meaningful analysis.

**To fix (optional):**
```bash
pip install tensorflow          # adds TF to your current venv
```
Or point to a Python that already has TF:
```bash
# Windows PowerShell
$env:ABC1_PYTHON = "C:\Python312\python.exe"
```
If you don't need the CNN, comment it out in your config:
```yaml
methods:
  - BackProjection
  - GaussNewton
  # - CompetitionCNN    ← comment out to skip
```

### Issue: CompetitionCNN runs but shows `+0.0000` degradation slope
The `otsu_segmentation` bug in ABC1's `solver.py` is causing this.
Apply the two-line fix described in the **Set up CompetitionCNN** section above.

### Issue: "Streamlit ModuleNotFoundError"
**Solution:**
```bash
pip install streamlit
streamlit run app.py
```

### Issue: "Port 8501 already in use" (Streamlit)
**Solution:**
```bash
streamlit run app.py --server.port=8502
```

---

## 9️⃣ FULL WORKFLOW EXAMPLE

### Complete from-scratch setup:
```bash
# 1. Clone
git clone https://github.com/Tannaz2001/ktc-eit-framework.git
cd ktc-eit-framework

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify data is in place
ls EvaluationData/level1/SampleA.mat

# 5. Quick test
python run.py --config configs/ktc_all_methods.yaml

# 6. View results
streamlit run app.py
# Browser opens to http://localhost:8501
```

### Expected output after run:
```
[green]Mesh loaded:[/green] Codes_Matlab/Mesh_sparse.mat
[BENCH_PROGRESS] completed=1/1 method=BackProjection level=1 sample=A
✓ Experiment complete
✓ scores.json saved to: outputs/scores.json
✓ Figures saved: 1 PNGs -> outputs/figures
✓ HTML report: outputs/report.html
```

---

## 🔟 ENVIRONMENT VARIABLES (Optional)

```bash
# Override dataset location
export KTC_DATASET_ROOT=/custom/path/to/EvaluationData

# Override mesh location
export KTC_MESH_PATH=/custom/path/to/Mesh_sparse.mat

# Python 3.12 for CompetitionCNN (auto-detected if not set)
export ABC1_PYTHON=python3.12

# ABC1 submission location (auto-searched if not set)
export ABC1_SUBMISSION_PATH=/path/to/abc1/submission

# Streamlit config
export STREAMLIT_SERVER_PORT=8502
```

---

## 📊 Output Structure

After running benchmark:
```
outputs/
├── scores.json                    # All per-run metrics (flat)
├── scores_nested.json             # Nested by method/level/sample
├── per_run_metrics.json           # Dashboard-ready format
├── dashboard_scores.json          # Aggregated per-method
├── report.html                    # HTML summary report
│
├── figures/
│   ├── BackProjection_level1_sampleA.png
│   ├── comparison_grid.png
│   ├── failure_gallery.png
│   ├── degradation_curve.png
│   └── leaderboard.png
│
├── images/
│   ├── BackProjection_level1_sampleA.png
│
├── mat_predictions/
│   └── BackProjection/
│       ├── level_1/
│       │   ├── sample_A.mat
│       │   ├── sample_B.mat
│       │   └── sample_C.mat
│
└── overlays/
    └── BackProjection/
        └── level_1/
            ├── sample_A.png
```

---

## 💡 Tips

1. **Start with quick test** — Run 1 level to verify setup works
2. **Monitor progress** — Watch console for `[BENCH_PROGRESS]` markers
3. **Check HTML report first** — Fastest way to see results
4. **Use Streamlit for exploration** — Better for interactive analysis
5. **Keep outputs folder** — Each run overwrites previous results
6. **Backup good runs** — Copy outputs/ to outputs_run_name/ before re-running
