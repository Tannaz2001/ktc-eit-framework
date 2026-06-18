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

If integrating competition submissions:

```bash
# Create external_methods directory
mkdir external_methods

# Clone or copy ABC1 submission to:
external_methods/abc1/
├── solver.py
├── main_python.py
├── Mesh_sparse.mat
└── TrainingData/
    └── ref.mat
```

Set environment variable (optional — auto-detected if not set):
```bash
export ABC1_SUBMISSION_PATH=/path/to/abc1
# or on Windows:
set ABC1_SUBMISSION_PATH=C:\path\to\abc1
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

Edit `configs/ktc_all_methods.yaml`:

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

### Issue: "CompetitionCNN failed: TensorFlow not found"
**Solution (optional — skip if not using CompetitionCNN):**
```bash
# Install Python 3.12 with TensorFlow
python3.12 -m pip install tensorflow

# Set env var
export ABC1_PYTHON=python3.12

# Or auto-detection will find it
```

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
