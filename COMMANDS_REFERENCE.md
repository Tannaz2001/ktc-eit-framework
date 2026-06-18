# KTC EIT Framework — Commands Reference Card

## 🚀 Essential Commands

### Initial Setup
```bash
# Clone repository
git clone https://github.com/Tannaz2001/ktc-eit-framework.git
cd ktc-eit-framework

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Run Benchmark

### Quick Test (5 min)
```bash
python run.py --config configs/ktc_all_methods.yaml
```
Edit `configs/ktc_all_methods.yaml` first:
```yaml
levels: [1]
samples: [A]
methods: [BackProjection]
```

### Full Benchmark (1 hour)
```bash
python run.py --config configs/ktc_all_methods.yaml
```
Make sure all levels/samples/methods are enabled in YAML

### Custom Configuration
```bash
# Create custom config
cat > configs/my_run.yaml << 'EOF'
data_plugin: KTCDataPlugin
dataset_root: EvaluationData
mesh_path: Codes_Matlab/Mesh_sparse.mat
levels: [1, 2]
samples: [A, B]
methods:
  - GaussNewton
  - BackProjection
output_dir: outputs_my_run/
EOF

# Run with it
python run.py --config configs/my_run.yaml
```

---

## 📊 View Results

### Interactive Dashboard (Recommended)
```bash
streamlit run app.py
```
Opens: `http://localhost:8501`

### View HTML Report
```bash
# macOS
open outputs/report.html

# Linux
xdg-open outputs/report.html

# Windows
start outputs/report.html

# Python server
cd outputs
python -m http.server 8000
# Visit: http://localhost:8000/report.html
```

### View Scores (JSON)
```bash
# Pretty print all scores
python -c "import json; print(json.dumps(json.load(open('outputs/scores.json')), indent=2))" | head -100

# Count runs
python -c "import json; print(f'Runs: {len(json.load(open(\"outputs/scores.json\")))}')"

# Get method averages
python -c "
import json
data = json.load(open('outputs/dashboard_scores.json'))
for m, metrics in sorted(data.items()):
    print(f'{m:30} KTC={metrics.get(\"ktc_score\", 0):.4f}')
"

# Get per-method detection rates
python -c "
import json
data = json.load(open('outputs/scores_nested.json'))
for method in data:
    qual = data[method].get('_qualitative_summary', {})
    res_pct = qual.get('resistive_detected_pct', 0)
    con_pct = qual.get('conductive_detected_pct', 0)
    print(f'{method:30} R={res_pct:.1f}% C={con_pct:.1f}%')
"
```

---

## 🧪 Run Tests

### All Tests
```bash
pytest tests/ -v
```

### Hull Plugin Tests Only
```bash
pytest tests/test_hull_plugin.py -v
```

### With Coverage
```bash
pytest tests/ --cov=src/ktc_framework --cov-report=html
open htmlcov/index.html
```

### Specific Test
```bash
pytest tests/test_hull_plugin.py::TestHullExtraction::test_circle_detection -v
```

---

## 🔧 Development

### Format Code
```bash
black src/ tests/
```

### Type Checking
```bash
mypy src/ktc_framework --ignore-missing-imports
```

### Lint
```bash
flake8 src/ --max-line-length=100
```

### Run Specific Method
```bash
python -c "
from pathlib import Path
from src.ktc_framework.runner.experiment_runner import BatchRunner

config = {
    'data_plugin': 'KTCDataPlugin',
    'dataset_root': 'EvaluationData',
    'mesh_path': 'Codes_Matlab/Mesh_sparse.mat',
    'levels': [1],
    'samples': ['A'],
    'methods': ['BackProjection'],
    'output_dir': 'outputs_test/'
}

runner = BatchRunner(config, Path('outputs_test'))
results = runner.run()
"
```

---

## 🌐 Streamlit Dashboard

### Launch
```bash
streamlit run app.py
```

### Custom Port
```bash
streamlit run app.py --server.port=8502
```

### Custom Host
```bash
streamlit run app.py --server.address=0.0.0.0
```

### Enable Remote Access
```bash
streamlit run app.py --server.enableXsrfProtection=false
```

---

## 📁 Project Structure

### View Files
```bash
# Tree structure
tree -L 2 src/ktc_framework/

# Key files
ls -la src/ktc_framework/runner/experiment_runner.py
ls -la src/ktc_framework/plugins/hull_plugin.py
ls -la src/ktc_framework/metrics/qualitative_metrics.py
```

### Check Config
```bash
# View active config
cat configs/ktc_all_methods.yaml

# List all configs
ls configs/*.yaml
```

### Check Output
```bash
# List all outputs
ls -lah outputs/

# Find latest report
ls -lt outputs/report.html | head -1

# Check file sizes
du -sh outputs/*
```

---

## 🔐 Environment Variables

### Set Variables
```bash
# Unix/Linux/macOS
export KTC_DATASET_ROOT=/path/to/EvaluationData
export ABC1_PYTHON=python3.12
export ABC1_SUBMISSION_PATH=/path/to/abc1

# Windows PowerShell
$env:KTC_DATASET_ROOT="C:\path\to\EvaluationData"
$env:ABC1_PYTHON="python3.12"
$env:ABC1_SUBMISSION_PATH="C:\path\to\abc1"

# Windows CMD
set KTC_DATASET_ROOT=C:\path\to\EvaluationData
set ABC1_PYTHON=python3.12
set ABC1_SUBMISSION_PATH=C:\path\to\abc1
```

### Check Variables
```bash
# Unix/Linux/macOS
echo $ABC1_PYTHON
env | grep ABC1

# Windows PowerShell
$env:ABC1_PYTHON
Get-ChildItem env: | grep ABC1

# Windows CMD
echo %ABC1_PYTHON%
set | findstr ABC1
```

---

## 🐛 Debugging

### Enable Verbose Output
```bash
# Streamlit debug mode
streamlit run app.py --logger.level=debug

# Python verbose
python -v run.py --config configs/ktc_all_methods.yaml
```

### Check Imports
```bash
python -c "from src.ktc_framework.runner.experiment_runner import BatchRunner; print('OK')"
python -c "from src.ktc_framework.plugins.hull_plugin import HullAnalyzer; print('OK')"
python -c "from src.ktc_framework.metrics.qualitative_metrics import aggregate_qualitative; print('OK')"
```

### Verify Data
```bash
python -c "
import scipy.io as io
import numpy as np

# Check ground truth
gt = io.loadmat('EvaluationData/GroundTruths/GTs_BINARY.mat')
print('GT keys:', gt.keys())
print('GT shape:', gt.get('GTs_BINARY', np.array([])).shape)

# Check level 1 data
data = io.loadmat('EvaluationData/level1/SampleA.mat')
print('Sample keys:', data.keys())
"
```

### Check Dependencies
```bash
python -m pip show numpy scipy scikit-image streamlit
pip list | grep -E "numpy|scipy|scikit|streamlit"
```

---

## 📦 Dependency Management

### View Requirements
```bash
cat requirements.txt
```

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Export Current Environment
```bash
pip freeze > requirements_current.txt
```

### Install from Scratch
```bash
# Remove old environment
rm -rf venv

# Create new
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1

# Install fresh
pip install -r requirements.txt
```

---

## 💾 Backup & Storage

### Backup Outputs
```bash
# Backup with timestamp
cp -r outputs outputs_backup_$(date +%Y%m%d_%H%M%S)

# Windows
xcopy outputs outputs_backup /E /I

# Archive
tar -czf outputs_backup.tar.gz outputs/
zip -r outputs_backup.zip outputs/
```

### Clean Up
```bash
# Remove old outputs (CAREFUL!)
rm -rf outputs/*

# Keep only report
rm -rf outputs/figures outputs/images outputs/mat_predictions outputs/overlays
```

### Check Disk Usage
```bash
# macOS/Linux
du -sh outputs/
du -sh outputs/*

# Windows
dir outputs /s
```

---

## 🚨 Common Issues & Fixes

### Port Already in Use
```bash
# Streamlit on different port
streamlit run app.py --server.port=8502

# Find process using port
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Memory Issues
```bash
# Reduce dataset size in config
# levels: [1]
# samples: [A]
# methods: [BackProjection]

# Monitor memory
top  # macOS/Linux
Task Manager  # Windows
```

### Import Errors
```bash
# Verify project root
pwd  # should end in /ktc-eit-framework

# Reinstall in editable mode
pip install -e .
```

---

## 📝 Useful One-Liners

```bash
# Count reconstructions completed
grep -c "BENCH_PROGRESS" console.log

# Get best method
python -c "import json; d=json.load(open('outputs/dashboard_scores.json')); 
print(max(d.items(), key=lambda x: x[1].get('ktc_score', 0)))"

# Compare two runs
diff outputs_run1/scores.json outputs_run2/scores.json

# Monitor progress
watch 'tail -10 console.log'

# Get runtime stats
python -c "import json; d=json.load(open('outputs/scores.json'));
print(f'Total time: {sum(r[\"runtime_ms\"] for r in d)/1000:.1f}s')"

# List all methods that ran
python -c "import json; d=json.load(open('outputs/scores.json'));
print(sorted(set(r['method'] for r in d)))"

# Get detection rates
python -c "import json; d=json.load(open('outputs/scores_nested.json'));
[(print(f'{m}: {d[m].get(\"_qualitative_summary\", {}).get(\"resistive_detected_pct\", 0):.1f}%')) 
 for m in d if '_qualitative_summary' in d[m]]"
```

---

## 📖 Documentation Files

```bash
# Read guides
cat README.md
cat RUN_GUIDE.md
cat FRICTION_REPORT.md
cat PLUGINS.md

# View constraints
cat constraint.txt
```

---

## 🎓 Learning Resources

```bash
# Explore source code
ls -la src/ktc_framework/

# Read docstrings
python -c "from src.ktc_framework.plugins.hull_plugin import HullAnalyzer; help(HullAnalyzer)"

# View test examples
cat tests/test_hull_plugin.py | head -100

# Check configs
ls configs/
cat configs/ktc_all_methods.yaml
```

---

Last updated: 2026-06-18
