#!/bin/bash
# Quick Start Guide — KTC EIT Benchmarking Framework
# Copy and paste these commands to run the project

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  KTC EIT Benchmarking Framework — Quick Start                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ════════════════════════════════════════════════════════════════════
# STEP 1: Clone and Setup
# ════════════════════════════════════════════════════════════════════

echo "STEP 1: Clone Repository"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git clone https://github.com/Tannaz2001/ktc-eit-framework.git
cd ktc-eit-framework

echo ""
echo "STEP 2: Create Virtual Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m venv venv

# Activate (choose one for your OS)
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1

# Windows CMD:
# venv\Scripts\activate.bat

# macOS/Linux:
# source venv/bin/activate

echo "⚠️  Activate virtual environment manually:"
echo "   Windows PowerShell: .\venv\Scripts\Activate.ps1"
echo "   Windows CMD:        venv\Scripts\activate.bat"
echo "   macOS/Linux:        source venv/bin/activate"
echo ""

echo "STEP 3: Install Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install --upgrade pip
pip install -r requirements.txt

# ════════════════════════════════════════════════════════════════════
# STEP 2: Verify Data
# ════════════════════════════════════════════════════════════════════

echo ""
echo "STEP 4: Verify Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -la EvaluationData/level1/ 2>/dev/null && echo "✓ EvaluationData found" || echo "⚠️  EvaluationData missing"
ls -la Codes_Matlab/Mesh_sparse.mat 2>/dev/null && echo "✓ Mesh found" || echo "⚠️  Mesh missing"

echo ""
echo "⚠️  If data is missing:"
echo "   1. Download KTC 2023 Dataset"
echo "   2. Place in project root:"
echo "      EvaluationData/"
echo "      Codes_Matlab/Mesh_sparse.mat"

# ════════════════════════════════════════════════════════════════════
# STEP 3: Run Tests (Optional)
# ════════════════════════════════════════════════════════════════════

echo ""
echo "STEP 5: Run Tests (Optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "pytest tests/ -v"
echo "(Expected: 12/12 passing)"

# ════════════════════════════════════════════════════════════════════
# STEP 4: Run Benchmark
# ════════════════════════════════════════════════════════════════════

echo ""
echo "STEP 6: Run Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Quick Test (1 level, 2 methods, ~1 min):"
echo "  python run.py --config configs/ktc_all_methods.yaml"

echo ""
echo "Full Benchmark (7 levels, 6 methods, ~1 hour):"
echo "  python run.py --config configs/ktc_all_methods.yaml"

# ════════════════════════════════════════════════════════════════════
# STEP 5: View Results
# ════════════════════════════════════════════════════════════════════

echo ""
echo "STEP 7: View Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Option A: Interactive Dashboard (Recommended)"
echo "  streamlit run app.py"
echo "  Browser: http://localhost:8501"
echo ""

echo "Option B: HTML Report"
echo "  open outputs/report.html"
echo ""

echo "Option C: View Scores"
echo "  cat outputs/scores.json | python -m json.tool | head -50"

# ════════════════════════════════════════════════════════════════════
# Complete
# ════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete! Run: streamlit run app.py                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
