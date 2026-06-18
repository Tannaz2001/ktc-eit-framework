@echo off
REM Quick Start Guide — KTC EIT Benchmarking Framework (Windows)

echo.
echo ========================================================================
echo   KTC EIT Benchmarking Framework - Quick Start
echo ========================================================================
echo.

REM ========================================================================
REM STEP 1: Clone and Setup
REM ========================================================================

echo STEP 1: Clone Repository
echo ────────────────────────────────────────────────────────────────
git clone https://github.com/Tannaz2001/ktc-eit-framework.git
cd ktc-eit-framework

echo.
echo STEP 2: Create Virtual Environment
echo ────────────────────────────────────────────────────────────────
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo STEP 3: Install Dependencies
echo ────────────────────────────────────────────────────────────────
python -m pip install --upgrade pip
pip install -r requirements.txt

REM ========================================================================
REM STEP 2: Verify Data
REM ========================================================================

echo.
echo STEP 4: Verify Dataset
echo ────────────────────────────────────────────────────────────────
if exist "EvaluationData\level1" (
    echo ✓ EvaluationData found
) else (
    echo ⚠ EvaluationData missing - download from KTC 2023
)

if exist "Codes_Matlab\Mesh_sparse.mat" (
    echo ✓ Mesh found
) else (
    echo ⚠ Mesh missing - place in project root
)

REM ========================================================================
REM STEP 3: Run Tests (Optional)
REM ========================================================================

echo.
echo STEP 5: Run Tests (Optional)
echo ────────────────────────────────────────────────────────────────
echo Command: pytest tests/ -v
echo Expected: 12/12 passing

REM ========================================================================
REM STEP 4: Run Benchmark
REM ========================================================================

echo.
echo STEP 6: Run Benchmark
echo ────────────────────────────────────────────────────────────────
echo Quick Test (1 level, 2 methods - ~1 min):
echo   python run.py --config configs/ktc_all_methods.yaml
echo.
echo Full Benchmark (7 levels, 6 methods - ~1 hour):
echo   Edit configs/ktc_all_methods.yaml then:
echo   python run.py --config configs/ktc_all_methods.yaml

REM ========================================================================
REM STEP 5: View Results
REM ========================================================================

echo.
echo STEP 7: View Results
echo ────────────────────────────────────────────────────────────────
echo Option A: Interactive Dashboard (Recommended)
echo   streamlit run app.py
echo   Browser: http://localhost:8501
echo.
echo Option B: HTML Report
echo   start outputs\report.html
echo.
echo Option C: View Scores
echo   type outputs\scores.json

REM ========================================================================
REM Complete
REM ========================================================================

echo.
echo ========================================================================
echo   Setup Complete! Run: streamlit run app.py
echo ========================================================================
echo.

pause
