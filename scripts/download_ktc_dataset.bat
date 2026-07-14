@echo off
REM Download KTC 2023 Evaluation Dataset Instructions
REM
REM This script provides instructions for downloading the official KTC 2023 dataset.
REM The dataset is NOT downloaded automatically because:
REM - It's large (~2-5 GB per difficulty level)
REM - It requires registration/authentication
REM - License terms must be reviewed
REM
REM Usage:
REM   download_ktc_dataset.bat
REM   or just read the instructions below

setlocal enabledelayedexpansion

REM Get project root (parent directory of scripts/)
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~dpI"
set "EVALUATION_DATA_DIR=%PROJECT_ROOT%EvaluationData"

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════════
echo   KTC 2023 Evaluation Dataset Download Instructions
echo ════════════════════════════════════════════════════════════════════════════════
echo.
echo STEP 1: Register at the KTC 2023 Challenge
echo   URL: https://ktc2023.uta.fi/
echo   - Create an account
echo   - Accept license terms
echo   - Download 'Evaluation Data' (all 7 levels)
echo.
echo STEP 2: Extract to project root
echo   The extracted folder should be placed at:
echo   %EVALUATION_DATA_DIR%\
echo.
echo   After extraction, your structure should look like:
echo   %EVALUATION_DATA_DIR%\
echo   - evaluation_datasets\
echo     - level1\ (contains data1.mat, data2.mat, data3.mat, ref.mat)
echo     - level2\ ... level7\
echo     - GroundTruths\
echo       - GroundTruths.mat
echo.
echo STEP 3: Verify dataset is present
if exist "%EVALUATION_DATA_DIR%\evaluation_datasets\level1" (
    echo ✓ EvaluationData\evaluation_datasets\level1 found
    dir "%EVALUATION_DATA_DIR%\evaluation_datasets\level1"
) else (
    echo ✗ EvaluationData\evaluation_datasets\level1 NOT FOUND
    echo   Please download and extract the dataset to: %EVALUATION_DATA_DIR%\
)

if exist "%EVALUATION_DATA_DIR%\evaluation_datasets\GroundTruths\GroundTruths.mat" (
    echo ✓ GroundTruths found
) else (
    echo ✗ GroundTruths NOT FOUND
)

echo.
echo STEP 4: Run the benchmark
echo   Once the dataset is in place, run:
echo   ^> python run.py --config configs/ktc_all_methods.yaml
echo.
pause
