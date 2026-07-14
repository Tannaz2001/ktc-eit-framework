#!/bin/bash
# Download KTC 2023 Evaluation Dataset
#
# This script provides instructions for downloading the official KTC 2023 dataset.
# The dataset is NOT downloaded automatically because:
# - It's large (~2-5 GB per difficulty level)
# - It requires registration/authentication
# - License terms must be reviewed
#
# Usage:
#   bash scripts/download_ktc_dataset.sh
#   or just read the instructions below

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATION_DATA_DIR="${PROJECT_ROOT}/EvaluationData"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  KTC 2023 Evaluation Dataset Download Instructions                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "STEP 1: Register at the KTC 2023 Challenge"
echo "  URL: https://ktc2023.uta.fi/"
echo "  - Create an account"
echo "  - Accept license terms"
echo "  - Download 'Evaluation Data' (all 7 levels)"
echo ""
echo "STEP 2: Extract to project root"
echo "  The extracted folder should be placed at:"
echo "  ${EVALUATION_DATA_DIR}/"
echo ""
echo "  After extraction, your structure should look like:"
echo "  ${EVALUATION_DATA_DIR}/"
echo "  ├── evaluation_datasets/"
echo "  │   ├── level1/"
echo "  │   │   ├── data1.mat"
echo "  │   │   ├── data2.mat"
echo "  │   │   ├── data3.mat"
echo "  │   │   └── ref.mat"
echo "  │   ├── level2/ ... level7/"
echo "  │   └── GroundTruths/"
echo "  │       └── GroundTruths.mat"
echo ""
echo "STEP 3: Verify dataset is present"
if [ -d "${EVALUATION_DATA_DIR}/evaluation_datasets/level1" ]; then
    echo "✓ EvaluationData/evaluation_datasets/level1 found"
    ls -la "${EVALUATION_DATA_DIR}/evaluation_datasets/level1" | head -6
else
    echo "✗ EvaluationData/evaluation_datasets/level1 NOT FOUND"
    echo "  Please download and extract the dataset to: ${EVALUATION_DATA_DIR}/"
fi

if [ -f "${EVALUATION_DATA_DIR}/evaluation_datasets/GroundTruths/GroundTruths.mat" ]; then
    echo "✓ GroundTruths found"
else
    echo "✗ GroundTruths NOT FOUND"
fi

echo ""
echo "STEP 4: Run the benchmark"
echo "  Once the dataset is in place, run:"
echo "  $ python run.py --config configs/ktc_all_methods.yaml"
echo ""
