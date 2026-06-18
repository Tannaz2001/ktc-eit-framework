# Plugin Architecture: Adding Competition Submissions

This framework supports pluggable reconstruction methods. Each method integrates as a `MethodPlugin` subclass that wraps untouched competition code, allowing algorithms to be benchmarked without modification.

## Directory Structure

```
KTC_WORK_HIS/
├── src/ktc_framework/methods/
│   ├── backprojection.py          # Built-in method
│   ├── gauss_newton.py            # Built-in method
│   ├── competition_cnn.py         # ABC1 (UFABC) competition submission wrapper
│   └── [your_method].py           # Your plugin here
│
├── src/ktc_framework/models/abc1/
│   ├── __init__.py
│   ├── solver.py                  # Competition code (ABC1) — DO NOT EDIT
│   ├── ultimate_cnn1.h5           # Pre-trained weights
│   └── MiscCodes/                 # Supporting libraries (untouched)
│
├── KTC2023-ABC1/                  # Competition submission repo (git clone)
│   ├── KTC2023_Python_A01+/
│   │   ├── main_python.py
│   │   ├── Mesh_sparse.mat
│   │   ├── MiscCodes/
│   │   ├── TrainingData/ref.mat
│   │   └── ultimate_cnn1.h5
│   └── README.md
│
└── configs/
    └── ktc_all_methods.yaml       # Register your method here
```

## How Plugins Work

### Pattern: Subprocess Isolation

Competition code runs in an **isolated subprocess**. The wrapper:
1. Receives a `DataBatch` (voltages, level, sample_id)
2. Locates the raw measurement file from the framework's dataset
3. Calls `subprocess.run(["python", "main_competition.py", input_dir, output_dir, level])`
4. Reads back the reconstruction from `output_dir/*.mat`
5. Returns a `(256, 256)` uint8 array

This approach:
- **Preserves immutability** — competition code untouched
- **Isolates dependencies** — TensorFlow, PyTorch, etc. don't conflict
- **Allows relocation** — no hardcoded absolute paths
- **Works in Docker** — portable across machines

### Path Discovery (No Hardcoding)

The wrapper searches for the submission directory in this order:

1. **Relative to framework root**
   ```
   Framework/../KTC2023-ABC1/KTC2023_Python_A01+
   ```

2. **Sibling directory** (if framework is in a subdirectory)
   ```
   Framework/KTC2023-ABC1/KTC2023_Python_A01+
   ```

3. **Environment variable** (highest priority for customization)
   ```bash
   export ABC1_SUBMISSION_PATH=/path/to/KTC2023-ABC1/KTC2023_Python_A01+
   ```

4. **CWD** (for Docker volumes, alternate layouts)

This means:
- Clone the framework anywhere
- Clone `KTC2023-ABC1` as a sibling directory
- Everything just works — no config files needed
- Users on different machines can set `$ABC1_SUBMISSION_PATH` if needed

### Python Interpreter Selection

Similarly, the Python version is discovered at runtime:

1. **Environment variable** (user override)
   ```bash
   export ABC1_PYTHON=/usr/bin/python3.12
   ```

2. **Current interpreter** (default)
   - The same Python running the framework
   - If TensorFlow is installed, it runs there
   - If not, the wrapper logs a clear error

This avoids machine-specific hardcoding while allowing power users to specify a custom Python if needed.

---

## How to Add Your Own Competition Entry

### Step 1: Set Up the Submission Directory

Clone or place your competition repo as a sibling to the framework:

```bash
cd /path/to/parent
git clone https://github.com/yourteam/KTC2023-YourEntry.git
cd KTC2023-YourEntry/YourEntryPythonFolder

# Verify structure:
# - main_python.py          (CLI: python main_python.py input output level)
# - solver.py or equivalent (core algorithm)
# - supporting files/data
```

### Step 2: Create a Wrapper

In `src/ktc_framework/methods/your_method.py`:

```python
"""Wraps YourEntry KTC2023 submission."""

from __future__ import annotations
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.ktc_framework.adapters.method_registry import register
from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch

_logger = logging.getLogger(__name__)

def _find_submission_dir() -> Optional[Path]:
    """Find YourEntry submission directory."""
    framework_root = Path(__file__).resolve().parents[3]
    candidates = [
        framework_root.parent / "KTC2023-YourEntry" / "YourEntryPythonFolder",
        Path(os.environ.get("YOURENTRY_SUBMISSION_PATH", "")),
        Path.cwd() / "KTC2023-YourEntry" / "YourEntryPythonFolder",
    ]
    for path in candidates:
        if path and (path / "main_python.py").exists():
            return path
    _logger.warning(f"YourEntry submission not found. Set $YOURENTRY_SUBMISSION_PATH")
    return None

def _find_python_interpreter() -> str:
    """Find Python interpreter (user can override with env var)."""
    if "YOURENTRY_PYTHON" in os.environ:
        return os.environ["YOURENTRY_PYTHON"]
    return sys.executable

_SUBMISSION_CWD = _find_submission_dir()
_SUBMISSION_MAIN = str((_SUBMISSION_CWD / "main_python.py")) if _SUBMISSION_CWD else None
_PYTHON = _find_python_interpreter()

@register
class YourEntry(MethodPlugin):
    """Wraps your KTC2023 competition entry."""

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        if _SUBMISSION_CWD is None:
            _logger.error("YourEntry submission directory not found")
            return np.zeros((256, 256), dtype=np.uint8)

        tmp_input = None
        tmp_output = None
        try:
            # Find data file using same pattern as ABC1
            sample_id = str(getattr(batch, "sample_id", ""))
            level = int(batch.level)
            letter = sample_id.split("_", 1)[1].upper() if "_" in sample_id else "A"
            file_num = {"A": "1", "B": "2", "C": "3"}.get(letter, "1")

            framework_root = Path(__file__).resolve().parents[3]
            for root in [framework_root, Path.cwd()]:
                for subdir in ["EvaluationData/evaluation_datasets", "evaluation_datasets"]:
                    data_file = root / subdir / f"level{level}" / f"data{file_num}.mat"
                    if data_file.exists():
                        break
                else:
                    continue
                break
            else:
                _logger.error(f"Data file not found for level={level} sample={file_num}")
                return np.zeros((256, 256), dtype=np.uint8)

            # Run submission in isolated subprocess
            tmp_input = tempfile.mkdtemp(prefix="your_in_")
            tmp_output = tempfile.mkdtemp(prefix="your_out_")

            import shutil
            shutil.copy2(str(data_file), os.path.join(tmp_input, data_file.name))

            cmd = [_PYTHON, _SUBMISSION_MAIN, tmp_input, tmp_output, str(level)]
            result = subprocess.run(cmd, cwd=str(_SUBMISSION_CWD), 
                                   capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                _logger.error(f"YourEntry failed:\n{result.stderr[-1000:]}")
                return np.zeros((256, 256), dtype=np.uint8)

            # Read output
            import scipy.io
            output_files = [f for f in os.listdir(tmp_output) if f.endswith(".mat")]
            if not output_files:
                _logger.error("No output .mat file produced")
                return np.zeros((256, 256), dtype=np.uint8)

            mat = scipy.io.loadmat(os.path.join(tmp_output, output_files[0]))
            reconstruction = np.asarray(mat.get("reconstruction", np.zeros((256, 256))), 
                                       dtype=np.uint8)

            if reconstruction.shape != (256, 256):
                _logger.error(f"Shape mismatch: {reconstruction.shape}")
                return np.zeros((256, 256), dtype=np.uint8)

            self.validate_output(reconstruction)
            return reconstruction

        except subprocess.TimeoutExpired:
            _logger.error("YourEntry timed out")
            return np.zeros((256, 256), dtype=np.uint8)
        except Exception as exc:
            _logger.error(f"YourEntry error: {exc}", exc_info=True)
            return np.zeros((256, 256), dtype=np.uint8)
        finally:
            for d in [tmp_input, tmp_output]:
                if d and os.path.isdir(d):
                    try:
                        shutil.rmtree(d)
                    except:
                        pass
```

### Step 3: Register in Framework

1. **Add import** in `src/ktc_framework/methods/__init__.py`:
   ```python
   from src.ktc_framework.methods.your_method import YourEntry  # noqa: F401
   ```

2. **Add to config** in `configs/ktc_all_methods.yaml`:
   ```yaml
   methods:
     - BackProjection
     - GaussNewton
     - CompetitionCNN
     - YourEntry  # ← Add your method here
   ```

### Step 4: Run Benchmark

```bash
# Run all methods including your entry
python example_usage.py --config configs/ktc_all_methods.yaml

# Or just your method
python example_usage.py --config configs/ktc_all_methods.yaml --methods YourEntry
```

---

## Environment Setup for Users

For users who don't have TensorFlow installed, or need a specific Python version:

```bash
# Create a Python 3.12 environment (if not present)
conda create -n abc1 python=3.12
conda activate abc1
pip install tensorflow opencv-python scikit-image scipy

# Set environment variables
export ABC1_PYTHON=/path/to/abc1/bin/python
export ABC1_SUBMISSION_PATH=/path/to/KTC2023-ABC1/KTC2023_Python_A01+

# Run benchmark
python example_usage.py --config configs/ktc_all_methods.yaml
```

Or for Docker:

```dockerfile
FROM python:3.12

# Install TensorFlow and dependencies
RUN pip install tensorflow opencv-python scikit-image scipy matplotlib

# Copy framework
COPY . /app
WORKDIR /app

# Submission should be mounted or cloned at runtime
ENV ABC1_SUBMISSION_PATH=/submissions/KTC2023-ABC1/KTC2023_Python_A01+

CMD ["python", "example_usage.py", "--config", "configs/ktc_all_methods.yaml"]
```

Then:

```bash
docker run -v /path/to/KTC2023-ABC1:/submissions:ro ktc-benchmark
```

---

## Key Principles

1. **Zero modifications to competition code** — `main_python.py`, `solver.py`, weights stay pristine
2. **Dynamic discovery** — no hardcoded absolute paths; environment variables for customization
3. **Graceful degradation** — missing submissions log clear errors, benchmark continues
4. **Isolated execution** — subprocess keeps dependencies from conflicting
5. **Portable** — works after `git clone`, in Docker, on any machine
