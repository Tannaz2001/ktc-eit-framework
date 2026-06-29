# Architecture Gap: ML Method Upload

## Current Reality

### In-Process Methods (✅ Frictionless)
- **BackProjection, GaussNewton, LinearDifference**: Pure Python, import directly
- Flow: Upload `.py` → `importlib.exec_module()` → `@register_method` → runs in-process
- **Works because**: Same Python interpreter, same `sys.path`, same environment
- **Upload complexity**: Just the `.py` file

### Out-of-Process Methods (❌ Manual Glue)
- **CompetitionCNN**: TensorFlow solver + weights file + subprocess wrapper
- Flow: 
  1. User hardcodes ABC1 path detection (`_find_submission_dir()`)
  2. User writes Python launcher (`_ABC1_PYTHON = py_launcher("-3.11")`)
  3. User builds temp-file I/O protocol (input `.mat` → output `.mat`)
  4. User implements subprocess call + error handling + cleanup
  5. User handles timeout (5 min), return shape validation, logging
- **Manual steps in competition_cnn.py**: 200+ lines for solver discovery, subprocess, I/O glue
- **Upload complexity**: Framework must know about solver location, weights path, expected I/O format

---

## The Gap

| Aspect | In-Process | Out-of-Process |
|--------|-----------|-----------------|
| **What user uploads** | `.py` with classes | `.py` + solver + weights + wrapper config |
| **Python detection** | Automatic (same process) | Manual search (`_find_submission_dir()`, `_has_tensorflow()`) |
| **Dependencies** | Assumed available in venv | Must find separate interpreter |
| **Data I/O** | Python objects in memory | Temp files, .mat, JSON, etc. |
| **Subprocess wrapper** | None needed | User must write it |
| **Error handling** | Implicit (Python exceptions) | Manual: timeout, missing files, shape validation |
| **Frictionless upload?** | ✅ Yes | ❌ No |

---

## What Users Actually Need for ML Methods

Currently, to add a new ML solver like `MyNewCNN`, a user must:

1. **Place solver directory** somewhere discoverable:
   ```
   external_methods/my_cnn/
   ├── solver.py         (entry point)
   ├── model.h5          (weights, 100+ MB)
   └── requirements.txt  (tensorflow==2.15, etc)
   ```

2. **Write discovery logic**:
   ```python
   def _find_my_cnn() -> Optional[Path]:
       candidates = [...]  # search multiple paths
       for path in candidates:
           if (path / "solver.py").exists():
               return path
   ```

3. **Write Python detection**:
   ```python
   _MY_CNN_PYTHON = _find_python_with_tf()  # what TensorFlow version? PyTorch?
   ```

4. **Write I/O bridge** (input format? output format?):
   ```python
   tmp_input = tempfile.mkdtemp()
   # How to pass data to solver.py?
   subprocess.run([_MY_CNN_PYTHON, "solver.py", tmp_input, tmp_output])
   ```

5. **Write 50+ lines** of error handling, validation, logging

---

## The Real Solution: Manifest-Based Method Registry

### User Perspective

User uploads a **method bundle**:
```
my_cnn.tar.gz
├── method.yaml          ← MANIFEST (user-filled template)
├── solver.py
├── model.h5
└── requirements.txt
```

**method.yaml** (standardized, but user-filled):
```yaml
name: MyNewCNN
description: "Post-processing CNN for EIT segmentation"

# Runtime environment
python_version: "3.11"
dependencies:
  - tensorflow==2.15
  - scipy

# Solver interface
solver:
  entry_point: "solver.py"
  entry_function: "reconstruct"  # solve(voltages, injection_patterns) -> ndarray
  
# Data I/O contract
input_format: "python_dict"     # {voltages, injection_patterns, level, sample_id}
output_format: "ndarray"         # shape (256, 256), dtype uint8

# Resource constraints
timeout_seconds: 300
weights_file: "model.h5"

# Optional: subprocess if in-process not possible
subprocess: false               # true = auto-wrap in subprocess
```

### Framework Perspective

Framework auto-generates wrapper:

```python
@register_method
class MyNewCNN(MethodPlugin):
    def __init__(self):
        self.manifest = load_manifest("method.yaml")
        self.python_interpreter = find_or_create_venv(
            name="my_new_cnn",
            python_version=self.manifest.python_version,
            dependencies=self.manifest.dependencies
        )
        self.weights = Path(self.manifest.weights_file)
    
    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        if self.manifest.subprocess:
            # Auto-generated subprocess wrapper
            return self._run_subprocess(batch)
        else:
            # Direct import (if pure Python)
            return self._run_in_process(batch)
    
    def _run_subprocess(self, batch):
        # Auto-generated I/O bridge based on manifest
        tmp_input = tempfile.mkdtemp()
        tmp_output = tempfile.mkdtemp()
        
        # Serialize input according to manifest
        input_data = self.manifest.serialize_input({
            "voltages": batch.voltages,
            "injection_patterns": batch.injection_patterns,
            ...
        })
        Path(tmp_input / "input.pkl").write_bytes(input_data)
        
        # Call solver with manifest-specified interpreter
        result = subprocess.run(
            [self.python_interpreter, "solver.py", tmp_input, tmp_output],
            timeout=self.manifest.timeout_seconds,
            ...
        )
        
        # Deserialize output according to manifest
        output_data = Path(tmp_output / "output.pkl").read_bytes()
        reconstruction = self.manifest.deserialize_output(output_data)
        
        # Validate shape according to manifest
        assert reconstruction.shape == self.manifest.output_shape
        
        return reconstruction
```

---

## Implementation Roadmap

### Sprint 7 (Current)
- ✅ Understand gap
- ✅ Verify design
- [ ] Document manifest spec (JSON schema)

### Sprint 8
- [ ] `MethodManifest` dataclass + validation
- [ ] `VenvManager`: find or create isolated Python env with `pip install`
- [ ] Auto-wrapper generator: inspect manifest → generate `_SubprocessMethodWrapper`
- [ ] Manifest upload UI in Streamlit (unzip, validate, register)

### Sprint 9
- [ ] Bundled method registry (marketplace?)
- [ ] Solver compatibility test harness
- [ ] Documentation + user template

---

## Why This Works

| Aspect | Before | After |
|--------|--------|-------|
| **User burden** | Write 200 lines of subprocess code | Fill 10-line YAML template |
| **Discovery** | Framework searches hardcoded paths | Manifest declares solver location |
| **Dependencies** | User figures out Python version | Manifest declares, framework isolates |
| **I/O contract** | User writes custom serialization | Manifest specifies, framework generates |
| **Error handling** | User catches timeout, validation | Framework handles (declared in manifest) |
| **Testability** | Bespoke per-method | Standardized per-manifest spec |

---

## What Stays Manual

- **Solver algorithm** (obviously)
- **Weights training** (outside framework scope)
- **Hyperparameter tuning** (user's responsibility)
- **Benchmark selection** (which methods to run together)

---

## Next Steps

1. **Design manifest schema** — JSON schema file
2. **Prototype `MethodManifest`** class
3. **Write VenvManager** stub
4. **Update upload flow** to accept `.tar.gz` bundles + manifest validation
5. **Test with ABC1** — wrap existing ABC1 wrapper with manifest (verify it works)

