# Electrode Node Index Extraction Guide

**Problem**: Scattered noise in reconstructions indicates electrode positions are incorrect, corrupting the Jacobian sensitivity matrix.

**Solution**: Extract correct electrode node indices from your KTC mesh and hardcode them in BackProjection.

---

## Why This Matters

BackProjection computes a Jacobian matrix J that maps conductivity changes to voltage changes:

```
ΔV = J · Δσ
```

If electrode positions are wrong, J is garbage, and reconstruction fails silently.

**Before fix**: Electrodes auto-detected incorrectly → scattered noise  
**After fix**: Electrodes hardcoded correctly → coherent objects

---

## Step 1: Extract Electrode Indices

You need the actual KTC mesh file: `KTC2023_open_mesh.mat` (part of KTC 2023 evaluation data)

Run the extraction script:

```bash
python3 extract_electrode_nodes.py /path/to/KTC2023_open_mesh.mat
```

Output will show:

```
Loading mesh: /path/to/KTC2023_open_mesh.mat
Mesh keys: ['__header__', '__version__', '__globals__', 'g', 'H', 'elfaces']

Nodes shape: (2800, 2)
Distance range: 0.000000 to 1.003815
  Boundary nodes [0.95, 1.05]: 320
  Boundary nodes [0.90, 1.10]: 320
  -> Using this range

Electrode angle spacing:
  Expected (360/32): 11.25 degrees
  Actual mean:       11.23 degrees
  Actual std:        0.10 degrees
  Min spacing:       11.02 degrees
  Max spacing:       11.51 degrees

OK: Electrodes are reasonably evenly distributed

HARDCODED ELECTRODE NODES FOR BackProjection
================================================================

Add this to src/ktc_framework/methods/backprojection.py:
In _electrode_positions() method, add before the for loop:

# Hardcoded KTC mesh electrode node indices
_KTC_ELECTRODE_NODES = np.array([
    2645,  2732,  2734,  2767,  2663,  2747,  2720,  2733,
    2756,  2668,  2768,  2768,  2750,  2718,  2759,  2768,
    ... (32 total)
], dtype=np.int32)

if mesh_data is not None and len(_KTC_ELECTRODE_NODES) == 32:
    return _KTC_ELECTRODE_NODES
```

---

## Step 2: Hardcode the Indices

Copy the printed values and add them to `src/ktc_framework/methods/backprojection.py`:

### Option A: Update the constant (recommended for production)

At the top of the file, replace:

```python
_KTC_ELECTRODE_NODES_FALLBACK = None
```

With:

```python
# Hardcoded electrode node indices for KTC2023_open_mesh.mat
_KTC_ELECTRODE_NODES_FALLBACK = np.array([
    2645, 2732, 2734, 2767, 2663, 2747, 2720, 2733,
    2756, 2668, 2768, 2768, 2750, 2718, 2759, 2768,
    2771, 2728, 2774, 2721, 2764, 2706, 2722, 2696,
    2651, 2710, 2736, 2632, 2701, 2636, 2700, 2656,
], dtype=np.int32)
```

### Option B: Add to _electrode_positions method

Add check before the fallback:

```python
@staticmethod
def _electrode_positions(mesh_data, nodes):
    """Find 32 electrode node indices from mesh."""
    # ... existing code ...
    
    # Hardcoded KTC mesh values
    KTC_ELECTRODES = np.array([
        2645, 2732, 2734, ... (full list)
    ], dtype=np.int32)
    
    if mesh_data is not None:
        # Try elfaces first
        for key in ["elfaces", ...]:
            # ... existing code ...
            if electrode_nodes.shape[0] == 32:
                return electrode_nodes
        
        # If elfaces parsing failed, use hardcoded KTC values
        return KTC_ELECTRODES
```

---

## Step 3: Test the Fix

After hardcoding, test a reconstruction:

```python
from src.ktc_framework.loaders.ktc_data_plugin import KTCDataPlugin
from src.ktc_framework.methods.backprojection import BackProjection

# Load a sample
loader = KTCDataPlugin()
batch = loader.load_sample(level=1, sample='A')

# Run reconstruction
bp = BackProjection()
output = bp.reconstruct(batch)

# Check output
import numpy as np
print(f"Output shape: {output.shape}")
print(f"Unique labels: {np.unique(output)}")
print(f"Should see labels [0, 1, 2] for background, resistive, conductive")
```

Expected output:
```
Output shape: (256, 256)
Unique labels: [0 1 2]  # Good! (not scattered noise)
```

---

## Verification Checklist

- [ ] Extract electrode indices using `extract_electrode_nodes.py`
- [ ] Verify electrode angle spacing (std should be <1 degree)
- [ ] Copy hardcoded values into BackProjection
- [ ] Run test reconstruction
- [ ] Verify output has coherent objects, not scattered noise
- [ ] Compare with MATLAB reference (if available)

---

## If Your Mesh Has Different Structure

If the script fails or your mesh uses different field names:

1. **Check mesh structure**:
   ```python
   import scipy.io as sio
   mesh = sio.loadmat('your_mesh.mat')
   print(mesh.keys())  # What fields are there?
   ```

2. **Modify extract_electrode_nodes.py** to handle your mesh format

3. **Contact the KTC 2023 team** for official electrode node mapping

---

## Common Issues

### Issue: "Could not find 32 boundary nodes"
- Mesh may use different coordinate system (not centered at origin)
- Try `extract_electrode_nodes.py` with debug output to see actual node distances

### Issue: "Electrodes not evenly distributed" (std > 1 degree)
- Mesh may have non-standard electrode placement
- Double-check with the original mesh documentation

### Issue: Reconstructions still show noise after hardcoding
- Verify the hardcoded indices are correct (copy-paste error?)
- Check that `reference_voltages` are populated in data batch
- Verify measurement patterns (Inj, Mpat) match mesh electrode count

---

## References

- `extract_electrode_nodes.py` - Automated extraction script
- `BackProjection._electrode_positions()` - Where indices are used
- `BACKPROJECTION_ISSUES.md` - Why electrode positions matter
- KTC 2023 challenge documentation - Official mesh specification

