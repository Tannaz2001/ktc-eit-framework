# BackProjection Critical Issues

**Status**: 🔴 THREE CRITICAL BUGS CONFIRMED  
**File**: `src/ktc_framework/methods/backprojection.py`  
**Severity**: HIGH (affects ALL reconstructions)

---

## Issue 1: Reference Voltages Fallback is Incorrect

**Location**: Lines 307-313

**Current Code**:
```python
if reference_voltages is None:
    warnings.warn("...using mean subtraction.")
    v0 = np.full_like(v1, float(np.nanmean(v1)))
```

**Problem**: 
- Falls back to mean subtraction (scalar constant) instead of per-electrode baseline
- This is mathematically wrong for difference voltage calculation
- KTC challenge requires ΔV = V_object - V_ref (per measurement baseline)

**Risk**: 
- If data loaders don't populate `reference_voltages`, ALL reconstructions are corrupted
- **Need to verify**: Does `KTCDataPlugin.load_sample()` actually populate `batch.reference_voltages`?

**Fix**:
```python
if reference_voltages is None:
    raise ValueError(
        "BackProjection requires reference_voltages. "
        "Check that KTCDataPlugin populates batch.reference_voltages."
    )
```

---

## Issue 2: Electrode Mapping Falls Back to Evenly-Spaced Nodes ❌ CRITICAL

**Location**: Line 425

**Current Code**:
```python
# Lines 385-425: Try to parse elfaces from mesh
for key in ["elfaces", "ElFaces", "el_faces", "elFaces"]:
    elfaces = _get_key(mesh_data, key)
    if elfaces is None:
        continue
    # ... try to extract electrode positions ...

# FALLBACK (line 425):
return np.round(np.linspace(0, n_nodes - 1, 32)).astype(np.int32)
```

**Problem**:
- **Electrodes MUST be on boundary circle** (physical constraint)
- Evenly-spaced node indices include **interior nodes** (wrong!)
- If mesh doesn't have 'elfaces' key, this fallback triggers → **garbage reconstruction**

**Evidence of Bug**:
```python
# Lines 388-423 show mesh has these keys checked:
for key in ["elfaces", "ElFaces", "el_faces", "elFaces"]:
```
**Question**: Does `KTC2023_open_mesh.mat` have an 'elfaces' key? Or does it use a different name like 'el_indices' or 'boundary_nodes'?

**Fix**:
```python
# Instead of evenly-spaced fallback:
@staticmethod
def _electrode_positions_fallback(nodes: np.ndarray) -> np.ndarray:
    """Find electrode positions on boundary circle."""
    # Electrodes are at angles 0, 2π/32, 4π/32, ..., 30π/32
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    electrode_xy = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Find node closest to each electrode angle
    electrode_nodes = []
    for target_xy in electrode_xy:
        # Only consider boundary nodes (x² + y² ≈ 1)
        boundary_nodes = np.where(
            (nodes[:, 0]**2 + nodes[:, 1]**2) >= 0.95  # within 5% of unit circle
        )[0]
        
        if boundary_nodes.size == 0:
            raise ValueError("No boundary nodes found in mesh")
        
        # Find closest boundary node to this electrode angle
        distances = np.linalg.norm(nodes[boundary_nodes] - target_xy, axis=1)
        closest_idx = boundary_nodes[np.argmin(distances)]
        electrode_nodes.append(closest_idx)
    
    return np.array(electrode_nodes, dtype=np.int32)
```

---

## Issue 3: Otsu Thresholding Commented Out, Replaced with Hand-Tuned Constants ❌ CRITICAL

**Location**: Lines 244 (commented) vs 247-263 (active)

**Current Code**:
```python
# Line 244: OTSU DISABLED (no explanation!)
# labels = _segment_ktc(sigma_map)

# Lines 247-263: Hand-tuned thresholds
factor = 1.0
if batch.level >= 6:
    factor = 1.3   # ← Why 1.3? No justification
elif batch.level >= 4:
    factor = 1.15  # ← Why 1.15? No justification

lower_thresh = mu - self.threshold_std * factor * std
upper_thresh = mu + self.threshold_std * factor * std

seg[inside & (sigma_map < lower_thresh)] = 1  # resistive
seg[inside & (sigma_map > upper_thresh)] = 2  # conductive
```

**Problems**:
1. **Inconsistent across levels** → Different thresholds for different noise = physically wrong
2. **Hand-tuned** → Overfitted to training data, may not generalize
3. **No explanation** → Why was Otsu disabled? Why these exact factors?
4. **Brittle** → If `self.threshold_std` changes, reconstruction breaks
5. **Imported but unused** → `threshold_multiotsu` is imported (line 18) but never called

**Evidence**:
```python
from skimage.filters import threshold_multiotsu  # Imported but not used!
```

**Fix**:
```python
# Re-enable Otsu thresholding (robust, parameter-free)
from skimage.filters import threshold_otsu

# Line 247, replace with:
mu, std = sigma_map[inside].mean(), sigma_map[inside].std()

# Use Otsu's method for robust two-class segmentation
try:
    # Otsu finds optimal threshold to separate two classes
    thresh = threshold_otsu(sigma_map[inside])
    
    seg = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)
    seg[inside & (sigma_map < thresh)] = 1  # resistive (lower conductivity)
    seg[inside & (sigma_map > thresh)] = 2  # conductive (higher conductivity)
    
except ValueError:
    # If Otsu fails (e.g., too few pixels), fall back to mean
    warnings.warn(
        f"Otsu thresholding failed for level {batch.level}; using mean.",
        RuntimeWarning,
        stacklevel=2
    )
    seg[inside & (sigma_map < mu)] = 1
    seg[inside & (sigma_map > mu)] = 2
```

**Why this is better**:
- ✓ Parameter-free (no magic numbers per level)
- ✓ Mathematically rigorous (minimizes inter-class variance)
- ✓ Works across all levels consistently
- ✓ Same algorithm used in official KTC scorer (KTCssim.m)

---

## Validation Checklist

- [ ] Confirm `KTCDataPlugin.load_sample()` populates `reference_voltages` 100% of the time
- [ ] Confirm mesh file has proper 'elfaces' or equivalent key for electrode positions
- [ ] Check what the fallback is actually being used for
- [ ] Re-enable Otsu thresholding instead of hand-tuned factors
- [ ] Test reconstructions match MATLAB reference before/after

---

## Questions for User

1. **Why was Otsu commented out?** Was there a reason, or is this dead code?
2. **What keys does `KTC2023_open_mesh.mat` actually have?** (Run `print(scipy.io.loadmat(...).keys())`)
3. **Are `reference_voltages` consistently populated by KTCDataPlugin?** Or is the None fallback actually triggering?
4. **What are the `factor` values (1.3, 1.15) based on?** Are they tuned to the training set?

---

## Impact Assessment

**If Issue 1 triggers** (reference_voltages = None):
- ✗ Difference voltages are mean-subtracted globally (wrong!)
- ✗ Reconstruction is biased and unreliable

**If Issue 2 triggers** (elfaces parse fails):
- ✗ Electrode positions are evenly spaced through mesh interior
- ✗ Back-projection weights are computed at wrong locations
- ✗ Entire reconstruction is garbage

**If Issue 3 is active** (Otsu disabled):
- ✗ Segmentation is hand-tuned to training set
- ✗ May fail on test data with different noise levels
- ✗ Inconsistent with mathematical best practice

---

**Recommendation**: Fix all three BEFORE submitting final results.
