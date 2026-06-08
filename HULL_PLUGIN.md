# Hull Plugin — Geometric Analysis for EIT Reconstructions

**Status:** ✅ Implemented and tested  
**Component:** Post-processing utility (NOT part of core pipeline)  
**Scope:** Optional error analysis for inclusion localization accuracy

---

## What Is the Hull Plugin?

The Hull Plugin is a **post-processing analysis tool** that extracts convex hulls of detected resistive and conductive inclusions from a 256×256 EIT reconstruction.

### It answers:
- "Where did the method locate the inclusions?"
- "How accurate were the size/shape estimates?"
- "Did the method systematically bias localization?"

### It does NOT:
- Affect KTC scores (purely post-hoc analysis)
- Require integration into the core pipeline
- Depend on any additional data files

---

## API

### Basic Usage

```python
from src.ktc_framework.methods.hull_plugin import HullPlugin
import numpy as np

# Analyze a reconstruction
pred = np.zeros((256, 256), dtype=np.uint8)
# ... populate with labels 0=water, 1=resistive, 2=conductive ...

result = HullPlugin.analyze(pred)

# Access geometric properties
print(f"Conductive center: {result.conductive_center}")
print(f"Conductive area:   {result.conductive_area} pixels²")
print(f"Conductive perim:  {result.conductive_perimeter} pixels")
```

### Compare Against Ground Truth

```python
result_pred = HullPlugin.analyze(prediction)
result_gt   = HullPlugin.analyze(ground_truth)

errors = HullPlugin.compare_hulls(result_pred, result_gt)
# Returns dict with keys:
# - resistive_center_error (pixels)
# - resistive_area_error (pixels²)
# - resistive_perimeter_error (pixels)
# - conductive_center_error (pixels)
# - conductive_area_error (pixels²)
# - conductive_perimeter_error (pixels)
```

---

## Return Types

### `HullResult` (NamedTuple)

```python
HullResult(
    # Resistive region (label=1)
    resistive_center: tuple[float, float] | None,      # (y, x) in pixels
    resistive_area: float | None,                       # pixels²
    resistive_perimeter: float | None,                  # pixels
    resistive_hull: np.ndarray | None,                  # (N, 2) vertices
    
    # Conductive region (label=2)
    conductive_center: tuple[float, float] | None,
    conductive_area: float | None,
    conductive_perimeter: float | None,
    conductive_hull: np.ndarray | None,
    
    # Metadata
    prediction_shape: tuple[int, int],                  # Always (256, 256)
    num_pixels_resistive: int,                          # Total pixels with label=1
    num_pixels_conductive: int,                         # Total pixels with label=2
)
```

**Note:** All geometric values (`center`, `area`, `perimeter`, `hull`) are `None` if fewer than 3 pixels exist in the region (degenerate case).

---

## Implementation Details

### Algorithm

1. **Boundary Detection:** Extract all pixels with label=1 or label=2
2. **Convex Hull:** Use scipy's ConvexHull to compute the minimum convex polygon
3. **Geometric Descriptors:**
   - **Center:** Center of mass via scipy.ndimage.center_of_mass
   - **Area:** Hull.volume (2D convex hull area)
   - **Perimeter:** Sum of Euclidean distances between consecutive hull vertices
4. **Vertex Extraction:** Return hull vertices for visualization

### Robustness

- Returns `None` for all properties if region has <3 pixels (cannot form hull)
- Handles collinear points gracefully (ConvexHull exception caught)
- No dependencies on specific mesh geometry or electrode positions

---

## Testing

All functionality is covered by 10 unit tests in `tests/test_hull_plugin.py`:

- ✅ Circle center and area computation
- ✅ Degenerate cases (single pixel, collinear)
- ✅ Multiple regions (resistive + conductive)
- ✅ Input validation (shape, dtype)
- ✅ Comparison metrics (perfect match, offsets, missing regions)
- ✅ Pixel counts and metadata

**Run tests:**
```bash
python -m pytest tests/test_hull_plugin.py -v
# Result: 10/10 passed in 2.2s
```

---

## Use Cases

### ✅ When to Use Hull Plugin

- **Localization Error Analysis:** Where did the method place the inclusion?
- **Size Accuracy:** How close are the estimated area/perimeter?
- **Systematic Bias Detection:** Does the method consistently underestimate or shift locations?
- **Visualization:** Overlay hulls on reconstructions to show detected regions
- **Supplementary Metrics:** Add geometric descriptors to output reports

### ❌ When NOT to Use Hull Plugin

- **Direct Scoring:** Use KTC score instead
- **Real-time Reconstruction:** It's post-processing only
- **Segmentation Metrics:** Dice, IoU are better for pixel-level comparison

---

## Integration Example (Optional)

To optionally integrate hull analysis into the experiment runner:

```python
# In experiment_runner.py, after reconstruction:

if self.config.get("compute_hull_analysis", False):
    from src.ktc_framework.methods.hull_plugin import HullPlugin
    
    hull_pred = HullPlugin.analyze(pred, None)
    hull_gt   = HullPlugin.analyze(batch.ground_truth, None)
    hull_errors = HullPlugin.compare_hulls(hull_pred, hull_gt)
    
    # Store in results dict for reporting
    result["hull_errors"] = hull_errors
```

Then enable in YAML config:
```yaml
# configs/ktc_all_methods.yaml
compute_hull_analysis: true
```

---

## Example Output

```
Prediction Hull Result:
  Resistive center:    (145.2, 98.5) pixels
  Resistive area:      3421 pixels²
  Resistive perim:     198.3 pixels
  Detected:            2847 pixels
  
  Conductive center:   (89.1, 187.3) pixels
  Conductive area:     2156 pixels²
  Conductive perim:    165.8 pixels
  Detected:            1923 pixels

Comparison vs Ground Truth:
  Resistive center error:    8.4 pixels
  Resistive area error:      157 pixels²
  Conductive center error:   3.2 pixels
  Conductive area error:     102 pixels²
```

---

## Constraint Compliance

✅ **Allowed under constraint.txt:**
- Post-processing only (not a reconstruction solver)
- Supplementary analysis (does not affect KTC score)
- Uses standard scipy algorithms (no novel math)
- Optional feature (can be disabled)

---

## Files

| File | Purpose |
|------|---------|
| `src/ktc_framework/methods/hull_plugin.py` | Implementation (300 lines) |
| `tests/test_hull_plugin.py` | Unit tests (10 tests, all passing) |
| `HULL_PLUGIN.md` | This documentation |

---

## Summary

The Hull Plugin is a complete, tested post-processing utility for geometric error analysis of EIT reconstructions. It requires no changes to the core pipeline, no additional data, and can be used independently or integrated optionally.

**Status:** Ready for production use ✅
