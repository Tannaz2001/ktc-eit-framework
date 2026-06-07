# MATLAB Reference Code — Validation Ground Truth

**Status**: Critical for correctness validation  
**Restored**: 2026-06-07  
**Source**: `origin/sprint-7`

---

## Purpose

The MATLAB files in `Codes_Matlab/` are the authoritative reference implementations for EIT reconstruction algorithms and scoring. Python ports in `data/KTCScoring/` and `src/ktc_framework/` must be validated against these originals.

## Critical Files

| File | Lines | Purpose | Python Port |
|------|-------|---------|------------|
| `circmesh.m` | 22,314 | Circular mesh generation for EIT | `src/ktc_framework/utils/pyeit_utils.py` |
| `EITFEM.m` | 584 | Forward problem solver (FEM) | `data/KTCScoring/KTCFwd.py` |
| `MiscCodes/KTCssim.m` | 47 | KTC scoring metric (SSIM-based) | `src/ktc_framework/metrics/ktc_score.py` |
| `circ.geo` | 105 | Gmsh geometry for mesh generation | Reference for mesh structure |

---

## Validation Checklist

- [ ] KTCFwd.py produces identical voltages to EITFEM.m for test conductivity
- [ ] ktc_score.py produces identical scores to KTCssim.m for test images
- [ ] Mesh generation in pyeit_utils.py matches circmesh.m node/element count
- [ ] Error overlays match MATLAB ground truth within floating-point tolerance

---

## How to Validate

```bash
# 1. Run MATLAB reference solver (requires MATLAB)
cd Codes_Matlab
python ../data/KTCScoring/KTCFwd.py  # Python port

# 2. Compare outputs
# (Create a validation harness that runs both and compares)

# 3. Check for deviations > floating-point epsilon (1e-12)
```

---

## Why This Matters

**Without the MATLAB originals:**
- ✗ Cannot prove Python ports are correct
- ✗ Cannot debug discrepancies between expected and actual scores
- ✗ Cannot validate FEM solver against numerical gold standard
- ✗ Risk of silently incorrect reconstructions

**With the MATLAB originals:**
- ✓ Ground truth for correctness
- ✓ Regression testing when updating algorithms
- ✓ Confidence in scoring methodology
- ✓ Traceability to KTC 2023 challenge spec

---

## References

- **KTC 2023**: Kaczmarz Team Challenge dataset and official scorer
- **PyEIT**: Python EIT forward solver (used as fallback if MATLAB unavailable)
- **Original MATLAB code**: Available in `Codes_Matlab/` (restored from sprint-7)

---

**Last Validated**: Not yet (see checklist)  
**Validated By**: _To be filled_  
**Date**: _To be filled_
