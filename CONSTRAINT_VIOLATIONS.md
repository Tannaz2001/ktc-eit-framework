# Constraint Violations in Upstream Refactor

**Document**: Recovered from git history (commit 00105fe deleted it)  
**Status**: 🔴 CRITICAL - Core scope violated  
**Impact**: Harder to hand over, added untested novel solvers, violates UI constraints

---

## Original Constraints (from constraint.txt)

### HARD CONSTRAINTS (OUT OF BOUNDS)
1. ✓ NO 3D RECONSTRUCTION - Only 2D, 256×256 (followed)
2. ✓ NO HARDWARE/PHYSICS FOCUS - Framework only (followed)
3. ❌ **NO NOVEL SOLVER INVENTION** - Only baseline solvers (VIOLATED)
4. ❌ **NO HEAVY DESKTOP GUIs** - CLI + YAML only (VIOLATED)

### CORE REQUIREMENTS (MUST-HAVES)
5. ✓ MODULAR PLUGIN ARCHITECTURE (maintained)
6. ⚠️ REPRODUCIBILITY - Tests deleted (COMPROMISED)
7. ✓ STANDARDIZED I/O (maintained)
8. ✓ BATCH RUNNER (maintained)
9. ⚠️ METRICS: KTC score reduced 70→41 lines (SUSPICIOUS)
10. ❌ **STATIC EXPORT** - Replaced with interactive dashboard (VIOLATED)

### STRETCH GOALS (ALLOWED IF CORE DONE FIRST)
- Dashboard using Streamlit (was allowed, but NOT as replacement for static export)
- ML vs traditional comparison (added, okay)
- Simulated data (not added)

---

## Specific Violations

### Violation 1: Novel Solver Invention ❌

**Constraint**: "Only baseline solvers (e.g., Gauss-Newton, Back-projection) to prove framework works"

**Violation**:
| File | Lines | Status | Problem |
|------|-------|--------|---------|
| `reference_fem.py` | 288 | NEW | Full FEM solver (novel implementation) |
| `backprojection.py` | 32→483 | 1400% increase | Massive rewrite, no longer "baseline" |
| `gauss_newton.py` | 30→475 | 1500% increase | Massive rewrite, no longer "baseline" |
| `groundtruth_oracle.py` | 18 | NEW | Additional solver |
| `back_projection_plugin.py` | 127 | NEW | Plugin wrapper for new implementation |

**Why This Matters**:
- Baseline solvers should be simple, well-understood, off-the-shelf
- Large rewrites indicate novel algorithm development
- No validation that 483-line BackProjection produces correct results
- **Risk**: Novel bugs, difficult to debug, hard to hand over

### Violation 2: Heavy Desktop GUI ❌

**Constraint**: "CLI and YAML configuration files, not PyQt or Tkinter GUIs"

**Violation**:
| File | Lines | Type | Status |
|------|-------|------|--------|
| `app.py` | 912 | Streamlit web app | VIOLATES |
| `viz.py` | 498 | Visualization layer | VIOLATES |
| `diagnose_dashboard.py` | 247 | Diagnostic UI | VIOLATES |
| `report_writer.py` | 480 | Report generator | VIOLATES |
| `prepare_dashboard.py` | 122 | Dashboard prep | VIOLATES |

**Why This Matters**:
- Constraint explicitly said "not... GUIs" (Streamlit IS a GUI framework)
- CLI + YAML was the design requirement
- Dashboard was a STRETCH GOAL, meant to supplement static export, not replace it
- **Risk**: Harder to reproduce, harder to test, harder to use in CI/CD

### Violation 3: Static Export Requirement ❌

**Constraint**: "Must generate leaderboard-style report (e.g., HTML/JSON)"

**What We Got**:
- ✓ HTML report (`html_report.py` - 164 lines)
- ✓ JSON outputs (`per_run_metrics.json`, `scores.json`)
- ✓ **BUT**: Primary interface is interactive Streamlit dashboard (`app.py`)
- ❌ Static leaderboard is secondary/optional

**Why This Matters**:
- Static HTML is reproducible and CI/CD-friendly
- Interactive dashboard is harder to automate
- Streaming dashboards don't work well in batch pipelines
- **Risk**: Can't easily integrate into evaluation pipeline

---

## Tests Deleted ❌

**Constraint**: "REPRODUCIBILITY: All code... must be fully reproducible"

**Violation**:
| File | Status | Impact |
|------|--------|--------|
| `test_runner.py` | DELETED | No batch runner validation |
| `test_config_validator.py` | DELETED | No config validation tests |
| `test_file_validator.py` | DELETED | No file format tests |
| Only `test_methods.py` remains | 87 lines | Minimal coverage |

**Why This Matters**:
- Deleted tests means no way to verify changes
- No regression testing
- Harder to hand over (no test suite to run)
- **Risk**: Silent failures, undetected bugs

---

## Suspicious Deletions

### ktc_score.py: 70 → 41 Lines (41% Reduction)

**Questions**:
- What metrics were deleted?
- Was HD95 validation removed?
- Was Otsu thresholding logic deleted?

**We Fixed This**: Re-enabled Otsu, restored HD95 function  
**But**: Original code deletion is concerning for what was lost

---

## Impact Assessment

### On Handover
- ❌ No test suite to validate correctness
- ❌ Heavy GUI makes it hard to integrate into pipelines
- ❌ Novel solvers need documentation/validation
- ✓ Code is well-organized and modular

### On Reproducibility
- ❌ Streamlit app is stateful and hard to script
- ⚠️ Missing tests mean no automated verification
- ✓ Configuration system (YAML) is still present
- ✓ Code is version-controlled

### On Scope Creep
- ❌ Went from benchmarking framework to visualization platform
- ❌ Added 3000+ lines of GUI code
- ❌ Added 400+ lines of novel solver code
- ✓ Core plugin architecture still works

---

## Recommendations

### Option A: Restore Constraints (Recommended)
1. **Revert GUI code**: Remove `app.py`, `viz.py`, `diagnose_dashboard.py`
2. **Keep static export**: Keep `html_report.py`, `report_writer.py`
3. **Validate solvers**: Validate BackProjection/GaussNewton against MATLAB reference
4. **Restore tests**: Rebuild test suite (can be auto-generated from test_methods.py)
5. **Simplify**: Focus on CLI + YAML + static reports

**Effort**: Medium (refactoring, but maintains scope)

### Option B: Acknowledge Scope Change
1. **Document**: Explicitly update constraints.txt to allow GUI/dashboard
2. **Validate**: Ensure novel solvers work correctly (baseline establishment)
3. **Test**: Rebuild test suite
4. **Handover**: Create UI documentation

**Effort**: Low (documentation), but violates original intent

### Option C: Hybrid (Recommended)
1. **Keep static export** as primary interface (requirement)
2. **Keep dashboard** as optional visualization tool (stretch goal, not replacement)
3. **Validate solvers** thoroughly (baseline establishment)
4. **Restore tests** for reproducibility (requirement)
5. **Document changes** and updated constraints

**Effort**: Medium, maintains both core requirement + stretch goal

---

## What We've Already Fixed

✅ Re-enabled Otsu thresholding (was commented out)  
✅ Restored electrode detection (was broken)  
✅ Re-enabled reference voltage validation  
✅ Restored MATLAB reference code  
✅ Removed HD95/DICE from metrics (simplified to KTC score only)

**Still Needs**:
- [ ] Validate BackProjection/GaussNewton work correctly
- [ ] Restore test suite
- [ ] Document what "novel solver" means (are we OK with 483-line BackProjection?)
- [ ] Clarify dashboard role (primary interface or supplementary?)

---

## Decision Point

**Before building test harness**: Decide on constraints.

If GUI + novel solvers are OK → document it, validate thoroughly  
If they violate scope → revert to CLI + YAML + static reports

Can't test what the scope should be.

