#!/usr/bin/env python3
"""diagnose_ktc.py — validates the KTC dataset setup before running experiments.

Checks mesh geometry, reference voltages, training data, evaluation data,
ground truth masks, and difference-imaging signal quality.  Prints a
human-readable report and exits 0 if ready, 1 if anything is missing.

Usage
-----
    python diagnose_ktc.py --root EvaluationData --mesh Codes_Matlab/Mesh_sparse.mat

    # Training-data layout (root = Codes_Matlab)
    python diagnose_ktc.py --root Codes_Matlab   --mesh Codes_Matlab/Mesh_sparse.mat

Dependencies: stdlib + numpy + scipy  (h5py optional, for .mat v7.3 files)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

import numpy as np

try:
    import scipy.io as _sio
    _SCIPY = True
except ImportError:                     # pragma: no cover
    _sio   = None                       # type: ignore[assignment]
    _SCIPY = False

try:
    import h5py as _h5py
except ImportError:
    _h5py = None                        # type: ignore[assignment]

# ── Unicode safety ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TICK  = "✓"   # ✓
CROSS = "✗"   # ✗
WARN  = "!"

def _s(text: str) -> str:
    """Replace Unicode symbols with ASCII fallbacks if the terminal can't print them."""
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return text
    except (UnicodeEncodeError, LookupError):
        return text.replace(TICK, "OK").replace(CROSS, "--")


# ── Line-width and print helpers ──────────────────────────────────────────
_W = 72

def _hr(ch: str = "-") -> None:
    print(ch * _W)

def _section(title: str) -> None:
    print()
    _hr("=")
    print(f"  {title}")
    _hr("=")

def _ok(msg: str)   -> None: print(_s(f"  {TICK}  {msg}"))
def _fail(msg: str) -> None: print(_s(f"  {CROSS}  {msg}"))
def _warn(msg: str) -> None: print(       f"  {WARN}  {msg}")
def _info(msg: str) -> None: print(       f"     {msg}")


# ── .mat file loader (v5 + v7.3 HDF5) ────────────────────────────────────

def _load_mat(path: str) -> dict[str, Any]:
    """Load a .mat file and return a plain dict mapping key → array."""
    if not _SCIPY:
        raise ImportError("scipy is required.  Run: pip install scipy")
    try:
        return _sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if _h5py is None:
            raise ImportError(
                "h5py is required for .mat v7.3 files.  Run: pip install h5py"
            )
        result: dict[str, Any] = {}
        with _h5py.File(path, "r") as fh:
            for key in fh.keys():
                result[key] = np.array(fh[key])
        return result


def _pub_keys(mat: dict) -> list[str]:
    """Return non-private keys from a loadmat result dict."""
    return [k for k in mat if not k.startswith("_")]


def _first_key(mat: dict, candidates: list[str]) -> Optional[str]:
    """Return the first candidate key that exists in *mat*, or None."""
    for k in candidates:
        if k in mat:
            return k
    return None


def _mat_struct_fields(obj: Any) -> list[str]:
    """Return field names of a scipy mat_struct, or [] if not applicable."""
    try:
        return list(obj._fieldnames)
    except AttributeError:
        return []


# ── Status accumulator ────────────────────────────────────────────────────

class _Results:
    """Collect pass/warn/fail results for the final summary."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    def __init__(self) -> None:
        self._rows: list[tuple[str, str]] = []

    def record(self, section: str, status: str) -> None:
        self._rows.append((section, status))

    def print_summary(self) -> bool:
        _section("SUMMARY")
        col = 34
        has_fail = False
        has_warn = False
        for section, status in self._rows:
            if status == self.PASS:
                sym = TICK
            elif status == self.WARN:
                sym = WARN
                has_warn = True
            else:
                sym = CROSS
                has_fail = True
            print(_s(f"  {sym}  {section.ljust(col)} {status}"))
        print()
        _hr("=")
        if not has_fail and not has_warn:
            print(_s(f"  {TICK}  READY TO RUN — all checks passed"))
        elif not has_fail:
            # Warnings mean degraded but runnable (e.g. training data absent
            # when using evaluation root, or ref.mat absent → mean-subtraction)
            print(_s(f"  {WARN}  READY TO RUN (with warnings) — "
                     f"check the WARN items above"))
        else:
            fails   = [s for s, st in self._rows if st == self.FAIL]
            missing = "  Missing: " + ", ".join(fails)
            print(_s(f"  {CROSS}  NOT READY — fix FAIL items before running.  "
                     f"{missing}"))
        _hr("=")
        return not has_fail


# ── Check 1: MESH ─────────────────────────────────────────────────────────

def check_mesh(mesh_path: str, results: _Results) -> None:
    _section("1 / 6  MESH CHECK")

    if not os.path.isfile(mesh_path):
        _fail(f"Mesh file not found: {mesh_path}")
        results.record("Mesh file", _Results.FAIL)
        return

    _ok(f"File found: {mesh_path}  ({os.path.getsize(mesh_path):,} bytes)")

    try:
        mat = _load_mat(mesh_path)
    except Exception as exc:
        _fail(f"Could not load mesh: {exc}")
        results.record("Mesh file", _Results.FAIL)
        return

    pub = _pub_keys(mat)
    _info(f"Top-level variables: {pub}")

    # Print all fields at the top level and inside any mat_struct
    node_arr:    Optional[np.ndarray] = None
    element_arr: Optional[np.ndarray] = None

    for key in pub:
        val = mat[key]
        fields = _mat_struct_fields(val)

        if fields:
            _info(f"  {key}  (struct)  fields = {fields}")
            # Track node/element pair within this struct only — different structs
            # (e.g. Mesh vs Mesh2) reference different node sets so pairing
            # across structs would produce misleading index-range messages.
            local_node: Optional[np.ndarray]    = None
            local_elem: Optional[np.ndarray]    = None
            for f in fields:
                try:
                    arr = np.asarray(getattr(val, f))
                    _info(f"    .{f:<12}  shape={str(arr.shape):<16}  dtype={arr.dtype}")

                    # Identify node-coordinate array: 2-D, N×2 or N×3, float
                    if (arr.ndim == 2
                            and arr.shape[1] in (2, 3)
                            and arr.dtype.kind == "f"
                            and arr.shape[0] > 100):
                        local_node = arr
                        _info(f"      ^ node coordinates ({arr.shape[0]} nodes, "
                              f"{arr.shape[1]}D)")

                    # Identify element-connectivity array: 2-D, M×3, integer
                    if (arr.ndim == 2
                            and arr.shape[1] == 3
                            and arr.dtype.kind in ("u", "i")
                            and arr.shape[0] > 100):
                        local_elem = arr
                        _info(f"      ^ element connectivity ({arr.shape[0]} elements)")

                except Exception:
                    _info(f"    .{f}  <unreadable>")

            # Promote first valid pair to the top-level trackers
            if node_arr is None and local_node is not None:
                node_arr = local_node
            if element_arr is None and local_elem is not None:
                element_arr = local_elem
        else:
            try:
                arr = np.asarray(val)
                _info(f"  {key:<14}  shape={str(arr.shape):<16}  dtype={arr.dtype}")
            except Exception:
                _info(f"  {key}  <unreadable>")

    # Node / element summary
    print()
    if node_arr is not None:
        _ok(f"Nodes identified    : {node_arr.shape[0]:,}  "
            f"(shape {node_arr.shape}, dtype {node_arr.dtype})")
    else:
        _warn("No node-coordinate array identified (expected N×2 or N×3 float)")

    if element_arr is not None:
        _ok(f"Elements identified : {element_arr.shape[0]:,}  "
            f"(shape {element_arr.shape}, dtype {element_arr.dtype})")
        # Sanity: elements should reference valid node indices
        idx_max = int(element_arr.max())
        one_indexed = idx_max >= (node_arr.shape[0] if node_arr is not None else 1)
        _info(f"  Max element index = {idx_max}  "
              f"({'1-indexed MATLAB' if one_indexed else '0-indexed Python'})")
    else:
        _warn("No element-connectivity array identified (expected M×3 integer)")

    status = _Results.PASS if (node_arr is not None and element_arr is not None) \
             else _Results.WARN
    results.record("Mesh file", status)


# ── Check 2: REFERENCE VOLTAGES ───────────────────────────────────────────

def check_reference(root: str, results: _Results) -> Optional[np.ndarray]:
    """Return ref voltage array if found, else None."""
    _section("2 / 6  REFERENCE VOLTAGE CHECK")

    # Try several candidate locations and key names
    candidates = [
        os.path.join(root, "ref.mat"),
        os.path.join(root, "TrainingData", "ref.mat"),
    ]
    ref_keys = ["Uelref", "Uel", "ref", "Uref"]

    ref_path: Optional[str] = None
    for p in candidates:
        if os.path.isfile(p):
            ref_path = p
            break

    if ref_path is None:
        _fail(f"ref.mat not found.  Tried:")
        for p in candidates:
            _info(f"  {p}")
        _warn("BackProjection / GaussNewton will fall back to mean-subtraction.")
        results.record("Reference voltages (ref.mat)", _Results.FAIL)
        return None

    _ok(f"Found: {ref_path}  ({os.path.getsize(ref_path):,} bytes)")

    try:
        mat = _load_mat(ref_path)
    except Exception as exc:
        _fail(f"Could not load ref.mat: {exc}")
        results.record("Reference voltages (ref.mat)", _Results.FAIL)
        return None

    pub = _pub_keys(mat)
    _info(f"Keys: {pub}")

    key = _first_key(mat, ref_keys)
    if key is None:
        _fail(f"No voltage key found in ref.mat.  Keys present: {pub}")
        results.record("Reference voltages (ref.mat)", _Results.FAIL)
        return None

    ref_v = np.asarray(mat[key], dtype=np.float64).ravel()
    _ok(f"Voltage array  key='{key}'  shape={ref_v.shape}  dtype={ref_v.dtype}")
    _info(f"  Range : [{ref_v.min():.6g},  {ref_v.max():.6g}]")
    _info(f"  Mean  : {ref_v.mean():.6g}")
    _info(f"  Std   : {ref_v.std():.6g}")

    results.record("Reference voltages (ref.mat)", _Results.PASS)
    return ref_v


# ── Check 3: TRAINING DATA ────────────────────────────────────────────────

def check_training_data(root: str, results: _Results) -> None:
    _section("3 / 6  TRAINING DATA CHECK")

    td_dir = os.path.join(root, "TrainingData")
    gt_dir = os.path.join(root, "GroundTruths")

    if not os.path.isdir(td_dir):
        _warn(f"TrainingData/ not found under '{root}'  (expected {td_dir})")
        _info("Skipping training-data check.")
        results.record("Training data", _Results.WARN)
        return

    _ok(f"TrainingData folder : {td_dir}")

    n_found = 0
    for i in range(1, 5):
        fname = f"data{i}.mat"
        fpath = os.path.join(td_dir, fname)
        if not os.path.isfile(fpath):
            _fail(f"  {fname}  NOT FOUND  ({fpath})")
            continue

        try:
            mat    = _load_mat(fpath)
            pub    = _pub_keys(mat)
            u_key  = _first_key(mat, ["Uel", "Uelref", "U"])
            inj_key = _first_key(mat, ["Inj", "Injref", "I"])
            u_shape   = np.asarray(mat[u_key]).shape   if u_key   else "missing"
            inj_shape = np.asarray(mat[inj_key]).shape if inj_key else "missing"
            _ok(f"  {fname}  Uel={u_shape}  Inj={inj_shape}  keys={pub}")
            n_found += 1
        except Exception as exc:
            _fail(f"  {fname}  load error: {exc}")

    # Ground truths for training
    print()
    if os.path.isdir(gt_dir):
        _ok(f"GroundTruths folder : {gt_dir}")
        for i in range(1, 5):
            fname = f"true{i}.mat"
            fpath = os.path.join(gt_dir, fname)
            sym   = TICK if os.path.isfile(fpath) else CROSS
            print(_s(f"  {sym}  {fname}"))
    else:
        _warn(f"GroundTruths/ not found under '{root}'  (expected {gt_dir})")

    status = _Results.PASS if n_found == 4 else (_Results.WARN if n_found > 0 else _Results.FAIL)
    results.record("Training data", status)


# ── Check 4: EVALUATION DATA ──────────────────────────────────────────────

def check_evaluation_data(root: str, results: _Results) -> Optional[str]:
    """Return the evaluation folder path if found, else None."""
    _section("4 / 6  EVALUATION DATA CHECK")

    eval_candidates = ["evaluation_datasets", "EvaluationData"]
    eval_dir: Optional[str] = None
    for name in eval_candidates:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            eval_dir = p
            break

    if eval_dir is None:
        _fail(f"Evaluation data folder not found under '{root}'.  Tried: {eval_candidates}")
        results.record("Evaluation data", _Results.FAIL)
        return None

    _ok(f"Evaluation folder : {eval_dir}")
    print()

    # ── 7×3 existence grid ─────────────────────────────────────────────────
    n_samples = 3
    n_levels  = 7
    grid: list[list[bool]] = []

    col_w = 10
    header = "         " + "".join(f"Sample {j+1}".center(col_w) for j in range(n_samples))
    print(header)
    _hr("-")

    for lv in range(1, n_levels + 1):
        row: list[bool] = []
        row_str = f"Level {lv} : "
        for samp in range(1, n_samples + 1):
            fpath = os.path.join(eval_dir, f"level{lv}", f"data{samp}.mat")
            exists = os.path.isfile(fpath)
            row.append(exists)
            row_str += (_s(TICK) if exists else _s(CROSS)).center(col_w)
        grid.append(row)
        print(row_str)

    _hr("-")
    n_present = sum(cell for row in grid for cell in row)
    total     = n_levels * n_samples
    print(f"\n  {n_present}/{total} data files present")

    # ── Detail for level1/data1 ────────────────────────────────────────────
    detail_path = os.path.join(eval_dir, "level1", "data1.mat")
    if os.path.isfile(detail_path):
        print()
        _info(f"Detail — {detail_path}")
        try:
            mat   = _load_mat(detail_path)
            u_key = _first_key(mat, ["Uel", "U"])
            if u_key:
                u = np.asarray(mat[u_key], dtype=np.float64).ravel()
                _info(f"  Uel  shape={u.shape}  dtype={mat[u_key].dtype if hasattr(mat[u_key],'dtype') else 'n/a'}")
                _info(f"  Range : [{u.min():.6g},  {u.max():.6g}]")
                _info(f"  Std   : {u.std():.6g}")
        except Exception as exc:
            _warn(f"  Could not load detail: {exc}")

    status = _Results.PASS if n_present == total else \
             (_Results.WARN if n_present > 0 else _Results.FAIL)
    results.record("Evaluation data", status)
    return eval_dir


# ── Check 5: GROUND TRUTH ─────────────────────────────────────────────────

def check_ground_truth(root: str, results: _Results) -> None:
    _section("5 / 6  GROUND TRUTH CHECK")

    gt_candidates = ["GroundTruths", "groundtruths", "GroundTruth"]
    gt_dir: Optional[str] = None
    for name in gt_candidates:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            gt_dir = p
            break

    if gt_dir is None:
        _fail(f"Ground truth folder not found under '{root}'.  Tried: {gt_candidates}")
        results.record("Ground truths", _Results.FAIL)
        return

    _ok(f"GroundTruths folder : {gt_dir}")
    print()

    # ── 7×3 existence grid ─────────────────────────────────────────────────
    n_samples = 3
    n_levels  = 7
    col_w     = 10

    header = "         " + "".join(f"Sample {j+1}".center(col_w) for j in range(n_samples))
    print(header)
    _hr("-")

    n_present = 0
    for lv in range(1, n_levels + 1):
        row_str = f"Level {lv} : "
        for samp in range(1, n_samples + 1):
            fpath = os.path.join(gt_dir, f"level_{lv}", f"{samp}_true.mat")
            exists = os.path.isfile(fpath)
            if exists:
                n_present += 1
            row_str += (_s(TICK) if exists else _s(CROSS)).center(col_w)
        print(row_str)

    _hr("-")
    total = n_levels * n_samples
    print(f"\n  {n_present}/{total} ground-truth files present")

    # ── Detail for level_1/1_true ──────────────────────────────────────────
    detail_path = os.path.join(gt_dir, "level_1", "1_true.mat")
    if os.path.isfile(detail_path):
        print()
        _info(f"Detail — {detail_path}")
        try:
            mat = _load_mat(detail_path)
            gt_key = _first_key(mat, ["truth", "Truth", "gt", "GT", "labels"])
            if gt_key:
                gt = np.asarray(mat[gt_key])
                _info(f"  Key   = '{gt_key}'  shape={gt.shape}  dtype={gt.dtype}")
                labels, counts = np.unique(gt, return_counts=True)
                total_px = gt.size
                for lbl, cnt in zip(labels, counts):
                    pct = cnt / total_px * 100
                    names = {0: "water/background", 1: "resistive", 2: "conductive"}
                    _info(f"  Label {int(lbl)} ({names.get(int(lbl), '?')}) : "
                          f"{cnt:,} px  ({pct:.1f} %)")
            else:
                _warn(f"  No recognised GT key.  Keys: {_pub_keys(mat)}")
        except Exception as exc:
            _warn(f"  Could not load detail: {exc}")

    status = _Results.PASS if n_present == total else \
             (_Results.WARN if n_present > 0 else _Results.FAIL)
    results.record("Ground truths", status)


# ── Check 6: DIFFERENCE IMAGING ───────────────────────────────────────────

def check_difference_imaging(
    root: str,
    ref_voltages: Optional[np.ndarray],
    eval_dir:     Optional[str],
    results:      _Results,
) -> None:
    _section("6 / 6  DIFFERENCE IMAGING TEST")

    if eval_dir is None:
        _warn("Evaluation data unavailable — skipping difference imaging test.")
        results.record("Difference imaging", _Results.WARN)
        return

    data1_path = os.path.join(eval_dir, "level1", "data1.mat")
    if not os.path.isfile(data1_path):
        _warn(f"level1/data1.mat not found at {data1_path} — skipping.")
        results.record("Difference imaging", _Results.WARN)
        return

    try:
        mat   = _load_mat(data1_path)
        u_key = _first_key(mat, ["Uel", "U"])
        if u_key is None:
            _fail(f"No voltage key in {data1_path}.  Keys: {_pub_keys(mat)}")
            results.record("Difference imaging", _Results.FAIL)
            return
        voltages = np.asarray(mat[u_key], dtype=np.float64).ravel()
    except Exception as exc:
        _fail(f"Could not load level1/data1.mat: {exc}")
        results.record("Difference imaging", _Results.FAIL)
        return

    _ok(f"Measurement voltages loaded  shape={voltages.shape}")

    # Method A: mean subtraction
    mean_sub = voltages - np.mean(voltages)
    _info(f"Mean-subtraction  (v - mean(v)):")
    _info(f"  Range [{mean_sub.min():.6g},  {mean_sub.max():.6g}]   "
          f"Std = {mean_sub.std():.6g}")

    # Method B: reference subtraction
    if ref_voltages is not None:
        ref_arr = ref_voltages.ravel().astype(np.float64)
        if len(ref_arr) >= len(voltages):
            ref_aligned = ref_arr[:len(voltages)]
        else:
            _warn(f"ref length {len(ref_arr)} < voltages length {len(voltages)}"
                  f" — zero-padding ref")
            ref_aligned = np.pad(ref_arr, (0, len(voltages) - len(ref_arr)))

        ref_sub = voltages - ref_aligned
        _info(f"Reference-subtraction  (v - v_ref):")
        _info(f"  Range [{ref_sub.min():.6g},  {ref_sub.max():.6g}]   "
              f"Std = {ref_sub.std():.6g}")

        if ref_sub.std() > 1e-12:
            ratio = mean_sub.std() / ref_sub.std()
            _info(f"Signal improvement ratio: {ratio:.1f}x  "
                  f"({'ref_sub cleaner' if ratio > 1 else 'mean_sub cleaner'})")
            print()
            if ratio >= 1.0:
                _ok("Reference subtraction produces cleaner difference signal "
                    "(lower residual std).")
            else:
                _warn("Mean subtraction has lower residual std — "
                      "check that ref.mat matches the evaluation protocol.")
        else:
            _warn("ref_sub std ≈ 0 — ref voltages may be identical to measurement.")

        results.record("Difference imaging", _Results.PASS)
    else:
        _warn("No reference voltages — can only assess mean-subtraction.")
        _info("Set up ref.mat to enable proper difference imaging.")
        results.record("Difference imaging", _Results.WARN)


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate KTC dataset setup before running experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python diagnose_ktc.py "
            "--root EvaluationData --mesh Codes_Matlab/Mesh_sparse.mat\n"
            "  python diagnose_ktc.py "
            "--root Codes_Matlab   --mesh Codes_Matlab/Mesh_sparse.mat"
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root path (folder containing evaluation_datasets/ and GroundTruths/).",
    )
    parser.add_argument(
        "--mesh",
        required=True,
        help="Path to Mesh_sparse.mat (or the directory that contains it).",
    )
    args = parser.parse_args()

    # Resolve mesh path: if a directory, append Mesh_sparse.mat
    mesh_path: str = args.mesh
    if os.path.isdir(mesh_path):
        mesh_path = os.path.join(mesh_path, "Mesh_sparse.mat")

    print()
    _hr("=")
    print("  KTC DATASET DIAGNOSTICS")
    _hr("=")
    print(f"  Root : {os.path.abspath(args.root)}")
    print(f"  Mesh : {os.path.abspath(mesh_path)}")
    _hr("=")

    results  = _Results()
    ref_v    = None
    eval_dir = None

    check_mesh(mesh_path, results)
    ref_v    = check_reference(args.root, results)
    check_training_data(args.root, results)
    eval_dir = check_evaluation_data(args.root, results)
    check_ground_truth(args.root, results)
    check_difference_imaging(args.root, ref_v, eval_dir, results)

    ready = results.print_summary()
    print()
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
