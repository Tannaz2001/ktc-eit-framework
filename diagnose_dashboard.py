"""
diagnose_dashboard.py - Find out exactly why views aren't working
"""

import json
from pathlib import Path
import sys

print("=" * 70)
print("DASHBOARD DIAGNOSTIC TOOL")
print("=" * 70)

# ============================================================
# CHECK 1: Data Files Exist
# ============================================================
print("\n[CHECK 1] Verifying data files exist...")

scores_exists = Path("scores.json").exists()
per_run_exists = Path("outputs/per_run_metrics.json").exists()

if scores_exists:
    print("  ✓ scores.json found")
else:
    print("  ✗ scores.json NOT FOUND")
    print("    → Run: python example_usage.py")
    sys.exit(1)

if per_run_exists:
    print("  ✓ outputs/per_run_metrics.json found")
else:
    print("  ✗ outputs/per_run_metrics.json NOT FOUND")
    print("    → Run: python example_usage.py")
    sys.exit(1)

# ============================================================
# CHECK 2: Load and Inspect Data Structure
# ============================================================
print("\n[CHECK 2] Loading and inspecting data structure...")

with open("scores.json", 'r') as f:
    scores = json.load(f)

with open("outputs/per_run_metrics.json", 'r') as f:
    per_run = json.load(f)

print(f"\n  scores.json structure:")
print(f"  ├─ Number of methods: {len(scores)}")
for method_name in scores.keys():
    print(f"  │  ├─ '{method_name}'")
    metrics = list(scores[method_name].keys())
    print(f"  │  │  └─ Metrics: {metrics[:3]}{'...' if len(metrics) > 3 else ''}")

print(f"\n  per_run_metrics.json structure:")
print(f"  ├─ Number of methods: {len(per_run)}")
for method_key in per_run.keys():
    samples = list(per_run[method_key].keys())
    print(f"  │  ├─ '{method_key}'")
    print(f"  │  │  ├─ Samples: {samples}")
    if samples:
        first_sample = samples[0]
        metrics = list(per_run[method_key][first_sample].keys())
        print(f"  │  │  └─ Sample '{first_sample}' metrics: {metrics[:3]}{'...' if len(metrics) > 3 else ''}")

# ============================================================
# CHECK 3: Method Name Mapping
# ============================================================
print("\n[CHECK 3] Testing method name mapping...")

scores_methods = list(scores.keys())
per_run_methods = list(per_run.keys())

print(f"\n  Display names (from scores.json):")
for name in scores_methods:
    print(f"    - '{name}'")

print(f"\n  Internal keys (from per_run_metrics.json):")
for key in per_run_methods:
    print(f"    - '{key}'")

# Test the mapping function
def create_method_mapping(scores, per_run):
    """Create mapping between display names and internal keys."""
    mapping = {}
    
    for display_name in scores.keys():
        display_lower = display_name.lower()
        
        for internal_key in per_run.keys():
            if internal_key.replace('_', '-') in display_lower or internal_key.replace('_', ' ') in display_lower:
                mapping[display_name] = internal_key
                break
            elif display_name.split()[0].lower().replace('-', '_') == internal_key.split('_')[0]:
                mapping[display_name] = internal_key
                break
    
    return mapping

mapping = create_method_mapping(scores, per_run)

print(f"\n  Mapping results ({len(mapping)}/{len(scores)} matched):")
for display, internal in mapping.items():
    print(f"    ✓ '{display}' → '{internal}'")

# Check for unmapped methods
unmapped = [m for m in scores.keys() if m not in mapping]
if unmapped:
    print(f"\n  ✗ UNMAPPED METHODS:")
    for m in unmapped:
        print(f"    - '{m}' (no match found in per_run_metrics.json)")

# ============================================================
# CHECK 4: Sample Data Availability
# ============================================================
print("\n[CHECK 4] Checking sample data for each method...")

for display_method, internal_key in mapping.items():
    if internal_key in per_run:
        samples = per_run[internal_key]
        print(f"\n  {display_method}:")
        print(f"    Internal key: '{internal_key}'")
        print(f"    Samples available: {list(samples.keys())}")
        
        # Show one sample's data
        if samples:
            sample_id = list(samples.keys())[0]
            sample_data = samples[sample_id]
            print(f"    Sample '{sample_id}' has metrics:")
            for metric, value in sample_data.items():
                print(f"      - {metric}: {value}")
    else:
        print(f"\n  ✗ {display_method}: Internal key '{internal_key}' NOT in per_run_metrics.json")

# ============================================================
# CHECK 5: Image Files
# ============================================================
print("\n[CHECK 5] Checking for visualization images...")

outputs_dir = Path("outputs")
if not outputs_dir.exists():
    print("  ✗ outputs/ directory does not exist")
else:
    print("  ✓ outputs/ directory exists")
    
    # Check subdirectories
    subdirs = ["comparison_panels", "error_overlays", "charts", "visualization"]
    
    for subdir in subdirs:
        subdir_path = outputs_dir / subdir
        if subdir_path.exists():
            images = list(subdir_path.glob("*.png"))
            print(f"\n  {subdir}/:")
            if images:
                print(f"    ✓ Found {len(images)} images")
                for img in images[:5]:  # Show first 5
                    print(f"      - {img.name}")
                if len(images) > 5:
                    print(f"      ... and {len(images) - 5} more")
            else:
                print(f"    ⚠ Directory exists but no PNG files found")
        else:
            print(f"\n  {subdir}/:")
            print(f"    ✗ Directory does not exist")

# ============================================================
# CHECK 6: Data Compatibility Test
# ============================================================
print("\n[CHECK 6] Testing data compatibility with dashboard views...")

# Test degradation curve data
print("\n  Testing Degradation Curve data:")
degradation_ok = True
for display_method, internal_key in mapping.items():
    if internal_key not in per_run:
        print(f"    ✗ {display_method}: No per_run data")
        degradation_ok = False
    else:
        samples = per_run[internal_key]
        ktc_scores = [s.get('ktc_score', None) for s in samples.values()]
        if None in ktc_scores:
            print(f"    ✗ {display_method}: Missing ktc_score in some samples")
            degradation_ok = False
        else:
            print(f"    ✓ {display_method}: {len(ktc_scores)} samples with KTC scores")

if degradation_ok:
    print("  ✓ Degradation Curve should work")
else:
    print("  ✗ Degradation Curve will have issues")

# Test comparison data
print("\n  Testing Comparison view data:")
comparison_ok = True
required_metrics = ['ktc_score', 'dice_resistive', 'dice_conductive', 'iou_resistive', 'iou_conductive']
for display_method, internal_key in mapping.items():
    if internal_key not in per_run:
        print(f"    ✗ {display_method}: No per_run data")
        comparison_ok = False
    else:
        samples = per_run[internal_key]
        first_sample = list(samples.values())[0] if samples else {}
        missing = [m for m in required_metrics if m not in first_sample]
        if missing:
            print(f"    ✗ {display_method}: Missing metrics: {missing}")
            comparison_ok = False
        else:
            print(f"    ✓ {display_method}: All required metrics present")

if comparison_ok:
    print("  ✓ Comparison view should work")
else:
    print("  ✗ Comparison view will have issues")

# Test failure gallery data
print("\n  Testing Failure Gallery data:")
gallery_ok = degradation_ok  # Same requirements
if gallery_ok:
    print("  ✓ Failure Gallery should work")
else:
    print("  ✗ Failure Gallery will have issues")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_ok = degradation_ok and comparison_ok and gallery_ok and len(mapping) == len(scores)

if all_ok:
    print("\n✓ ALL CHECKS PASSED")
    print("\nYour data looks good! If views are still empty, the issue might be:")
    print("  1. Browser cache - Try hard refresh (Ctrl+Shift+R)")
    print("  2. Streamlit cache - Stop and restart: streamlit run app.py")
    print("  3. Check browser console for JavaScript errors")
else:
    print("\n✗ ISSUES FOUND")
    print("\nProblems detected:")
    if len(mapping) != len(scores):
        print(f"  - Only {len(mapping)}/{len(scores)} methods mapped correctly")
    if not degradation_ok:
        print("  - Degradation Curve data incomplete")
    if not comparison_ok:
        print("  - Comparison view data incomplete")
    print("\nFIX: Run 'python example_usage.py' to regenerate data files")

print("\n" + "=" * 70)
