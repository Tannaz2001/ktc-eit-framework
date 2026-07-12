"""
diagnose_dashboard.py - Find out exactly why views aren't working

Standalone CLI diagnostic for the dashboard's data pipeline. Kept
dependency-free (stdlib only) so it can be run to check *why* a dashboard
view might be empty without needing Streamlit/pandas/plotly installed or
importable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def create_method_mapping(scores: dict[str, Any], per_run: dict[str, Any]) -> dict[str, str]:
    """Map scores.json display names to per_run_metrics.json internal keys."""
    mapping: dict[str, str] = {}

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


def check_data_files_exist() -> None:
    """CHECK 1: verify scores.json and outputs/per_run_metrics.json exist.

    Exits the process (status 1) if either is missing — every later check
    assumes both files are already loadable.
    """
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


def load_data() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load scores.json and outputs/per_run_metrics.json."""
    with open("scores.json", 'r') as f:
        scores = json.load(f)

    with open("outputs/per_run_metrics.json", 'r') as f:
        per_run = json.load(f)

    return scores, per_run


def check_data_structure(scores: dict[str, Any], per_run: dict[str, Any]) -> None:
    """CHECK 2: print the shape of both loaded JSON files."""
    print("\n[CHECK 2] Loading and inspecting data structure...")

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


def check_method_mapping(scores: dict[str, Any], per_run: dict[str, Any]) -> dict[str, str]:
    """CHECK 3: run create_method_mapping() and report matched/unmapped methods."""
    print("\n[CHECK 3] Testing method name mapping...")

    scores_methods = list(scores.keys())
    per_run_methods = list(per_run.keys())

    print(f"\n  Display names (from scores.json):")
    for name in scores_methods:
        print(f"    - '{name}'")

    print(f"\n  Internal keys (from per_run_metrics.json):")
    for key in per_run_methods:
        print(f"    - '{key}'")

    mapping = create_method_mapping(scores, per_run)

    print(f"\n  Mapping results ({len(mapping)}/{len(scores)} matched):")
    for display, internal in mapping.items():
        print(f"    ✓ '{display}' → '{internal}'")

    unmapped = [m for m in scores.keys() if m not in mapping]
    if unmapped:
        print(f"\n  ✗ UNMAPPED METHODS:")
        for m in unmapped:
            print(f"    - '{m}' (no match found in per_run_metrics.json)")

    return mapping


def check_sample_data(mapping: dict[str, str], per_run: dict[str, Any]) -> None:
    """CHECK 4: print sample availability and one example sample's metrics per mapped method."""
    print("\n[CHECK 4] Checking sample data for each method...")

    for display_method, internal_key in mapping.items():
        if internal_key in per_run:
            samples = per_run[internal_key]
            print(f"\n  {display_method}:")
            print(f"    Internal key: '{internal_key}'")
            print(f"    Samples available: {list(samples.keys())}")

            if samples:
                sample_id = list(samples.keys())[0]
                sample_data = samples[sample_id]
                print(f"    Sample '{sample_id}' has metrics:")
                for metric, value in sample_data.items():
                    print(f"      - {metric}: {value}")
        else:
            print(f"\n  ✗ {display_method}: Internal key '{internal_key}' NOT in per_run_metrics.json")


def check_visualization_images() -> None:
    """CHECK 5: report which output image subdirectories exist and how many PNGs each has."""
    print("\n[CHECK 5] Checking for visualization images...")

    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("  ✗ outputs/ directory does not exist")
        return

    print("  ✓ outputs/ directory exists")

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


def check_data_compatibility(
    mapping: dict[str, str], per_run: dict[str, Any]
) -> tuple[bool, bool, bool]:
    """CHECK 6: verify mapped methods have what the Degradation Curve,
    Comparison, and Failure Gallery views need.

    Returns (degradation_ok, comparison_ok, gallery_ok).
    """
    print("\n[CHECK 6] Testing data compatibility with dashboard views...")

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

    print("\n  Testing Comparison view data:")
    comparison_ok = True
    required_metrics = ['ktc_score']
    for display_method, internal_key in mapping.items():
        if internal_key not in per_run:
            print(f"    ✗ {display_method}: No per_run data")
            comparison_ok = False
        else:
            samples = per_run[internal_key]
            first_sample = list(samples.values())[0] if samples else {}
            missing = [m for m in required_metrics if m not in first_sample]
            if missing:
                print(f"    ✗ {display_method}: Missing KTC score")
                comparison_ok = False
            else:
                print(f"    ✓ {display_method}: KTC score present")

    if comparison_ok:
        print("  ✓ Comparison view should work")
    else:
        print("  ✗ Comparison view will have issues")

    print("\n  Testing Failure Gallery data:")
    gallery_ok = degradation_ok  # Same requirements
    if gallery_ok:
        print("  ✓ Failure Gallery should work")
    else:
        print("  ✗ Failure Gallery will have issues")

    return degradation_ok, comparison_ok, gallery_ok


def print_summary(
    mapping: dict[str, str],
    scores: dict[str, Any],
    degradation_ok: bool,
    comparison_ok: bool,
    gallery_ok: bool,
) -> None:
    """Print the final pass/fail summary and next-step guidance."""
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


def main() -> None:
    print("=" * 70)
    print("DASHBOARD DIAGNOSTIC TOOL")
    print("=" * 70)

    check_data_files_exist()
    scores, per_run = load_data()

    check_data_structure(scores, per_run)
    mapping = check_method_mapping(scores, per_run)
    check_sample_data(mapping, per_run)
    check_visualization_images()
    degradation_ok, comparison_ok, gallery_ok = check_data_compatibility(mapping, per_run)

    print_summary(mapping, scores, degradation_ok, comparison_ok, gallery_ok)


if __name__ == "__main__":
    main()
