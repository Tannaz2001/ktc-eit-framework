#!/usr/bin/env python3
"""
Establish ground truth baseline scores from training data.

This script:
1. Loads the 4 real KTC training samples (we have these locally)
2. Runs all methods on each sample
3. Computes KTC scores
4. Determines what "normal" performance looks like
5. Generates baseline table for test harness to compare against

These baselines are then used in test_harness.py to flag methods that
deviate significantly from expected performance.

Usage:
    python3 establish_baselines.py

Output:
    baselines.json - scores for each method on each training sample
    BASELINE_REPORT.md - human-readable baseline table
"""

import json
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

print("=" * 80)
print("ESTABLISHING BASELINE SCORES FROM TRAINING DATA")
print("=" * 80)

try:
    from src.ktc_framework.loaders.training_data_plugin import TrainingDataPlugin
    from src.ktc_framework.metrics.ktc_score import compute_ktc_score

    # Load training data
    print("\nLoading training data plugin...")
    loader = TrainingDataPlugin()

    # Get available samples
    training_samples = ["1", "2", "3", "4"]

    baselines = {}

    print(f"Found {len(training_samples)} training samples")

    for sample_id in training_samples:
        print(f"\n[Sample {sample_id}]")

        try:
            batch = loader.load_sample(sample=sample_id)

            if batch is None or batch.ground_truth is None:
                print(f"  Could not load sample {sample_id}")
                continue

            gt = batch.ground_truth
            print(f"  GT shape: {gt.shape}, labels: {np.unique(gt)}")

            baselines[f"sample_{sample_id}"] = {
                "gt_shape": list(gt.shape),
                "gt_labels": list(np.unique(gt)),
                "methods": {},
            }

            # Try each method
            methods_to_test = [
                ("MockBaseline", "mock_method_plugin"),
                ("BackProjection", "backprojection"),
                ("GaussNewton", "gauss_newton"),
            ]

            for method_name, module_name in methods_to_test:
                try:
                    if module_name == "mock_method_plugin":
                        from src.ktc_framework.methods.mock_method_plugin import MockMethod
                        method = MockMethod()
                    elif module_name == "backprojection":
                        from src.ktc_framework.methods.backprojection import BackProjection
                        method = BackProjection()
                    elif module_name == "gauss_newton":
                        from src.ktc_framework.methods.gauss_newton import GaussNewton
                        method = GaussNewton()
                    else:
                        continue

                    print(f"  {method_name:20} ", end="", flush=True)

                    pred = method.reconstruct(batch)
                    score = compute_ktc_score(pred, gt)

                    print(f"  score={score:+.6f}")

                    baselines[f"sample_{sample_id}"]["methods"][method_name] = {
                        "score": float(score),
                        "pred_shape": list(pred.shape),
                        "pred_labels": list(np.unique(pred).astype(int)),
                    }

                except Exception as e:
                    print(f"  {method_name:20} FAILED: {e}")
                    baselines[f"sample_{sample_id}"]["methods"][method_name] = {
                        "error": str(e)
                    }

        except Exception as e:
            print(f"  Error loading sample: {e}")
            continue

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    all_scores = {}
    for sample_key, sample_data in baselines.items():
        for method_name, method_data in sample_data.get("methods", {}).items():
            if "score" in method_data:
                if method_name not in all_scores:
                    all_scores[method_name] = []
                all_scores[method_name].append(method_data["score"])

    print("\nPer-method statistics (across all training samples):")
    for method_name, scores in all_scores.items():
        scores = np.array(scores)
        print(f"\n{method_name}:")
        print(f"  Mean:  {scores.mean():+.6f}")
        print(f"  Std:   {scores.std():.6f}")
        print(f"  Min:   {scores.min():+.6f}")
        print(f"  Max:   {scores.max():+.6f}")
        print(f"  Range: [{scores.min():+.6f}, {scores.max():+.6f}]")

    # Save baselines
    baselines_file = Path("baselines.json")
    with open(baselines_file, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\n✓ Baselines saved to: {baselines_file}")

    # Generate markdown report
    report = []
    report.append("# Established Baselines (Training Data)\n")
    report.append("These baselines are computed from the 4 KTC training samples.\n")
    report.append("The test harness uses these to flag methods that deviate significantly.\n\n")

    report.append("## Per-Sample Scores\n\n")
    report.append("| Sample | MockBaseline | BackProjection | GaussNewton |\n")
    report.append("|--------|--------------|----------------|-------------|\n")

    for sample_id in training_samples:
        key = f"sample_{sample_id}"
        if key in baselines:
            methods_data = baselines[key].get("methods", {})
            mock_score = methods_data.get("MockBaseline", {}).get("score", "N/A")
            bp_score = methods_data.get("BackProjection", {}).get("score", "N/A")
            gn_score = methods_data.get("GaussNewton", {}).get("score", "N/A")

            mock_str = f"{mock_score:+.6f}" if isinstance(mock_score, float) else str(mock_score)
            bp_str = f"{bp_score:+.6f}" if isinstance(bp_score, float) else str(bp_score)
            gn_str = f"{gn_score:+.6f}" if isinstance(gn_score, float) else str(gn_score)

            report.append(f"| {sample_id} | {mock_str} | {bp_str} | {gn_str} |\n")

    report.append("\n## Method Statistics\n\n")

    for method_name, scores in all_scores.items():
        scores = np.array(scores)
        report.append(f"### {method_name}\n\n")
        report.append(f"- **Mean**: {scores.mean():+.6f}\n")
        report.append(f"- **Std**: {scores.std():.6f}\n")
        report.append(f"- **Range**: [{scores.min():+.6f}, {scores.max():+.6f}]\n")
        report.append(f"- **Confidence interval (95%)**: [{scores.mean() - 1.96*scores.std():+.6f}, {scores.mean() + 1.96*scores.std():+.6f}]\n\n")

    report.append("## Interpretation\n\n")
    report.append("These baselines represent what we can achieve on the **training set**.\n\n")
    report.append("On the **evaluation set** (especially higher levels), scores will naturally be lower due to:\n")
    report.append("- Reduced measurement coverage (Level 4 has 71% data loss)\n")
    report.append("- Different noise characteristics\n")
    report.append("- Generalization to unseen conductivity patterns\n\n")
    report.append("**Expected degradation**: 30-70% lower scores at Level 4+ compared to training data.\n")

    report_file = Path("BASELINE_REPORT.md")
    with open(report_file, "w") as f:
        f.writelines(report)

    print(f"✓ Report saved to: {report_file}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Next: Use baselines.json as reference in test_harness.py")
print("=" * 80)
