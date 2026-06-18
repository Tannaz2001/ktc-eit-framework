"""Qualitative metrics — aggregates hull detection results.

This module computes detection flags for individual samples and aggregates
them across all samples for a method. It reads boolean detection flags
produced by HullAnalyzer.compare() and counts them.

Does NOT perform any reconstruction. Operates only on outputs of hull analysis.
"""

from __future__ import annotations

import numpy as np


def compute_qualitative_sample(
    pred: np.ndarray, gt: np.ndarray, hull_analyzer
) -> dict:
    """Compute qualitative detection flags for ONE sample.

    Args:
        pred: (256, 256) uint8, predicted segmentation
        gt: (256, 256) uint8, ground truth segmentation
        hull_analyzer: HullAnalyzer instance

    Returns:
        dict with detection flags for both resistive (class=1) and
        conductive (class=2) classes.
    """
    result = {}

    # Process each class (1=resistive, 2=conductive)
    for class_id in [1, 2]:
        class_name = "resistive" if class_id == 1 else "conductive"

        gt_hull = hull_analyzer.extract(gt, class_id)
        pred_hull = hull_analyzer.extract(pred, class_id)

        comparison = hull_analyzer.compare(pred_hull, gt_hull)

        result[f"{class_name}_in_gt"] = not gt_hull.empty
        result[f"{class_name}_in_pred"] = not pred_hull.empty
        result[f"{class_name}_detected"] = comparison["detected"]
        result[f"{class_name}_hull_iou"] = comparison["hull_iou"]
        result[f"{class_name}_centroid_dist"] = comparison["centroid_distance_px"]

        # False positive: pred has class but GT doesn't
        result[f"false_positive_{class_name}"] = (
            comparison["detection_reason"] == "false_positive"
        )

    return result


def aggregate_qualitative(all_sample_results: list[dict]) -> dict:
    """Aggregate qualitative flags across ALL samples for ONE method.

    Args:
        all_sample_results: list of dicts from compute_qualitative_sample()

    Returns:
        dict with aggregated counts, percentages, and averages.
    """
    if not all_sample_results:
        return {}

    total_samples = len(all_sample_results)

    # Initialize accumulators
    resistive_detected_count = 0
    resistive_gt_count = 0
    conductive_detected_count = 0
    conductive_gt_count = 0
    false_positive_count = 0

    resistive_iou_sum = 0.0
    resistive_iou_count = 0
    conductive_iou_sum = 0.0
    conductive_iou_count = 0

    resistive_centroid_sum = 0.0
    resistive_centroid_count = 0
    conductive_centroid_sum = 0.0
    conductive_centroid_count = 0

    # Aggregate across all samples
    for sample in all_sample_results:
        # Resistive class
        if sample["resistive_in_gt"]:
            resistive_gt_count += 1
            if sample["resistive_detected"]:
                resistive_detected_count += 1
            resistive_iou_sum += sample["resistive_hull_iou"]
            resistive_iou_count += 1
            resistive_centroid_sum += sample["resistive_centroid_dist"]
            resistive_centroid_count += 1

        # Conductive class
        if sample["conductive_in_gt"]:
            conductive_gt_count += 1
            if sample["conductive_detected"]:
                conductive_detected_count += 1
            conductive_iou_sum += sample["conductive_hull_iou"]
            conductive_iou_count += 1
            conductive_centroid_sum += sample["conductive_centroid_dist"]
            conductive_centroid_count += 1

        # False positives (count any occurrence)
        false_positive_count += int(
            sample["false_positive_resistive"] or sample["false_positive_conductive"]
        )

    # Compute percentages and averages
    resistive_detected_pct = (
        100.0 * resistive_detected_count / resistive_gt_count
        if resistive_gt_count > 0
        else 0.0
    )
    conductive_detected_pct = (
        100.0 * conductive_detected_count / conductive_gt_count
        if conductive_gt_count > 0
        else 0.0
    )

    avg_resistive_hull_iou = (
        resistive_iou_sum / resistive_iou_count if resistive_iou_count > 0 else 0.0
    )
    avg_conductive_hull_iou = (
        conductive_iou_sum / conductive_iou_count if conductive_iou_count > 0 else 0.0
    )

    avg_resistive_centroid_dist = (
        resistive_centroid_sum / resistive_centroid_count
        if resistive_centroid_count > 0
        else 0.0
    )
    avg_conductive_centroid_dist = (
        conductive_centroid_sum / conductive_centroid_count
        if conductive_centroid_count > 0
        else 0.0
    )

    return {
        "total_samples": total_samples,
        "resistive_gt_count": resistive_gt_count,
        "resistive_detected_count": resistive_detected_count,
        "resistive_detected_str": f"{resistive_detected_count}/{resistive_gt_count}",
        "resistive_detected_pct": round(resistive_detected_pct, 1),
        "conductive_gt_count": conductive_gt_count,
        "conductive_detected_count": conductive_detected_count,
        "conductive_detected_str": f"{conductive_detected_count}/{conductive_gt_count}",
        "conductive_detected_pct": round(conductive_detected_pct, 1),
        "false_positive_count": false_positive_count,
        "avg_resistive_hull_iou": round(avg_resistive_hull_iou, 3),
        "avg_conductive_hull_iou": round(avg_conductive_hull_iou, 3),
        "avg_resistive_centroid_dist": round(avg_resistive_centroid_dist, 1),
        "avg_conductive_centroid_dist": round(avg_conductive_centroid_dist, 1),
    }
