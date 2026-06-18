"""Hull Plugin — Convex hull geometry extraction and comparison.

This is an ANALYSIS plugin, not a MethodPlugin. It operates on
already-produced (256, 256) segmentation arrays with labels {0, 1, 2}.

It extracts convex hulls from each class (resistive=1, conductive=2),
compares predicted vs ground-truth hulls via rasterized IoU, and produces
qualitative detection summaries.

Design rationale:
- Rasterization-based IoU avoids polygon clipping (no new deps, robust to edge cases)
- Connected component filtering removes noise (noisy methods won't inflate hull areas)
- Detection threshold 0.3 (vs standard 0.5) accounts for EIT's physics-limited resolution
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from skimage.draw import polygon as draw_polygon
from skimage.measure import label as connected_components


class HullDescriptor:
    """Geometric descriptor for one class's convex hull."""

    __slots__ = (
        "area_px",
        "centroid",
        "vertex_count",
        "hull_points",
        "perimeter_px",
        "empty",
    )

    def __init__(
        self,
        area_px: float = 0.0,
        centroid: tuple = (0.0, 0.0),
        vertex_count: int = 0,
        hull_points: np.ndarray | None = None,
        perimeter_px: float = 0.0,
        empty: bool = True,
    ):
        self.area_px = area_px
        self.centroid = centroid
        self.vertex_count = vertex_count
        self.hull_points = hull_points if hull_points is not None else np.empty((0, 2))
        self.perimeter_px = perimeter_px
        self.empty = empty


class HullAnalyzer:
    """Extracts convex hulls from segmentations and compares them."""

    GRID_SHAPE = (256, 256)
    MIN_COMPONENT_SIZE = 50  # Pixels in largest connected component to avoid noise

    def _filter_noise(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component if >= MIN_COMPONENT_SIZE pixels.

        This is critical for robustness: noisy algorithms can produce scattered pixels
        that form huge but meaningless convex hulls. By filtering, we ensure that
        small noise patches don't inflate detection metrics.

        Args:
            mask: (256, 256) boolean array where True = class pixels

        Returns:
            (256, 256) boolean array with only the largest component (or all zeros)
        """
        labeled = connected_components(mask)
        if labeled.max() == 0:
            return mask  # all zeros, no components

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background (label 0)

        if sizes.size == 0 or sizes.max() < self.MIN_COMPONENT_SIZE:
            return np.zeros_like(mask)  # treat as noise

        largest = sizes.argmax()
        return (labeled == largest).astype(mask.dtype)

    def extract(
        self, segmentation: np.ndarray, class_id: int
    ) -> HullDescriptor:
        """Extract convex hull for a single class from a (256,256) label array.

        Args:
            segmentation: (256, 256) uint8 array with labels {0, 1, 2}
            class_id: which class to extract (1=resistive, 2=conductive)

        Returns:
            HullDescriptor with geometry, or empty descriptor if class
            has fewer than 3 non-collinear pixels.
        """
        # Step 1: Extract pixel coordinates where segmentation == class_id
        mask = segmentation == class_id

        # Step 2: Filter noise (keep only largest connected component)
        mask = self._filter_noise(mask)

        coords = np.argwhere(mask)  # shape (N, 2), row-col format

        # Step 3: Edge case: fewer than 3 points
        if len(coords) < 3:
            return HullDescriptor(empty=True)

        # Step 4: Compute ConvexHull (with error handling for degenerate cases)
        try:
            hull = ConvexHull(coords)
        except QhullError:
            # Collinear points or other degenerate geometry
            return HullDescriptor(empty=True)

        # Step 5: Extract descriptors
        # CRITICAL: scipy quirk — in 2D, hull.volume = area, hull.area = perimeter
        area_px = float(hull.volume)
        perimeter_px = float(hull.area)
        hull_points = coords[hull.vertices]  # ordered boundary vertices
        centroid = coords.mean(axis=0)  # mean of ALL class pixels, not just vertices
        vertex_count = len(hull.vertices)

        return HullDescriptor(
            area_px=area_px,
            centroid=tuple(centroid),
            vertex_count=vertex_count,
            hull_points=hull_points,
            perimeter_px=perimeter_px,
            empty=False,
        )

    def _rasterize_hull(self, hull_points: np.ndarray) -> np.ndarray:
        """Rasterize a convex hull polygon onto a (256, 256) boolean mask.

        Design: compute IoU via pixel counting (rasterization), not polygon clipping.
        This avoids floating-point edge cases and requires no new dependencies.

        Args:
            hull_points: (V, 2) array of hull vertex coordinates in row-col format

        Returns:
            (256, 256) boolean mask where True = inside hull
        """
        if len(hull_points) < 3:
            return np.zeros(self.GRID_SHAPE, dtype=bool)

        mask = np.zeros(self.GRID_SHAPE, dtype=bool)

        # skimage.draw.polygon expects (row, col) ordering
        rr, cc = draw_polygon(
            hull_points[:, 0], hull_points[:, 1], shape=self.GRID_SHAPE
        )

        mask[rr, cc] = True
        return mask

    def compare(
        self, pred_desc: HullDescriptor, gt_desc: HullDescriptor
    ) -> dict:
        """Compare predicted hull vs ground-truth hull.

        Returns:
            dict with keys:
            - hull_iou: float (0.0 to 1.0) — rasterized IoU of the two hulls
            - centroid_distance_px: float — Euclidean distance between centroids
            - area_ratio: float — pred_area / gt_area (1.0 = perfect)
            - detected: bool — whether the prediction qualifies as a valid detection
            - detection_reason: str — why it was or wasn't detected

        Detection logic:
            detected = True iff hull_iou >= DETECTION_THRESHOLD AND pred is not empty
            Threshold 0.3 justifies lower bar than PASCAL VOC (0.5) because EIT
            reconstructions are inherently blurry (ill-posed inverse problem).
        """
        DETECTION_THRESHOLD = 0.3

        # Edge cases (handle FIRST, before any math)
        if gt_desc.empty and pred_desc.empty:
            return {
                "hull_iou": 0.0,
                "centroid_distance_px": 0.0,
                "area_ratio": 1.0,
                "detected": False,
                "detection_reason": "no_class_in_gt_or_pred",
            }

        if gt_desc.empty and not pred_desc.empty:
            return {
                "hull_iou": 0.0,
                "centroid_distance_px": 0.0,
                "area_ratio": 0.0,
                "detected": False,
                "detection_reason": "false_positive",
            }

        if gt_desc.empty or pred_desc.empty:  # GT non-empty but pred empty
            return {
                "hull_iou": 0.0,
                "centroid_distance_px": float(
                    np.linalg.norm(np.array(gt_desc.centroid))
                ),
                "area_ratio": 0.0,
                "detected": False,
                "detection_reason": "missed",
            }

        # Both non-empty: compute rasterized IoU
        gt_mask = self._rasterize_hull(gt_desc.hull_points)
        pred_mask = self._rasterize_hull(pred_desc.hull_points)

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        hull_iou = float(intersection / union) if union > 0 else 0.0

        centroid_distance = float(
            np.linalg.norm(np.array(pred_desc.centroid) - np.array(gt_desc.centroid))
        )

        area_ratio = (
            float(pred_desc.area_px / gt_desc.area_px)
            if gt_desc.area_px > 0
            else 0.0
        )

        detected = hull_iou >= DETECTION_THRESHOLD
        reason = "detected" if detected else "below_iou_threshold"

        return {
            "hull_iou": hull_iou,
            "centroid_distance_px": centroid_distance,
            "area_ratio": area_ratio,
            "detected": detected,
            "detection_reason": reason,
        }
