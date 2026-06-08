"""Hull Plugin — Post-processing geometry analysis for EIT reconstructions.

Extracts convex hulls of detected inclusions (resistive/conductive regions)
and computes geometric descriptors for error analysis.

Used to answer: "Where did the reconstruction locate the inclusions?"
"How accurate were the size/shape estimates?"

This is purely post-processing and does NOT affect KTC score.
"""

from __future__ import annotations

from typing import NamedTuple, Optional
import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import center_of_mass


class HullResult(NamedTuple):
    """Geometric analysis of a reconstruction."""

    # Resistive (label=1) region
    resistive_center: tuple[float, float] | None  # (y, x) in pixels
    resistive_area: float | None  # pixels²
    resistive_perimeter: float | None  # pixels
    resistive_hull: np.ndarray | None  # (N, 2) vertices

    # Conductive (label=2) region
    conductive_center: tuple[float, float] | None
    conductive_area: float | None
    conductive_perimeter: float | None
    conductive_hull: np.ndarray | None

    # Metadata
    prediction_shape: tuple[int, int]  # (256, 256)
    num_pixels_resistive: int
    num_pixels_conductive: int


class HullPlugin:
    """Analyze convex hulls of reconstructed inclusion regions.

    This is a post-processing utility for geometric error analysis.
    It does NOT affect KTC scores or core reconstruction logic.
    """

    @staticmethod
    def analyze(
        prediction: np.ndarray,
        ground_truth: np.ndarray | None = None,
    ) -> HullResult:
        """Compute hull geometry for resistive and conductive regions.

        Parameters
        ----------
        prediction : np.ndarray
            Shape (256, 256), dtype uint8.
            Labels: 0=water, 1=resistive, 2=conductive
        ground_truth : np.ndarray, optional
            Same format. If provided, can be used for comparison.
            (Currently not used in this version.)

        Returns
        -------
        HullResult
            Named tuple with center, area, perimeter, hull for each label.

        Raises
        ------
        ValueError
            If prediction shape is not (256, 256) or dtype is not uint8.
        """

        # Validate input
        if prediction.shape != (256, 256):
            raise ValueError(
                f"Expected shape (256, 256), got {prediction.shape}"
            )
        if prediction.dtype != np.uint8:
            raise ValueError(
                f"Expected dtype uint8, got {prediction.dtype}"
            )

        pred = np.asarray(prediction, dtype=np.uint8)

        # Analyze resistive region (label=1)
        resistive_mask = (pred == 1)
        r_center, r_area, r_perim, r_hull = HullPlugin._compute_hull(
            resistive_mask
        )

        # Analyze conductive region (label=2)
        conductive_mask = (pred == 2)
        c_center, c_area, c_perim, c_hull = HullPlugin._compute_hull(
            conductive_mask
        )

        return HullResult(
            resistive_center=r_center,
            resistive_area=r_area,
            resistive_perimeter=r_perim,
            resistive_hull=r_hull,
            conductive_center=c_center,
            conductive_area=c_area,
            conductive_perimeter=c_perim,
            conductive_hull=c_hull,
            prediction_shape=pred.shape,
            num_pixels_resistive=int(resistive_mask.sum()),
            num_pixels_conductive=int(conductive_mask.sum()),
        )

    @staticmethod
    def _compute_hull(
        mask: np.ndarray,
    ) -> tuple[
        tuple[float, float] | None,
        float | None,
        float | None,
        np.ndarray | None,
    ]:
        """Compute convex hull of a binary region.

        Parameters
        ----------
        mask : np.ndarray
            Shape (256, 256), dtype bool. True where region exists.

        Returns
        -------
        tuple
            (center, area, perimeter, hull_vertices)
            All None if fewer than 3 pixels exist (degenerate case).
        """

        # Find all pixels in the region
        region_pixels = np.where(mask)

        if region_pixels[0].size < 3:
            # Need at least 3 points for a hull
            return None, None, None, None

        # Stack as (N, 2) array of (y, x) coordinates
        points = np.column_stack(region_pixels)  # shape (N, 2)

        try:
            # Compute convex hull
            hull = ConvexHull(points)
        except Exception:
            # Degenerate case (e.g., collinear points)
            return None, None, None, None

        # Extract hull vertices
        hull_vertices = hull.points[hull.vertices]  # shape (M, 2)

        # Center of mass
        center_y, center_x = center_of_mass(mask)
        center = (float(center_y), float(center_x))

        # Area (hull.volume in 2D is the area)
        area = float(hull.volume)

        # Perimeter (sum of edge lengths)
        perimeter = HullPlugin._compute_perimeter(hull_vertices)

        return center, area, perimeter, hull_vertices

    @staticmethod
    def _compute_perimeter(vertices: np.ndarray) -> float:
        """Compute perimeter of a convex hull.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (M, 2), the hull vertices in order.

        Returns
        -------
        float
            Perimeter in pixels.
        """

        if vertices.shape[0] < 2:
            return 0.0

        # Close the hull by appending first vertex at end
        closed = np.vstack([vertices, vertices[0:1]])

        # Compute Euclidean distances between consecutive vertices
        diffs = np.diff(closed, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        return float(np.sum(distances))

    @staticmethod
    def compare_hulls(
        pred_result: HullResult,
        gt_result: HullResult,
    ) -> dict[str, float | None]:
        """Compare geometric descriptors between prediction and ground truth.

        Parameters
        ----------
        pred_result : HullResult
            Hull analysis of the reconstruction.
        gt_result : HullResult
            Hull analysis of the ground truth.

        Returns
        -------
        dict
            Error metrics:
            - resistive_center_error (pixels)
            - conductive_center_error (pixels)
            - resistive_area_error (pixels²)
            - conductive_area_error (pixels²)
            - resistive_perimeter_error (pixels)
            - conductive_perimeter_error (pixels)

        Returns None values where comparison cannot be made.
        """

        errors = {}

        # Resistive region
        if pred_result.resistive_center and gt_result.resistive_center:
            pred_c = np.array(pred_result.resistive_center)
            gt_c = np.array(gt_result.resistive_center)
            errors["resistive_center_error"] = float(np.linalg.norm(pred_c - gt_c))
        else:
            errors["resistive_center_error"] = None

        if (
            pred_result.resistive_area is not None
            and gt_result.resistive_area is not None
        ):
            errors["resistive_area_error"] = abs(
                pred_result.resistive_area - gt_result.resistive_area
            )
        else:
            errors["resistive_area_error"] = None

        if (
            pred_result.resistive_perimeter is not None
            and gt_result.resistive_perimeter is not None
        ):
            errors["resistive_perimeter_error"] = abs(
                pred_result.resistive_perimeter - gt_result.resistive_perimeter
            )
        else:
            errors["resistive_perimeter_error"] = None

        # Conductive region (same pattern)
        if pred_result.conductive_center and gt_result.conductive_center:
            pred_c = np.array(pred_result.conductive_center)
            gt_c = np.array(gt_result.conductive_center)
            errors["conductive_center_error"] = float(np.linalg.norm(pred_c - gt_c))
        else:
            errors["conductive_center_error"] = None

        if (
            pred_result.conductive_area is not None
            and gt_result.conductive_area is not None
        ):
            errors["conductive_area_error"] = abs(
                pred_result.conductive_area - gt_result.conductive_area
            )
        else:
            errors["conductive_area_error"] = None

        if (
            pred_result.conductive_perimeter is not None
            and gt_result.conductive_perimeter is not None
        ):
            errors["conductive_perimeter_error"] = abs(
                pred_result.conductive_perimeter - gt_result.conductive_perimeter
            )
        else:
            errors["conductive_perimeter_error"] = None

        return errors


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Example usage."""

    # Create a dummy prediction (circle in center)
    pred = np.zeros((256, 256), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    center = (128, 128)
    radius = 30
    circle_mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2
    pred[circle_mask] = 2  # Conductive

    # Create ground truth (slightly offset circle)
    gt = np.zeros((256, 256), dtype=np.uint8)
    center_gt = (130, 128)
    circle_mask_gt = (
        (y - center_gt[0]) ** 2 + (x - center_gt[1]) ** 2 <= (radius + 2) ** 2
    )
    gt[circle_mask_gt] = 2

    # Analyze
    result = HullPlugin.analyze(pred, gt)
    print("Prediction Hull Result:")
    print(f"  Conductive center: {result.conductive_center}")
    print(f"  Conductive area:   {result.conductive_area:.1f}")
    print(f"  Conductive perim:  {result.conductive_perimeter:.1f}")

    result_gt = HullPlugin.analyze(gt, gt)
    errors = HullPlugin.compare_hulls(result, result_gt)
    print("\nComparison Errors:")
    for key, val in errors.items():
        if val is not None:
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: None")
