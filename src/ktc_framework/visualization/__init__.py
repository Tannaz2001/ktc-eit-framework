"""
visualization/__init__.py
-------------------------
Saves side-by-side PNG panels: ground_truth | prediction

Colour map (matches KTC convention):
  0 = background  → dark blue  (#1f4e79)
  1 = resistive   → red        (#c00000)
  2 = conductive  → green      (#375623)

Usage
-----
from src.ktc_framework.visualization import save_panel

path = save_panel(
    gt=batch.ground_truth,
    pred=reconstruction,
    method="BackProjection",
    level=1,
    sample="A",
    output_dir=Path("outputs/images"),
    ktc_score=0.312,
)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive — safe for headless / CI runs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Label → hex colour
_COLOURS = {
    0: "#1f4e79",  # background — dark blue
    1: "#c00000",  # resistive  — red
    2: "#375623",  # conductive — green
}


def _label_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert (H, W) integer label array → (H, W, 3) uint8 RGB image."""
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label, hex_col in _COLOURS.items():
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        rgb[arr == label] = [r, g, b]
    return rgb


def save_panel(
    gt: np.ndarray,
    pred: np.ndarray,
    method: str,
    level: int | str,
    sample: str,
    output_dir: Path,
    ktc_score: float = 0.0,
) -> Path:
    """Save a side-by-side PNG: ground truth (left) | prediction (right).

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth label array, shape (256, 256), values {0, 1, 2}.
    pred : np.ndarray
        Predicted label array, shape (256, 256), values {0, 1, 2}.
    method : str
        Method name — used in the panel title and filename.
    level : int | str
        Difficulty level — used in title and filename.
    sample : str
        Sample identifier — used in title and filename.
    output_dir : Path
        Directory to write the PNG into (created if absent).
    ktc_score : float
        KTC score to display in the subtitle.

    Returns
    -------
    Path
        Absolute path of the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(_label_to_rgb(gt.astype(np.uint8)))
    axes[0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(_label_to_rgb(pred.astype(np.uint8)))
    axes[1].set_title(f"Prediction — {method}", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    legend_patches = [
        mpatches.Patch(color=_COLOURS[0], label="Background"),
        mpatches.Patch(color=_COLOURS[1], label="Resistive"),
        mpatches.Patch(color=_COLOURS[2], label="Conductive"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Level {level}  |  Sample {sample}  |  KTC score: {ktc_score:.3f}",
        fontsize=10,
    )

    fname = f"{method}_level{level}_sample{sample}.png"
    out_path = output_dir / fname
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
