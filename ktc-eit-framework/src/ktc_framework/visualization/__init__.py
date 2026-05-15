"""
visualization/__init__.py
--------------------------
Plot helpers for EIT reconstruction results.

Main entry points
-----------------
plot_sample(gt, pred)          — side-by-side ground-truth vs prediction panel
plot_overlay(gt, pred)         — difference overlay on a single axis
save_panel(gt, pred, path)     — write a panel PNG to disk
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Label → display colour  (matches KTC convention)
_CMAP = {0: "#2196F3", 1: "#F44336", 2: "#4CAF50"}  # blue, red, green
_LABEL_NAMES = {0: "Background", 1: "Resistive", 2: "Conductive"}

# Discrete colourmap shared by all plots
_COLOURS = [_CMAP[k] for k in sorted(_CMAP)]


def _discrete_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(_COLOURS, name="ktc")


def plot_sample(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_id: str = "",
    show: bool = True,
) -> plt.Figure:
    """Side-by-side ground-truth / prediction panel.

    Parameters
    ----------
    gt, pred : np.ndarray
        256×256 integer arrays with labels {0, 1, 2}.
    sample_id : str
        Optional title suffix.
    show : bool
        Call plt.show() if True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cmap = _discrete_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmin, vmax = 0, 2

    axes[0].imshow(gt,   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("Ground Truth")

    axes[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("Prediction")

    diff = (pred != gt).astype(np.uint8)
    axes[2].imshow(diff, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    accuracy = 100 * (1 - diff.mean())
    axes[2].set_title(f"Errors  (px acc {accuracy:.1f}%)")

    for ax in axes:
        ax.axis("off")

    title = f"Sample: {sample_id}" if sample_id else "EIT Reconstruction"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    _add_legend(fig)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_id: str = "",
    show: bool = True,
) -> plt.Figure:
    """Single-axis overlay: ground-truth contours on top of prediction fill.

    Parameters
    ----------
    gt, pred : np.ndarray
        256×256 integer arrays with labels {0, 1, 2}.
    """
    cmap = _discrete_cmap()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pred, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", alpha=0.8)

    # Draw GT boundary contours per class
    from matplotlib.colors import to_rgba
    for lbl, colour in _CMAP.items():
        if lbl == 0:
            continue  # skip background contour
        mask = (gt == lbl).astype(np.uint8)
        ax.contour(mask, levels=[0.5], colors=[colour], linewidths=1.5)

    ax.set_title(f"Overlay{' — ' + sample_id if sample_id else ''}")
    ax.axis("off")
    _add_legend(fig)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def save_panel(
    gt: np.ndarray,
    pred: np.ndarray,
    path: str | Path,
    sample_id: str = "",
    dpi: int = 150,
) -> Path:
    """Write a plot_sample panel to *path* as a PNG.

    Returns
    -------
    Path
        Resolved path of the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_sample(gt, pred, sample_id=sample_id, show=False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_legend(fig: plt.Figure) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(color=_CMAP[k], label=_LABEL_NAMES[k]) for k in sorted(_CMAP)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
