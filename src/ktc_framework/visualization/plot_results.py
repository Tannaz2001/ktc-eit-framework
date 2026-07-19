"""Visualization helpers — integrated from Areeba's viz.py into the main framework."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for CLI use
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch
import seaborn as sns

# ── EIT colour scheme ──────────────────────────────────────────────────────
COLORMAP = {
    0: "#1a3a5c",   # water
    1: "#D85A30",   # resistive
    2: "#1D9E75",   # conductive
}

GRADE_COLORS = {
    "A": "#1D9E75",
    "B": "#4A90E2",
    "C": "#F5A623",
    "D": "#D85A30",
}

_EIT_CMAP = ListedColormap([COLORMAP[0], COLORMAP[1], COLORMAP[2]])

_logger = logging.getLogger(__name__)

# Deterministic per-method colors — stable across all charts in one run
_PALETTE = [
    "#1D9E75", "#D85A30", "#4A90E2", "#F5A623",
    "#9B59B6", "#E74C3C", "#16A085", "#E67E22",
]


def _method_color(method: str) -> str:
    """Hash method name to a palette color — same name always gets same color."""
    return _PALETTE[sum(ord(c) for c in method) % len(_PALETTE)]


# ── Panel plot ─────────────────────────────────────────────────────────────

def plot_panel(
    pred: np.ndarray,
    gt: np.ndarray,
    method: str = "",
    level: int = 0,
    sample: str = "",
    ktc_score: float = 0.0,
    save_path: str | Path = "outputs/panel.png",
) -> None:
    """Save GT | Prediction | Error Map panel as PNG."""
    error = pred != gt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"{method}  |  Level {level}  Sample {sample}  |  KTC: {ktc_score:.3f}",
        fontsize=11, fontweight="bold",
    )

    axes[0].imshow(gt,    cmap=_EIT_CMAP, vmin=0, vmax=2)
    axes[0].set_title("Ground Truth"); axes[0].axis("off")

    axes[1].imshow(pred,  cmap=_EIT_CMAP, vmin=0, vmax=2)
    axes[1].set_title("Prediction");   axes[1].axis("off")

    axes[2].imshow(error, cmap="Reds")
    axes[2].set_title("Error Map");    axes[2].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and rename over the target instead of
    # overwriting it in place. A direct overwrite can hit a sharing-violation
    # PermissionError on a Windows host (via Docker Desktop's bind mount) if
    # the existing file is open elsewhere (e.g. the dashboard displaying it);
    # os.replace succeeds even then, since it never opens the old file.
    tmp_path = save_path.with_name(f".{save_path.name}.tmp")
    try:
        plt.savefig(tmp_path, dpi=150, bbox_inches="tight")
        os.replace(tmp_path, save_path)
    finally:
        plt.close(fig)
        tmp_path.unlink(missing_ok=True)


# ── Runner integration: save one PNG per result ────────────────────────────

def save_figures(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Save one comparison PNG per result. Returns list of saved paths."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for r in results:
        gt   = r.get("_gt")
        pred = r.get("_pred")
        if gt is None or pred is None:
            continue

        fname = f"{r['method']}_level{r['level']}_sample{r['sample']}.png"
        path  = fig_dir / fname

        try:
            plot_panel(
                pred=pred, gt=gt,
                method=r["method"], level=r["level"], sample=r["sample"],
                ktc_score=r.get("metrics", {}).get("ktc_score", 0.0),
                save_path=path,
            )
        except OSError as exc:
            # One locked/inaccessible file (e.g. a Windows sharing violation
            # while the dashboard has it open) must not cancel every other
            # method/sample's figure in this run.
            _logger.warning("Skipping figure %s: %s", path, exc)
            continue
        saved.append(path)

    return saved


# ── Failure gallery ────────────────────────────────────────────────────────

def plot_failure_gallery(
    results: list[dict[str, Any]],
    output_dir: Path,
    n: int = 3,
) -> Path | None:
    """Save a gallery of the n worst samples per method."""
    if not results:
        return None

    methods = list({r["method"] for r in results})
    worst_per_method: dict[str, list] = {}

    for method in methods:
        method_results = [
            r for r in results
            if r["method"] == method
            and r.get("_gt") is not None
            and r.get("_pred") is not None
        ]
        if not method_results:
            continue
        worst = sorted(method_results, key=lambda r: r.get("metrics", {}).get("ktc_score", 0.0))[:n]
        worst_per_method[method] = worst

    if not worst_per_method:
        return None

    total_rows = sum(len(v) for v in worst_per_method.values())
    fig, axes = plt.subplots(total_rows, 2, figsize=(8, 3.5 * total_rows))
    if total_rows == 1:
        axes = [axes]

    fig.suptitle("Failure Gallery — Worst Samples per Method", fontsize=12, fontweight="bold")

    row = 0
    for method, samples in worst_per_method.items():
        for r in samples:
            score = r.get("metrics", {}).get("ktc_score", 0.0)
            title = f"{method} | L{r['level']} {r['sample']} | KTC:{score:.3f}"
            for ax, img, side in zip(axes[row], [r["_gt"], r["_pred"]], ["GT", "Pred"]):
                ax.imshow(img, cmap=_EIT_CMAP, vmin=0, vmax=2, interpolation="nearest")
                ax.set_title(f"{side} — {title}", fontsize=7)
                ax.axis("off")
            row += 1

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(output_dir) / "figures" / "failure_gallery.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Error overlay ──────────────────────────────────────────────────────────

def plot_error_overlay(
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: str | Path = "outputs/error_overlay.png",
) -> np.ndarray:
    """Save error overlay: grey=correct, red=missed, orange=false inclusion."""
    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[pred == gt]                    = [128, 128, 128]   # correct
    overlay[(gt > 0) & (pred == 0)]        = [216,  90,  48]   # missed
    overlay[(gt == 0) & (pred > 0)]        = [245, 166,  35]   # false

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.set_title("Error Overlay\n(Grey=Correct, Red=Missed, Orange=False)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return overlay


# ── Degradation curve ──────────────────────────────────────────────────────

def plot_degradation_curve(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Plot KTC score vs difficulty level for each method."""
    if not results:
        return None

    methods = sorted({r["method"] for r in results})

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        levels = sorted({r["level"] for r in method_results})
        avg_scores = [
            float(np.mean([
                r.get("metrics", {}).get("ktc_score", 0.0)
                for r in method_results if r["level"] == lv
            ]))
            for lv in levels
        ]
        slope = (
            float(np.polyfit(levels, avg_scores, 1)[0])
            if len(levels) >= 2 else 0.0
        )
        label = f"{method} (slope: {slope:+.3f})"
        ax.plot(levels, avg_scores, marker="o", linewidth=2.5, markersize=8,
                label=label, color=_method_color(method))

    ax.set_xlabel("Difficulty Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("KTC Score",        fontsize=12, fontweight="bold")
    ax.set_title("KTC Score vs Difficulty Level", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True, shadow=True)
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.5, 7.5)

    path = Path(output_dir) / "figures" / "degradation_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Leaderboard chart ──────────────────────────────────────────────────────

def plot_leaderboard(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Horizontal bar chart ranking methods by mean KTC score."""
    if not results:
        return None

    buckets: dict[str, list] = {}
    for r in results:
        buckets.setdefault(r["method"], []).append(
            r.get("metrics", {}).get("ktc_score", 0.0)
        )

    avg = {m: float(np.mean(v)) for m, v in buckets.items()}
    sorted_methods = sorted(avg.items(), key=lambda x: x[1], reverse=True)

    names  = [m for m, _ in sorted_methods]
    scores = [s for _, s in sorted_methods]
    grades = [
        "A" if s >= 0.60 else "B" if s >= 0.30 else "C" if s >= 0.10 else "D"
        for s in scores
    ]
    colors = [GRADE_COLORS[g] for g in grades]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))
    y_pos = np.arange(len(names))
    bars  = ax.barh(y_pos, scores, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("KTC Score", fontsize=12, fontweight="bold")
    ax.set_title("Method Leaderboard", fontsize=14, fontweight="bold")
    x_min = min(min(scores) - 0.05, -0.05)
    x_max = max(max(scores) + 0.18, 1.0)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, score, grade in zip(bars, scores, grades):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f} ({grade})", ha="left", va="center",
                fontsize=10, fontweight="bold")

    legend_elements = [
        Patch(facecolor=GRADE_COLORS["A"], edgecolor="black", label="Grade A (>=0.60)"),
        Patch(facecolor=GRADE_COLORS["B"], edgecolor="black", label="Grade B (>=0.30)"),
        Patch(facecolor=GRADE_COLORS["C"], edgecolor="black", label="Grade C (>=0.10)"),
        Patch(facecolor=GRADE_COLORS["D"], edgecolor="black", label="Grade D (<0.10)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    path = Path(output_dir) / "figures" / "leaderboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: str | Path = "outputs/confusion_matrix.png",
) -> None:
    """3x3 confusion matrix heatmap."""
    n = 3
    confusion = np.zeros((n, n), dtype=int)
    for tc in range(n):
        for pc in range(n):
            confusion[tc, pc] = int(np.sum((gt == tc) & (pred == pc)))

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_pct = confusion.astype(float) / row_sums * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    class_names = ["Water", "Resistive", "Conductive"]
    sns.heatmap(confusion_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Percentage (%)"},
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor="black", ax=ax)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class",      fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix (% of True Class)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Metrics summary heatmap ────────────────────────────────────────────────

_METRIC_COLS = ["ktc_score", "dice_resistive", "dice_conductive", "iou_resistive", "iou_conductive"]
_METRIC_LABELS = ["KTC Score", "Dice Res", "Dice Con", "IoU Res", "IoU Con"]


def plot_metrics_heatmap(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Seaborn heatmap: rows=methods, columns=5 metrics, annotated with 2 dp values."""
    if not results:
        return None

    methods = sorted({r["method"] for r in results})
    matrix = np.array([
        [
            float(np.mean([
                r.get("metrics", {}).get(metric, 0.0)
                for r in results if r["method"] == method
            ]))
            for metric in _METRIC_COLS
        ]
        for method in methods
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(_METRIC_COLS) * 1.8), max(4, len(methods) * 0.9)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        xticklabels=_METRIC_LABELS,
        yticklabels=methods,
        linewidths=0.5,
        linecolor="grey",
        ax=ax,
    )
    ax.set_title("Metrics Summary Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric",  fontsize=11)
    ax.set_ylabel("Method",  fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    path = Path(output_dir) / "figures" / "metrics_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Runtime comparison ─────────────────────────────────────────────────────

def plot_runtime_comparison(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Horizontal bar chart of mean runtime per method, sorted descending."""
    if not results:
        return None

    buckets: dict[str, list] = {}
    for r in results:
        buckets.setdefault(r["method"], []).append(r.get("runtime_ms", 0.0))

    avg_rt = {m: float(np.mean(v)) for m, v in buckets.items()}
    sorted_items = sorted(avg_rt.items(), key=lambda x: x[1], reverse=True)

    names    = [m for m, _ in sorted_items]
    runtimes = [rt for _, rt in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))
    y_pos = np.arange(len(names))
    bars  = ax.barh(y_pos, runtimes, color="#4A90E2", edgecolor="black", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Mean runtime (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Runtime Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(runtimes) * 1.3 if runtimes else 1)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, rt in zip(bars, runtimes):
        label = f"{rt / 1000:.1f} s" if rt >= 1000 else f"{rt:.0f} ms"
        ax.text(
            bar.get_width() + max(runtimes) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            label, ha="left", va="center", fontsize=10, fontweight="bold",
        )

    path = Path(output_dir) / "figures" / "runtime_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Electrode layout ───────────────────────────────────────────────────────

def plot_electrodes(save_path: str | Path = "outputs/figures/electrodes.png") -> None:
    """Draw circular tank with 32 electrodes."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Circle((0, 0), 1, fill=False, linewidth=2))
    for i in range(32):
        theta = 2 * np.pi * i / 32
        ax.plot(np.cos(theta), np.sin(theta), "o", markersize=6)
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    plt.title("32 Electrode Layout")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Multi-method comparison panel ──────────────────────────────────────────

def plot_comparison_panel(
    gt: np.ndarray,
    methods_dict: dict[str, np.ndarray],
    save_path: str | Path = "outputs/comparison_panel.png",
) -> None:
    """Save GT + one column per method side-by-side.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth (256, 256).
    methods_dict : dict[str, np.ndarray]
        ``{"Method Name": prediction_array, ...}`` in display order.
    save_path : str | Path
        Output file path.
    """
    n_panels = 1 + len(methods_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(gt, cmap=_EIT_CMAP, vmin=0, vmax=2)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for idx, (name, pred) in enumerate(methods_dict.items(), start=1):
        axes[idx].imshow(pred, cmap=_EIT_CMAP, vmin=0, vmax=2)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Structured single-method panel (runner-style output) ───────────────────

def save_method_panel(
    pred: np.ndarray,
    gt: np.ndarray,
    level: int,
    sample: str,
    method: str,
    output_dir: str | Path = "outputs",
) -> str:
    """Save a GT | Pred | Error panel at ``output_dir/level_{L}/sample_{S}/{method}.png``.

    Returns the saved file path as a string.
    """
    save_dir = Path(output_dir) / f"level_{level}" / f"sample_{sample}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{method}.png"
    plot_panel(pred=pred, gt=gt, method=method, level=level, sample=str(sample),
               save_path=save_path)
    return str(save_path)
