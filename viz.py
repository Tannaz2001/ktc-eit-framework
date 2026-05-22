import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import seaborn as sns
import json
from pathlib import Path

# =========================================================
# COLOUR MAP
# =========================================================

COLORMAP = {
    0: "#1a3a5c",   # water
    1: "#D85A30",   # resistive
    2: "#1D9E75"    # conductive
}

cmap = ListedColormap([
    COLORMAP[0],
    COLORMAP[1],
    COLORMAP[2]
])

# Grade colors for leaderboard
GRADE_COLORS = {
    'A': '#1D9E75',  # green
    'B': '#4A90E2',  # blue
    'C': '#F5A623',  # amber
    'D': '#D85A30'   # red
}

# =========================================================
# PANEL PLOT (Original)
# =========================================================

def plot_panel(pred, gt, save_path="outputs/panel.png"):
    """
    Creates:
    Ground Truth | Prediction | Error Map
    """

    error = pred != gt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # -----------------------------------------------------
    # Ground Truth
    # -----------------------------------------------------

    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # -----------------------------------------------------
    # Error Map
    # -----------------------------------------------------

    axes[2].imshow(error, cmap="Reds")
    axes[2].set_title("Error Map")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved panel to {save_path}")

# =========================================================
# MULTI-METHOD COMPARISON PANEL
# =========================================================

def plot_comparison_panel(gt, methods_dict, save_path="outputs/comparison_panel.png"):
    """
    Creates comparison panel with ground truth and multiple reconstruction methods.
    
    Parameters:
    -----------
    gt : np.ndarray
        Ground truth image (256, 256)
    methods_dict : dict
        Dictionary mapping method names to reconstruction arrays
        e.g., {'Back-projection': bp_result, 'Gauss-Newton': gn_result}
    save_path : str
        Path to save the output image
    
    Example:
    --------
    plot_comparison_panel(gt, {
        'Back-projection': bp_pred,
        'Gauss-Newton': gn_pred
    })
    """
    
    n_methods = len(methods_dict)
    n_panels = 1 + n_methods  # GT + all methods
    
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    
    if n_panels == 1:
        axes = [axes]
    
    # Ground Truth
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Each method
    for idx, (method_name, pred) in enumerate(methods_dict.items(), start=1):
        axes[idx].imshow(pred, cmap=cmap, vmin=0, vmax=2)
        axes[idx].set_title(method_name, fontsize=12, fontweight='bold')
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison panel to {save_path}")

# =========================================================
# ERROR OVERLAY
# =========================================================

def plot_error_overlay(pred, gt, save_path="outputs/error_overlay.png"):
    """
    Creates a 256×256 error overlay image:
    - Correct pixels: grey
    - Missed inclusions (false negative): red
    - False inclusions (false positive): orange
    
    Parameters:
    -----------
    pred : np.ndarray
        Predicted segmentation (256, 256)
    gt : np.ndarray
        Ground truth segmentation (256, 256)
    save_path : str
        Path to save the output image
    """
    
    # Create RGB image
    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Correct predictions: grey (128, 128, 128)
    correct_mask = pred == gt
    overlay[correct_mask] = [128, 128, 128]
    
    # Missed inclusions (false negatives): red (216, 90, 48)
    # True class is inclusion (1 or 2), but predicted as water (0)
    missed_inclusion_mask = (gt > 0) & (pred == 0)
    overlay[missed_inclusion_mask] = [216, 90, 48]  # Red from team palette
    
    # False inclusions (false positives): orange (245, 166, 35)
    # True class is water (0), but predicted as inclusion (1 or 2)
    false_inclusion_mask = (gt == 0) & (pred > 0)
    overlay[false_inclusion_mask] = [245, 166, 35]  # Orange/amber
    
    # Save the overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.set_title("Error Overlay\n(Grey=Correct, Red=Missed, Orange=False)", 
                 fontsize=11, fontweight='bold')
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error overlay to {save_path}")
    
    return overlay

# =========================================================
# DEGRADATION CURVE
# =========================================================

def plot_degradation_curve(scores_json, save_path="outputs/degradation_curve.png"):
    """
    Plots each method's KTC score against difficulty level 1-7.
    Multiple methods shown in different colors on the same chart.
    
    Parameters:
    -----------
    scores_json : str or dict
        Path to scores JSON file or dictionary containing scores
        Expected format:
        {
            "method_name": {
                "level_1": 0.85,
                "level_2": 0.78,
                ...
            }
        }
    save_path : str
        Path to save the output image
    """
    
    # Load scores
    if isinstance(scores_json, str):
        with open(scores_json, "r") as f:
            data = json.load(f)
    else:
        data = scores_json
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette for different methods
    colors = ['#1D9E75', '#D85A30', '#4A90E2', '#F5A623', '#9B59B6', '#E74C3C', '#1ABC9C']
    
    for idx, (method_name, scores) in enumerate(data.items()):
        levels = []
        ktc_scores = []
        
        # Extract level and score data
        for key, value in scores.items():
            if key.startswith('level_'):
                level_num = int(key.split('_')[1])
                levels.append(level_num)
                ktc_scores.append(value)
        
        # Sort by level
        if levels:
            sorted_data = sorted(zip(levels, ktc_scores))
            levels, ktc_scores = zip(*sorted_data)
            
            # Plot line with markers
            color = colors[idx % len(colors)]
            ax.plot(levels, ktc_scores, marker='o', linewidth=2.5, 
                   markersize=8, label=method_name, color=color)
    
    ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('KTC Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Degradation Across Difficulty Levels', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved degradation curve to {save_path}")

# =========================================================
# LEADERBOARD
# =========================================================

def plot_leaderboard(scores_json, save_path="outputs/leaderboard.png"):
    """
    Creates a horizontal bar chart ranking all methods by composite score.
    Bars colored by grade (green=A, blue=B, amber=C, red=D).
    
    Parameters:
    -----------
    scores_json : str or dict
        Path to scores JSON file or dictionary containing scores
        Expected format:
        {
            "method_name": {
                "composite_score": 0.82,
                "grade": "A"
            }
        }
        Or if grades not provided, will auto-assign based on score
    save_path : str
        Path to save the output image
    """
    
    # Load scores
    if isinstance(scores_json, str):
        with open(scores_json, "r") as f:
            data = json.load(f)
    else:
        data = scores_json
    
    # Extract methods and scores
    methods = []
    scores = []
    grades = []
    
    for method_name, method_data in data.items():
        methods.append(method_name)
        
        # Get composite score
        if isinstance(method_data, dict):
            score = method_data.get('composite_score', method_data.get('score', 0))
            grade = method_data.get('grade', None)
        else:
            score = method_data
            grade = None
        
        scores.append(score)
        
        # Auto-assign grade if not provided
        if grade is None:
            if score >= 0.85:
                grade = 'A'
            elif score >= 0.70:
                grade = 'B'
            elif score >= 0.55:
                grade = 'C'
            else:
                grade = 'D'
        
        grades.append(grade)
    
    # Sort by score
    sorted_data = sorted(zip(methods, scores, grades), key=lambda x: x[1], reverse=True)
    methods, scores, grades = zip(*sorted_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(methods) * 0.5)))
    
    # Color bars by grade
    colors = [GRADE_COLORS[g] for g in grades]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title('Method Leaderboard', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add score labels on bars
    for i, (bar, score, grade) in enumerate(zip(bars, scores, grades)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.3f} ({grade})',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add legend for grades
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GRADE_COLORS['A'], edgecolor='black', label='Grade A (≥0.85)'),
        Patch(facecolor=GRADE_COLORS['B'], edgecolor='black', label='Grade B (≥0.70)'),
        Patch(facecolor=GRADE_COLORS['C'], edgecolor='black', label='Grade C (≥0.55)'),
        Patch(facecolor=GRADE_COLORS['D'], edgecolor='black', label='Grade D (<0.55)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved leaderboard to {save_path}")

# =========================================================
# CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(pred, gt, save_path="outputs/confusion_matrix.png"):
    """
    Creates a 3×3 heatmap showing classification confusion.
    Rows = true class, Columns = predicted class.
    
    Parameters:
    -----------
    pred : np.ndarray
        Predicted segmentation (256, 256)
    gt : np.ndarray
        Ground truth segmentation (256, 256)
    save_path : str
        Path to save the output image
    """
    
    # Create confusion matrix
    n_classes = 3
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            confusion[true_class, pred_class] = np.sum((gt == true_class) & (pred == pred_class))
    
    # Normalize by row (true class) to get percentages
    confusion_pct = confusion.astype(float)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_pct = confusion_pct / row_sums * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create heatmap
    class_names = ['Water', 'Resistive', 'Conductive']
    sns.heatmap(confusion_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Percentage (%)'},
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (% of True Class)', fontsize=14, fontweight='bold')
    
    # Add counts as secondary annotation
    for i in range(n_classes):
        for j in range(n_classes):
            count = confusion[i, j]
            ax.text(j + 0.5, i + 0.7, f'n={count}', 
                   ha='center', va='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {save_path}")

# =========================================================
# ELECTRODE PLOT (Original)
# =========================================================

def plot_electrodes(save_path="outputs/electrodes.png"):
    """
    Draws circular tank with 32 electrodes.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    # Tank boundary
    circle = Circle((0, 0), 1, fill=False, linewidth=2)
    ax.add_patch(circle)

    # 32 electrodes
    n_electrodes = 32

    for i in range(n_electrodes):
        theta = 2 * np.pi * i / n_electrodes

        x = np.cos(theta)
        y = np.sin(theta)

        ax.plot(x, y, 'o', markersize=6)

    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.axis('off')

    plt.title("32 Electrode Layout")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved electrodes to {save_path}")

# =========================================================
# UTILITY: SAVE PANEL FOR RUNNER INTEGRATION
# =========================================================

def save_method_panel(pred, gt, level, sample, method, output_dir="outputs"):
    """
    Saves a method comparison panel in the standardized directory structure.
    Path: outputs/level_X/sample_Y/method_Z.png
    
    Parameters:
    -----------
    pred : np.ndarray
        Predicted reconstruction
    gt : np.ndarray
        Ground truth
    level : int
        Difficulty level (1-7)
    sample : int or str
        Sample identifier
    method : str
        Method name (e.g., 'back_projection', 'gauss_newton')
    output_dir : str
        Base output directory
    """
    
    # Create directory structure
    save_dir = Path(output_dir) / f"level_{level}" / f"sample_{sample}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save path
    save_path = save_dir / f"{method}.png"
    
    # Create panel
    plot_panel(pred, gt, save_path=str(save_path))
    
    return str(save_path)
