import os
from pathlib import Path
import scipy.io
import numpy as np
from src.ktc_framework.types import DataBatch
from src.ktc_framework.adapters.method_registry import get as registry_get
from src.ktc_framework.metrics.ktc_score import compute_all_metrics
from src.ktc_framework.metrics.composite_score import composite_score, letter_grade

# --- Paths ---
# To the actual Windows path where you extracted the dataset:
dataset_root = Path(r"D:/MS_HIS/SoSe2026/HIS Project/Project/ktc-eit-framework/Evaluation_Data_Full/10418802/EvaluationData")
eval_root = dataset_root  # directly point to the folder containing level1, level2, ...
gt_root = dataset_root / "GroundTruths"
output_dir = Path(r"D:/MS_HIS/SoSe2026/HIS Project/Project/ktc-eit-framework/outputs_optimized")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Levels ---
levels = sorted([d for d in eval_root.iterdir() if d.is_dir()])

# --- Helper: Load sample ---
def load_sample(level_path, sample_idx):
    data_file = level_path / f"data{sample_idx}.mat"
    ref_file = level_path / "ref.mat"

    data = scipy.io.loadmat(data_file)
    ref = scipy.io.loadmat(ref_file)

    voltages = data.get("voltages", data.get("V"))
    inj = data.get("injection_patterns", data.get("Inj"))

    gt_file = gt_root / f"{level_path.name.replace('level','level_')}/{sample_idx}_true.mat"
    gt_mat = scipy.io.loadmat(gt_file)
    gt = gt_mat.get("ground_truth", gt_mat.get("GT"))

    ref_voltages = ref.get("Uelref", None)
    return DataBatch(
        voltages=np.array(voltages, dtype=np.float64).ravel(),
        injection_patterns=np.array(inj, dtype=np.float64),
        ground_truth=np.array(gt, dtype=np.uint8),
        level=int(level_path.name[-1]),
        sample_id=f"{level_path.name}_{sample_idx}",
        mesh=None,
        reference_voltages=np.array(ref_voltages, dtype=np.float64).ravel() if ref_voltages is not None else None
    )

# --- Run optimized reconstruction ---
results = []

for level_path in levels:
    for sample_idx in range(1, 4):  # 3 samples per level
        batch = load_sample(level_path, sample_idx)

        # Run BackProjection
        bp_plugin = registry_get("BackProjection")()
        bp_seg = bp_plugin.reconstruct(batch)

        # Run Gauss-Newton
        gn_plugin = registry_get("GaussNewton")()
        gn_seg = gn_plugin.reconstruct(batch)

        # Fusion for higher levels
        final_seg = np.maximum(bp_seg, gn_seg) if batch.level >= 6 else bp_seg

        # Compute metrics
        metrics = run_all_metrics(final_seg, batch.ground_truth)
        comp_score = composite_score(metrics)
        grade = letter_grade(comp_score)

        # Save final segmentation
        out_file = output_dir / f"{batch.sample_id}_seg.npy"
        np.save(out_file, final_seg)

        # Append result
        results.append({
            "sample": batch.sample_id,
            "level": batch.level,
            "bp_seg": bp_seg,
            "gn_seg": gn_seg,
            "final_seg": final_seg,
            "metrics": metrics,
            "composite_score": comp_score,
            "grade": grade,
            "ground_truth": batch.ground_truth,
            "seg_file": str(out_file)
        })

# --- Print summary ---
for r in results:
    print(f"{r['sample']} | Level {r['level']} | KTC: {r['metrics'].get('ktc_score',0):.3f} "
          f"| Dice Res: {r['metrics'].get('dice_resistive',0):.3f} "
          f"| Dice Cond: {r['metrics'].get('dice_conductive',0):.3f} "
          f"| Grade: {r['grade']} | Saved: {r['seg_file']}")

print(f"\nAll optimized segmentations saved in {output_dir}")