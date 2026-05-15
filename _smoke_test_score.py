import sys
sys.path.insert(0, r"C:\Users\sahil\OneDrive\Desktop\KTC_WORK_HIS\ktc-eit-framework")

import scipy.io
import numpy as np
from src.ktc_framework.metrics.ktc_score import compute_ktc_score, compute_all_metrics

base = r"C:\Users\sahil\OneDrive\Desktop\KTC_WORK_HIS\Codes_Matlab"

for i in range(1, 5):
    t   = scipy.io.loadmat(f"{base}/GroundTruths/true{i}.mat", squeeze_me=True)["truth"]
    out = scipy.io.loadmat(f"{base}/Output/{i}.mat",           squeeze_me=True)["reconstruction"]
    m   = compute_all_metrics(out.astype(int), t.astype(int))
    print(
        f"Sample {i}: "
        f"ktc={m['ktc_score']:.4f}  "
        f"dice_r={m['dice_resistive']:.3f}  "
        f"dice_c={m['dice_conductive']:.3f}  "
        f"iou_r={m['iou_resistive']:.3f}  "
        f"iou_c={m['iou_conductive']:.3f}"
    )
