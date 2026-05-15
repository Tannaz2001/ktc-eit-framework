import sys
sys.path.insert(0, '.')
from src.ktc_framework.metrics.ktc_score import compute_ktc_score
import numpy as np

rng = np.random.default_rng(hash("mock_level1_A") % (2**32))
rng.standard_normal(2356)
gt_flat = rng.choice([0,1,2], size=256*256, p=[0.90,0.05,0.05])
gt = gt_flat.reshape(256, 256).astype("uint8")
pred = np.zeros((256,256), dtype="int32")

score = compute_ktc_score(pred, gt)
print("KTC score (all-zero pred):", round(score, 6), "-> displays as", round(score, 3))
score_perfect = compute_ktc_score(gt.astype("int32"), gt)
print("KTC score (perfect pred): ", round(score_perfect, 6), "-> displays as", round(score_perfect, 3))
print("OK - scoring pipeline works")
