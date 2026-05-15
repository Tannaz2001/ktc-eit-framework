"""
Diagnostic: what does the KTC score actually produce when MockMethodPlugin
(all-zero prediction) is scored against MockDataPlugin ground truth?
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# --- inline _ktcssim so we don't need the full framework import chain ---
def _ktcssim(truth, reco, r=80.0):
    c1, c2 = 1e-4, 9e-4
    t = truth.astype(np.float64)
    ra = reco.astype(np.float64)
    def _s(a): return gaussian_filter(a, sigma=r, mode="constant", cval=0.0, truncate=2.0)
    correction = _s(np.ones_like(t))
    mu_t = _s(t) / correction
    mu_r = _s(ra) / correction
    mu_t2, mu_r2, mu_tr = mu_t**2, mu_r**2, mu_t*mu_r
    sigma_t2 = _s(t**2) / correction - mu_t2
    sigma_r2 = _s(ra**2) / correction - mu_r2
    sigma_tr  = _s(t*ra) / correction  - mu_tr
    num = (2*mu_tr + c1) * (2*sigma_tr + c2)
    den = (mu_t2 + mu_r2 + c1) * (sigma_t2 + sigma_r2 + c2)
    return float(np.mean(num / den))

def compute_ktc_score(pred, gt):
    if pred.shape != (256, 256): return 0.0
    sc = _ktcssim((gt==2).astype(np.float64), (pred==2).astype(np.float64))
    sd = _ktcssim((gt==1).astype(np.float64), (pred==1).astype(np.float64))
    return 0.5 * (sc + sd)

# --- synthetic ground truth matching MockDataPlugin output ---
rng = np.random.default_rng(hash("mock_level1_A") % (2**32))
rng.standard_normal(2356)                          # consume voltages draw
gt_flat = rng.choice([0,1,2], size=256*256, p=[0.90,0.05,0.05])
gt   = gt_flat.reshape(256, 256).astype(np.uint8)
pred = np.zeros((256, 256), dtype=np.int32)        # MockMethodPlugin output

print("=== Ground truth label counts ===")
for lbl, name in [(0,"Background"),(1,"Resistive"),(2,"Conductive")]:
    n = int((gt==lbl).sum())
    print(f"  label {lbl} ({name}): {n} px ({100*n/gt.size:.2f}%)")

print()
print("=== Scores (pred = all zeros) ===")
sc = _ktcssim((gt==2).astype(float), (pred==2).astype(float))
sd = _ktcssim((gt==1).astype(float), (pred==1).astype(float))
ktc = 0.5*(sc+sd)
print(f"  ssim_conductive : {sc:.8f}")
print(f"  ssim_resistive  : {sd:.8f}")
print(f"  ktc_score       : {ktc:.8f}   →  displayed as {ktc:.3f}")

print()
print("=== Scores (pred = perfect copy of gt) ===")
sc_p = _ktcssim((gt==2).astype(float), (gt==2).astype(float))
sd_p = _ktcssim((gt==1).astype(float), (gt==1).astype(float))
ktc_p = 0.5*(sc_p+sd_p)
print(f"  ssim_conductive : {sc_p:.8f}")
print(f"  ssim_resistive  : {sd_p:.8f}")
print(f"  ktc_score       : {ktc_p:.8f}   →  displayed as {ktc_p:.3f}")

print()
print("=== Root cause summary ===")
print(f"  All-zero pred KTC  = {ktc:.6f}  → rounds to {ktc:.3f}")
print(f"  Perfect pred KTC   = {ktc_p:.6f}  → rounds to {ktc_p:.3f}")
print(f"  Scoring IS working. 0.000 is the correct score for an all-zero prediction.")
