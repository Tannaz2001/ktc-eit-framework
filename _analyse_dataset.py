import scipy.io
import numpy as np

base = r"C:\Users\sahil\OneDrive\Desktop\KTC_WORK_HIS\Codes_Matlab"

# --- reference measurement ---
ref = scipy.io.loadmat(f"{base}/TrainingData/ref.mat", squeeze_me=True, struct_as_record=False)
print("=== ref.mat ===")
for k, v in ref.items():
    if k.startswith("_"):
        continue
    a = np.asarray(v)
    print(f"  [{k}]  shape={a.shape}  dtype={a.dtype}  min={a.min():.6f}  max={a.max():.6f}")

print()

# --- per-sample inspection ---
for i in range(1, 5):
    d   = scipy.io.loadmat(f"{base}/TrainingData/data{i}.mat", squeeze_me=True, struct_as_record=False)
    t   = scipy.io.loadmat(f"{base}/GroundTruths/true{i}.mat", squeeze_me=True, struct_as_record=False)
    out = scipy.io.loadmat(f"{base}/Output/{i}.mat",           squeeze_me=True, struct_as_record=False)

    print(f"=== Sample {i} ===")

    for k, v in d.items():
        if k.startswith("_"):
            continue
        a = np.asarray(v)
        print(f"  data[{k}]  shape={a.shape}  dtype={a.dtype}  "
              f"min={a.min():.6f}  max={a.max():.6f}  "
              f"mean={a.mean():.6f}  std={a.std():.6f}")

    for k, v in t.items():
        if k.startswith("_"):
            continue
        a = np.asarray(v)
        u = np.unique(a)
        counts = {int(lbl): int(np.sum(a == lbl)) for lbl in u}
        total = a.size
        pct = {lbl: round(100 * cnt / total, 2) for lbl, cnt in counts.items()}
        print(f"  true[{k}]  shape={a.shape}  dtype={a.dtype}  "
              f"unique={u.tolist()}  counts={counts}  pct={pct}")

    for k, v in out.items():
        if k.startswith("_"):
            continue
        a = np.asarray(v)
        print(f"  out[{k}]   shape={a.shape}  dtype={a.dtype}  "
              f"min={a.min():.6f}  max={a.max():.6f}  "
              f"mean={a.mean():.6f}  std={a.std():.6f}")

    print()

# --- mesh files ---
print("=== Mesh files ===")
for name in ("Mesh_dense.mat", "Mesh_sparse.mat"):
    m = scipy.io.loadmat(f"{base}/{name}", squeeze_me=True, struct_as_record=False)
    print(f"  {name}:")
    for k, v in m.items():
        if k.startswith("_"):
            continue
        a = np.asarray(v) if not hasattr(v, "shape") else v
        try:
            a = np.asarray(a, dtype=float)
            print(f"    [{k}]  shape={a.shape}  dtype={a.dtype}  "
                  f"min={a.min():.6f}  max={a.max():.6f}")
        except Exception:
            print(f"    [{k}]  (non-numeric or struct)")
