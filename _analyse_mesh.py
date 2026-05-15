import scipy.io
import numpy as np

base = r"C:\Users\sahil\OneDrive\Desktop\KTC_WORK_HIS\Codes_Matlab"

def inspect_struct(obj, prefix="", depth=0):
    if depth > 3:
        return
    if hasattr(obj, "_fieldnames"):
        for field in obj._fieldnames:
            val = getattr(obj, field, None)
            if val is None:
                continue
            try:
                a = np.asarray(val, dtype=float)
                print(f"{prefix}{field}: shape={a.shape} dtype={a.dtype} "
                      f"min={a.min():.4f} max={a.max():.4f}")
            except Exception:
                print(f"{prefix}{field}: (struct or non-numeric)")
                inspect_struct(val, prefix + "  ", depth + 1)
    else:
        try:
            a = np.asarray(obj, dtype=float)
            print(f"{prefix}array shape={a.shape} dtype={a.dtype}")
        except Exception:
            print(f"{prefix}(unreadable type: {type(obj).__name__})")

for name in ("Mesh_dense.mat", "Mesh_sparse.mat"):
    print(f"\n=== {name} ===")
    m = scipy.io.loadmat(f"{base}/{name}", squeeze_me=True, struct_as_record=False)
    for k, v in m.items():
        if k.startswith("_"):
            continue
        print(f"  Key: {k}")
        inspect_struct(v, prefix="    ")
