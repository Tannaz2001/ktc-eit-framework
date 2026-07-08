import argparse
import hashlib
from pathlib import Path

import numpy as np
import scipy.io as sio


def _circle(mask, cx, cy, radius, label):
    yy, xx = np.ogrid[:256, :256]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = label


def reconstruct(data_path, level):
    mat = sio.loadmat(str(data_path), squeeze_me=True, struct_as_record=False)
    values = []
    for key, value in mat.items():
        if key.startswith("_"):
            continue
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            values.append(arr.astype(float).ravel())

    signal = np.concatenate(values) if values else np.array([level], dtype=float)
    digest = hashlib.md5(signal[: min(signal.size, 2048)].tobytes() + bytes([level])).digest()
    nums = list(digest)

    recon = np.zeros((256, 256), dtype=np.uint8)
    _circle(recon, 45 + nums[0] % 115, 50 + nums[1] % 115, 18 + nums[2] % 20, 1)
    _circle(recon, 105 + nums[3] % 95, 95 + nums[4] % 95, 16 + nums[5] % 18, 2)
    return recon


def main():
    parser = argparse.ArgumentParser(
        description="Toy ML inverse reconstruction method for dashboard zip-upload testing."
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("level", type=int)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(p for p in input_dir.glob("*.mat") if p.name.lower() != "ref.mat")
    if not data_files:
        raise FileNotFoundError(f"No data .mat file found in {input_dir}")

    for data_file in data_files:
        reconstruction = reconstruct(data_file, args.level)
        sio.savemat(str(output_dir / data_file.name), {"reconstruction": reconstruction})


if __name__ == "__main__":
    main()
