"""
CLI wrapper for KTC2023-ABC2 (DIP + TV method).

Accepts the same arguments as main.py but saves .mat files (with a
'reconstruction' field) instead of PNG files, conforming to the KTC
framework output contract.

Usage:
    python run_method.py <input_path> <output_path> <category>
"""
import os
import re
import sys
import argparse

import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import solver

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str)
parser.add_argument("output_path", type=str)
parser.add_argument("category", type=int, choices=range(1, 8), metavar="[1-7]")
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

img_names = sorted(f for f in os.listdir(args.input_path) if f.endswith(".mat") and f != "ref.mat")
print(f"{len(img_names)} image(s) found.")

for idx, img_name in enumerate(img_names, start=1):
    path_in = os.path.join(args.input_path, img_name)
    print(f"Processing {img_name} ...")
    img_seg = solver.solve(path_in, args.category)

    # solver.solve() returns a 256×256 float array with values ≈ {0, 1, 2}
    img_seg = np.round(img_seg).clip(0, 2).astype(np.int32)

    out_path = os.path.join(args.output_path, f"{idx}.mat")
    scipy.io.savemat(out_path, {"reconstruction": img_seg})
    print(f"  → saved {out_path}")
