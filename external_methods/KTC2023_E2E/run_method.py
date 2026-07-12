"""
CLI wrapper for KTC2023_E2E (end-to-end UNet method).

main.py defines main(inputFolder, outputFolder, difficulty) but has no
__main__ / argparse block. This wrapper adds the CLI interface.

Two fixes applied transparently:
  1. matplotlib Agg backend — prevents plt.show() from blocking.
  2. torch.load map_location='cpu' fallback — allows CPU-only inference
     on models that were saved on GPU.

Usage:
    python run_method.py <input_folder> <output_folder> <difficulty>
"""
import os
import sys
import argparse

# 1. Agg before any plt import
import matplotlib
matplotlib.use("Agg")

# 2. Ensure model loads work on CPU-only machines
import torch
_orig_load = torch.load
def _load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _orig_load(*args, **kwargs)
torch.load = _load_cpu

# 3. deepinv.utils.get_freer_gpu returns None when no GPU found;
#    main.py passes this directly to .to(device), which crashes on None.
import deepinv as dinv
_orig_gpu = dinv.utils.get_freer_gpu
def _safe_gpu():
    dev = _orig_gpu()
    return dev if dev is not None else torch.device("cpu")
dinv.utils.get_freer_gpu = _safe_gpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str)
parser.add_argument("output_folder", type=str)
parser.add_argument("difficulty", type=int)
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

from main import main
main(args.input_folder, args.output_folder, args.difficulty)
