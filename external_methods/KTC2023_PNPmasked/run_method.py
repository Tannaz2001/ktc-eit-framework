"""
CLI wrapper for KTC2023_PNPmasked (Plug-and-Play + synthetic masks method).

main.py defines main(inputFolder, outputFolder, categoryNbr) but has no
__main__ / argparse block. This wrapper adds the CLI interface.

Two fixes applied transparently:
  1. matplotlib Agg backend — prevents plt.show() calls in main.py from
     blocking the subprocess.
  2. Working directory is set to this file's directory so that relative
     paths inside main.py resolve correctly:
       - 'Mesh_sparse.mat'
       - './weights_denoiser.pth'
       - 'Mask_Synthetic/lev{N}.mat'

Usage:
    python run_method.py <input_folder> <output_folder> <category>
"""
import os
import sys
import argparse

# 1. Agg before any plt import
import matplotlib
matplotlib.use("Agg")

# 2. Change CWD to this file's directory so all relative paths resolve
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str)
parser.add_argument("output_folder", type=str)
parser.add_argument("category", type=int)
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

from main import main
main(args.input_folder, args.output_folder, args.category)
