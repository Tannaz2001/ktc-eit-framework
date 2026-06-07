#!/usr/bin/env python3
"""
Extract and hardcode electrode node indices from KTC mesh.

Run this script with the KTC2023_open_mesh.mat file to determine the correct
32 electrode node indices for your mesh, then update BackProjection to use
these hardcoded values.

Usage:
    python3 extract_electrode_nodes.py /path/to/KTC2023_open_mesh.mat
"""

import sys
import scipy.io as sio
import numpy as np


def extract_electrode_nodes(mesh_file):
    """Extract 32 electrode node indices from KTC mesh."""
    print(f"Loading mesh: {mesh_file}")
    mesh = sio.loadmat(mesh_file)

    print(f"Mesh keys: {list(mesh.keys())}")

    # Check for elfaces
    elfaces_key = None
    for key in ["elfaces", "ElFaces", "el_faces", "elFaces"]:
        if key in mesh:
            elfaces_key = key
            break

    if elfaces_key:
        print(f"\nUsing elfaces from key: '{elfaces_key}'")
        elfaces = mesh[elfaces_key]
        print(f"  Shape: {elfaces.shape}")
        print(f"  Type: {type(elfaces)}")

    # Extract nodes
    if "g" not in mesh:
        print("ERROR: Mesh does not have 'g' (nodes) key")
        return

    nodes = mesh["g"]
    print(f"\nNodes shape: {nodes.shape}")

    distances = np.linalg.norm(nodes, axis=1)
    print(f"Distance range: {distances.min():.6f} to {distances.max():.6f}")

    # Find boundary nodes
    for r_min, r_max in [(0.95, 1.05), (0.90, 1.10), (0.80, 1.20)]:
        boundary_mask = (distances >= r_min) & (distances <= r_max)
        boundary_indices = np.where(boundary_mask)[0]
        print(f"  Boundary nodes [{r_min:.2f}, {r_max:.2f}]: {len(boundary_indices)}")

        if len(boundary_indices) >= 32:
            print(f"  -> Using this range")
            break

    if len(boundary_indices) < 32:
        print(f"\nERROR: Could not find 32 boundary nodes (found {len(boundary_indices)})")
        return

    # Sort by angle
    angles = np.arctan2(nodes[boundary_indices, 1], nodes[boundary_indices, 0])
    angle_indices = np.argsort(angles)
    sorted_boundary = boundary_indices[angle_indices]
    electrode_nodes = sorted_boundary[:32]

    # Verify spacing
    angles_sorted = angles[angle_indices[:32]]
    angle_diffs = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2 * np.pi]]))

    print(f"\nElectrode angle spacing:")
    print(f"  Expected (360/32): {360.0/32:.2f} degrees")
    print(f"  Actual mean:       {np.degrees(angle_diffs.mean()):.2f} degrees")
    print(f"  Actual std:        {np.degrees(angle_diffs.std()):.2f} degrees")
    print(f"  Min spacing:       {np.degrees(angle_diffs.min()):.2f} degrees")
    print(f"  Max spacing:       {np.degrees(angle_diffs.max()):.2f} degrees")

    if angle_diffs.std() < np.radians(5):
        print("\nOK: Electrodes are reasonably evenly distributed")
    else:
        print("\nWARNING: Electrodes may not be evenly distributed")

    # Output hardcoded values
    print("\n" + "=" * 70)
    print("HARDCODED ELECTRODE NODES FOR BackProjection")
    print("=" * 70)
    print("\nAdd this to src/ktc_framework/methods/backprojection.py:")
    print("In _electrode_positions() method, add before the for loop:\n")

    print("# Hardcoded KTC mesh electrode node indices (from extract_electrode_nodes.py)")
    print("_KTC_ELECTRODE_NODES = np.array([")
    for i in range(0, 32, 8):
        vals = electrode_nodes[i : i + 8]
        print("    " + ", ".join(f"{v:5d}" for v in vals) + ",")
    print("], dtype=np.int32)\n")

    print("if mesh_data is not None and len(_KTC_ELECTRODE_NODES) == 32:")
    print("    # Use hardcoded electrode positions for KTC mesh")
    print("    return _KTC_ELECTRODE_NODES\n")

    print("=" * 70)
    print("\nElectrode node indices:")
    print(list(electrode_nodes))

    return electrode_nodes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python3 extract_electrode_nodes.py ~/data/KTC2023_open_mesh.mat")
        sys.exit(1)

    mesh_file = sys.argv[1]
    extract_electrode_nodes(mesh_file)
