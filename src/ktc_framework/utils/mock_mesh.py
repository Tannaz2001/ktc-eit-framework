# src/ktc_framework/utils/mock_mesh.py
import numpy as np
from pyeit.mesh import create as mesh_create
from collections import namedtuple

DataBatch = namedtuple(
    "DataBatch",
    ["voltages", "injection_patterns", "ground_truth", "mesh"]
)

def make_adjacent_protocol(n_electrodes=32):
    """
    Build an adjacent-pair injection protocol manually.
    Returns a matrix of shape (n_electrodes, n_patterns)
    """
    n_patterns = n_electrodes * (n_electrodes - 1) // 2
    patterns = np.zeros((n_electrodes, n_patterns))
    for i in range(n_electrodes):
        source = i % n_patterns
        sink = (i + 1) % n_patterns
        patterns[i, source] = 1
        patterns[i, sink] = -1
    return patterns

def make_dummy_batch_with_mesh():
    """
    Create a dummy DataBatch for testing GaussNewton and BackProjection.
    Uses a valid PyEIT mesh and manual adjacent-pair protocol.
    """
    voltages = np.random.randn(2356).astype(np.float64)
    mesh_obj = mesh_create(n_el=32, h0=0.1)  # valid PyEIT mesh
    injection_patterns = make_adjacent_protocol(n_electrodes=32)
    ground_truth = np.zeros((256, 256), dtype=int)

    return DataBatch(
        voltages=voltages,
        injection_patterns=injection_patterns,
        ground_truth=ground_truth,
        mesh=mesh_obj
    )