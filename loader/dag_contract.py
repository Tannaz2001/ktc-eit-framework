import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class EITSample:
    """The standard token that flows through the DAG pipeline."""
    sample_id: str
    level: str
    
    # Raw Data (from DataLoader)
    v_meas: np.ndarray          # Measured voltages
    v_ref: np.ndarray           # Reference voltages (empty tank)
    
    # Engineered Features (Populated by FeatureTransformer)
    delta_v: Optional[np.ndarray] = None
    gp_mask: Optional[np.ndarray] = None # Tracks which electrodes were interpolated
    
    # Ground Truth (Kept hidden from the solver!)
    ground_truth_image: Optional[np.ndarray] = None