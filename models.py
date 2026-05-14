from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

@dataclass(slots=True)
class EITSample:
    """The raw contract representing a loaded dataset sample."""
    sample_id: str
    level: int
    v_meas: np.ndarray
    v_ref: np.ndarray | float = 0.0
    source_path: Path | str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ProcessedSample:
    """The clean contract for downstream algorithms."""
    sample_id: str
    level: int
    delta_v: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
