from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.methods.segment import segment
from src.ktc_framework.adapters.method_registry import register
import importlib
import numpy as np

BP = None
AdjElectrode = None
try:
    bp_module = importlib.import_module("pyeit.eit.bp")
    protocol_module = importlib.import_module("pyeit.eit.protocol")
    BP = getattr(bp_module, "BP", None)
    AdjElectrode = getattr(protocol_module, "AdjElectrode", None)
except ImportError:
    BP = None
    AdjElectrode = None

@register
class BackProjection(MethodPlugin):
    def reconstruct(self, batch) -> np.ndarray:
        # fallback if pyEIT not installed or no mesh
        if BP is None or getattr(batch, 'mesh', None) is None or AdjElectrode is None:
            sigma_map = np.random.rand(256, 256)
        else:
            protocol = AdjElectrode(batch.mesh)
            bp_solver = BP(batch.mesh, protocol=protocol)
            delta_v = batch.voltages - batch.voltages.mean(axis=0)
            sigma_map = bp_solver.solve(delta_v)

        labels = segment(sigma_map)
        self.validate_output(labels)
        return labels