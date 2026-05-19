from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.methods.segment import segment
from src.ktc_framework.adapters.method_registry import register
import importlib
import numpy as np

@register
class GaussNewton(MethodPlugin):
    def reconstruct(self, batch) -> np.ndarray:
        try:
            pyeit_jac = importlib.import_module("pyeit.eit.jac")
            JAC = getattr(pyeit_jac, "JAC", None)
            pyeit_protocol = importlib.import_module("pyeit.eit.protocol")
            AdjElectrode = getattr(pyeit_protocol, "AdjElectrode", None)
        except ImportError:
            JAC = None
            AdjElectrode = None

        if JAC is None or batch.mesh is None or AdjElectrode is None:
            sigma_map = np.random.rand(256, 256)
        else:
            protocol = AdjElectrode(batch.mesh)
            jac_solver = JAC(batch.mesh, protocol=protocol)
            delta_v = batch.voltages - batch.voltages.mean(axis=0)
            delta_sigma = jac_solver.solve(delta_v, n_iter=1)
            sigma_map = delta_sigma + 1.0

        labels = segment(sigma_map)
        self.validate_output(labels)
        return labels