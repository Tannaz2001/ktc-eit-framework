"""
Reconstruction method registration manifest.

Every method listed here is registered in the MethodRegistry on import
and becomes available by its class name in experiment configs.

To add a new method:
  1. Create your class in this directory.
  2. Decorate it with @register_method (from src.ktc_framework.registry).
  3. Add one import line below.
"""

# fmt: off
from src.ktc_framework.methods.mock_method_plugin import MockMethodPlugin          # noqa: F401
from src.ktc_framework.methods.backprojection import BackProjection                # noqa: F401
from src.ktc_framework.methods.gauss_newton import GaussNewton                    # noqa: F401
from src.ktc_framework.methods.reference_fem import (                             # noqa: F401
    LinearDifferenceReconstruction,
    RegularizedFEMReconstruction,
    ReferenceFEM,
)
from src.ktc_framework.methods.groundtruth_oracle import GroundTruthOracle        # noqa: F401
# fmt: on
