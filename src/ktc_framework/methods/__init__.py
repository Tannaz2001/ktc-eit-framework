"""
Reconstruction method registration manifest.

Every method listed here is registered in the MethodRegistry on import
and becomes available by its class name in experiment configs.

To add a new method:
  1. Create your class in this directory.
  2. Decorate it with @register_method (from ktc_framework.registry).
  3. Add one import line below.
"""

# fmt: off
from ktc_framework.methods.backprojection import BackProjection                # noqa: F401
from ktc_framework.methods.gauss_newton import GaussNewton                    # noqa: F401
from ktc_framework.methods.reference_fem import (                             # noqa: F401
    LinearDifferenceReconstruction,
    RegularizedFEMReconstruction,
    ReferenceFEM,
)
from ktc_framework.methods.competition_cnn import CompetitionCNN              # noqa: F401
# fmt: on
