"""
Data plugin registration manifest.

Every plugin listed here is registered in the DataPluginRegistry on import
and becomes available by name in the data_plugin field of experiment configs.

To add a new data plugin:
  1. Create your class in this directory.
  2. Decorate it with @DataPluginRegistry.register("YourName") (from src.ktc_framework.registry).
  3. Add one import line below.
"""

# fmt: off
from src.ktc_framework.loaders.mock_data_plugin import MockDataPlugin              # noqa: F401
from src.ktc_framework.loaders.ktc_data_plugin import KTCDataPlugin               # noqa: F401
from src.ktc_framework.loaders.training_data_plugin import TrainingDataPlugin      # noqa: F401
from src.ktc_framework.loaders.phantom_data_plugin import PhantomDataPlugin        # noqa: F401
# fmt: on
