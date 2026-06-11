# Runtime Method Adapter

This framework can load reconstruction methods at runtime from external Python
files. This lets an evaluator add a new method without editing the framework's
source package.

## 1. Add a plugin folder

Create a folder outside or inside the project, for example:

```text
external_methods/
```

## 2. Add the method file

Example file:

```text
external_methods/professor_method.py
```

```python
import numpy as np

from src.ktc_framework.registry import register_method
from src.ktc_framework.types import DataBatch


@register_method
class ProfessorMethod:
    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        """Return a 256x256 label image: 0=background, 1=resistive, 2=conductive."""
        reconstruction = np.zeros((256, 256), dtype=np.uint8)

        # Implement reconstruction using:
        # batch.voltages
        # batch.injection_patterns
        # batch.mesh
        # batch.reference_voltages
        # batch.measurement_patterns

        return reconstruction
```

The class name is the method name used in YAML.

## 3. Enable it in config

```yaml
methods:
  - BackProjection
  - GaussNewton
  - ProfessorMethod

method_plugin_paths:
  - external_methods
```

When the experiment starts, every `.py` file in `method_plugin_paths` is
imported. Importing the file runs `@register_method`, so `ProfessorMethod`
becomes available like any built-in method.

## Output Contract

External methods must implement:

```python
def reconstruct(self, batch: DataBatch) -> np.ndarray:
    ...
```

The method may return either:

```python
return reconstruction
```

or:

```python
return {"reconstruction": reconstruction, "metadata": {}}
```

The adapter validates that the final reconstruction is a `256x256` array and
converts it to `uint8` before scoring, saving, and reporting.
