from __future__ import annotations

import numpy as np

from ktc_framework.contracts import StandardizedKtcInput
from ktc_framework.factory import GreitAdapterFactory


class DemoOnlyCommand:
    """
    This is not a data loader.
    It only demonstrates how Sahil/Tannaz's modules should call the adapter
    after they have already produced standardized input.
    """

    def run(self) -> None:
        adapter = GreitAdapterFactory().from_yaml("config/greit_adapter.yaml")

        sample = StandardizedKtcInput(
            sample_id="demo_sample",
            level=1,
            delta_v=np.zeros(2356),
            active_measurement_mask=np.ones(2356, dtype=bool),
            metadata={"source": "provided_by_batch_engine"},
        )

        result = adapter.run(sample)
        print(result.method_name, result.sample_id, result.level, result.segmentation_mask.shape)


if __name__ == "__main__":
    DemoOnlyCommand().run()
