# GREIT Method Adapter Only

This package contains only the GREIT method adapter and its small supporting contract classes.
It does **not** include the data loader because data loading and preprocessing are handled by another team member.

## What this adapter expects

The data-loader / batch-engine side must provide a `StandardizedKtcInput` object:

```python
StandardizedKtcInput(
    sample_id="level_4_sample_01",
    level=4,
    delta_v=standardized_vector_2356,
    active_measurement_mask=active_mask_2356,
    ground_truth=optional_256_by_256_mask,
)
```

The adapter expects:

- `delta_v`: already computed difference-voltage vector
- length: exactly `2356`
- missing measurements already padded by the pipeline
- optional `active_measurement_mask`
- level value from `1` to `7`

## Adapter contract

The GREIT adapter follows:

```python
preprocess(sample)
reconstruct(preprocessed_input)
postprocess(raw_reconstruction)
run(sample)
```

## Output

The adapter returns `AdapterResult` with:

- `raw_image`
- `normalized_image`
- `segmentation_mask`
- `runtime_seconds`
- `success`
- `metadata`

The final mask is always:

```text
256 x 256
labels: 0 = water, 1 = resistive, 2 = conductive
```

## Required GREIT model file

The model file should be:

```text
models/greit_fixed_2356.npz
```

It must contain:

```text
reconstruction_matrix
```

For a 64 x 64 GREIT image, matrix shape must be:

```text
4096 x 2356
```

## How to use from the batch engine

```python
from ktc_framework.contracts import StandardizedKtcInput
from ktc_framework.factory import GreitAdapterFactory

adapter = GreitAdapterFactory().from_yaml("config/greit_adapter.yaml")

sample = StandardizedKtcInput(
    sample_id="sample_01",
    level=1,
    delta_v=vector_2356,
    active_measurement_mask=mask_2356,
)

result = adapter.run(sample)
```

## Run tests

```bash
PYTHONPATH=src pytest -q
```
