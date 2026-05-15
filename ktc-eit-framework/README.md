# ktc-eit-framework
Modular framework for evaluating Electrical Impedance Tomography (EIT) reconstruction methods using the Kuopio Tomography Challenge dataset. Supports data loading, method integration, evaluation metrics, and visualization, with async batch processing to run multiple difficulty levels in parallel for efficient large-scale experiments.

## Testing batch processing

You do not need the full KTC dataset to test the batch module.

- Unit and integration tests in `tests/` create mini synthetic files automatically using `tmp_path`.
- This validates level classification, file-format filtering, missing-value handling, and asyncio parallel level execution.

Install requirements and run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Where to put the real dataset

For real experiment runs, keep the KTC dataset outside or inside the repo, then pass the path with `--dataset-root`.

Common options:

- Inside repo: `ktc-eit-framework/data/ktc/`
- Outside repo: any folder (recommended for large data), e.g. `D:/datasets/ktc/`

Example:

```bash
python run_experiment.py --dataset-root "D:/datasets/ktc" --levels 1 7 --data-format numpy --method greit
```
