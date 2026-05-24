"""MockDataPlugin — synthetic EIT DataBatch generator for testing.

Generates deterministic DataBatch objects whose array shapes exactly match
the real KTC 2023 evaluation dataset:

    voltages           (2356,)    float32   — 76 injections × 31 voltage pairs
    injection_patterns (32, 76)   float32   — adjacent-pair current protocol
    ground_truth       (256, 256) uint8     — labels {0=water, 1=resistive, 2=conductive}

The RNG is seeded from ``hash(f'{level}_{sample}')`` so the same
``(level, sample)`` pair **always** produces identical arrays — essential for
reproducible unit tests and pipeline smoke tests.
"""

from __future__ import annotations

from typing import Generator, Optional

import numpy as np

from src.ktc_framework.loaders.ktc_loader import PluginRegistry
from src.ktc_framework.types import DataBatch


# ---------------------------------------------------------------------------
# Constants — fixed by the KTC dataset specification
# ---------------------------------------------------------------------------

_N_VOLTAGES:   int = 2356   # 76 injection patterns × 31 differential pairs
_N_ELECTRODES: int = 32
_N_INJ_COLS:   int = 76

# Ground-truth class weights: background dominates at ~90 %
_LABEL_PROBS: list[float] = [0.90, 0.05, 0.05]


# ---------------------------------------------------------------------------
# MockDataPlugin
# ---------------------------------------------------------------------------

@PluginRegistry.register("MockDataPlugin")
class MockDataPlugin:
    """Synthetic DataBatch generator with KTC-correct array shapes.

    Accepts the same constructor argument as the real data plugins so
    ``BatchRunner`` can swap between them transparently.  The argument is
    stored but never used to load files.

    Parameters
    ----------
    dataset_root_or_config:
        Either a plain string (how ``BatchRunner`` calls it — passes the
        ``dataset_root`` value from the config) or a config dict with an
        optional ``'dataset_root'`` key.  Both are accepted; neither is used
        for synthetic generation.
    """

    def __init__(self, dataset_root_or_config: str | dict = "") -> None:
        # Accept both plain string (BatchRunner API) and config dict (direct use)
        if isinstance(dataset_root_or_config, dict):
            self.config: dict       = dataset_root_or_config
            self.dataset_root: str  = str(dataset_root_or_config.get("dataset_root", ""))
        else:
            self.config        = {}
            self.dataset_root  = str(dataset_root_or_config)

    # ── public interface ────────────────────────────────────────────────────

    def load_sample(self, level: int, sample: str) -> DataBatch:
        """Return a deterministic synthetic DataBatch for *level* / *sample*.

        The RNG is seeded from ``hash(f'{level}_{sample}')`` so every call
        with the same arguments returns bit-for-bit identical arrays.

        Parameters
        ----------
        level:
            Difficulty level (1–7).  Used only as part of the seed.
        sample:
            Sample identifier (``'A'``, ``'B'``, ``'C'``, …).  Used only as
            part of the seed.

        Returns
        -------
        DataBatch
            Fully populated batch.  ``mesh`` and ``reference_voltages`` are
            always ``None`` — ``BatchRunner`` attaches the shared resources.
        """
        seed = hash(f"{level}_{sample}") % (2 ** 32)
        rng  = np.random.RandomState(seed)

        # ── voltages: (2356,) float32 ──────────────────────────────────────
        voltages = rng.randn(_N_VOLTAGES).astype(np.float32)

        # ── injection patterns: (32, 76) float32 ──────────────────────────
        injection_patterns = self._make_injection_patterns()

        # ── ground truth: (256, 256) uint8, 90 % / 5 % / 5 % split ───────
        gt_flat      = rng.choice([0, 1, 2], size=(256 * 256), p=_LABEL_PROBS)
        ground_truth = gt_flat.reshape(256, 256).astype(np.uint8)

        return DataBatch(
            voltages           = voltages,
            injection_patterns = injection_patterns,
            ground_truth       = ground_truth,
            level              = level,
            sample_id          = f"mock_level{level}_{sample}",
            mesh               = None,   # filled by BatchRunner._run_one
            reference_voltages = None,   # filled by BatchRunner._run_one
        )

    # ── extra helpers (kept for test-suite / notebook use) ─────────────────

    def get_batch(
        self,
        n_samples: int = 1,
        level: int = 1,
        sample_id: Optional[str] = None,
    ) -> DataBatch:
        """Return one synthetic DataBatch, addressable by an arbitrary *sample_id*.

        Delegates to :meth:`load_sample` using the last character of
        *sample_id* as the sample letter so seeding is consistent.

        Parameters
        ----------
        n_samples:
            Accepted for API symmetry; ignored (one batch is always returned).
        level:
            Difficulty level passed through to :meth:`load_sample`.
        sample_id:
            Arbitrary identifier; defaults to ``'mock-0000'``.
        """
        sid    = sample_id if sample_id is not None else "mock-0000"
        sample = sid[-1] if sid else "A"
        batch  = self.load_sample(level=level, sample=sample)
        # Override sample_id so callers get back the id they supplied
        return batch._replace(sample_id=sid)

    def iter_batches(
        self,
        n_batches: int,
        n_samples: int = 1,
        level: int = 1,
    ) -> Generator[DataBatch, None, None]:
        """Yield *n_batches* distinct synthetic DataBatch objects.

        Each batch has a unique ``sample_id`` (``'mock-0000'``, ``'mock-0001'``, …)
        so seeds differ and the arrays are independent.
        """
        for i in range(n_batches):
            yield self.get_batch(
                n_samples=n_samples,
                level=level,
                sample_id=f"mock-{i:04d}",
            )

    # ── private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_injection_patterns() -> np.ndarray:
        """Build the KTC-style adjacent-pair injection matrix.

        Each row corresponds to one electrode; each column to one injection
        pattern.  For injection pattern ``j``:
        * electrode ``j % 32`` injects  (+1)
        * electrode ``(j+1) % 32`` sinks (−1)

        Returns
        -------
        np.ndarray
            Shape ``(32, 76)`` float32.
        """
        patterns = np.zeros((_N_ELECTRODES, _N_INJ_COLS), dtype=np.float32)
        for j in range(_N_INJ_COLS):
            source = j % _N_ELECTRODES
            sink   = (j + 1) % _N_ELECTRODES
            patterns[source, j] =  1.0
            patterns[sink,   j] = -1.0
        return patterns
