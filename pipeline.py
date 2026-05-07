from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any
import numpy as np

from batch_processing.parallel_runner import ParallelLevelRunner
from batch_processing.level_selector import (
    SampleFile,
    classify_sample_level,
    get_extensions_for_format,
    normalize_levels,
)
from batch_processing.sample_loader import ensure_standardized_sample
from loader.feature_eng import FeatureTransformer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an EIT reconstruction experiment."""
    dataset_root: Path
    levels: list[int]
    method: str
    data_format: str
    per_level_concurrency: int
    verbose_missing: bool


class ExperimentPipeline:
    """
    Constructs and executes the data pipeline for the experiment.
    Encapsulates all domain logic outside the CLI entrypoint.
    """
    
    def __init__(self, config: ExperimentConfig):
        logger.info("Initializing ExperimentPipeline...")
        self.config = config
        self.levels = normalize_levels(config.levels)
        self.extensions = get_extensions_for_format(config.data_format)
        
        # Instantiate the feature transformer once per pipeline
        self.transformer = FeatureTransformer(use_gp_interpolation=True)

    def run(self):
        """Executes the pipeline across all configured levels."""
        logger.info("Starting pipeline execution across levels: %s", self.levels)
        
        # 0. Fit global scalers to avoid data leakage
        self._fit_transformer()

        runner = ParallelLevelRunner(
            process_sample=self.process_sample,
            per_level_concurrency=self.config.per_level_concurrency,
        )
        reports = runner.run_levels(
            self.config.dataset_root, 
            self.levels, 
            extensions=self.extensions
        )
        
        self._print_reports(reports)
        return reports

    def _print_reports(self, reports):
        """Helper to print batch execution results."""
        logger.info("Batch execution results gathered.")
        logger.info("Data format: %s | extensions: %s", self.config.data_format, self.extensions)
        logger.info("Parallel levels launched: %s", self.levels)
        for level in self.levels:
            report = reports[level]
            logger.info(
                "Level %d: processed=%d, skipped=%d, errors=%d, duration=%.2fs",
                level, report.processed, report.skipped, report.errors, report.duration_seconds
            )

    def _fit_transformer(self):
        """
        Fits the feature transformer on a small reference set to compute global scaling
        statistics, ensuring consistent scaling across all parallel workers.
        """
        logger.info("Preparing to fit global scaler on a reference subset...")
        reference_samples = []
        import itertools
        from batch_processing.sample_loader import load_standardized_sample
        from batch_processing.level_selector import iter_samples_for_levels

        sample_iterator = iter_samples_for_levels(
            self.config.dataset_root, 
            self.levels, 
            extensions=self.extensions
        )
        
        # Grab the first 10 valid samples to act as our "held-out" scaling reference
        for sample_file in itertools.islice(sample_iterator, 10):
            try:
                eit_sample = load_standardized_sample(sample_file)
                reference_samples.append(eit_sample)
            except Exception:
                pass
                
        if reference_samples:
            self.transformer.fit(reference_samples)
            logger.info("Scaler fitted successfully. Global Mean: %.4f, Global Std: %.4f", 
                        self.transformer.global_mean, self.transformer.global_std)
        else:
            logger.warning("No valid reference samples found for fitting. Scaling might fail.")

    def process_sample(self, sample: SampleFile | np.ndarray | dict[str, Any]) -> None:
        """
        Processes a single sample through the pipeline.
        This handles loading, standardization, missing value imputation,
        and eventually feature engineering & reconstruction.
        """
        # Note: this executes within a worker process
        eit_sample = ensure_standardized_sample(sample)
        
        # 1. Feature Engineering & Clean Contract
        processed = self.transformer.process(eit_sample)

        if self.config.verbose_missing:
            # Report imputed values if any
            missing_count = int(np.sum(processed.gp_mask)) if processed.gp_mask is not None else 0
            if missing_count > 0:
                logger.info(
                    "[gp-imputed] sample=%s level=%s imputed=%d values", 
                    processed.sample_id, processed.level, missing_count
                )

        # Validate Level Agreement
        if isinstance(sample, SampleFile):
            detected_level = classify_sample_level(sample.path)
            if detected_level != sample.level:
                raise ValueError(
                    f"Level mismatch for sample {sample.sample_id}: "
                    f"selector={sample.level}, classifier={detected_level}"
                )

        # TODO: Pass processed.delta_v to the selected reconstruction method
        _ = processed.delta_v
        time.sleep(0.001)
        _ = self.config.method
