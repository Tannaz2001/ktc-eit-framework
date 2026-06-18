"""Experiment runner — loops over levels, samples, and methods from config."""

from __future__ import annotations

import json
import os
import subprocess
import time
import warnings
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from rich.console import Console  # type: ignore[import]
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn  # type: ignore[import]
from rich.table import Table  # type: ignore[import]

from src.ktc_framework.types import DataBatch
from src.ktc_framework.adapters.method_adapter import MethodAdapter
from src.ktc_framework.registry import (
    get_method as registry_get,
    list_methods as registry_list_methods,
    load_external_methods,
    PluginRegistry,
)
from src.ktc_framework.metrics.metric_registry import register_metric, run_all_metrics
from src.ktc_framework.metrics.ktc_score import compute_ktc_score
from src.ktc_framework.metrics.composite_score import composite_score, letter_grade
from src.ktc_framework.visualization import save_panel
from src.ktc_framework.visualization.plot_results import (
    save_figures,
    plot_failure_gallery,
    plot_degradation_curve,
    plot_leaderboard,
    plot_error_overlay,
)
from src.ktc_framework.plugins.hull_plugin import HullAnalyzer
from src.ktc_framework.metrics.qualitative_metrics import (
    compute_qualitative_sample,
    aggregate_qualitative,
)
from src.ktc_framework.reporting.html_report import generate_html_report

# Importing each package runs its __init__.py, which registers all plugins.
# To register a new method or data plugin, add it to the relevant __init__.py.
import src.ktc_framework.methods   # noqa: F401 — registers all reconstruction methods
import src.ktc_framework.loaders   # noqa: F401 — registers all data plugins

# Register built-in metrics once at module load.
# KTC score is the only metric (challenge constraint — see constraint.txt).
register_metric("ktc_score", compute_ktc_score)

console = Console(safe_box=True)  # ASCII box-drawing — safe on Windows cp1252 terminals


class BatchRunner:
    """Reads an experiment config dict and runs each method across
    all selected difficulty levels and samples.

    Parameters
    ----------
    config : dict
        Parsed YAML config with keys: data_plugin, mesh_path, dataset_root,
        levels, samples, methods, output_dir.
    output_dir : Path
        Directory where scores.json, images/, and figures/ are written.
    """

    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        method_plugin_paths = list(self.config.get("method_plugin_paths", []))
        default_external_path = Path("external_methods")
        if default_external_path.is_dir() and str(default_external_path) not in method_plugin_paths:
            method_plugin_paths.append(str(default_external_path))
        if method_plugin_paths:
            methods_before = set(registry_list_methods())
            load_external_methods(method_plugin_paths)
            if self.config.get("include_external_methods", False):
                configured_methods = list(self.config.get("methods", []))
                for method_name in registry_list_methods():
                    if method_name not in methods_before and method_name not in configured_methods:
                        configured_methods.append(method_name)
                self.config["methods"] = configured_methods

        # Load shared resources once — passed to every DataBatch at run time
        self.mesh = self._load_mesh(config.get("mesh_path", ""))
        self.ref_voltages = self._load_reference(config.get("dataset_root", ""))
        self.data_plugin = self._load_data_plugin()

    # ------------------------------------------------------------------
    # Resource loaders
    # ------------------------------------------------------------------

    def _load_mesh(self, mesh_path: str):
        """Load Mesh_sparse.mat and return the raw ``mat['Mesh']`` struct.

        Accepts a directory path (looks for ``Mesh_sparse.mat`` inside) or a
        direct path to the ``.mat`` file.  Falls back to a generated pyEIT
        32-electrode mesh if the file is absent; returns ``None`` if pyEIT is
        not installed.
        """
        if mesh_path:
            candidate = Path(mesh_path)
            mat_file = (candidate / "Mesh_sparse.mat") if candidate.is_dir() else candidate

            if mat_file.exists():
                try:
                    mat = scipy.io.loadmat(
                        str(mat_file), squeeze_me=True, struct_as_record=False
                    )
                    mesh_struct = mat["Mesh"]
                    mesh2_struct = mat.get("Mesh2")
                    if mesh2_struct is not None:
                        mesh_struct = SimpleNamespace(
                            Mesh=mesh_struct,
                            Mesh2=mesh2_struct,
                            H=mesh_struct.H,
                            g=mesh_struct.g,
                            elfaces=mesh_struct.elfaces,
                            Node=mesh_struct.Node,
                            Element=mesh_struct.Element,
                        )
                    console.print(
                        f"[green]Mesh loaded:[/green] {mat_file} "
                        f"({mesh_struct.g.shape[0]} nodes, "
                        f"{mesh_struct.H.shape[0]} elements)"
                    )
                    return mesh_struct
                except Exception as exc:
                    warnings.warn(
                        f"Could not load Mesh_sparse.mat ({exc}) — falling back to generated mesh.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    console.print(f"[yellow]Mesh load failed: {exc}[/yellow]")
            else:
                console.print(
                    f"[yellow]mesh_path '{mesh_path}' not found — "
                    f"falling back to generated mesh.[/yellow]"
                )

        # Fallback: generate a 32-electrode mesh using pyEIT
        try:
            from pyeit.mesh import create as mesh_create  # type: ignore[import]
            mesh_obj = mesh_create(n_el=32, h0=0.1)
            console.print(
                "[yellow]Using generated 32-electrode pyEIT mesh (no Mesh_sparse.mat).[/yellow]"
            )
            return mesh_obj
        except ImportError:
            console.print(
                "[yellow]pyeit not installed — mesh unavailable; "
                "BP/GaussNewton will use random fallback.[/yellow]"
            )
            return None
        except Exception as exc:
            console.print(f"[yellow]pyEIT mesh generation failed ({exc}) — returning None.[/yellow]")
            return None

    def _load_data_plugin(self):
        """Resolve and instantiate the configured data plugin once.

        Falls back to MockDataPlugin if the configured name is not registered,
        matching the previous per-run behaviour.
        """
        plugin_name  = self.config.get("data_plugin", "MockDataPlugin")
        dataset_root = self.config.get("dataset_root", "")

        try:
            data_plugin_cls = PluginRegistry.get(plugin_name)
        except KeyError:
            console.print(
                f"[yellow]data_plugin '{plugin_name}' not registered — "
                f"falling back to MockDataPlugin.[/yellow]"
            )
            from src.ktc_framework.loaders.mock_data_plugin import MockDataPlugin
            data_plugin_cls = MockDataPlugin

        return data_plugin_cls(dataset_root)

    def _load_reference(self, dataset_root: str) -> np.ndarray | None:
        """Load the empty-tank reference voltages from ``ref.mat``.

        Checks these locations in order:
          1. ``<dataset_root>/ref.mat``          (evaluation layout)
          2. ``<dataset_root>/TrainingData/ref.mat``  (training layout)

        Tries multiple key names: ``Uelref``, ``Uel``, ``Uref``, ``ref``.
        Returns a flat float32 array of shape ``(N,)`` on success, or
        ``None`` if the file is absent at both locations.
        """
        if not dataset_root:
            return None

        candidates = [
            os.path.join(dataset_root, "ref.mat"),
            os.path.join(dataset_root, "TrainingData", "ref.mat"),
        ]
        ref_keys = ["Uelref", "Uel", "Uref", "ref"]

        ref_path = next((p for p in candidates if os.path.exists(p)), None)

        if ref_path is None:
            console.print(
                f"[yellow]ref.mat not found — tried:[/yellow]\n"
                + "\n".join(f"  [yellow]{p}[/yellow]" for p in candidates)
                + "\n[yellow]Reconstruction methods will use mean-subtraction fallback.[/yellow]"
            )
            return None

        try:
            mat = scipy.io.loadmat(ref_path, squeeze_me=True, struct_as_record=False)
            key = next((k for k in ref_keys if k in mat), None)
            if key is None:
                pub = [k for k in mat if not k.startswith("_")]
                warnings.warn(
                    f"ref.mat loaded but no voltage key found. "
                    f"Keys present: {pub}. Using mean-subtraction fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            ref = np.asarray(mat[key], dtype=np.float32).ravel()
            console.print(
                f"[green]Reference voltages loaded:[/green] {ref_path} "
                f"(key='{key}', shape {ref.shape})"
            )
            return ref
        except Exception as exc:
            warnings.warn(
                f"Could not load ref.mat ({exc}) — using mean-subtraction fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            console.print(f"[yellow]ref.mat load failed: {exc}[/yellow]")
            return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Run all method × level × sample combinations and return results."""
        results: list[dict[str, Any]] = []

        levels  = self.config.get("levels",  [])
        samples = self.config.get("samples", [])
        methods = self.config.get("methods", [])

        total = len(methods) * len(levels) * len(samples)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("-"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiment...", total=total)

            done = 0
            for method in methods:
                for level in levels:
                    for sample in samples:
                        progress.update(
                            task,
                            description=f"method={method}  level={level}  sample={sample}",
                        )
                        result = self._run_one(method, level, sample)
                        if result is not None:
                            results.append(result)
                        done += 1
                        progress.advance(task)
                        # Machine-readable marker so the dashboard progress bar
                        # can parse completion without scraping Rich escape codes.
                        console.print(
                            f"[BENCH_PROGRESS] completed={done}/{total} "
                            f"method={method} level={level} sample={sample}",
                            highlight=False,
                        )

        n_gt_missing = sum(1 for r in results if r.get("gt_missing"))
        if n_gt_missing:
            console.print(
                f"\n[bold red]{n_gt_missing} of {len(results)} runs were scored "
                f"against an all-zero ground truth — their scores are "
                f"meaningless. Check the dataset layout / GT folder.[/bold red]"
            )

        self._save(results)
        self._print_summary(results)
        self._print_degradation(results)
        self._generate_visuals(results)
        return results

    def _run_one(
        self, method: str, level: int, sample: str
    ) -> dict[str, Any] | None:
        """Load one sample, run one reconstruction method, return scored result."""
        dataset_root = self.config.get("dataset_root", "")

        # ── load sample (plugin is constructed once in __init__) ──────────
        try:
            raw_batch = self.data_plugin.load_sample(level=level, sample=sample)
        except FileNotFoundError:
            console.print(
                f"[yellow]Skipping level={level} sample={sample} — "
                f"file not found in '{dataset_root}'.[/yellow]"
            )
            return None
        except ValueError as exc:
            console.print(
                f"[red]Skipping level={level} sample={sample} — "
                f"validation error: {exc}[/red]"
            )
            return None

        # ── augment batch with shared resources ───────────────────────────
        # Use the explicit DataBatch constructor (not _replace) so every field
        # is visible and the type signature stays correct.
        batch = DataBatch(
            voltages           = raw_batch.voltages,
            injection_patterns = raw_batch.injection_patterns,
            ground_truth       = raw_batch.ground_truth,
            level              = raw_batch.level,
            sample_id          = raw_batch.sample_id,
            mesh               = self.mesh,
            reference_voltages = (
                getattr(raw_batch, "reference_voltages", None)
                if getattr(raw_batch, "reference_voltages", None) is not None
                else self.ref_voltages
            ),
            measurement_patterns=getattr(raw_batch, "measurement_patterns", None),
        )

        # ── load method ───────────────────────────────────────────────────
        try:
            method_plugin = MethodAdapter(registry_get(method)())
        except KeyError:
            console.print(f"[red]Method '{method}' not registered — skipping.[/red]")
            return None

        # ── reconstruct ───────────────────────────────────────────────────
        start = time.perf_counter()
        try:
            reconstruction = method_plugin.reconstruct(batch)
        except Exception as exc:
            console.print(
                f"[red]Reconstruction failed for method={method} "
                f"level={level} sample={sample}: {exc}[/red]"
            )
            return None
        runtime_ms = (time.perf_counter() - start) * 1000

        # ── score ─────────────────────────────────────────────────────────
        gt = batch.ground_truth
        # An all-zero GT almost always means the file was missing/unreadable
        # and the loader fell back to zeros — every score against it is 0.0.
        gt_missing = not np.any(gt)
        if gt_missing:
            console.print(
                f"[red]Ground truth for level={level} sample={sample} is all "
                f"zeros (missing or unreadable GT file?) — KTC score for this "
                f"run is meaningless.[/red]"
            )
        metrics = run_all_metrics(reconstruction, gt)
        comp    = composite_score(metrics)
        grade   = letter_grade(comp)

        # ── save panel image ──────────────────────────────────────────────
        png_path = save_panel(
            gt         = gt,
            pred       = reconstruction,
            method     = method,
            level      = level,
            sample     = sample,
            output_dir = self.output_dir / "images",
            ktc_score  = metrics.get("ktc_score", 0.0),
        )

        mat_dir = self.output_dir / "mat_predictions" / method / f"level_{level}"
        mat_dir.mkdir(parents=True, exist_ok=True)
        mat_path = mat_dir / f"sample_{sample}.mat"
        scipy.io.savemat(
            str(mat_path),
            {"reconstruction": reconstruction.astype(np.uint8)},
        )

        overlay_dir = self.output_dir / "overlays" / method / f"level_{level}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"sample_{sample}.png"
        plot_error_overlay(
            pred=reconstruction,
            gt=gt,
            save_path=overlay_path,
        )

        # ── qualitative metrics (stored for later aggregation by method) ──
        # Store prediction and GT for hull analysis aggregation
        # Internal arrays (_pred, _gt) are stripped before JSON serialization

        return {
            "method":          method,
            "level":           level,
            "sample":          sample,
            "gt_missing":      gt_missing,
            "output_shape":    list(reconstruction.shape),
            "metrics":         metrics,
            "composite_score": comp,
            "grade":           grade,
            "runtime_ms":      round(runtime_ms, 3),
            "git_sha":         self._git_sha(),
            "png_path":        str(png_path),
            "mat_path":        str(mat_path),
            "overlay_path":    str(overlay_path),
            # internal arrays — stripped before JSON serialisation
            # used for hull analysis aggregation in _save()
            "_gt":   gt,
            "_pred": reconstruction,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save(self, results: list[dict[str, Any]]) -> None:
        """Write scores.json and scores_nested.json to output_dir."""
        # Strip arrays before JSON serialisation
        clean = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ]

        # Flat list
        out = self.output_dir / "scores.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)

        # Nested: method -> level -> sample -> metrics
        nested: dict[str, Any] = {}
        for r in clean:
            m  = r["method"]
            lv = str(r["level"])
            s  = r["sample"]
            nested.setdefault(m, {}).setdefault(lv, {})[s] = {
                "metrics":           r["metrics"],
                "composite_score":   r.get("composite_score", 0.0),
                "grade":             r.get("grade", "D"),
                "runtime_ms":        r["runtime_ms"],
                "degradation_slope": r.get("degradation_slope", 0.0),
            }

        nested_out = self.output_dir / "scores_nested.json"
        with nested_out.open("w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2)

        # Compute qualitative metrics (hull detection) per method
        self._compute_qualitative_metrics(results, nested)

        dashboard_scores: dict[str, dict[str, float]] = {}
        per_run: dict[str, dict[str, Any]] = {}

        for method in sorted({r["method"] for r in clean}):
            method_rows = [r for r in clean if r["method"] == method]
            metric_names = sorted(
                {
                    metric
                    for row in method_rows
                    for metric in row.get("metrics", {}).keys()
                }
            )
            dashboard_scores[method] = {
                metric: round(
                    float(np.mean([row["metrics"].get(metric, 0.0) for row in method_rows])),
                    6,
                )
                for metric in metric_names
            }

            per_run[method] = {}
            for row in method_rows:
                key = f"L{row['level']}_{row['sample']}"
                per_run[method][key] = {
                    **row.get("metrics", {}),
                    "composite_score": row.get("composite_score", 0.0),
                    "grade": row.get("grade", "D"),
                    "runtime_ms": row.get("runtime_ms", 0.0),
                    "level": row["level"],
                    "sample": row["sample"],
                    "gt_missing": row.get("gt_missing", False),
                    "png_path": row.get("png_path", ""),
                    "mat_path": row.get("mat_path", ""),
                    "overlay_path": row.get("overlay_path", ""),
                    "hull": row.get("hull", {}),
                }

        with (self.output_dir / "dashboard_scores.json").open("w", encoding="utf-8") as f:
            json.dump(dashboard_scores, f, indent=2)

        with (self.output_dir / "per_run_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(per_run, f, indent=2)

    def _compute_qualitative_metrics(
        self, results: list[dict[str, Any]], nested: dict[str, Any]
    ) -> None:
        """Compute qualitative (hull detection) metrics per method.

        Hull analysis extracts convex hulls from predicted and ground-truth
        segmentations, compares them via rasterized IoU (threshold=0.3),
        and counts detection successes across all samples.

        Groups results by method, computes per-sample flags (resistive_detected,
        conductive_detected, hull_iou, centroid_distance, false_positives),
        and aggregates across all samples.

        Example output in scores_nested.json:
            "method_name": {
                "_qualitative_summary": {
                    "resistive_detected_str": "19/21",
                    "resistive_detected_pct": 90.5,
                    "avg_resistive_hull_iou": 0.723,
                    ...
                },
                "_qualitative_per_sample": [...]
            }

        Wrapped in try/except — pipeline continues even if hull analysis fails.
        """
        hull_analyzer = HullAnalyzer()

        methods = {r["method"] for r in results}

        for method in methods:
            method_results = [r for r in results if r["method"] == method]

            # Skip if no valid internal arrays
            if not any(r.get("_pred") is not None and r.get("_gt") is not None for r in method_results):
                continue

            qual_samples = []

            try:
                for result in method_results:
                    pred = result.get("_pred")
                    gt = result.get("_gt")

                    if pred is None or gt is None:
                        continue

                    # Compute qualitative flags for this sample
                    qual = compute_qualitative_sample(pred, gt, hull_analyzer)
                    qual["sample_id"] = f"L{result['level']}_{result['sample']}"
                    qual["level"] = result["level"]
                    qual["sample"] = result["sample"]
                    qual_samples.append(qual)

                # Aggregate across all samples for this method
                if qual_samples:
                    qual_summary = aggregate_qualitative(qual_samples)

                    # Store in nested dict under method
                    if method in nested:
                        nested[method]["_qualitative_summary"] = qual_summary
                        nested[method]["_qualitative_per_sample"] = qual_samples

            except Exception as exc:
                console.print(
                    f"[yellow]Qualitative metrics aggregation failed for {method}: {exc}[/yellow]"
                )

    def _print_summary(self, results: list[dict[str, Any]]) -> None:
        """Print a rich table of per-run metrics."""
        table = Table(
            title="Experiment Summary",
            show_header=True,
            header_style="bold cyan",
            min_width=80,
        )
        table.add_column("Method",        style="bold", min_width=20)
        table.add_column("Level",         justify="center", min_width=7)
        table.add_column("Sample",        justify="center", min_width=8)
        table.add_column("KTC Score",     justify="right",  min_width=10)
        table.add_column("Runtime (ms)",  justify="right",  min_width=13)

        for r in results:
            m = r["metrics"]
            table.add_row(
                r["method"],
                str(r["level"]),
                r["sample"],
                f"{m['ktc_score']:.3f}",
                f"{r['runtime_ms']:.2f}",
            )

        console.print()
        console.print(table)
        console.print(
            f"\n[green]scores.json saved to:[/green] {self.output_dir / 'scores.json'}"
        )

    def _print_degradation(self, results: list[dict[str, Any]]) -> None:
        """Compute and print the degradation slope (KTC score vs level) per method."""
        methods = list({r["method"] for r in results})
        slopes: dict[str, float] = {}

        for method in methods:
            method_results = [r for r in results if r["method"] == method]
            levels = sorted({r["level"] for r in method_results})
            avg_scores = []
            for lv in levels:
                level_scores = [
                    r["metrics"]["ktc_score"]
                    for r in method_results
                    if r["level"] == lv
                ]
                avg_scores.append(float(np.mean(level_scores)))

            slope = (
                float(np.polyfit(levels, avg_scores, 1)[0])
                if len(levels) >= 2
                else 0.0
            )
            slopes[method] = round(slope, 4)

        # Attach slope to every result dict for _save
        for result in results:
            result["degradation_slope"] = slopes[result["method"]]

        table = Table(
            title="Degradation Slope by Method",
            show_header=True,
            header_style="bold magenta",
            min_width=50,
        )
        table.add_column("Method",            style="bold", min_width=20)
        table.add_column("Slope (per level)", justify="right", min_width=20)

        for method, slope in slopes.items():
            direction = "v degrades" if slope < 0 else "^ improves"
            table.add_row(method, f"{slope:+.4f}  {direction}")

        console.print()
        console.print(table)
        console.print(
            "[dim]Steeper negative slope = method degrades faster at harder levels[/dim]"
        )

    def _generate_visuals(self, results: list[dict[str, Any]]) -> None:
        """Save comparison PNGs, failure gallery, degradation curve, leaderboard, HTML report."""
        try:
            saved = save_figures(results, self.output_dir)
            plot_failure_gallery(results, self.output_dir)
            plot_degradation_curve(results, self.output_dir)
            plot_leaderboard(results, self.output_dir)

            # Read qualitative metrics from scores_nested.json for HTML report
            nested_path = self.output_dir / "scores_nested.json"
            qualitative_data = {}
            if nested_path.exists():
                try:
                    with nested_path.open("r", encoding="utf-8") as f:
                        nested = json.load(f)
                        for method in nested:
                            if "_qualitative_summary" in nested[method]:
                                qualitative_data[method] = nested[method]["_qualitative_summary"]
                except Exception as e:
                    console.print(f"[yellow]Could not load qualitative data: {e}[/yellow]")

            report_path = generate_html_report(results, self.output_dir, qualitative_data)
            console.print(
                f"\n[green]Figures saved:[/green] {len(saved)} PNGs -> {self.output_dir / 'figures'}"
            )
            console.print(f"[green]HTML report:[/green]   {report_path}")
        except Exception as exc:
            console.print(f"[yellow]Visualization skipped: {exc}[/yellow]")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _git_sha() -> str:
        """Return the current git short SHA, or 'unknown' if git is unavailable."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"
