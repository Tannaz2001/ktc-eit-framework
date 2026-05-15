"""Experiment runner — loops over levels, samples, and methods from config."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.ktc_framework.adapters.method_registry import get as registry_get
from src.ktc_framework.loaders.ktc_loader import PluginRegistry
from src.ktc_framework.metrics.metric_registry import register_metric, run_all_metrics
from src.ktc_framework.metrics.ktc_score import compute_ktc_score, dice, iou
from src.ktc_framework.metrics.composite_score import composite_score, letter_grade
import src.ktc_framework.loaders.mock_data_plugin   # noqa: F401 — registers MockDataPlugin
import src.ktc_framework.loaders.ktc_data_plugin    # noqa: F401 — registers KTCDataPlugin
import src.ktc_framework.methods.mock_method_plugin  # noqa: F401 — registers MockMethodPlugin

# Register built-in metrics
register_metric("ktc_score",       compute_ktc_score)
register_metric("dice_resistive",  lambda pred, gt: dice(pred, gt, label=1))
register_metric("dice_conductive", lambda pred, gt: dice(pred, gt, label=2))
register_metric("iou_resistive",   lambda pred, gt: iou(pred, gt, label=1))
register_metric("iou_conductive",  lambda pred, gt: iou(pred, gt, label=2))

console = Console()


class BatchRunner:
    """
    Reads experiment.yaml config and runs each method across
    all selected difficulty levels and samples.
    """

    def __init__(self, config: dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[dict[str, Any]]:
        results = []

        levels = self.config.get("levels", [])
        samples = self.config.get("samples", [])
        methods = self.config.get("methods", [])

        total = len(methods) * len(levels) * len(samples)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiment...", total=total)

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
                        progress.advance(task)

        self._save(results)
        self._print_summary(results)
        self._print_degradation(results)
        return results

    def _run_one(self, method: str, level: int, sample: str) -> dict[str, Any] | None:
        plugin_name = self.config.get("data_plugin", "MockDataPlugin")
        dataset_root = self.config.get("dataset_root", "")

        try:
            data_plugin_cls = PluginRegistry.get(plugin_name)
        except KeyError:
            console.print(f"[yellow]data_plugin '{plugin_name}' not registered — falling back to MockDataPlugin.[/yellow]")
            from src.ktc_framework.loaders.mock_data_plugin import MockDataPlugin
            data_plugin_cls = MockDataPlugin

        data_plugin = data_plugin_cls(dataset_root)

        try:
            batch = data_plugin.load_sample(level=level, sample=sample)
        except FileNotFoundError:
            console.print(f"[yellow]Skipping level={level} sample={sample} — file not found in '{dataset_root}'.[/yellow]")
            return None
        except ValueError as exc:
            console.print(f"[red]Skipping level={level} sample={sample} — validation error: {exc}[/red]")
            return None

        try:
            method_plugin = registry_get(method)()
        except KeyError:
            console.print(f"[red]Method '{method}' not registered — skipping.[/red]")
            return None

        start = time.perf_counter()
        reconstruction = method_plugin.reconstruct(batch)
        runtime_ms = (time.perf_counter() - start) * 1000

        gt = batch.ground_truth
        metrics = run_all_metrics(reconstruction, gt)

        comp = composite_score(metrics)
        grade = letter_grade(comp)

        return {
            "method": method,
            "level": level,
            "sample": sample,
            "output_shape": list(reconstruction.shape),
            "metrics": metrics,
            "composite_score": comp,
            "grade": grade,
            "runtime_ms": round(runtime_ms, 3),
            "git_sha": self._git_sha(),
        }

    def _print_summary(self, results: list[dict[str, Any]]) -> None:
        table = Table(title="Experiment Summary", show_header=True, header_style="bold cyan", min_width=80)
        table.add_column("Method", style="bold", min_width=16)
        table.add_column("Level", justify="center", min_width=7)
        table.add_column("Sample", justify="center", min_width=8)
        table.add_column("KTC Score", justify="right", min_width=10)
        table.add_column("Dice Res.", justify="right", min_width=10)
        table.add_column("Dice Cond.", justify="right", min_width=11)
        table.add_column("Runtime (ms)", justify="right", min_width=13)

        for r in results:
            m = r["metrics"]
            table.add_row(
                r["method"],
                str(r["level"]),
                r["sample"],
                f"{m['ktc_score']:.3f}",
                f"{m['dice_resistive']:.3f}",
                f"{m['dice_conductive']:.3f}",
                f"{r['runtime_ms']:.2f}",
            )

        console.print()
        console.print(table)
        console.print(f"\n[green]scores.json saved to:[/green] {self.output_dir / 'scores.json'}")

    def _print_degradation(self, results: list[dict[str, Any]]) -> None:
        """Compute and print degradation slope per method using numpy.polyfit."""
        methods = list({r["method"] for r in results})
        slopes: dict[str, float] = {}

        for method in methods:
            method_results = [r for r in results if r["method"] == method]
            levels = sorted({r["level"] for r in method_results})
            avg_scores = []
            for level in levels:
                level_scores = [
                    r["metrics"]["ktc_score"]
                    for r in method_results
                    if r["level"] == level
                ]
                avg_scores.append(np.mean(level_scores))

            if len(levels) >= 2:
                slope = float(np.polyfit(levels, avg_scores, 1)[0])
            else:
                slope = 0.0

            slopes[method] = round(slope, 4)

        for result in results:
            result["degradation_slope"] = slopes[result["method"]]

        table = Table(
            title="Degradation Slope by Method",
            show_header=True,
            header_style="bold magenta",
            min_width=50,
        )
        table.add_column("Method", style="bold", min_width=20)
        table.add_column("Slope (per level)", justify="right", min_width=20)

        for method, slope in slopes.items():
            direction = "v degrades" if slope < 0 else "^ improves"
            table.add_row(method, f"{slope:+.4f}  {direction}")

        console.print()
        console.print(table)
        console.print("[dim]Steeper negative slope = method degrades faster at harder levels[/dim]")

    def _save(self, results: list[dict[str, Any]]) -> None:
        # Flat list for easy iteration
        out = self.output_dir / "scores.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Nested structure: method → level → sample → metrics
        nested: dict[str, Any] = {}
        for r in results:
            m = r["method"]
            lv = str(r["level"])
            s = r["sample"]
            nested.setdefault(m, {}).setdefault(lv, {})[s] = {
                "metrics": r["metrics"],
                "composite_score": r.get("composite_score", 0.0),
                "grade": r.get("grade", "D"),
                "runtime_ms": r["runtime_ms"],
                "degradation_slope": r.get("degradation_slope", 0.0),
            }

        nested_out = self.output_dir / "scores_nested.json"
        with nested_out.open("w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2)

    @staticmethod
    def _git_sha() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"
