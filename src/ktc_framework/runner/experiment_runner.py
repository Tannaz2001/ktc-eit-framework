"""Experiment runner — loops over levels, samples, and methods from config."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

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
                        results.append(result)
                        progress.advance(task)

        self._save(results)
        self._print_summary(results)
        return results

    def _run_one(self, method: str, level: int, sample: str) -> dict[str, Any]:
        start = time.perf_counter()

        # Placeholder — reconstruction logic will be wired in W4
        runtime_ms = (time.perf_counter() - start) * 1000

        return {
            "method": method,
            "level": level,
            "sample": sample,
            "metrics": {
                "ktc_score": 0.0,
                "dice_resistive": 0.0,
                "dice_conductive": 0.0,
                "iou_resistive": 0.0,
                "iou_conductive": 0.0,
            },
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

    def _save(self, results: list[dict[str, Any]]) -> None:
        out = self.output_dir / "scores.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def _git_sha() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"
