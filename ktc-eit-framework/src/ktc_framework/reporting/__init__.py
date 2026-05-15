"""
reporting/__init__.py
---------------------
Generates a self-contained HTML leaderboard from a scores.json file produced
by BatchRunner.  Call generate_report() after a benchmark run.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def generate_report(scores_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Read scores.json and write a self-contained HTML leaderboard.

    Parameters
    ----------
    scores_path : str | Path
        Path to the scores.json file written by BatchRunner.
    output_path : str | Path | None
        Where to write the HTML file.  Defaults to scores.json's directory
        as ``report.html``.

    Returns
    -------
    Path
        Absolute path of the written HTML file.
    """
    scores_path = Path(scores_path)
    output_path = Path(output_path) if output_path else scores_path.parent / "report.html"

    with open(scores_path) as f:
        scores = json.load(f)

    rows = _build_rows(scores)
    html = _render_html(rows, scores_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_rows(scores: list[dict] | dict) -> list[dict]:
    """Flatten scores.json (list or dict) into a list of row dicts."""
    if isinstance(scores, dict):
        # nested format: {method: {level: {sample: {metric: val}}}}
        rows = []
        for method, levels in scores.items():
            for level, samples in levels.items():
                for sample, metrics in samples.items():
                    rows.append({"method": method, "level": level,
                                 "sample": sample, **metrics})
        return rows
    # flat list format
    return scores


def _metric_columns(rows: list[dict]) -> list[str]:
    skip = {"method", "level", "sample", "git_sha"}
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in skip and k not in cols:
                cols.append(k)
    return cols


def _render_html(rows: list[dict], scores_path: Path) -> str:
    cols = _metric_columns(rows)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    header_cells = "".join(
        f"<th>{c.replace('_', ' ').title()}</th>" for c in ["method", "level", "sample"] + cols
    )

    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{row.get(c, '')}</td>" for c in ["method", "level", "sample"] + cols
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>KTC EIT Benchmark — Leaderboard</title>
  <style>
    body {{ font-family: sans-serif; padding: 2rem; background: #f8f9fa; }}
    h1 {{ color: #2c3e50; }}
    p.meta {{ color: #666; font-size: 0.85rem; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff;
             box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    th {{ background: #2c3e50; color: #fff; padding: .6rem 1rem; text-align: left; }}
    td {{ padding: .5rem 1rem; border-bottom: 1px solid #e0e0e0; }}
    tr:hover td {{ background: #f0f4ff; }}
  </style>
</head>
<body>
  <h1>KTC EIT Benchmark — Leaderboard</h1>
  <p class="meta">Source: {scores_path.name} &nbsp;|&nbsp; Generated: {generated}</p>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>"""
