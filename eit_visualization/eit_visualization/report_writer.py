"""
report_writer.py — Minimal real-data HTML report.

Reads a scores.json file produced by example_usage.py and writes a
self-contained HTML page summarising the real metrics computed by the
ktc-eit-framework on the real KTC training samples.

No external template engines, no dummy values.
"""

from __future__ import annotations

import json
import html
from datetime import datetime
from pathlib import Path
from typing import Any


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EIT Reconstruction Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 2rem auto; max-width: 960px; color: #1a3a5c;
    background: #fafafa;
  }}
  h1   {{ border-bottom: 3px solid #1D9E75; padding-bottom: .4rem; }}
  h2   {{ color: #1a3a5c; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; background: white; }}
  th, td {{ padding: .55rem .9rem; border: 1px solid #e0e0e0; text-align: left; }}
  th     {{ background: #1a3a5c; color: white; }}
  tr:nth-child(even) td {{ background: #f4f7fa; }}
  .meta  {{ color: #666; font-size: .9rem; margin-bottom: 1.5rem; }}
  .gallery {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem; margin-top: 1rem;
  }}
  .gallery figure {{ margin: 0; background: white; padding: .5rem;
                     border: 1px solid #e0e0e0; border-radius: 4px; }}
  .gallery img {{ width: 100%; height: auto; display: block; }}
  .gallery figcaption {{ font-size: .85rem; text-align: center;
                         padding-top: .3rem; color: #555; }}
  code {{ background: #eef; padding: 1px 5px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>EIT Reconstruction Report</h1>
<div class="meta">Generated {when} from <code>{src}</code></div>
{body}
</body>
</html>
"""


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return html.escape(str(v))


def _scores_table(scores: dict) -> str:
    """Render a metrics dict as an HTML table.

    Supports two shapes:
      flat  : { "Dice": 0.82, "IoU": 0.75, ... }
      nested: { "method_name": { "metric": value, ... }, ... }
    """
    if not scores:
        return "<p><em>scores.json was empty.</em></p>"

    first_val = next(iter(scores.values()))
    rows: list[str] = []

    if isinstance(first_val, dict):
        # nested: rows are methods, columns are metrics
        all_metrics: list[str] = []
        for m in scores.values():
            for k in m.keys():
                if k not in all_metrics:
                    all_metrics.append(k)
        header = "<tr><th>Method</th>" + "".join(
            f"<th>{html.escape(m)}</th>" for m in all_metrics
        ) + "</tr>"
        for name, metrics in scores.items():
            row = f"<tr><td><strong>{html.escape(str(name))}</strong></td>"
            for m in all_metrics:
                row += f"<td>{_format_value(metrics.get(m, '–'))}</td>"
            row += "</tr>"
            rows.append(row)
        return f"<table>{header}{''.join(rows)}</table>"

    # flat metrics
    header = "<tr><th>Metric</th><th>Value</th></tr>"
    for k, v in scores.items():
        rows.append(
            f"<tr><td>{html.escape(str(k))}</td><td>{_format_value(v)}</td></tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def _gallery(outputs_dir: Path, rel_prefix: str = "../outputs") -> str:
    """Pick up every real PNG produced by example_usage.py and embed it."""
    if not outputs_dir.exists():
        return ""
    pngs = sorted(outputs_dir.glob("*.png"))
    if not pngs:
        return ""
    figs: list[str] = []
    for p in pngs:
        rel = f"{rel_prefix}/{p.name}"
        figs.append(
            f'<figure><img src="{html.escape(rel)}" alt="{html.escape(p.stem)}">'
            f"<figcaption>{html.escape(p.stem)}</figcaption></figure>"
        )
    return "<div class='gallery'>" + "".join(figs) + "</div>"


def generate_report(
    scores_path: str = "scores.json",
    out_path: str = "reports/report.html",
    outputs_dir: str = "outputs",
) -> str:
    """Render the HTML report. Returns the output path."""
    scores_p = Path(scores_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if scores_p.exists():
        with scores_p.open(encoding="utf-8") as f:
            scores = json.load(f)
    else:
        scores = {}

    body_parts: list[str] = []
    body_parts.append("<h2>Real metrics</h2>")
    body_parts.append(_scores_table(scores))

    body_parts.append("<h2>Figures</h2>")
    body_parts.append(_gallery(Path(outputs_dir)))

    page = _PAGE.format(
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        src=html.escape(str(scores_p)),
        body="\n".join(body_parts),
    )
    out_p.write_text(page, encoding="utf-8")
    print(f"Saved report to {out_p}")
    return str(out_p)


if __name__ == "__main__":
    generate_report()
