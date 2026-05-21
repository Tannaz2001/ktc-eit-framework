"""HTML report generator — integrated from Areeba's report_writer.py."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_html_report(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate outputs/report.html from experiment results."""
    output_dir  = Path(output_dir)
    figures_dir = output_dir / "figures"

    # -- summary stats -------------------------------------------------------
    methods     = list({r["method"] for r in results})
    total_runs  = len(results)
    best        = max(results, key=lambda r: r.get("composite_score", 0), default=None)
    best_method = best["method"] if best else "N/A"
    best_score  = round(best.get("composite_score", 0), 4) if best else 0.0

    # -- leaderboard ---------------------------------------------------------
    leaderboard: dict[str, dict] = {}
    for r in results:
        m = r["method"]
        if m not in leaderboard:
            leaderboard[m] = {"ktc": [], "dice_r": [], "dice_c": [], "runtime": [], "composite": []}
        metrics = r.get("metrics", {})
        leaderboard[m]["ktc"].append(metrics.get("ktc_score", 0))
        leaderboard[m]["dice_r"].append(metrics.get("dice_resistive", 0))
        leaderboard[m]["dice_c"].append(metrics.get("dice_conductive", 0))
        leaderboard[m]["runtime"].append(r.get("runtime_ms", 0))
        leaderboard[m]["composite"].append(r.get("composite_score", 0))

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    leaderboard_rows = sorted(
        [{"method": m,
          "avg_composite": avg(v["composite"]),
          "avg_ktc":       avg(v["ktc"]),
          "avg_dice_r":    avg(v["dice_r"]),
          "avg_dice_c":    avg(v["dice_c"]),
          "avg_runtime":   avg(v["runtime"]),
          "runs":          len(v["ktc"])}
         for m, v in leaderboard.items()],
        key=lambda x: x["avg_composite"], reverse=True,
    )

    lb_html = ""
    for i, row in enumerate(leaderboard_rows, 1):
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else str(i)
        lb_html += f"""
        <tr>
            <td>{medal}</td>
            <td><strong>{html.escape(row['method'])}</strong></td>
            <td>{row['avg_composite']}</td>
            <td>{row['avg_ktc']}</td>
            <td>{row['avg_dice_r']}</td>
            <td>{row['avg_dice_c']}</td>
            <td>{row['avg_runtime']} ms</td>
            <td>{row['runs']}</td>
        </tr>"""

    # -- per-run detail rows -------------------------------------------------
    detail_rows = ""
    grade_color = {"A": "#1D9E75", "B": "#4A90E2", "C": "#F5A623", "D": "#D85A30"}
    for r in results:
        m     = r.get("metrics", {})
        grade = r.get("grade", "D")
        color = grade_color.get(grade, "#888")
        fname = f"{r['method']}_level{r['level']}_sample{r['sample']}.png"
        img   = f'<img src="figures/{html.escape(fname)}" width="180">' \
                if (figures_dir / fname).exists() else "—"
        detail_rows += f"""
        <tr>
            <td>{html.escape(r['method'])}</td>
            <td>{r['level']}</td>
            <td>{r['sample']}</td>
            <td>{m.get('ktc_score', 0):.4f}</td>
            <td>{m.get('dice_resistive', 0):.4f}</td>
            <td>{m.get('dice_conductive', 0):.4f}</td>
            <td>{r.get('composite_score', 0):.4f}</td>
            <td style="color:{color};font-weight:bold">{grade}</td>
            <td>{r.get('runtime_ms', 0):.2f} ms</td>
            <td>{img}</td>
        </tr>"""

    # -- figure gallery ------------------------------------------------------
    gallery_html = ""
    for png in sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []:
        gallery_html += (
            f'<figure style="display:inline-block;margin:8px">'
            f'<img src="figures/{html.escape(png.name)}" style="max-width:300px">'
            f'<figcaption style="text-align:center;font-size:.8em">'
            f'{html.escape(png.stem)}</figcaption></figure>'
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EIT Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #fafafa; color: #1a3a5c; }}
  h1   {{ border-bottom: 3px solid #1D9E75; padding-bottom: 10px; }}
  h2   {{ color: #1a3a5c; margin-top: 40px; }}
  .cards {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
  .card  {{ background: white; border-radius: 8px; padding: 20px; flex: 1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 140px; }}
  .card .value {{ font-size: 2em; font-weight: bold; color: #1D9E75; }}
  .card .label {{ color: #7f8c8d; font-size: .9em; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
           overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 12px; }}
  th {{ background: #1a3a5c; color: white; padding: 12px; text-align: left; font-size: .9em; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #ecf0f1; font-size: .85em; }}
  tr:hover {{ background: #f0fdf8; }}
  .footer {{ text-align: center; color: #95a5a6; margin-top: 40px; font-size: .8em; }}
</style>
</head>
<body>
<h1>EIT Benchmark Report</h1>
<p>KTC 2023 Dataset — Levels 1-7, Samples A/B/C &nbsp;|&nbsp;
   Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="cards">
  <div class="card"><div class="value">{total_runs}</div><div class="label">Total Runs</div></div>
  <div class="card"><div class="value">{len(methods)}</div><div class="label">Methods</div></div>
  <div class="card"><div class="value">{html.escape(best_method)}</div><div class="label">Best Method</div></div>
  <div class="card"><div class="value">{best_score}</div><div class="label">Best Score</div></div>
</div>

<h2>Leaderboard</h2>
<table>
  <tr><th>#</th><th>Method</th><th>Composite</th><th>KTC Score</th>
      <th>Dice Res.</th><th>Dice Cond.</th><th>Avg Runtime</th><th>Runs</th></tr>
  {lb_html}
</table>

<h2>Per-Run Results</h2>
<table>
  <tr><th>Method</th><th>Level</th><th>Sample</th><th>KTC Score</th>
      <th>Dice Res.</th><th>Dice Cond.</th><th>Composite</th>
      <th>Grade</th><th>Runtime</th><th>Overlay</th></tr>
  {detail_rows}
</table>

<h2>Figures</h2>
<div style="background:white;border-radius:8px;padding:16px;box-shadow:0 2px 4px rgba(0,0,0,0.1)">
  {gallery_html if gallery_html else "<p><em>No figures generated yet.</em></p>"}
</div>

<div class="footer">Generated by KTC EIT Benchmarking Framework</div>
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    return report_path
