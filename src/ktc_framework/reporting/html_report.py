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
    qualitative_data: dict[str, dict] | None = None,
) -> Path:
    """Generate outputs/report.html from experiment results.

    Parameters
    ----------
    results : list[dict]
        Per-run results with metrics
    output_dir : Path
        Directory to save report.html
    qualitative_data : dict, optional
        Qualitative metrics per method from _qualitative_summary
    """
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
            leaderboard[m] = {"ktc": [], "runtime": [], "composite": []}
        metrics = r.get("metrics", {})
        leaderboard[m]["ktc"].append(metrics.get("ktc_score", 0))
        leaderboard[m]["runtime"].append(r.get("runtime_ms", 0))
        leaderboard[m]["composite"].append(r.get("composite_score", 0))

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    leaderboard_rows = sorted(
        [{"method": m,
          "avg_composite": avg(v["composite"]),
          "avg_ktc":       avg(v["ktc"]),
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
            <td>{row['avg_runtime']} ms</td>
            <td>{row['runs']}</td>
        </tr>"""

    # -- qualitative detection summary -------------------------------------------
    qual_html = ""
    qual_summaries = ""
    if qualitative_data:
        for method, qual_data in sorted(qualitative_data.items()):
            res_pct = qual_data.get("resistive_detected_pct", 0.0)
            con_pct = qual_data.get("conductive_detected_pct", 0.0)
            res_iou = qual_data.get("avg_resistive_hull_iou", 0.0)
            con_iou = qual_data.get("avg_conductive_hull_iou", 0.0)
            fp_count = qual_data.get("false_positive_count", 0)

            # Color coding: green ≥90%, blue ≥70%, amber ≥50%, red <50%
            def pct_color(pct):
                if pct >= 90:
                    return "#1D9E75"  # green
                elif pct >= 70:
                    return "#4A90E2"  # blue
                elif pct >= 50:
                    return "#F5A623"  # amber
                else:
                    return "#D85A30"  # red

            res_color = pct_color(res_pct)
            con_color = pct_color(con_pct)

            qual_html += f"""
        <tr>
            <td><strong>{html.escape(method)}</strong></td>
            <td style="color:{res_color};font-weight:bold">{res_pct:.1f}%</td>
            <td style="color:{con_color};font-weight:bold">{con_pct:.1f}%</td>
            <td>{fp_count}</td>
            <td>{res_iou:.3f}</td>
            <td>{con_iou:.3f}</td>
        </tr>"""

            # Natural language summary
            res_str = qual_data.get("resistive_detected_str", "0/0")
            con_str = qual_data.get("conductive_detected_str", "0/0")
            qual_summaries += f"""
    <p><strong>{html.escape(method)}</strong>: Resistive regions detected in {res_str} ({res_pct:.1f}%),
       conductive regions in {con_str} ({con_pct:.1f}%).
       Avg hull IoU: {res_iou:.3f} (R), {con_iou:.3f} (C).
       False positives: {fp_count}.</p>"""

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
      <th>Avg Runtime</th><th>Runs</th></tr>
  {lb_html}
</table>

<h2>Qualitative Detection Summary</h2>
<table>
  <tr><th>Method</th><th>Resistive Detected</th><th>Conductive Detected</th>
      <th>False Positives</th><th>Avg Hull IoU (R)</th><th>Avg Hull IoU (C)</th></tr>
  {qual_html if qual_html else "<tr><td colspan='6' style='text-align:center;color:#999'>No qualitative data available</td></tr>"}
</table>

<h3>Detection Summaries</h3>
<div style="background:white;border-radius:8px;padding:16px;box-shadow:0 2px 4px rgba(0,0,0,0.1);line-height:1.6">
  {qual_summaries if qual_summaries else "<p><em>No qualitative summaries available.</em></p>"}
</div>

<h2>Per-Run Results</h2>
<table>
  <tr><th>Method</th><th>Level</th><th>Sample</th><th>KTC Score</th>
      <th>Composite</th><th>Grade</th><th>Runtime</th><th>Overlay</th></tr>
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
