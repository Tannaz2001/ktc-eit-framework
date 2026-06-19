"""HTML report generator — fully self-contained, no external dependencies."""

from __future__ import annotations

import base64
import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_png(path: str | Path) -> str:
    """Return a base64 PNG data URI, or empty string if the file is missing."""
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _runtime_fmt(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


_GRADE_COLOR = {"A": "#1D9E75", "B": "#4A90E2", "C": "#F5A623", "D": "#D85A30"}


def _grade(ktc: float) -> str:
    if ktc >= 0.60:
        return "A"
    if ktc >= 0.30:
        return "B"
    if ktc >= 0.10:
        return "C"
    return "D"


def _avg(lst: list[float]) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def _get_metric(r: dict, key: str) -> float:
    return float(r.get("metrics", {}).get(key, r.get(key, 0.0)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html_report(
    scores_json_path,   # str | Path | list[dict]  (list = old runner calling convention)
    figures_dir,        # str | Path
    output_path=None,   # str | Path | dict | None (dict = old qualitative_data arg)
) -> str:
    """Generate a fully self-contained HTML report. Returns the output file path.

    New calling convention (standalone):
        generate_html_report(scores_json_path, figures_dir, output_path)

    Old calling convention (from experiment_runner — kept for compatibility):
        generate_html_report(results_list, output_dir, qualitative_data)

    All PNG figures are embedded as base64 data URIs so the report is
    completely self-contained with no external image references.
    """
    qualitative_data: dict | None = None

    # ── detect calling convention ───────────────────────────────────────────
    if isinstance(scores_json_path, list):
        # Old: (results, output_dir, qualitative_data)
        results: list[dict] = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in scores_json_path
        ]
        _output_dir   = Path(figures_dir)
        _figures_dir  = _output_dir / "figures"
        _output_file  = _output_dir / "report.html"
        if isinstance(output_path, dict):
            qualitative_data = output_path
    else:
        # New: (scores_json_path, figures_dir, output_path)
        _sp           = Path(scores_json_path)
        _figures_dir  = Path(figures_dir)
        _output_file  = (
            Path(output_path) if output_path
            else _figures_dir.parent / "report.html"
        )
        results = []
        if _sp.exists():
            with _sp.open(encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                results = raw

    # ── summary stats ───────────────────────────────────────────────────────
    total_runs = len(results)
    methods    = sorted({r["method"] for r in results})

    # Best by mean KTC per method
    method_ktc: dict[str, list[float]] = {}
    for r in results:
        method_ktc.setdefault(r["method"], []).append(_get_metric(r, "ktc_score"))
    best_method = max(method_ktc, key=lambda m: _avg(method_ktc[m])) if method_ktc else "N/A"
    best_method_score = round(_avg(method_ktc[best_method]), 4) if best_method != "N/A" else 0.0

    # Single highest KTC run
    best_run = max(results, key=lambda r: _get_metric(r, "ktc_score"), default=None)
    best_ktc_val = round(_get_metric(best_run, "ktc_score"), 4) if best_run else 0.0
    best_ktc_lbl = (
        f"{best_ktc_val} (L{best_run['level']} {best_run['sample']})"
        if best_run else "N/A"
    )

    # ── leaderboard ─────────────────────────────────────────────────────────
    lb_buckets: dict[str, dict[str, list]] = {}
    for r in results:
        m = r["method"]
        if m not in lb_buckets:
            lb_buckets[m] = {k: [] for k in
                             ["ktc", "dice_r", "dice_c", "iou_r", "iou_c", "rt"]}
        b = lb_buckets[m]
        b["ktc"].append(_get_metric(r, "ktc_score"))
        b["dice_r"].append(_get_metric(r, "dice_resistive"))
        b["dice_c"].append(_get_metric(r, "dice_conductive"))
        b["iou_r"].append(_get_metric(r, "iou_resistive"))
        b["iou_c"].append(_get_metric(r, "iou_conductive"))
        b["rt"].append(float(r.get("runtime_ms", 0.0)))

    lb_rows = sorted(
        [{"method": m,
          "ktc":    _avg(v["ktc"]),
          "dice_r": _avg(v["dice_r"]),
          "dice_c": _avg(v["dice_c"]),
          "iou_r":  _avg(v["iou_r"]),
          "iou_c":  _avg(v["iou_c"]),
          "rt":     _avg(v["rt"]),
          "grade":  _grade(_avg(v["ktc"]))}
         for m, v in lb_buckets.items()],
        key=lambda x: x["ktc"], reverse=True,
    )

    lb_html = ""
    for i, row in enumerate(lb_rows, 1):
        gc    = _GRADE_COLOR.get(row["grade"], "#888")
        hl    = ' style="background:#f0fdf8;font-weight:600"' if row["method"] == best_method else ""
        lb_html += f"""
        <tr{hl}>
          <td>{i}</td>
          <td>{_html.escape(row['method'])}</td>
          <td>{row['ktc']:.3f}</td>
          <td>{row['dice_r']:.3f}</td>
          <td>{row['dice_c']:.3f}</td>
          <td>{row['iou_r']:.3f}</td>
          <td>{row['iou_c']:.3f}</td>
          <td>{_runtime_fmt(row['rt'])}</td>
          <td style="background:{gc};color:white;font-weight:bold;text-align:center">{row['grade']}</td>
        </tr>"""

    # ── per-run results ──────────────────────────────────────────────────────
    detail_rows = ""
    for r in results:
        ktc   = _get_metric(r, "ktc_score")
        dr    = _get_metric(r, "dice_resistive")
        dc    = _get_metric(r, "dice_conductive")
        ir    = _get_metric(r, "iou_resistive")
        ic    = _get_metric(r, "iou_conductive")
        rt    = float(r.get("runtime_ms", 0.0))
        grade = r.get("grade") or _grade(ktc)
        gc    = _GRADE_COLOR.get(grade, "#888")
        stem  = f"{r['method']}_level{r['level']}_sample{r['sample']}"
        fig_btn = (
            f'<button class="fig-btn no-print" onclick="openFig(\'{_html.escape(stem)}\')">'
            f'&#128247;</button>'
            if (_figures_dir / f"{stem}.png").exists() else "—"
        )
        detail_rows += f"""
        <tr>
          <td>{_html.escape(r['method'])}</td>
          <td>{r['level']}</td>
          <td>{r['sample']}</td>
          <td>{ktc:.3f}</td>
          <td>{dr:.3f}</td>
          <td>{dc:.3f}</td>
          <td>{ir:.3f}</td>
          <td>{ic:.3f}</td>
          <td>{_runtime_fmt(rt)}</td>
          <td style="background:{gc};color:white;font-weight:bold;text-align:center">{grade}</td>
          <td>{fig_btn}</td>
        </tr>"""

    # ── qualitative detection summary (hull analysis) ────────────────────────
    qual_html = ""
    if qualitative_data:
        for method, qd in sorted(qualitative_data.items()):
            rp = qd.get("resistive_detected_pct", 0.0)
            cp = qd.get("conductive_detected_pct", 0.0)

            def _pct_color(p: float) -> str:
                return "#1D9E75" if p >= 90 else "#4A90E2" if p >= 70 else "#F5A623" if p >= 50 else "#D85A30"

            qual_html += f"""
        <tr>
          <td><strong>{_html.escape(method)}</strong></td>
          <td style="color:{_pct_color(rp)};font-weight:bold">{rp:.1f}%</td>
          <td style="color:{_pct_color(cp)};font-weight:bold">{cp:.1f}%</td>
          <td>{qd.get('false_positive_count', 0)}</td>
          <td>{qd.get('avg_resistive_hull_iou', 0.0):.3f}</td>
          <td>{qd.get('avg_conductive_hull_iou', 0.0):.3f}</td>
        </tr>"""

    # ── JS figure dict (embed each PNG once) ────────────────────────────────
    fig_js_entries: list[str] = []
    gallery_html = ""
    for png in sorted(_figures_dir.glob("*.png")):
        uri = _embed_png(png)
        if not uri:
            continue
        safe_stem = _html.escape(png.stem)
        # One JSON entry per figure
        fig_js_entries.append(f'  "{safe_stem}": "{uri}"')
        # Gallery thumbnail
        gallery_html += (
            f'<figure class="thumb no-print" onclick="openFig(\'{safe_stem}\')">'
            f'<img data-fig="{safe_stem}" alt="{safe_stem}">'
            f'<figcaption>{safe_stem}</figcaption></figure>'
        )

    fig_js = "const FIGS = {\n" + ",\n".join(fig_js_entries) + "\n};"

    # ── aggregate chart embeds ───────────────────────────────────────────────
    def _chart_section(title: str, fname: str) -> str:
        uri = _embed_png(_figures_dir / fname)
        if not uri:
            return ""
        safe = _html.escape(fname.replace(".png", ""))
        return (
            f'<section><h2>{title}</h2>'
            f'<div class="chart-wrap">'
            f'<img src="{uri}" class="chart-img" alt="{safe}">'
            f'</div></section>'
        )

    degradation_section  = _chart_section("KTC Score vs Difficulty Level",
                                           "degradation_curve.png")
    heatmap_section      = _chart_section("Metrics Summary Matrix",
                                           "metrics_heatmap.png")
    runtime_section      = _chart_section("Runtime Comparison",
                                           "runtime_comparison.png")
    failure_section      = _chart_section("Failure Gallery — Worst Samples",
                                           "failure_gallery.png")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── assemble page ────────────────────────────────────────────────────────
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KTC EIT Benchmarking Report</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  max-width: 1300px; margin: 0 auto; padding: 24px 20px;
  background: #f4f6f8; color: #1a3a5c;
}}
header {{ border-bottom: 3px solid #1D9E75; padding-bottom: 14px; margin-bottom: 24px; }}
header h1 {{ margin: 0 0 4px; font-size: 1.9em; }}
header .subtitle {{ color: #4A90E2; font-size: 0.95em; margin: 0; }}
header .ts {{ color: #7f8c8d; font-size: 0.82em; }}
h2 {{ color: #1a3a5c; margin: 36px 0 12px; font-size: 1.25em; border-left: 4px solid #1D9E75; padding-left: 10px; }}
section {{ margin-bottom: 32px; }}
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.card {{
  background: white; border-radius: 10px; padding: 20px 24px;
  flex: 1 1 180px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  text-align: center;
}}
.card .val {{ font-size: 1.7em; font-weight: 700; color: #1D9E75; word-break: break-word; }}
.card .lbl {{ color: #7f8c8d; font-size: 0.83em; margin-top: 6px; }}
table {{
  width: 100%; border-collapse: collapse;
  background: white; border-radius: 10px; overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08); font-size: 0.84em;
}}
th {{
  background: #1a3a5c; color: white; padding: 11px 10px;
  text-align: left; cursor: pointer; user-select: none; white-space: nowrap;
}}
th:hover {{ background: #243e5e; }}
td {{ padding: 9px 10px; border-bottom: 1px solid #ecf0f1; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: #f0fdf8; }}
.chart-wrap {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.chart-img {{ max-width: 100%; height: auto; border-radius: 6px; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 12px; background: white; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
figure.thumb {{ margin: 0; cursor: pointer; border-radius: 6px; overflow: hidden; border: 2px solid transparent; transition: border-color .2s; }}
figure.thumb:hover {{ border-color: #1D9E75; }}
figure.thumb img {{ display: block; width: 200px; height: 140px; object-fit: cover; }}
figure.thumb figcaption {{ font-size: 0.72em; text-align: center; padding: 4px; background: #f4f6f8; color: #555; max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.fig-btn {{ background: #1D9E75; color: white; border: none; border-radius: 4px; padding: 3px 8px; cursor: pointer; font-size: 0.85em; }}
.fig-btn:hover {{ background: #17866a; }}
#lightbox {{
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.82); z-index: 9999;
  justify-content: center; align-items: center;
  flex-direction: column; cursor: zoom-out;
}}
#lightbox img {{ max-width: 92vw; max-height: 82vh; border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.6); }}
#lightbox .lb-cap {{ color: #e0e0e0; margin-top: 12px; font-size: 0.88em; }}
footer {{ text-align: center; color: #95a5a6; margin-top: 48px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.8em; line-height: 1.8; }}
@media (max-width: 768px) {{
  .cards {{ flex-direction: column; }}
  table {{ font-size: 0.75em; }}
  th, td {{ padding: 6px 8px; }}
  figure.thumb img {{ width: 140px; height: 100px; }}
}}
@media print {{
  .no-print, #lightbox {{ display: none !important; }}
  .card {{ box-shadow: none; border: 1px solid #ddd; }}
  table {{ box-shadow: none; }}
  body {{ background: white; }}
}}
</style>
</head>
<body>

<header>
  <h1>KTC EIT Benchmarking Report</h1>
  <p class="subtitle">Generated by KTC Framework &mdash; Sprint 7</p>
  <p class="ts">Generated {now}</p>
</header>

<section>
  <div class="cards">
    <div class="card"><div class="val">{total_runs}</div><div class="lbl">Total Runs</div></div>
    <div class="card"><div class="val">{len(methods)}</div><div class="lbl">Methods Tested</div></div>
    <div class="card"><div class="val">{_html.escape(best_method)}</div><div class="lbl">Best Method</div></div>
    <div class="card"><div class="val">{best_ktc_lbl}</div><div class="lbl">Best KTC Score</div></div>
  </div>
</section>

<section>
  <h2>Leaderboard</h2>
  <table id="lb-table">
    <thead>
      <tr>
        <th onclick="sortTable('lb-table',0,true)">Rank</th>
        <th onclick="sortTable('lb-table',1,false)">Method</th>
        <th onclick="sortTable('lb-table',2,true)">Mean KTC Score</th>
        <th onclick="sortTable('lb-table',3,true)">Dice (R)</th>
        <th onclick="sortTable('lb-table',4,true)">Dice (C)</th>
        <th onclick="sortTable('lb-table',5,true)">IoU (R)</th>
        <th onclick="sortTable('lb-table',6,true)">IoU (C)</th>
        <th onclick="sortTable('lb-table',7,false)">Mean Runtime</th>
        <th>Grade</th>
      </tr>
    </thead>
    <tbody>{lb_html}</tbody>
  </table>
</section>

{degradation_section}

{heatmap_section}

{runtime_section}

<section>
  <h2>Per-Run Results</h2>
  <table id="run-table">
    <thead>
      <tr>
        <th onclick="sortTable('run-table',0,false)">Method</th>
        <th onclick="sortTable('run-table',1,true)">Level</th>
        <th onclick="sortTable('run-table',2,false)">Sample</th>
        <th onclick="sortTable('run-table',3,true)">KTC Score</th>
        <th onclick="sortTable('run-table',4,true)">Dice R</th>
        <th onclick="sortTable('run-table',5,true)">Dice C</th>
        <th onclick="sortTable('run-table',6,true)">IoU R</th>
        <th onclick="sortTable('run-table',7,true)">IoU C</th>
        <th onclick="sortTable('run-table',8,false)">Runtime</th>
        <th>Grade</th>
        <th class="no-print">Fig</th>
      </tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>
</section>

{failure_section}

{"" if not qualitative_data else f"""
<section>
  <h2>Hull Detection Summary</h2>
  <table>
    <thead>
      <tr><th>Method</th><th>Resistive Detected</th><th>Conductive Detected</th>
          <th>False Positives</th><th>Avg Hull IoU (R)</th><th>Avg Hull IoU (C)</th></tr>
    </thead>
    <tbody>{qual_html}</tbody>
  </table>
</section>
"""}

<section class="no-print">
  <h2>Figures Gallery</h2>
  <div class="gallery">
    {gallery_html if gallery_html else "<p><em>No figures generated yet.</em></p>"}
  </div>
</section>

<div id="lightbox" onclick="closeLightbox()">
  <img id="lb-img" src="" alt="">
  <p class="lb-cap" id="lb-cap"></p>
</div>

<footer>
  <p>Generated by KTC EIT Framework</p>
  <p>KTC 2023 Dataset &mdash; <a href="https://zenodo.org/record/10986692" style="color:#4A90E2">Zenodo record 10986692</a>, CC-BY 4.0</p>
  <p>Framework repository: <em>[placeholder]</em></p>
</footer>

<script>
{fig_js}

// Populate gallery thumbnails from FIGS dict
document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('[data-fig]').forEach(function(el) {{
    var src = FIGS[el.dataset.fig];
    if (src) el.src = src;
  }});
}});

// Lightbox
function openFig(name) {{
  var src = FIGS[name];
  if (!src) return;
  document.getElementById('lb-img').src = src;
  document.getElementById('lb-cap').textContent = name;
  document.getElementById('lightbox').style.display = 'flex';
}}
function closeLightbox() {{
  document.getElementById('lightbox').style.display = 'none';
  document.getElementById('lb-img').src = '';
}}
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeLightbox(); }});

// Table sort
function sortTable(tableId, colIdx, numeric) {{
  var table  = document.getElementById(tableId);
  var tbody  = table.querySelector('tbody');
  var rows   = Array.from(tbody.rows);
  var prevCol = table.dataset.sortCol;
  var prevDir = table.dataset.sortDir;
  var asc = (prevCol === String(colIdx) && prevDir === 'asc') ? false : true;
  rows.sort(function(a, b) {{
    var va = a.cells[colIdx] ? a.cells[colIdx].innerText.trim() : '';
    var vb = b.cells[colIdx] ? b.cells[colIdx].innerText.trim() : '';
    if (numeric) {{
      var na = parseFloat(va) || 0;
      var nb = parseFloat(vb) || 0;
      return asc ? na - nb : nb - na;
    }}
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
  table.dataset.sortCol = colIdx;
  table.dataset.sortDir = asc ? 'asc' : 'desc';
}}
</script>

</body>
</html>"""

    _output_file.parent.mkdir(parents=True, exist_ok=True)
    _output_file.write_text(page, encoding="utf-8")
    return str(_output_file)
