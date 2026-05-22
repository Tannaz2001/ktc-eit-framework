"""
report_writer.py — Full real-data HTML report.

Reads:
    scores.json                            (headline averages from example_usage.py)
    outputs/scores.json                    (one row per (method, level, sample) from BatchRunner)
    outputs/per_run_metrics.json           (per-sample metrics from example_usage.py)
    outputs/*.png                          (all the charts written by viz.py)
    outputs/level_X/sample_Y/*.png         (per-sample panels written by save_method_panel)
    outputs/error_overlay_*.png            (per-sample error overlays)

Writes:
    reports/report.html                    (one self-contained file — every image
                                            embedded as base64 so it works offline
                                            and survives email/upload)

Sections built:
    1. Header + run summary
    2. Method comparison table (every metric, every method × level × sample)
    3. Charts (degradation curve, leaderboard, confusion matrix, others)
    4. Per-sample panels (GT | predictions for each real sample)
    5. Failure gallery — 3 worst samples per method, each with the panel and
       its error overlay (satisfies "failure case highlighting" from the
       constraint file)

No external template engines, no dummy values. One command rebuilds the
whole report:

    python -c "from report_writer import generate_report; generate_report()"
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps the framework's method class name to the filename stem used by
# example_usage.save_method_panel / plot_error_overlay. Keep these in
# sync with the keys in REAL_METHODS / method_key_map in example_usage.py.
_METHOD_KEY = {
    "MockMethodPlugin":     "mock_baseline",
    "BackProjectionPlugin": "back_projection",
    "BackProjection":       "back_projection_pyeit",
    "GaussNewton":          "gauss_newton",
    "UNetPlugin":           "unet",
}


def _embed_png(path: Path) -> str:
    """Inline a PNG as a base64 data URI so the HTML stays self-contained."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if v is None:
        return "–"
    return html.escape(str(v))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section_summary(runner_rows: list[dict]) -> str:
    if not runner_rows:
        return ("<section><h2>Run summary</h2>"
                "<p><em>outputs/scores.json was not found. Run:</em><br>"
                "<code>python run.py --config configs/training_experiment.yaml</code><br>"
                "<code>python example_usage.py</code></p></section>")

    n_methods = len({r["method"] for r in runner_rows})
    n_samples = len({str(r["sample"]) for r in runner_rows})
    n_levels  = len({r["level"]  for r in runner_rows})
    n_runs    = len(runner_rows)

    return f"""
    <section>
      <h2>Run summary</h2>
      <ul>
        <li><strong>{n_runs}</strong> real reconstructions across
            <strong>{n_methods}</strong> method(s),
            <strong>{n_levels}</strong> difficulty level(s),
            <strong>{n_samples}</strong> sample(s)</li>
        <li>Loader: <code>TrainingDataPlugin</code> / <code>KTCDataPlugin</code></li>
        <li>Runner: <code>src.ktc_framework.runner.experiment_runner.BatchRunner</code></li>
      </ul>
    </section>
    """


def _section_headline_table(scores: dict) -> str:
    """Top-level scores.json — averaged metrics per method."""
    if not scores:
        return ""
    first_val = next(iter(scores.values()), None)
    if not isinstance(first_val, dict):
        # Old flat shape — render as 2-column table
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{_fmt(v)}</td></tr>"
            for k, v in scores.items()
        )
        return ("<section><h2>Headline metrics</h2>"
                f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table></section>")

    all_metrics: list[str] = []
    for m in scores.values():
        for k in m.keys():
            if k not in all_metrics:
                all_metrics.append(k)
    header = "<tr><th>Method (averaged)</th>" + "".join(
        f"<th>{html.escape(m)}</th>" for m in all_metrics
    ) + "</tr>"
    rows = []
    for name, metrics in scores.items():
        cells = "".join(f"<td>{_fmt(metrics.get(m, '–'))}</td>" for m in all_metrics)
        rows.append(f"<tr><td><strong>{html.escape(str(name))}</strong></td>{cells}</tr>")
    return ("<section><h2>Headline metrics</h2>"
            f"<div class='scroll-x'><table>{header}{''.join(rows)}</table></div></section>")


def _section_comparison_table(runner_rows: list[dict]) -> str:
    """Full per-run table — one row per (method, level, sample)."""
    if not runner_rows:
        return ""

    cols = ["method", "level", "sample",
            "ktc_score", "dice_resistive", "dice_conductive",
            "iou_resistive", "iou_conductive",
            "hd95_resistive", "hd95_conductive",
            "composite_score", "grade", "runtime_ms"]

    header = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    rows = []
    for r in runner_rows:
        m = r.get("metrics", {})
        cells = [
            r.get("method"), r.get("level"), r.get("sample"),
            m.get("ktc_score"),
            m.get("dice_resistive"), m.get("dice_conductive"),
            m.get("iou_resistive"), m.get("iou_conductive"),
            m.get("hd95_resistive"), m.get("hd95_conductive"),
            r.get("composite_score"), r.get("grade"), r.get("runtime_ms"),
        ]
        rows.append("<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in cells) + "</tr>")

    return f"""
    <section>
      <h2>Method comparison table</h2>
      <p class="muted">Every real reconstruction, with every metric the framework computed.</p>
      <div class="scroll-x"><table>{header}{''.join(rows)}</table></div>
    </section>
    """


def _section_charts(outputs_dir: Path) -> str:
    chart_files = [
        ("Performance degradation",          "degradation_curve.png"),
        ("Method leaderboard",               "leaderboard.png"),
        ("Confusion matrix (pooled)",        "confusion_matrix.png"),
        ("Noise sensitivity",                "noise_sensitivity.png"),
        ("Electrode layout",                 "electrodes.png"),
    ]
    figs = []
    for caption, fname in chart_files:
        uri = _embed_png(outputs_dir / fname)
        if uri:
            figs.append(
                f'<figure><img src="{uri}" alt="{html.escape(caption)}">'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
    if not figs:
        return ""
    return ("<section><h2>Charts</h2>"
            "<div class='gallery'>" + "".join(figs) + "</div></section>")


def _section_per_sample_panels(runner_rows: list[dict], outputs_dir: Path) -> str:
    """One row per (level, sample) showing every method's panel side by side."""
    if not runner_rows:
        return ""

    grid: dict[tuple, dict[str, Path]] = {}
    for r in runner_rows:
        key = (r["level"], str(r["sample"]))
        mkey = _METHOD_KEY.get(r["method"], r["method"].lower())
        panel = outputs_dir / f"level_{r['level']}" / f"sample_{r['sample']}" / f"{mkey}.png"
        if panel.exists():
            grid.setdefault(key, {})[r["method"]] = panel

    if not grid:
        return ""

    blocks = []
    for (lv, sid), method_panels in sorted(grid.items()):
        figs = []
        for method_name, png_path in method_panels.items():
            uri = _embed_png(png_path)
            figs.append(
                f'<figure><img src="{uri}" alt="{html.escape(method_name)}">'
                f"<figcaption>{html.escape(method_name)}</figcaption></figure>"
            )
        blocks.append(
            f"<h3>Level {lv} · sample {html.escape(sid)}</h3>"
            f"<div class='gallery'>{''.join(figs)}</div>"
        )

    return ("<section><h2>Per-sample panels</h2>"
            "<p class='muted'>Each panel: ground truth | prediction | error overlay.</p>"
            + "".join(blocks) + "</section>")


def _section_failure_gallery(runner_rows: list[dict], outputs_dir: Path,
                             k: int = 3) -> str:
    """For each method, show the k worst (lowest composite_score) samples.

    Each card shows the per-sample panel (GT | pred | error) plus the standalone
    error overlay. This is the "failure case highlighting" called out in the
    constraint file.
    """
    if not runner_rows:
        return ""

    by_method: dict[str, list[dict]] = {}
    for r in runner_rows:
        by_method.setdefault(r["method"], []).append(r)

    blocks = []
    for method, rows in by_method.items():
        worst = sorted(rows, key=lambda r: r.get("composite_score", 0.0))[:k]
        mkey = _METHOD_KEY.get(method, method.lower())

        cards = []
        for r in worst:
            lv  = r["level"]
            sid = str(r["sample"])
            comp = r.get("composite_score", 0.0)
            grade = r.get("grade", "?")
            ktc = r.get("metrics", {}).get("ktc_score", "–")

            panel   = outputs_dir / f"level_{lv}" / f"sample_{sid}" / f"{mkey}.png"
            overlay = outputs_dir / f"error_overlay_{mkey}_sample_{sid}.png"

            img_html = ""
            for tag, p in (("Panel (GT | pred | error)", panel),
                           ("Standalone error overlay", overlay)):
                uri = _embed_png(p)
                if uri:
                    img_html += (
                        f"<figure><img src='{uri}' alt='{html.escape(tag)}'>"
                        f"<figcaption>{html.escape(tag)}</figcaption></figure>"
                    )

            cards.append(
                f"<article class='failure'>"
                f"<h4>level {lv} · sample {html.escape(sid)} "
                f"<span class='badge grade-{html.escape(grade)}'>{html.escape(grade)}</span>"
                f"<span class='score'>composite = {_fmt(comp)} · KTC = {_fmt(ktc)}</span></h4>"
                f"<div class='gallery'>{img_html}</div>"
                f"</article>"
            )

        blocks.append(f"<h3>{html.escape(method)}</h3>" + "".join(cards))

    return (f"<section><h2>Failure gallery — worst {k} samples per method</h2>"
            "<p class='muted'>Sorted by composite score, ascending. Each entry shows the "
            "ground truth, the method's prediction, and an error overlay highlighting "
            "missed (red) and false (orange) inclusions.</p>"
            + "".join(blocks) + "</section>")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 2rem auto; max-width: 1100px; color: #1a3a5c; background: #fafafa; padding: 0 1rem; }
h1 { border-bottom: 3px solid #1D9E75; padding-bottom: .4rem; }
h2 { color: #1a3a5c; margin-top: 2.5rem; border-bottom: 1px solid #d0d8e0; padding-bottom: .3rem; }
h3 { color: #1a3a5c; margin-top: 1.5rem; }
h4 { color: #1a3a5c; margin: .5rem 0; }
.meta  { color: #666; font-size: .9rem; margin-bottom: 1.5rem; }
.muted { color: #666; font-size: .9rem; margin: .25rem 0 .75rem; }
.scroll-x { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: white; font-size: .9rem; }
th, td { padding: .45rem .7rem; border: 1px solid #e0e0e0; text-align: left; white-space: nowrap; }
th { background: #1a3a5c; color: white; }
tr:nth-child(even) td { background: #f4f7fa; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
           gap: 1rem; margin: 1rem 0; }
.gallery figure { margin: 0; background: white; padding: .5rem;
                  border: 1px solid #e0e0e0; border-radius: 4px; }
.gallery img { width: 100%; height: auto; display: block; }
.gallery figcaption { font-size: .85rem; text-align: center;
                      padding-top: .3rem; color: #555; }
.failure { background: white; border-left: 4px solid #D85A30; padding: .75rem 1rem;
           margin: 1rem 0; border-radius: 4px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: .75rem; color: white; vertical-align: middle; margin-left: .4rem; }
.grade-A { background: #1D9E75; }
.grade-B { background: #4A90E2; }
.grade-C { background: #F5A623; }
.grade-D { background: #D85A30; }
.score { color: #666; font-size: .85rem; margin-left: .8rem; }
code { background: #eef; padding: 1px 5px; border-radius: 3px; }
nav { background: white; padding: .8rem 1rem; border-radius: 4px;
      border: 1px solid #d0d8e0; margin-bottom: 2rem; }
nav a { color: #1a3a5c; text-decoration: none; margin-right: 1rem; font-weight: 600; }
nav a:hover { color: #1D9E75; }
"""


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_report(
    scores_path: str = "scores.json",
    out_path: str = "reports/report.html",
    outputs_dir: str = "outputs",
) -> str:
    """Build the full real-data HTML report.

    One command, one self-contained file. All images embedded as base64 so
    the report works offline and survives being emailed or zipped.
    """
    out_p     = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    outputs_p = Path(outputs_dir)

    headline    = _read_json(Path(scores_path)) or {}
    runner_rows = _read_json(outputs_p / "scores.json") or []

    nav = ("<nav>"
           "<a href='#summary'>Summary</a>"
           "<a href='#headline'>Headline</a>"
           "<a href='#table'>Comparison table</a>"
           "<a href='#charts'>Charts</a>"
           "<a href='#panels'>Per-sample panels</a>"
           "<a href='#failures'>Failure gallery</a>"
           "</nav>")

    sections = [
        f"<a id='summary'></a>{_section_summary(runner_rows)}",
        f"<a id='headline'></a>{_section_headline_table(headline)}",
        f"<a id='table'></a>{_section_comparison_table(runner_rows)}",
        f"<a id='charts'></a>{_section_charts(outputs_p)}",
        f"<a id='panels'></a>{_section_per_sample_panels(runner_rows, outputs_p)}",
        f"<a id='failures'></a>{_section_failure_gallery(runner_rows, outputs_p)}",
    ]

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EIT Reconstruction Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>EIT Reconstruction Report</h1>
<div class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 · {len(runner_rows)} real reconstructions · self-contained (images embedded)</div>
{nav}
{''.join(sections)}
</body>
</html>
"""
    out_p.write_text(page, encoding="utf-8")
    print(f"Saved report to {out_p}")
    return str(out_p)


if __name__ == "__main__":
    generate_report()