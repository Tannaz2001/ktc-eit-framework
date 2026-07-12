"""Focused self-contained HTML report generator.

The report uses active-run PNGs for reconstruction/hull figures and renders
dashboard-style SVG charts from the same filtered data used by Streamlit.
It intentionally avoids dumping all failure/gallery images into the report.
"""

from __future__ import annotations

import base64
import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ktc_framework.reporting.constants import (
    METRIC_SHORT_LABELS,
    METRIC_SPECS,
    get_method_color,
    letter_grade as _letter_grade,
)

# Short-label metric list used for dense tables in this report.
METRICS = [(METRIC_SHORT_LABELS[key], key) for _, key in METRIC_SPECS]
METRIC_BY_KEY = {key: (label, key) for label, key in METRICS}

PALETTE = [
    "#2da44e", "#8250df", "#0969da", "#bf8700", "#cf222e",
    "#1a7f37", "#d4a72c", "#0550ae", "#9a3ece", "#068a39",
]

def _embed_png(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _esc(value: Any) -> str:
    return _html.escape(str(value))


def _metric(row: dict, key: str) -> float:
    return float(row.get("metrics", {}).get(key, row.get(key, 0.0)) or 0.0)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _grade(ktc: float) -> str:
    return _letter_grade(ktc)


def _runtime_fmt(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _load_results(scores_json_path, figures_dir, output_path):
    qualitative_data = output_path if isinstance(output_path, dict) else {}
    if isinstance(scores_json_path, list):
        results = [{k: v for k, v in row.items() if not k.startswith("_")}
                   for row in scores_json_path]
        output_dir = Path(figures_dir)
        return results, output_dir / "figures", output_dir / "report.html", qualitative_data

    scores_path = Path(scores_json_path)
    figs_dir = Path(figures_dir)
    report_path = Path(output_path) if output_path else figs_dir.parent / "report.html"
    results = []
    if scores_path.exists():
        raw = json.loads(scores_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            results = raw
    return results, figs_dir, report_path, qualitative_data


def _metric_defs(options: dict) -> list[tuple[str, str]]:
    selected = options.get("_metric_keys") if isinstance(options, dict) else None
    if not selected:
        return METRICS
    defs = [METRIC_BY_KEY[key] for key in selected if key in METRIC_BY_KEY]
    return defs or METRICS


def _method_summary(results: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in results:
        buckets.setdefault(str(row["method"]), []).append(row)

    rows = []
    for method, method_rows in buckets.items():
        item = {
            "method": method,
            "runs": len(method_rows),
            "runtime": _avg([float(r.get("runtime_ms", 0.0) or 0.0) for r in method_rows]),
        }
        for _, key in METRICS:
            item[key] = _avg([_metric(r, key) for r in method_rows])
        item["grade"] = _grade(item["ktc_score"])
        rows.append(item)
    return sorted(rows, key=lambda r: r["ktc_score"], reverse=True)


def _table(headers: list[str], rows: list[list[Any]], table_id: str = "") -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    ident = f' id="{_esc(table_id)}"' if table_id else ""
    return f"<table{ident}><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _chart_img(figures_dir: Path, filename: str, title: str) -> str:
    uri = _embed_png(figures_dir / filename)
    if not uri:
        return ""
    return (
        f'<section><h2>{_esc(title)}</h2>'
        f'<div class="figure"><img src="{uri}" alt="{_esc(title)}"></div></section>'
    )


def _chart_img_any(figures_dir: Path, filenames: list[str], title: str) -> str:
    for filename in filenames:
        chart = _chart_img(figures_dir, filename, title)
        if chart:
            return chart
    return ""


def _method_colors(methods: list[str]) -> dict[str, str]:
    return {method: get_method_color(method) for method in methods}


def _leaderboard_svg(summary: list[dict]) -> str:
    if not summary:
        return ""
    colors = _method_colors([str(row["method"]) for row in summary])
    width, height = 920, 310
    left, right, top, bottom = 48, 18, 20, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 12
    bar_w = max(22, (plot_w - bar_gap * (len(summary) - 1)) / len(summary))

    # Axis domain must include any negative scores (a method can do *worse*
    # than an empty-tank guess, giving a negative KTC). Clamping those to 0
    # -- as the old code did -- made failing methods look like they scored
    # zero, disagreeing with the dashboard which draws them below the axis.
    #
    # We also auto-scale the top of the axis to the data instead of pinning
    # it at 100: KTC scores in practice sit well below 100, so a fixed 0-100
    # axis squashed every bar into the bottom third and looked lopsided. A
    # data-fitted axis (with headroom) fills the chart and makes the ranking
    # and the small/negative bars legible.
    import math
    raw = [float(row.get("ktc_score", 0.0)) * 100.0 for row in summary]
    data_hi = max(raw + [0.0])
    data_lo = min(raw + [0.0])
    spread = (data_hi - data_lo) or 1.0
    axis_hi = data_hi + spread * 0.18
    axis_lo = min(0.0, data_lo - (spread * 0.10 if data_lo < 0 else 0.0))

    def _nice_step(rng: float) -> int:
        for s in (5, 10, 20, 25, 50, 100):
            if rng / s <= 6:
                return s
        return 100

    step = _nice_step(axis_hi - axis_lo)
    axis_hi = math.ceil(axis_hi / step) * step
    axis_lo = math.floor(axis_lo / step) * step
    span = (axis_hi - axis_lo) or 1.0

    def y_of(v: float) -> float:
        return top + plot_h - ((v - axis_lo) / span) * plot_h

    zero_y = y_of(0.0)

    # Ticks on the nice step across the (possibly negative) domain, plus 0.
    ticks = sorted({0} | {
        int(axis_lo + i * step) for i in range(int(round(span / step)) + 1)
    })
    grid = ""
    labels_y = ""
    for tick in ticks:
        y = y_of(tick)
        cls = "axis" if tick == 0 else "chart-grid"
        grid += f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="{cls}"/>'
        labels_y += f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis-label">{tick}</text>'

    bars = ""
    labels_x = ""
    for idx, row in enumerate(summary):
        method = str(row["method"])
        score = float(row.get("ktc_score", 0.0)) * 100.0
        x = left + idx * (bar_w + bar_gap)
        yv = y_of(score)
        # Positive bars grow up from the zero line; negative bars hang below it.
        bar_y = min(yv, zero_y)
        bar_h = abs(yv - zero_y)
        color = colors[method]
        label = method if len(method) <= 18 else method[:16] + ".."
        # Value label sits above positive bars, below negative ones.
        val_y = (bar_y - 7) if score >= 0 else (bar_y + bar_h + 14)
        bars += (
            f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'rx="4" fill="{color}"/>'
            f'<text x="{x + bar_w/2:.1f}" y="{max(12, val_y):.1f}" text-anchor="middle" class="bar-label">'
            f'{score:.1f} ({_esc(row["grade"])})</text>'
        )
        labels_x += (
            f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 22:.1f}" text-anchor="middle" '
            f'class="axis-label">{_esc(label)}</text>'
        )

    return (
        '<section><h2>Leaderboard Chart</h2><div class="figure">'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Leaderboard Chart">'
        f'{grid}{labels_y}<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>'
        f'{bars}{labels_x}<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" class="axis-label">'
        'Method Rankings - KTC Score</text></svg></div></section>'
    )


def _degradation_svg(results: list[dict]) -> str:
    methods = sorted({str(r["method"]) for r in results})
    levels = sorted({int(r.get("level", 1)) for r in results})
    if not methods or not levels:
        return ""
    colors = _method_colors(methods)
    width, height = 920, 330
    left, right, top, bottom = 52, 150, 24, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_level, max_level = min(levels), max(levels)
    level_span = max(1, max_level - min_level)

    grid = ""
    labels_y = ""
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - tick * plot_h
        grid += f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="chart-grid"/>'
        labels_y += f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis-label">{tick:.2f}</text>'

    labels_x = ""
    for level in levels:
        x = left + ((level - min_level) / level_span) * plot_w
        labels_x += f'<text x="{x:.1f}" y="{top+plot_h+22:.1f}" text-anchor="middle" class="axis-label">L{level}</text>'

    lines = ""
    legend = ""
    for method in methods:
        points = []
        method_rows = [r for r in results if str(r["method"]) == method]
        for level in levels:
            vals = [_metric(r, "ktc_score") for r in method_rows if int(r.get("level", 1)) == level]
            if not vals:
                continue
            x = left + ((level - min_level) / level_span) * plot_w
            y = top + plot_h - max(0.0, min(1.0, _avg(vals))) * plot_h
            points.append((x, y, _avg(vals)))
        if len(points) < 1:
            continue
        color = colors[method]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        lines += f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>'
        for x, y, val in points:
            lines += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="#fff" stroke-width="2"><title>{_esc(method)} L{round((x-left)/plot_w*level_span+min_level)}: {val:.4f}</title></circle>'
        legend += (
            f'<span class="legend-item"><span style="background:{color}"></span>'
            f'{_esc(method)}</span>'
        )

    return (
        '<section><h2>Degradation Curve</h2><div class="figure">'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Degradation Curve">'
        f'{grid}{labels_y}{labels_x}<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>'
        f'{lines}<text x="{left + plot_w/2:.1f}" y="{height-14}" text-anchor="middle" class="axis-label">'
        'Difficulty Level</text><text x="16" y="160" transform="rotate(-90 16 160)" text-anchor="middle" class="axis-label">'
        'KTC Score</text></svg><div class="legend chart-legend">'
        f'{legend}</div></div></section>'
    )


def _radar_svg(summary: list[dict], metric_defs: list[tuple[str, str]]) -> str:
    if not summary:
        return ""
    cx = cy = 190
    radius = 130
    axes = metric_defs
    if len(axes) < 2:
        return ""
    pts_axis = []
    labels = []
    import math
    for i, (label, _) in enumerate(axes):
        angle = -math.pi / 2 + (2 * math.pi * i / len(axes))
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        lx = cx + (radius + 28) * math.cos(angle)
        ly = cy + (radius + 28) * math.sin(angle)
        pts_axis.append((x, y))
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">{_esc(label)}</text>')

    grid = ""
    for scale in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for x, y in pts_axis:
            pts.append(f"{cx + (x - cx) * scale:.1f},{cy + (y - cy) * scale:.1f}")
        grid += f'<polygon points="{" ".join(pts)}" class="grid"/>'
    spokes = "".join(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" class="axis"/>'
                     for x, y in pts_axis)

    polys = ""
    legend = ""
    for idx, row in enumerate(summary):
        color = PALETTE[idx % len(PALETTE)]
        pts = []
        for axis_idx, (_, key) in enumerate(axes):
            x, y = pts_axis[axis_idx]
            val = max(0.0, min(1.0, float(row.get(key, 0.0))))
            pts.append(f"{cx + (x - cx) * val:.1f},{cy + (y - cy) * val:.1f}")
        polys += (
            f'<polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="0.14" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        legend += (
            f'<span class="legend-item"><span style="background:{color}"></span>'
            f'{_esc(row["method"])}</span>'
        )

    return (
        '<section><h2>Radar Chart — Balanced Skill Profile</h2>'
        '<p class="chart-note"><b>How to read this shape.</b> Each colored polygon is one method; '
        'each spoke is a different quality metric, scored from 0 at the centre to 1 at the outer '
        'edge. A <b>large, evenly balanced</b> shape means the method is strong on every metric at '
        'once. A <b>lopsided</b> shape reveals a trade-off &mdash; for example recovering resistive '
        'targets well while missing conductive ones, or scoring on overlap (Dice) but not on '
        'boundary accuracy (IoU). More area covered = better all-round reconstruction.</p>'
        '<div class="radar-wrap">'
        f'<svg viewBox="0 0 380 380" role="img">{grid}{spokes}{polys}{"".join(labels)}</svg>'
        f'<div class="legend">{legend}</div></div></section>'
    )


def _degradation_table(results: list[dict]) -> str:
    methods = sorted({str(r["method"]) for r in results})
    levels = sorted({int(r.get("level", 1)) for r in results})
    rows = []
    for method in methods:
        row = [_esc(method)]
        method_rows = [r for r in results if str(r["method"]) == method]
        for level in levels:
            vals = [_metric(r, "ktc_score") for r in method_rows if int(r.get("level", 1)) == level]
            row.append(f"{_avg(vals):.3f}" if vals else "-")
        rows.append(row)
    return _table(["Method"] + [f"L{lv}" for lv in levels], rows)


def _metrics_table(summary: list[dict], metric_defs: list[tuple[str, str]]) -> str:
    rows = []
    for row in summary:
        cells = [_esc(row["method"])]
        cells.extend(f'{row[key]:.3f}' for _, key in metric_defs)
        cells.extend([_runtime_fmt(float(row["runtime"])), _esc(row["grade"])])
        rows.append(cells)
    return _table(["Method"] + [label for label, _ in metric_defs] + ["Runtime", "Grade"], rows)


def _failure_summary(results: list[dict]) -> str:
    rows = []
    for method in sorted({str(r["method"]) for r in results}):
        method_rows = [r for r in results if str(r["method"]) == method]
        ktc_vals = [_metric(r, "ktc_score") for r in method_rows]
        worst = min(method_rows, key=lambda r: _metric(r, "ktc_score"), default=None)
        rows.append([
            _esc(method),
            str(sum(1 for v in ktc_vals if v < 0.30)),
            f"{min(ktc_vals):.3f}" if ktc_vals else "-",
            f"{_avg(ktc_vals):.3f}" if ktc_vals else "-",
            _esc(f"L{worst.get('level')} {worst.get('sample')}" if worst else "-"),
        ])
    return _table(["Method", "Runs < 0.30", "Worst KTC", "Mean KTC", "Worst Sample"], rows)


def _reconstruction_section(results: list[dict], figures_dir: Path) -> str:
    rows = []
    cards = []
    for method in sorted({str(r["method"]) for r in results}):
        method_rows = [r for r in results if str(r["method"]) == method]
        best = max(method_rows, key=lambda r: _metric(r, "ktc_score"), default=None)
        worst = min(method_rows, key=lambda r: _metric(r, "ktc_score"), default=None)
        rows.append([
            _esc(method),
            str(len(method_rows)),
            f"{_metric(best, 'ktc_score'):.3f}" if best else "-",
            _esc(f"L{best.get('level')} {best.get('sample')}" if best else "-"),
            f"{_metric(worst, 'ktc_score'):.3f}" if worst else "-",
            _esc(f"L{worst.get('level')} {worst.get('sample')}" if worst else "-"),
        ])

    # Curate the image gallery instead of embedding every run. Dumping all
    # runs (e.g. 8 methods x 21 = 168 images) pushed the report past 7 MB of
    # base64, which made it slow to build and the download button sluggish.
    # For storytelling we only need to *show the range*: for each method,
    # its best case, its worst case, and a typical (median) case. That is
    # the visual evidence a reader needs -- "here is this method at its
    # best, its worst, and on an average day" -- at a fraction of the size.
    def sort_key(row: dict) -> tuple[str, int, str]:
        return (str(row.get("method", "")), int(row.get("level", 1)), str(row.get("sample", "")))

    curated: list[tuple[dict, str]] = []
    for method in sorted({str(r["method"]) for r in results}):
        method_rows = sorted(
            [r for r in results if str(r["method"]) == method],
            key=lambda r: _metric(r, "ktc_score"),
        )
        if not method_rows:
            continue
        seen: set[int] = set()
        worst_i, best_i, mid_i = 0, len(method_rows) - 1, len(method_rows) // 2
        for idx, tag in ((best_i, "best case"), (mid_i, "typical"), (worst_i, "worst case")):
            if idx not in seen:
                seen.add(idx)
                curated.append((method_rows[idx], tag))

    for item, tag in sorted(curated, key=lambda t: sort_key(t[0])):
        method = str(item.get("method", ""))
        level = int(item.get("level", 1))
        sample = str(item.get("sample", ""))
        stem = f"{method}_level{level}_sample{sample}.png"
        image_path = item.get("image_path")
        uri = _embed_png(image_path) if image_path else ""
        if not uri:
            uri = _embed_png(figures_dir / stem)
        if uri:
            cards.append(
                f'<figure><img src="{uri}" alt="{_esc(stem)}">'
                f'<figcaption><b>{_esc(method)}</b> &middot; {tag}<br>'
                f'L{level} {sample} &middot; KTC {_metric(item, "ktc_score"):.3f}</figcaption></figure>'
            )
    caption = (
        '<p style="font-size:11px;color:#57606a;margin:2px 0 10px">'
        "For each method we show three representative reconstructions &mdash; its "
        "<b>best</b>, a <b>typical</b>, and its <b>worst</b> case &mdash; so you see the full "
        "range of what it produces rather than one cherry-picked example. A good reconstruction "
        "places objects in the right spot with roughly the right size; a poor one smears them "
        "or invents structure that is not there.</p>"
    )
    return (
        "<section><h2>Reconstruction Images</h2>"
        + _table(["Method", "Runs", "Best KTC", "Best Sample", "Worst KTC", "Worst Sample"], rows)
        + caption
        + f'<div class="recon-grid">{"".join(cards)}</div></section>'
    )


def _hull_section(results: list[dict]) -> str:
    rows = []
    buckets: dict[str, list[dict]] = {}
    for row in results:
        hull = row.get("hull") or {}
        if hull:
            buckets.setdefault(str(row["method"]), []).append(hull)
    for method, hulls in sorted(buckets.items()):
        def avg_key(key: str) -> float:
            vals = [float(h.get(key, 0.0) or 0.0) for h in hulls]
            return _avg(vals)
        rows.append([
            _esc(method),
            f'{avg_key("hull_resistive_center_error"):.2f}',
            f'{avg_key("hull_resistive_area_error"):.2f}',
            f'{avg_key("hull_conductive_center_error"):.2f}',
            f'{avg_key("hull_conductive_area_error"):.2f}',
            str(len(hulls)),
        ])
    if not rows:
        return '<section><h2>Hull Analysis</h2><p class="muted">No hull data available for this report.</p></section>'
    return (
        "<section><h2>Hull Analysis</h2>"
        + _table(["Method", "Res Center Err", "Res Area Err", "Con Center Err", "Con Area Err", "Runs"], rows)
        + "</section>"
    )


def _hull_svg_figures(results: list[dict]) -> str:
    buckets: dict[str, list[dict]] = {}
    for row in results:
        hull = row.get("hull") or {}
        if hull:
            buckets.setdefault(str(row["method"]), []).append({**hull, "ktc_score": _metric(row, "ktc_score"), "level": row.get("level", 1)})
    if not buckets:
        return '<section><h2>Hull Analysis Figures</h2><p class="muted">No hull data available for charting.</p></section>'

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    methods = sorted(buckets)
    center = [avg([float(h.get("hull_resistive_center_error", 0.0) or 0.0) for h in buckets[m]]) for m in methods]
    area = [avg([float(h.get("hull_resistive_area_error", 0.0) or 0.0) for h in buckets[m]]) for m in methods]

    def axis_ticks(max_v: float, count: int = 4) -> list[float]:
        return [(max_v * i / count) for i in range(count + 1)]

    def bar_svg(title: str, values: list[float], unit: str) -> str:
        width, height = 520, 210
        left, top, plot_w, plot_h = 115, 28, 360, 126
        max_v = max(values) if values else 1.0
        max_v = max(max_v, 1.0)
        grid = ""
        for tick in axis_ticks(max_v):
            x = left + (tick / max_v) * plot_w
            grid += (
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="chart-grid"/>'
                f'<text x="{x:.1f}" y="{top + plot_h + 13}" text-anchor="middle" class="axis-label">{tick:.0f}</text>'
            )
        rows = ""
        for idx, (method, value) in enumerate(zip(methods, values)):
            y = top + idx * (plot_h / max(1, len(methods)))
            bar_h = max(10, plot_h / max(1, len(methods)) - 7)
            bar_w = plot_w * (value / max_v) if max_v else 0
            color = PALETTE[idx % len(PALETTE)]
            label = method if len(method) <= 19 else method[:17] + ".."
            rows += (
                f'<text x="4" y="{y + bar_h * .7:.1f}" class="axis-label">{_esc(label)}</text>'
                f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
                f'<text x="{min(left + bar_w + 5, width - 38):.1f}" y="{y + bar_h * .7:.1f}" class="bar-label">{value:.1f}</text>'
            )
        return (
            f'<figure><svg viewBox="0 0 {width} {height}" role="img">'
            f'<text x="4" y="14" class="bar-label">{_esc(title)}</text>'
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" class="axis-label">{_esc(unit)}</text>'
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>'
            f'{grid}{rows}</svg></figure>'
        )

    levels = sorted({int(h.get("level", 1)) for hs in buckets.values() for h in hs})
    line_svg = ""
    if levels:
        width, height = 520, 210
        left, top, plot_w, plot_h = 48, 26, 420, 126
        max_center = max(center) if center else 1.0
        max_center = max(max_center, 1.0)
        level_span = max(1, max(levels) - min(levels))
        grid = ""
        for tick in axis_ticks(max_center):
            y = top + plot_h - (tick / max_center) * plot_h
            grid += (
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="chart-grid"/>'
                f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" class="axis-label">{tick:.0f}</text>'
            )
        xlabels = ""
        for level in levels:
            x = left + ((level - min(levels)) / level_span) * plot_w
            xlabels += f'<text x="{x:.1f}" y="{top + plot_h + 13}" text-anchor="middle" class="axis-label">L{level}</text>'
        lines = ""
        for idx, method in enumerate(methods):
            points = []
            for level in levels:
                vals = [float(h.get("hull_resistive_center_error", 0.0) or 0.0) for h in buckets[method] if int(h.get("level", 1)) == level]
                if vals:
                    x = left + ((level - min(levels)) / level_span) * plot_w
                    y = top + plot_h - (avg(vals) / max_center) * plot_h
                    points.append(f"{x:.1f},{y:.1f}")
            if points:
                color = PALETTE[idx % len(PALETTE)]
                lines += f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        line_svg = (
            f'<figure><svg viewBox="0 0 {width} {height}" role="img">'
            f'<text x="4" y="14" class="bar-label">Hull Error Degradation</text>'
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" class="axis-label">Difficulty level</text>'
            f'<text x="12" y="{top + plot_h / 2:.1f}" transform="rotate(-90 12 {top + plot_h / 2:.1f})" text-anchor="middle" class="axis-label">Center error (px)</text>'
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>'
            f'{grid}{xlabels}{lines}</svg></figure>'
        )

    scatter = '<figure><svg viewBox="0 0 520 210" role="img"><text x="4" y="14" class="bar-label">KTC vs Center Error</text>'
    all_points = [(float(h.get("ktc_score", 0.0) or 0.0), float(h.get("hull_resistive_center_error", 0.0) or 0.0), idx)
                  for idx, method in enumerate(methods) for h in buckets[method]]
    max_err = max([p[1] for p in all_points], default=1.0)
    max_err = max(max_err, 1.0)
    left, top, plot_w, plot_h = 48, 26, 420, 126
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * plot_w
        scatter += (
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="chart-grid"/>'
            f'<text x="{x:.1f}" y="{top + plot_h + 13}" text-anchor="middle" class="axis-label">{tick:.2f}</text>'
        )
    for tick in axis_ticks(max_err):
        y = top + plot_h - (tick / max_err) * plot_h
        scatter += (
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="chart-grid"/>'
            f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" class="axis-label">{tick:.0f}</text>'
        )
    scatter += (
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>'
        f'<text x="{left + plot_w / 2:.1f}" y="202" text-anchor="middle" class="axis-label">KTC score</text>'
        f'<text x="12" y="{top + plot_h / 2:.1f}" transform="rotate(-90 12 {top + plot_h / 2:.1f})" text-anchor="middle" class="axis-label">Center error (px)</text>'
    )
    for ktc, err, idx in all_points[::max(1, len(all_points)//90)]:
        x = left + max(0.0, min(1.0, ktc)) * plot_w
        y = top + plot_h - (err / max_err) * plot_h if max_err else top + plot_h
        scatter += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{PALETTE[idx % len(PALETTE)]}"/>'
    scatter += '</svg></figure>'

    return (
        '<section><h2>Hull Analysis Figures</h2><div class="hull-grid">'
        + bar_svg("Center Error by Method", center, "Center error (px)")
        + bar_svg("Hull Area Error by Method", area, "Area error (px2)")
        + scatter
        + line_svg
        + '</div></section>'
    )


def _hull_figures(figures_dir: Path, results: list[dict]) -> str:
    if not figures_dir.exists():
        return _hull_svg_figures(results)
    images = []
    for path in sorted(figures_dir.glob("hull*.png")):
        uri = _embed_png(path)
        if uri:
            images.append(
                f'<figure><img src="{uri}" alt="{_esc(path.stem)}">'
                f'<figcaption>{_esc(path.stem.replace("_", " ").title())}</figcaption></figure>'
            )
    if not images:
        return _hull_svg_figures(results)
    return f'<section><h2>Hull Analysis Figures</h2><div class="hull-grid">{"".join(images)}</div></section>'


def _grade_meaning(grade: str) -> str:
    """Plain-language meaning of a letter grade, for the narrative."""
    return {
        "A": "an excellent, near-perfect reconstruction",
        "B": "a strong reconstruction that recovers the real structure",
        "C": "a moderate result — it locates inclusions but blurs their shape",
        "D": "a poor result, barely above the empty-tank baseline",
    }.get(str(grade), "an unclassified result")


# ---------------------------------------------------------------------------
# Scientific narrative computations (5 stats that turn the executive summary
# from descriptive into scientific — see reporting/html_report.py callers).
# ---------------------------------------------------------------------------

def _mean_std_by_method(results: list[dict]) -> dict[str, tuple[float, float]]:
    """Computation 1: per-method (mean, std) of ktc_score across all its runs."""
    buckets: dict[str, list[float]] = {}
    for r in results:
        buckets.setdefault(str(r["method"]), []).append(_metric(r, "ktc_score"))
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in buckets.items() if v}


def _degradation_slopes(results: list[dict]) -> dict[str, float]:
    """Computation 2: per-method degradation slope of mean KTC score vs level.

    Mirrors BatchRunner._print_degradation()'s np.polyfit(levels, means, 1)
    approach, recomputed here rather than read from scores.json: the runner
    attaches "degradation_slope" to each result dict only *after* _save()
    has already written scores.json, so that field is never actually
    persisted to disk.
    """
    per_level: dict[str, dict[int, list[float]]] = {}
    for r in results:
        m = str(r["method"])
        lv = int(r.get("level", 1) or 1)
        per_level.setdefault(m, {}).setdefault(lv, []).append(_metric(r, "ktc_score"))

    slopes: dict[str, float] = {}
    for m, by_level in per_level.items():
        levels = sorted(by_level)
        means = [float(np.mean(by_level[lv])) for lv in levels]
        slopes[m] = float(np.polyfit(levels, means, 1)[0]) if len(levels) >= 2 else 0.0
    return slopes


def _resistive_conductive_gaps(summary: list[dict]) -> dict[str, float]:
    """Computation 4: per-method (dice_resistive - dice_conductive) / dice_resistive * 100.

    Positive = conductive is harder for that method; negative = easier.
    """
    gaps: dict[str, float] = {}
    for row in summary:
        dr, dc = row.get("dice_resistive"), row.get("dice_conductive")
        if isinstance(dr, (int, float)) and isinstance(dc, (int, float)) and dr:
            gaps[str(row["method"])] = (dr - dc) / dr * 100.0
    return gaps


def _easy_hard_means(results: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """Computation 5 groundwork: per-method mean KTC for levels 1-4 vs 5-7."""
    easy: dict[str, list[float]] = {}
    hard: dict[str, list[float]] = {}
    for r in results:
        m = str(r["method"])
        lv = int(r.get("level", 1) or 1)
        v = _metric(r, "ktc_score")
        (easy if 1 <= lv <= 4 else hard if 5 <= lv <= 7 else {}).setdefault(m, []).append(v)
    easy_means = {m: float(np.mean(v)) for m, v in easy.items() if v}
    hard_means = {m: float(np.mean(v)) for m, v in hard.items() if v}
    return easy_means, hard_means


def _leaderboard_table_with_stats(
    summary: list[dict],
    metric_defs: list[tuple[str, str]],
    mean_std: dict[str, tuple[float, float]],
    slopes: dict[str, float],
) -> str:
    """Leaderboard table with Computations 1-2 as extra columns (± Std, Slope).

    Deliberately a separate function from _metrics_table(), which the Metric
    Breakdown section also uses unchanged — adding KTC-score stats columns
    there would be off-topic for a per-class Dice/IoU table.
    """
    headers = ["Method"]
    for label, key in metric_defs:
        headers.append(label)
        if key == "ktc_score":
            headers.append("± Std")
    headers.extend(["Runtime", "Grade", "Slope"])

    rows = []
    for row in summary:
        method = str(row["method"])
        cells = [_esc(method)]
        for label, key in metric_defs:
            cells.append(f'{row[key]:.3f}')
            if key == "ktc_score":
                std = mean_std.get(method, (0.0, 0.0))[1]
                cells.append(f'&plusmn;{std:.3f}')
        cells.extend([_runtime_fmt(float(row["runtime"])), _esc(row["grade"])])
        cells.append(f'{slopes.get(method, 0.0):+.4f}')
        rows.append(cells)
    return _table(headers, rows)


def _build_narratives(summary: list[dict], results: list[dict]) -> dict:
    """Turn the raw numbers into interpreted, story-telling HTML snippets.

    Every snippet is computed from this specific run, so the report explains
    *what* the reader is looking at, *why* it matters, *which* method wins, and
    *how* to read it — instead of leaving bare tables to interpret themselves.
    """
    keys = ["exec", "context", "lb", "deg", "metrics", "fail", "recon", "hull", "reco", "recommendation"]
    if not summary:
        return {k: "" for k in keys}

    best, worst = summary[0], summary[-1]
    n, total = len(summary), len(results)
    bg = str(best.get("grade", "?"))
    gap = float(best.get("ktc_score", 0.0)) - float(worst.get("ktc_score", 0.0))

    # Overall score by difficulty level (for the degradation story)
    by_level: dict[int, list] = {}
    for r in results:
        by_level.setdefault(int(r.get("level", 1) or 1), []).append(_metric(r, "ktc_score"))
    lvl_mean = {lv: sum(v) / len(v) for lv, v in by_level.items() if v}
    low = [lvl_mean[l] for l in lvl_mean if l <= 2]
    high = [lvl_mean[l] for l in lvl_mean if l >= 6]
    drop = (sum(low) / len(low) - sum(high) / len(high)) if low and high else None

    # Most robust method = smallest easy→hard drop
    per_method: dict[str, dict] = {}
    for r in results:
        per_method.setdefault(str(r["method"]), {}).setdefault(
            int(r.get("level", 1) or 1), []).append(_metric(r, "ktc_score"))
    most_robust, best_flat = None, 1e9
    for m, lv in per_method.items():
        lm = {k: sum(v) / len(v) for k, v in lv.items() if v}
        lo = [lm[l] for l in lm if l <= 2]; hi = [lm[l] for l in lm if l >= 6]
        if lo and hi:
            d = sum(lo) / len(lo) - sum(hi) / len(hi)
            if d < best_flat:
                best_flat, most_robust = d, m

    # ---- Metric breakdown: which class is genuinely harder, this run? ------
    def _cls_mean(key: str) -> float | None:
        vals = [_metric(r, key) for r in results if isinstance(r.get("metrics", {}).get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    dice_res, dice_con = _cls_mean("dice_resistive"), _cls_mean("dice_conductive")
    harder_class = None
    if dice_res is not None and dice_con is not None and abs(dice_res - dice_con) > 0.01:
        harder_class = ("conductive", dice_con, dice_res) if dice_con < dice_res else ("resistive", dice_res, dice_con)

    # Most lopsided method (largest gap between its two Dice classes)
    lop_method, lop_gap = None, 0.0
    for row in summary:
        dr, dc = row.get("dice_resistive"), row.get("dice_conductive")
        if isinstance(dr, (int, float)) and isinstance(dc, (int, float)) and abs(dr - dc) > lop_gap:
            lop_gap, lop_method, lop_pair = abs(dr - dc), str(row["method"]), (dr, dc)

    # ---- Failures: where do the worst reconstructions actually cluster? -----
    ranked_runs = sorted(results, key=lambda r: _metric(r, "ktc_score"))
    n_worst = max(3, len(ranked_runs) // 10)
    worst_runs = ranked_runs[:n_worst]
    worst_levels = [int(r.get("level", 1) or 1) for r in worst_runs]
    hard_share = sum(1 for l in worst_levels if l >= 5) / len(worst_levels) if worst_levels else 0.0
    worst_lo = float(_metric(worst_runs[0], "ktc_score")) if worst_runs else 0.0
    fail_methods = sorted({str(r["method"]) for r in worst_runs})

    # ---- Hull: real geometric errors -----------------------------------------
    hull_center = {}
    for r in results:
        h = r.get("hull") or {}
        ce = h.get("hull_resistive_center_error")
        if isinstance(ce, (int, float)):
            hull_center.setdefault(str(r["method"]), []).append(float(ce))
    hull_avg = {m: sum(v) / len(v) for m, v in hull_center.items() if v}
    hull_best = min(hull_avg.items(), key=lambda kv: kv[1]) if hull_avg else None
    hull_overall = (sum(hull_avg.values()) / len(hull_avg)) if hull_avg else None

    def box(text: str) -> str:
        return f'<div class="narrative">{text}</div>'

    deg_line = ""
    if drop is not None:
        trend = "declines steadily" if drop > 0.02 else "holds roughly steady"
        deg_line = (f" Overall accuracy {trend} as the problem gets harder — mean KTC falls by "
                    f"about {drop:.2f} from the easiest levels to the hardest.")

    # ---- 5 scientific computations, each producing one sentence ---------------
    mean_std = _mean_std_by_method(results)
    slopes = _degradation_slopes(results)
    rc_gaps = _resistive_conductive_gaps(summary)
    easy_means, hard_means = _easy_hard_means(results)

    # Computation 1: confidence — best method's score with its std, in context.
    best_mean, best_std = mean_std.get(best["method"], (best["ktc_score"], 0.0))
    best_runs = sum(1 for r in results if str(r["method"]) == best["method"])
    consistency = "tightly clustered" if (best_mean <= 0 or best_std / best_mean < 0.3) else "notably variable"
    sent1 = (f"<b>Confidence.</b> {_esc(best['method'])}: {best_mean:.3f} &plusmn; {best_std:.2f} "
             f"across {best_runs} runs (grade {bg}) — scores are {consistency} run to run.")

    # Computation 2: degradation slope — least vs most steeply degrading method.
    if len(slopes) >= 2:
        least_steep = min(slopes, key=lambda m: abs(slopes[m]))
        most_steep = max(slopes, key=lambda m: abs(slopes[m]))
        sent2 = (f"<b>Degradation rate.</b> {_esc(least_steep)} degrades least steeply "
                  f"({slopes[least_steep]:+.3f}/level) while {_esc(most_steep)} degrades fastest "
                  f"({slopes[most_steep]:+.3f}/level).")
    elif len(slopes) == 1:
        only = next(iter(slopes))
        sent2 = (f"<b>Degradation rate.</b> {_esc(only)}'s score changes by {slopes[only]:+.3f} "
                  f"per difficulty level.")
    else:
        sent2 = ""

    # Computation 3: speed/accuracy trade-off — fastest method vs the top scorer.
    sent3 = ""
    if len(summary) >= 2:
        fastest = min(summary, key=lambda r: r["runtime"])
        if fastest["method"] == best["method"]:
            sent3 = (f"<b>Speed vs. accuracy.</b> {_esc(best['method'])} is both the most accurate "
                      f"and the fastest method here — no trade-off to make.")
        elif best["ktc_score"] > 0 and best["runtime"] > 0:
            accuracy_pct = 100.0 * fastest["ktc_score"] / best["ktc_score"]
            cost_pct = 100.0 * fastest["runtime"] / best["runtime"]
            sent3 = (f"<b>Speed vs. accuracy.</b> {_esc(fastest['method'])} achieves {accuracy_pct:.0f}% "
                      f"of {_esc(best['method'])}'s accuracy at {cost_pct:.2f}% of its compute cost "
                      f"({_runtime_fmt(fastest['runtime'])} vs {_runtime_fmt(best['runtime'])} per run).")

    # Computation 4: resistive vs conductive difficulty gap, across all methods.
    sent4 = ""
    if rc_gaps:
        min_gap, max_gap = min(rc_gaps.values()), max(rc_gaps.values())
        if max_gap >= 0:
            sent4 = (f"<b>Resistive vs. conductive.</b> Conductive recovery is "
                      f"{min_gap:.0f}&ndash;{max_gap:.0f}% harder than resistive across all methods.")
        else:
            sent4 = (f"<b>Resistive vs. conductive.</b> Conductive recovery is actually "
                      f"{abs(max_gap):.0f}&ndash;{abs(min_gap):.0f}% <i>easier</i> than resistive "
                      f"across all methods.")

    # Computation 5: conditional recommendation — best method for easy vs hard levels.
    reco_sentence = ""
    if easy_means and hard_means:
        best_easy = max(easy_means, key=easy_means.get)
        best_hard = max(hard_means, key=hard_means.get)
        if best_easy == best_hard:
            reco_sentence = f"{_esc(best_easy)} dominates across all difficulty levels."
        else:
            slope_easy, slope_hard = slopes.get(best_easy), slopes.get(best_hard)
            if slope_easy and abs(slope_hard or 0) < abs(slope_easy):
                pct_less = (1 - abs(slope_hard) / abs(slope_easy)) * 100.0
                reco_sentence = (
                    f"For levels 1&ndash;4, use {_esc(best_easy)}; for levels 5&ndash;7, "
                    f"{_esc(best_hard)} degrades {pct_less:.0f}% less steeply and pulls ahead.")
            else:
                reco_sentence = (
                    f"For levels 1&ndash;4, use {_esc(best_easy)}; for levels 5&ndash;7, switch to "
                    f"{_esc(best_hard)}, which scores higher in that harder range.")
    sent5 = f"<b>Recommendation.</b> {reco_sentence}" if reco_sentence else ""


    N = {}
    N["exec"] = box(
        f"<b>Executive summary.</b> This report evaluates <b>{n}</b> reconstruction methods across "
        f"<b>{total}</b> test cases (difficulty levels &times; samples). "
        f"<b>{_esc(best['method'])}</b> performs best, with a mean KTC of "
        f"<b>{best['ktc_score']:.3f}</b> (grade {bg}) — {_grade_meaning(bg)}. "
        f"The weakest method, {_esc(worst['method'])}, averages {worst['ktc_score']:.3f}.{deg_line} "
        f"{sent1} {sent2} {sent3} {sent4} {sent5}")
    N["recommendation"] = box(
        f"<b>Recommendation.</b> {reco_sentence}" if reco_sentence else
        "<b>Recommendation.</b> Not enough level coverage in this run to split easy vs. hard "
        "difficulty levels (need results across both levels 1&ndash;4 and 5&ndash;7).")
    N["context"] = box(
        "<b>How to read this report.</b> EIT reconstructs what is inside a 32-electrode tank from "
        "boundary voltage measurements — an <i>ill-posed</i> problem where noise easily corrupts the "
        "result, so a method's regularization matters. Scores use the official <b>KTC score</b>: "
        "<b>0</b> = an empty-tank guess, <b>1</b> = a perfect match — higher is better. Difficulty "
        "levels 1&rarr;7 progressively <b>remove electrode data</b>, so higher levels are genuinely harder.")
    N["lb"] = box(
        f"<b>What</b> this shows: the overall ranking of every method. "
        f"<b>Why</b> it matters: it is the single-glance verdict on which method to trust. "
        f"<b>Which</b> wins: <b>{_esc(best['method'])}</b> (grade {bg}). "
        f"<b>How</b> to read it: a longer bar / higher score is a better reconstruction; a short bar "
        f"means the method barely beats an empty-tank guess. "
        + (f"A {gap:.3f} KTC spread separates best from worst, so the choice of method matters."
           if gap > 0.1 else "The methods are fairly close, so the trade-offs are subtle."))
    N["deg"] = box(
        f"<b>What</b> this shows: how each method's accuracy changes as electrode data is removed. "
        f"<b>Why</b> it matters: real systems operate with varying coverage, so robustness is as "
        f"important as peak accuracy. "
        + (f"<b>Which</b> holds up best: <b>{_esc(most_robust)}</b>, which degrades the least. "
           if most_robust else "")
        + "<b>How</b> to read it: a flatter line means the method stays reliable even when most "
          "measurement data is gone.")
    metrics_insight = ""
    if harder_class is not None:
        cls, lo_v, hi_v = harder_class
        other = "resistive" if cls == "conductive" else "conductive"
        metrics_insight = (
            f"<b>Which</b> is harder here: across every method the <b>{cls}</b> region is "
            f"reconstructed worse (mean Dice {lo_v:.2f}) than the {other} region ({hi_v:.2f}) — "
            f"so {cls} inclusions are the common weak spot in this run. ")
    if lop_method is not None and lop_gap > 0.05:
        metrics_insight += (
            f"<b>{_esc(lop_method)}</b> is the most unbalanced method (Dice "
            f"{lop_pair[0]:.2f} resistive vs {lop_pair[1]:.2f} conductive), a {lop_gap:.2f} gap. ")
    N["metrics"] = box(
        "<b>What</b> this shows: per-class overlap metrics (Dice / IoU) for the resistive and "
        "conductive regions. <b>Why</b> it matters: the KTC score is one number — these reveal "
        "<i>where</i> a method succeeds or fails. "
        + metrics_insight +
        "<b>How</b> to read it: higher overlap is better; a low value on one class "
        "pinpoints exactly what a method struggles to reconstruct.")

    if hard_share >= 0.6:
        fail_where = (f"<b>Which</b> cases: {int(round(hard_share*100))}% of the worst "
                      f"reconstructions land on the hardest levels (5–7), so failure here is "
                      f"<i>systematic</i> — driven by sparse data, not random noise. ")
    elif hard_share <= 0.3:
        fail_where = (f"<b>Which</b> cases: the worst reconstructions are <i>spread across</i> "
                      f"difficulty levels rather than concentrated on the hard ones, which points to "
                      f"case-specific noise more than a clean difficulty effect. ")
    else:
        fail_where = ("<b>Which</b> cases: the worst reconstructions lean toward the harder levels "
                      "but also appear on easier ones — a mix of difficulty and case-specific noise. ")
    N["fail"] = box(
        "<b>What</b> this shows: the lowest-scoring reconstructions. <b>Why</b> it matters: averages "
        f"hide the failure modes that decide whether a method is trustworthy (the worst case here "
        f"scored just {worst_lo:.3f} KTC). "
        + fail_where +
        f"<b>How</b> to use it: the methods appearing most in this tail "
        f"({_esc(', '.join(fail_methods[:3]))}{'…' if len(fail_methods) > 3 else ''}) are the ones to "
        f"treat with caution on difficult inputs.")
    N["recon"] = box(
        f"<b>What</b> this shows: the actual reconstructed images beside the ground truth. "
        f"<b>Why</b> it matters: numbers summarize quality, but the images <i>prove</i> it — you can "
        f"see directly how much structure each method recovers. <b>How</b> to read it: compare each "
        f"reconstruction to its ground truth; the closer the shapes match, the higher the score.")
    hull_insight = ""
    if hull_best is not None and hull_overall is not None:
        hull_insight = (
            f"<b>Which</b> is most faithful geometrically: <b>{_esc(hull_best[0])}</b>, placing the "
            f"resistive target's centre just {hull_best[1]:.0f} px from the truth on average "
            f"(run-wide average is {hull_overall:.0f} px). ")
    N["hull"] = box(
        "<b>What</b> this shows: the geometric accuracy of each detected inclusion — its shape, area "
        "and position versus the truth. <b>Why</b> it matters: a method can get the score roughly "
        "right yet misplace or distort the inclusion; hull analysis catches that. "
        + hull_insight +
        "<b>How</b> to read it: smaller position and area errors mean a more faithful reconstruction "
        "of the real object.")

    # ---- Bottom line: an actual recommendation, computed from this run -------
    reco_parts = [
        f"<b>Bottom line.</b> For overall accuracy on this benchmark, <b>{_esc(best['method'])}</b> "
        f"is the strongest choice (mean KTC {best['ktc_score']:.3f}, grade {bg})."]
    if most_robust and most_robust != best["method"]:
        reco_parts.append(
            f" If your system may run with reduced electrode coverage, <b>{_esc(most_robust)}</b> "
            f"degrades the least and is the safer pick under sparse data.")
    elif most_robust == best["method"]:
        reco_parts.append(
            f" It is also the most robust as data is removed, so it is the safe default across "
            f"both easy and hard conditions.")
    if harder_class is not None:
        reco_parts.append(
            f" Across the board, expect {harder_class[0]} inclusions to be the limiting factor — "
            f"that is where all methods lose the most accuracy.")
    N["reco"] = box("".join(reco_parts))
    return N


def _project_background() -> str:
    """General, run-independent information about what this project is and how
    to interpret its scores — the context a first-time reader needs before the
    numbers mean anything."""
    return (
        "<section><h2>About This Project</h2>"
        '<div class="chart-note" style="font-size:11.5px;color:#24292f">'
        "<p style='margin:0 0 7px'><b>The problem.</b> Electrical Impedance Tomography (EIT) "
        "reconstructs the electrical conductivity inside a body from voltage measurements taken by "
        "electrodes around its boundary. In this benchmark the &ldquo;body&rdquo; is a 32-electrode "
        "water tank containing resistive and conductive inclusions. Recovering the interior from "
        "boundary data is a classic <i>ill-posed inverse problem</i>: small measurement errors can "
        "cause large reconstruction errors, so every method must trade sharpness against stability "
        "through some form of regularization.</p>"
        "<p style='margin:0 0 7px'><b>The benchmark.</b> This framework evaluates several "
        "reconstruction methods on the <b>KTC 2023</b> (Kuopio Tomography Challenge) dataset, which "
        "provides <b>7 difficulty levels</b>; each higher level removes more electrode data, so the "
        "inverse problem becomes progressively harder and less determined. Every method is run on the "
        "same samples, scored the same way, and compared head-to-head.</p>"
        "<p style='margin:0'><b>The score.</b> Quality is measured with the official <b>KTC score</b>, "
        "an SSIM-based similarity to the ground truth normalized so that <b>0</b> = an empty-tank "
        "guess and <b>1</b> = a perfect reconstruction. A <i>negative</i> score is possible and means "
        "the method did worse than simply guessing an empty tank. Letter grades (A&ndash;D) summarize "
        "the score band for quick reading.</p></div></section>"
    )


def _statistics_section(summary: list[dict], results: list[dict]) -> str:
    """Generic descriptive statistics for the whole run — the numeric backbone
    behind the narrative."""
    if not results:
        return ""
    import statistics as _stats
    ktcs = [_metric(r, "ktc_score") for r in results]
    levels = sorted({int(r.get("level", 1) or 1) for r in results})
    samples = sorted({str(r.get("sample", "")) for r in results if str(r.get("sample", ""))})
    best_run = max(results, key=lambda r: _metric(r, "ktc_score"))
    worst_run = min(results, key=lambda r: _metric(r, "ktc_score"))
    mean_k = sum(ktcs) / len(ktcs)
    med_k = _stats.median(ktcs)
    std_k = _stats.pstdev(ktcs) if len(ktcs) > 1 else 0.0
    pct_pos = 100.0 * sum(1 for k in ktcs if k > 0) / len(ktcs)

    def cell(value: str, label: str) -> str:
        return f'<div class="stat-cell"><b>{value}</b><span>{label}</span></div>'

    # NOTE: total runs and method count are deliberately omitted here — they are
    # already shown in the KPI cards at the top of the page. This block adds only
    # the *distribution* statistics the cards don't cover, so the two don't repeat.
    # Eight cells => a clean 4x2 grid with no dead space.
    grid = (
        cell(f"{min(levels)}&ndash;{max(levels)}", f"Difficulty levels ({len(levels)})")
        + cell(str(len(samples)) or "&mdash;", "Samples / level")
        + cell(f"{mean_k:.3f}", "Mean KTC (all runs)")
        + cell(f"{med_k:.3f}", "Median KTC")
        + cell(f"{std_k:.3f}", "Std. deviation (spread)")
        + cell(f"{pct_pos:.0f}%", "Runs beating empty-tank (KTC &gt; 0)")
        + cell(f"{_metric(best_run, 'ktc_score'):.3f}",
               f"Best single run &middot; {_esc(str(best_run['method']))} L{best_run.get('level')}")
        + cell(f"{_metric(worst_run, 'ktc_score'):.3f}",
               f"Worst single run &middot; {_esc(str(worst_run['method']))} L{worst_run.get('level')}")
    )
    return (
        "<section><h2>Key Statistics</h2>"
        '<p class="chart-note">The distribution behind the headline numbers — how scores spread across '
        "every reconstruction, from the single best run to the worst. This spread is what the rest of "
        "the report explains.</p>"
        f'<div class="stat-grid">{grid}</div></section>'
    )


# Short, plain-language profile for each method: what it is, and *why* it
# behaves (and therefore scores) the way it does. Matched by substring so
# aliases resolve; unknown/plugin methods fall through to a generic profile so
# the section stays correct as new methods are added.
_METHOD_PROFILES: list[tuple[tuple[str, ...], str, str]] = [
    (("backprojection", "back_projection"),
     "One-shot linear back-projection — applies the transpose of the sensitivity (Jacobian) "
     "matrix to the measured voltage differences in a single, non-iterative step.",
     "Because it takes one step with no explicit regularization it is very fast and reliably finds "
     "roughly <i>where</i> an inclusion is, but it smears edges and mis-estimates area — so it "
     "typically lands mid-table: decent at position, weak on shape."),
    (("gaussnewton", "gauss_newton", "gauss-newton"),
     "Single-step Tikhonov-regularized Gauss-Newton — one linearized solve about a homogeneous "
     "background, using the noise-weighted Jacobian together with a spatial <i>smoothness prior</i> "
     "that keeps neighbouring pixels similar.",
     "The smoothness prior suppresses noise and yields clean, well-placed inclusions on easier "
     "levels, which is why it scores near the top there; but because it linearizes only once about a "
     "fixed background and assumes a set smoothness scale, it cannot resolve fine structure when "
     "measurements are sparse, so its accuracy drops on the hardest levels."),
    (("lineardifference", "reference_fem", "referencefem", "regularizedfem", "regularized_fem",
      "linear_difference"),
     "Linear difference imaging — reconstructs the <i>change</i> in conductivity relative to a "
     "reference (empty-tank) measurement using a single regularized linear solve.",
     "It is accurate when the perturbation is small and the reference is good, so it competes near "
     "the top on typical cases; but its linear assumption breaks down under large contrasts and "
     "heavy data loss, which is why it degrades on the hardest levels."),
    (("competitioncnn", "cnn", "abc1", "deepdbar", "deep_dbar"),
     "A trained convolutional neural network used as a <i>post-processor</i> — the KTC2023 "
     "&lsquo;ABC1&rsquo; competition entry (Beraldo et al., UFABC) — which takes an initial "
     "reconstruction and refines and segments it into the final image.",
     "Having learned from training examples, it cleans up and sharpens reconstructions that resemble "
     "its training distribution extremely well — hence its high score — but it can generalize poorly "
     "or introduce learned artefacts on cases unlike anything it saw in training, so its errors are "
     "less predictable than the purely physics-based methods."),
    (("dampedleastsquares", "damped_ls", "dampedls"),
     "One-step damped (Tikhonov) least-squares — a single regularized linear solve (via the LSQR "
     "algorithm) for the conductivity change, with a penalty (damping &asymp; 0.15) on the overall "
     "<i>magnitude</i> of that change.",
     "Unlike the Gauss-Newton method it uses <i>no spatial smoothness prior</i> — only a magnitude "
     "penalty — so the solution tends to be noisy and speckled. That noise survives segmentation as "
     "spurious structure, which is why it can score at or <i>below zero</i>, worse than guessing an "
     "empty tank."),
    (("noser",),
     "NOSER (Newton&rsquo;s One-Step Error Reconstructor) — a single Gauss-Newton step regularized "
     "with the diagonal of J&#7488;J instead of a full smoothness prior.",
     "One linearized step makes it fast and stable, but stopping after a single update limits how "
     "much detail it can recover, giving middling scores that neither fail badly nor reach the top."),
    (("truncatedsvd", "tsvd", "truncated_svd"),
     "Truncated SVD — inverts the sensitivity matrix while discarding its smallest singular values "
     "to suppress noise amplification.",
     "Truncation controls noise well, but the discarded components carry the fine spatial detail, so "
     "it trades sharpness for stability — steady, but rarely the highest-scoring."),
]


def _method_profile(name: str) -> tuple[str, str]:
    key = name.lower().replace(" ", "")
    for aliases, definition, why in _METHOD_PROFILES:
        if any(a.replace("_", "").replace("-", "") in key for a in aliases):
            return definition, why
    return (
        "A registered reconstruction method evaluated through the same pipeline as the built-ins.",
        "Its behaviour is characterised here directly from its results — see the statistics below "
        "for where it is strong and where it loses accuracy.",
    )


def _method_deepdive_section(summary: list[dict], results: list[dict]) -> str:
    """Per-method deep-dive: a short definition, this-run statistics, and the
    reason the method scores the way it does. Answers 'what is the reason
    behind the score' and 'why/how a method is failing'."""
    if not summary:
        return ""
    cards = []
    for row in summary:
        method = str(row["method"])
        definition, why = _method_profile(method)
        rows_m = [r for r in results if str(r["method"]) == method]
        # per-level means -> strongest / weakest difficulty
        by_lv: dict[int, list] = {}
        for r in rows_m:
            by_lv.setdefault(int(r.get("level", 1) or 1), []).append(_metric(r, "ktc_score"))
        lv_mean = {lv: sum(v) / len(v) for lv, v in by_lv.items() if v}
        best_lv = max(lv_mean, key=lv_mean.get) if lv_mean else None
        worst_lv = min(lv_mean, key=lv_mean.get) if lv_mean else None
        drop = (lv_mean[best_lv] - lv_mean[worst_lv]) if lv_mean else 0.0
        dr = row.get("dice_resistive"); dc = row.get("dice_conductive")
        weak_cls = ""
        if isinstance(dr, (int, float)) and isinstance(dc, (int, float)):
            weak_cls = ("conductive" if dc < dr else "resistive")
            weak_val = min(dr, dc)

        grade = str(row.get("grade", "?"))
        stat_bits = [f"mean KTC <b>{row['ktc_score']:.3f}</b> (grade {grade})"]
        if best_lv is not None and worst_lv is not None and best_lv != worst_lv:
            stat_bits.append(f"strongest at level {best_lv} ({lv_mean[best_lv]:.3f}), "
                             f"weakest at level {worst_lv} ({lv_mean[worst_lv]:.3f})")
        if weak_cls:
            stat_bits.append(f"recovers {weak_cls} regions least well (Dice {weak_val:.2f})")

        # A data-tied closing sentence that connects mechanism to this run.
        tie = ""
        if drop > 0.15:
            tie = (f" In this run that shows up as a steep {drop:.2f} KTC fall from its best to its "
                   f"worst difficulty level — the instability described above.")
        elif drop > 0.0:
            tie = (f" Here it holds up fairly well, losing only {drop:.2f} KTC from its easiest to "
                   f"hardest level.")
        if row["ktc_score"] < 0:
            tie += (" Its negative mean confirms it is, on average, not beating the empty-tank "
                    "baseline on this data.")

        cards.append(
            f'<div class="method-card"><h3>{_esc(method)}'
            f'<span class="grade-pill">grade {grade}</span></h3>'
            f'<p style="margin:2px 0 5px"><b>What it is.</b> {definition}</p>'
            f'<p class="stat">{" &middot; ".join(stat_bits)}</p>'
            f'<p style="margin:4px 0 0"><b>Why it scores this way.</b> {why}{tie}</p></div>'
        )
    return (
        "<section><h2>Each Method Scores and What It Does</h2>"
        '<p class="chart-note">For every method: a plain-language definition, its statistics on this '
        "run, and the mechanistic reason behind those numbers — so the ranking is explained, not just "
        "listed.</p>" + "".join(cards) + "</section>"
    )


def generate_html_report(scores_json_path, figures_dir, output_path=None) -> str:
    """Generate the focused dashboard report and return the HTML path."""
    results, _figures_dir, _output_file, qualitative_data = _load_results(
        scores_json_path, figures_dir, output_path
    )
    summary = _method_summary(results)
    metric_defs = _metric_defs(qualitative_data)
    total_runs = len(results)
    best = summary[0] if summary else None
    best_ktc = f'{best["ktc_score"]:.3f}' if best else "0.000"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    nar = _build_narratives(summary, results)

    leaderboard_chart = (
        _chart_img_any(_figures_dir, ["leaderboard_dashboard.png"], "Leaderboard Chart")
        or _leaderboard_svg(summary)
    )
    degradation_caption = (
        '<p class="chart-note"><b>How to read this curve.</b> Each line is one method. '
        'The horizontal axis runs from the easiest cases (level 1) to the hardest (level 7), '
        'where progressively more of the boundary measurements are removed; the vertical axis '
        'is the KTC score. A line that stays <b>high and flat</b> means the method keeps working '
        'even as data disappears &mdash; that is robustness. A line that <b>dives toward zero</b> '
        'means the method only succeeds on easy problems and collapses on hard ones. Where two '
        'lines cross, the better method for that difficulty flips.</p>'
    )
    degradation_chart = (
        _chart_img_any(_figures_dir, ["degradation_dashboard.png"], "Degradation Curve")
        or _degradation_svg(results)
    )
    metrics_chart = ""

    project_background = _project_background()
    statistics = _statistics_section(summary, results)
    method_deepdive = _method_deepdive_section(summary, results)
    leaderboard_table = _leaderboard_table_with_stats(
        summary, metric_defs, _mean_std_by_method(results), _degradation_slopes(results)
    )
    degradation_table = _degradation_table(results)
    radar = _radar_svg(summary, metric_defs)
    metrics_table = _metrics_table(summary, metric_defs)
    failures = _failure_summary(results)
    recon = _reconstruction_section(results, _figures_dir)
    hull = _hull_figures(_figures_dir, results) + _hull_section(results)

    quality_rows = []
    if qualitative_data:
        for method, qd in sorted(qualitative_data.items()):
            if str(method).startswith("_") or not isinstance(qd, dict):
                continue
            quality_rows.append([
                _esc(method),
                f'{float(qd.get("resistive_detected_pct", 0.0)):.1f}%',
                f'{float(qd.get("conductive_detected_pct", 0.0)):.1f}%',
                str(qd.get("false_positive_count", 0)),
            ])
    quality = (
        "<section><h2>Detection Summary</h2>"
        + _table(["Method", "Resistive Detected", "Conductive Detected", "False Positives"], quality_rows)
        + "</section>"
        if quality_rows else ""
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EIT Dashboard Report</title>
<style>
* {{ box-sizing: border-box; }}
html,body {{ width:100%; max-width:100%; overflow-x:hidden; }}
body {{ margin:0; padding:20px; background:#eef2f7; color:#1f2328; font-family:Inter, Segoe UI, Arial, sans-serif; }}
main {{ width:100%; max-width:1120px; margin:0 auto; }}
.report-page {{ width:100%; max-width:100%; margin:0 0 18px; padding:18px 22px; background:#ffffff; border:1px solid #d0d7de; border-radius:8px; box-shadow:0 10px 26px rgba(31,35,40,.08); overflow:hidden; }}
section {{ max-width:100%; overflow-x:auto; -webkit-overflow-scrolling:touch; }}
header {{ border-bottom:1px solid #d0d7de; padding-bottom:12px; margin-bottom:14px; }}
h1 {{ margin:0 0 4px; font-size:24px; font-weight:650; overflow-wrap:anywhere; }}
h2 {{ margin:12px 0 8px; font-size:15px; border-bottom:1px solid #d0d7de; padding-bottom:5px; }}
.sub,.muted {{ color:#57606a; font-size:12px; }}
.narrative {{ background:#f6f8fa; border-left:3px solid #0969da; border-radius:0 6px 6px 0; padding:9px 12px; margin:6px 0 12px; font-size:11.5px; line-height:1.55; color:#24292f; }}
.narrative b {{ color:#24292f; }}
.chart-note {{ font-size:11px; line-height:1.5; color:#57606a; margin:2px 0 8px; }}
.chart-note b {{ color:#24292f; }}
.stat-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:6px 0 12px; max-width:100%; }}
.stat-cell {{ background:#f6f8fa; border:1px solid #d0d7de; border-radius:6px; padding:8px 10px; }}
.stat-cell b {{ display:block; font-size:15px; color:#1f2328; line-height:1.15; }}
.stat-cell span {{ font-size:9.5px; color:#57606a; }}
.method-card {{ background:#fff; border:1px solid #d0d7de; border-left:3px solid #57606a; border-radius:0 7px 7px 0; padding:9px 12px; margin:7px 0; font-size:11.5px; line-height:1.5; color:#24292f; }}
.method-card h3 {{ margin:0 0 4px; font-size:13px; }}
.method-card .stat {{ color:#57606a; font-size:10.5px; margin:3px 0; font-family:JetBrains Mono, Consolas, monospace; }}
.method-card .stat b {{ color:#1f2328; }}
.grade-pill {{ display:inline-block; font-size:10px; font-weight:500; padding:1px 8px; border-radius:10px; background:#f6f8fa; border:1px solid #d0d7de; margin-left:8px; vertical-align:middle; color:#57606a; }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; margin-bottom:12px; max-width:100%; }}
.card {{ background:#f6f8fa; border:1px solid #d0d7de; border-radius:7px; padding:11px; }}
.card b {{ display:block; font-size:17px; margin-bottom:3px; overflow-wrap:anywhere; line-height:1.18; }}
table {{ width:100%; min-width:520px; border-collapse:collapse; background:white; border:1px solid #d0d7de; border-radius:7px; overflow:hidden; font-size:10px; margin-bottom:8px; }}
th,td {{ padding:5px 6px; border-bottom:1px solid #eaeef2; text-align:left; }}
th {{ background:#f6f8fa; color:#57606a; font-size:9.5px; text-transform:uppercase; letter-spacing:.04em; }}
td,th {{ overflow-wrap:anywhere; }}
tr:last-child td {{ border-bottom:none; }}
.figure {{ max-width:100%; overflow:hidden; background:white; border:1px solid #d0d7de; border-radius:7px; padding:7px; margin-bottom:8px; text-align:center; }}
.figure img,.figure svg {{ display:block; width:100%; max-width:100%; max-height:330px; height:auto; object-fit:contain; }}
.radar-wrap {{ max-width:100%; overflow:hidden; display:grid; grid-template-columns:minmax(0,360px) minmax(0,1fr); gap:14px; align-items:center; background:white; border:1px solid #d0d7de; border-radius:7px; padding:10px; }}
.radar-wrap svg {{ width:100%; max-width:100%; height:auto; }}
svg text {{ font-size:12px; fill:#57606a; }}
.grid {{ fill:none; stroke:#d0d7de; stroke-width:1; }}
.chart-grid {{ stroke:#d0d7de; stroke-width:1; }}
.axis {{ stroke:#d0d7de; stroke-width:1; }}
.axis-label {{ fill:#57606a; font-size:11px; font-family:JetBrains Mono, Consolas, monospace; }}
.bar-label {{ fill:#1f2328; font-size:11px; font-family:JetBrains Mono, Consolas, monospace; font-weight:600; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 14px; }}
.chart-legend {{ justify-content:center; margin-top:6px; }}
.legend-item {{ font-size:13px; color:#57606a; display:inline-flex; align-items:center; gap:6px; }}
.legend-item span {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.recon-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:7px; margin-top:7px; max-width:100%; }}
.hull-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:6px; margin-top:6px; max-width:100%; }}
figure {{ min-width:0; max-width:100%; overflow:hidden; margin:0; background:white; border:1px solid #d0d7de; border-radius:7px; padding:6px; }}
figure img {{ width:100%; max-height:132px; object-fit:contain; display:block; }}
figure svg {{ width:100%; max-width:100%; height:auto; display:block; }}
.hull-grid figure img {{ max-height:155px; }}
figcaption {{ color:#57606a; font-size:10px; margin-top:4px; overflow-wrap:anywhere; }}
.page-label {{ float:right; color:#848d97; font-size:11px; margin-top:4px; }}
@page {{ size:A4 landscape; margin:0; }}
@media print {{
  body {{ padding:0; background:white; }}
  main {{ max-width:none; margin:0; }}
  .report-page {{ width:297mm; height:210mm; overflow:visible; min-height:0; margin:0; padding:7mm; border:0; border-radius:0; box-shadow:none; page-break-after:always; break-after:page; }}
  .report-page:last-child {{ page-break-after:auto; break-after:auto; }}
  h1 {{ font-size:18px; }}
  h2 {{ margin:7px 0 5px; font-size:12px; padding-bottom:3px; }}
  header {{ padding-bottom:6px; margin-bottom:6px; }}
  .cards {{ gap:5px; margin-bottom:6px; }}
  .card {{ padding:6px; }}
  .card b {{ font-size:13px; }}
  table {{ font-size:7.5px; margin-bottom:5px; }}
  th,td {{ padding:3px 4px; }}
  th {{ font-size:7px; letter-spacing:.02em; }}
  .figure {{ padding:4px; margin-bottom:5px; }}
  .figure img,.figure svg {{ max-height:46mm; }}
  .radar-wrap {{ grid-template-columns:82mm 1fr; gap:6px; padding:5px; }}
  .radar-wrap svg {{ max-height:58mm; }}
  .legend-item {{ font-size:8px; }}
  .recon-grid {{ grid-template-columns:repeat(6,minmax(0,1fr)); gap:4px; margin-top:4px; }}
  .hull-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); gap:4px; margin-top:4px; }}
  figure {{ padding:3px; border-radius:4px; }}
  figure img {{ max-height:18mm; }}
  .hull-grid figure img {{ max-height:31mm; }}
  figcaption {{ font-size:6.5px; margin-top:2px; }}
  .page-label {{ font-size:8px; }}
}}
@media (max-width:800px) {{
  body {{ padding:12px; }}
  .report-page {{ min-height:0; padding:14px; border-radius:7px; }}
  .cards,.stat-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .radar-wrap,.hull-grid {{ grid-template-columns:1fr; }}
  .recon-grid {{ grid-template-columns:repeat(auto-fit,minmax(128px,1fr)); }}
  table {{ min-width:480px; }}
  .figure img,.figure svg {{ max-height:none; }}
}}
@media (max-width:520px) {{
  body {{ padding:8px; }}
  .report-page {{ padding:10px; margin-bottom:10px; }}
  h1 {{ font-size:19px; }}
  h2 {{ font-size:13px; }}
  .cards,.recon-grid,.hull-grid,.stat-grid {{ grid-template-columns:1fr; }}
  .card {{ padding:8px; }}
  table {{ min-width:420px; font-size:9px; }}
  th,td {{ padding:4px 5px; }}
  .legend {{ gap:6px 10px; }}
  .legend-item {{ font-size:11px; }}
  figure img {{ max-height:none; }}
}}
</style>
</head>
<body>
<main>
<div class="report-page">
<span class="page-label">Page 1 / 5</span>
<header>
  <h1>EIT Reconstruction Dashboard Report</h1>
  <div class="sub">Generated {now}</div>
</header>
<section class="cards">
  <div class="card"><b>{total_runs}</b><span class="muted">Runs</span></div>
  <div class="card"><b>{len(summary)}</b><span class="muted">Methods</span></div>
  <div class="card"><b>{_esc(best["method"]) if best else "N/A"}</b><span class="muted">Best Method</span></div>
  <div class="card"><b>{best_ktc}</b><span class="muted">Best Mean KTC</span></div>
</section>
{nar['exec']}
{nar['recommendation']}
{nar['reco']}
{project_background}
{statistics}
<section><h2>Leaderboard — Overall Standings</h2>{nar['lb']}{leaderboard_table}</section>
{leaderboard_chart}
</div>
<div class="report-page">
<span class="page-label">Page 2 / 5</span>
<section><h2>Performance vs Difficulty</h2>{nar['deg']}{degradation_table}</section>
{degradation_caption}
{degradation_chart}
{radar}
</div>
<div class="report-page">
<span class="page-label">Page 3 / 5</span>
<section><h2>Metric Breakdown</h2>{nar['metrics']}{metrics_table}</section>
{metrics_chart}
{method_deepdive}
<section><h2>Failures &amp; Limitations</h2>{nar['fail']}{failures}</section>
{quality}
</div>
<div class="report-page">
<span class="page-label">Page 4 / 5</span>
<section><h2>Geometric Accuracy — Hull Analysis</h2>{nar['hull']}</section>
{hull}
</div>
<div class="report-page">
<span class="page-label">Page 5 / 5</span>
<section><h2>Reconstruction Quality — Visual Evidence</h2>{nar['recon']}</section>
{recon}
</div>
</main>
</body>
</html>"""

    _output_file.parent.mkdir(parents=True, exist_ok=True)
    _output_file.write_text(page, encoding="utf-8")
    return str(_output_file)
