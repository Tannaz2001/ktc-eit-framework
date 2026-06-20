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


METRICS = [
    ("KTC", "ktc_score"),
    ("Dice R", "dice_resistive"),
    ("Dice C", "dice_conductive"),
    ("IoU R", "iou_resistive"),
    ("IoU C", "iou_conductive"),
]
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
    if ktc >= 0.60:
        return "A"
    if ktc >= 0.30:
        return "B"
    if ktc >= 0.10:
        return "C"
    return "D"


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
    return {method: PALETTE[idx % len(PALETTE)] for idx, method in enumerate(sorted(methods))}


def _leaderboard_svg(summary: list[dict]) -> str:
    if not summary:
        return ""
    colors = _method_colors([str(row["method"]) for row in summary])
    width, height = 920, 310
    left, right, top, bottom = 48, 18, 20, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = 100.0
    bar_gap = 12
    bar_w = max(22, (plot_w - bar_gap * (len(summary) - 1)) / len(summary))

    grid = ""
    labels_y = ""
    for tick in [0, 25, 50, 75, 100]:
        y = top + plot_h - (tick / max_score) * plot_h
        grid += f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="chart-grid"/>'
        labels_y += f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis-label">{tick}</text>'

    bars = ""
    labels_x = ""
    for idx, row in enumerate(summary):
        method = str(row["method"])
        score = max(0.0, min(100.0, float(row.get("ktc_score", 0.0)) * 100.0))
        x = left + idx * (bar_w + bar_gap)
        h = (score / max_score) * plot_h
        y = top + plot_h - h
        color = colors[method]
        label = method if len(method) <= 18 else method[:16] + ".."
        bars += (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'rx="4" fill="{color}"/>'
            f'<text x="{x + bar_w/2:.1f}" y="{max(12, y-7):.1f}" text-anchor="middle" class="bar-label">'
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
        '<section><h2>Radar Chart</h2><div class="radar-wrap">'
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

    def sort_key(row: dict) -> tuple[str, int, str]:
        return (str(row.get("method", "")), int(row.get("level", 1)), str(row.get("sample", "")))

    for item in sorted(results, key=sort_key):
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
                f'<figcaption>{_esc(method)} - L{level} {sample} - KTC {_metric(item, "ktc_score"):.3f}</figcaption></figure>'
            )
    return (
        "<section><h2>Reconstruction Images</h2>"
        + _table(["Method", "Runs", "Best KTC", "Best Sample", "Worst KTC", "Worst Sample"], rows)
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

    leaderboard_chart = (
        _chart_img_any(_figures_dir, ["leaderboard_dashboard.png"], "Leaderboard Chart")
        or _leaderboard_svg(summary)
    )
    degradation_chart = (
        _chart_img_any(_figures_dir, ["degradation_dashboard.png"], "Degradation Curve")
        or _degradation_svg(results)
    )
    metrics_chart = ""

    leaderboard_table = _metrics_table(summary, metric_defs)
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
  .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
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
  .cards,.recon-grid,.hull-grid {{ grid-template-columns:1fr; }}
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
<section><h2>Leaderboard Table</h2>{leaderboard_table}</section>
{leaderboard_chart}
</div>
<div class="report-page">
<span class="page-label">Page 2 / 5</span>
<section><h2>Degradation Table</h2>{degradation_table}</section>
{degradation_chart}
{radar}
</div>
<div class="report-page">
<span class="page-label">Page 3 / 5</span>
<section><h2>Metrics Table</h2>{metrics_table}</section>
{metrics_chart}
<section><h2>Failure Summary</h2>{failures}</section>
{quality}
</div>
<div class="report-page">
<span class="page-label">Page 4 / 5</span>
{hull}
</div>
<div class="report-page">
<span class="page-label">Page 5 / 5</span>
{recon}
</div>
</main>
</body>
</html>"""

    _output_file.parent.mkdir(parents=True, exist_ok=True)
    _output_file.write_text(page, encoding="utf-8")
    return str(_output_file)
