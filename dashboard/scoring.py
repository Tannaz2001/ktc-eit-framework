"""Scoring helpers: grades, method colors, leaderboard builders, and UI components."""
from __future__ import annotations

import html
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    pd.options.future.infer_string = False
except AttributeError:
    pass

from ktc_framework.reporting.constants import (
    METHOD_COLORS,
    METHOD_COLOR_FALLBACK as _METHOD_COLOR_FALLBACK,
    get_method_color,
    letter_grade as _canonical_letter_grade,
)
from ktc_framework.reporting.data_layer import filter_by_level

from dashboard.data import METRIC_SPECS, METRIC_KEY_TO_LABEL
import dashboard.state as SS


def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float] = None) -> float:
    ktc = metrics.get('KTC score', metrics.get('ktc_score', 0))
    return round(ktc * 100, 2)


def letter_grade(score: float) -> str:
    # Dashboard uses composite score (0–100); canonical function uses raw KTC (0–1).
    return _canonical_letter_grade(score / 100)


def all_methods(scores: Dict) -> List[str]:
    removed_external = SS.removed_external()
    return [
        m for m in list(scores.keys())
        + [m for m in SS.custom_methods() if m not in scores]
        if m not in removed_external
    ]


def method_display_name(method_name: str) -> str:
    """Format internal method IDs for compact, readable sidebar labels."""
    label = re.sub(r"(Reconstruction|Method)$", "", method_name)
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", label)
    return " ".join(label.split()) or method_name


def render_empty_bar(fig: go.Figure, method_name: str, x_position) -> None:
    """Add a zero-height 'No data' placeholder bar to *fig* at *x_position*."""
    fig.add_trace(go.Bar(
        name=method_name, x=[x_position], y=[0],
        marker=dict(
            color=_METHOD_COLOR_FALLBACK,
            pattern=dict(shape="/", fgcolor="#94a3b8", bgcolor=_METHOD_COLOR_FALLBACK, size=6),
        ),
        showlegend=False,
        hovertemplate=f"<b>{method_name}</b><br>No data<extra></extra>",
    ))
    fig.add_annotation(
        x=x_position, y=0, yshift=14,
        text="No data", showarrow=False,
        font=dict(family="JetBrains Mono, monospace", size=9, color=_METHOD_COLOR_FALLBACK),
    )


def render_grade_key() -> None:
    """Render a small, persistent grade-band legend strip (Streamlit)."""
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        'color:var(--tx3);margin:2px 0 8px">'
        'Grade bands: '
        '<span style="color:#1a7f37">A &ge; 60 (green)</span> &middot; '
        '<span style="color:#0969da">B &ge; 30 (blue)</span> &middot; '
        '<span style="color:#9a6700">C &ge; 10 (amber)</span> &middot; '
        '<span style="color:#cf222e">D &lt; 10 (red)</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, tooltip: str = "") -> None:
    """Render a small-caps section label with an optional hover-help icon."""
    icon_html = ""
    if tooltip:
        icon_html = f'<span class="info-tip" data-tip="{html.escape(tooltip)}">?</span>'
    st.markdown(
        f'<div class="slbl" style="display:flex;align-items:center">{title}{icon_html}</div>',
        unsafe_allow_html=True,
    )


def render_what_why_how(what: str, why: str, how: str) -> None:
    """Render a compact WHAT / WHY / HOW strip at the top of a tab."""
    rows = [("WHAT", what, "#0969da"), ("WHY", why, "#8250df"), ("HOW", how, "#1a7f37")]
    html_parts = ['<div style="margin-bottom:12px">']
    for label, text, color in rows:
        html_parts.append(
            f'<div style="display:flex;gap:10px;margin-bottom:4px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:11px;color:var(--tx2);line-height:1.5">'
            f'<span style="flex:0 0 34px;font-weight:700;color:{color}">{label}</span>'
            f'<span>{text}</span></div>'
        )
    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)


def build_leaderboard_df(
    scores: Dict, per_run: Dict, mm: Dict, level_range: tuple = (1, 7)
) -> pd.DataFrame:
    """Build the exact leaderboard data used by both dashboard and report export."""
    lvl_min, lvl_max = level_range
    rows = []
    for method_name, metrics in scores.items():
        ik = mm.get(method_name)
        entries = filter_by_level(per_run.get(ik, {}), lvl_min, lvl_max) if ik else {}
        if entries:
            ktc_val = float(np.mean([e.get('ktc_score', 0.0) for e in entries.values()]))
        else:
            ktc_val = metrics.get('KTC score', metrics.get('ktc_score', 0))
        comp = calculate_composite_score({'ktc_score': ktc_val})
        row = {
            'Method': method_name,
            'Composite Score': comp,
            'Grade': letter_grade(comp),
            'KTC Score': ktc_val,
        }
        for label, key in METRIC_SPECS:
            if label != "KTC Score":
                row[label] = metrics.get(key, 0)
        rows.append(row)
    rows.sort(key=lambda x: x['Composite Score'], reverse=True)
    return pd.DataFrame(rows)


def build_leaderboard_figure(scores: Dict, df: pd.DataFrame) -> go.Figure:
    """Build the exact leaderboard Plotly figure used on dashboard and in report PNG."""
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row['Method'], x=[row['Composite Score']], y=[row['Method']],
            orientation='h',
            marker_color=get_method_color(row['Method']),
            text=f"{row['Composite Score']:.1f} ({row['Grade']})", textposition='outside',
            textfont=dict(family="JetBrains Mono", size=9, color="#1f2328"),
            hovertemplate=(
                f"<b>{row['Method']}</b><br>Score: {row['Composite Score']:.1f} ({row['Grade']})<br>"
                f"KTC: {row['KTC Score']:.4f}<br><extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title="Score (0-100)", yaxis_title="Method", xaxis_range=[0, 115],
        showlegend=False, height=max(320, 54 * len(df)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f6f8fa',
        font=dict(family="JetBrains Mono,monospace", color="#848d97", size=9),
        xaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickfont=dict(size=9)),
        margin=dict(l=150, r=50, t=20, b=38),
    )
    fig.update_yaxes(autorange="reversed")
    return fig
