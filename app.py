"""
app.py — Streamlit Dashboard for EIT Reconstruction Analysis
MODERNIZED DESIGN matching eit_final_dashboard.html

Five Views:
1. Leaderboard table with composite scores
2. Degradation curve with method selector
3. Side-by-side comparison (any two methods + any sample)
4. Failure gallery (worst 3 samples per method)
5. Per-metric radar chart

Features:
- Interactive composite weight editor (5 sliders for metric tiers)
- Real-time leaderboard updates based on weight changes
- Loads from scores.json and per_run_metrics.json
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
from PIL import Image

# =========================================================
# MODERN STYLING - Matching eit_final_dashboard.html
# =========================================================

MODERN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --sidebar: #0F172A;
    --sidebar2: #1E293B;
    --surface: #FFFFFF;
    --bg: #F1F5F9;
    --border: #E2E8F0;
    --text: #0F172A;
    --text2: #64748B;
    --text3: #94A3B8;
    --teal: #0F766E;
    --teal-light: #CCFBF1;
    --teal-mid: #14B8A6;
    --blue: #1D4ED8;
    --blue-light: #EFF6FF;
    --coral: #BE123C;
    --coral-light: #FFF1F2;
    --amber: #B45309;
    --amber-light: #FFFBEB;
    --green: #15803D;
    --green-light: #F0FDF4;
    --method-bp: #6366F1;
    --method-gn: #0EA5E9;
    --method-un: #10B981;
}

/* Main App Styling */
.stApp {
    background: var(--bg) !important;
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
}

/* Main Content Area */
.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

/* Headers - Matching eit_final_dashboard sizes */
h1 {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.2 !important;
}

h2 {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    border-bottom: 2px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1rem !important;
    margin-top: 1.5rem !important;
    line-height: 1.2 !important;
}

h3 {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.75rem !important;
    line-height: 1.2 !important;
}

/* Regular text */
p, .stMarkdown p {
    font-size: 13px !important;
    line-height: 1.5 !important;
    color: var(--text2) !important;
}

/* Strong text */
strong, b {
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* Lists */
ul, ol {
    font-size: 13px !important;
    color: var(--text2) !important;
    line-height: 1.5 !important;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    background: var(--bg) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    color: var(--text) !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    padding-top: 1rem !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* Sidebar Logo/Title Area - Compact like eit_final_dashboard */
section[data-testid="stSidebar"] > div > div:first-child {
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--sidebar2) !important;
    margin-bottom: 0.75rem !important;
}

section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

/* Sidebar Headers - Compact sizing like eit_final_dashboard */
section[data-testid="stSidebar"] h1 {
    color: #F8FAFC !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    margin-bottom: 0.25rem !important;
    line-height: 1.2 !important;
}

section[data-testid="stSidebar"] h2 {
    color: #F8FAFC !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    margin-top: 0.75rem !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
    line-height: 1.2 !important;
}

section[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    margin-bottom: 0.4rem !important;
    margin-top: 0.6rem !important;
    line-height: 1.2 !important;
}

/* Sidebar Text and Captions */
section[data-testid="stSidebar"] .stMarkdown {
    color: #94A3B8 !important;
    font-size: 10px !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 10px !important;
    margin-bottom: 0.4rem !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] .stMarkdown strong {
    font-size: 10px !important;
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Sidebar Captions - Very compact */
section[data-testid="stSidebar"] .element-container .stMarkdown:has(> p) p {
    font-size: 9px !important;
    color: #64748B !important;
    margin-top: 0.15rem !important;
    margin-bottom: 0.15rem !important;
    line-height: 1.3 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar Buttons - Compact */
section[data-testid="stSidebar"] button {
    background: transparent !important;
    border: 1px solid var(--sidebar2) !important;
    color: #94A3B8 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    line-height: 1.2 !important;
    margin: 2px 0 !important;
}

section[data-testid="stSidebar"] button:hover {
    background: var(--sidebar2) !important;
    color: #E2E8F0 !important;
}

/* Sidebar Sliders - Compact */
section[data-testid="stSidebar"] .stSlider {
    padding: 0.3rem 0 !important;
    margin-bottom: 0.4rem !important;
}

section[data-testid="stSidebar"] .stSlider label {
    font-size: 10px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    margin-bottom: 0.25rem !important;
    line-height: 1.2 !important;
}

section[data-testid="stSidebar"] .stSlider label div {
    font-size: 10px !important;
}

/* Sidebar Inputs - Compact */
section[data-testid="stSidebar"] input[type="text"] {
    font-size: 11px !important;
    padding: 6px 8px !important;
    background: var(--sidebar2) !important;
    border: 1px solid #334155 !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
}

section[data-testid="stSidebar"] input[type="text"]::placeholder {
    color: #64748B !important;
    font-size: 10px !important;
}

/* Sidebar Labels */
section[data-testid="stSidebar"] label {
    font-size: 10px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
}

/* Sidebar Dividers */
section[data-testid="stSidebar"] hr {
    margin: 0.75rem 0 !important;
    border-color: var(--sidebar2) !important;
}

/* Sidebar Success/Warning/Info messages */
section[data-testid="stSidebar"] .stAlert {
    font-size: 10px !important;
    padding: 6px 10px !important;
    margin: 0.4rem 0 !important;
    line-height: 1.3 !important;
}

/* Sidebar Progress bars */
section[data-testid="stSidebar"] .stProgress {
    height: 4px !important;
    margin: 0.3rem 0 !important;
}

section[data-testid="stSidebar"] .stProgress > div > div > div {
    background: var(--teal) !important;
}

/* Main Content Cards */
.element-container {
    background: transparent !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1.2 !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    color: var(--text2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-weight: 600 !important;
    margin-bottom: 0.25rem !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin-top: 0.25rem !important;
}

/* Tabs - Compact like eit_final_dashboard */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
    padding: 8px 16px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    line-height: 1.2 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* DataFrames/Tables - Compact like eit_final_dashboard */
.dataframe {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.dataframe thead th {
    background: var(--text) !important;
    color: white !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 10px !important;
    letter-spacing: 0.05em !important;
    padding: 8px 10px !important;
    line-height: 1.2 !important;
}

.dataframe tbody td {
    padding: 8px 10px !important;
    border-bottom: 1px solid #F8FAFC !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
}

.dataframe tbody tr:hover {
    background: #F8FAFC !important;
}

.dataframe tbody tr:last-child td {
    border-bottom: none !important;
}

/* Select Boxes - Compact */
.stSelectbox > div > div {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    padding: 6px 10px !important;
    min-height: 36px !important;
}

.stSelectbox label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    margin-bottom: 0.4rem !important;
}

/* Multiselect - Compact */
.stMultiSelect > div > div {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 4px 8px !important;
}

.stMultiSelect span {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11px !important;
}

.stMultiSelect label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    margin-bottom: 0.4rem !important;
}

/* Sidebar Select Boxes - Compact */
section[data-testid="stSidebar"] .stSelectbox {
    margin-bottom: 0.5rem !important;
}

section[data-testid="stSidebar"] .stSelectbox > label {
    font-size: 10px !important;
    color: #94A3B8 !important;
    margin-bottom: 0.25rem !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--sidebar2) !important;
    border: 1px solid #334155 !important;
    font-size: 11px !important;
    padding: 6px 8px !important;
    min-height: 32px !important;
}

section[data-testid="stSidebar"] .stSelectbox option {
    font-size: 11px !important;
}

/* Info/Warning/Error boxes - Compact */
.stAlert {
    border-radius: 8px !important;
    border-left: 4px solid var(--teal) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    padding: 10px 12px !important;
    line-height: 1.4 !important;
}

.stAlert p {
    font-size: 12px !important;
    margin: 0 !important;
}

/* Progress bars */
.stProgress > div > div > div {
    background: var(--teal) !important;
    border-radius: 4px !important;
}

/* Expander - Compact */
.streamlit-expanderHeader {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}

.streamlit-expanderContent {
    font-size: 12px !important;
    line-height: 1.5 !important;
}

/* Captions and small text */
.stCaption, small {
    font-size: 11px !important;
    color: var(--text3) !important;
    line-height: 1.4 !important;
}

/* Image captions */
.stImage > div > div {
    font-size: 11px !important;
    color: var(--text2) !important;
    text-align: center !important;
    margin-top: 0.5rem !important;
}

/* Columns - Tighter spacing */
.row-widget.stHorizontal {
    gap: 12px !important;
}

[data-testid="column"] {
    padding: 0 6px !important;
}

/* Markdown spacing */
.stMarkdown {
    margin-bottom: 0.75rem !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    margin-top: 1rem !important;
}

/* Success/Warning/Info messages */
.stSuccess, .stWarning, .stInfo, .stError {
    font-size: 12px !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    line-height: 1.4 !important;
}

/* Download button */
.stDownloadButton > button {
    font-size: 11px !important;
    padding: 6px 14px !important;
}

/* Buttons - Compact */
.stButton > button {
    background: var(--teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    transition: all 0.15s !important;
    line-height: 1.4 !important;
}

.stButton > button:hover {
    background: var(--teal-mid) !important;
    transform: translateY(-1px) !important;
}

/* Custom metric cards - matching eit_final_dashboard */
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.metric-card .metric-value {
    font-size: 24px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text);
    margin-bottom: 2px;
    line-height: 1.2;
}

.metric-card .metric-label {
    font-size: 11px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.metric-card .metric-sub {
    font-size: 10px;
    color: var(--text3);
    margin-top: 4px;
    line-height: 1.4;
}

/* Badge styles */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
}

.badge-success {
    background: var(--green-light);
    color: var(--green);
}

.badge-warning {
    background: var(--amber-light);
    color: var(--amber);
}

.badge-danger {
    background: var(--coral-light);
    color: var(--coral);
}

.badge-info {
    background: var(--blue-light);
    color: var(--blue);
}

.badge-teal {
    background: var(--teal-light);
    color: var(--teal);
}

/* Grade badges */
.grade-A { color: var(--green) !important; font-weight: 700 !important; }
.grade-B { color: var(--blue) !important; font-weight: 700 !important; }
.grade-C { color: var(--amber) !important; font-weight: 700 !important; }
.grade-D { color: var(--coral) !important; font-weight: 700 !important; }

/* Status pills */
.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.status-live {
    background: var(--green-light);
    color: var(--green);
}
</style>
"""

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EIT Reconstruction Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern CSS
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# Color scheme matching the dashboard
COLORS = {
    'water': '#1a3a5c',
    'resistive': '#D85A30',
    'conductive': '#1D9E75',
    'primary': '#0F172A',
    'success': '#1D9E75',
    'warning': '#F5A623',
    'danger': '#D85A30',
    'teal': '#0F766E',
    'method_bp': '#6366F1',
    'method_gn': '#0EA5E9',
    'method_un': '#10B981'
}

GRADE_COLORS = {
    'A': '#15803D',  # green
    'B': '#1D4ED8',  # blue
    'C': '#B45309',  # amber
    'D': '#BE123C'   # coral/red
}

ALL_LEVELS = [1, 2, 3, 4, 5, 6, 7]

# Color map for segmentation visualization
COLORMAP = ListedColormap([COLORS['water'], COLORS['resistive'], COLORS['conductive']])

# =========================================================
# DATA LOADING (UNCHANGED)
# =========================================================

def create_method_mapping(scores: Dict, per_run: Dict) -> Dict[str, str]:
    """
    Create mapping between display names (from scores.json) and internal keys (from per_run_metrics.json).
    
    Examples:
    - "Back-projection (avg across 4 real samples)" -> "back_projection"
    - "Mock baseline (avg across 4 real samples)" -> "mock_baseline"
    - "Gauss-Newton (avg across 4 real samples)" -> "gauss_newton"
    """
    mapping = {}
    
    # Extract base names from display names
    for display_name in scores.keys():
        # Try to match with per_run keys
        display_lower = display_name.lower()
        
        for internal_key in per_run.keys():
            # Check if internal key is contained in display name
            if internal_key.replace('_', '-') in display_lower or internal_key.replace('_', ' ') in display_lower:
                mapping[display_name] = internal_key
                break
            # Also try matching first word
            elif display_name.split()[0].lower().replace('-', '_') == internal_key.split('_')[0]:
                mapping[display_name] = internal_key
                break
    
    return mapping

@st.cache_data
def load_data(scores_path: str = "scores.json", 
              per_run_path: str = "outputs/per_run_metrics.json") -> Tuple[Dict, Dict, Dict]:
    """Load scores and per-run metrics from JSON files."""
    
    # Try scores.json in current directory or outputs/
    scores = {}
    scores_file = None
    if Path(scores_path).exists():
        scores_file = Path(scores_path)
    elif Path("outputs/scores.json").exists():
        scores_file = Path("outputs/scores.json")
    
    if scores_file:
        with open(scores_file, 'r') as f:
            scores = json.load(f)
        st.sidebar.markdown(f'<p style="font-size: 9px; color: #64748B; font-family: \'JetBrains Mono\', monospace; margin: 0.15rem 0;">📄 {scores_file.name}</p>', unsafe_allow_html=True)
    
    # Try per_run_metrics.json in outputs/
    per_run = {}
    per_run_file = None
    if Path(per_run_path).exists():
        per_run_file = Path(per_run_path)
    
    if per_run_file:
        with open(per_run_file, 'r') as f:
            per_run = json.load(f)
        st.sidebar.markdown(f'<p style="font-size: 9px; color: #64748B; font-family: \'JetBrains Mono\', monospace; margin: 0.15rem 0;">📄 {per_run_file.name}</p>', unsafe_allow_html=True)
    
    # Create mapping
    method_mapping = create_method_mapping(scores, per_run)
    
    return scores, per_run, method_mapping

@st.cache_data
def load_images_for_sample(sample_id: str, level: int = 1, outputs_dir: str = "outputs") -> Dict[str, Image.Image]:
    """Load all method images for a specific sample and level."""
    images = {}
    outputs_path = Path(outputs_dir)

    sample_dir = outputs_path / "reconstructions" / f"level_{level}" / f"sample_{sample_id}"
    if sample_dir.exists():
        for img_file in sample_dir.glob("*.png"):
            # Use the filename (without .png) as the method key
            method_key = img_file.stem  # e.g., 'back_projection', 'mock_baseline'
            images[method_key] = Image.open(img_file)
    
    # Also look for error overlays
    error_dir = outputs_path / "error_overlays"
    if error_dir.exists():
        for img_file in error_dir.glob(f"*_sample_{sample_id}.png"):
            # Extract method name from filename (e.g., "back_projection_sample_1.png")
            method_key = img_file.stem.replace(f"_sample_{sample_id}", "")
            if method_key not in images:  # Don't overwrite reconstructions
                images[method_key] = Image.open(img_file)
    
    return images

@st.cache_data
def load_comparison_panel(sample_id: str, outputs_dir: str = "outputs") -> Image.Image:
    """Load the multi-method comparison panel for a sample."""
    outputs_path = Path(outputs_dir)
    
    # Look for comparison panel
    comparison_file = outputs_path / "comparison_panels" / f"sample_{sample_id}.png"
    if comparison_file.exists():
        return Image.open(comparison_file)
    
    # Try the main variant
    comparison_main = outputs_path / "comparison_panels" / f"sample_{sample_id}_main.png"
    if comparison_main.exists():
        return Image.open(comparison_main)
    
    return None

# =========================================================
# COMPOSITE SCORE CALCULATION (UNCHANGED)
# =========================================================

def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted composite score from metrics.
    
    Metric tiers:
    - Tier 1: KTC Score (primary benchmark metric)
    - Tier 2: Dice coefficients (overlap metrics)
    - Tier 3: IoU scores (intersection over union)
    - Tier 4: Hausdorff distance (boundary accuracy)
    - Tier 5: Overall balance
    
    Returns score in range 0-100
    """
    
    # Extract metrics with defaults
    ktc = metrics.get('KTC score', metrics.get('ktc_score', 0))
    dice_r = metrics.get('Dice (resistive)', metrics.get('dice_resistive', 0))
    dice_c = metrics.get('Dice (conductive)', metrics.get('dice_conductive', 0))
    iou_r = metrics.get('IoU (resistive)', metrics.get('iou_resistive', 0))
    iou_c = metrics.get('IoU (conductive)', metrics.get('iou_conductive', 0))
    
    # Hausdorff distance (lower is better, so invert)
    hd95_r = metrics.get('hd95_resistive', 0)
    hd95_c = metrics.get('hd95_conductive', 0)
    # Normalize HD95 to 0-1 range (assuming max reasonable value is 100 pixels)
    hd95_r_norm = max(0, 1 - (hd95_r / 100))
    hd95_c_norm = max(0, 1 - (hd95_c / 100))
    
    # Calculate tier scores
    tier1 = (1 - ktc) * 100  # KTC is error, so invert
    tier2 = ((dice_r + dice_c) / 2) * 100
    tier3 = ((iou_r + iou_c) / 2) * 100
    tier4 = ((hd95_r_norm + hd95_c_norm) / 2) * 100
    tier5 = tier1  # Overall balance - use KTC as baseline
    
    # Weighted combination
    composite = (
        weights['tier1'] * tier1 +
        weights['tier2'] * tier2 +
        weights['tier3'] * tier3 +
        weights['tier4'] * tier4 +
        weights['tier5'] * tier5
    ) / sum(weights.values())
    
    return composite

def letter_grade(score: float) -> str:
    """Convert composite score to letter grade."""
    if score >= 85:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 55:
        return 'C'
    else:
        return 'D'

# =========================================================
# ADD METHOD AT RUNTIME (UNCHANGED)
# =========================================================

def sidebar_add_method():
    """Sidebar section to register a new method name at runtime."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="margin-bottom: 0.5rem;">
        <div style="font-size: 9px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.5rem;">
            ADD METHOD
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.caption("Register a method now; connect backend later.")

    if 'custom_methods' not in st.session_state:
        st.session_state.custom_methods = []

    new_name = st.sidebar.text_input("Method name:", key="new_method_input",
                                     placeholder="e.g. gauss_newton_v2", label_visibility="collapsed")
    if st.sidebar.button("➕ Add Method", use_container_width=True) and new_name.strip():
        name = new_name.strip()
        if name not in st.session_state.custom_methods:
            st.session_state.custom_methods.append(name)
            st.sidebar.success(f"✓ Added: {name}")
        else:
            st.sidebar.warning("Already in list")

    if st.session_state.custom_methods:
        st.sidebar.markdown("**Custom methods:**")
        to_remove = None
        for m in st.session_state.custom_methods:
            col_a, col_b = st.sidebar.columns([4, 1])
            col_a.caption(f"• {m}  *(pending)*")
            if col_b.button("✕", key=f"rm_{m}"):
                to_remove = m
        if to_remove:
            st.session_state.custom_methods.remove(to_remove)
            st.rerun()


def all_methods(scores: Dict) -> List[str]:
    """Merge scores-file methods with any runtime-added custom methods."""
    loaded = list(scores.keys())
    custom = st.session_state.get('custom_methods', [])
    return loaded + [m for m in custom if m not in loaded]


# =========================================================
# VIEW 1: LEADERBOARD WITH INTERACTIVE WEIGHTS
# =========================================================

def view_leaderboard(scores: Dict, per_run: Dict):
    """Interactive leaderboard with composite weight editor."""
    
    st.markdown("## 🏆 Leaderboard")
    
    # Sidebar: Weight Editor
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="margin-bottom: 0.75rem;">
        <div style="font-size: 9px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.5rem;">
            COMPOSITE WEIGHTS
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('<p style="font-size: 10px; color: #94A3B8; margin-bottom: 0.75rem; line-height: 1.4;">Adjust weights for each metric tier:</p>', unsafe_allow_html=True)
    
    # Initialize session state for weights if not exists
    if 'weights' not in st.session_state:
        st.session_state.weights = {
            'tier1': 0.40,  # KTC Score
            'tier2': 0.25,  # Dice
            'tier3': 0.20,  # IoU
            'tier4': 0.10,  # HD95
            'tier5': 0.05   # Balance
        }
    
    # Weight sliders
    weights = {}
    weights['tier1'] = st.sidebar.slider(
        "Tier 1: KTC Score",
        0.0, 1.0, st.session_state.weights['tier1'], 0.05,
        help="KTC benchmark score - lower is better"
    )
    weights['tier2'] = st.sidebar.slider(
        "Tier 2: Dice Coefficients",
        0.0, 1.0, st.session_state.weights['tier2'], 0.05,
        help="Overlap metrics for resistive/conductive regions"
    )
    weights['tier3'] = st.sidebar.slider(
        "Tier 3: IoU Scores",
        0.0, 1.0, st.session_state.weights['tier3'], 0.05,
        help="Intersection over Union metrics"
    )
    weights['tier4'] = st.sidebar.slider(
        "Tier 4: Hausdorff Distance",
        0.0, 1.0, st.session_state.weights['tier4'], 0.05,
        help="Boundary accuracy (95th percentile)"
    )
    weights['tier5'] = st.sidebar.slider(
        "Tier 5: Overall Balance",
        0.0, 1.0, st.session_state.weights['tier5'], 0.05,
        help="Balancing factor for overall performance"
    )
    
    # Normalize button
    if st.sidebar.button("⚖️ Normalize", use_container_width=True):
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
            st.session_state.weights = weights
            st.rerun()
    
    # Reset button
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.weights = {
            'tier1': 0.40, 'tier2': 0.25, 'tier3': 0.20, 'tier4': 0.10, 'tier5': 0.05
        }
        st.rerun()
    
    # Show weight distribution
    st.sidebar.markdown("""
    <div style="margin-top: 0.75rem;">
        <div style="font-size: 9px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.5rem;">
            CURRENT DISTRIBUTION
        </div>
    </div>
    """, unsafe_allow_html=True)
    total_weight = sum(weights.values())
    for tier, weight in weights.items():
        pct = (weight / total_weight * 100) if total_weight > 0 else 0
        st.sidebar.progress(weight)
        st.sidebar.caption(f"{tier}: {pct:.1f}%")
    
    # Calculate composite scores for all methods
    leaderboard_data = []
    
    # Define distinct colors for each method - these stay consistent
    method_color_palette = [
        '#6366F1',  # Indigo
        '#0EA5E9',  # Cyan
        '#10B981',  # Green
        '#F59E0B',  # Amber
        '#EC4899',  # Pink
        '#8B5CF6',  # Purple
        '#14B8A6',  # Teal
        '#F97316',  # Orange
        '#06B6D4',  # Sky
        '#84CC16',  # Lime
        '#EF4444',  # Red
        '#6366F1',  # Indigo (repeat for more methods)
    ]
    
    # Create consistent color mapping for methods
    if 'method_colors' not in st.session_state:
        st.session_state.method_colors = {}
    
    for method_name, metrics in scores.items():
        # Assign color if method doesn't have one yet
        if method_name not in st.session_state.method_colors:
            color_index = len(st.session_state.method_colors) % len(method_color_palette)
            st.session_state.method_colors[method_name] = method_color_palette[color_index]
        
        composite = calculate_composite_score(metrics, weights)
        grade = letter_grade(composite)
        
        leaderboard_data.append({
            'Method': method_name,
            'Composite Score': composite,
            'Grade': grade,
            'Color': st.session_state.method_colors[method_name],
            'KTC Score': metrics.get('KTC score', metrics.get('ktc_score', 0)),
            'Dice (R)': metrics.get('Dice (resistive)', metrics.get('dice_resistive', 0)),
            'Dice (C)': metrics.get('Dice (conductive)', metrics.get('dice_conductive', 0)),
            'IoU (R)': metrics.get('IoU (resistive)', metrics.get('iou_resistive', 0)),
            'IoU (C)': metrics.get('IoU (conductive)', metrics.get('iou_conductive', 0)),
        })
    
    # Sort by composite score (descending)
    leaderboard_data.sort(key=lambda x: x['Composite Score'], reverse=True)
    df = pd.DataFrame(leaderboard_data)
    
    # Display metrics in modern card style
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.iloc[0]['Composite Score']:.1f}</div>
            <div class="metric-label">Top Score</div>
            <div class="metric-sub">{df.iloc[0]['Method']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['Composite Score'].mean():.1f}</div>
            <div class="metric-label">Average Score</div>
            <div class="metric-sub">±{df['Composite Score'].std():.1f} std</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        grade_counts = df['Grade'].value_counts()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Methods Evaluated</div>
            <div class="metric-sub">{grade_counts.get('A', 0)}A, {grade_counts.get('B', 0)}B, {grade_counts.get('C', 0)}C, {grade_counts.get('D', 0)}D</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_ktc = df['KTC Score'].min()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_ktc:.4f}</div>
            <div class="metric-label">Best KTC Score</div>
            <div class="metric-sub">Lower is better</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive bar chart - Each method gets unique persistent color
    fig = go.Figure()
    
    # Assign unique colors to each method (colors stay with method, not position)
    method_color_palette = [
        '#6366F1',  # Indigo
        '#0EA5E9',  # Cyan
        '#10B981',  # Green
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#8B5CF6',  # Purple
        '#EC4899',  # Pink
        '#14B8A6',  # Teal
        '#F97316',  # Orange
        '#06B6D4',  # Sky
        '#84CC16',  # Lime
        '#A855F7',  # Violet
    ]
    
    # Create a persistent color mapping for all methods
    all_method_names = sorted(scores.keys())  # Use original scores dict for consistency
    method_colors = {}
    for idx, method in enumerate(all_method_names):
        method_colors[method] = method_color_palette[idx % len(method_color_palette)]
    
    # Plot bars with persistent method colors
    for idx, row in df.iterrows():
        method_color = method_colors.get(row['Method'], '#64748B')  # Fallback to gray
        
        fig.add_trace(go.Bar(
            name=row['Method'],
            x=[row['Method']],
            y=[row['Composite Score']],
            marker_color=method_color,
            text=f"{row['Composite Score']:.1f} ({row['Grade']})",
            textposition='outside',
            hovertemplate=(
                f"<b>{row['Method']}</b><br>"
                f"Composite: {row['Composite Score']:.1f}<br>"
                f"Grade: {row['Grade']}<br>"
                f"KTC: {row['KTC Score']:.4f}<br>"
                f"Dice (R/C): {row['Dice (R)']:.4f} / {row['Dice (C)']:.4f}<br>"
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title="Method Rankings by Composite Score",
        xaxis_title="Method",
        yaxis_title="Composite Score (0-100)",
        yaxis_range=[0, 105],
        showlegend=False,
        height=400,
        template="plotly_white",
        font=dict(family="Space Grotesk, sans-serif")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("### 📊 Detailed Metrics")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['Composite Score'] = display_df['Composite Score'].apply(lambda x: f"{x:.2f}")
    display_df['KTC Score'] = display_df['KTC Score'].apply(lambda x: f"{x:.4f}")
    display_df['Dice (R)'] = display_df['Dice (R)'].apply(lambda x: f"{x:.4f}")
    display_df['Dice (C)'] = display_df['Dice (C)'].apply(lambda x: f"{x:.4f}")
    display_df['IoU (R)'] = display_df['IoU (R)'].apply(lambda x: f"{x:.4f}")
    display_df['IoU (C)'] = display_df['IoU (C)'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# =========================================================
# VIEW 2: DEGRADATION CURVE
# =========================================================

def view_degradation_curve(scores: Dict, per_run: Dict, method_mapping: Dict):
    """Degradation curve showing performance across difficulty levels."""
    
    st.markdown("## 📉 Degradation Curve")
    st.markdown("Performance trends across different samples/difficulty levels")
    
    # Extract method keys from per_run data
    if not per_run:
        st.warning("No per-run metrics available. Run the benchmark first.")
        return
    
    display_methods = all_methods(scores)

    col_lvl, col_methods = st.columns([1, 3])
    with col_lvl:
        selected_level = st.selectbox("Level:", ALL_LEVELS, index=0, key="deg_level")
    with col_methods:
        selected_display_methods = st.multiselect(
            "Select methods to display:",
            display_methods,
            default=[m for m in display_methods[:3] if m not in st.session_state.get('custom_methods', [])]
        )
    
    if not selected_display_methods:
        st.info("Please select at least one method to display.")
        return
    
    # Prepare data
    fig = go.Figure()
    
    colors_palette = [COLORS['method_bp'], COLORS['method_gn'], COLORS['method_un'], 
                      '#9B59B6', '#E74C3C', '#1ABC9C', '#F39C12']
    
    for idx, display_method in enumerate(selected_display_methods):
        # Get internal key from mapping
        internal_key = method_mapping.get(display_method)
        if not internal_key or internal_key not in per_run:
            continue
        
        samples = per_run[internal_key]
        sample_ids = sorted(samples.keys())
        ktc_scores = [samples[sid]['ktc_score'] for sid in sample_ids]
        
        # Convert sample IDs to numeric for plotting
        x_values = [int(sid) if sid.isdigit() else idx for idx, sid in enumerate(sample_ids, 1)]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=ktc_scores,
            mode='lines+markers',
            name=display_method,
            line=dict(width=3, color=colors_palette[idx % len(colors_palette)]),
            marker=dict(size=10),
            hovertemplate=(
                f"<b>{display_method}</b><br>"
                "Sample: %{x}<br>"
                "KTC Score: %{y:.4f}<br>"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=f"KTC Score Degradation Across Samples — Level {selected_level}",
        xaxis_title="Sample ID",
        yaxis_title="KTC Score (lower is better)",
        height=500,
        template="plotly_white",
        hovermode='x unified',
        font=dict(family="Space Grotesk, sans-serif")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.markdown("### 📈 Performance Statistics")
    
    stats_data = []
    for display_method in selected_display_methods:
        internal_key = method_mapping.get(display_method)
        if not internal_key or internal_key not in per_run:
            continue
        
        samples = per_run[internal_key]
        ktc_values = [s['ktc_score'] for s in samples.values()]
        
        stats_data.append({
            'Method': display_method,
            'Mean KTC': np.mean(ktc_values),
            'Std Dev': np.std(ktc_values),
            'Min': np.min(ktc_values),
            'Max': np.max(ktc_values),
            'Range': np.max(ktc_values) - np.min(ktc_values)
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.round(4)
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# =========================================================
# VIEW 3: SIDE-BY-SIDE COMPARISON
# =========================================================

def view_comparison(scores: Dict, per_run: Dict, method_mapping: Dict):
    """Side-by-side comparison of any two methods on any sample."""
    
    st.markdown("## 🔍 Side-by-Side Comparison")
    
    if not per_run:
        st.warning("No per-run metrics available.")
        return
    
    display_methods = all_methods(scores)

    first_internal = list(per_run.keys())[0] if per_run else None
    samples = list(per_run[first_internal].keys()) if first_internal else []

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        method1_display = st.selectbox("Method 1:", display_methods, index=0)

    with col2:
        method2_display = st.selectbox("Method 2:", display_methods,
                                       index=1 if len(display_methods) > 1 else 0)

    with col3:
        sample_id = st.selectbox("Sample:", samples)

    with col4:
        selected_level = st.selectbox("Level:", ALL_LEVELS, index=0)
    
    if method1_display and method2_display and sample_id:
        method1_internal = method_mapping.get(method1_display)
        method2_internal = method_mapping.get(method2_display)

        # Custom (pending) methods have no backend data yet
        m1_pending = method1_display in st.session_state.get('custom_methods', [])
        m2_pending = method2_display in st.session_state.get('custom_methods', [])
        if m1_pending:
            st.info(f"**{method1_display}** is a custom method — connect its backend to see metrics.")
        if m2_pending:
            st.info(f"**{method2_display}** is a custom method — connect its backend to see metrics.")

        metrics1 = per_run.get(method1_internal, {}).get(sample_id, {}) if not m1_pending else {}
        metrics2 = per_run.get(method2_internal, {}).get(sample_id, {}) if not m2_pending else {}
        
        # Display metrics comparison
        st.markdown("### 📊 Metric Comparison")
        
        comparison_data = []
        for key in metrics1.keys():
            comparison_data.append({
                'Metric': key.replace('_', ' ').title(),
                method1_display: metrics1.get(key, 0),
                method2_display: metrics2.get(key, 0),
                'Difference': abs(metrics1.get(key, 0) - metrics2.get(key, 0))
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Format numbers
        for col in [method1_display, method2_display, 'Difference']:
            comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Radar chart
        st.markdown("### 📡 Radar Chart")
        
        # Select key metrics for radar chart
        radar_metrics = ['ktc_score', 'dice_resistive', 'dice_conductive', 
                        'iou_resistive', 'iou_conductive']
        
        categories = [m.replace('_', ' ').title() for m in radar_metrics]
        
        fig = go.Figure()
        
        # Method 1
        values1 = [metrics1.get(m, 0) for m in radar_metrics]
        values1.append(values1[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values1,
            theta=categories + [categories[0]],
            fill='toself',
            name=method1_display,
            line_color=COLORS['method_bp']
        ))
        
        # Method 2
        values2 = [metrics2.get(m, 0) for m in radar_metrics]
        values2.append(values2[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values2,
            theta=categories + [categories[0]],
            fill='toself',
            name=method2_display,
            line_color=COLORS['method_gn']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=500,
            font=dict(family="Space Grotesk, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Try to load comparison panel first
        st.markdown("### 🖼️ Visual Comparison")
        
        comparison_panel = load_comparison_panel(sample_id)
        if comparison_panel:
            st.markdown(f"**All Methods - Sample {sample_id}**")
            st.image(comparison_panel, use_container_width=True)
            st.markdown("---")
        
        images = load_images_for_sample(sample_id, level=selected_level)
        
        if images:
            img_col1, img_col2 = st.columns(2)
            
            # Find matching images for the methods
            method1_img = None
            method2_img = None
            
            for img_key, img in images.items():
                if method1_internal and method1_internal.lower() in img_key.lower():
                    method1_img = img
                if method2_internal and method2_internal.lower() in img_key.lower():
                    method2_img = img
            
            with img_col1:
                st.markdown(f"**{method1_display}**")
                if method1_img:
                    st.image(method1_img, use_container_width=True)
                else:
                    st.info(f"Image not found for {method1_display}")
            
            with img_col2:
                st.markdown(f"**{method2_display}**")
                if method2_img:
                    st.image(method2_img, use_container_width=True)
                else:
                    st.info(f"Image not found for {method2_display}")
        elif not comparison_panel:
            st.info("No visualization images found. Run example_usage.py to generate images.")

# =========================================================
# VIEW 4: FAILURE GALLERY
# =========================================================

def view_failure_gallery(scores: Dict, per_run: Dict, method_mapping: Dict):
    """Gallery showing worst 3 samples per method."""
    
    st.markdown("## ⚠️ Failure Gallery")
    st.markdown("Worst performing samples for each method (highest KTC scores)")
    
    if not per_run:
        st.warning("No per-run metrics available.")
        return
    
    for display_method in scores.keys():
        st.markdown(f"### 🔴 {display_method}")
        
        # Get internal key
        internal_key = method_mapping.get(display_method)
        if not internal_key or internal_key not in per_run:
            st.info(f"No per-run data available for {display_method}")
            continue
        
        samples = per_run[internal_key]
        
        # Sort samples by KTC score (descending - worst first)
        sorted_samples = sorted(
            samples.items(),
            key=lambda x: x[1]['ktc_score'],
            reverse=True
        )
        
        # Take worst 3
        worst_samples = sorted_samples[:3]
        
        # Display in columns
        cols = st.columns(3)
        
        for idx, (sample_id, metrics) in enumerate(worst_samples):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sample {sample_id}</div>
                    <div class="metric-value" style="font-size: 20px;">{metrics['ktc_score']:.4f}</div>
                    <div class="metric-sub">
                        Dice (R): {metrics.get('dice_resistive', 0):.4f}<br>
                        Dice (C): {metrics.get('dice_conductive', 0):.4f}<br>
                        IoU (R): {metrics.get('iou_resistive', 0):.4f}<br>
                        IoU (C): {metrics.get('iou_conductive', 0):.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Try to load image from reconstructions directory
                sample_dir = Path("outputs") / "reconstructions" / "level_1" / f"sample_{sample_id}"
                img_file = sample_dir / f"{internal_key}.png"
                
                if img_file.exists():
                    st.image(Image.open(img_file), use_container_width=True)
                else:
                    # Try error overlay as fallback
                    error_file = Path("outputs") / "error_overlays" / f"{internal_key}_sample_{sample_id}.png"
                    if error_file.exists():
                        st.image(Image.open(error_file), use_container_width=True)
                    else:
                        st.caption("🖼️ Image not available")
        
        st.markdown("---")

# =========================================================
# VIEW 5: PER-METRIC RADAR CHART
# =========================================================

def view_radar_chart(scores: Dict, per_run: Dict):
    """Comprehensive radar chart for all methods across all metrics."""
    
    st.markdown("## 📡 Per-Metric Radar Analysis")
    st.markdown("Compare all methods across different metric dimensions")
    
    if not scores:
        st.warning("No scores available.")
        return
    
    # Select metrics to include
    st.markdown("### Select Metrics")
    
    available_metrics = set()
    for method_scores in scores.values():
        available_metrics.update(method_scores.keys())
    
    available_metrics = sorted(list(available_metrics))
    
    selected_metrics = st.multiselect(
        "Choose metrics to display:",
        available_metrics,
        default=available_metrics[:5] if len(available_metrics) >= 5 else available_metrics
    )
    
    if not selected_metrics:
        st.info("Please select at least one metric.")
        return
    
    # Prepare data
    fig = go.Figure()
    
    colors_palette = [COLORS['method_bp'], COLORS['method_gn'], COLORS['method_un'],
                      '#9B59B6', '#E74C3C', '#1ABC9C', '#F39C12']
    
    for idx, (method_name, metrics) in enumerate(scores.items()):
        # Extract values for selected metrics
        values = []
        for metric in selected_metrics:
            val = metrics.get(metric, 0)
            # Normalize KTC score (invert since lower is better)
            if 'ktc' in metric.lower():
                val = max(0, 1 - val)
            values.append(val)
        
        # Close the polygon
        values.append(values[0])
        categories = [m.replace('_', ' ').title() for m in selected_metrics]
        categories.append(categories[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=method_name,
            line_color=colors_palette[idx % len(colors_palette)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=600,
        title="Method Performance Across Selected Metrics",
        font=dict(family="Space Grotesk, sans-serif")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metric statistics
    st.markdown("### 📊 Metric Statistics")
    
    stats_data = []
    for metric in selected_metrics:
        metric_values = []
        for method_scores in scores.values():
            val = method_scores.get(metric, 0)
            # Normalize KTC score
            if 'ktc' in metric.lower():
                val = max(0, 1 - val)
            metric_values.append(val)
        
        stats_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Mean': np.mean(metric_values),
            'Std Dev': np.std(metric_values),
            'Min': np.min(metric_values),
            'Max': np.max(metric_values)
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.round(4)
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# =========================================================
# MAIN APP
# =========================================================

def main():
    """Main application entry point."""
    
    # Sidebar Title - Compact like eit_final_dashboard
    st.sidebar.markdown("""
    <div style="padding: 0 0 0.75rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 0.75rem;">
        <h1 style="font-size: 14px; font-weight: 700; color: #F8FAFC; letter-spacing: -0.3px; margin: 0;">
            EIT BENCHMARKING
        </h1>
        <p style="font-size: 10px; color: #475569; margin: 0.25rem 0 0 0; font-family: 'JetBrains Mono', monospace;">
            Real Data Analysis v1.0
        </p>
        <span style="display: inline-block; font-size: 9px; padding: 2px 7px; background: #0F766E22; border: 1px solid #0F766E44; color: #2DD4BF; border-radius: 20px; margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace;">
            LIVE DASHBOARD
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Header with modern badge
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="display: inline-block; margin-right: 1rem;">🔬 EIT Reconstruction Dashboard</h1>
        <span class="badge badge-teal">REAL DATA</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Interactive dashboard for analyzing Electrical Impedance Tomography (EIT) reconstruction methods.
    
    **Features:** 🏆 Interactive leaderboard • 📉 Performance degradation • 🔍 Method comparisons • ⚠️ Failure analysis • 📡 Radar charts
    """)
    
    st.markdown("---")
    
    sidebar_add_method()

    try:
        scores, per_run, method_mapping = load_data()
        
        if not scores and not per_run:
            st.error("❌ No data found! Please run the benchmark first to generate scores.json and per_run_metrics.json")
            st.info("Run: `python example_usage.py` to generate the required data files.")
            return
        
        # Data info in modern expandable card
        with st.expander("ℹ️ Dataset Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 20px;">{len(scores)}</div>
                    <div class="metric-label">Methods Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 20px;">{len(per_run.get(list(per_run.keys())[0], {})) if per_run else 0}</div>
                    <div class="metric-label">Total Samples</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 20px;">{sum(len(v) for v in per_run.values()) if per_run else 0}</div>
                    <div class="metric-label">Total Reconstructions</div>
                </div>
                """, unsafe_allow_html=True)
            
            if scores:
                st.markdown("**Available methods:**")
                for method in scores.keys():
                    st.markdown(f"  • {method}")
        
        # View tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Leaderboard",
            "📉 Degradation Curve",
            "🔍 Comparison",
            "⚠️ Failures",
            "📡 Radar Chart"
        ])
        
        with tab1:
            view_leaderboard(scores, per_run)
        
        with tab2:
            view_degradation_curve(scores, per_run, method_mapping)
        
        with tab3:
            view_comparison(scores, per_run, method_mapping)
        
        with tab4:
            view_failure_gallery(scores, per_run, method_mapping)
        
        with tab5:
            view_radar_chart(scores, per_run)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()