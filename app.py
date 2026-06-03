"""
app.py — EIT Reconstruction Dashboard
Layout: pixel-exact to approved white mockup
Data:   all original logic preserved unchanged
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

st.set_page_config(
    page_title="EIT Reconstruction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS — exact mockup spec
# =========================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── tokens ── */
:root{
  --bg:#f6f8fa; --sur:#ffffff; --bd:#d0d7de; --bd2:#b0bac5;
  --tx:#1f2328;  --tx2:#57606a; --tx3:#848d97;
  --grn:#1a7f37; --grn-bg:#dafbe1; --grn-bd:#a7f3c0;
  --red:#cf222e; --blu:#0969da; --amb:#9a6700; --pur:#8250df;
  --c1:#2da44e;  --c2:#8250df;  --c3:#0969da; --c4:#bf8700;
  --c5:#cf222e;  --c6:#1a7f37; --c7:#d4a72c; --c8:#0550ae;
}

/* ── base ── */
html,body,.stApp{background:var(--bg)!important;font-family:'Inter',system-ui,sans-serif!important;}
.main .block-container{padding:12px 18px 40px!important;max-width:100%!important;}

/* ── sidebar ── */
section[data-testid="stSidebar"]{background:var(--sur)!important;border-right:1px solid var(--bd)!important;}
section[data-testid="stSidebar"]>div:first-child{padding:14px 12px!important;}
section[data-testid="stSidebar"] *{font-family:'JetBrains Mono',monospace!important;color:var(--tx2)!important;}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  font-size:8px!important;font-weight:600!important;color:var(--tx3)!important;
  text-transform:uppercase!important;letter-spacing:.14em!important;
  border:none!important;padding:0!important;margin:11px 0 6px!important;line-height:1!important;}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown p{font-size:9px!important;margin:3px 0!important;line-height:1.4!important;}
section[data-testid="stSidebar"] label{font-size:8px!important;color:var(--tx3)!important;text-transform:uppercase!important;letter-spacing:.1em!important;}
section[data-testid="stSidebar"] input{font-size:9px!important;padding:5px 7px!important;background:var(--bg)!important;border:1px solid var(--bd)!important;color:var(--tx)!important;border-radius:5px!important;}
section[data-testid="stSidebar"] button{font-size:9px!important;padding:5px 0!important;width:100%!important;border:1px solid var(--grn-bd)!important;border-radius:5px!important;color:var(--grn)!important;background:transparent!important;text-align:center!important;}
section[data-testid="stSidebar"] button:hover{background:var(--grn-bg)!important;}
section[data-testid="stSidebar"] hr{border-color:var(--bd)!important;margin:11px 0!important;}
section[data-testid="stSidebar"] .stProgress>div>div>div{background:var(--grn)!important;}

/* ── topbar stripe/header ── */
.dash-header{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:12px 18px 11px;margin-bottom:10px;position:relative;overflow:hidden;}
.dash-header::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#2da44e,#0969da,#8250df);}
.dash-title{font-family:'Inter',sans-serif;font-size:13px;font-weight:500;color:var(--tx);line-height:1.2;margin-bottom:3px;}
.dash-sub{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--tx3);}

/* ── kpi cards ── */
.kpi-row{display:flex;gap:8px;margin-bottom:10px;}
.kpi{flex:1;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:8px 10px;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--kc,#2da44e);opacity:.8;}
.kpi-n{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:500;color:var(--tx);line-height:1;}
.kpi-l{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin-top:4px;}
.kpi-s{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--tx3);margin-top:2px;}

/* ── chips ── */
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:9px;}
.chip{display:inline-flex;align-items:center;gap:4px;font-size:9px;color:var(--tx2);background:var(--bg);border:1px solid var(--bd);padding:2px 8px 2px 5px;border-radius:20px;font-family:'JetBrains Mono',monospace;}
.chip-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;}

/* ── section label ── */
.slbl{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.13em;margin-bottom:8px;}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"]{background:var(--sur)!important;border:1px solid var(--bd)!important;border-radius:7px!important;padding:4px!important;gap:2px!important;margin-bottom:11px!important;}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;font-weight:500!important;color:var(--tx3)!important;background:transparent!important;border:none!important;border-radius:5px 5px 0 0!important;padding:5px 11px!important;letter-spacing:.05em!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--tx)!important;}
.stTabs [aria-selected="true"]{background:var(--bg)!important;color:var(--grn)!important;border:1px solid var(--bd)!important;border-bottom:none!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:0!important;}

/* ── selectbox ── */
.stSelectbox>div>div,.stMultiSelect>div>div{background:var(--sur)!important;border:1px solid var(--bd)!important;border-radius:5px!important;font-family:'JetBrains Mono',monospace!important;font-size:9px!important;padding:4px 8px!important;color:var(--tx2)!important;}
.stSelectbox label,.stMultiSelect label{font-family:'JetBrains Mono',monospace!important;font-size:8px!important;color:var(--tx3)!important;text-transform:uppercase!important;letter-spacing:.1em!important;}

/* ── dataframes — th 8px 4px 7px, td 9px 5px 7px ── */
[data-testid="stDataFrame"]>div{border:1px solid var(--bd)!important;border-radius:7px!important;overflow:hidden!important;}
.stDataFrame thead th{background:var(--bg)!important;color:var(--tx3)!important;font-family:'JetBrains Mono',monospace!important;font-size:8px!important;text-transform:uppercase!important;letter-spacing:.1em!important;border-bottom:1px solid var(--bd)!important;padding:4px 7px!important;}
.stDataFrame tbody td{background:var(--sur)!important;color:var(--tx)!important;font-family:'JetBrains Mono',monospace!important;font-size:9px!important;border-bottom:1px solid var(--bg)!important;padding:5px 7px!important;}
.stDataFrame tbody tr:hover td{background:var(--bg)!important;}

/* ── buttons ── */
.stButton>button{background:transparent!important;color:var(--grn)!important;border:1px solid var(--grn-bd)!important;border-radius:5px!important;font-family:'JetBrains Mono',monospace!important;font-size:9px!important;font-weight:600!important;padding:5px 10px!important;}
.stButton>button:hover{background:var(--grn-bg)!important;}

/* ── alerts, expander ── */
.stAlert{background:var(--sur)!important;border:1px solid var(--bd)!important;border-radius:7px!important;font-size:9px!important;}
.streamlit-expanderHeader{background:var(--sur)!important;border:1px solid var(--bd)!important;border-radius:5px!important;font-size:9px!important;font-weight:600!important;color:var(--tx)!important;padding:5px 10px!important;}
.streamlit-expanderContent{background:var(--sur)!important;border:1px solid var(--bd)!important;border-top:none!important;}

/* ── metric cards (for views) ── */
.mcard{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:8px 10px;margin-bottom:8px;}
.mcard .mv{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:500;color:var(--tx);line-height:1;margin-bottom:4px;}
.mcard .ml{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;}
.mcard .ms{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--tx3);margin-top:2px;line-height:1.4;}

/* ── failure cards ── */
.fcard{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 11px;}
.frank{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:600;color:#cf222e;letter-spacing:.1em;margin-bottom:4px;}
.fktc{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:500;color:var(--tx);line-height:1;}
.flbl{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--tx3);letter-spacing:.08em;margin:3px 0 6px;}
.fbar{height:3px;border-radius:2px;background:var(--bg);overflow:hidden;margin-top:6px;}
.fbar-f{height:3px;border-radius:2px;}

/* ── live badge ── */
.sb-live{display:inline-flex;align-items:center;gap:4px;font-size:9px;color:#1a7f37;background:#dafbe1;border:1px solid #a7f3c0;padding:2px 8px;border-radius:20px;margin-top:7px;}
.ldot{width:5px;height:5px;background:#2da44e;border-radius:50%;}

/* ── tier bar ── */
.tier-bar-wrap{height:3px;background:#d0d7de;border-radius:2px;margin-top:3px;}
.tier-bar-fill{height:3px;background:var(--grn);border-radius:2px;}

/* ── columns ── */
[data-testid="column"]{padding:0 4px!important;}

/* ── typography (main area) ── */
h1{font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:500!important;color:var(--tx)!important;margin:0 0 2px!important;line-height:1.2!important;}
h2{font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:600!important;color:var(--tx)!important;border-bottom:1px solid var(--bd)!important;padding-bottom:4px!important;margin:14px 0 9px!important;}
h3{font-family:'Inter',sans-serif!important;font-size:11px!important;font-weight:600!important;color:var(--tx2)!important;margin:9px 0 5px!important;}
p,.stMarkdown p{font-size:9px!important;color:var(--tx2)!important;line-height:1.4!important;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# CONSTANTS  (original)
# =========================================================
COLORS = {
    'water':'#1a3a5c','resistive':'#D85A30','conductive':'#1D9E75',
    'primary':'#0F172A','success':'#1D9E75','warning':'#F5A623','danger':'#D85A30',
    'teal':'#0F766E','method_bp':'#6366F1','method_gn':'#0EA5E9','method_un':'#10B981'
}
ALL_LEVELS = [1,2,3,4,5,6,7]
COLORMAP = ListedColormap([COLORS['water'],COLORS['resistive'],COLORS['conductive']])
PALETTE = ['#2da44e','#8250df','#0969da','#bf8700','#cf222e',
           '#1a7f37','#d4a72c','#0550ae','#9a3ece','#068a39',
           '#6366F1','#0EA5E9']


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# =========================================================
# DATA LOADING  (original — untouched)
# =========================================================
def create_method_mapping(scores: Dict, per_run: Dict) -> Dict[str, str]:
    mapping = {}
    for display_name in scores.keys():
        display_lower = display_name.lower()
        for internal_key in per_run.keys():
            if internal_key.replace('_','-') in display_lower or internal_key.replace('_',' ') in display_lower:
                mapping[display_name] = internal_key; break
            elif display_name.split()[0].lower().replace('-','_') == internal_key.split('_')[0]:
                mapping[display_name] = internal_key; break
    return mapping

def find_latest_run() -> Path:
    """Return the most recent run folder, or fallback to outputs/."""
    runs_root = Path("outputs")
    # Check the pointer file written by example_usage.py
    pointer = runs_root / "latest.txt"
    if pointer.exists():
        latest = Path(pointer.read_text().strip())
        if latest.exists():
            return latest
    # Fallback: find newest run_YYYYMMDD_HHMMSS folder
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True)
    if run_dirs:
        return run_dirs[0]
    # Final fallback: flat outputs/ folder (old structure)
    return runs_root


@st.cache_data
def load_data(_cache_key: str = "") -> Tuple[Dict, Dict, Dict]:
    """Load scores + per-run metrics from the latest run folder."""
    run_dir = find_latest_run()

    scores = {}
    # Try run folder first, then root fallback
    for candidate in [run_dir / "scores.json", Path("scores.json"),
                      Path("outputs/scores.json")]:
        if candidate.exists():
            with open(candidate, 'r') as f:
                scores = json.load(f)
            break

    per_run = {}
    for candidate in [run_dir / "per_run_metrics.json",
                      Path("outputs/per_run_metrics.json")]:
        if candidate.exists():
            with open(candidate, 'r') as f:
                per_run = json.load(f)
            break

    return scores, per_run, create_method_mapping(scores, per_run)

@st.cache_data
def load_images_for_sample(sample_id:str, level:int=1, outputs_dir:str="") -> Dict[str,Image.Image]:
    images = {}
    op = Path(outputs_dir) if outputs_dir else find_latest_run()
    sd = op/"reconstructions"/f"level_{level}"/f"sample_{sample_id}"
    if sd.exists():
        for f in sd.glob("*.png"): images[f.stem] = Image.open(f)
    ed = op/"error_overlays"
    if ed.exists():
        for f in ed.glob(f"*_sample_{sample_id}.png"):
            k = f.stem.replace(f"_sample_{sample_id}","")
            if k not in images: images[k] = Image.open(f)
    return images

@st.cache_data
def load_comparison_panel(sample_id:str, outputs_dir:str="") -> Image.Image:
    op = Path(outputs_dir) if outputs_dir else find_latest_run()
    for fname in [f"sample_{sample_id}.png", f"sample_{sample_id}_main.png"]:
        p = op/"comparison_panels"/fname
        if p.exists(): return Image.open(p)
    return None

# =========================================================
# SCORING  (original — untouched)
# =========================================================
def calculate_composite_score(metrics:Dict[str,float], weights:Dict[str,float]) -> float:
    ktc    = metrics.get('KTC score',        metrics.get('ktc_score',0))
    dice_r = metrics.get('Dice (resistive)', metrics.get('dice_resistive',0))
    dice_c = metrics.get('Dice (conductive)',metrics.get('dice_conductive',0))
    iou_r  = metrics.get('IoU (resistive)',  metrics.get('iou_resistive',0))
    iou_c  = metrics.get('IoU (conductive)', metrics.get('iou_conductive',0))
    hd95_r = metrics.get('hd95_resistive',0); hd95_c = metrics.get('hd95_conductive',0)
    h_r = max(0,1-(hd95_r/100)); h_c = max(0,1-(hd95_c/100))
    t1=(1-ktc)*100; t2=((dice_r+dice_c)/2)*100; t3=((iou_r+iou_c)/2)*100
    t4=((h_r+h_c)/2)*100; t5=t1
    return (weights['tier1']*t1+weights['tier2']*t2+weights['tier3']*t3+weights['tier4']*t4+weights['tier5']*t5)/sum(weights.values())

def letter_grade(score:float) -> str:
    return 'A' if score>=85 else 'B' if score>=70 else 'C' if score>=55 else 'D'

def all_methods(scores:Dict) -> List[str]:
    return list(scores.keys())+[m for m in st.session_state.get('custom_methods',[]) if m not in scores]

def mcol(idx:int) -> str:
    return PALETTE[idx % len(PALETTE)]

# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    # Brand
    st.sidebar.markdown("""
    <div style="border-bottom:1px solid #d0d7de;padding-bottom:11px;margin-bottom:11px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:500;color:#1f2328;letter-spacing:.06em">EIT BENCH</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#848d97;margin-top:2px;letter-spacing:.08em">RECONSTRUCTION ANALYSIS</div>
      <div class="sb-live"><div class="ldot"></div>LIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # T1 KTC weight bar (display only — fixed at 1.00)
    st.sidebar.markdown("## Composite Weights")
    st.sidebar.markdown("""
    <div style="margin-bottom:9px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="font-size:9px;color:#57606a">T1  KTC Score</span>
        <span style="font-size:9px;color:#1f2328;font-weight:500">1.00</span>
      </div>
      <div class="tier-bar-wrap"><div class="tier-bar-fill" style="width:100%"></div></div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#1a7f37;margin-bottom:4px">&#x3A3; = 1.00</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.sidebar.columns(2)
    c1.button("Norm",  key="sb_norm",  use_container_width=True)
    c2.button("Reset", key="sb_reset", use_container_width=True)

    st.sidebar.markdown("---")

    # Data files
    st.sidebar.markdown("## Data Files")
    for p, lbl in [("scores.json","scores.json"),("outputs/per_run_metrics.json","per_run_metrics.json")]:
        ok = Path(p).exists()
        color = "#1a7f37" if ok else "#cf222e"
        icon  = "✓" if ok else "✗"
        st.sidebar.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:{color};margin:3px 0">{icon}  {lbl}</div>',
            unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Add method
    st.sidebar.markdown("## Add Method")
    st.sidebar.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:#848d97;margin-bottom:5px">Type a name then click Register.</div>', unsafe_allow_html=True)

    if 'custom_methods' not in st.session_state:
        st.session_state.custom_methods = []

    new_name = st.sidebar.text_input("name", key="new_method_input",
                                     placeholder="e.g. gauss_newton_v2",
                                     label_visibility="collapsed")
    if st.sidebar.button("+ Register", use_container_width=True, key="sb_add"):
        n = new_name.strip()
        if n and n not in st.session_state.custom_methods:
            st.session_state.custom_methods.append(n)
        elif n:
            st.sidebar.warning("Already registered")

    if st.session_state.custom_methods:
        to_remove = None
        for m in st.session_state.custom_methods:
            ca, cb = st.sidebar.columns([5,1])
            ca.markdown(f'<div style="font-size:9px;color:#57606a;padding:2px 0">• {m}</div>', unsafe_allow_html=True)
            if cb.button("✕", key=f"rm_{m}"): to_remove = m
        if to_remove:
            st.session_state.custom_methods.remove(to_remove)
            st.rerun()

    # ── Run Benchmark Button ───────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Run Benchmark")
    config_options = {
        "Full Benchmark (42 runs)": "configs/ktc_all_methods.yaml",
        "Training Data (4 samples)": "configs/training_experiment.yaml",
        "Mock / Smoke Test": "configs/mock_experiment.yaml",
    }
    chosen_config = st.sidebar.selectbox(
        "Config:", list(config_options.keys()), key="run_config_sel",
        label_visibility="collapsed")

    if st.sidebar.button("▶  Run Now", use_container_width=True, key="run_btn"):
        import subprocess, sys, os
        config_path = config_options[chosen_config]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        with st.spinner(f"Running benchmark — {chosen_config} …"):
            try:
                # Step 1 — run the benchmark
                r1 = subprocess.run(
                    [sys.executable, "run.py", "--config", config_path],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", cwd=str(Path.cwd()), env=env
                )
                if r1.returncode != 0:
                    st.sidebar.error(f"Benchmark failed:\n{r1.stderr[-500:]}")
                else:
                    # Step 2 — prepare dashboard data
                    r2 = subprocess.run(
                        [sys.executable, "prepare_dashboard.py"],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=str(Path.cwd()), env=env
                    )
                    if r2.returncode != 0:
                        st.sidebar.error(f"Prepare failed:\n{r2.stderr[-300:]}")
                    else:
                        st.cache_data.clear()
                        st.sidebar.success("Done! Dashboard updated.")
                        st.rerun()
            except Exception as ex:
                st.sidebar.error(f"Error: {ex}")

    # ── Run selector ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Run History")
    runs_root = Path("outputs")
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True) if runs_root.exists() else []

    if run_dirs:
        # Show latest pointer
        latest = find_latest_run()
        st.sidebar.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:#1a7f37;margin:3px 0">'
            f'▶ Latest: {latest.name}</div>', unsafe_allow_html=True)

        run_names = [d.name for d in run_dirs]
        chosen_run = st.sidebar.selectbox(
            "Load a previous run:",
            run_names, index=0, key="selected_run",
            label_visibility="collapsed")

        if st.sidebar.button("↺ Load selected run", use_container_width=True, key="load_run_btn"):
            selected_path = runs_root / chosen_run
            (runs_root / "latest.txt").write_text(str(selected_path))
            st.cache_data.clear()
            st.rerun()
    else:
        st.sidebar.markdown(
            '<div style="font-size:9px;color:#848d97;margin:3px 0">No runs yet.<br>Run example_usage.py first.</div>',
            unsafe_allow_html=True)

# =========================================================
# VIEW 1 — LEADERBOARD  (original logic)
# =========================================================
def view_leaderboard(scores:Dict, per_run:Dict):
    weights = {'tier1':0.40,'tier2':0.00,'tier3':0.20,'tier4':0.10,'tier5':0.05}

    if 'method_colors' not in st.session_state: st.session_state.method_colors = {}
    leaderboard_data = []
    for i,(method_name,metrics) in enumerate(scores.items()):
        if method_name not in st.session_state.method_colors:
            st.session_state.method_colors[method_name] = mcol(len(st.session_state.method_colors))
        comp  = calculate_composite_score(metrics, weights)
        grade = letter_grade(comp)
        leaderboard_data.append({
            'Method':method_name,'Composite Score':comp,'Grade':grade,
            'Color':st.session_state.method_colors[method_name],
            'KTC Score': metrics.get('KTC score', metrics.get('ktc_score',0)),
            'Dice (R)':  metrics.get('Dice (resistive)',  metrics.get('dice_resistive',0)),
            'Dice (C)':  metrics.get('Dice (conductive)', metrics.get('dice_conductive',0)),
            'IoU (R)':   metrics.get('IoU (resistive)',   metrics.get('iou_resistive',0)),
            'IoU (C)':   metrics.get('IoU (conductive)',  metrics.get('iou_conductive',0)),
            'HD95 (R)':  metrics.get('hd95_resistive',  0),
            'HD95 (C)':  metrics.get('hd95_conductive', 0),
        })
    leaderboard_data.sort(key=lambda x: x['Composite Score'], reverse=True)
    df = pd.DataFrame(leaderboard_data)

    # KPI cards — exact mockup spec
    gc = df['Grade'].value_counts()
    kpis = [
        (f"{df.iloc[0]['Composite Score']:.1f}", "TOP SCORE",  df.iloc[0]['Method'][:22], "--c1"),
        (f"{df['Composite Score'].mean():.1f}",  "AVG SCORE",  f"σ = {df['Composite Score'].std():.1f}", "--c2"),
        (str(len(df)),                           "METHODS",    f"{gc.get('A',0)}A  {gc.get('B',0)}B  {gc.get('C',0)}C  {gc.get('D',0)}D", "--c3"),
        (f"{df['KTC Score'].min():.4f}",         "BEST KTC",   "lower is better", "--c4"),
    ]
    kpi_html = '<div class="kpi-row">'
    for num, lbl, sub, kc in kpis:
        kpi_html += f'<div class="kpi" style="--kc:var({kc})"><div class="kpi-n">{num}</div><div class="kpi-l">{lbl}</div><div class="kpi-s">{sub}</div></div>'
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Method chips
    chips_html = '<div class="chips">'
    for i, name in enumerate(scores.keys()):
        chips_html += f'<span class="chip"><span class="chip-dot" style="background:{mcol(i)}"></span>{name}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    # Bar chart
    st.markdown('<div class="slbl">METHOD RANKINGS — KTC SCORE</div>', unsafe_allow_html=True)
    fig = go.Figure()
    all_names = sorted(scores.keys())
    col_map = {n: mcol(i) for i,n in enumerate(all_names)}
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row['Method'], x=[row['Method']], y=[row['Composite Score']],
            marker_color=col_map.get(row['Method'],'#64748B'),
            text=f"{row['Composite Score']:.1f} ({row['Grade']})", textposition='outside',
            textfont=dict(family="JetBrains Mono",size=9,color="#1f2328"),
            hovertemplate=(f"<b>{row['Method']}</b><br>Score: {row['Composite Score']:.1f} ({row['Grade']})<br>"
                           f"KTC: {row['KTC Score']:.4f}<br>Dice R/C: {row['Dice (R)']:.4f}/{row['Dice (C)']:.4f}<br><extra></extra>")
        ))
    fig.update_layout(
        xaxis_title="Method", yaxis_title="Score (0–100)", yaxis_range=[0,115],
        showlegend=False, height=380,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f6f8fa',
        font=dict(family="JetBrains Mono,monospace",color="#848d97",size=9),
        xaxis=dict(gridcolor='#d0d7de',linecolor='#d0d7de',tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#d0d7de',linecolor='#d0d7de',tickfont=dict(size=9)),
        margin=dict(l=0,r=10,t=20,b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown('<div class="slbl">DETAILED METRICS</div>', unsafe_allow_html=True)
    disp = df.drop(columns=['Color']).copy()
    disp['Composite Score'] = disp['Composite Score'].round(2)
    for c in ['KTC Score','Dice (R)','Dice (C)','IoU (R)','IoU (C)','HD95 (R)','HD95 (C)']:
        if c in disp.columns:
            disp[c] = disp[c].round(4)
    st.dataframe(disp, use_container_width=True, hide_index=True)

# =========================================================
# VIEW 2 — DEGRADATION  (original logic)
# =========================================================
def view_degradation_curve(scores:Dict, per_run:Dict, mm:Dict):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    dm = all_methods(scores)
    cl, cm = st.columns([1,3])
    with cl: lvl = st.selectbox("Level:", ALL_LEVELS, index=0, key="deg_level")
    with cm:
        chosen = st.multiselect("Select methods:",dm,
            default=[m for m in dm[:3] if m not in st.session_state.get('custom_methods',[])])

    if not chosen:
        st.info("Select at least one method.")
        return

    fig = go.Figure()
    stats = []
    for i, disp in enumerate(chosen):
        ik = mm.get(disp, disp)
        if ik not in per_run: continue
        samps = per_run[ik]

        # Group by level — average KTC across samples A/B/C per level
        level_scores: dict = {}
        for run_key, metrics in samps.items():
            lv = metrics.get("level")
            if lv is not None:
                level_scores.setdefault(int(lv), []).append(metrics.get("ktc_score", 0))

        if not level_scores:
            continue

        levels_x = sorted(level_scores.keys())
        ktc_avg  = [float(np.mean(level_scores[lv])) for lv in levels_x]
        ktc_min  = [float(np.min(level_scores[lv]))  for lv in levels_x]
        ktc_max  = [float(np.max(level_scores[lv]))  for lv in levels_x]

        c = mcol(i)
        # Shaded band (min/max range)
        fig.add_trace(go.Scatter(
            x=levels_x + levels_x[::-1],
            y=ktc_max + ktc_min[::-1],
            fill='toself', fillcolor=hex_to_rgba(c, 0.10),
            line=dict(width=0), showlegend=False, hoverinfo='skip'))
        # Average line
        fig.add_trace(go.Scatter(
            x=levels_x, y=ktc_avg, mode='lines+markers', name=disp,
            line=dict(width=2.5, color=c),
            marker=dict(size=7, color=c, line=dict(width=2, color='#ffffff')),
            hovertemplate=f"<b>{disp}</b><br>Level: %{{x}}<br>Avg KTC: %{{y:.4f}}<extra></extra>"))

        stats.append({'Method': disp, 'Mean KTC': np.mean(ktc_avg),
                      'Std Dev': np.std(ktc_avg), 'Min': np.min(ktc_avg),
                      'Max': np.max(ktc_avg), 'Range': np.max(ktc_avg)-np.min(ktc_avg)})

    fig.update_layout(
        title="KTC Score vs Difficulty Level (avg across samples A/B/C)",
        xaxis_title="Level (1=easiest → 7=hardest)", yaxis_title="KTC Score ↑",
        height=420,hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#f6f8fa',
        font=dict(family="JetBrains Mono,monospace",color="#848d97",size=9),
        xaxis=dict(gridcolor='#d0d7de',linecolor='#d0d7de'),
        yaxis=dict(gridcolor='#d0d7de',linecolor='#d0d7de'),
        legend=dict(bgcolor='rgba(255,255,255,.9)',bordercolor='#d0d7de',borderwidth=1,font=dict(size=9)),
        margin=dict(l=0,r=0,t=36,b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="slbl">KTC STATISTICS</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(stats).round(4), use_container_width=True, hide_index=True)

# =========================================================
# VIEW 3 — COMPARISON  (original logic)
# =========================================================
def view_comparison(scores:Dict, per_run:Dict, mm:Dict):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    dm = all_methods(scores)
    fi = list(per_run.keys())[0] if per_run else None
    samps = list(per_run[fi].keys()) if fi else []

    c1,c2,c3,c4 = st.columns(4)
    m1 = c1.selectbox("Method 1:", dm, index=0)
    m2 = c2.selectbox("Method 2:", dm, index=min(1,len(dm)-1))
    sid = c3.selectbox("Sample:", samps)
    lvl = c4.selectbox("Level:", ALL_LEVELS, index=0)

    m1i = mm.get(m1); m2i = mm.get(m2)
    p1 = m1 in st.session_state.get('custom_methods', [])
    p2 = m2 in st.session_state.get('custom_methods', [])
    if p1: st.info(f"**{m1}** is a custom method — connect its backend to see metrics.")
    if p2: st.info(f"**{m2}** is a custom method — connect its backend to see metrics.")

    met1 = per_run.get(m1i or m1, {}).get(sid, {}) if not p1 else {}
    met2 = per_run.get(m2i or m2, {}).get(sid, {}) if not p2 else {}

    st.markdown("### Metric Comparison")
    if not met1 and not met2:
        st.warning(f"No data found for sample '{sid}'. Run `python prepare_dashboard.py` to refresh.")
    else:
        numeric_keys = [k for k in met1.keys() if isinstance(met1.get(k), (int, float))]
        comp_data = [{'Metric':k.replace('_',' ').title(), m1:met1.get(k,0), m2:met2.get(k,0),
                      'Difference':abs(met1.get(k,0)-met2.get(k,0))} for k in numeric_keys]
        comp_df = pd.DataFrame(comp_data)
        if not comp_df.empty:
            for col in [m1, m2, 'Difference']:
                if col in comp_df.columns:
                    comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("### Radar Chart")
    radar_keys = ['ktc_score','dice_resistive','dice_conductive','iou_resistive','iou_conductive']
    cats = [k.replace('_',' ').title() for k in radar_keys]
    fig = go.Figure()
    for label,met,c in [(m1,met1,PALETTE[0]),(m2,met2,PALETTE[1])]:
        v = [met.get(k,0) for k in radar_keys]; v.append(v[0])
        fig.add_trace(go.Scatterpolar(r=v,theta=cats+[cats[0]],fill='toself',name=label,
            line_color=c,fillcolor=hex_to_rgba(c, 0.13)))
    fig.update_layout(
        polar=dict(bgcolor='#f6f8fa',
            radialaxis=dict(visible=True,range=[0,1],gridcolor='#d0d7de',linecolor='#d0d7de',tickfont=dict(size=8)),
            angularaxis=dict(gridcolor='#d0d7de',linecolor='#d0d7de',tickfont=dict(size=9))),
        showlegend=True,height=460,paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="JetBrains Mono,monospace",color="#848d97",size=9),
        legend=dict(bgcolor='rgba(255,255,255,.9)',bordercolor='#d0d7de',borderwidth=1,font=dict(size=9)),
        margin=dict(l=50,r=50,t=30,b=50))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Visual Comparison")
    panel = load_comparison_panel(sid)
    if panel:
        st.markdown(f"**All Methods – Sample {sid}**")
        st.image(panel, use_container_width=True)
        st.markdown("---")
    imgs = load_images_for_sample(sid, level=lvl)
    if imgs:
        ic1,ic2 = st.columns(2)
        i1 = next((img for k,img in imgs.items() if m1i and m1i.lower() in k.lower()),None)
        i2 = next((img for k,img in imgs.items() if m2i and m2i.lower() in k.lower()),None)
        with ic1:
            st.markdown(f"**{m1}**")
            if i1:
                st.image(i1, use_container_width=True)
            else:
                st.info(f"Image not found for {m1}")
        with ic2:
            st.markdown(f"**{m2}**")
            if i2:
                st.image(i2, use_container_width=True)
            else:
                st.info(f"Image not found for {m2}")
    elif not panel:
        st.info("No visualization images found. Run example_usage.py to generate images.")

# =========================================================
# VIEW 4 — FAILURE GALLERY  (original logic)
# =========================================================
def view_failure_gallery(scores:Dict, per_run:Dict, mm:Dict):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    for disp in scores.keys():
        ik = mm.get(disp)
        if not ik or ik not in per_run:
            st.info(f"No per-run data for {disp}")
            continue
        worst3 = sorted(per_run[ik].items(), key=lambda x:x[1]['ktc_score'], reverse=True)[:3]

        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:600;color:#1f2328;padding:6px 0 4px;border-bottom:1px solid #d0d7de;margin-bottom:8px">{disp}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        max_ktc = max((m['ktc_score'] for _,m in worst3), default=1) or 1
        for idx,(sid,metrics) in enumerate(worst3):
            with cols[idx]:
                pct = int(metrics['ktc_score']/max_ktc*100)
                bc  = ['#cf222e','#bf8700','#0969da'][idx]
                st.markdown(f"""<div class="fcard">
                  <div class="frank">#{idx+1} WORST · SAMPLE {sid}</div>
                  <div class="fktc">{metrics['ktc_score']:.4f}</div>
                  <div class="flbl">KTC SCORE</div>
                  <div class="fbar"><div class="fbar-f" style="width:{pct}%;background:{bc}"></div></div>
                </div>""", unsafe_allow_html=True)
                img_shown = False
                for p in [Path("outputs/reconstructions/level_1")/f"sample_{sid}"/f"{ik}.png",
                          Path("outputs/error_overlays")/f"{ik}_sample_{sid}.png"]:
                    if p.exists():
                        st.image(Image.open(p), use_container_width=True)
                        img_shown = True
                        break
                if not img_shown:
                    st.caption("Image not available")
        st.markdown("---")

# =========================================================
# VIEW 5 — RADAR CHART  (original logic)
# =========================================================
def view_radar_chart(scores:Dict, per_run:Dict):
    if not scores:
        st.warning("No scores available.")
        return

    avail = sorted({k for m in scores.values() for k in m.keys()})
    chosen = st.multiselect("Choose metrics:", avail,
        default=avail[:5] if len(avail)>=5 else avail)
    if not chosen:
        st.info("Select at least one metric.")
        return

    fig = go.Figure()
    for i,(name,metrics) in enumerate(scores.items()):
        vals = [max(0,1-metrics.get(m,0)) if 'ktc' in m.lower() else metrics.get(m,0) for m in chosen]
        vals.append(vals[0])
        cats = [m.replace('_',' ').title() for m in chosen]; cats.append(cats[0])
        c = mcol(i)
        fig.add_trace(go.Scatterpolar(r=vals,theta=cats,fill='toself',name=name,
            line_color=c,fillcolor=hex_to_rgba(c, 0.13)))
    fig.update_layout(
        polar=dict(bgcolor='#f6f8fa',
            radialaxis=dict(visible=True,range=[0,1],gridcolor='#d0d7de',linecolor='#d0d7de',tickfont=dict(size=8)),
            angularaxis=dict(gridcolor='#d0d7de',linecolor='#d0d7de',tickfont=dict(size=10))),
        showlegend=True,height=560,title="Method Performance Across Selected Metrics",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="JetBrains Mono,monospace",color="#848d97",size=9),
        legend=dict(bgcolor='rgba(255,255,255,.9)',bordercolor='#d0d7de',borderwidth=1,font=dict(size=9)),
        margin=dict(l=55,r=55,t=45,b=55))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="slbl">METRIC STATISTICS</div>', unsafe_allow_html=True)
    rows = []
    for m in chosen:
        vals = [max(0,1-ms.get(m,0)) if 'ktc' in m.lower() else ms.get(m,0) for ms in scores.values()]
        rows.append({'Metric':m.replace('_',' ').title(),'Mean':np.mean(vals),'Std Dev':np.std(vals),'Min':np.min(vals),'Max':np.max(vals)})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)

# =========================================================
# VIEW 6 — ALL RUNS  (same as terminal table)
# =========================================================
def view_all_runs(per_run: Dict):
    """Show every individual run — same as terminal Experiment Summary + Degradation Slope."""

    if not per_run:
        st.warning("No per-run data available. Click 'Run Now' in the sidebar or run `python prepare_dashboard.py`.")
        return

    # ── Build full runs table ─────────────────────────────────
    rows = []
    for method, runs in per_run.items():
        for run_key, metrics in runs.items():
            rows.append({
                "Method":     method,
                "Level":      metrics.get("level", run_key),
                "Sample":     metrics.get("sample", run_key),
                "KTC Score":  round(metrics.get("ktc_score", 0), 3),
                "Dice Res.":  round(metrics.get("dice_resistive", 0), 3),
                "Dice Cond.": round(metrics.get("dice_conductive", 0), 3),
                "HD95 Res.":  round(metrics.get("hd95_resistive", 0), 1),
                "HD95 Cond.": round(metrics.get("hd95_conductive", 0), 1),
                "Runtime ms": round(metrics.get("runtime_ms", 0), 1),
                "Grade":      metrics.get("grade", "-"),
            })

    df = pd.DataFrame(rows).sort_values(["Method", "Level", "Sample"]).reset_index(drop=True)

    # ── Table 1: Experiment Summary ───────────────────────────
    st.markdown("### Experiment Summary")
    col1, col2 = st.columns(2)
    methods_list = ["All"] + sorted(df["Method"].unique().tolist())
    sel_method = col1.selectbox("Method:", methods_list, key="ar_method")
    levels_list = ["All"] + sorted(df["Level"].unique().tolist())
    sel_level = col2.selectbox("Level:", levels_list, key="ar_level")

    filtered = df.copy()
    if sel_method != "All":
        filtered = filtered[filtered["Method"] == sel_method]
    if sel_level != "All":
        filtered = filtered[filtered["Level"] == sel_level]

    st.markdown(f"**{len(filtered)} / {len(df)} runs**")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # ── Table 2: Degradation Slope ────────────────────────────
    st.markdown("### Degradation Slope by Method")
    slope_rows = []
    for method, runs in per_run.items():
        level_scores: Dict[int, list] = {}
        for metrics in runs.values():
            lv = metrics.get("level")
            if lv is not None:
                level_scores.setdefault(int(lv), []).append(metrics.get("ktc_score", 0))
        levels_sorted = sorted(level_scores.keys())
        avg_scores = [float(np.mean(level_scores[lv])) for lv in levels_sorted]
        if len(levels_sorted) >= 2:
            slope = float(np.polyfit(levels_sorted, avg_scores, 1)[0])
        else:
            slope = 0.0
        direction = "degrades" if slope < 0 else "improves"
        slope_rows.append({
            "Method":          method,
            "Slope (per level)": round(slope, 4),
            "Direction":       f"▼ {direction}" if slope < 0 else f"▲ {direction}",
        })

    st.dataframe(pd.DataFrame(slope_rows), use_container_width=True, hide_index=True)
    st.caption("Steeper negative slope = method degrades faster at harder levels")


# =========================================================
# MAIN
# =========================================================
def main():
    render_sidebar()

    # Header — exact mockup: 2px stripe, 13px/500 title, 9px sub
    st.markdown("""
    <div class="dash-header">
      <div class="dash-title">&#9889; EIT Reconstruction Dashboard</div>
      <div class="dash-sub">Electrical Impedance Tomography &#8212; Benchmarking &amp; Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Cache key = latest run folder name so cache refreshes on new run
        latest_run = find_latest_run()
        _cache_key = latest_run.name
        scores, per_run, mm = load_data(_cache_key)

        # Show which run is loaded
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'color:#848d97;margin:-6px 0 10px;padding:0 2px">'
            f'📂 Loaded: <span style="color:#1a7f37">{latest_run.name}</span></div>',
            unsafe_allow_html=True)

        if not scores and not per_run:
            st.error("No data found. Run `python example_usage.py` first.")
            return

        # Dataset info expander
        with st.expander("Dataset Information", expanded=False):
            c1,c2,c3 = st.columns(3)
            n_s = len(per_run.get(list(per_run.keys())[0],{})) if per_run else 0
            n_t = sum(len(v) for v in per_run.values()) if per_run else 0
            stat_items = [
                (str(len(scores)), "Methods Analyzed"),
                (str(n_s),         "Total Samples"),
                (str(n_t),         "Total Reconstructions"),
            ]
            for col_, (num, lbl) in zip([c1, c2, c3], stat_items):
                with col_:
                    st.markdown(
                        f'<div class="mcard"><div class="mv" style="font-size:18px">{num}</div>'
                        f'<div class="ml">{lbl}</div></div>',
                        unsafe_allow_html=True)
            if scores:
                st.markdown("**Available methods:**")
                for m in scores.keys(): st.markdown(f"  • {m}")

        # Tabs — exact mockup labels
        t1,t2,t3,t4,t5,t6 = st.tabs([
            "01  LEADERBOARD","02  DEGRADATION",
            "03  COMPARISON","04  FAILURES","05  RADAR",
            "06  ALL RUNS"])

        with t1: view_leaderboard(scores, per_run)
        with t2: view_degradation_curve(scores, per_run, mm)
        with t3: view_comparison(scores, per_run, mm)
        with t4: view_failure_gallery(scores, per_run, mm)
        with t5: view_radar_chart(scores, per_run)
        with t6: view_all_runs(per_run)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# Run the app
main()