"""
app.py — EIT Reconstruction Dashboard
Layout: pixel-exact to approved white mockup
Data:   all original logic preserved unchanged
"""

import streamlit as st
import json
import subprocess
import sys
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

/* ── hide Streamlit sidebar collapse arrow button (keyboard_double_arrow_left icon) ── */
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
.st-emotion-cache-1q1n0ol,
.eyeqlp51,
[data-testid="stBaseButton-headerNoPadding"]{display:none!important;}

/* ── hide Material icon text that leaks through the collapse button ── */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] p,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="stSidebarCollapseButton"] p,
[data-testid="stSidebarCollapseButton"] span {
  visibility:hidden!important; font-size:0!important; color:transparent!important;
}

/* ── LIGHT theme tokens ── */
:root{
  --bg:#f6f8fa; --sur:#ffffff; --bd:#d0d7de;
  --tx:#1f2328;  --tx2:#57606a; --tx3:#848d97;
  --grn:#1a7f37; --grn-bg:#dafbe1; --grn-bd:#a7f3c0;
  --c1:#2da44e; --c2:#8250df; --c3:#0969da; --c4:#bf8700; --c5:#cf222e;
  --amb:#9a6700; --red:#cf222e;
  --inp-bg:#ffffff; --inp-bd:#d0d7de; --inp-tx:#1f2328;
  --chk-bg:#ffffff;
}

/* ── DARK theme tokens — applied when .eit-dark on stApp ── */
.eit-dark{
  --bg:#0d1117; --sur:#161b22; --bd:#30363d;
  --tx:#e6edf3;  --tx2:#c9d1d9; --tx3:#8b949e;
  --grn:#3fb950; --grn-bg:#0d2119; --grn-bd:#1a4e2a;
  --c1:#3fb950; --c2:#bc8cff; --c3:#58a6ff; --c4:#d29922; --c5:#f85149;
  --amb:#d29922; --red:#f85149;
  --inp-bg:#21262d; --inp-bd:#30363d; --inp-tx:#e6edf3;
  --chk-bg:#21262d;
}

/* ── entire app background + text ── */
html,body{background:var(--bg)!important;color:var(--tx)!important;}
[data-testid="stApp"]{background:var(--bg)!important;color:var(--tx)!important;}
.main,.main .block-container{
  background:var(--bg)!important;color:var(--tx)!important;
  padding:8px 20px 42px!important;max-width:100%!important;
}

/* ── sidebar — background + all text ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"]>div{
  background:var(--sur)!important;border-right:1px solid var(--bd)!important;
}
section[data-testid="stSidebar"]>div:first-child{padding:14px 12px!important;}
section[data-testid="stSidebar"] *{
  font-family:'JetBrains Mono',monospace!important;color:var(--tx2)!important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  font-size:10px!important;font-weight:700!important;color:var(--tx3)!important;
  text-transform:uppercase!important;letter-spacing:.14em!important;
  border:none!important;padding:0!important;margin:14px 0 7px!important;line-height:1!important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown p{
  font-size:11px!important;margin:4px 0!important;line-height:1.5!important;color:var(--tx2)!important;
}
section[data-testid="stSidebar"] label{
  font-size:11px!important;color:var(--tx2)!important;
  letter-spacing:.02em!important;
  /* NO text-transform — method names like BackProjection must stay readable */
}
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="number"]{
  font-size:11px!important;padding:6px 8px!important;
  background:var(--inp-bg)!important;border:1px solid var(--inp-bd)!important;
  color:var(--inp-tx)!important;border-radius:5px!important;
}
section[data-testid="stSidebar"] button{
  font-size:11px!important;padding:7px 0!important;width:100%!important;
  border:1px solid var(--grn-bd)!important;border-radius:6px!important;
  color:var(--grn)!important;background:transparent!important;text-align:center!important;
  min-height:32px!important;
}
section[data-testid="stSidebar"] button:hover{background:var(--grn-bg)!important;}
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button{
  margin-top:6px!important;background:var(--grn-bg)!important;
  border-color:var(--grn-bd)!important;color:var(--grn)!important;
}
/* ── ▶ run-method buttons: compact, sit beside the checkbox ── */
section[data-testid="stSidebar"] [data-testid^="stColumn"] button[kind="secondary"]{
  font-size:11px!important;padding:4px 0!important;
  border-color:var(--bd)!important;color:var(--tx3)!important;
}
section[data-testid="stSidebar"] [data-testid^="stColumn"] button[kind="secondary"]:hover{
  background:var(--grn-bg)!important;border-color:var(--grn-bd)!important;color:var(--grn)!important;
}
section[data-testid="stSidebar"] hr{border-color:var(--bd)!important;margin:13px 0!important;}

/* ── file uploader: hide drag-drop UI, keep Browse button ── */
[data-testid="stFileUploaderDropzone"]{
  border:1px solid var(--bd)!important;border-radius:6px!important;
  background:transparent!important;padding:2px 4px!important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"]{
  display:none!important;
}
/* target the inner text nodes that show "Drag and drop" */
[data-testid="stFileUploaderDropzone"] > div > div:not(:has(button)){
  display:none!important;
}
[data-testid="stFileUploaderDropzone"] button{
  width:100%!important;font-family:'JetBrains Mono',monospace!important;
  font-size:11px!important;font-weight:600!important;
  color:var(--grn)!important;background:transparent!important;
  border:1px solid var(--grn-bd)!important;border-radius:6px!important;
  padding:7px 0!important;cursor:pointer!important;
}
[data-testid="stFileUploaderDropzone"] button:hover{background:var(--grn-bg)!important;}

/* ── checkboxes ── */
[data-testid="stCheckbox"] label{color:var(--tx2)!important;font-size:11px!important;}
[data-testid="stCheckbox"] input + div,
[data-testid="stCheckbox"] span[data-baseweb="checkbox"]{
  background:var(--chk-bg)!important;border-color:var(--bd)!important;
}

/* ── slider ── */
[data-testid="stSlider"] label{color:var(--tx3)!important;font-size:10px!important;}
[data-testid="stSlider"] [data-baseweb="slider"] div{background:var(--bd)!important;}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"]{
  color:var(--tx)!important;background:var(--sur)!important;border-color:var(--bd)!important;
}

/* ── selectbox + multiselect ── */
.stSelectbox>div>div,.stMultiSelect>div>div{
  background:var(--sur)!important;border:1px solid var(--bd)!important;
  border-radius:6px!important;font-family:'JetBrains Mono',monospace!important;
  font-size:11px!important;padding:6px 10px!important;color:var(--tx2)!important;
}
.stSelectbox label,.stMultiSelect label{
  font-family:'JetBrains Mono',monospace!important;font-size:10px!important;
  color:var(--tx3)!important;text-transform:uppercase!important;letter-spacing:.1em!important;
}
/* dropdown popup */
[data-baseweb="popover"] ul,[data-baseweb="menu"]{
  background:var(--sur)!important;border:1px solid var(--bd)!important;
}
[data-baseweb="menu"] li{
  background:var(--sur)!important;color:var(--tx2)!important;
  font-size:11px!important;
}
[data-baseweb="menu"] li:hover{background:var(--bg)!important;}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--sur)!important;border:1px solid var(--bd)!important;
  border-radius:7px!important;padding:4px!important;gap:2px!important;margin-bottom:12px!important;
}
.stTabs [data-baseweb="tab"]{
  font-family:'JetBrains Mono',monospace!important;font-size:11px!important;font-weight:500!important;
  color:var(--tx3)!important;background:transparent!important;border:none!important;
  border-radius:5px 5px 0 0!important;padding:6px 14px!important;letter-spacing:.04em!important;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--tx)!important;}
.stTabs [aria-selected="true"]{
  background:var(--bg)!important;color:var(--grn)!important;
  border:1px solid var(--bd)!important;border-bottom:none!important;
}
.stTabs [data-baseweb="tab-panel"]{padding-top:0!important;}

/* ── dataframe ── */
[data-testid="stDataFrame"]>div{
  border:1px solid var(--bd)!important;border-radius:7px!important;overflow:hidden!important;
  background:var(--sur)!important;
}
.stDataFrame thead th{
  background:var(--bg)!important;color:var(--tx3)!important;
  font-family:'JetBrains Mono',monospace!important;font-size:10px!important;
  text-transform:uppercase!important;letter-spacing:.08em!important;
  border-bottom:1px solid var(--bd)!important;padding:6px 9px!important;
}
.stDataFrame tbody td{
  background:var(--sur)!important;color:var(--tx)!important;
  font-family:'JetBrains Mono',monospace!important;font-size:11px!important;
  border-bottom:1px solid var(--bg)!important;padding:6px 9px!important;
}
.stDataFrame tbody tr:hover td{background:var(--bg)!important;}

/* ── buttons ── */
.stButton>button,.stDownloadButton>button{
  background:transparent!important;color:var(--grn)!important;
  border:1px solid var(--grn-bd)!important;border-radius:6px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:11px!important;
  font-weight:600!important;padding:7px 14px!important;min-height:32px!important;
}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--grn-bg)!important;}

/* ── alerts + expander ── */
.stAlert{
  background:var(--sur)!important;border:1px solid var(--bd)!important;
  border-radius:7px!important;font-size:11px!important;color:var(--tx)!important;
}
.streamlit-expanderHeader{
  background:var(--sur)!important;border:1px solid var(--bd)!important;
  border-radius:6px!important;font-size:11px!important;font-weight:600!important;
  color:var(--tx)!important;padding:7px 12px!important;
}
.streamlit-expanderContent{
  background:var(--sur)!important;border:1px solid var(--bd)!important;border-top:none!important;
}

/* ── typography ── */
h1{font-family:'Inter',sans-serif!important;font-size:15px!important;font-weight:500!important;color:var(--tx)!important;margin:0 0 3px!important;}
h2{font-family:'Inter',sans-serif!important;font-size:15px!important;font-weight:600!important;color:var(--tx)!important;border-bottom:1px solid var(--bd)!important;padding-bottom:5px!important;margin:16px 0 10px!important;}
h3{font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:600!important;color:var(--tx2)!important;margin:10px 0 6px!important;}
p,.stMarkdown p{font-size:11px!important;color:var(--tx2)!important;line-height:1.5!important;}
.slbl{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;}

/* ── header ── */
.dash-header{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:14px 20px 13px;margin:0 0 12px;position:relative;overflow:hidden;}
.dash-header::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#2da44e,#0969da,#8250df);}
.dash-title{font-family:'Inter',sans-serif;font-size:15px;font-weight:500;color:var(--tx);line-height:1.2;margin-bottom:4px;}
.dash-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tx3);}

/* ── KPI cards ── */
.kpi-row{display:flex;gap:10px;margin-bottom:12px;}
.kpi{flex:1;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:11px 13px;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--kc,#2da44e);opacity:.8;}
.kpi-n{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;color:var(--tx);line-height:1;}
.kpi-l{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin-top:5px;}
.kpi-s{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);margin-top:3px;}

/* ── chips ── */
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2);background:var(--bg);border:1px solid var(--bd);padding:3px 10px 3px 7px;border-radius:20px;font-family:'JetBrains Mono',monospace;}
.chip-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}

/* ── metric + failure cards ── */
.mcard{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 13px;margin-bottom:9px;}
.mcard .mv{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:500;color:var(--tx);line-height:1;margin-bottom:5px;}
.mcard .ml{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;}
.fcard{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:11px 13px;}
.frank{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--red);letter-spacing:.1em;margin-bottom:5px;}
.fktc{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;color:var(--tx);line-height:1;}
.flbl{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);letter-spacing:.07em;margin:4px 0 7px;}
.fbar{height:4px;border-radius:2px;background:var(--bg);overflow:hidden;margin-top:7px;}
.fbar-f{height:4px;border-radius:2px;}

/* ── live badge ── */
.sb-live{display:inline-flex;align-items:center;gap:5px;font-size:10px;color:var(--grn);background:var(--grn-bg);border:1px solid var(--grn-bd);padding:3px 10px;border-radius:20px;margin-top:7px;}
.ldot{width:6px;height:6px;background:var(--grn);border-radius:50%;}

/* ── tier bar ── */
.tier-bar-wrap{height:3px;background:var(--bd);border-radius:2px;margin-top:3px;}
.tier-bar-fill{height:3px;background:var(--grn);border-radius:2px;}

/* ── columns ── */
[data-testid="column"]{padding:0 4px!important;}

/* ── scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px;}

/* ── dark mode: stApp class override — covers every element ── */
.eit-dark [data-testid="stApp"],
.eit-dark .main,
.eit-dark .main .block-container,
.eit-dark [data-testid="stHeader"],
.eit-dark [data-testid="stToolbar"],
.eit-dark [data-testid="stDecoration"],
.eit-dark [data-testid="stStatusWidget"]{
  background:var(--bg)!important;color:var(--tx)!important;
}
.eit-dark section[data-testid="stSidebar"],
.eit-dark section[data-testid="stSidebar"]>div{
  background:var(--sur)!important;border-right-color:var(--bd)!important;
}
.eit-dark section[data-testid="stSidebar"] *{color:var(--tx2)!important;}
.eit-dark .stMarkdown,.eit-dark .stMarkdown *,.eit-dark p{color:var(--tx2)!important;}
.eit-dark [data-testid="stDataFrame"]>div{background:var(--sur)!important;border-color:var(--bd)!important;}
.eit-dark .stDataFrame thead th{background:var(--bg)!important;color:var(--tx3)!important;border-color:var(--bd)!important;}
.eit-dark .stDataFrame tbody td{background:var(--sur)!important;color:var(--tx)!important;}
.eit-dark .stDataFrame tbody tr:hover td{background:var(--bd)!important;}
.eit-dark .stTabs [data-baseweb="tab-list"]{background:var(--sur)!important;border-color:var(--bd)!important;}
.eit-dark .stTabs [data-baseweb="tab"]{color:var(--tx3)!important;background:transparent!important;}
.eit-dark .stTabs [aria-selected="true"]{background:var(--bg)!important;color:var(--grn)!important;border-color:var(--bd)!important;}
.eit-dark .dash-header{background:var(--sur)!important;border-color:var(--bd)!important;}
.eit-dark .stSelectbox>div>div,.eit-dark .stMultiSelect>div>div{background:var(--sur)!important;border-color:var(--bd)!important;color:var(--tx2)!important;}
.eit-dark [data-baseweb="menu"]{background:var(--sur)!important;border-color:var(--bd)!important;}
.eit-dark [data-baseweb="menu"] li{background:var(--sur)!important;color:var(--tx2)!important;}
.eit-dark [data-baseweb="menu"] li:hover{background:var(--bd)!important;}
.eit-dark .stButton>button{color:var(--grn)!important;border-color:var(--grn-bd)!important;background:transparent!important;}
.eit-dark .stButton>button:hover{background:var(--grn-bg)!important;}
.eit-dark .stAlert{background:var(--sur)!important;border-color:var(--bd)!important;color:var(--tx)!important;}
.eit-dark .streamlit-expanderHeader{background:var(--sur)!important;border-color:var(--bd)!important;color:var(--tx)!important;}
.eit-dark .streamlit-expanderContent{background:var(--sur)!important;border-color:var(--bd)!important;}
.eit-dark [data-testid="stCheckbox"] label{color:var(--tx2)!important;}
.eit-dark input[type="text"],.eit-dark input[type="number"]{background:var(--inp-bg)!important;border-color:var(--inp-bd)!important;color:var(--inp-tx)!important;}
.eit-dark h1,.eit-dark h2,.eit-dark h3{color:var(--tx)!important;}

/* ---- UI layout polish: readable scale, aligned controls, less top whitespace ---- */
[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important;}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
.main,.main .block-container{padding:4px 20px 42px!important;}

section[data-testid="stSidebar"]>div:first-child{padding:12px 12px!important;}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  font-size:11px!important;margin:12px 0 7px!important;line-height:1!important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label{font-size:12px!important;line-height:1.45!important;}
section[data-testid="stSidebar"] button,
.stButton>button,
.stDownloadButton>button{
  font-size:12px!important;min-height:34px!important;padding:8px 14px!important;
}
section[data-testid="stSidebar"] button{padding-left:0!important;padding-right:0!important;}
section[data-testid="stSidebar"] [data-testid^="stColumn"] button[kind="secondary"]{
  font-size:12px!important;min-height:30px!important;padding:5px 0!important;
}

[data-testid="stCheckbox"] label{font-size:12px!important;}
[data-testid="stSlider"] label{font-size:11px!important;}
.stSelectbox>div>div,.stMultiSelect>div>div{
  font-size:12px!important;min-height:36px!important;padding:7px 10px!important;
}
.stSelectbox label,.stMultiSelect label{font-size:11px!important;}
[data-baseweb="menu"] li{font-size:12px!important;}

.stTabs [data-baseweb="tab-list"]{
  gap:4px!important;flex-wrap:wrap!important;margin-bottom:12px!important;
}
.stTabs [data-baseweb="tab"]{
  font-size:12px!important;font-weight:600!important;min-height:34px!important;
  padding:8px 12px!important;border-radius:5px!important;letter-spacing:.02em!important;
}
.stTabs [data-baseweb="tab-panel"]{padding-top:0!important;}

.stDataFrame thead th{font-size:11px!important;padding:7px 10px!important;}
.stDataFrame tbody td{font-size:12px!important;padding:7px 10px!important;}

h1{font-size:18px!important;font-weight:600!important;margin:0 0 4px!important;}
h2{font-size:16px!important;margin:14px 0 10px!important;padding-bottom:6px!important;}
h3{font-size:14px!important;margin:10px 0 7px!important;}
p,.stMarkdown p{font-size:13px!important;line-height:1.45!important;}
.slbl{font-size:11px!important;letter-spacing:.1em!important;margin-bottom:10px!important;}

.dash-header{padding:13px 18px 12px!important;margin:0 0 10px!important;}
.dash-title{font-size:18px!important;font-weight:600!important;margin-bottom:3px!important;}
.dash-sub{font-size:12px!important;}

.kpi-row{gap:10px!important;margin-bottom:12px!important;align-items:stretch!important;}
.kpi{padding:12px 14px!important;min-height:76px!important;}
.kpi-n{font-size:21px!important;font-weight:600!important;}
.kpi-l,.kpi-s{font-size:11px!important;}
.chip{font-size:12px!important;padding:4px 10px 4px 8px!important;}

.mcard{padding:12px 14px!important;margin-bottom:10px!important;min-height:72px!important;}
.mcard .mv{font-size:20px!important;font-weight:600!important;}
.mcard .ml{font-size:11px!important;letter-spacing:.08em!important;}
.fcard{padding:12px 14px!important;min-height:108px!important;}
.frank,.flbl{font-size:11px!important;}
.fktc{font-size:21px!important;font-weight:600!important;}

.main [style*="font-size:8px"]{font-size:11px!important;}
.main [style*="font-size:9px"]{font-size:11px!important;}
.main [style*="font-size:10px"]{font-size:12px!important;}
.main [style*="font-size:11px"]{font-size:12px!important;}

/* Final control polish for Streamlit/BaseWeb dropdowns and compact widgets. */
.stSelectbox>div>div,.stMultiSelect>div>div{
  padding:0!important;
}
[data-baseweb="select"]{
  min-height:38px!important;border-radius:6px!important;
}
[data-baseweb="select"] > div{
  min-height:38px!important;height:auto!important;padding:0 10px!important;
  display:flex!important;align-items:center!important;
  border-color:var(--bd)!important;background:var(--sur)!important;
}
[data-baseweb="select"] *{
  font-family:'JetBrains Mono',monospace!important;
  font-size:12px!important;line-height:18px!important;
  color:var(--tx2)!important;box-sizing:border-box!important;
}
[data-baseweb="select"] input{
  min-height:20px!important;height:20px!important;padding:0!important;margin:0!important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div[role="button"],
[data-baseweb="select"] div[role="combobox"]{
  min-height:20px!important;display:flex!important;align-items:center!important;
  overflow:visible!important;
}
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  min-height:36px!important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] *{
  font-size:12px!important;line-height:18px!important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li{
  font-family:'JetBrains Mono',monospace!important;
  font-size:12px!important;line-height:1.35!important;
  padding-top:7px!important;padding-bottom:7px!important;
}
.stSlider,.stSelectbox,.stMultiSelect{margin-bottom:8px!important;}
.stCheckbox{margin-bottom:2px!important;}
</style>
<script>
(function(){
  /* Hide Streamlit's sidebar collapse/expand icon (shows as raw text when
     Material Symbols font fails to load).  Uses MutationObserver so it fires
     even after Streamlit's own JS re-renders the sidebar. */
  function purgeSidebarIcon(){
    try {
      var doc = window.parent ? window.parent.document : document;
      /* 1. Hide the collapse-button element by every known data-testid variant */
      ['stSidebarCollapseButton','collapsedControl','stBaseButton-headerNoPadding'].forEach(function(tid){
        doc.querySelectorAll('[data-testid="'+tid+'"]').forEach(function(el){
          el.style.setProperty('display','none','important');
        });
      });
      /* 2. Also target any leaf node in the sidebar whose text starts with
         "keyboard" — that is the raw icon fallback text */
      doc.querySelectorAll('section[data-testid="stSidebar"] *').forEach(function(el){
        if(!el.children.length && /^keyboard/i.test((el.textContent||'').trim())){
          el.style.setProperty('display','none','important');
        }
      });
    } catch(e){}
  }
  purgeSidebarIcon();
  /* Re-run on every DOM mutation so Streamlit re-renders don't bring it back */
  try {
    var target = (window.parent||window).document.body;
    if(target) new MutationObserver(purgeSidebarIcon).observe(target,{childList:true,subtree:true});
  } catch(e){}
})();
</script>
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
# DATA LOADING — all schema knowledge lives in the framework data layer
# =========================================================
from src.ktc_framework.reporting.data_layer import (
    find_latest_run,
    load_run_data,
    create_method_mapping,
    filter_by_level,
    count_gt_missing,
)


@st.cache_data
def load_data(cache_key: str = "") -> Tuple[Dict, Dict, Dict]:
    """Load scores + per-run metrics from the latest run folder."""
    scores, per_run = load_run_data(find_latest_run())
    return scores, per_run, create_method_mapping(scores, per_run)


def apply_dashboard_filters(scores: Dict, per_run: Dict, mm: Dict,
                            selected_methods: List[str], level_range: tuple,
                            selected_samples: List[str]) -> Tuple[Dict, Dict, Dict]:
    """Apply sidebar method, level, and sample filters for every tab/report."""
    selected_methods = selected_methods or list(scores.keys())
    selected_samples = selected_samples if selected_samples is not None else ['A', 'B', 'C']
    sample_set = {str(s).strip().lower() for s in selected_samples}
    lvl_min, lvl_max = level_range
    scores_f, per_run_f, mm_f = {}, {}, {}

    for display_name in selected_methods:
        if display_name not in scores:
            continue
        ik = mm.get(display_name, display_name)
        kept = {}
        for run_key, entry in per_run.get(ik, {}).items():
            try:
                level_ok = lvl_min <= int(entry.get("level", 1)) <= lvl_max
            except Exception:
                level_ok = True
            sample_val = str(entry.get("sample", run_key)).strip().lower()
            sample_key = str(run_key).split("_")[-1].strip().lower()
            sample_ok = not sample_set or sample_val in sample_set or sample_key in sample_set
            if level_ok and sample_ok:
                kept[run_key] = entry
        if kept:
            metrics = dict(scores.get(display_name, {}))
            ktc_vals = [float(v.get("ktc_score", 0)) for v in kept.values()]
            if ktc_vals:
                mean_ktc = float(np.mean(ktc_vals))
                metrics["ktc_score"] = mean_ktc
                metrics["KTC score"] = mean_ktc
            scores_f[display_name] = metrics
            per_run_f[ik] = kept
            mm_f[display_name] = ik
    return scores_f, per_run_f, mm_f

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
def calculate_composite_score(metrics:Dict[str,float], weights:Dict[str,float]=None) -> float:
    ktc = metrics.get('KTC score', metrics.get('ktc_score', 0))
    return round(ktc * 100, 2)

def letter_grade(score:float) -> str:
    # Same thresholds as src/ktc_framework/metrics/composite_score.py
    return 'A' if score>=80 else 'B' if score>=60 else 'C' if score>=40 else 'D'

def all_methods(scores:Dict) -> List[str]:
    return list(scores.keys())+[m for m in st.session_state.get('custom_methods',[]) if m not in scores]

def mcol(idx:int) -> str:
    return PALETTE[idx % len(PALETTE)]

def inject_theme(dark: bool):
    """Add/remove eit-dark class on the stApp element so CSS vars cascade everywhere."""
    if dark:
        js = """
        (function(){
          var el=window.parent.document.querySelector('[data-testid="stApp"]');
          if(el){el.classList.add('eit-dark');}
          // also target iframes
          window.parent.document.querySelectorAll('iframe').forEach(function(f){
            try{
              var d=f.contentDocument||f.contentWindow.document;
              var a=d.querySelector('[data-testid="stApp"]')||d.body;
              if(a){a.classList.add('eit-dark');}
            }catch(e){}
          });
        })();
        """
    else:
        js = """
        (function(){
          window.parent.document.querySelectorAll('.eit-dark').forEach(function(el){
            el.classList.remove('eit-dark');
          });
          window.parent.document.querySelectorAll('iframe').forEach(function(f){
            try{
              var d=f.contentDocument||f.contentWindow.document;
              d.querySelectorAll('.eit-dark').forEach(function(el){el.classList.remove('eit-dark');});
            }catch(e){}
          });
        })();
        """
    st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)

# =========================================================
# BENCHMARK LAUNCHER — the dashboard drives the backend
# =========================================================
BENCH_LOG = Path("outputs/benchmark_log.txt")


def launch_benchmark(config_path: str | Path) -> bool:
    """Start `python example_usage.py --no-app --config <cfg>` in the background.

    The bridge writes latest.txt last, so the dashboard only flips to the new
    run once it is fully prepared.
    """
    proc = st.session_state.get('bench_proc')
    if proc is not None and proc.poll() is None:
        st.sidebar.warning("A benchmark is already running — wait for it to finish.")
        return False
    import os as _os
    BENCH_LOG.parent.mkdir(exist_ok=True)
    log = open(BENCH_LOG, "w", encoding="utf-8")
    _env = {**_os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    st.session_state.bench_proc = subprocess.Popen(
        [sys.executable, "example_usage.py", "--no-app", "--config", str(config_path)],
        stdout=log, stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent),
        env=_env,
    )
    st.session_state.bench_config = Path(config_path).stem
    return True


def write_runtime_config(method_name: str) -> Path:
    """Generate a one-method benchmark config for a runtime-registered method."""
    cfg_path = Path("configs") / f"runtime_{method_name}.yaml"
    cfg_path.write_text(
        "# Auto-generated by the dashboard Register button.\n"
        "data_plugin: KTCDataPlugin\n"
        "dataset_root: EvaluationData\n"
        "mesh_path: Codes_Matlab/Mesh_sparse.mat\n\n"
        "levels: [1, 2, 3, 4, 5, 6, 7]\n"
        "samples: [A, B, C]\n\n"
        f"methods:\n  - {method_name}\n\n"
        "method_plugin_paths:\n  - external_methods\n\n"
        "output_dir: outputs/\n",
        encoding="utf-8",
    )
    return cfg_path


def render_benchmark_status() -> None:
    """Sidebar status for the running/finished benchmark subprocess."""
    proc = st.session_state.get('bench_proc')
    if proc is not None:
        code = proc.poll()
        cfg = st.session_state.get('bench_config', '')
        if code is None:
            st.sidebar.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:var(--amb);margin:4px 0">RUNNING · {cfg}</div>',
                unsafe_allow_html=True)
            if st.sidebar.button("Refresh status", use_container_width=True, key="bench_refresh"):
                st.rerun()
        elif code == 0:
            st.session_state.bench_proc = None
            st.cache_data.clear()
            st.rerun()  # latest.txt now points at the new run
        else:
            st.session_state.bench_proc = None
            st.sidebar.error(f"Benchmark failed (exit {code}) — see log below.")
    if proc is not None and proc.poll() is None and BENCH_LOG.exists():
        tail = BENCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [ln for ln in tail if ln.strip()][-8:]
        if tail:
            log_html = "".join(
                f'<div style="white-space:pre;overflow-x:hidden;text-overflow:ellipsis">'
                f'{ln.replace("&","&amp;").replace("<","&lt;")}</div>'
                for ln in tail
            )
            st.sidebar.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                f'color:var(--tx3);background:var(--bg);border:1px solid var(--bd);'
                f'border-radius:6px;padding:8px 10px;margin-top:7px;line-height:1.6">'
                f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:var(--tx3);margin-bottom:5px">Benchmark log</div>'
                f'{log_html}</div>',
                unsafe_allow_html=True)


def render_bench_progress() -> None:
    """Full-width progress banner in the main area while a benchmark subprocess runs."""
    proc = st.session_state.get('bench_proc')
    if proc is None:
        return
    if proc.poll() is not None:
        # Benchmark just finished — do one final rerun so the dashboard flips
        # to the freshly prepared run, then stop auto-refreshing.
        if st.session_state.get('_bench_was_running'):
            st.session_state['_bench_was_running'] = False
            st.rerun()
        return
    st.session_state['_bench_was_running'] = True

    import re as _re
    completed, total = 0, 0
    cur_method, cur_level, cur_sample = "", "", ""

    if BENCH_LOG.exists():
        for line in BENCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _re.search(
                r'\[BENCH_PROGRESS\] completed=(\d+)/(\d+)'
                r'(?:\s+method=(\S+)\s+level=(\S+)\s+sample=(\S+))?',
                line,
            )
            if m:
                completed  = int(m.group(1))
                total      = int(m.group(2))
                cur_method = m.group(3) or ""
                cur_level  = m.group(4) or ""
                cur_sample = m.group(5) or ""

    if total == 0:
        try:
            import yaml as _yaml
            cfg_name  = st.session_state.get('bench_config', 'ktc_all_methods')
            cfg_path  = Path("configs") / f"{cfg_name}.yaml"
            if not cfg_path.exists():
                cfg_path = Path("configs/ktc_all_methods.yaml")
            cfg   = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            total = (len(cfg.get("methods", [])) *
                     len(cfg.get("levels",  [])) *
                     len(cfg.get("samples", [])))
        except Exception:
            total = 105  # 5 methods × 7 levels × 3 samples

    pct      = min(completed / total, 1.0) if total > 0 else 0.0
    pct_px   = f"{pct * 100:.1f}%"
    cfg_lbl  = st.session_state.get('bench_config', 'ktc_all_methods')
    cur_info = (f"&nbsp;·&nbsp; {cur_method} &nbsp;L{cur_level}/{cur_sample}"
                if cur_method else "&nbsp;·&nbsp; initialising…")

    st.markdown(
        f'<div style="background:var(--sur);border:1px solid var(--bd);border-radius:7px;'
        f'padding:14px 18px 12px;margin-bottom:14px;position:relative;overflow:hidden">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#2da44e,#0969da,#8250df)"></div>'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:10px">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;'
        f'color:var(--amb);letter-spacing:.05em">BENCHMARK RUNNING</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx3)">'
        f'{cfg_lbl}{cur_info}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:600;'
        f'color:var(--tx)">{completed}&nbsp;/&nbsp;{total} runs &nbsp;'
        f'<span style="color:var(--grn)">{pct*100:.0f}%</span></div>'
        f'</div>'
        f'<div style="background:var(--bd);border-radius:4px;height:8px;overflow:hidden;margin-bottom:8px">'
        f'<div style="background:linear-gradient(90deg,#2da44e,#0969da);height:8px;'
        f'border-radius:4px;width:{pct_px};transition:width .4s ease"></div>'
        f'</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx3)">'
        f'Progress updates automatically every 2s &nbsp;·&nbsp; '
        f'dashboard reloads when complete</div>'
        f'</div>',
        unsafe_allow_html=True)

    if st.button("↻  Refresh progress", key="main_bench_refresh"):
        st.rerun()
    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--bd);margin:4px 0 14px">',
        unsafe_allow_html=True)

    # Auto-refresh: while the benchmark subprocess is running, poll its progress
    # every 2s and rerun automatically so the counter updates without the user
    # having to click "Refresh progress".
    import time as _time
    _time.sleep(2)
    st.rerun()


def append_method_to_config(method_name: str,
                            config_path: Path = Path("configs/ktc_all_methods.yaml")) -> bool:
    """Insert a method name under the 'methods:' key, preserving YAML comments."""
    if not config_path.exists():
        return False
    text = config_path.read_text(encoding="utf-8")
    if f"- {method_name}" in text:
        return False
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.strip() == "methods:":
            lines.insert(i + 1, f"  - {method_name}\n")
            config_path.write_text("".join(lines), encoding="utf-8")
            return True
    return False


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    # ── Brand ────────────────────────────────────────────────
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    # Data-age label: read latest.txt mtime to tell user how fresh the dashboard is
    import time as _time
    _lt = Path("outputs/latest.txt")
    if _lt.exists():
        _age_s = _time.time() - _lt.stat().st_mtime
        if _age_s < 3600:
            _age_str = f"{int(_age_s//60)}m ago"
        elif _age_s < 86400:
            _age_str = f"{int(_age_s//3600)}h ago"
        else:
            _age_str = f"{int(_age_s//86400)}d ago"
        _live_sub = f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx3);margin-top:4px">data from {_age_str}</div>'
    else:
        _live_sub = ""

    st.sidebar.markdown(f"""
    <div style="border-bottom:1px solid var(--bd);padding-bottom:13px;margin-bottom:13px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:var(--tx);letter-spacing:.06em">EIT BENCH</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);margin-top:3px;letter-spacing:.08em">RECONSTRUCTION ANALYSIS</div>
      <div class="sb-live"><div class="ldot"></div>LIVE</div>
      {_live_sub}
    </div>
    """, unsafe_allow_html=True)

    # ── Dark mode toggle ──────────────────────────────────────
    st.sidebar.markdown("---")

    # ── Run Benchmark — one button drives the whole backend ───
    st.sidebar.markdown("## Run Benchmark")
    # Estimate runtime: read method count from YAML (~4 min per method on real data)
    try:
        import yaml as _yaml
        _cfg_path = Path("configs/ktc_all_methods.yaml")
        _n_methods = len(_yaml.safe_load(_cfg_path.read_text()).get("methods", []))
        _eta = f"~{_n_methods * 4} min"
        _n_str = str(_n_methods)
    except Exception:
        _eta = "several min"
        _n_str = "all"
    st.sidebar.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#848d97;'
        f'margin-bottom:7px;line-height:1.45">'
        f'<div>Config: <span style="color:var(--tx2)">ktc_all_methods.yaml</span></div>'
        f'<div>{_n_str} methods | est. <b style="color:#9a6700">{_eta}</b></div>'
        f'<div>Reloads automatically when done.</div></div>',
        unsafe_allow_html=True)
    if st.sidebar.button("Run all methods", use_container_width=True, key="run_full_btn"):
        if launch_benchmark(Path("configs/ktc_all_methods.yaml")):
            st.rerun()
    render_benchmark_status()

    st.sidebar.markdown("---")

    # ── Reset All Filters ─────────────────────────────────────
    ALL_METRICS_SIDEBAR = ['KTC Score']
    if st.sidebar.button("Reset All Filters", key="reset_all_btn", use_container_width=True):
        st.session_state.selected_metrics  = ALL_METRICS_SIDEBAR.copy()
        st.session_state.selected_methods  = st.session_state.get('_available_methods', []).copy()
        st.session_state.level_range       = (1, 7)
        st.session_state.selected_samples  = ['A', 'B', 'C']
        # clear checkbox widget state so they re-render checked
        for m in ALL_METRICS_SIDEBAR:
            st.session_state[f'metric_{m}'] = True
        for s in ['A','B','C']:
            st.session_state[f'samp_{s}'] = True
        for m in st.session_state.get('_available_methods', []):
            st.session_state[f'method_{m}'] = True
        st.rerun()

    st.sidebar.markdown("---")

    # ── Metric Selector ──────────────────────────────────────
    st.sidebar.markdown("## Metrics")
    ALL_METRICS_SIDEBAR = ['KTC Score']
    if 'selected_metrics' not in st.session_state:
        st.session_state.selected_metrics = ALL_METRICS_SIDEBAR.copy()
    for m in ALL_METRICS_SIDEBAR:
        checked = m in st.session_state.selected_metrics
        if st.sidebar.checkbox(m, value=checked, key=f"metric_{m}"):
            if m not in st.session_state.selected_metrics:
                st.session_state.selected_metrics.append(m)
        else:
            if m in st.session_state.selected_metrics:
                st.session_state.selected_metrics.remove(m)

    st.sidebar.markdown("---")

    # ── Method Selector ──────────────────────────────────────
    st.sidebar.markdown("## Methods")
    st.sidebar.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#848d97;'
        'margin-bottom:8px">Checkbox = show in charts &nbsp;·&nbsp; ▶ = run that method only</div>',
        unsafe_allow_html=True)
    available_methods = st.session_state.get('_available_methods', [])
    if 'selected_methods' not in st.session_state:
        st.session_state.selected_methods = available_methods.copy()

    if available_methods:
        for m in available_methods:
            # Abbreviate for display so long names don't truncate
            short = m.replace("Reconstruction","Recon").replace("Difference","Diff").replace("Projection","Proj")
            c_chk, c_run = st.sidebar.columns([4, 1])
            with c_chk:
                checked = m in st.session_state.selected_methods
                new_val = st.checkbox(short, value=checked, key=f"method_{m}",
                                      help=m)  # full name in tooltip
                if new_val and m not in st.session_state.selected_methods:
                    st.session_state.selected_methods.append(m)
                elif not new_val and m in st.session_state.selected_methods:
                    st.session_state.selected_methods.remove(m)
            with c_run:
                if st.button("▶", key=f"run_m_{m}",
                             help=f"Benchmark {m} only\n→ configs/runtime_{m}.yaml"):
                    if launch_benchmark(write_runtime_config(m)):
                        st.rerun()
    else:
        st.sidebar.markdown('<div style="font-size:11px;color:var(--tx3)">Loading methods...</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # ── Level Filter ─────────────────────────────────────────
    st.sidebar.markdown("## Level Filter")
    if 'level_range' not in st.session_state:
        st.session_state.level_range = (1, 7)
    cur_min, cur_max = st.session_state.level_range
    c_min, c_max = st.sidebar.columns(2)
    with c_min:
        lvl_min = st.selectbox(
            "From", list(range(1, 8)), index=max(0, min(6, int(cur_min) - 1)),
            key="sb_level_min")
    with c_max:
        lvl_max = st.selectbox(
            "To", list(range(1, 8)), index=max(0, min(6, int(cur_max) - 1)),
            key="sb_level_max")
    if lvl_min > lvl_max:
        lvl_min, lvl_max = lvl_max, lvl_min
    st.session_state.level_range = (lvl_min, lvl_max)
    st.sidebar.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:var(--tx3);margin-top:-4px">Showing levels {lvl_min}-{lvl_max}</div>',
        unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # ── Sample Filter ─────────────────────────────────────────
    st.sidebar.markdown("## Samples")
    if 'selected_samples' not in st.session_state:
        st.session_state.selected_samples = ['A','B','C']
    for s in ['A','B','C']:
        checked = s in st.session_state.selected_samples
        if st.sidebar.checkbox(f"Sample {s}", value=checked, key=f"samp_{s}"):
            if s not in st.session_state.selected_samples:
                st.session_state.selected_samples.append(s)
        else:
            if s in st.session_state.selected_samples:
                st.session_state.selected_samples.remove(s)

    st.sidebar.markdown("---")

    # Data files — checked inside the ACTIVE run folder, not the project root
    active_run = find_latest_run()
    required_files = ["scores.json", "per_run_metrics.json"]
    missing_files = [lbl for lbl in required_files if not (active_run / lbl).exists()]
    data_color = "#cf222e" if missing_files else "var(--tx3)"
    data_status = "Missing: " + ", ".join(missing_files) if missing_files else "Data files: OK"
    st.sidebar.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
        f'color:{data_color};margin:6px 0 8px;line-height:1.35">{data_status}</div>',
        unsafe_allow_html=True)

    # ── Add / Register external methods ──────────────────────
    st.sidebar.markdown("## Add Method")

    # ── 1. Scan button — always visible ──────────────────────
    #    Picks up any .py already in external_methods/ so the user
    #    doesn't have to re-upload files that are already on disk.
    from src.ktc_framework.registry import (
        get_method as _get_method, list_methods as _list_methods,
        load_external_methods as _load_ext,
    )
    if 'uploaded_methods' not in st.session_state:
        st.session_state.uploaded_methods = {}

    if st.sidebar.button("Scan external_methods/", key="scan_ext_btn",
                         use_container_width=True,
                         help="Detect any @register_method classes already in external_methods/"):
        ext_dir = Path("external_methods")
        ext_dir.mkdir(exist_ok=True)
        py_files = list(ext_dir.glob("*.py"))
        if py_files:
            before = set(_list_methods())
            try:
                _load_ext([str(ext_dir)])
                new_methods = sorted(set(_list_methods()) - before)
                for nm in new_methods:
                    fname = next(
                        (f.name for f in py_files
                         if nm.lower() in f.stem.lower()), py_files[0].name
                    )
                    st.session_state.uploaded_methods[nm] = fname
                if new_methods:
                    st.sidebar.success(f"Found: {', '.join(new_methods)}")
                else:
                    st.sidebar.info("No new methods found in external_methods/")
            except Exception as exc:
                st.sidebar.error(f"Scan failed: {exc}")
        else:
            st.sidebar.info("external_methods/ is empty — upload a .py file below.")

    # ── 2. Registered methods — always show with action buttons ─
    if st.session_state.uploaded_methods:
        st.sidebar.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            'color:var(--tx3);margin:6px 0 4px;text-transform:uppercase;'
            'letter-spacing:.1em">Registered plugins</div>',
            unsafe_allow_html=True)
        for nm, fname in list(st.session_state.uploaded_methods.items()):
            st.sidebar.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:var(--tx2);padding:3px 0 1px">• {nm}<br>'
                f'<span style="font-size:10px;color:var(--tx3)">{fname}</span></div>',
                unsafe_allow_html=True)
            ca, cb, cc = st.sidebar.columns([2, 1, 1])
            if ca.button("Register", key=f"reg_{nm}",
                         help=f"Run benchmark for {nm} only → configs/runtime_{nm}.yaml"):
                if launch_benchmark(write_runtime_config(nm)):
                    st.rerun()
            if cb.button("+all", key=f"cfg_{nm}",
                         help="Add to ktc_all_methods.yaml for future full runs"):
                if append_method_to_config(nm):
                    st.sidebar.success(f"{nm} added to ktc_all_methods.yaml")
                else:
                    st.sidebar.info("Already in config")
            if cc.button("x", key=f"rm_up_{nm}", help="Remove plugin file from disk"):
                (Path("external_methods") / fname).unlink(missing_ok=True)
                del st.session_state.uploaded_methods[nm]
                st.rerun()
    else:
        st.sidebar.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            'color:var(--tx3);margin:3px 0 5px;line-height:1.35">'
            'No plugins yet. Upload .py or scan.</div>',
            unsafe_allow_html=True)

    # ── 3. Upload new plugin ─────────────────────────────────
    st.sidebar.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'color:var(--tx3);margin:8px 0 5px;text-transform:uppercase;'
        'letter-spacing:.1em">Upload new plugin</div>',
        unsafe_allow_html=True)
    up = st.sidebar.file_uploader("Upload plugin (.py)", type=["py"], key="method_upload",
                                  label_visibility="collapsed")
    if up is not None:
        sig = f"{up.name}:{up.size}"
        if st.session_state.get('_last_method_upload') != sig:
            st.session_state['_last_method_upload'] = sig
            dest_dir = Path("external_methods")
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / Path(up.name).name
            before = set(_list_methods())
            dest.write_bytes(up.getbuffer())
            try:
                _load_ext([str(dest_dir)])
                new_methods = sorted(set(_list_methods()) - before)
                if new_methods:
                    for nm in new_methods:
                        if not callable(getattr(_get_method(nm), "reconstruct", None)):
                            st.sidebar.warning(f"{nm} has no reconstruct(batch) — will fail at run time.")
                        st.session_state.uploaded_methods[nm] = dest.name
                    st.sidebar.success(f"Registered: {', '.join(new_methods)}")
                else:
                    dest.unlink(missing_ok=True)
                    st.sidebar.warning("No @register_method class found — file removed.")
            except Exception as exc:
                dest.unlink(missing_ok=True)
                st.sidebar.error(f"Rejected {dest.name}: {exc}")

    # ── Export ───────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Export")
    if st.sidebar.button("Export PDF Report", use_container_width=True, key="pdf_sidebar_btn"):
        st.session_state['_trigger_pdf'] = True
    pdf_export_slot = st.sidebar.empty()

    # ── Run selector ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Run History")
    runs_root = Path("outputs")
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True) if runs_root.exists() else []

    if run_dirs:
        run_names = [d.name for d in run_dirs]
        # Current loaded run name
        current_run = find_latest_run().name
        st.sidebar.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--grn);margin:4px 0">'
            f'Active: {current_run}</div>', unsafe_allow_html=True)

        default_idx = run_names.index(current_run) if current_run in run_names else 0
        chosen_run = st.sidebar.selectbox(
            "Load run:", run_names, index=default_idx, key="selected_run",
            label_visibility="collapsed")

        if st.sidebar.button("Load selected run", use_container_width=True, key="load_run_btn"):
            selected_path = runs_root / chosen_run
            (runs_root / "latest.txt").write_text(str(selected_path))
            st.cache_data.clear()
            st.rerun()

        # Show scores for the selected run (not just active) as preview
        preview_scores_path = runs_root / chosen_run / "scores.json"
        if preview_scores_path.exists() and chosen_run != current_run:
            with open(preview_scores_path) as f:
                preview = json.load(f)
            st.sidebar.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx3);margin:5px 0 3px">Preview: {chosen_run}</div>',
                unsafe_allow_html=True)
            for method, mets in preview.items():
                ktc = mets.get('KTC score', mets.get('ktc_score', '—'))
                ktc_str = f"{ktc:.4f}" if isinstance(ktc, float) else str(ktc)
                st.sidebar.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx2);padding:2px 0">'
                    f'  {method}: KTC={ktc_str}</div>',
                    unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            '<div style="font-size:11px;color:var(--tx3);margin:4px 0">No runs yet.<br>Run example_usage.py first.</div>',
            unsafe_allow_html=True)
    return pdf_export_slot

# =========================================================
# VIEW 1 — LEADERBOARD  (original logic)
# =========================================================
def view_leaderboard(scores:Dict, per_run:Dict, sel_metrics:list=None, mm:Dict=None, level_range:tuple=(1,7)):
    if sel_metrics is None:
        sel_metrics = ['KTC Score']
    if mm is None:
        mm = {}
    lvl_min, lvl_max = level_range

    if lvl_min != 1 or lvl_max != 7:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--amb);'
            f'margin-bottom:6px">Filtered: levels {lvl_min}–{lvl_max} &nbsp;·&nbsp; '
            f'{len(scores)} method(s) selected</div>', unsafe_allow_html=True)

    if 'method_colors' not in st.session_state: st.session_state.method_colors = {}
    leaderboard_data = []
    for i,(method_name,metrics) in enumerate(scores.items()):
        if method_name not in st.session_state.method_colors:
            st.session_state.method_colors[method_name] = mcol(len(st.session_state.method_colors))
        # Average KTC over the level-filtered per-run entries when available;
        # the precomputed scores.json average covers all levels.
        ik = mm.get(method_name)
        entries = filter_by_level(per_run.get(ik, {}), lvl_min, lvl_max) if ik else {}
        if entries:
            ktc_val = float(np.mean([e.get('ktc_score', 0.0) for e in entries.values()]))
        else:
            ktc_val = metrics.get('KTC score', metrics.get('ktc_score', 0))
        comp  = calculate_composite_score({'ktc_score': ktc_val})
        grade = letter_grade(comp)
        leaderboard_data.append({
            'Method':method_name,'Composite Score':comp,'Grade':grade,
            'Color':st.session_state.method_colors[method_name],
            'KTC Score': ktc_val,
        })
    leaderboard_data.sort(key=lambda x: x['Composite Score'], reverse=True)
    df = pd.DataFrame(leaderboard_data)

    # KPI cards — exact mockup spec
    gc = df['Grade'].value_counts()
    kpis = [
        (f"{df.iloc[0]['Composite Score']:.1f}", "TOP SCORE",  df.iloc[0]['Method'][:22], "--c1"),
        (f"{df['Composite Score'].mean():.1f}",  "AVG SCORE",  f"σ = {df['Composite Score'].std():.1f}", "--c2"),
        (str(len(df)),                           "METHODS",    f"{gc.get('A',0)}A  {gc.get('B',0)}B  {gc.get('C',0)}C  {gc.get('D',0)}D", "--c3"),
        (f"{df['KTC Score'].max():.4f}",         "BEST KTC",   "higher is better", "--c4"),
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
                           f"KTC: {row['KTC Score']:.4f}<br><extra></extra>")
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

    # Table — filter columns by selected metrics in real time
    st.markdown('<div class="slbl">DETAILED METRICS</div>', unsafe_allow_html=True)

    # Build full display_df first
    display_df = df.drop(columns=['Color']).copy()
    display_df['Composite Score'] = display_df['Composite Score'].round(2)
    if 'KTC Score' in display_df.columns:
        display_df['KTC Score'] = display_df['KTC Score'].round(4)

    # Map sidebar metric names to table column names
    METRIC_COL_MAP = {
        'KTC Score':  'KTC Score',
    }
    # Always keep Method, Composite Score, Grade; filter the rest
    always_cols = ['Method','Composite Score','Grade']
    optional_cols = [METRIC_COL_MAP[m] for m in sel_metrics
                     if METRIC_COL_MAP.get(m) and METRIC_COL_MAP[m] in display_df.columns]
    show_cols = always_cols + optional_cols
    filtered_df = display_df[[c for c in show_cols if c in display_df.columns]]

    def grade_color(grade):
        return {'A':'#dafbe1','B':'#ddf4ff','C':'#fff8c5','D':'#ffebe9'}.get(grade,'')

    pc = st.session_state.get('_pcolors', {})
    row_bg    = pc.get('bg', '#f6f8fa')
    cell_col  = pc.get('text', '#1f2328') if st.session_state.get('dark_mode') else '#1f2328'
    hdr_bg    = pc.get('bg', '#f6f8fa')
    hdr_col   = pc.get('text', '#848d97')
    brd       = '#30363d' if st.session_state.get('dark_mode') else '#d0d7de'
    sep       = '#30363d' if st.session_state.get('dark_mode') else '#f6f8fa'

    rows_html = ''
    for _, row in filtered_df.iterrows():
        gc  = grade_color(row['Grade'])
        bg  = f'background:{gc};' if gc else f'background:{pc.get("bg","")};'
        rows_html += f'<tr style="{bg}">'
        for col in filtered_df.columns:
            rows_html += (f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                          f'padding:5px 8px;border-bottom:1px solid {sep};color:{cell_col}">'
                          f'{row[col]}</td>')
        rows_html += '</tr>'

    headers_html = ''.join(
        f'<th style="font-family:\'JetBrains Mono\',monospace;font-size:8px;text-transform:uppercase;'
        f'letter-spacing:.1em;color:{hdr_col};padding:4px 8px;border-bottom:1px solid {brd};'
        f'background:{hdr_bg}">{c}</th>' for c in filtered_df.columns)
    st.markdown(
        f'<div style="border:1px solid {brd};border-radius:7px;overflow:hidden">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>{headers_html}</tr></thead><tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True)

# =========================================================
# VIEW 2 — DEGRADATION  (original logic)
# =========================================================
def view_degradation_curve(scores:Dict, per_run:Dict, mm:Dict, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    lvl_min, lvl_max = level_range
    dm = all_methods(scores)
    # Default: all methods so GroundTruthOracle is always visible as the reference ceiling
    chosen = st.multiselect("Select methods:", dm,
        default=[m for m in dm if m not in st.session_state.get('custom_methods',[])])

    if not chosen:
        st.info("Select at least one method.")
        return

    if lvl_min != 1 or lvl_max != 7:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--amb);margin-bottom:6px">'
            f'Showing levels {lvl_min}–{lvl_max}</div>', unsafe_allow_html=True)

    pc  = st.session_state.get('_pcolors', {})
    pb  = pc.get('bg',     '#f6f8fa')
    pp  = pc.get('paper',  'rgba(0,0,0,0)')
    pg  = pc.get('grid',   '#d0d7de')
    pt  = pc.get('text',   '#848d97')
    pleg= pc.get('legend', 'rgba(255,255,255,.9)')

    fig = go.Figure()
    stats = []
    for i, disp_name in enumerate(chosen):
        ik = mm.get(disp_name)
        if not ik or ik not in per_run: continue
        samps = filter_by_level(per_run[ik], lvl_min, lvl_max)
        if not samps:
            continue

        # x = difficulty level; y = mean KTC over that level's samples;
        # band = ±1 std across the samples within each level
        levels   = sorted({int(e['level']) for e in samps.values()})
        by_level = {lv: [e['ktc_score'] for e in samps.values() if int(e['level']) == lv]
                    for lv in levels}
        ktc = [float(np.mean(by_level[lv])) for lv in levels]
        sds = [float(np.std(by_level[lv]))  for lv in levels]
        x   = levels
        c   = mcol(i)
        mu  = np.mean(ktc); sd = np.std(ktc)
        upper = [v + s for v, s in zip(ktc, sds)]; lower = [max(0, v - s) for v, s in zip(ktc, sds)]
        # Confidence band
        fig.add_trace(go.Scatter(
            x=x+x[::-1], y=upper+lower[::-1],
            fill='toself', fillcolor=hex_to_rgba(c, 0.10),
            line=dict(width=0), showlegend=False, hoverinfo='skip'))
        # Main line
        fig.add_trace(go.Scatter(x=x, y=ktc, mode='lines+markers', name=disp_name,
            line=dict(width=2.5, color=c),
            marker=dict(size=7, color=c, line=dict(width=2, color='#ffffff')),
            hovertemplate=f"<b>{disp_name}</b><br>Level: %{{x}}<br>KTC: %{{y:.4f}}<extra></extra>"))
        # Horizontal mean line
        fig.add_trace(go.Scatter(
            x=[min(x), max(x)], y=[mu, mu],
            mode='lines', line=dict(width=1, color=c, dash='dot'),
            showlegend=False, hoverinfo='skip'))
        stats.append({'Method':disp_name,'Mean KTC':mu,'Std Dev':sd,
                      'Min':np.min(ktc),'Max':np.max(ktc),'Range':np.max(ktc)-np.min(ktc)})

    fig.update_layout(
        title=f"KTC Score — Levels {lvl_min}–{lvl_max}",
        xaxis_title="Difficulty Level", yaxis_title="KTC Score (higher = better)",
        height=420, hovermode='x unified',
        paper_bgcolor=pp, plot_bgcolor=pb,
        font=dict(family="JetBrains Mono,monospace", color=pt, size=9),
        xaxis=dict(gridcolor=pg, linecolor=pg),
        yaxis=dict(gridcolor=pg, linecolor=pg),
        legend=dict(bgcolor=pleg, bordercolor=pg, borderwidth=1, font=dict(size=9)),
        margin=dict(l=0, r=0, t=36, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    if stats:
        st.markdown('<div class="slbl">KTC STATISTICS</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(stats).round(4), use_container_width=True, hide_index=True)

        best  = max(stats, key=lambda r: r['Mean KTC'])
        worst = max(stats, key=lambda r: r['Std Dev'])
        st.markdown(
            f'<div style="background:var(--grn-bg);border:1px solid var(--grn-bd);border-radius:7px;padding:10px 14px;margin-top:8px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:600;color:var(--grn);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">KEY INSIGHTS</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:var(--tx);line-height:1.6">'
            f'- Best average KTC: <b>{best["Method"]}</b> ({best["Mean KTC"]:.4f})<br>'
            f'- Most variability: <b>{worst["Method"]}</b> (σ = {worst["Std Dev"]:.4f})<br>'
            f'- Shaded bands = ±1 std deviation. Narrower = more consistent performance.'
            f'</div></div>',
            unsafe_allow_html=True)

# =========================================================
# VIEW 3 — COMPARISON  (original logic)
# =========================================================
def view_comparison(scores:Dict, per_run:Dict, mm:Dict, sel_metrics:list=None, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    # Map sidebar metric display names to internal per_run keys
    METRIC_DISPLAY_MAP = {
        'KTC Score':  'ktc_score',
    }
    # Which internal keys to show — driven by sel_metrics
    if sel_metrics:
        show_keys = [METRIC_DISPLAY_MAP[sm] for sm in sel_metrics if sm in METRIC_DISPLAY_MAP]
    else:
        show_keys = list(METRIC_DISPLAY_MAP.values())

    lvl_min, lvl_max = level_range

    dm = all_methods(scores)
    fi = list(per_run.keys())[0] if per_run else None
    entries = filter_by_level(per_run[fi], lvl_min, lvl_max) if fi else {}
    if not entries and fi:
        entries = per_run[fi]  # fallback: show all if filter leaves nothing
    levels_avail = sorted({int(e['level']) for e in entries.values()}) or ALL_LEVELS

    c1,c2,c3,c4 = st.columns(4)
    m1  = c1.selectbox("Method 1:", dm, index=0)
    m2  = c2.selectbox("Method 2:", dm, index=min(1,len(dm)-1))
    lvl = c3.selectbox("Level:", levels_avail, index=0)
    samps = sorted({e['sample'] for e in entries.values() if int(e['level']) == lvl})
    sid = c4.selectbox("Sample:", samps) if samps else None
    run_key = f"L{lvl}_{sid}"

    m1i = mm.get(m1); m2i = mm.get(m2)
    p1  = m1 in st.session_state.get('custom_methods', [])
    p2  = m2 in st.session_state.get('custom_methods', [])
    if p1: st.info(f"{m1} is a custom method — connect its backend to see metrics.")
    if p2: st.info(f"{m2} is a custom method — connect its backend to see metrics.")

    met1 = per_run.get(m1i or m1, {}).get(run_key, {}) if not p1 else {}
    met2 = per_run.get(m2i or m2, {}).get(run_key, {}) if not p2 else {}

    # Metric comparison table — only show selected metrics
    st.markdown('<div class="slbl">METRIC COMPARISON</div>', unsafe_allow_html=True)
    keys_to_show = [k for k in show_keys if k in met1 or k in met2]
    if not keys_to_show:
        keys_to_show = list(met1.keys())  # fallback: show everything
    comp_data = [{'Metric': k.replace('_',' ').title(),
                  m1: met1.get(k,0), m2: met2.get(k,0),
                  'Diff': abs(met1.get(k,0)-met2.get(k,0))}
                 for k in keys_to_show]
    comp_df = pd.DataFrame(comp_data)
    for col in [m1, m2, 'Diff']:
        if col in comp_df.columns:
            comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Visual comparison — images from backend
    st.markdown('<div class="slbl">VISUAL COMPARISON</div>', unsafe_allow_html=True)
    panel = load_comparison_panel(sid)
    if panel:
        st.markdown(f"All Methods — Sample {sid}")
        st.image(panel, use_container_width=True)
        st.markdown("---")
    imgs = load_images_for_sample(sid, level=lvl)
    if imgs:
        ic1, ic2 = st.columns(2)
        i1 = next((img for k,img in imgs.items() if m1i and m1i.lower() in k.lower()), None)
        i2 = next((img for k,img in imgs.items() if m2i and m2i.lower() in k.lower()), None)
        with ic1:
            st.markdown(f"**{m1}**")
            if i1: st.image(i1, use_container_width=True)
            else:  st.info(f"No image for {m1}")
        with ic2:
            st.markdown(f"**{m2}**")
            if i2: st.image(i2, use_container_width=True)
            else:  st.info(f"No image for {m2}")
    elif not panel:
        st.info("No visualization images found. Run example_usage.py to generate images.")

# =========================================================
# VIEW 4 — FAILURE GALLERY  (original logic)
# =========================================================
def view_failure_gallery(scores:Dict, per_run:Dict, mm:Dict, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    lvl_min, lvl_max = level_range

    if lvl_min != 1 or lvl_max != 7:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--amb);margin-bottom:6px">'
            f'Filtered: levels {lvl_min}–{lvl_max}</div>', unsafe_allow_html=True)

    # ── Pass 1: collect all per-run data for KPI counts, root cause, summary ─
    # KTC score is higher = better: worst runs have the LOWEST scores.
    all_ktc      = []
    grade_counts = {'A':0,'B':0,'C':0,'D':0}
    root_causes  = []
    summary_rows = []

    for disp in scores.keys():
        ik = mm.get(disp)
        if not ik or ik not in per_run:
            continue
        samples_f = filter_by_level(per_run[ik], lvl_min, lvl_max)
        if not samples_f:
            continue
        ktc_vals = [v['ktc_score'] for v in samples_f.values()]
        worst3   = sorted(samples_f.items(), key=lambda x: x[1]['ktc_score'])[:3]

        # summary row (one per method)
        summary_rows.append({
            'Method':              disp,
            'Worst KTC':           min(ktc_vals),
            'Mean KTC':            np.mean(ktc_vals),
            'Failures (KTC<0.3)':  sum(1 for v in ktc_vals if v < 0.3),
            'Worst Samples':       ', '.join(str(s) for s, _ in worst3),
        })

        # per-sample KPI + failing-run list (filtered)
        for sid, m in samples_f.items():
            ktc = m.get('ktc_score', 0)
            all_ktc.append(ktc)
            # grade/composite come from the backend JSON; compute only if absent
            comp = m.get('composite_score', calculate_composite_score({'ktc_score': ktc}))
            g = m.get('grade', letter_grade(comp))
            grade_counts[g] = grade_counts.get(g, 0) + 1
            if g in ('C', 'D'):
                root_causes.append({
                    'Method': disp, 'Sample': sid, 'Grade': g,
                    'KTC': round(ktc, 4), 'Composite': round(comp, 1),
                })

    # ── KPI overview cards ────────────────────────────────────
    total    = len(all_ktc)
    kpi2     = [
        (str(total),             "TOTAL RUNS",         "all methods × samples", "--c3"),
        (str(grade_counts['D']), "D-GRADE FAILED",     "composite < 40",        "--c5"),
        (str(grade_counts['C']), "C-GRADE STRUGGLING", "marginal performance",  "--c4"),
        (str(grade_counts['B']), "B-GRADE STRONG",     "solid performance",     "--c1"),
    ]
    kpi2_html = '<div class="kpi-row">'
    for num, lbl, sub, kc in kpi2:
        kpi2_html += (f'<div class="kpi" style="--kc:var({kc})">'
                      f'<div class="kpi-n">{num}</div>'
                      f'<div class="kpi-l">{lbl}</div>'
                      f'<div class="kpi-s">{sub}</div></div>')
    kpi2_html += '</div>'
    st.markdown(kpi2_html, unsafe_allow_html=True)

    # ── Failure summary table (one row per method) ────────────
    if summary_rows:
        st.markdown('<div class="slbl">FAILURE SUMMARY</div>', unsafe_allow_html=True)
        fdf = pd.DataFrame(summary_rows)
        fdf['Worst KTC'] = fdf['Worst KTC'].round(4)
        fdf['Mean KTC']  = fdf['Mean KTC'].round(4)
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        # Explain negative KTC if any method scores below zero
        if any(r['Worst KTC'] < 0 for r in summary_rows):
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                'color:#9a6700;background:#fff8c5;border:1px solid #f0d847;border-radius:5px;'
                'padding:5px 10px;margin-top:4px">'
                'KTC < 0 means the reconstruction is <b>worse than the all-water baseline</b> '
                '(random noise artefacts confuse the SSIM metric). '
                'Composite scores below zero are expected for poorly-initialised solvers.</div>',
                unsafe_allow_html=True)

    # ── Root cause analysis table ─────────────────────────────
    if root_causes:
        st.markdown('<div class="slbl">ROOT CAUSE ANALYSIS — FAILING RUNS (C + D GRADE)</div>', unsafe_allow_html=True)
        rc_df = pd.DataFrame(root_causes)

        def _rc_color(grade):
            return '#ffebe9' if grade == 'D' else '#fff8c5'

        rows_html2 = ''
        for _, row in rc_df.iterrows():
            bg = _rc_color(row['Grade'])
            rows_html2 += f'<tr style="background:{bg}">'
            for col in rc_df.columns:
                rows_html2 += (f'<td style="font-family:\'JetBrains Mono\',monospace;'
                               f'font-size:9px;padding:5px 8px;border-bottom:1px solid #f6f8fa;'
                               f'color:#1f2328">{row[col]}</td>')
            rows_html2 += '</tr>'
        hdrs2 = ''.join(
            f'<th style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'text-transform:uppercase;letter-spacing:.1em;color:#848d97;'
            f'padding:4px 8px;border-bottom:1px solid #d0d7de;background:#f6f8fa">{c}</th>'
            for c in rc_df.columns)
        st.markdown(
            f'<div style="border:1px solid #d0d7de;border-radius:7px;overflow:hidden">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{hdrs2}</tr></thead><tbody>{rows_html2}</tbody></table></div>',
            unsafe_allow_html=True)

        # Patterns by level / sample
        st.markdown('<div class="slbl" style="margin-top:12px">FAILURE PATTERNS BY SAMPLE</div>', unsafe_allow_html=True)
        level_counts = {}
        for r in root_causes:
            key = str(r['Sample'])
            level_counts[key] = level_counts.get(key, 0) + 1
        if level_counts:
            lc_df = pd.DataFrame([{'Sample': k, 'Failures': v}
                                   for k, v in sorted(level_counts.items())])
            st.dataframe(lc_df, use_container_width=True, hide_index=True)

        # Patterns by method
        st.markdown('<div class="slbl" style="margin-top:8px">FAILURE PATTERNS BY METHOD</div>', unsafe_allow_html=True)
        method_fail = {}
        for r in root_causes:
            method_fail[r['Method']] = method_fail.get(r['Method'], 0) + 1
        if method_fail:
            mf_df = pd.DataFrame([{'Method': k, 'Failures': v}
                                   for k, v in sorted(method_fail.items(), key=lambda x: -x[1])])
            st.dataframe(mf_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Per-method worst-3 cards (lowest KTC = worst, level-filtered) ─
    run_dir = find_latest_run()
    for disp in scores.keys():
        ik = mm.get(disp)
        if not ik or ik not in per_run:
            st.info(f"No per-run data for {disp}")
            continue
        samples_f2 = filter_by_level(per_run[ik], lvl_min, lvl_max)
        if not samples_f2:
            continue
        worst3 = sorted(samples_f2.items(), key=lambda x: x[1]['ktc_score'])[:3]

        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:600;'
            f'color:#1f2328;padding:6px 0 4px;border-bottom:1px solid #d0d7de;margin-bottom:8px">'
            f'{disp}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        max_ktc = max((m['ktc_score'] for _, m in worst3), default=1) or 1
        for idx, (sid, metrics) in enumerate(worst3):
            with cols[idx]:
                pct = int(metrics['ktc_score'] / max_ktc * 100)
                bc  = ['#cf222e', '#bf8700', '#0969da'][idx]
                st.markdown(f"""<div class="fcard">
                  <div class="frank">#{idx+1} WORST · SAMPLE {sid}</div>
                  <div class="fktc">{metrics['ktc_score']:.4f}</div>
                  <div class="flbl">KTC SCORE</div>
                  <div class="fbar"><div class="fbar-f" style="width:{pct}%;background:{bc}"></div></div>
                </div>""", unsafe_allow_html=True)
                img_shown = False
                lv = int(metrics.get('level', 1))
                sp = metrics.get('sample', sid)
                for p in [run_dir / "reconstructions" / f"level_{lv}" / f"sample_{sp}" / f"{ik}.png",
                          run_dir / "error_overlays" / f"{ik}_sample_{sp}.png"]:
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
def view_radar_chart(scores:Dict, per_run:Dict, sel_metrics:list=None):
    if not scores:
        st.warning("No scores available.")
        return

    # Map sidebar display names → scores.json key names
    SIDEBAR_TO_SCORE_KEY = {
        'KTC Score':  ['KTC score','ktc_score'],
    }
    avail_score_keys = sorted({k for m in scores.values() for k in m.keys()})

    # If sel_metrics provided, use those as default (filtered to what actually exists)
    if sel_metrics:
        default_keys = []
        for sm in sel_metrics:
            for candidate in SIDEBAR_TO_SCORE_KEY.get(sm, []):
                if candidate in avail_score_keys:
                    default_keys.append(candidate)
                    break
        if not default_keys:
            default_keys = avail_score_keys[:min(7,len(avail_score_keys))]
    else:
        RADAR_PREFERRED = ['KTC score','ktc_score']
        default_keys = [m for m in RADAR_PREFERRED if m in avail_score_keys]
        if not default_keys:
            default_keys = avail_score_keys[:min(7,len(avail_score_keys))]

    chosen = st.multiselect("Choose metrics (7-axis):", avail_score_keys,
        default=default_keys if default_keys else avail_score_keys[:min(7,len(avail_score_keys))])
    if not chosen:
        st.info("Select at least one metric.")
        return

    # Radar is only meaningful with 2+ axes. With 1 metric it degrades to dots on a line.
    if len(chosen) == 1:
        st.info(
            f"Radar chart needs 2+ metrics to draw a polygon — currently only **{chosen[0]}** is available. "
            "Showing a bar comparison instead. Add more metrics (Dice, IoU…) to unlock the full radar."
        )
        metric = chosen[0]
        pc2 = st.session_state.get('_pcolors', {})
        bar_data = [(name, max(0, metrics.get(metric, 0))) for name, metrics in scores.items()]
        bar_data.sort(key=lambda x: x[1], reverse=True)
        fig2 = go.Figure()
        for i, (name, val) in enumerate(bar_data):
            fig2.add_trace(go.Bar(
                name=name, x=[name], y=[val],
                marker_color=mcol(i),
                text=f"{val:.4f}", textposition='outside',
                textfont=dict(family="JetBrains Mono", size=9),
            ))
        fig2.update_layout(
            xaxis_title="Method", yaxis_title=metric.replace('_', ' ').title(),
            yaxis_range=[0, 1.1], showlegend=False, height=360,
            paper_bgcolor=pc2.get('paper', 'rgba(0,0,0,0)'),
            plot_bgcolor=pc2.get('bg', '#f6f8fa'),
            font=dict(family="JetBrains Mono,monospace", color=pc2.get('text', '#848d97'), size=9),
            xaxis=dict(gridcolor=pc2.get('grid', '#d0d7de'), linecolor=pc2.get('grid', '#d0d7de')),
            yaxis=dict(gridcolor=pc2.get('grid', '#d0d7de'), linecolor=pc2.get('grid', '#d0d7de')),
            margin=dict(l=0, r=10, t=20, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)
        return

    fig = go.Figure()
    for i,(name,metrics) in enumerate(scores.items()):
        # KTC is already higher = better on [0, 1]; clamp negatives (worse than
        # all-water baseline) to 0 for the polar axis.
        vals = [max(0, metrics.get(m,0)) for m in chosen]
        vals.append(vals[0])
        cats = [m.replace('_',' ').title() for m in chosen]; cats.append(cats[0])
        c = mcol(i)
        fig.add_trace(go.Scatterpolar(r=vals,theta=cats,fill='toself',name=name,
            line_color=c,fillcolor=hex_to_rgba(c, 0.13)))
    pc = st.session_state.get('_pcolors',{})
    fig.update_layout(
        polar=dict(bgcolor=pc.get('bg','#f6f8fa'),
            radialaxis=dict(visible=True,range=[0,1],gridcolor=pc.get('grid','#d0d7de'),linecolor=pc.get('grid','#d0d7de'),tickfont=dict(size=8,color=pc.get('text','#848d97'))),
            angularaxis=dict(gridcolor=pc.get('grid','#d0d7de'),linecolor=pc.get('grid','#d0d7de'),tickfont=dict(size=10,color=pc.get('text','#848d97')))),
        showlegend=True,height=560,title="Method Performance Across Selected Metrics",
        paper_bgcolor=pc.get('paper','rgba(0,0,0,0)'),
        font=dict(family="JetBrains Mono,monospace",color=pc.get('text','#848d97'),size=9),
        legend=dict(bgcolor=pc.get('legend','rgba(255,255,255,.9)'),bordercolor=pc.get('grid','#d0d7de'),borderwidth=1,font=dict(size=9)),
        margin=dict(l=55,r=55,t=45,b=55))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="slbl">METRIC STATISTICS</div>', unsafe_allow_html=True)
    rows = []
    for m in chosen:
        vals = [max(0, ms.get(m,0)) for ms in scores.values()]
        rows.append({'Metric':m.replace('_',' ').title(),'Mean':np.mean(vals),'Std Dev':np.std(vals),'Min':np.min(vals),'Max':np.max(vals)})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)

# =========================================================
# VIEW HEATMAP — COLOR GRID (all 42 runs at once)
# =========================================================
def view_heatmap(scores:Dict, per_run:Dict, mm:Dict, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    lvl_min, lvl_max = level_range

    # Build run-key list filtered by each entry's level, ordered by (level, sample)
    key_order: dict = {}
    for entries in per_run.values():
        for sid, e in filter_by_level(entries, lvl_min, lvl_max).items():
            key_order.setdefault(sid, (int(e.get('level', 1)), str(e.get('sample', sid))))
    all_sample_ids = sorted(key_order, key=lambda k: key_order[k])

    if not all_sample_ids:
        st.info(f"No samples found for levels {lvl_min}–{lvl_max}.")
        return

    method_names = list(scores.keys())
    # Build metric options from what actually exists in per_run data
    sample_metrics = set()
    for ik in per_run.values():
        for s in list(ik.values())[:1]:
            sample_metrics.update(s.keys())
    metric_opts = sorted(sample_metrics)
    if not metric_opts:
        metric_opts = ['ktc_score']

    # Map sidebar selected_metrics display names → internal keys for default selection
    DISPLAY_TO_INTERNAL = {
        'KTC Score': 'ktc_score',
    }
    sel_metrics_sidebar = st.session_state.get('selected_metrics', [])
    default_hm_metric = 'ktc_score'
    for sm in sel_metrics_sidebar:
        candidate = DISPLAY_TO_INTERNAL.get(sm)
        if candidate and candidate in metric_opts:
            default_hm_metric = candidate
            break
    default_hm_idx = metric_opts.index(default_hm_metric) if default_hm_metric in metric_opts else 0

    mc1, mc2 = st.columns([2, 4])
    with mc1:
        chosen_metric = st.selectbox("Metric:", metric_opts, index=default_hm_idx, key="hm_metric")
    with mc2:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--tx3);padding-top:28px">'
            f'Levels {lvl_min}–{lvl_max} · {len(all_sample_ids)} samples · {len(method_names)} methods. '
            f'Higher = greener (better).</div>',
            unsafe_allow_html=True)

    pc = st.session_state.get('_pcolors', {})
    pb = pc.get('bg', '#f6f8fa')
    pg = pc.get('grid', '#d0d7de')
    pt = pc.get('text', '#848d97')

    z, text, y_labels = [], [], []
    for disp in method_names:
        ik = mm.get(disp)
        row, trow = [], []
        for sid in all_sample_ids:
            val = per_run.get(ik,{}).get(sid,{}).get(chosen_metric, None) if ik else None
            row.append(val if val is not None else float('nan'))
            trow.append(f"{val:.4f}" if val is not None else "—")
        z.append(row)
        text.append(trow)
        y_labels.append(disp)

    colorscale = 'RdYlGn'  # higher = greener; KTC score is higher = better

    # When columns exceed 12 the cell text overlaps — rely on hover tooltips instead
    show_cell_text = len(all_sample_ids) <= 12

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(s) for s in all_sample_ids],
        y=y_labels,
        text=text,
        texttemplate="%{text}" if show_cell_text else "",
        textfont=dict(family="JetBrains Mono,monospace", size=8),
        colorscale=colorscale,
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Sample: %{x}<br>Value: %{text}<extra></extra>",
        colorbar=dict(
            thickness=10, len=0.9,
            tickfont=dict(family="JetBrains Mono,monospace", size=8, color="#848d97"),
            outlinecolor="#d0d7de", outlinewidth=1,
        )
    ))
    fig.update_layout(
        height=max(220, len(method_names)*54 + 80),
        paper_bgcolor=pc.get('paper','rgba(0,0,0,0)'),
        plot_bgcolor=pb,
        font=dict(family="JetBrains Mono,monospace", color=pt, size=9),
        xaxis=dict(side='top', gridcolor=pb, linecolor=pg,
                   tickfont=dict(size=8, color=pt), title='Sample'),
        yaxis=dict(gridcolor=pb, linecolor=pg,
                   tickfont=dict(size=9, color=pt), autorange='reversed'),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick per-method stats row below heatmap
    st.markdown('<div class="slbl">PER-METHOD SUMMARY</div>', unsafe_allow_html=True)
    hm_rows = []
    for disp in method_names:
        ik = mm.get(disp)
        vals = [per_run[ik][s].get(chosen_metric, float('nan'))
                for s in all_sample_ids
                if ik and s in per_run.get(ik,{})]
        vals = [v for v in vals if not (v != v)]  # drop NaN
        if vals:
            hm_rows.append({
                'Method': disp,
                'Mean':   round(float(np.mean(vals)),4),
                'Std':    round(float(np.std(vals)),4),
                'Min':    round(float(np.min(vals)),4),
                'Max':    round(float(np.max(vals)),4),
                'Runs':   len(vals),
            })
    if hm_rows:
        st.dataframe(pd.DataFrame(hm_rows), use_container_width=True, hide_index=True)


def _render_pdf_export(scores:Dict, per_run:Dict, mm:Dict, run_name:str, target=None):
    """Styled PDF report: curated figures, compact tables, max 5 pages."""
    target = target or st
    try:
        import datetime
        import importlib
        importlib.invalidate_caches()
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        max_pages = 5
        display_methods = sorted(scores.items(), key=lambda item: calculate_composite_score(item[1]), reverse=True)[:10]
        run_dir = Path("outputs") / run_name
        asset_dir = Path("outputs") / "report_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)

        def allowed_png(path: Path) -> bool:
            low = str(path).lower()
            return path.suffix.lower() == ".png" and not any(x in low for x in ("failure", "error_overlay", "error_overlays", "overlay"))

        def make_chart_path(name: str) -> Path:
            return asset_dir / f"{run_name}_{name}.png"

        def save_leaderboard_png() -> Path | None:
            if not display_methods:
                return None
            out = make_chart_path("leaderboard")
            try:
                names = [m for m, _ in display_methods[:8]][::-1]
                vals = [calculate_composite_score(scores[m]) for m in names]
                fig, ax = plt.subplots(figsize=(6.6, 3.25), dpi=150)
                ax.barh(names, vals, color=[mcol(i) for i in range(len(names))])
                ax.set_xlim(0, 100); ax.set_title("Leaderboard - Composite Score", fontsize=11, fontweight="bold")
                ax.grid(axis="x", color="#d0d7de", linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(axis="both", labelsize=7)
                for i, v in enumerate(vals): ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=7)
                fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
                return out
            except Exception:
                plt.close("all"); return None

        def save_degradation_png() -> Path | None:
            rows = []
            for method, _ in display_methods:
                ik = mm.get(method)
                for entry in per_run.get(ik, {}).values() if ik else []:
                    rows.append((method, int(entry.get("level", 1)), float(entry.get("ktc_score", 0))))
            if not rows: return None
            out = make_chart_path("degradation")
            try:
                df = pd.DataFrame(rows, columns=["Method", "Level", "KTC"])
                fig, ax = plt.subplots(figsize=(6.6, 3.25), dpi=150)
                for i, method in enumerate(df["Method"].unique()[:6]):
                    s = df[df["Method"] == method].groupby("Level")["KTC"].mean().sort_index()
                    ax.plot(s.index, s.values, marker="o", linewidth=1.5, markersize=3.2, color=mcol(i), label=method.replace("Reconstruction", "Recon"))
                ax.set_title("Degradation - KTC by Difficulty", fontsize=11, fontweight="bold")
                ax.set_xlabel("Level", fontsize=8); ax.set_ylabel("Mean KTC", fontsize=8)
                ax.grid(color="#d0d7de", linewidth=.7); ax.legend(fontsize=5.8, frameon=False, loc="best")
                ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
                return out
            except Exception:
                plt.close("all"); return None

        def save_radar_png() -> Path | None:
            keys = sorted({k for metrics in scores.values() for k in metrics.keys()})
            chosen = [k for k in ("KTC score", "ktc_score") if k in keys] or keys[:4]
            if not chosen: return None
            out = make_chart_path("radar")
            try:
                fig = plt.figure(figsize=(6.2, 3.75), dpi=150); ax = fig.add_subplot(111, polar=True)
                labels = [k.replace("_", " ").title() for k in chosen]
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
                for i, (method, metrics) in enumerate(display_methods[:6]):
                    vals = [max(0.0, min(1.0, float(metrics.get(k, 0)))) for k in chosen]; vals += vals[:1]
                    ax.plot(angles, vals, color=mcol(i), linewidth=1.4, label=method.replace("Reconstruction", "Recon"))
                    ax.fill(angles, vals, color=mcol(i), alpha=.08)
                ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=7); ax.set_ylim(0, 1); ax.set_yticklabels([])
                ax.grid(color="#d0d7de", linewidth=.7); ax.set_title("Radar Performance", fontsize=11, fontweight="bold", pad=10)
                ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.08), fontsize=5.8, frameon=False)
                fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
                return out
            except Exception:
                plt.close("all"); return None

        def save_hull_png() -> Path | None:
            rows = []
            for method, _ in display_methods:
                ik = mm.get(method)
                for entry in per_run.get(ik, {}).values() if ik else []:
                    hull = entry.get("hull", {})
                    center, area = hull.get("hull_resistive_center_error"), hull.get("hull_resistive_area_error")
                    if center is not None or area is not None: rows.append((method, center, area))
            if not rows: return None
            out = make_chart_path("hull")
            try:
                df = pd.DataFrame(rows, columns=["Method", "Center", "Area"])
                fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), dpi=150)
                for ax, title, col, ylabel in [(axes[0], "Center Error", "Center", "px"), (axes[1], "Area Error", "Area", "px^2")]:
                    s = df.dropna(subset=[col]).groupby("Method")[col].mean().sort_values().head(6)
                    ax.bar(s.index, s.values, color=[mcol(i) for i in range(len(s))])
                    ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_ylabel(ylabel, fontsize=7)
                    ax.grid(axis="y", color="#d0d7de", linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
                    ax.tick_params(axis="x", labelrotation=35, labelsize=5.4); ax.tick_params(axis="y", labelsize=6.4)
                fig.suptitle("Hull Analysis - Resistive Region", fontsize=11, fontweight="bold")
                fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
                return out
            except Exception:
                plt.close("all"); return None

        def collect_recon_pngs() -> List[Path]:
            roots = [run_dir / "reconstructions", Path("outputs") / "reconstructions", Path("outputs") / "figures", Path("outputs") / "images"]
            chosen, seen = [], set()
            for method, _ in display_methods:
                internal = mm.get(method, method)
                patterns = [f"{internal}.png", f"{method}.png", f"{method}_level*_sample*.png", f"{internal}_level*_sample*.png"]
                for root in roots:
                    if not root.exists(): continue
                    matches = []
                    for pattern in patterns: matches.extend(root.rglob(pattern))
                    matches = [p for p in sorted(matches) if allowed_png(p) and p.resolve() not in seen]
                    if matches:
                        chosen.append(matches[0]); seen.add(matches[0].resolve()); break
                if len(chosen) >= 4: return chosen
            return chosen

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.2*cm, rightMargin=1.2*cm, topMargin=1.1*cm, bottomMargin=1.1*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("r_title", parent=styles["Heading1"], fontSize=15, leading=18, spaceAfter=3, textColor=rl_colors.HexColor("#1f2328"))
        h2_style = ParagraphStyle("r_h2", parent=styles["Heading2"], fontSize=9.4, leading=11, spaceBefore=2, spaceAfter=4, textColor=rl_colors.HexColor("#57606a"), fontName="Courier-Bold")
        body_style = ParagraphStyle("r_body", parent=styles["Normal"], fontSize=6.2, leading=7.6, textColor=rl_colors.HexColor("#1f2328"), fontName="Courier")
        label_style = ParagraphStyle("r_label", parent=styles["Normal"], fontSize=6.3, leading=7.5, textColor=rl_colors.HexColor("#848d97"), fontName="Courier")
        caption_style = ParagraphStyle("r_caption", parent=styles["Normal"], fontSize=6.2, leading=7.5, textColor=rl_colors.HexColor("#57606a"), fontName="Courier")

        def para(v): return Paragraph(str(v), body_style)
        def fit(path, max_w, max_h):
            img = RLImage(str(path)); scale = min(max_w/img.imageWidth, max_h/img.imageHeight, 1)
            img.drawWidth, img.drawHeight = img.imageWidth*scale, img.imageHeight*scale; return img
        def table_style(tbl, fs=5.8):
            tbl.setStyle(TableStyle([
                ("FONTNAME",(0,0),(-1,0),"Courier-Bold"),("FONTNAME",(0,1),(-1,-1),"Courier"),("FONTSIZE",(0,0),(-1,-1),fs),
                ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#f6f8fa")),("TEXTCOLOR",(0,0),(-1,0),rl_colors.HexColor("#57606a")),
                ("GRID",(0,0),(-1,-1),.25,rl_colors.HexColor("#d0d7de")),("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white,rl_colors.HexColor("#fbfbfb")]),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2.3),("BOTTOMPADDING",(0,0),(-1,-1),2.3),
            ])); return tbl
        def frame(title, path, w=8.25*cm, h=4.85*cm):
            if not path or not path.exists():
                return Table([[Paragraph(f"{title}<br/><font color='#848d97'>Not available</font>", caption_style)]], colWidths=[w])
            t = Table([[Paragraph(title, caption_style)], [fit(path, w-.45*cm, h-.65*cm)]], colWidths=[w])
            t.setStyle(TableStyle([("BOX",(0,0),(-1,-1),.35,rl_colors.HexColor("#d0d7de")),("LINEBELOW",(0,0),(-1,0),.25,rl_colors.HexColor("#d0d7de")),("ALIGN",(0,1),(-1,1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
            return t

        class FivePageCanvas(canvas.Canvas):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs); self._states = []
            def showPage(self):
                if self._pageNumber <= max_pages: self._states.append(dict(self.__dict__))
                self._startPage()
            def save(self):
                total = min(len(self._states), max_pages)
                for state in self._states[:max_pages]:
                    self.__dict__.update(state); self.setFont("Courier", 7); self.setFillColor(rl_colors.HexColor("#848d97"))
                    self.drawRightString(A4[0]-1.2*cm, .68*cm, f"Page {self._pageNumber} of {total}")
                    canvas.Canvas.showPage(self)
                canvas.Canvas.save(self)

        story = [
            Paragraph("EIT Reconstruction Dashboard - PDF Report", title_style),
            Paragraph(f"Run: {run_name} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Max 5 pages", label_style),
            Paragraph("Curated report: leaderboard, hull analysis, reconstructed images, radar, degradation, compact metric tables. Failure figures are excluded.", label_style),
            HRFlowable(width="100%", thickness=.6, color=rl_colors.HexColor("#d0d7de")), Spacer(1, .14*cm),
        ]
        story.append(Paragraph("01 LEADERBOARD SUMMARY", h2_style))
        lb_rows = [[para("Method"), para("Composite"), para("Grade"), para("KTC")]]
        for method, metrics in display_methods:
            comp = calculate_composite_score(metrics)
            lb_rows.append([para(method), f"{comp:.2f}", letter_grade(comp), f"{metrics.get('KTC score', metrics.get('ktc_score', 0)):.4f}"])
        story.append(table_style(Table(lb_rows, repeatRows=1, colWidths=[7.7*cm,2.5*cm,1.8*cm,2.8*cm]), 6.0))
        story.append(Spacer(1, .12*cm))

        story.append(Paragraph("02 VISUAL SUMMARY", h2_style))
        chart_grid = Table([
            [frame("Leaderboard Figure", save_leaderboard_png()), frame("Radar Chart", save_radar_png())],
            [frame("Degradation Chart", save_degradation_png()), frame("Hull Analysis Figures", save_hull_png())],
        ], colWidths=[8.35*cm, 8.35*cm])
        chart_grid.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story.append(chart_grid)

        recon = collect_recon_pngs()
        if recon:
            story.append(Paragraph("03 RECONSTRUCTED IMAGES", h2_style))
            rows = []
            for i in range(0, len(recon), 2):
                cells = [frame(p.stem.replace("_", " ").title(), p, 8.15*cm, 4.25*cm) for p in recon[i:i+2]]
                while len(cells) < 2: cells.append("")
                rows.append(cells)
            story.append(Table(rows, colWidths=[8.35*cm, 8.35*cm]))

        stat_rows = [[para("Method"),para("Runs"),para("Mean KTC"),para("Std"),para("Min"),para("Max")]]
        fail_rows = [[para("Method"),para("Worst KTC"),para("Mean KTC"),para("Failures"),para("Worst Samples")]]
        sample_rows = [[para("Method"),para("Sample"),para("Level"),para("KTC")]]
        for method, _ in display_methods:
            ik = mm.get(method); samples = per_run.get(ik, {}) if ik else {}
            vals = [float(v.get("ktc_score", 0)) for v in samples.values()]
            if not vals: continue
            stat_rows.append([para(method), str(len(vals)), f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}", f"{np.min(vals):.4f}", f"{np.max(vals):.4f}"])
            worst = sorted(samples.items(), key=lambda item: item[1].get("ktc_score", 0))[:3]
            fail_rows.append([para(method), f"{min(vals):.4f}", f"{np.mean(vals):.4f}", str(sum(1 for v in vals if v < .3)), para(", ".join(str(s) for s,_ in worst))])
            for sid, metrics in worst[:2]:
                sample_rows.append([para(method), str(sid), str(metrics.get("level", "")), f"{metrics.get('ktc_score', 0):.4f}"])
        story.append(Paragraph("04 COMPACT STATISTICS", h2_style))
        story.append(Paragraph("Degradation statistics", caption_style))
        story.append(table_style(Table(stat_rows, repeatRows=1, colWidths=[6.2*cm,1.4*cm,2.2*cm,1.8*cm,1.8*cm,1.8*cm]), 5.7))
        story.append(Paragraph("Failure analysis - one table, no failure figures", caption_style))
        story.append(table_style(Table(fail_rows, repeatRows=1, colWidths=[5.2*cm,1.9*cm,1.9*cm,1.6*cm,5.4*cm]), 5.6))
        story.append(Paragraph("Sample metrics snapshot", caption_style))
        story.append(table_style(Table(sample_rows[:17], repeatRows=1, colWidths=[6.6*cm,3.7*cm,1.8*cm,2.2*cm]), 5.6))

        doc.build(story, canvasmaker=FivePageCanvas)
        buf.seek(0)
        target.download_button("Download PDF Report", data=buf.getvalue(), file_name=f"eit_report_{run_name}.pdf", mime="application/pdf", key="pdf_download_btn", use_container_width=True)
    except ImportError as e:
        target.error(f"ReportLab import failed in `{sys.executable}`: {e}")
    except Exception as e:
        target.error(f"PDF generation error: {e}")


# =========================================================
# MAIN
# =========================================================
# =========================================================
# VIEW 7 — HULL ANALYSIS
# =========================================================
def view_hull_analysis(scores: Dict, per_run: Dict, mm: Dict, level_range: tuple = (1, 7)):
    """Convex-hull geometric error analysis — pred vs GT hulls."""

    pc = st.session_state.get('_pcolors', {})
    sel_methods = list(scores.keys())
    lvl_min, lvl_max = level_range

    # ── Collect hull data across all methods ──────────────────
    hull_rows = []
    for method in sel_methods:
        pr_key = mm.get(method, method)
        entries = per_run.get(pr_key, {})
        for key, entry in entries.items():
            lv = int(entry.get("level", 1))
            if lv < lvl_min or lv > lvl_max:
                continue
            hull = entry.get("hull", {})
            if not hull:
                continue
            hull_rows.append({
                "Method": method,
                "Level": lv,
                "Sample": entry.get("sample", "?"),
                "KTC": entry.get("ktc_score", 0.0),
                "Res Center Err": hull.get("hull_resistive_center_error"),
                "Res Area Err": hull.get("hull_resistive_area_error"),
                "Res Perim Err": hull.get("hull_resistive_perimeter_error"),
                "Con Center Err": hull.get("hull_conductive_center_error"),
                "Con Area Err": hull.get("hull_conductive_area_error"),
                "Con Perim Err": hull.get("hull_conductive_perimeter_error"),
                "Res Area": hull.get("hull_res_area"),
                "Con Area": hull.get("hull_con_area"),
                "Res Pixels": hull.get("hull_res_pixels", 0),
                "Con Pixels": hull.get("hull_con_pixels", 0),
            })

    if not hull_rows:
        st.info("No hull data available. Run the benchmark or `python compute_hull_data.py` to generate hull analysis.")
        return

    import pandas as _pd
    df = _pd.DataFrame(hull_rows)

    # ── KPI cards — average geometric errors per method ───────
    st.markdown("### Method Comparison — Average Geometric Errors")
    err_cols = ["Res Center Err", "Res Area Err", "Res Perim Err"]
    summary = df.groupby("Method")[err_cols].mean().round(1)

    kpi_html = '<div class="kpi-row">'
    for i, method in enumerate(summary.index):
        row = summary.loc[method]
        c = PALETTE[i % len(PALETTE)]
        center_e = row["Res Center Err"]
        area_e = row["Res Area Err"]
        center_str = f"{center_e:.1f}px" if pd.notna(center_e) else "—"
        area_str = f"{area_e:.0f}px²" if pd.notna(area_e) else "—"
        short = method.replace("Reconstruction", "Recon").replace("Difference", "Diff").replace("Projection", "Proj")
        kpi_html += (
            f'<div class="kpi" style="--kc:{c}">'
            f'<div class="kpi-n">{center_str}</div>'
            f'<div class="kpi-l">Center Error</div>'
            f'<div class="kpi-s">{short} · Area Err: {area_str}</div></div>'
        )
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Bar chart: Center error by method ─────────────────────
    st.markdown("### Resistive Region — Center Error by Method")
    avg_center = df.dropna(subset=["Res Center Err"]).groupby("Method")["Res Center Err"].mean().sort_values()
    if not avg_center.empty:
        fig_ce = go.Figure()
        colors = [PALETTE[list(scores.keys()).index(m) % len(PALETTE)] if m in scores else "#848d97"
                  for m in avg_center.index]
        fig_ce.add_trace(go.Bar(
            x=avg_center.index, y=avg_center.values,
            marker_color=colors,
            text=[f"{v:.1f}px" for v in avg_center.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_ce.update_layout(
            yaxis_title="Center Error (px)",
            height=340,
            margin=dict(l=50, r=20, t=30, b=40),
            plot_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            paper_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            font=dict(family="JetBrains Mono", size=10, color=pc.get('text', '#848d97')),
            yaxis=dict(gridcolor=pc.get('grid', '#d0d7de'), zeroline=False),
        )
        st.plotly_chart(fig_ce, use_container_width=True)

    # ── Bar chart: Area error by method ───────────────────────
    st.markdown("### Resistive Region — Hull Area Error by Method")
    avg_area = df.dropna(subset=["Res Area Err"]).groupby("Method")["Res Area Err"].mean().sort_values()
    if not avg_area.empty:
        fig_ae = go.Figure()
        colors_a = [PALETTE[list(scores.keys()).index(m) % len(PALETTE)] if m in scores else "#848d97"
                    for m in avg_area.index]
        fig_ae.add_trace(go.Bar(
            x=avg_area.index, y=avg_area.values,
            marker_color=colors_a,
            text=[f"{v:.0f}px²" for v in avg_area.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_ae.update_layout(
            yaxis_title="Area Error (px²)",
            height=340,
            margin=dict(l=50, r=20, t=30, b=40),
            plot_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            paper_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            font=dict(family="JetBrains Mono", size=10, color=pc.get('text', '#848d97')),
            yaxis=dict(gridcolor=pc.get('grid', '#d0d7de'), zeroline=False),
        )
        st.plotly_chart(fig_ae, use_container_width=True)

    # ── Scatter: KTC score vs Center error ────────────────────
    st.markdown("### KTC Score vs Center Error — Correlation")
    scatter_df = df.dropna(subset=["Res Center Err", "KTC"])
    if not scatter_df.empty:
        fig_sc = go.Figure()
        for i, method in enumerate(sel_methods):
            mdf = scatter_df[scatter_df["Method"] == method]
            if mdf.empty:
                continue
            fig_sc.add_trace(go.Scatter(
                x=mdf["KTC"], y=mdf["Res Center Err"],
                mode="markers",
                name=method.replace("Reconstruction", "Recon"),
                marker=dict(size=7, color=PALETTE[i % len(PALETTE)]),
            ))
        fig_sc.update_layout(
            xaxis_title="KTC Score",
            yaxis_title="Center Error (px)",
            height=380,
            margin=dict(l=50, r=20, t=30, b=40),
            plot_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            paper_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            font=dict(family="JetBrains Mono", size=10, color=pc.get('text', '#848d97')),
            xaxis=dict(gridcolor=pc.get('grid', '#d0d7de')),
            yaxis=dict(gridcolor=pc.get('grid', '#d0d7de')),
            legend=dict(bgcolor=pc.get('legend', 'rgba(255,255,255,.9)'),
                        bordercolor=pc.get('grid', '#d0d7de'), borderwidth=1,
                        font=dict(size=10)),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Degradation: Center error across difficulty levels ────
    st.markdown("### Hull Error Degradation by Level")
    deg_df = df.dropna(subset=["Res Center Err"])
    if not deg_df.empty:
        fig_deg = go.Figure()
        for i, method in enumerate(sel_methods):
            mdf = deg_df[deg_df["Method"] == method]
            if mdf.empty:
                continue
            by_level = mdf.groupby("Level")["Res Center Err"].mean().sort_index()
            fig_deg.add_trace(go.Scatter(
                x=by_level.index, y=by_level.values,
                mode="lines+markers",
                name=method.replace("Reconstruction", "Recon"),
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                marker=dict(size=6),
            ))
        fig_deg.update_layout(
            xaxis_title="Difficulty Level",
            yaxis_title="Avg Center Error (px)",
            height=360,
            margin=dict(l=50, r=20, t=30, b=40),
            plot_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            paper_bgcolor=pc.get('paper', 'rgba(0,0,0,0)'),
            font=dict(family="JetBrains Mono", size=10, color=pc.get('text', '#848d97')),
            xaxis=dict(gridcolor=pc.get('grid', '#d0d7de'), dtick=1),
            yaxis=dict(gridcolor=pc.get('grid', '#d0d7de')),
            legend=dict(bgcolor=pc.get('legend', 'rgba(255,255,255,.9)'),
                        bordercolor=pc.get('grid', '#d0d7de'), borderwidth=1,
                        font=dict(size=10)),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    # ── Detailed table ────────────────────────────────────────
    st.markdown("### Per-Run Hull Metrics")
    display_cols = ["Method", "Level", "Sample", "KTC",
                    "Res Center Err", "Res Area Err", "Res Perim Err",
                    "Res Pixels", "Con Pixels"]
    tbl = df[display_cols].copy()
    for c in ["KTC", "Res Center Err", "Res Area Err", "Res Perim Err"]:
        tbl[c] = tbl[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    st.dataframe(tbl, use_container_width=True, hide_index=True)


def main():
    pdf_export_slot = render_sidebar()
    dark = st.session_state.get('dark_mode', False)
    inject_theme(dark)

    # Plot color helpers that respect dark mode
    plot_bg     = '#161b22' if dark else '#f6f8fa'
    plot_paper  = 'rgba(22,27,34,0)' if dark else 'rgba(0,0,0,0)'
    plot_grid   = '#30363d' if dark else '#d0d7de'
    plot_text   = '#8b949e' if dark else '#848d97'
    plot_legend = 'rgba(22,27,34,.9)' if dark else 'rgba(255,255,255,.9)'
    st.session_state['_pcolors'] = dict(bg=plot_bg, paper=plot_paper,
                                        grid=plot_grid, text=plot_text, legend=plot_legend)

    header_bg = 'var(--sur)'  # CSS vars handle dark mode automatically now

    # Header
    st.markdown("""
    <div class="dash-header">
      <div class="dash-title">EIT Reconstruction Dashboard</div>
      <div class="dash-sub">Electrical Impedance Tomography &mdash; Benchmarking &amp; Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress banner — visible whenever a benchmark subprocess is running
    render_bench_progress()

    try:
        latest_run = find_latest_run()
        cache_key = latest_run.name
        scores, per_run, mm = load_data(cache_key)

        # Active run label (+ red badge when runs were scored against a
        # missing ground truth — their 0.0 scores are meaningless)
        n_gt_missing = count_gt_missing(per_run)
        gt_badge = (
            f' &nbsp;·&nbsp; <span style="background:#ffebe9;border:1px solid #cf222e;'
            f'color:#cf222e;border-radius:5px;padding:1px 6px;font-weight:600">'
            f'{n_gt_missing} runs scored without ground truth</span>'
        ) if n_gt_missing else ''
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
            f'color:var(--tx3);margin:-2px 0 10px;padding:0 2px;line-height:1.35">'
            f'Run: <span style="color:var(--grn)">{latest_run.name}</span> &nbsp;·&nbsp; '
            f'{len(scores)} method(s) &nbsp;·&nbsp; '
            f'{sum(len(v) for v in per_run.values()) if per_run else 0} total reconstructions'
            f'{gt_badge}</div>',
            unsafe_allow_html=True)

        if not scores and not per_run:
            st.error("No data found. Run `python example_usage.py` first.")
            return

        # Store available methods so sidebar checkboxes can render them
        st.session_state['_available_methods'] = list(scores.keys())

        # Selected metrics + methods + level range from sidebar
        sel_metrics = st.session_state.get('selected_metrics', ['KTC Score'])
        sel_methods = st.session_state.get('selected_methods', list(scores.keys()))
        if not sel_methods:
            sel_methods = list(scores.keys())  # fallback: show all if none checked
        level_range = st.session_state.get('level_range', (1, 7))
        sel_samples = st.session_state.get('selected_samples', ['A', 'B', 'C'])

        # Apply method, level, and sample filters once so every tab/report agrees.
        scores_f, per_run_f, mm_f = apply_dashboard_filters(
            scores, per_run, mm, sel_methods, level_range, sel_samples
        )
        if not scores_f:
            st.info("No dashboard data matches the selected sidebar filters.")
            return

        # Dataset info — inline KPI row (no expander = no icon issue)
        n_s = len(per_run_f.get(list(per_run_f.keys())[0], {})) if per_run_f else 0
        n_t = sum(len(v) for v in per_run_f.values()) if per_run_f else 0
        method_list = " &nbsp;·&nbsp; ".join(
            f'<span style="color:var(--tx)">{m}</span>' for m in scores_f.keys())
        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:12px;align-items:stretch">'
            f'<div style="flex:1;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 12px;min-height:72px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:19px;font-weight:600;color:var(--tx);line-height:1">{len(scores_f)}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-top:6px">Methods</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:var(--tx2);margin-top:4px;line-height:1.35">{method_list}</div>'
            f'</div>'
            f'<div style="flex:0 0 112px;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 12px;min-height:72px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:19px;font-weight:600;color:var(--tx);line-height:1">{n_s}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-top:6px">Runs / method</div>'
            f'</div>'
            f'<div style="flex:0 0 112px;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 12px;min-height:72px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:19px;font-weight:600;color:var(--tx);line-height:1">{n_t}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-top:6px">Total recons</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True)

        # PDF export — triggered from sidebar button
        if st.session_state.get('_trigger_pdf'):
            st.session_state['_trigger_pdf'] = False
            _render_pdf_export(scores_f, per_run_f, mm_f, latest_run.name, target=pdf_export_slot)

        # Tabs
        t1,t2,t3,t4,t5,t6,t7 = st.tabs([
            "LEADERBOARD", "HEATMAP",
            "DEGRADATION", "RADAR",
            "FAILURES", "RECONSTRUCTION",
            "HULL ANALYSIS"])

        with t1: view_leaderboard(scores_f, per_run_f, sel_metrics, mm_f, level_range)
        with t2: view_heatmap(scores_f, per_run_f, mm_f, level_range)
        with t3: view_degradation_curve(scores_f, per_run_f, mm_f, level_range)
        with t4: view_radar_chart(scores_f, per_run_f, sel_metrics)
        with t5: view_failure_gallery(scores_f, per_run_f, mm_f, level_range)
        with t6: view_comparison(scores_f, per_run_f, mm_f, sel_metrics, level_range)
        with t7: view_hull_analysis(scores_f, per_run_f, mm_f, level_range)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# Run the app
main()
