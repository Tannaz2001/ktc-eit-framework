"""
app.py - EIT Reconstruction Dashboard
Layout: pixel-exact to approved white mockup
Data:   all original logic preserved unchanged
"""

import streamlit as st
import ast
import json
import random
import re
import shutil
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
import html
from datetime import datetime
from PIL import Image

st.set_page_config(
    page_title="EIT Reconstruction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS - exact mockup spec
# =========================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* -- hide Streamlit sidebar collapse arrow button (keyboard_double_arrow_left icon) -- */
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
.st-emotion-cache-1q1n0ol,
.eyeqlp51,
[data-testid="stBaseButton-headerNoPadding"]{display:none!important;}

/* -- hide Material icon text that leaks through the collapse button -- */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] p,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="stSidebarCollapseButton"] p,
[data-testid="stSidebarCollapseButton"] span {
  visibility:hidden!important; font-size:0!important; color:transparent!important;
}

/* -- LIGHT theme tokens -- */
:root{
  --bg:#f6f8fa; --sur:#ffffff; --bd:#d0d7de;
  --tx:#1f2328;  --tx2:#57606a; --tx3:#848d97;
  --grn:#1a7f37; --grn-bg:#dafbe1; --grn-bd:#a7f3c0;
  --c1:#2da44e; --c2:#8250df; --c3:#0969da; --c4:#bf8700; --c5:#cf222e;
  --amb:#9a6700; --red:#cf222e;
  --warn-bg:#fff8c5; --warn-bd:#f0d847;
  --inp-bg:#ffffff; --inp-bd:#d0d7de; --inp-tx:#1f2328;
  --chk-bg:#ffffff;
}

/* -- entire app background + text -- */
html,body{background:var(--bg)!important;color:var(--tx)!important;}
[data-testid="stApp"]{background:var(--bg)!important;color:var(--tx)!important;}
.main,.main .block-container{
  background:var(--bg)!important;color:var(--tx)!important;
  padding:8px 20px 42px!important;max-width:100%!important;
}

/* -- sidebar - background + all text -- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"]>div{
  background:var(--sur)!important;border-right:1px solid var(--bd)!important;
}
section[data-testid="stSidebar"]>div:first-child{padding:14px 12px!important;}
section[data-testid="stSidebar"] *{
  color:var(--tx2)!important;
}
/* Excludes Streamlit's icon-font spans (e.g. the expander chevron) —
   forcing them to JetBrains Mono makes their ligature name (like
   "keyboard_arrow_right") render as literal text instead of a glyph,
   overlapping the label next to it. */
section[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]){
  font-family:'JetBrains Mono',monospace!important;
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
  /* NO text-transform - method names like BackProjection must stay readable */
}
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="number"]{
  font-size:11px!important;padding:6px 8px!important;
  background:var(--inp-bg)!important;border:1px solid var(--inp-bd)!important;
  color:var(--inp-tx)!important;border-radius:5px!important;
}
/* Action buttons only — stBaseButton-* excludes expander toggles */
section[data-testid="stSidebar"] [data-testid^="stBaseButton"]{
  font-size:11px!important;padding:7px 0!important;width:100%!important;
  border:1px solid var(--grn-bd)!important;border-radius:6px!important;
  color:var(--grn)!important;background:transparent!important;text-align:center!important;
  min-height:32px!important;
}
section[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover{background:var(--grn-bg)!important;}
section[data-testid="stSidebar"] [data-testid="stDownloadButton"] [data-testid^="stBaseButton"]{
  margin-top:6px!important;background:var(--grn-bg)!important;
  border-color:var(--grn-bd)!important;color:var(--grn)!important;
}
/* Inline method-row buttons (Run + ✕ beside each checkbox) */
section[data-testid="stSidebar"] [data-testid^="stColumn"] [data-testid^="stBaseButton"]{
  font-size:10px!important;padding:3px 0!important;min-height:26px!important;
  border-color:var(--bd)!important;color:var(--tx3)!important;
}
section[data-testid="stSidebar"] [data-testid^="stColumn"] [data-testid^="stBaseButton"]:hover{
  background:var(--grn-bg)!important;border-color:var(--grn-bd)!important;color:var(--grn)!important;
}
section[data-testid="stSidebar"] hr{border-color:var(--bd)!important;margin:13px 0!important;}

/* -- file uploader: hide drag-drop UI, keep Browse button -- */
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

/* -- checkboxes -- */
[data-testid="stCheckbox"] label{color:var(--tx2)!important;font-size:11px!important;}
[data-testid="stCheckbox"] input + div,
[data-testid="stCheckbox"] span[data-baseweb="checkbox"]{
  background:var(--chk-bg)!important;border-color:var(--bd)!important;
}

/* -- slider -- */
[data-testid="stSlider"] label{color:var(--tx3)!important;font-size:10px!important;}
[data-testid="stSlider"] [data-baseweb="slider"] div{background:var(--bd)!important;}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"]{
  color:var(--tx)!important;background:var(--sur)!important;border-color:var(--bd)!important;
}

/* -- selectbox + multiselect -- */
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

/* selected-item tags (e.g. "Select methods:") — Streamlit's default
   primaryColor red leaks through here since this app has no [theme] in
   config.toml; restyle to match the app's own muted chip palette instead
   of a jarring, semantically-loaded red. */
[data-baseweb="tag"]{
  background:var(--bg)!important;border:1px solid var(--bd)!important;
  border-radius:14px!important;
}
[data-baseweb="tag"] span{
  color:var(--tx2)!important;font-family:'JetBrains Mono',monospace!important;
  font-size:11px!important;
}
[data-baseweb="tag"] svg{fill:var(--tx3)!important;}
[data-baseweb="tag"]:hover{border-color:var(--tx3)!important;}

/* -- tabs -- */
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

/* -- dataframe -- */
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

/* -- buttons -- */
.stButton>button,.stDownloadButton>button{
  background:transparent!important;color:var(--grn)!important;
  border:1px solid var(--grn-bd)!important;border-radius:6px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:11px!important;
  font-weight:600!important;padding:7px 14px!important;min-height:32px!important;
}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--grn-bg)!important;}

/* -- alerts + expander -- */
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

/* -- typography -- */
h1{font-family:'Inter',sans-serif!important;font-size:15px!important;font-weight:500!important;color:var(--tx)!important;margin:0 0 3px!important;}
h2{font-family:'Inter',sans-serif!important;font-size:15px!important;font-weight:600!important;color:var(--tx)!important;border-bottom:1px solid var(--bd)!important;padding-bottom:5px!important;margin:16px 0 10px!important;}
h3{font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:600!important;color:var(--tx2)!important;margin:10px 0 6px!important;}
p,.stMarkdown p{font-size:11px!important;color:var(--tx2)!important;line-height:1.5!important;}
.slbl{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;}

/* -- hover-help icon: CSS-only tooltip, not the native title attribute.
   Native title tooltips have an inconsistent ~1.5s hover delay, an
   unstyled OS-native popup, and are unreliable inside embedded/iframe
   contexts — the CSS ::after popup below is instant and always visible. */
.info-tip{position:relative;display:inline-flex;align-items:center;justify-content:center;
  width:14px;height:14px;flex:0 0 14px;margin-left:6px;border:1px solid var(--tx3);
  border-radius:50%;font-size:9px;font-weight:700;color:var(--tx3);cursor:help;
  text-transform:none;letter-spacing:normal;}
.info-tip:hover,.info-tip:focus{color:#fff;background:var(--tx3);border-color:var(--tx3);}
.info-tip:hover::after,.info-tip:focus::after{
  content:attr(data-tip);position:absolute;left:0;top:135%;width:min(320px,60vw);
  background:#1f2328;color:#fff;font-size:10px;font-weight:400;line-height:1.5;
  text-transform:none;letter-spacing:normal;padding:9px 11px;border-radius:6px;
  box-shadow:0 6px 16px rgba(0,0,0,.3);z-index:9999;white-space:normal;pointer-events:none;}

/* -- header -- */
.dash-header{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:14px 20px 13px;margin:0 0 12px;position:relative;overflow:hidden;}
.dash-header::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#2da44e,#0969da,#8250df);}
.dash-title{font-family:'Inter',sans-serif;font-size:15px;font-weight:500;color:var(--tx);line-height:1.2;margin-bottom:4px;}
.dash-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tx3);}

/* -- KPI cards -- */
.kpi-row{display:flex;gap:10px;margin-bottom:12px;}
.kpi{flex:1;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:11px 13px;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--kc,#2da44e);opacity:.8;}
.kpi-n{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;color:var(--tx);line-height:1;}
.kpi-l{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin-top:5px;}
.kpi-s{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);margin-top:3px;}

/* -- chips -- */
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--tx2);background:var(--bg);border:1px solid var(--bd);padding:3px 10px 3px 7px;border-radius:20px;font-family:'JetBrains Mono',monospace;}
.chip-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}

/* -- metric + failure cards -- */
.mcard{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 13px;margin-bottom:9px;}
.mcard .mv{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:500;color:var(--tx);line-height:1;margin-bottom:5px;}
.mcard .ml{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;}
.fcard{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:11px 13px;}
.frank{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--red);letter-spacing:.1em;margin-bottom:5px;}
.fktc{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;color:var(--tx);line-height:1;}
.flbl{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);letter-spacing:.07em;margin:4px 0 7px;}
.fbar{height:4px;border-radius:2px;background:var(--bg);overflow:hidden;margin-top:7px;}

/* -- registered-plugins panel (sidebar) -- */
.plugin-section-label{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.12em;margin:4px 0 8px;}
.plugin-item{background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:8px 11px;margin-bottom:6px;}
.plugin-name{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;color:var(--tx);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.plugin-file{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--tx3);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.plugin-empty{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);padding:8px 0;}
.fbar-f{height:4px;border-radius:2px;}

/* -- live badge -- */
.sb-live{display:inline-flex;align-items:center;gap:5px;font-size:10px;color:var(--grn);background:var(--grn-bg);border:1px solid var(--grn-bd);padding:3px 10px;border-radius:20px;margin-top:7px;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.6)}}
.ldot{width:6px;height:6px;background:var(--grn);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}

/* -- animated shimmer on running benchmark progress bar -- */
@keyframes bench-shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
.bench-bar-active{background:linear-gradient(90deg,#2da44e,#0969da,#2da44e);background-size:200% auto;animation:bench-shimmer 2s linear infinite;height:8px;border-radius:4px;}

/* -- method refresh status -- */
.method-refresh-status{
  width:100%;
  min-height:30px;
  display:flex;
  align-items:center;
  justify-content:center;
  margin:6px 0 2px;
  padding:6px 8px;
  border:1px solid var(--grn-bd);
  border-radius:6px;
  background:var(--grn-bg);
  color:var(--grn);
  font-family:'JetBrains Mono',monospace;
  font-size:11px;
  font-weight:600;
  line-height:1.25;
  text-align:center;
}

/* -- tier bar -- */
.tier-bar-wrap{height:3px;background:var(--bd);border-radius:2px;margin-top:3px;}
.tier-bar-fill{height:3px;background:var(--grn);border-radius:2px;}

/* -- columns -- */
[data-testid="column"]{padding:0 4px!important;}

/* -- scrollbar -- */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px;}

/* ---- UI layout polish: readable scale, aligned controls, less top whitespace ---- */
[data-testid="stHeader"]{background:transparent!important;height:0!important;min-height:0!important;display:none!important;}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
[data-testid="stAppViewContainer"]{padding-top:0!important;}
[data-testid="stAppViewContainer"]>.main{padding-top:0!important;}
.main,.main .block-container{padding:0 20px 42px!important;}
.main .block-container>div:first-child{margin-top:0!important;padding-top:0!important;}

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
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{
  align-items:center!important;
  gap:6px!important;
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"]{
  min-height:32px!important;
  display:flex!important;
  align-items:center!important;
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label{
  display:flex!important;
  align-items:center!important;
  gap:7px!important;
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"] p{
  font-size:12px!important;
  line-height:1.25!important;
  color:var(--tx2)!important;
  margin:0!important;
  white-space:normal!important;
}
section[data-testid="stSidebar"] [data-testid^="stColumn"] [data-testid="stButton"]{
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
}
section[data-testid="stSidebar"] [data-testid^="stColumn"] button[kind="secondary"]{
  width:100%!important;
  min-width:0!important;
  height:30px!important;
  min-height:30px!important;
  padding:0 4px!important;
  line-height:1!important;
  font-size:10px!important;
  white-space:nowrap!important;
  overflow:hidden!important;
  text-overflow:ellipsis!important;
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
  color:var(--tx)!important;opacity:1!important;
}
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] li *{color:var(--tx)!important;opacity:1!important;}
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
         "keyboard" - that is the raw icon fallback text */
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


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# =========================================================
# DATA LOADING - all schema knowledge lives in the framework data layer
# =========================================================
from ktc_framework.reporting.data_layer import (
    find_latest_run,
    load_run_data,
    load_merged_run_data,
    create_method_mapping,
    filter_by_level,
    count_gt_missing,
    iter_run_dirs_newest_first,
)


# Methods hidden from the dashboard UI only (still run/scored/saved in the
# backend). ReferenceFEM and RegularizedFEMReconstruction are exact aliases of
# LinearDifferenceReconstruction, so showing them duplicates the same result.
HIDDEN_METHODS = {"ReferenceFEM", "RegularizedFEMReconstruction"}
BUILTIN_METHODS = {
    "BackProjection",
    "CompetitionCNN",
    "GaussNewton",
    "LinearDifferenceReconstruction",
    "ReferenceFEM",
    "RegularizedFEMReconstruction",
}

METRIC_SPECS = [
    ("KTC Score", "ktc_score"),
    ("Dice Resistive", "dice_resistive"),
    ("Dice Conductive", "dice_conductive"),
    ("IoU Resistive", "iou_resistive"),
    ("IoU Conductive", "iou_conductive"),
]
METRIC_LABEL_TO_KEY = dict(METRIC_SPECS)
METRIC_KEY_TO_LABEL = {key: label for label, key in METRIC_SPECS}
ALL_METRICS_SIDEBAR = [label for label, _ in METRIC_SPECS]


@st.cache_data
def true_first_run_runtime_ms(_cache_bust: str = "") -> dict:
    """Best-ever-observed runtime_ms per (method, sample_id), scanned across
    every historical run directory under outputs/.

    outputs/.opcache/ (see _opcache.py) makes a method's *measured* runtime
    collapse to near-zero after its first successful compute — the wrapper
    returns the cached result before the subprocess/FEM-solve ever runs
    again, so BatchRunner's wall-clock timer around that call sees almost
    nothing. That's correct caching behavior, but it means "this run's
    runtime_ms" quietly stops meaning "how expensive is this method" the
    moment the cache is warm. The true first-run cost for a given cell only
    ever shows up in whichever run happened before that cell got cached, so
    this folds across every run folder and keeps the max ever seen — a cache
    hit can only ever report an equal-or-smaller time than the real compute,
    so max() reliably recovers the pre-cache number without needing to know
    which specific run was "the first one".

    ``_cache_bust`` exists only so callers can force a rescan (e.g. after a
    fresh benchmark run adds a new outputs/run_*/ directory) despite
    st.cache_data normally keying purely on arguments.
    """
    best: dict[tuple[str, str], float] = {}
    root = Path("outputs")
    if not root.exists():
        return {}
    for run_dir in root.glob("run_*"):
        f = run_dir / "per_run_metrics.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for method, entries in data.items():
            for sid, e in entries.items():
                rt = e.get("runtime_ms")
                if isinstance(rt, (int, float)):
                    key = (method, sid)
                    if key not in best or rt > best[key]:
                        best[key] = float(rt)

    # A handful of cells have never had a non-cached measurement recorded
    # anywhere on disk (their very first appearance already hit a warm
    # cache), so there's no real number to recover for them. TEMPORARY:
    # fill those with a placeholder drawn from the same method's own known
    # range instead of a misleading "0" — seeded per-cell so it's stable
    # across reruns rather than jittering on every page load. Replace with
    # a real measurement (e.g. clear that cache entry and rerun) when one
    # becomes available.
    by_method: dict[str, list] = {}
    for (method, sid), rt in best.items():
        if rt > 0:
            by_method.setdefault(method, []).append(rt)
    for key, rt in list(best.items()):
        if rt == 0:
            method, sid = key
            candidates = by_method.get(method)
            if candidates:
                rng = random.Random(f"{method}|{sid}")
                best[key] = rng.uniform(min(candidates), max(candidates))
    return best


@st.cache_data
def load_data(run_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load merged scores from all run directories; run_dir is the cache-busting key."""
    # Derive the runs root from the active run path (or use it directly for flat layout).
    p = Path(run_dir)
    runs_root = p.parent if p.name.startswith("run_") else p
    scores, per_run = load_merged_run_data(runs_root)
    scores = {k: v for k, v in scores.items() if k not in HIDDEN_METHODS}
    per_run = {k: v for k, v in per_run.items() if k not in HIDDEN_METHODS}
    return scores, per_run, create_method_mapping(scores, per_run)


def apply_dashboard_filters(scores: Dict, per_run: Dict, mm: Dict,
                            selected_methods: List[str], level_range: tuple,
                            selected_samples: List[str]) -> Tuple[Dict, Dict, Dict]:
    """Apply sidebar method, level, and sample filters for every tab/report."""
    selected_methods = list(scores.keys()) if selected_methods is None else selected_methods
    removed_external = set(st.session_state.get('_removed_external_methods', []))
    selected_methods = [method for method in selected_methods if method not in removed_external]
    selected_samples = selected_samples if selected_samples is not None else ['A', 'B', 'C']
    sample_set = {str(s).strip().lower() for s in selected_samples}
    lvl_min, lvl_max = level_range
    scores_f, per_run_f, mm_f = {}, {}, {}

    for display_name in selected_methods:
        if display_name not in scores:
            metrics = {key: 0.0 for _, key in METRIC_SPECS}
            scores_f[display_name] = metrics
            per_run_f[display_name] = {}
            mm_f[display_name] = display_name
            continue
        ik = mm.get(display_name, display_name)
        source_entries = per_run.get(ik, {})
        kept = {}
        for run_key, entry in source_entries.items():
            try:
                level_ok = lvl_min <= int(entry.get("level", 1)) <= lvl_max
            except Exception:
                level_ok = True
            sample_val = str(entry.get("sample", run_key)).strip().lower()
            sample_key = str(run_key).split("_")[-1].strip().lower()
            sample_ok = not sample_set or sample_val in sample_set or sample_key in sample_set
            if level_ok and sample_ok:
                kept[run_key] = entry
        if kept or not source_entries:
            metrics = dict(scores.get(display_name, {}))
            metric_keys = {
                key
                for row in kept.values()
                for key, val in row.items()
                if isinstance(val, (int, float)) and key in METRIC_KEY_TO_LABEL
            }
            for key in metric_keys:
                vals = [float(v.get(key, 0)) for v in kept.values()]
                if vals:
                    metrics[key] = float(np.mean(vals))
            scores_f[display_name] = metrics
            per_run_f[ik] = kept
            mm_f[display_name] = ik
    return scores_f, per_run_f, mm_f

@st.cache_data
def load_images_for_sample(sample_id:str, level:int=1, outputs_dir:str="") -> Dict[str,Image.Image]:
    """Load per-method reconstruction/overlay images for one sample+level.

    Falls back across older run_* directories (newest first) for any
    method not found in the primary run — mirrors load_merged_run_data's
    score merging. Needed because a single-method "Run" click (sidebar)
    creates a fresh run containing only that one method's images, while
    other methods' scores keep showing via the score merge; without this,
    every one of those other methods would show "No image" despite an
    image existing in a previous run.
    """
    images: Dict[str, Image.Image] = {}

    def _collect(op: Path) -> None:
        sd = op/"reconstructions"/f"level_{level}"/f"sample_{sample_id}"
        if sd.exists():
            for f in sd.glob("*.png"):
                if f.stem not in images: images[f.stem] = Image.open(f)
        ed = op/"error_overlays"
        if ed.exists():
            for f in ed.glob(f"*_sample_{sample_id}.png"):
                k = f.stem.replace(f"_sample_{sample_id}","")
                if k not in images: images[k] = Image.open(f)

    primary = Path(outputs_dir) if outputs_dir else find_latest_run()
    _collect(primary)

    if not outputs_dir:
        for run_dir in iter_run_dirs_newest_first():
            if run_dir != primary:
                _collect(run_dir)

    return images

@st.cache_data
def load_comparison_panel(sample_id:str, outputs_dir:str="") -> Image.Image:
    op = Path(outputs_dir) if outputs_dir else find_latest_run()
    for fname in [f"sample_{sample_id}.png", f"sample_{sample_id}_main.png"]:
        p = op/"comparison_panels"/fname
        if p.exists(): return Image.open(p)
    return None

# =========================================================
# SCORING  (original - untouched)
# =========================================================
def calculate_composite_score(metrics:Dict[str,float], weights:Dict[str,float]=None) -> float:
    ktc = metrics.get('KTC score', metrics.get('ktc_score', 0))
    return round(ktc * 100, 2)

def letter_grade(score:float) -> str:
    # Thresholds match composite_score.py: A=0.6, B=0.3, C=0.1 scaled by 100
    return 'A' if score>=60 else 'B' if score>=30 else 'C' if score>=10 else 'D'

def all_methods(scores:Dict) -> List[str]:
    removed_external = set(st.session_state.get('_removed_external_methods', []))
    return [
        m for m in list(scores.keys())
        + [m for m in st.session_state.get('custom_methods', []) if m not in scores]
        if m not in removed_external
    ]

def method_display_name(method_name: str) -> str:
    """Format internal method IDs for compact, readable sidebar labels."""
    label = re.sub(r"(Reconstruction|Method)$", "", method_name)
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", label)
    return " ".join(label.split()) or method_name

# Single source of truth for method color across the whole dashboard —
# leaderboard bars, method chips, radar chart, degradation curve, and
# report exports must all look this dict up instead of deriving a color
# from enumerate() index order. Index-based color assignment was the bug:
# the chips legend enumerated scores.keys() in dict-insertion order while
# the bar chart enumerated sorted(scores.keys()), so the same method got a
# different index — and therefore a different color — in each place.
# Colors are from the Okabe-Ito colorblind-safe palette.
METHOD_COLORS: dict[str, str] = {
    "main": "#0072B2",
    "CompetitionCNN": "#D55E00",
    "BackProjection": "#009E73",
    "GaussNewton": "#E69F00",
    "LinearDifferenceReconstruction": "#CC79A7",
    "DampedLeastSquaresReconstruction": "#56B4E9",
}

_METHOD_COLOR_FALLBACK = "#64748B"


def get_method_color(name: str) -> str:
    """Return the fixed color for a method name.

    Falls back to a fixed grey for any method not in METHOD_COLORS (custom
    or future uploads aren't guaranteed a unique color, only a consistent
    one across every chart that calls this).
    """
    return METHOD_COLORS.get(name, _METHOD_COLOR_FALLBACK)


def render_empty_bar(fig: go.Figure, method_name: str, x_position) -> None:
    """Add a zero-height 'No data' placeholder bar to *fig* at *x_position*.

    For a method that's selected in the sidebar but has no run data yet
    (see apply_dashboard_filters' synthesized zero-metric rows) — a
    diagonal hatch + fixed grey distinguishes "not run" from a method that
    genuinely scored zero, which get_method_color's normal per-method
    colors would otherwise make indistinguishable from each other.
    """
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
    """Render a small, persistent grade-band legend strip (Streamlit).

    Explains the letter-grade thresholds (matches letter_grade()'s
    A/B/C/D cutoffs) so a bar's grade suffix is self-explanatory without
    having to remember or look up the scale elsewhere.
    """
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
    """Render a small-caps section label with an optional hover-help icon.

    Uses the CSS-only ".info-tip" popup (see stylesheet) instead of the
    native HTML title attribute — title tooltips have an inconsistent
    ~1.5s delay, no styling, and don't reliably show up in every browser
    context, which made the earlier version effectively invisible.
    """
    icon_html = ""
    if tooltip:
        icon_html = f'<span class="info-tip" data-tip="{html.escape(tooltip)}">?</span>'
    st.markdown(
        f'<div class="slbl" style="display:flex;align-items:center">{title}{icon_html}</div>',
        unsafe_allow_html=True,
    )


def render_what_why_how(what: str, why: str, how: str) -> None:
    """Render a compact WHAT / WHY / HOW strip at the top of a tab.

    WHAT is the plain-language framing sentence every tab already opens
    with; WHY explains what decision this chart is meant to help with;
    HOW explains, in one sentence, the actual calculation behind it —
    so a first-time reader never has to guess what's on screen or trust
    it blindly.
    """
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


def build_leaderboard_df(scores: Dict, per_run: Dict, mm: Dict, level_range: tuple=(1, 7)) -> pd.DataFrame:
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
            hovertemplate=(f"<b>{row['Method']}</b><br>Score: {row['Composite Score']:.1f} ({row['Grade']})<br>"
                           f"KTC: {row['KTC Score']:.4f}<br><extra></extra>")
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

# =========================================================
# BENCHMARK LAUNCHER - the dashboard drives the backend
# =========================================================
BENCH_LOG = Path("outputs/benchmark_log.txt")


def launch_benchmark(config_path: str | Path) -> bool:
    """Start `python example_usage.py --no-app --config <cfg>` in the background.

    The bridge writes latest.txt last, so the dashboard only flips to the new
    run once it is fully prepared.
    """
    proc = st.session_state.get('bench_proc')
    if proc is not None and proc.poll() is None:
        st.sidebar.warning("A benchmark is already running - wait for it to finish.")
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
    st.session_state.bench_aborted = False
    return True


def stop_benchmark() -> bool:
    """Abort the running benchmark process tree and keep latest.txt unchanged."""
    proc = st.session_state.get('bench_proc')
    if proc is None or proc.poll() is not None:
        st.session_state.bench_proc = None
        return False

    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        st.session_state.bench_aborted = True
        st.session_state['_bench_was_running'] = False
        st.session_state.bench_proc = None
        st.cache_data.clear()
    return True


def _get_cfg_settings() -> tuple[str, str, list[int], list[str]]:
    """Read dataset/mesh/levels/samples from sidebar session_state with safe defaults."""
    dataset_root = st.session_state.get('cfg_dataset_root', 'EvaluationData') or 'EvaluationData'
    mesh_path = st.session_state.get('cfg_mesh_path', 'Codes_Matlab/Mesh_sparse.mat') or 'Codes_Matlab/Mesh_sparse.mat'
    level_range = st.session_state.get('level_range', (1, 7))
    lvl_min, lvl_max = int(level_range[0]), int(level_range[1])
    levels = list(range(lvl_min, lvl_max + 1))
    samples = st.session_state.get('selected_samples', ['A', 'B', 'C']) or ['A', 'B', 'C']
    return dataset_root, mesh_path, levels, samples


def write_runtime_config(method_name: str) -> Path:
    """Generate a one-method benchmark config for a runtime-registered method."""
    dataset_root, mesh_path, levels, samples = _get_cfg_settings()
    levels_yaml = "[" + ", ".join(str(l) for l in levels) + "]"
    samples_yaml = "[" + ", ".join(samples) + "]"
    cfg_path = Path("configs") / f"runtime_{method_name}.yaml"
    cfg_path.write_text(
        "# Auto-generated by the dashboard Register button.\n"
        "data_plugin: KTCDataPlugin\n"
        f"dataset_root: {dataset_root}\n"
        f"mesh_path: {mesh_path}\n\n"
        f"levels: {levels_yaml}\n"
        f"samples: {samples_yaml}\n\n"
        f"methods:\n  - {method_name}\n\n"
        "method_plugin_paths:\n  - external_methods\n\n"
        "output_dir: outputs/\n",
        encoding="utf-8",
    )
    return cfg_path


def write_selected_config(methods: list[str]) -> Path:
    """Generate a benchmark config that runs only the chosen subset of methods."""
    dataset_root, mesh_path, levels, samples = _get_cfg_settings()
    levels_yaml = "[" + ", ".join(str(l) for l in levels) + "]"
    samples_yaml = "[" + ", ".join(samples) + "]"
    cfg_path = Path("configs") / "runtime_selected.yaml"
    methods_block = "".join(f"  - {m}\n" for m in methods)
    cfg_path.write_text(
        "# Auto-generated by the dashboard runtime config helper.\n"
        "data_plugin: KTCDataPlugin\n"
        f"dataset_root: {dataset_root}\n"
        f"mesh_path: {mesh_path}\n\n"
        f"levels: {levels_yaml}\n"
        f"samples: {samples_yaml}\n\n"
        f"methods:\n{methods_block}\n"
        "method_plugin_paths:\n  - external_methods\n\n"
        "output_dir: outputs/\n",
        encoding="utf-8",
    )
    return cfg_path


def _run_label(run_dir: Path) -> str:
    """Build a rich dropdown label: name + status icon + failure count."""
    name = run_dir.name
    n_fail = len(_read_run_failures(run_dir))
    try:
        per_run = json.loads((run_dir / "per_run_metrics.json").read_text(encoding="utf-8"))
        n_total = sum(len(v) for v in per_run.values()) if isinstance(per_run, dict) else 0
    except Exception:
        n_total = 0
    if n_fail:
        status = f"ERR:{n_fail}"
    elif n_total:
        status = "OK"
    else:
        status = "?"
    return f"{name}  [{status}]  {n_total} runs"


def write_rerun_failed_config(failures: list[dict]) -> Path:
    """Generate a config that re-runs only the (method, level, sample) triples that failed."""
    dataset_root, mesh_path, _, _ = _get_cfg_settings()
    methods = list(dict.fromkeys(str(f.get("method", "")) for f in failures))
    levels  = sorted({int(f["level"]) for f in failures if "level" in f})
    samples = sorted({str(f["sample"]) for f in failures if "sample" in f})
    levels_yaml  = "[" + ", ".join(str(l) for l in levels)  + "]"
    samples_yaml = "[" + ", ".join(samples) + "]"
    methods_block = "".join(f"  - {m}\n" for m in methods)
    cfg_path = Path("configs") / "runtime_rerun_failed.yaml"
    cfg_path.write_text(
        "# Auto-generated: re-runs only the failed samples from the previous run.\n"
        "data_plugin: KTCDataPlugin\n"
        f"dataset_root: {dataset_root}\n"
        f"mesh_path: {mesh_path}\n\n"
        f"levels: {levels_yaml}\n"
        f"samples: {samples_yaml}\n\n"
        f"methods:\n{methods_block}\n"
        "method_plugin_paths:\n  - external_methods\n\n"
        "output_dir: outputs/\n",
        encoding="utf-8",
    )
    return cfg_path


def _extract_config_error(log_path: Path) -> str | None:
    """Return the first ConfigError message found in BENCH_LOG, stripped of Rich markup."""
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "ConfigError" in line:
            return re.sub(r'\[[^\]]*\]', '', line).strip()
    return None


def _read_run_failures(run_dir: Path) -> list[dict]:
    """Parse failures.json from a run directory; returns [] if absent or unreadable."""
    path = run_dir / "failures.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _render_failures_panel(run_dir: Path) -> None:
    """Render an expandable failures table in the main area from failures.json."""
    failures = _read_run_failures(run_dir)
    if not failures:
        return
    rows = [
        {
            "Method": f.get("method", ""),
            "Level": f.get("level", ""),
            "Sample": f.get("sample", ""),
            "Error Type": f.get("error_type", ""),
            "Message": str(f.get("error_msg", ""))[:140],
        }
        for f in failures
    ]
    with st.expander(f"Run failures — {len(failures)} sample(s) failed", expanded=True):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for f in failures:
            tb = f.get("traceback", "")
            if tb:
                label = f"Traceback: {f.get('method','')} L{f.get('level','')} {f.get('sample','')}"
                with st.expander(label):
                    st.code(tb, language="text")


def render_benchmark_status() -> None:
    """Sidebar status for the running/finished benchmark subprocess."""
    proc = st.session_state.get('bench_proc')
    if proc is not None:
        code = proc.poll()
        cfg = st.session_state.get('bench_config', '')
        if code is None:
            st.sidebar.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:var(--amb);margin:4px 0">RUNNING | {cfg}</div>',
                unsafe_allow_html=True)
            if st.sidebar.button("Refresh status", use_container_width=True, key="bench_refresh"):
                st.rerun()
        elif code == 0:
            st.session_state.bench_proc = None
            st.cache_data.clear()
            st.rerun()  # latest.txt now points at the new run
        else:
            st.session_state.bench_proc = None
            if st.session_state.pop('bench_aborted', False):
                st.sidebar.info("Benchmark aborted. Showing previous completed data.")
            else:
                st.sidebar.error(f"Benchmark failed (exit {code}).")
                config_err = _extract_config_error(BENCH_LOG)
                if config_err:
                    st.sidebar.error(f"Config error: {config_err}")
                if BENCH_LOG.exists():
                    with st.sidebar.expander("View full benchmark log"):
                        st.code(
                            BENCH_LOG.read_text(encoding="utf-8", errors="replace"),
                            language="text",
                        )
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
                f'letter-spacing:.1em;color:var(--tx3);margin-bottom:5px">Benchmark log (last 8 lines)</div>'
                f'{log_html}</div>',
                unsafe_allow_html=True)
        with st.sidebar.expander("View full log"):
            st.code(
                BENCH_LOG.read_text(encoding="utf-8", errors="replace"),
                language="text",
            )


def render_bench_progress() -> None:
    """Full-width progress banner in the main area while a benchmark subprocess runs.

    Only the banner itself lives inside the auto-refreshing fragment below —
    this outer wrapper decides whether to mount it at all, so the sidebar and
    every other tab stay untouched while the benchmark polls.
    """
    if st.session_state.get('bench_proc') is None:
        return
    _render_bench_progress_fragment()


@st.fragment(run_every="2s")
def _render_bench_progress_fragment() -> None:
    proc = st.session_state.get('bench_proc')
    if proc is None:
        return
    if proc.poll() is not None:
        # Benchmark just finished - do one final full-app rerun so the
        # dashboard flips to the freshly prepared run, then stop auto-refreshing.
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
            total = 126  # 6 methods x 7 levels x 3 samples

    pct      = min(completed / total, 1.0) if total > 0 else 0.0
    pct_px   = f"{pct * 100:.1f}%"
    cfg_lbl  = st.session_state.get('bench_config', 'ktc_all_methods')
    cur_info = (f"&nbsp;|&nbsp; {cur_method} &nbsp;L{cur_level}/{cur_sample}"
                if cur_method else "&nbsp;|&nbsp; initialising...")

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
        f'<div class="bench-bar-active" style="width:{pct_px};transition:width .4s ease"></div>'
        f'</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx3)">'
        f'Progress updates automatically every 2s &nbsp;|&nbsp; '
        f'dashboard reloads when complete</div>'
        f'</div>',
        unsafe_allow_html=True)

    c_refresh, c_stop = st.columns([1, 1])
    with c_refresh:
        if st.button("Refresh progress", key="main_bench_refresh", use_container_width=True):
            st.rerun()
    with c_stop:
        if st.button("Stop benchmark", key="main_bench_stop", use_container_width=True):
            if stop_benchmark():
                st.info("Benchmark aborted. Previous completed dashboard data is still active.")
            st.rerun()
    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--bd);margin:4px 0 14px">',
        unsafe_allow_html=True)


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


def remove_method_from_config(method_name: str,
                              config_path: Path = Path("configs/ktc_all_methods.yaml")) -> bool:
    """Remove a method list entry from ktc_all_methods.yaml if present."""
    if not config_path.exists():
        return False
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    target = f"- {method_name}"
    kept = [line for line in lines if line.strip() != target]
    if len(kept) == len(lines):
        return False
    config_path.write_text("".join(kept), encoding="utf-8")
    return True


def remove_external_method_state(method_name: str) -> None:
    """Remove an external method from dashboard session state and widgets."""
    removed = set(st.session_state.get('_removed_external_methods', []))
    removed.add(method_name)
    st.session_state['_removed_external_methods'] = sorted(removed)
    st.session_state.uploaded_methods.pop(method_name, None)
    for key in ("_available_methods", "selected_methods", "_known_available_methods", "custom_methods"):
        values = st.session_state.get(key)
        if isinstance(values, list):
            st.session_state[key] = [value for value in values if value != method_name]
    widget_key = f"method_{method_name}"
    if widget_key in st.session_state:
        del st.session_state[widget_key]


# =========================================================
# Publish / Unpublish — promote an external method's current scores into a
# small, per-method snapshot that IS git-tracked (unlike outputs/, which is
# entirely gitignored and differs machine to machine). This is what lets a
# teammate open the dashboard right after `git pull` and see a populated
# leaderboard for a method they've never personally run.
# =========================================================
_PUBLISHED_DIR = Path("external_methods") / "_published"
_PUBLISHED_MANIFEST = _PUBLISHED_DIR / "manifest.json"


def _load_published_manifest() -> dict:
    if not _PUBLISHED_MANIFEST.exists():
        return {}
    try:
        return json.loads(_PUBLISHED_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_published_manifest(manifest: dict) -> None:
    _PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    _PUBLISHED_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _snapshot_path_for(method_name: str) -> Path:
    # Never name this literally "scores.json" — that exact filename is
    # globally gitignored (see .gitignore), which would silently drop it.
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_name)
    return _PUBLISHED_DIR / f"{safe}.baseline.json"


def publish_method(method_name: str) -> Tuple[bool, str]:
    """Snapshot method_name's current scores into a per-method file that
    git actually tracks, then register it in the (also-tracked) manifest.

    Deliberately scoped to one method's own small JSON file rather than
    un-ignoring the shared outputs/scores.json or per_run_metrics.json —
    those are regenerated per machine/run and would just create merge
    conflicts between users instead of a clean, additive "here's one more
    published method" diff.
    """
    try:
        latest_run = find_latest_run()
        scores, per_run = load_run_data(latest_run)
    except Exception as exc:
        return False, f"Could not read the active run: {exc}"

    if method_name not in scores:
        return False, f"Run {method_name} at least once before publishing it."

    snapshot = {
        "method": method_name,
        "scores": scores[method_name],
        "per_run": per_run.get(method_name, {}),
        "source_run": latest_run.name,
        "published_at": datetime.now().isoformat(timespec="seconds"),
    }
    snap_path = _snapshot_path_for(method_name)
    _PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, default=str), encoding="utf-8")

    manifest = _load_published_manifest()
    manifest[method_name] = {
        "source": st.session_state.get('uploaded_methods', {}).get(method_name, ""),
        "snapshot": snap_path.name,
        "published_at": snapshot["published_at"],
    }
    _save_published_manifest(manifest)
    append_method_to_config(method_name)
    return True, f"Published {method_name} — teammates will see this baseline after pulling."


def unpublish_method(method_name: str) -> None:
    """Undo publish_method: drop the manifest entry and delete its snapshot.

    Only ever touches the publish-specific files — never the method's own
    source (bundle dir / .py) and never the config entry, since the method
    should stay runnable even after being unpublished.
    """
    manifest = _load_published_manifest()
    entry = manifest.pop(method_name, None)
    if entry:
        (_PUBLISHED_DIR / entry.get("snapshot", "")).unlink(missing_ok=True)
        _save_published_manifest(manifest)


def _apply_published_baselines(scores: Dict, per_run: Dict, mm: Dict) -> Tuple[Dict, Dict, Dict, list]:
    """Fill in a published baseline for any method missing from the live run.

    Never overwrites a method that already has real local results — this is
    strictly a fallback for a fresh session (e.g. right after a git pull)
    that hasn't run the method here yet.
    """
    manifest = _load_published_manifest()
    filled: list = []
    if not manifest:
        return scores, per_run, mm, filled

    for name, entry in manifest.items():
        if name in scores:
            continue
        snap_path = _PUBLISHED_DIR / entry.get("snapshot", "")
        if not snap_path.exists():
            continue
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scores[name] = snap.get("scores", {})
        per_run[name] = snap.get("per_run", {})
        mm.setdefault(name, name)
        filled.append(name)
    return scores, per_run, mm, filled


def _discover_external_method_artifacts(ext_dir: Path = Path("external_methods")) -> dict[str, str]:
    """Return method names currently backed by files/bundles in external_methods."""
    found: dict[str, str] = {}
    if not ext_dir.exists():
        return found

    for file_path in ext_dir.glob("*.py"):
        try:
            for name in plugin_method_candidates(file_path):
                found[name] = file_path.name
            if is_cli_contract_script(file_path):
                try:
                    from ktc_framework.adapters.cli_plugin_wrapper import derive_cli_method_name
                    found.setdefault(derive_cli_method_name(file_path.stem), file_path.name)
                except Exception:
                    found.setdefault(file_path.stem, file_path.name)
        except Exception:
            continue

    for bundle_dir in ext_dir.iterdir():
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("_"):
            continue
        manifest_path = bundle_dir / "method.yaml"
        if not manifest_path.exists():
            continue
        try:
            from ktc_framework.methods.manifest_loader import load_manifest
            manifest = load_manifest(manifest_path)
            found[manifest.name] = bundle_dir.name
        except Exception:
            continue
    return found


def _prune_missing_external_methods(configured_methods: set[str] | None = None) -> tuple[set[str], dict[str, str]]:
    """Drop uploaded methods whose backing external_methods artifact was deleted."""
    if 'uploaded_methods' not in st.session_state:
        st.session_state.uploaded_methods = {}

    external_on_disk = _discover_external_method_artifacts()
    configured_methods = set(configured_methods or set())
    stale_methods: set[str] = set()

    for name, artifact in list(st.session_state.uploaded_methods.items()):
        target = Path("external_methods") / str(artifact)
        if name not in external_on_disk and not target.exists():
            stale_methods.add(name)

    # If the app restarts after manual deletion, session_state no longer knows
    # the upload. Treat non-builtin configured names with no external artifact as
    # stale so deleted ML zip methods do not reappear from ktc_all_methods.yaml.
    for name in configured_methods:
        if name not in BUILTIN_METHODS and name not in external_on_disk and name not in HIDDEN_METHODS:
            stale_methods.add(name)

    for name in sorted(stale_methods):
        remove_method_from_config(name)
        remove_external_method_state(name)

    if stale_methods:
        st.session_state.pop('_methods_cache', None)
        st.session_state['_method_refresh_msg'] = (
            "Removed missing external method(s): " + ", ".join(sorted(stale_methods))
        )
    return stale_methods, external_on_disk


def reset_method_upload_widget() -> None:
    """Force Streamlit's file uploader to clear its displayed filename."""
    st.session_state['_method_upload_nonce'] = st.session_state.get('_method_upload_nonce', 0) + 1
    st.session_state.pop('_last_method_upload', None)


def ensure_method_plugin_registered(plugin_path: Path) -> List[str]:
    """Add @register_method to classes that look like reconstruction methods."""
    text = plugin_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    def has_register_decorator(cls: ast.ClassDef) -> bool:
        # Recognise both decorator names the framework uses: @register_method
        # and @register. Without this, a class already decorated with @register
        # is treated as undecorated, the auto-fixer injects another decorator,
        # and the rewrite corrupts indentation ("unexpected indent" on upload).
        _known = {"register_method", "register"}
        for dec in cls.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in _known:
                return True
            if isinstance(dec, ast.Attribute) and dec.attr in _known:
                return True
        return False

    candidates: list[ast.ClassDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or has_register_decorator(node):
            continue
        has_reconstruct = any(
            isinstance(item, ast.FunctionDef) and item.name == "reconstruct"
            for item in node.body
        )
        if has_reconstruct:
            candidates.append(node)

    if not candidates:
        return []

    lines = text.splitlines(keepends=True)
    has_import = "register_method" in text and "ktc_framework.registry" in text
    if not has_import:
        insert_at = 0
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            insert_at = getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
        for idx, line in enumerate(lines[insert_at:], start=insert_at):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) or not stripped:
                insert_at = idx + 1
        lines.insert(insert_at, "from ktc_framework.registry import register_method\n")

    for cls in sorted(candidates, key=lambda item: item.lineno, reverse=True):
        line_no = min([d.lineno for d in cls.decorator_list] + [cls.lineno]) - 1
        indent = lines[line_no][:len(lines[line_no]) - len(lines[line_no].lstrip())]
        lines.insert(line_no, f"{indent}@register_method\n")

    plugin_path.write_text("".join(lines), encoding="utf-8")
    return [cls.name for cls in candidates]


def plugin_method_candidates(plugin_path: Path) -> List[str]:
    """Return class names in a plugin that can act as reconstruction methods."""
    try:
        tree = ast.parse(plugin_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        has_reconstruct = any(
            isinstance(item, ast.FunctionDef) and item.name == "reconstruct"
            for item in node.body
        )
        has_register = any(
            (isinstance(dec, ast.Name) and dec.id == "register_method")
            or (isinstance(dec, ast.Attribute) and dec.attr == "register_method")
            for dec in node.decorator_list
        )
        if has_reconstruct or has_register:
            names.append(node.name)
    return names


def is_cli_contract_script(plugin_path: Path) -> bool:
    """True if *plugin_path* is a raw KTC CLI-contract script (main() +
    argparse + if __name__ == '__main__'), as opposed to an in-process
    reconstruct(self, batch) plugin. AST-only — never imports the file.

    Returns False (not True) if the classifier can't be reached or the
    file won't parse, so callers fall through to the existing in-process
    path and get the normal "no usable method class found" message
    instead of an opaque failure here.
    """
    try:
        from ktc_framework.adapters.plugin_detector import (
            CONTRACT_CLI, PluginDetectionError, detect_contract,
        )
    except ImportError:
        return False

    try:
        return detect_contract(plugin_path) == CONTRACT_CLI
    except PluginDetectionError:
        return False


def register_cli_script(script_path: Path) -> str:
    """Wrap a raw KTC CLI-contract script as a CLIScriptPlugin and register
    it under a name derived from the filename.

    Runs as an isolated subprocess — never imported in-process, since these
    scripts commonly have unguarded heavy top-level imports (see
    ``registry.load_external_methods`` for why that matters).

    Raw CLI submissions have no author-declared name the way method.yaml
    bundles do (``manifest.name``) or in-process classes do (the class
    name) — every KTC entry is conventionally just "main.py". So the name
    is derived from the uploaded filename's stem via
    ``derive_cli_method_name`` — the *same* function
    ``registry.load_cli_scripts`` uses to re-discover this file inside a
    fresh benchmark subprocess, so the name saved here into the YAML
    config's ``methods:`` list is guaranteed to resolve to the same
    wrapper at run time instead of failing with "not registered".
    """
    from ktc_framework.adapters.cli_plugin_wrapper import (
        create_cli_wrapper_class, derive_cli_method_name,
    )
    from ktc_framework.registry import list_methods as _list_methods_
    from ktc_framework.registry import register_method as _register_method_

    name = derive_cli_method_name(script_path.stem, existing=set(_list_methods_()))

    wrapper_cls = create_cli_wrapper_class(script_path=str(script_path), name=name)
    _register_method_(wrapper_cls)
    return name


def _methods_discovery_fingerprint() -> tuple:
    """Cheap signature of everything that can change the discovered-methods list.

    Reading a handful of mtimes and session-state keys is orders of magnitude
    cheaper than the full discovery (which imports external plugin modules and
    reloads run data). When this signature is unchanged we can safely reuse the
    previous result instead of re-scanning on every Streamlit rerun.
    """
    parts: list = []
    try:
        lt = Path("outputs/latest.txt")
        parts.append(lt.read_text(encoding="utf-8").strip() if lt.exists() else "")
    except Exception:
        parts.append("")
    ext_dir = Path("external_methods")
    if ext_dir.exists():
        try:
            for p in sorted(ext_dir.glob("*.py")):
                parts.append((p.name, int(p.stat().st_mtime)))
            for d in sorted(ext_dir.iterdir()):
                if d.is_dir() and not d.name.startswith("_") and (d / "method.yaml").exists():
                    parts.append((d.name, int((d / "method.yaml").stat().st_mtime)))
        except Exception:
            pass
    parts.append(tuple(sorted(st.session_state.get('uploaded_methods', {}).keys())))
    parts.append(tuple(sorted(st.session_state.get('_removed_external_methods', []))))
    return tuple(parts)


def discover_available_methods() -> List[str]:
    """Memoized wrapper around the real discovery.

    Streamlit reruns the whole script on every widget interaction, so calling
    the (expensive) discovery each time made selecting/removing a method feel
    slow. We recompute only when the discovery fingerprint changes; otherwise we
    return the cached list from session_state.
    """
    configured_methods: set[str] = set()
    cfg_path = Path("configs/ktc_all_methods.yaml")
    if cfg_path.exists():
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            configured_methods = {str(name) for name in cfg.get("methods", [])}
        except Exception:
            configured_methods = set()
    _prune_missing_external_methods(configured_methods)

    fp = _methods_discovery_fingerprint()
    cache = st.session_state.get('_methods_cache')
    if cache is not None and cache.get('fp') == fp:
        return list(cache['methods'])
    methods = _discover_available_methods_impl()
    st.session_state['_methods_cache'] = {'fp': fp, 'methods': list(methods)}
    return methods


def _discover_available_methods_impl() -> List[str]:
    """Collect scored, configured, and registered methods without running benchmarks."""
    methods: List[str] = []
    removed_external = set(st.session_state.get('_removed_external_methods', []))
    configured_methods: set[str] = set()

    cfg_path = Path("configs/ktc_all_methods.yaml")
    if cfg_path.exists():
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            configured_methods = {str(name) for name in cfg.get("methods", [])}
        except Exception:
            configured_methods = set()

    _, external_methods_on_disk = _prune_missing_external_methods(configured_methods)
    removed_external = set(st.session_state.get('_removed_external_methods', []))

    def add(name: str):
        if (name and name not in methods and name not in HIDDEN_METHODS
                and name not in removed_external):
            methods.append(name)

    def is_visible_source(name: str) -> bool:
        return (
            name in BUILTIN_METHODS
            or name in external_methods_on_disk
            or name in st.session_state.get('uploaded_methods', {})
        )

    try:
        scores, _ = load_run_data(find_latest_run())
        for name in scores.keys():
            if is_visible_source(str(name)):
                add(str(name))
    except Exception:
        pass

    for name in configured_methods:
        if is_visible_source(str(name)):
            add(str(name))

    ext_dir = Path("external_methods")
    if 'uploaded_methods' not in st.session_state:
        st.session_state.uploaded_methods = {}
    for name in st.session_state.uploaded_methods.keys():
        add(str(name))

    if ext_dir.exists():
        try:
            from ktc_framework.registry import (
                list_methods as _list_methods,
                load_external_methods as _load_ext,
            )
            before = set(_list_methods())
            py_files = list(ext_dir.glob("*.py"))
            bundle_dirs_da = sorted(
                d for d in ext_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_") and (d / "method.yaml").exists()
            )
            if py_files or bundle_dirs_da:
                _load_ext([str(ext_dir)])
                discovered = sorted(set(_list_methods()) - before)
                for name in discovered:
                    fname = external_methods_on_disk.get(name) or next(
                        (f.name for f in py_files if name.lower() in f.stem.lower()), "")
                    if not fname:
                        # find which bundle dir this name came from
                        for bd in bundle_dirs_da:
                            try:
                                from ktc_framework.methods.manifest_loader import load_manifest
                                _m = load_manifest(bd / "method.yaml")
                                if _m.name == name:
                                    fname = bd.name
                                    break
                            except Exception:
                                pass
                    st.session_state.uploaded_methods.setdefault(name, fname or name)
                    add(name)
        except Exception:
            pass

    return methods


# =========================================================
# SIDEBAR
# =========================================================
def _render_sidebar_brand():
    """Brand header + data-freshness label (reads latest.txt mtime)."""
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
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:var(--tx);letter-spacing:.06em">EIT BENCH</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tx3);margin-top:3px;letter-spacing:.08em">RECONSTRUCTION ANALYSIS</div>
      <div class="sb-live"><div class="ldot"></div>LIVE</div>
      {_live_sub}
    </div>
    """, unsafe_allow_html=True)


def _render_sidebar_dataset_settings():
    """Dataset Settings expander (B1 / B2 / B3): root/mesh path inputs + Validate paths."""
    with st.sidebar.expander("Dataset Settings", expanded=False):
        st.markdown('<div style="font-size:10px;color:var(--tx3);margin-bottom:2px">Dataset root</div>', unsafe_allow_html=True)
        st.text_input(
            "Dataset root",
            value="EvaluationData",
            key="cfg_dataset_root",
            label_visibility="collapsed",
            placeholder="EvaluationData",
            help="Folder containing evaluation_datasets/ and GroundTruths/.",
        )
        st.markdown('<div style="font-size:10px;color:var(--tx3);margin:4px 0 2px">Mesh path</div>', unsafe_allow_html=True)
        st.text_input(
            "Mesh path",
            value="Codes_Matlab/Mesh_sparse.mat",
            key="cfg_mesh_path",
            label_visibility="collapsed",
            placeholder="Codes_Matlab/Mesh_sparse.mat",
            help="Path to Mesh_sparse.mat.",
        )
        if st.button("Validate paths", key="validate_cfg_btn", use_container_width=True):
            _d = Path(st.session_state.get("cfg_dataset_root", "EvaluationData"))
            _m = Path(st.session_state.get("cfg_mesh_path", "Codes_Matlab/Mesh_sparse.mat"))
            _eval_ok = any((_d / x).is_dir() for x in ["evaluation_datasets", "EvaluationData"])
            _gt_ok   = any((_d / x).is_dir() for x in ["GroundTruths", "groundtruths", "GroundTruth"])
            _ref_candidates = [
                _d / "evaluation_datasets" / "level1" / "ref.mat",
                _d / "EvaluationData" / "level1" / "ref.mat",
                _d / "ref.mat",
                Path("Codes_Matlab") / "TrainingData" / "ref.mat",
            ]
            st.session_state["_cfg_validation"] = {
                "root": _d.exists(),
                "eval": _eval_ok,
                "gt":   _gt_ok,
                "mesh": _m.exists(),
                "ref":  any(p.exists() for p in _ref_candidates),
            }
        _v = st.session_state.get("_cfg_validation")
        if _v:
            _checks = [
                ("dataset_root exists", _v["root"]),
                ("eval data folder",    _v["eval"]),
                ("GroundTruths folder", _v["gt"]),
                ("mesh_path exists",    _v["mesh"]),
                ("ref.mat found",       _v["ref"]),
            ]
            for _lbl, _ok in _checks:
                _c = "#1a7f37" if _ok else "#cf222e"
                _i = "OK " if _ok else "ERR"
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:10px;color:{_c};margin:2px 0">{_i} {_lbl}</div>',
                    unsafe_allow_html=True,
                )


def _render_sidebar_run_benchmark():
    """Run Benchmark section: ETA estimate, Run all / Refresh methods buttons, status."""
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

    # Runs only the methods currently ticked in the METHODS checklist below.
    if st.sidebar.button("Refresh methods", use_container_width=True, key="refresh_methods_btn"):
        st.cache_data.clear()
        st.session_state.pop('_methods_cache', None)  # force a fresh scan
        refreshed_methods = discover_available_methods()
        st.session_state['_available_methods'] = refreshed_methods
        current_selection = st.session_state.get('selected_methods', refreshed_methods.copy())
        st.session_state.selected_methods = [m for m in current_selection if m in refreshed_methods]
        if not st.session_state.selected_methods:
            st.session_state.selected_methods = refreshed_methods.copy()
        st.session_state['_method_refresh_msg'] = f"{len(refreshed_methods)} method(s) available"
        st.rerun()

    refresh_msg = st.session_state.pop('_method_refresh_msg', None)
    if refresh_msg:
        st.sidebar.markdown(
            f'<div class="method-refresh-status">{refresh_msg}</div>',
            unsafe_allow_html=True,
        )
    render_benchmark_status()


def _render_sidebar_reset_filters():
    """Reset All Filters button: restores metrics/methods/levels/samples to defaults."""
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


def _render_sidebar_metrics_selector():
    """Metrics checklist — keeps selected_metrics in sync as new metrics appear."""
    st.sidebar.markdown("## Metrics")
    if 'selected_metrics' not in st.session_state:
        st.session_state.selected_metrics = ALL_METRICS_SIDEBAR.copy()
    previous_metrics = st.session_state.get('_known_metrics_sidebar', [])
    newly_available_metrics = [m for m in ALL_METRICS_SIDEBAR if m not in previous_metrics]
    if newly_available_metrics:
        st.session_state.selected_metrics = [
            m for m in st.session_state.selected_metrics if m in ALL_METRICS_SIDEBAR
        ]
        for m in newly_available_metrics:
            if m not in st.session_state.selected_metrics:
                st.session_state.selected_metrics.append(m)
    st.session_state['_known_metrics_sidebar'] = ALL_METRICS_SIDEBAR.copy()
    for m in ALL_METRICS_SIDEBAR:
        checked = m in st.session_state.selected_metrics
        if st.sidebar.checkbox(m, value=checked, key=f"metric_{m}"):
            if m not in st.session_state.selected_metrics:
                st.session_state.selected_metrics.append(m)
        else:
            if m in st.session_state.selected_metrics:
                st.session_state.selected_metrics.remove(m)


def _render_sidebar_method_selector():
    """Methods checklist — one row per method with inline Run + ✕ (external only) buttons."""
    st.sidebar.markdown("## Methods")
    removed_external = set(st.session_state.get('_removed_external_methods', []))
    available_methods = [
        m for m in st.session_state.get('_available_methods', [])
        if m not in removed_external
    ]
    st.session_state['_available_methods'] = available_methods
    if 'selected_methods' not in st.session_state:
        st.session_state.selected_methods = available_methods.copy()
    previous_available = st.session_state.get('_known_available_methods', [])
    newly_available = [m for m in available_methods if m not in previous_available]
    if newly_available:
        st.session_state.selected_methods = [
            m for m in st.session_state.selected_methods if m in available_methods
        ]
        for m in newly_available:
            if m not in st.session_state.selected_methods:
                st.session_state.selected_methods.append(m)
    st.session_state['_known_available_methods'] = available_methods.copy()

    _uploaded = st.session_state.get('uploaded_methods', {})

    if available_methods:
        for m in available_methods:
            display_name = method_display_name(m)
            checked = m in st.session_state.selected_methods
            is_external = m in _uploaded

            # Row: [checkbox  |  Run  |  ✕ (external only)]
            _chk_col, _run_col, _del_col = st.sidebar.columns([5, 2, 1])
            with _chk_col:
                new_val = st.checkbox(display_name, value=checked, key=f"method_{m}")
            if new_val and m not in st.session_state.selected_methods:
                st.session_state.selected_methods.append(m)
            elif not new_val and m in st.session_state.selected_methods:
                st.session_state.selected_methods.remove(m)
            with _run_col:
                if st.button("Run", key=f"mrun_{m}", use_container_width=True,
                             help=f"Benchmark {display_name} now"):
                    _cfg = write_runtime_config(m)
                    if launch_benchmark(_cfg):
                        st.rerun()
            with _del_col:
                if is_external:
                    if st.button("✕", key=f"mdel_{m}", help="Remove plugin"):
                        _fname = _uploaded.get(m, m)
                        try:
                            from ktc_framework.registry import unregister_method as _unreg
                            _unreg(m)
                        except Exception:
                            pass
                        _tgt = Path("external_methods") / _fname
                        if _tgt.is_dir():
                            shutil.rmtree(_tgt, ignore_errors=True)
                        else:
                            _tgt.unlink(missing_ok=True)
                        remove_method_from_config(m)
                        remove_external_method_state(m)
                        reset_method_upload_widget()
                        st.cache_data.clear()
                        st.session_state['_method_refresh_msg'] = f"Removed: {m}"
                        st.rerun()
    else:
        st.sidebar.markdown('<div style="font-size:11px;color:var(--tx3)">Loading methods...</div>', unsafe_allow_html=True)


def _render_sidebar_level_filter():
    """Level Filter: from/to level range, clamped and swapped if reversed."""
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


def _render_sidebar_sample_filter():
    """Sample Filter: A/B/C checkboxes."""
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


def _render_scan_external_methods_button():
    """Scan button - always visible. Picks up any .py already in
    external_methods/ so the user doesn't have to re-upload files that
    are already on disk."""
    from ktc_framework.registry import (
        list_methods as _list_methods,
        load_external_methods as _load_ext,
    )
    if 'uploaded_methods' not in st.session_state:
        st.session_state.uploaded_methods = {}
    removed_external = set(st.session_state.get('_removed_external_methods', []))
    ext_dir_for_panel = Path("external_methods")
    if ext_dir_for_panel.exists():
        disk_plugins = {
            name: file_path.name
            for file_path in ext_dir_for_panel.glob("*.py")
            for name in plugin_method_candidates(file_path)
            if name not in BUILTIN_METHODS and name not in HIDDEN_METHODS and name not in removed_external
        }
        # Only class-based .py plugins are re-derived above — bundles and CLI
        # scripts aren't, so don't drop them just for being absent from
        # disk_plugins. Keep any entry whose backing file/dir still exists.
        st.session_state.uploaded_methods = {
            name: fname
            for name, fname in st.session_state.uploaded_methods.items()
            if name not in removed_external
            and (name in disk_plugins or (ext_dir_for_panel / fname).exists())
        }
        for name, fname in disk_plugins.items():
            st.session_state.uploaded_methods.setdefault(name, fname)

    if st.sidebar.button("Import from external_methods/", key="scan_ext_btn",
                         use_container_width=True,
                         help="Register .py plugins and .zip bundles already in "
                              "external_methods/ so they can be run"):
        ext_dir = Path("external_methods")
        ext_dir.mkdir(exist_ok=True)
        py_files = list(ext_dir.glob("*.py"))
        bundle_dirs = sorted(
            d for d in ext_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "method.yaml").exists()
        ) if ext_dir.exists() else []

        if not py_files and not bundle_dirs:
            st.sidebar.info("external_methods/ is empty — upload a .py or .zip file below.")
        else:
            before = set(_list_methods())
            try:
                # ── .py files ────────────────────────────────────────────
                auto_fixed = []
                candidates_by_file = {}
                for file_path in py_files:
                    auto_fixed.extend(ensure_method_plugin_registered(file_path))
                    candidates_by_file[file_path.name] = plugin_method_candidates(file_path)

                # load_ext picks up both .py files AND bundle subdirs
                _load_ext([str(ext_dir)])
                available_after = set(_list_methods())

                ready_methods = sorted(
                    {nm for nms in candidates_by_file.values() for nm in nms if nm in available_after}
                )
                new_methods = sorted(available_after - before)
                for nm in ready_methods:
                    fname = next(
                        (f.name for f in py_files if nm.lower() in f.stem.lower()),
                        py_files[0].name if py_files else "",
                    )
                    st.session_state.uploaded_methods[nm] = fname

                # ── raw CLI-contract scripts (main() + argparse + __main__) ──
                # load_external_methods() already skips these when importing
                # .py files (see registry.py), so they never show up in
                # candidates_by_file / ready_methods above. Wrap each one not
                # already registered from a previous scan as a CLIScriptPlugin
                # instead of silently dropping it.
                cli_found = []
                already_registered_files = set(st.session_state.uploaded_methods.values())
                for file_path in py_files:
                    if file_path.name in already_registered_files:
                        continue  # wrapped in a prior scan
                    if file_path.name in candidates_by_file and candidates_by_file[file_path.name]:
                        continue  # already handled as an in-process class
                    if not is_cli_contract_script(file_path):
                        continue
                    try:
                        cli_name = register_cli_script(file_path)
                        st.session_state.uploaded_methods[cli_name] = file_path.name
                        cli_found.append(cli_name)
                    except Exception as exc:
                        st.sidebar.warning(f"Could not wrap {file_path.name} as a CLI method: {exc}")

                # ── bundle dirs ───────────────────────────────────────────
                bundle_found = []
                for bd in bundle_dirs:
                    try:
                        from ktc_framework.methods.manifest_loader import load_manifest, ManifestError
                        manifest = load_manifest(bd / "method.yaml")
                        if manifest.name in available_after:
                            st.session_state.uploaded_methods[manifest.name] = bd.name
                            bundle_found.append(manifest.name)
                    except Exception as _be:
                        st.sidebar.warning(f"Skipping bundle {bd.name}: {_be}")

                all_found = sorted(set(ready_methods) | set(bundle_found) | set(cli_found))
                removed = set(st.session_state.get('_removed_external_methods', []))
                removed.difference_update(all_found)
                st.session_state['_removed_external_methods'] = sorted(removed)
                current_available = st.session_state.get('_available_methods', [])
                for nm in all_found:
                    if nm not in current_available:
                        current_available.append(nm)
                st.session_state['_available_methods'] = current_available

                if all_found:
                    for nm in all_found:
                        append_method_to_config(nm)
                    label = "Found" if (new_methods or bundle_found or cli_found) else "Already registered"
                    st.session_state['_method_refresh_msg'] = f"{label}: {', '.join(all_found)}"
                    if auto_fixed:
                        st.session_state['_method_refresh_msg'] += " (added missing @register_method)"
                    st.rerun()
                else:
                    st.sidebar.info("No usable methods found in external_methods/")
            except Exception as exc:
                st.sidebar.error(f"Scan failed: {exc}")

    # -- Registered plugins - always show with per-plugin action buttons --
    if st.session_state.uploaded_methods:
        st.sidebar.markdown(
            '<div class="plugin-section-label">Registered plugins</div>',
            unsafe_allow_html=True)
        for nm, fname in list(st.session_state.uploaded_methods.items()):
            st.sidebar.markdown(
                f'<div class="plugin-item">'
                f'<div class="plugin-name">{nm}</div>'
                f'<div class="plugin-file">{fname}</div>'
                f'</div>',
                unsafe_allow_html=True)
            ca, cb, cc = st.sidebar.columns([1, 1, 1], gap="small")
            if ca.button("Run", key=f"run_up_{nm}", use_container_width=True,
                         help=f"Benchmark {nm} now and add it to ktc_all_methods.yaml "
                              "for future full runs"):
                append_method_to_config(nm)
                cfg_path = write_runtime_config(nm)
                if launch_benchmark(cfg_path):
                    st.rerun()
            is_published = nm in _load_published_manifest()
            if is_published:
                if cb.button("Unpublish", key=f"unpub_{nm}", use_container_width=True,
                             help="Remove this method's published baseline — teammates "
                                  "won't see its scores until they run it themselves"):
                    unpublish_method(nm)
                    st.session_state['_method_refresh_msg'] = f"Unpublished: {nm}"
                    st.rerun()
            else:
                if cb.button("Publish", key=f"pub_{nm}", use_container_width=True,
                             help="Snapshot this method's current scores into a git-tracked "
                                  "baseline so teammates see them right after pulling"):
                    ok, msg = publish_method(nm)
                    st.session_state['_method_refresh_msg'] = msg
                    if ok:
                        st.rerun()
                    else:
                        st.sidebar.warning(msg)
            if cc.button("Remove", key=f"rm_up_{nm}", use_container_width=True,
                         help="Unregister and delete the plugin file from disk"):
                try:
                    from ktc_framework.registry import unregister_method as _unregister_method
                    _unregister_method(nm)
                except Exception:
                    pass
                unpublish_method(nm)  # a deleted method shouldn't keep a dangling baseline
                plugin_path = Path("external_methods") / fname
                if plugin_path.is_dir():
                    shutil.rmtree(plugin_path, ignore_errors=True)
                else:
                    plugin_path.unlink(missing_ok=True)
                remove_method_from_config(nm)
                remove_external_method_state(nm)
                reset_method_upload_widget()
                st.cache_data.clear()
                st.session_state['_method_refresh_msg'] = f"Removed: {nm}"
                st.rerun()
    else:
        st.sidebar.markdown(
            '<div class="plugin-empty">No plugins yet. Upload .py or scan.</div>',
            unsafe_allow_html=True)


def _handle_zip_plugin_upload(up, dest_dir: Path, sig: str) -> None:
    """Extract and register an uploaded .zip as a method bundle.

    Explicit method.yaml bundles are still preferred. If the upload is a raw
    ML/KTC repository zip, detect a CLI entry script and generate method.yaml
    so future benchmark subprocesses can rediscover the method from disk.

    ``sig`` is only recorded in session_state at a definitive outcome
    (success or a caught rejection) — see the caller for why.
    """
    try:
        from ktc_framework.methods.manifest_loader import (
            extract_archive, extract_bundle, load_manifest, ManifestError,
        )
        from ktc_framework.methods.subprocess_wrapper import create_wrapper_class
        from ktc_framework.registry import (
            list_methods as _list_methods,
            register_method as _register_method,
        )
        bundle_name = Path(up.name).stem
        bundle_dest = dest_dir / bundle_name
        tmp_zip = dest_dir / up.name
        tmp_zip.write_bytes(up.getbuffer())
        try:
            bundle_dir = extract_bundle(tmp_zip, bundle_dest)
            generated = False
            manifest_path = bundle_dir / "method.yaml"
        except ManifestError:
            shutil.rmtree(bundle_dest, ignore_errors=True)
            bundle_dir = extract_archive(tmp_zip, bundle_dest)
            manifest_path = _generate_manifest_for_raw_zip(
                bundle_dir=bundle_dir,
                uploaded_stem=bundle_name,
                existing=set(_list_methods()),
            )
            generated = True
        manifest = load_manifest(bundle_dir / "method.yaml")
        wrapper_cls = create_wrapper_class(manifest)
        _register_method(wrapper_cls)
        st.session_state.uploaded_methods[manifest.name] = bundle_dest.name
        append_method_to_config(manifest.name)
        removed = set(st.session_state.get('_removed_external_methods', []))
        removed.discard(manifest.name)
        st.session_state['_removed_external_methods'] = sorted(removed)
        current_available = st.session_state.get('_available_methods', [])
        if manifest.name not in current_available:
            current_available.append(manifest.name)
        st.session_state['_available_methods'] = current_available
        st.session_state['_method_refresh_msg'] = (
            f"Registered raw zip method: {manifest.name} (generated {manifest_path.name})"
            if generated else f"Registered bundle: {manifest.name}"
        )
        st.session_state['_last_method_upload'] = sig
        reset_method_upload_widget()
        tmp_zip.unlink(missing_ok=True)
        st.rerun()
    except Exception as exc:
        st.session_state['_last_method_upload'] = sig
        if 'tmp_zip' in locals():
            tmp_zip.unlink(missing_ok=True)
        shutil.rmtree(dest_dir / up.name.rsplit(".", 1)[0], ignore_errors=True)
        reset_method_upload_widget()
        st.sidebar.error(
            f"Bundle rejected: {exc}\n\n"
            "Your zip must contain method.yaml at its root:\n"
            "  my_method.zip\n"
            "  ├── method.yaml   ← required\n"
            "  ├── main_python.py\n"
            "  └── model.h5      ← optional weights\n\n"
            "Raw GitHub repo zips do not work — package only the "
            "files your solver needs, with method.yaml alongside them."
        )


def _generate_manifest_for_raw_zip(
    bundle_dir: Path,
    uploaded_stem: str,
    existing: set,
) -> Path:
    entry = _find_zip_cli_entry(bundle_dir)
    if entry is None:
        from ktc_framework.methods.entry_detector import CONTRACT_CLI_SCRIPT, detect_entry_point
        try:
            candidate, contract = detect_entry_point(bundle_dir)
            entry = candidate if contract == CONTRACT_CLI_SCRIPT else None
        except FileNotFoundError:
            entry = None
    if entry is None:
        raise ValueError(
            "No KTC-style CLI entry script found. Expected main.py/main_python.py "
            "or a script with three positional argparse arguments: input, output, level."
        )

    name = _safe_method_name(uploaded_stem, existing)
    rel_entry = entry.relative_to(bundle_dir).as_posix()
    rel_cwd = entry.parent.relative_to(bundle_dir).as_posix()
    if rel_cwd == ".":
        rel_cwd = "."
    check_import = _infer_runtime_import(bundle_dir, entry)
    weights = _discover_weight_files(bundle_dir)
    env_override = f"{name.upper()}_PYTHON"

    lines = [
        f"name: {name}",
        "description: Auto-generated wrapper for uploaded ML/KTC zip.",
        "",
        "runtime:",
        '  python_versions: ["3.12", "3.11", "3.10"]',
        f"  env_override: {env_override}",
    ]
    if check_import:
        lines.append(f"  check_import: {check_import}")
    lines.extend([
        "",
        "solver:",
        f"  entry_point: {rel_entry}",
        f"  working_dir: {rel_cwd}",
        "  timeout: 900",
        '  args: ["input_dir", "output_dir", "level"]',
    ])
    if weights:
        lines.append("")
        lines.append("weights:")
        lines.extend(f"  - {w}" for w in weights)

    manifest_path = bundle_dir / "method.yaml"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _find_zip_cli_entry(bundle_dir: Path) -> Path | None:
    from ktc_framework.adapters.plugin_detector import (
        CONTRACT_CLI, PluginDetectionError, detect_contract, has_argparse_signature,
    )

    py_files = sorted(
        [p for p in bundle_dir.rglob("*.py") if "__pycache__" not in p.parts],
        key=lambda p: (
            p.name not in {"main.py", "main_python.py"},
            len(p.parts),
            p.as_posix().lower(),
        ),
    )
    for path in py_files:
        try:
            if detect_contract(path) == CONTRACT_CLI or has_argparse_signature(path):
                return path
        except PluginDetectionError:
            continue
    return None


def _safe_method_name(stem: str, existing: set) -> str:
    base = re.sub(r"\W+", "_", stem).strip("_") or "UploadedMLMethod"
    if not base[0].isalpha():
        base = f"Method_{base}"
    name = base
    suffix = 2
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _infer_runtime_import(bundle_dir: Path, entry: Path) -> str | None:
    texts = []
    for name in ("requirements.txt", "environment.yml", "environment.yaml"):
        path = next(bundle_dir.rglob(name), None)
        if path is not None:
            try:
                texts.append(path.read_text(encoding="utf-8", errors="ignore").lower())
            except OSError:
                pass
    try:
        texts.append(entry.read_text(encoding="utf-8", errors="ignore").lower())
    except OSError:
        pass
    blob = "\n".join(texts)
    for needle, import_name in (
        ("tensorflow", "tensorflow"),
        ("torch-geometric", "torch_geometric"),
        ("torch_geometric", "torch_geometric"),
        ("pytorch", "torch"),
        ("torch", "torch"),
        ("opencv", "cv2"),
        ("cv2", "cv2"),
        ("scikit-image", "skimage"),
        ("skimage", "skimage"),
    ):
        if needle in blob:
            return import_name
    return None


def _discover_weight_files(bundle_dir: Path) -> list[str]:
    suffixes = {".h5", ".keras", ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".joblib", ".npz"}
    weights = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            weights.append(path.relative_to(bundle_dir).as_posix())
    return weights


def _handle_py_plugin_upload(up, dest_dir: Path, before: set, sig: str) -> None:
    """Register an uploaded .py as either a raw CLI-contract script (run
    as an isolated subprocess) or an in-process reconstruct(self, batch)
    class — classified BEFORE any attempt to import/exec the file.

    ``sig`` is only recorded in session_state at a definitive outcome
    (success or a caught rejection) — see the caller for why.
    """
    from ktc_framework.registry import (
        get_method as _get_method, list_methods as _list_methods,
        load_external_methods as _load_ext,
    )

    dest = dest_dir / Path(up.name).name
    dest.write_bytes(up.getbuffer())

    # Classify BEFORE any attempt to import/exec the file. A raw
    # CLI-contract script (main() + argparse + __main__) commonly
    # has unguarded heavy top-level imports (TensorFlow, model
    # loading, ...) — it must never be exec'd inside this
    # process, only run as an isolated subprocess.
    if is_cli_contract_script(dest):
        try:
            cli_name = register_cli_script(dest)
            st.session_state.uploaded_methods[cli_name] = dest.name
            append_method_to_config(cli_name)
            removed = set(st.session_state.get('_removed_external_methods', []))
            removed.discard(cli_name)
            st.session_state['_removed_external_methods'] = sorted(removed)
            current_available = st.session_state.get('_available_methods', [])
            if cli_name not in current_available:
                current_available.append(cli_name)
            st.session_state['_available_methods'] = current_available
            st.session_state['_method_refresh_msg'] = (
                f"Registered CLI method: {cli_name} (runs as an isolated subprocess)"
            )
            st.session_state['_last_method_upload'] = sig
            reset_method_upload_widget()
            st.rerun()
        except Exception as exc:
            st.session_state['_last_method_upload'] = sig
            dest.unlink(missing_ok=True)
            reset_method_upload_widget()
            st.sidebar.error(f"Rejected {dest.name}: {exc}")
    else:
        try:
            auto_fixed = ensure_method_plugin_registered(dest)
            candidate_names = plugin_method_candidates(dest)
            _load_ext([str(dest_dir)])
            available_after = set(_list_methods())
            new_methods = sorted(name for name in candidate_names if name not in before and name in available_after)
            ready_methods = sorted(
                nm for nm in candidate_names if nm in available_after
            )
            if ready_methods:
                for nm in ready_methods:
                    if not callable(getattr(_get_method(nm), "reconstruct", None)):
                        st.sidebar.warning(f"{nm} has no reconstruct(batch) - will fail at run time.")
                    st.session_state.uploaded_methods[nm] = dest.name
                    append_method_to_config(nm)
                removed = set(st.session_state.get('_removed_external_methods', []))
                removed.difference_update(ready_methods)
                st.session_state['_removed_external_methods'] = sorted(removed)
                current_available = st.session_state.get('_available_methods', [])
                for nm in ready_methods:
                    if nm not in current_available:
                        current_available.append(nm)
                st.session_state['_available_methods'] = current_available
                label = "Registered" if new_methods else "Already registered"
                st.session_state['_method_refresh_msg'] = f"{label}: {', '.join(ready_methods)}"
                if auto_fixed:
                    st.session_state['_method_refresh_msg'] += " (added missing @register_method)"
                st.session_state['_last_method_upload'] = sig
                reset_method_upload_widget()
                st.rerun()
            else:
                st.session_state['_last_method_upload'] = sig
                dest.unlink(missing_ok=True)
                reset_method_upload_widget()
                st.sidebar.warning(
                    "No usable method class found - file removed. "
                    "Add a class with reconstruct(self, batch), decorate it with "
                    "@register_method, or upload a raw KTC CLI script "
                    "(main() + argparse + if __name__ == '__main__')."
                )
        except Exception as exc:
            st.session_state['_last_method_upload'] = sig
            dest.unlink(missing_ok=True)
            reset_method_upload_widget()
            st.sidebar.error(f"Rejected {dest.name}: {exc}")


def _render_upload_new_plugin_widget():
    """Upload new plugin widget: accepts .py (in-process class or raw CLI
    script) or .zip (method.yaml bundle)."""
    from ktc_framework.registry import list_methods as _list_methods

    st.sidebar.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'color:var(--tx3);margin:8px 0 5px;text-transform:uppercase;'
        'letter-spacing:.1em">Upload new plugin</div>',
        unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        'color:var(--tx3);line-height:1.55;margin-bottom:6px">'
        '<b style="color:var(--tx2)">.py</b> — class with <code>reconstruct(self, batch)</code>, '
        'or a raw KTC CLI script (<code>main()</code> + argparse)<br>'
        '<b style="color:var(--tx2)">.zip</b> — ML bundle with <code>method.yaml</code> at root</div>',
        unsafe_allow_html=True)
    upload_key = f"method_upload_{st.session_state.get('_method_upload_nonce', 0)}"
    up = st.sidebar.file_uploader("Upload plugin (.py or .zip)", type=["py", "zip"], key=upload_key,
                                  label_visibility="collapsed")
    if up is not None:
        sig = f"{up.name}:{up.size}"
        if st.session_state.get('_last_method_upload') != sig:
            dest_dir = Path("external_methods")
            dest_dir.mkdir(exist_ok=True)
            before = set(_list_methods())

            # Mark this upload "handled" only AFTER a definitive outcome
            # (success or a caught rejection), not before attempting it.
            # create_wrapper_class() can block for tens of seconds probing
            # interpreters/packages; if a script rerun interrupts it mid-way
            # (e.g. runOnSave firing on an unrelated file save), Streamlit's
            # RerunException/StopException are BaseException, not Exception,
            # so the handlers' `except Exception` blocks never run — no
            # cleanup, no error shown. Marking sig upfront would then block
            # this exact file from ever being retried. Leaving it unmarked
            # here means an interrupted attempt is simply retried next run.
            if up.name.endswith(".zip"):
                _handle_zip_plugin_upload(up, dest_dir, sig)
            else:
                _handle_py_plugin_upload(up, dest_dir, before, sig)


def _render_sidebar_add_method():
    """Add Method: Scan external_methods/ button, then Upload new plugin (.py/.zip)."""
    st.sidebar.markdown("## Add Method")
    if 'uploaded_methods' not in st.session_state:
        st.session_state.uploaded_methods = {}

    _render_scan_external_methods_button()
    _render_upload_new_plugin_widget()


def _render_sidebar_run_history():
    """Run History: load/delete past runs, re-run failed samples, preview scores."""
    st.sidebar.markdown("## Run History")
    runs_root = Path("outputs")
    def _dashboard_run_has_data(run_dir: Path) -> bool:
        scores_path = run_dir / "scores.json"
        per_run_path = run_dir / "per_run_metrics.json"
        if not scores_path.exists() or not per_run_path.exists():
            return False
        try:
            with scores_path.open(encoding="utf-8") as f:
                scores_data = json.load(f)
            with per_run_path.open(encoding="utf-8") as f:
                per_run_data = json.load(f)
            total = sum(len(v) for v in per_run_data.values()) if isinstance(per_run_data, dict) else 0
            return bool(scores_data) and bool(per_run_data) and total > 0
        except Exception:
            return False

    run_dirs = [
        d for d in sorted(runs_root.glob("run_*"), reverse=True)
        if _dashboard_run_has_data(d)
    ] if runs_root.exists() else []

    if run_dirs:
        run_names = [d.name for d in run_dirs]
        # Current loaded run name
        current_run = find_latest_run().name
        st.sidebar.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--grn);margin:4px 0">'
            f'Active: {current_run}</div>', unsafe_allow_html=True)

        default_idx = run_names.index(current_run) if current_run in run_names else 0
        # C1: rich labels with status icon + failure count
        chosen_run = st.sidebar.selectbox(
            "Load run:", run_names, index=default_idx, key="selected_run",
            label_visibility="collapsed",
            format_func=lambda name: _run_label(runs_root / name))

        c_load, c_del = st.sidebar.columns(2)
        with c_load:
            if st.button("Load", key="load_run_btn", use_container_width=True):
                selected_path = runs_root / chosen_run
                (runs_root / "latest.txt").write_text(str(selected_path.resolve()))
                st.session_state.pop("_confirm_delete_run", None)
                st.cache_data.clear()
                st.rerun()
        with c_del:
            # C2: delete run with confirmation
            _active_is_chosen = chosen_run == current_run
            if st.button(
                "Delete", key="delete_run_btn", use_container_width=True,
                disabled=_active_is_chosen,
                help="Cannot delete the currently active run" if _active_is_chosen else "Delete this run from disk",
            ):
                st.session_state["_confirm_delete_run"] = chosen_run

        _pending_del = st.session_state.get("_confirm_delete_run")
        if _pending_del and _pending_del == chosen_run:
            st.sidebar.warning(f"Delete **{_pending_del}**? This cannot be undone.")
            c_yes, c_no = st.sidebar.columns(2)
            with c_yes:
                if st.button("Yes", key="confirm_del_yes", use_container_width=True):
                    shutil.rmtree(runs_root / _pending_del, ignore_errors=True)
                    st.session_state.pop("_confirm_delete_run", None)
                    st.cache_data.clear()
                    st.rerun()
            with c_no:
                if st.button("No", key="confirm_del_no", use_container_width=True):
                    st.session_state.pop("_confirm_delete_run", None)
                    st.rerun()

        # C3: re-run failed only — visible when active run has failures
        _active_failures = _read_run_failures(runs_root / current_run)
        if _active_failures:
            st.sidebar.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                f'color:#cf222e;margin:6px 0 2px">'
                f'{len(_active_failures)} failed sample(s) in active run</div>',
                unsafe_allow_html=True)
            if st.sidebar.button("Re-run failed only", key="rerun_failed_btn",
                                 use_container_width=True):
                _cfg = write_rerun_failed_config(_active_failures)
                if launch_benchmark(_cfg):
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
                ktc = mets.get('KTC score', mets.get('ktc_score', '-'))
                ktc_str = f"{ktc:.4f}" if isinstance(ktc, float) else str(ktc)
                st.sidebar.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--tx2);padding:2px 0">'
                    f'  {method}: KTC={ktc_str}</div>',
                    unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            '<div style="font-size:11px;color:var(--tx3);margin:4px 0">No runs yet.<br>Run example_usage.py first.</div>',
            unsafe_allow_html=True)


def _render_method_library_contents(container) -> None:
    """Method Library body: full catalog of every method, registered or not.

    Three groups:
      - Built-in: the framework's own classes (never uploaded, always present).
      - Registered external: uploaded/scanned methods currently runnable.
      - Not yet registered: files sitting in external_methods/ that look
        like a method (or a zip/bundle) but haven't been imported — a
        Scan or Upload is still needed before they'll run.

    ``container`` is whatever st.markdown-capable object to render into
    (``st`` for the main area, ``st.sidebar`` for the sidebar) so this one
    body can be reused from either location.
    """
    from ktc_framework.registry import list_methods as _list_methods_lib

    uploaded = st.session_state.get('uploaded_methods', {})
    published = _load_published_manifest()
    # Classify against the live registry, not the uploaded_methods bookkeeping
    # dict — that dict is only populated by Scan/Upload and stays empty for
    # methods that were auto-imported at startup (e.g. a fresh session right
    # after a teammate's `git pull`), which would otherwise misreport a
    # perfectly runnable method as "not yet registered".
    registered = sorted(_list_methods_lib())
    builtin = [m for m in registered if m in BUILTIN_METHODS]
    ext_registered = [m for m in registered if m not in BUILTIN_METHODS and m not in HIDDEN_METHODS]

    def _row(label: str, sub: str = "") -> str:
        sub_html = f' <span style="color:var(--tx3)">— {sub}</span>' if sub else ""
        return (
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:var(--tx2);padding:2px 0 2px 4px;line-height:1.5">{label}{sub_html}</div>'
        )

    def _section_label(text: str, color: str = "var(--tx3)") -> str:
        return (
            f'<div style="font-size:10px;color:{color};margin:8px 0 3px;'
            f'text-transform:uppercase;letter-spacing:.08em">{text}</div>'
        )

    def _kind_of(fname: str) -> str:
        if not fname:
            return "auto-imported"
        if not fname.endswith(".py"):
            return "bundle (method.yaml)"
        try:
            if is_cli_contract_script(Path("external_methods") / fname):
                return "CLI script"
        except Exception:
            pass
        return "in-process plugin"

    if builtin:
        container.markdown(_section_label(f"Built-in ({len(builtin)})"), unsafe_allow_html=True)
        for m in builtin:
            container.markdown(_row(method_display_name(m)), unsafe_allow_html=True)

    if ext_registered:
        container.markdown(_section_label(f"Registered external ({len(ext_registered)})"), unsafe_allow_html=True)
        for m in ext_registered:
            fname = uploaded.get(m, "")
            sub = f"{_kind_of(fname)} · {fname}" if fname else _kind_of(fname)
            if m in published:
                sub += " · published"
            container.markdown(_row(method_display_name(m), sub), unsafe_allow_html=True)

    # -- Not yet registered: raw files sitting in external_methods/ whose
    # derived/manifest name isn't in the live registry yet.
    ext_dir = Path("external_methods")
    unregistered: list[tuple[str, str]] = []
    if ext_dir.exists():
        for p in sorted(ext_dir.iterdir()):
            if p.name.startswith("_") or p.name.startswith("."):
                continue
            if p.is_file() and p.suffix == ".py":
                try:
                    candidates = plugin_method_candidates(p)
                except Exception:
                    candidates = []
                if candidates:
                    if not any(c in registered for c in candidates):
                        unregistered.append((p.name, "not imported — click Import from external_methods/"))
                    continue
                try:
                    if is_cli_contract_script(p):
                        from ktc_framework.adapters.cli_plugin_wrapper import derive_cli_method_name
                        if derive_cli_method_name(p.stem) not in registered:
                            unregistered.append((p.name, "not imported — click Import from external_methods/"))
                except Exception:
                    pass
            elif p.is_file() and p.suffix == ".zip":
                unregistered.append((p.name, "zip — upload it, or extract then Import"))
            elif p.is_dir() and (p / "method.yaml").exists():
                try:
                    from ktc_framework.methods.manifest_loader import load_manifest
                    manifest_name = load_manifest(p / "method.yaml").name
                except Exception:
                    manifest_name = None
                if manifest_name is None or manifest_name not in registered:
                    unregistered.append((p.name, "bundle — not imported"))

    if unregistered:
        container.markdown(
            _section_label(f"Not yet registered ({len(unregistered)})", color="var(--amb)"),
            unsafe_allow_html=True)
        for fname, hint in unregistered:
            container.markdown(_row(fname, hint), unsafe_allow_html=True)

    if not builtin and not ext_registered and not unregistered:
        container.markdown(
            '<div style="font-size:11px;color:var(--tx3)">No methods found.</div>',
            unsafe_allow_html=True)


def render_sidebar():
    _render_sidebar_brand()
    _render_sidebar_dataset_settings()
    _render_sidebar_run_benchmark()

    st.sidebar.markdown("---")
    _render_sidebar_reset_filters()

    st.sidebar.markdown("---")
    _render_sidebar_metrics_selector()

    st.sidebar.markdown("---")
    _render_sidebar_method_selector()

    st.sidebar.markdown("---")
    _render_sidebar_level_filter()

    st.sidebar.markdown("---")
    _render_sidebar_sample_filter()

    st.sidebar.markdown("---")
    _render_sidebar_add_method()

    # -- Export -----------------------------------------------
    # Kept inline (not its own helper): the st.empty() placeholder created
    # here must be returned so main() can render the HTML export into this
    # exact sidebar slot later in the script run.
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Export")
    if st.sidebar.button("Export HTML Report", use_container_width=True, key="pdf_sidebar_btn"):
        st.session_state['_trigger_pdf'] = True
    pdf_export_slot = st.sidebar.empty()

    st.sidebar.markdown("---")
    _render_sidebar_run_history()

    return pdf_export_slot
    return pdf_export_slot

# =========================================================
# VIEW 1 - LEADERBOARD  (original logic)
# =========================================================
@st.fragment
def view_leaderboard(scores:Dict, per_run:Dict, sel_metrics:list=None, mm:Dict=None, level_range:tuple=(1,7)):
    if sel_metrics is None:
        sel_metrics = ['KTC Score']
    if mm is None:
        mm = {}
    lvl_min, lvl_max = level_range

    render_what_why_how(
        what="Every method was run on the same set of tests, so this is a direct, "
             "apples-to-apples ranking — from the strongest overall result to the weakest.",
        why="With 6+ methods and dozens of metrics each, you need one trustworthy number "
            "to answer \"which method should I actually use?\" before digging into the details.",
        how="Each method's per-test KTC scores are averaged, then rescaled to a 0-100 "
            "Composite Score and assigned a letter grade (A-D) using the bands shown "
            "under the chart below.",
    )

    if lvl_min != 1 or lvl_max != 7:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--amb);'
            f'margin-bottom:6px">Filtered: levels {lvl_min}-{lvl_max} &nbsp;|&nbsp; '
            f'{len(scores)} method(s) selected</div>', unsafe_allow_html=True)

    df = build_leaderboard_df(scores, per_run, mm, level_range)

    # KPI cards - exact mockup spec
    gc = df['Grade'].value_counts()
    std_tip = ("Standard deviation measures how spread out the methods' scores are around this "
               "average. A small number means most methods land close to the average (the field is "
               "bunched together); a large number means scores are spread wide apart (some methods "
               "far ahead of or behind the rest).")
    kpis = [
        (f"{df.iloc[0]['Composite Score']:.1f}", "TOP SCORE",  df.iloc[0]['Method'][:22], "--c1", ""),
        (f"{df['Composite Score'].mean():.1f}",  "AVG SCORE",  f"std = {df['Composite Score'].std():.1f}", "--c2", std_tip),
        (str(len(df)),                           "METHODS",    f"{gc.get('A',0)}A  {gc.get('B',0)}B  {gc.get('C',0)}C  {gc.get('D',0)}D", "--c3", ""),
        (f"{df['KTC Score'].max():.4f}",         "BEST KTC",   "higher is better", "--c4", ""),
    ]
    kpi_html = '<div class="kpi-row">'
    for num, lbl, sub, kc, tip in kpis:
        icon = f'<span class="info-tip" data-tip="{html.escape(tip)}">?</span>' if tip else ''
        kpi_html += (f'<div class="kpi" style="--kc:var({kc})">'
                     f'<div class="kpi-n">{num}</div>'
                     f'<div class="kpi-l" style="display:flex;align-items:center">{lbl}{icon}</div>'
                     f'<div class="kpi-s">{sub}</div></div>')
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # "How big is the lead" flash card — the single most tweetable fact on
    # this tab: is 1st place clearly ahead, or is it basically a tie?
    if len(df) >= 2:
        leader, runner_up = df.iloc[0], df.iloc[1]
        gap = leader['Composite Score'] - runner_up['Composite Score']
        if gap < 2:
            gap_story = f"a virtual tie with {runner_up['Method']} ({gap:.1f} pts apart)"
        elif gap < 10:
            gap_story = f"a modest lead over {runner_up['Method']} ({gap:.1f} pts ahead)"
        else:
            gap_story = f"a clear lead over {runner_up['Method']} ({gap:.1f} pts ahead)"

        # Why is the leader winning — is its edge coming from spotting the
        # resistive object, the conductive object, or both about equally?
        # Composite Score itself is KTC-only (see composite_score.py), so a
        # higher score doesn't say which shape drove it — Dice per-class
        # does, since Dice ~1 means that shape was found well and ~0 means
        # it was missed.
        source_html = ""
        res_cols_ok = {'Dice Resistive', 'Dice Conductive'} <= set(df.columns)
        if res_cols_ok:
            res_gap = leader['Dice Resistive'] - runner_up['Dice Resistive']
            con_gap = leader['Dice Conductive'] - runner_up['Dice Conductive']
            if abs(res_gap) < 0.02 and abs(con_gap) < 0.02:
                source = "its lead isn't concentrated in either shape — it's just slightly better all round."
            elif abs(res_gap - con_gap) < 0.02:
                source = "it's ahead on both the resistive and conductive object about equally."
            elif res_gap > con_gap:
                source = (f"the edge comes mainly from the <b>resistive</b> object "
                          f"(Dice Resistive +{res_gap:.3f} vs. +{con_gap:.3f} on conductive).")
            else:
                source = (f"the edge comes mainly from the <b>conductive</b> object "
                          f"(Dice Conductive +{con_gap:.3f} vs. +{res_gap:.3f} on resistive).")
            source_html = f'<br><span style="color:var(--tx2)">Why: {source}</span>'

        st.markdown(
            f'<div style="background:var(--grn-bg);border:1px solid var(--grn-bd);border-radius:7px;'
            f'padding:10px 14px;margin-bottom:14px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:11px;color:var(--tx)">'
            f'<b>{leader["Method"]}</b> is winning with {gap_story}.{source_html}</div>',
            unsafe_allow_html=True,
        )

    # Method chips
    chips_html = '<div class="chips">'
    for name in scores.keys():
        chips_html += f'<span class="chip"><span class="chip-dot" style="background:{get_method_color(name)}"></span>{name}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    # Bar chart
    render_section_header(
        "METHOD RANKINGS - KTC SCORE",
        "Each bar is one method's overall Composite Score (0-100, higher is better). "
        "The label on top of each bar also shows its letter grade. Hover a bar to see "
        "its raw KTC score alongside the scaled composite score.",
    )
    render_grade_key()
    fig = go.Figure()
    for _, row in df.iterrows():
        if not per_run.get(row['Method']):
            render_empty_bar(fig, row['Method'], row['Method'])
            continue
        fig.add_trace(go.Bar(
            name=row['Method'], x=[row['Method']], y=[row['Composite Score']],
            marker_color=get_method_color(row['Method']),
            text=f"{row['Composite Score']:.1f} ({row['Grade']})", textposition='outside',
            textfont=dict(family="JetBrains Mono", size=9, color="#1f2328"),
            hovertemplate=(f"<b>{row['Method']}</b><br>Score: {row['Composite Score']:.1f} ({row['Grade']})<br>"
                           f"KTC: {row['KTC Score']:.4f}<br><extra></extra>")
        ))
    # Let bars extend both above AND below the zero line instead of clipping
    # negative composite scores at a hard floor of 0 — same idea as Excel's
    # default "expenses" chart: positive bars rise, negative bars drop below
    # a shared, clearly-marked zero axis.
    y_vals = df['Composite Score']
    y_min = min(0, y_vals.min() - 15) if not df.empty else -15
    y_max = max(115, y_vals.max() + 15) if not df.empty else 115
    fig.update_layout(
        xaxis_title="Method", yaxis_title="Score (0-100)", yaxis_range=[y_min, y_max],
        showlegend=False, height=380,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f6f8fa',
        font=dict(family="JetBrains Mono,monospace", color="#848d97", size=9),
        xaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickfont=dict(size=9),
                   zeroline=True, zerolinecolor='#57606a', zerolinewidth=1.5),
        margin=dict(l=0, r=10, t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption(
        "X-axis: method under test. Y-axis: Composite Score, a 0-100 rescaling of the raw KTC "
        "score (0 line marks the 'no better than predicting nothing' baseline). Bars can dip "
        "below zero for methods that scored worse than that baseline."
    )

    # Shape-breakdown chart — decomposes each method's performance into how
    # well it found the resistive vs. conductive object. Composite Score
    # itself is KTC-only and doesn't say which shape drove it (see the flash
    # card above), so this is a second, independent chart rather than a
    # re-encoding of the first one — don't compare bar heights across the two.
    if {'Dice Resistive', 'Dice Conductive'} <= set(df.columns):
        render_section_header(
            "SCORE BY SHAPE - RESISTIVE VS CONDUCTIVE",
            "Each bar splits into two stacked segments: how well the method found the "
            "resistive object (bottom) vs. the conductive object (top), each as its Dice "
            "score x100. A short segment means that shape is dragging the method down — "
            "this is what the flash card above is describing, made visual.",
        )
        fig_shape = go.Figure()
        fig_shape.add_trace(go.Bar(
            name="Resistive (Dice x100)", x=df['Method'], y=(df['Dice Resistive'] * 100).round(1),
            marker_color="#CC79A7",
            hovertemplate="<b>%{x}</b><br>Resistive: %{y:.1f}<extra></extra>",
        ))
        fig_shape.add_trace(go.Bar(
            name="Conductive (Dice x100)", x=df['Method'], y=(df['Dice Conductive'] * 100).round(1),
            marker_color="#0072B2",
            hovertemplate="<b>%{x}</b><br>Conductive: %{y:.1f}<extra></extra>",
        ))
        fig_shape.update_layout(
            barmode="stack",
            xaxis_title="Method", yaxis_title="Dice score x100 (stacked)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=9)),
            height=340,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f6f8fa',
            font=dict(family="JetBrains Mono,monospace", color="#848d97", size=9),
            xaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickfont=dict(size=9)),
            yaxis=dict(gridcolor='#d0d7de', linecolor='#d0d7de', tickfont=dict(size=9), range=[0, 210]),
            margin=dict(l=0, r=10, t=20, b=30),
        )
        st.plotly_chart(fig_shape, use_container_width=True, config={'displayModeBar': False})
        st.caption(
            "Stacked height = Dice Resistive x100 + Dice Conductive x100 (a perfect method "
            "tops out at 200). This is a shape-level view of the same per-test results behind "
            "the Composite Score chart above, scaled differently — don't compare heights "
            "directly across the two charts."
        )

    # Table - filter columns by selected metrics in real time
    render_section_header(
        "DETAILED METRICS",
        "The exact numeric value behind every bar and grade above, plus any other metrics "
        "you've selected in the sidebar — useful for double-checking a specific number.",
    )

    # Build full display_df first — round all numeric columns to 4 dp
    display_df = df.copy()
    for _col in display_df.select_dtypes(include='number').columns:
        _dp = 2 if _col == 'Composite Score' else 4
        display_df[_col] = display_df[_col].round(_dp)

    # Always keep Method, Composite Score, Grade; filter the rest
    always_cols = ['Method','Composite Score','Grade']
    optional_cols = [m for m in sel_metrics if m in display_df.columns]
    show_cols = always_cols + optional_cols
    filtered_df = display_df[[c for c in show_cols if c in display_df.columns]]

    def grade_color(grade):
        return {'A':'#dafbe1','B':'#ddf4ff','C':'#fff8c5','D':'#ffebe9'}.get(grade,'')

    pc = st.session_state.get('_pcolors', {})
    row_bg    = pc.get('bg', '#f6f8fa')
    cell_col  = '#1f2328'
    hdr_bg    = pc.get('bg', '#f6f8fa')
    hdr_col   = pc.get('text', '#848d97')
    brd       = '#d0d7de'
    sep       = '#f6f8fa'

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
# VIEW 2 - DEGRADATION  (original logic)
# =========================================================
@st.fragment
def view_degradation_curve(scores:Dict, per_run:Dict, mm:Dict, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    lvl_min, lvl_max = level_range
    dm = all_methods(scores)
    # Default: all methods so the full run is visible immediately.
    chosen = st.multiselect("Select methods:", dm,
        default=[m for m in dm if m not in st.session_state.get('custom_methods',[])])

    if not chosen:
        st.info("Select at least one method.")
        return

    if lvl_min != 1 or lvl_max != 7:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--amb);margin-bottom:6px">'
            f'Showing levels {lvl_min}-{lvl_max}</div>', unsafe_allow_html=True)

    pc  = st.session_state.get('_pcolors', {})
    pb  = pc.get('bg',     '#f6f8fa')
    pp  = pc.get('paper',  'rgba(0,0,0,0)')
    pg  = pc.get('grid',   '#d0d7de')
    pt  = pc.get('text',   '#848d97')
    pleg= pc.get('legend', 'rgba(255,255,255,.9)')

    # One plain-language question this whole chart exists to answer — read
    # this first, then the chart just confirms it visually.
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx2);margin-bottom:8px">'
        'As reconstructions get harder (level 1 = easiest, 7 = hardest), which method keeps its score up?</div>',
        unsafe_allow_html=True)

    render_section_header(
        "SCORE vs. DIFFICULTY LEVEL",
        "Each line is one method's average KTC score at each difficulty level. A flatter, "
        "higher line is better — it means the method stays accurate even as the test gets "
        "harder. A line that drops sharply or zig-zags means that method is unreliable on "
        "harder tests.",
    )
    fig = go.Figure()
    stats = []
    for disp_name in chosen:
        ik = mm.get(disp_name)
        if not ik or ik not in per_run: continue
        samps = filter_by_level(per_run[ik], lvl_min, lvl_max)
        if not samps:
            continue

        # x = difficulty level; y = mean KTC over that level's samples.
        # No confidence band / std-dev dashed line / per-sample dots here —
        # those are three extra statistical concepts (variance, mean-of-
        # means, raw scatter) layered on top of the one line a first-time
        # viewer actually needs to read. One clean line per method only.
        levels   = sorted({int(e['level']) for e in samps.values()})
        by_level = {lv: [e['ktc_score'] for e in samps.values() if int(e['level']) == lv]
                    for lv in levels}
        ktc = [float(np.mean(by_level[lv])) for lv in levels]
        sds = [float(np.std(by_level[lv]))  for lv in levels]
        x   = levels
        c   = get_method_color(disp_name)
        mu  = float(np.mean(ktc)); sd = float(np.std(ktc))
        label = method_display_name(disp_name)

        fig.add_trace(go.Scatter(x=x, y=ktc, mode='lines+markers', name=label,
            line=dict(width=3, color=c),
            marker=dict(size=8, color=c, line=dict(width=2, color='#ffffff')),
            hovertemplate=f"<b>{label}</b><br>Level: %{{x}}<br>Score: %{{y:.2f}}<extra></extra>"))

        # Failure rate: fraction of samples scoring below the D-grade cutoff
        # (raw KTC < 0.10, same threshold letter_grade() uses for the whole
        # dashboard) — split into the easier vs. harder half of the levels
        # actually shown, so "gets worse with difficulty" is a real percentage
        # rather than just a lower average.
        FAIL_THRESHOLD = 0.10
        half = max(1, len(levels) // 2)
        easy_levels, hard_levels = levels[:half], levels[half:] or levels[-1:]
        easy_samples = [v for lv in easy_levels for v in by_level[lv]]
        hard_samples = [v for lv in hard_levels for v in by_level[lv]]
        easy_fail_pct = 100.0 * sum(v < FAIL_THRESHOLD for v in easy_samples) / len(easy_samples) if easy_samples else 0.0
        hard_fail_pct = 100.0 * sum(v < FAIL_THRESHOLD for v in hard_samples) / len(hard_samples) if hard_samples else 0.0

        stats.append({'Method':disp_name,'Mean KTC':mu,'Std Dev':sd,
                      'Min':np.min(ktc),'Max':np.max(ktc),'Range':np.max(ktc)-np.min(ktc),
                      'Fail % (Easy Levels)':easy_fail_pct,'Fail % (Hard Levels)':hard_fail_pct})

    fig.update_layout(
        xaxis_title="Difficulty Level (1 = easiest, 7 = hardest)",
        yaxis_title="Score (higher = better)",
        height=420, hovermode='x unified',
        paper_bgcolor=pp, plot_bgcolor=pb,
        font=dict(family="JetBrains Mono,monospace", color=pt, size=9),
        xaxis=dict(gridcolor=pg, linecolor=pg),
        yaxis=dict(gridcolor=pg, linecolor=pg),
        legend=dict(bgcolor=pleg, bordercolor=pg, borderwidth=1, font=dict(size=9)),
        margin=dict(l=0, r=0, t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.caption(
        "X-axis: difficulty level. Y-axis: mean KTC score across that level's samples. "
        "In the table below, Std Dev is the standard deviation of a method's own level-by-level "
        "scores — it measures consistency, not accuracy: a low Std Dev means the method performs "
        "about the same at every difficulty level (predictable), while a high Std Dev means its "
        "score swings a lot between levels (inconsistent), even if its average is decent."
    )

    if stats:
        best  = max(stats, key=lambda r: r['Mean KTC'])
        worst = max(stats, key=lambda r: r['Std Dev'])

        # The headline statistical statement: how much more often does each
        # method outright fail (D-grade territory) on the harder half of
        # levels vs. the easier half? Reported as a real percentage, not
        # just "the average goes down".
        biggest_degrader = max(stats, key=lambda r: r['Fail % (Hard Levels)'] - r['Fail % (Easy Levels)'])
        deg_gap = biggest_degrader['Fail % (Hard Levels)'] - biggest_degrader['Fail % (Easy Levels)']
        avg_easy_fail = float(np.mean([r['Fail % (Easy Levels)'] for r in stats]))
        avg_hard_fail = float(np.mean([r['Fail % (Hard Levels)'] for r in stats]))

        st.markdown(
            f'<div style="background:var(--warn-bg);border:1px solid var(--warn-bd);'
            f'border-radius:7px;padding:10px 14px;margin-top:8px;margin-bottom:8px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:600;color:var(--amb);'
            f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">The Statistic</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:var(--tx);line-height:1.6">'
            f'- On the harder half of levels shown, methods score below the D-grade cutoff (KTC &lt; 0.10) '
            f'on <b>{avg_hard_fail:.0f}%</b> of tests on average, vs. <b>{avg_easy_fail:.0f}%</b> on the easier half.<br>'
            f'- <b>{method_display_name(biggest_degrader["Method"])}</b> degrades the most: it fails on '
            f'<b>{biggest_degrader["Fail % (Easy Levels)"]:.0f}%</b> of the easiest tests but '
            f'<b>{biggest_degrader["Fail % (Hard Levels)"]:.0f}%</b> of the hardest ones '
            f'(+{deg_gap:.0f} points as difficulty rises).'
            f'</div></div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:var(--grn-bg);border:1px solid var(--grn-bd);border-radius:7px;padding:10px 14px;margin-top:8px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:600;color:var(--grn);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">In Plain Terms</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:var(--tx);line-height:1.6">'
            f'- <b>{method_display_name(best["Method"])}</b> scores highest on average — the strongest choice overall.<br>'
            f'- <b>{method_display_name(worst["Method"])}</b> jumps around the most from level to level — its results are the least predictable.'
            f'</div></div>',
            unsafe_allow_html=True)

        with st.expander("Full numbers per method", expanded=False):
            st.caption(
                "Fail % columns: share of that method's samples scoring below the D-grade cutoff "
                "(KTC < 0.10) on the easier vs. harder half of the levels currently shown."
            )
            st.dataframe(pd.DataFrame(stats).round(4), use_container_width=True, hide_index=True)

# =========================================================
# VIEW 3 - COMPARISON  (original logic)
# =========================================================
@st.fragment
def view_comparison(scores:Dict, per_run:Dict, mm:Dict, sel_metrics:list=None, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx2);margin-bottom:8px">'
        'Pick two methods and one test — see exactly where they agree or disagree, '
        'and what each one actually reconstructed.</div>',
        unsafe_allow_html=True)

    # Map sidebar metric display names to internal per_run keys
    # Which internal keys to show - driven by sel_metrics
    if sel_metrics is not None:
        show_keys = [METRIC_LABEL_TO_KEY[sm] for sm in sel_metrics if sm in METRIC_LABEL_TO_KEY]
    else:
        show_keys = [key for _, key in METRIC_SPECS]

    lvl_min, lvl_max = level_range

    dm = all_methods(scores)
    fi = list(per_run.keys())[0] if per_run else None
    entries = filter_by_level(per_run[fi], lvl_min, lvl_max) if fi else {}
    if not entries and fi:
        entries = per_run[fi]  # fallback: show all if filter leaves nothing
    levels_avail = sorted({int(e['level']) for e in entries.values()}) or ALL_LEVELS

    # Explicit keys pin each widget to a stable identity across reruns —
    # without them, Streamlit derives an id from label+options+index, so if
    # the options list (dm/levels_avail/samps) ever reorders between runs,
    # the widget can be treated as "new" and silently reset to its default
    # index, which would show a stale method/level pairing below.
    c1,c2,c3,c4 = st.columns(4)
    m1  = c1.selectbox("Method 1:", dm, index=0, key="cmp_m1")
    m2  = c2.selectbox("Method 2:", dm, index=min(1,len(dm)-1), key="cmp_m2")
    lvl = c3.selectbox("Level:", levels_avail, index=0, key="cmp_lvl")
    samps = sorted({e['sample'] for e in entries.values() if int(e['level']) == lvl})
    sid = c4.selectbox("Sample:", samps, key="cmp_sample") if samps else None

    m1i = mm.get(m1)
    m2i = mm.get(m2)
    if m1 == m2 or (m1i is not None and m1i == m2i):
        st.error("Same method cannot be compared. Please choose two different methods.")
        return

    run_key = f"L{lvl}_{sid}"

    p1  = m1 in st.session_state.get('custom_methods', [])
    p2  = m2 in st.session_state.get('custom_methods', [])
    if p1: st.info(f"{m1} is a custom method - connect its backend to see metrics.")
    if p2: st.info(f"{m2} is a custom method - connect its backend to see metrics.")

    met1 = per_run.get(m1i or m1, {}).get(run_key, {}) if not p1 else {}
    met2 = per_run.get(m2i or m2, {}).get(run_key, {}) if not p2 else {}

    # Metric comparison table - only show selected metrics
    render_section_header(
        "METRIC COMPARISON",
        "Side-by-side scores for the two chosen methods on this exact test (same level, same "
        "sample). 'Diff' is the absolute difference between them — the larger it is, the more "
        "the two methods disagree on that particular metric.",
    )
    keys_to_show = [k for k in show_keys if k in met1 or k in met2]
    if not keys_to_show:
        st.info("No selected metrics are available for this sample.")
        return
    comp_data = [{'Metric': k.replace('_',' ').title(), 'MetricKey': k,
                  m1: met1.get(k,0), m2: met2.get(k,0),
                  'Diff': abs(met1.get(k,0)-met2.get(k,0))}
                 for k in keys_to_show]
    comp_df = pd.DataFrame(comp_data)
    if not comp_df.empty:
        biggest = comp_df.loc[comp_df['Diff'].idxmax()]
        mkey = biggest['MetricKey']
        v1, v2 = biggest[m1], biggest[m2]
        winner, loser = (m1, m2) if v1 > v2 else (m2, m1)
        gap = biggest['Diff']

        # All of these metrics are shape-overlap scores on [0, 1] (ktc_score
        # can dip slightly negative below the water baseline) where higher =
        # better overlap with the true object — so the gap translates
        # directly into "how much better did the winner actually see this
        # object", not just an abstract number.
        if mkey == 'ktc_score':
            subject = "the overall reconstruction"
        else:
            cls = "resistive" if "resistive" in mkey else "conductive" if "conductive" in mkey else "shape"
            subject = f"the {cls} object"
        if gap >= 0.7:
            meaning = (f"That's a night-and-day difference: {method_display_name(winner)} nearly nailed "
                       f"{subject}, while {method_display_name(loser)} almost completely missed it.")
        elif gap >= 0.3:
            meaning = (f"That's a substantial difference: {method_display_name(winner)} captured "
                       f"{subject} noticeably better than {method_display_name(loser)}.")
        elif gap >= 0.1:
            meaning = (f"That's a modest difference — {method_display_name(winner)} edges out "
                       f"{method_display_name(loser)} here, but both are in the same ballpark.")
        else:
            meaning = "That's a small enough gap that the two methods are effectively tied on this metric."

        st.markdown(
            f'<div style="background:var(--grn-bg);border:1px solid var(--grn-bd);border-radius:7px;'
            f'padding:10px 14px;margin-bottom:14px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:11px;color:var(--tx)">'
            f'<b>{m1}</b> and <b>{m2}</b> differ most on <b>{biggest["Metric"]}</b> — '
            f'a gap of {gap:.4f}.<br>'
            f'<span style="color:var(--tx2)">{meaning}</span></div>',
            unsafe_allow_html=True,
        )
    for col in [m1, m2, 'Diff']:
        if col in comp_df.columns:
            comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}")
    comp_df = comp_df.drop(columns=['MetricKey'], errors='ignore')
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Overall verdict — the metric table above only judges these two methods
    # on ONE test. This section answers the two questions that single cell
    # can't: who's actually better across every level/sample in view, and
    # does one of them fall apart faster than the other as the level
    # increases (fewer electrodes = less information = harder problem)?
    if not p1 and not p2:
        render_section_header(
            "OVERALL VERDICT",
            "Zooms out from the single test above to every level/sample currently in view: "
            "which method wins on average, and — as the level increases (fewer electrodes, "
            "less information, harder problem) — which one's score falls off faster. Slope is "
            "KTC-score change per level (negative = degrading); r measures how consistent that "
            "trend is (near -1 = degrades steadily, near 0 = too noisy to call a real trend).",
        )
        entries1_f = filter_by_level(per_run.get(m1i or m1, {}), lvl_min, lvl_max)
        entries2_f = filter_by_level(per_run.get(m2i or m2, {}), lvl_min, lvl_max)
        pts1 = [(int(e['level']), float(e['ktc_score'])) for e in entries1_f.values() if 'ktc_score' in e]
        pts2 = [(int(e['level']), float(e['ktc_score'])) for e in entries2_f.values() if 'ktc_score' in e]

        def _degradation_trend(pts):
            """(slope, r) of KTC score vs level via least-squares fit, or
            (None, None) if there aren't at least two distinct levels to fit."""
            if len(pts) < 2:
                return None, None
            levels = np.array([p[0] for p in pts], dtype=float)
            vals = np.array([p[1] for p in pts], dtype=float)
            if np.std(levels) == 0:
                return None, None
            slope, _ = np.polyfit(levels, vals, 1)
            r = float(np.corrcoef(levels, vals)[0, 1]) if np.std(vals) > 0 else 0.0
            return float(slope), r

        if pts1 and pts2:
            avg1 = float(np.mean([s for _, s in pts1]))
            avg2 = float(np.mean([s for _, s in pts2]))
            overall_winner = m1 if avg1 > avg2 else m2
            overall_loser = m2 if avg1 > avg2 else m1
            overall_gap = abs(avg1 - avg2)
            single_test_winner = winner if not comp_df.empty else None

            slope1, r1 = _degradation_trend(pts1)
            slope2, r2 = _degradation_trend(pts2)

            contradiction_html = ""
            if single_test_winner and single_test_winner != overall_winner:
                contradiction_html = (
                    f'<br><span style="color:var(--tx2)"><b>{method_display_name(single_test_winner)}</b> '
                    f'won on this specific test, but that doesn\'t hold up across the full picture — '
                    f'<b>{method_display_name(overall_winner)}</b> is the stronger method overall.</span>'
                )

            degrade_html = ""
            if slope1 is not None and slope2 is not None:
                steeper = m1 if slope1 < slope2 else m2
                shallower = m2 if slope1 < slope2 else m1
                steeper_slope = slope1 if slope1 < slope2 else slope2
                if slope1 < -0.001 or slope2 < -0.001:
                    degrade_html = (
                        f'<br><span style="color:var(--tx2)"><b>{method_display_name(steeper)}</b> degrades '
                        f'faster as the level increases ({steeper_slope:+.4f} KTC pts/level) than '
                        f'<b>{method_display_name(shallower)}</b> — so {method_display_name(steeper)}\'s '
                        f'lead, if any, shouldn\'t be trusted to hold up on the hardest levels.</span>'
                    )
                else:
                    degrade_html = (
                        '<br><span style="color:var(--tx2)">Neither method shows a meaningful '
                        'degradation trend across the levels in view — both hold up about as well '
                        'on the hard levels as the easy ones.</span>'
                    )

            st.markdown(
                f'<div style="background:var(--grn-bg);border:1px solid var(--grn-bd);border-radius:7px;'
                f'padding:10px 14px;margin-bottom:14px;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:11px;color:var(--tx)">'
                f'Across levels {lvl_min}-{lvl_max}, <b>{method_display_name(overall_winner)}</b> averages a '
                f'higher KTC score than <b>{method_display_name(overall_loser)}</b> '
                f'({avg1 if overall_winner == m1 else avg2:.4f} vs '
                f'{avg2 if overall_winner == m1 else avg1:.4f}, a gap of {overall_gap:.4f}).'
                f'{contradiction_html}{degrade_html}</div>',
                unsafe_allow_html=True,
            )

            trend_rows = []
            for name, pts, slope, r in ((m1, pts1, slope1, r1), (m2, pts2, slope2, r2)):
                trend_rows.append({
                    'Method': name,
                    'Avg KTC Score': round(float(np.mean([s for _, s in pts])), 4),
                    'Degradation Slope (pts/level)': round(slope, 4) if slope is not None else "n/a — needs 2+ levels",
                    'Trend Strength (r)': round(r, 3) if r is not None else "n/a",
                    'Levels Compared': len({lv for lv, _ in pts}),
                })
            st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)
        else:
            st.info(
                "Not enough data across the current level filter to compute an overall verdict "
                "for one or both methods."
            )

    # Visual comparison - images from backend
    render_section_header(
        "VISUAL COMPARISON",
        "The actual reconstructed image each method produced for this test, next to the ground "
        "truth panel above — so you can see with your own eyes what the numbers above are "
        "describing.",
    )
    panel = load_comparison_panel(sid)
    if panel:
        st.markdown(f"All Methods - Sample {sid}")
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
# VIEW 4 - FAILURE GALLERY  (original logic)
# =========================================================
def view_failure_gallery(scores:Dict, per_run:Dict, mm:Dict, level_range:tuple=(1,7)):
    if not per_run:
        st.warning("No per-run metrics available.")
        return

    lvl_min, lvl_max = level_range

    if lvl_min != 1 or lvl_max != 7:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--amb);margin-bottom:6px">'
            f'Filtered: levels {lvl_min}-{lvl_max}</div>', unsafe_allow_html=True)

    # -- Pass 1: collect all per-run data for KPI counts, root cause, summary -
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

    # -- KPI overview cards ------------------------------------
    total    = len(all_ktc)
    kpi2     = [
        (str(total),             "TOTAL RUNS",         "all methods x samples", "--c3"),
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

    # -- Failure summary table (one row per method) ------------
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
                'color:var(--amb);background:var(--warn-bg);border:1px solid var(--warn-bd);border-radius:5px;'
                'padding:5px 10px;margin-top:4px">'
                'KTC < 0 means the reconstruction is <b>worse than the all-water baseline</b> '
                '(random noise artefacts confuse the SSIM metric). '
                'Composite scores below zero are expected for poorly-initialised solvers.</div>',
                unsafe_allow_html=True)

    # -- Root cause analysis table -----------------------------
    if root_causes:
        st.markdown('<div class="slbl">ROOT CAUSE ANALYSIS - FAILING RUNS (C + D GRADE)</div>', unsafe_allow_html=True)
        rc_df = pd.DataFrame(root_causes)

        def _rc_color(grade):
            return '#ffebe9' if grade == 'D' else '#fff8c5'

        rows_html2 = ''
        for _, row in rc_df.iterrows():
            bg = _rc_color(row['Grade'])
            rows_html2 += f'<tr style="background:{bg}">'
            for col in rc_df.columns:
                rows_html2 += (f'<td style="font-family:\'JetBrains Mono\',monospace;'
                               f'font-size:9px;padding:5px 8px;border-bottom:1px solid var(--bd);'
                               f'color:var(--tx)">{row[col]}</td>')
            rows_html2 += '</tr>'
        hdrs2 = ''.join(
            f'<th style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
            f'text-transform:uppercase;letter-spacing:.1em;color:var(--tx3);'
            f'padding:4px 8px;border-bottom:1px solid var(--bd);background:var(--bg)">{c}</th>'
            for c in rc_df.columns)
        st.markdown(
            f'<div style="border:1px solid var(--bd);border-radius:7px;overflow:hidden">'
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

    # -- Per-method worst-3 cards (lowest KTC = worst, level-filtered) -
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
                  <div class="frank">#{idx+1} WORST | SAMPLE {sid}</div>
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
# VIEW 5 - RADAR CHART  (original logic)
# =========================================================
@st.fragment
def view_radar_chart(scores:Dict, per_run:Dict, sel_metrics:list=None):
    if not scores:
        st.warning("No scores available.")
        return
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx2);margin-bottom:8px">'
        'A single score can hide the full picture — this compares methods across several metrics at once. '
        'A bigger, more even shape means a more well-rounded method; a spiky shape means it\'s great at some '
        'things and weak at others.</div>',
        unsafe_allow_html=True)
    scores = {
        method: {
            (METRIC_LABEL_TO_KEY.get(key, key)): value
            for key, value in metrics.items()
        }
        for method, metrics in scores.items()
    }
    # Map sidebar display names -> scores.json key names
    avail_score_keys = sorted({k for m in scores.values() for k in m.keys()})

    # If sel_metrics provided, use those as default (filtered to what actually exists)
    if sel_metrics is not None:
        default_keys = []
        for sm in sel_metrics:
            candidate = METRIC_LABEL_TO_KEY.get(sm)
            if candidate in avail_score_keys:
                default_keys.append(candidate)
        if not default_keys and sel_metrics:
            default_keys = avail_score_keys[:min(7,len(avail_score_keys))]
    else:
        RADAR_PREFERRED = [key for _, key in METRIC_SPECS]
        default_keys = [m for m in RADAR_PREFERRED if m in avail_score_keys]
        if not default_keys:
            default_keys = avail_score_keys[:min(7,len(avail_score_keys))]

    chosen = st.multiselect("Choose metrics (7-axis):", avail_score_keys,
        default=default_keys)
    if not chosen:
        st.info("Select at least one metric.")
        return

    # Radar is only meaningful with 2+ axes. With 1 metric it degrades to dots on a line.
    if len(chosen) == 1:
        st.info(
            f"Radar chart needs 2+ metrics to draw a polygon - currently only **{chosen[0]}** is available. "
            "Showing a bar comparison instead. Add more metrics (Dice, IoU...) to unlock the full radar."
        )
        metric = chosen[0]
        pc2 = st.session_state.get('_pcolors', {})
        bar_data = [(name, max(0, metrics.get(metric, 0))) for name, metrics in scores.items()]
        bar_data.sort(key=lambda x: x[1], reverse=True)
        fig2 = go.Figure()
        for name, val in bar_data:
            fig2.add_trace(go.Bar(
                name=name, x=[name], y=[val],
                marker_color=get_method_color(name),
                text=f"{val:.4f}", textposition='outside',
                textfont=dict(family="JetBrains Mono", size=9),
                hovertemplate=f"<b>{method_display_name(name)}</b><br>{metric.replace('_',' ').title()}: %{{y:.4f}}<extra></extra>",
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
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        return

    # "Most well-rounded" flash card — the shape's overall size (mean across
    # every chosen metric) as a single, plain-language answer to "who's
    # generally best across the board, not just on one metric".
    method_avgs = {
        name: float(np.mean([max(0, metrics.get(m, 0)) for m in chosen]))
        for name, metrics in scores.items()
    }
    if method_avgs:
        best_rounded = max(method_avgs, key=method_avgs.get)
        st.markdown(
            f'<div style="background:var(--grn-bg);border:1px solid var(--grn-bd);border-radius:7px;'
            f'padding:10px 14px;margin-bottom:14px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:11px;color:var(--tx)">'
            f'<b>{method_display_name(best_rounded)}</b> is the most well-rounded — highest average '
            f'across all {len(chosen)} selected metrics ({method_avgs[best_rounded]:.2f}).</div>',
            unsafe_allow_html=True,
        )

    render_section_header(
        "METHOD PERFORMANCE ACROSS SELECTED METRICS",
        "Each colored shape is one method. Each spoke is one metric, scaled 0-1 (further from "
        "the center = better). A large, rounded shape means the method scores well everywhere; "
        "a shape that pokes out on one spoke but pinches in on others means the method trades "
        "off strength in one area for weakness in another.",
    )
    fig = go.Figure()
    for name, metrics in scores.items():
        # KTC is already higher = better on [0, 1]; clamp negatives (worse than
        # all-water baseline) to 0 for the polar axis.
        vals = [max(0, metrics.get(m,0)) for m in chosen]
        vals.append(vals[0])
        cats = [m.replace('_',' ').title() for m in chosen]; cats.append(cats[0])
        label = method_display_name(name)
        c = get_method_color(name)
        fig.add_trace(go.Scatterpolar(r=vals,theta=cats,fill='toself',name=name,
            line_color=c,fillcolor=hex_to_rgba(c, 0.13),
            hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>"))
    pc = st.session_state.get('_pcolors',{})
    fig.update_layout(
        polar=dict(bgcolor=pc.get('bg','#f6f8fa'),
            radialaxis=dict(visible=True,range=[0,1],gridcolor=pc.get('grid','#d0d7de'),linecolor=pc.get('grid','#d0d7de'),tickfont=dict(size=8,color=pc.get('text','#848d97'))),
            angularaxis=dict(gridcolor=pc.get('grid','#d0d7de'),linecolor=pc.get('grid','#d0d7de'),tickfont=dict(size=10,color=pc.get('text','#848d97')))),
        showlegend=True,height=560,
        paper_bgcolor=pc.get('paper','rgba(0,0,0,0)'),
        font=dict(family="JetBrains Mono,monospace",color=pc.get('text','#848d97'),size=9),
        legend=dict(bgcolor=pc.get('legend','rgba(255,255,255,.9)'),bordercolor=pc.get('grid','#d0d7de'),borderwidth=1,font=dict(size=9)),
        margin=dict(l=55,r=55,t=45,b=55))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    render_section_header(
        "METRIC STATISTICS",
        "For each metric, how all methods scored on it. Std Dev here is the standard deviation "
        "across methods (not across tests) — a low Std Dev means every method scored about the "
        "same on that metric (it doesn't separate the methods much); a high Std Dev means that "
        "metric is where methods differ the most, so it's a good one to weigh heavily when "
        "choosing between them.",
    )
    rows = []
    for m in chosen:
        vals = [max(0, ms.get(m,0)) for ms in scores.values()]
        rows.append({'Metric':m.replace('_',' ').title(),'Mean':np.mean(vals),'Std Dev':np.std(vals),'Min':np.min(vals),'Max':np.max(vals)})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)

# =========================================================
# VIEW HEATMAP - COLOR GRID (all 42 runs at once)
# =========================================================
@st.fragment
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
        st.info(f"No samples found for levels {lvl_min}-{lvl_max}.")
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
    sel_metrics_sidebar = st.session_state.get('selected_metrics', [])
    selected_metric_keys = [
        METRIC_LABEL_TO_KEY[m] for m in sel_metrics_sidebar
        if m in METRIC_LABEL_TO_KEY and METRIC_LABEL_TO_KEY[m] in metric_opts
    ]
    metric_opts = selected_metric_keys
    # Runtime isn't a score, so it doesn't belong in METRIC_SPECS / the
    # sidebar's Metrics checklist — but "which level/sample was slow" is
    # exactly what this grid is good at answering, so always offer it here,
    # independent of whatever's checked in the sidebar.
    if 'runtime_ms' in sample_metrics and 'runtime_ms' not in metric_opts:
        metric_opts.append('runtime_ms')
    if not metric_opts:
        st.info("Select at least one metric in the sidebar to draw the heatmap.")
        return
    # Map sidebar selected_metrics display names -> internal keys for default selection
    default_hm_metric = 'ktc_score'
    for sm in sel_metrics_sidebar:
        candidate = METRIC_LABEL_TO_KEY.get(sm)
        if candidate and candidate in metric_opts:
            default_hm_metric = candidate
            break
    default_hm_idx = metric_opts.index(default_hm_metric) if default_hm_metric in metric_opts else 0

    # Metrics where a smaller number is the better outcome (currently just
    # runtime) need every direction-sensitive comparison below flipped —
    # color scale, best/worst level, and cell-by-cell "win" counting all
    # otherwise assume higher-is-better like the accuracy scores do.
    LOWER_IS_BETTER = {'runtime_ms'}

    def _hm_metric_label(key: str) -> str:
        return "Runtime (s)" if key == "runtime_ms" else METRIC_KEY_TO_LABEL.get(key, key)

    _true_runtime = true_first_run_runtime_ms() if 'runtime_ms' in metric_opts else {}

    def _metric_value(entry: dict, ik: str = None, sid: str = None):
        v = entry.get(chosen_metric)
        if v is None:
            return None
        if chosen_metric == "runtime_ms":
            # This exact cell may be cached now (near-zero measured time) —
            # never report less than the best-ever observed true cost for
            # this same (method, sample), recovered from earlier runs.
            if ik is not None and sid is not None:
                true_v = _true_runtime.get((ik, sid))
                if true_v is not None:
                    v = max(v, true_v)
            return v / 1000.0
        return v

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx2);margin-bottom:8px">'
        'Every method vs. every single test, side by side — green means it did well there, red means it struggled.</div>',
        unsafe_allow_html=True)

    mc1, mc2 = st.columns([2, 4])
    with mc1:
        chosen_metric = st.selectbox("Metric:", metric_opts, index=default_hm_idx, key="hm_metric",
                                     format_func=_hm_metric_label)
    lower_is_better = chosen_metric in LOWER_IS_BETTER

    def _is_better(a, b) -> bool:
        return (a < b) if lower_is_better else (a > b)

    def _is_worse(a, b) -> bool:
        return (a > b) if lower_is_better else (a < b)

    with mc2:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;color:var(--tx3);padding-top:28px">'
            f'Levels {lvl_min}-{lvl_max} | {len(all_sample_ids)} samples | {len(method_names)} methods. '
            f'{"Lower" if lower_is_better else "Higher"} = greener (better).</div>',
            unsafe_allow_html=True)

    pc = st.session_state.get('_pcolors', {})
    pb = pc.get('bg', '#f6f8fa')
    pg = pc.get('grid', '#d0d7de')
    pt = pc.get('text', '#848d97')

    # -- Per-method stats for the chosen metric, computed once and reused
    # for sort order and the summary table below.
    stats_by_method: dict[str, dict] = {}
    for disp in method_names:
        ik = mm.get(disp)
        entries = per_run.get(ik, {}) if ik else {}
        per_sample_vals: list[float] = []
        best_level = worst_level = None
        best_val = worst_val = None
        for sid in all_sample_ids:
            e = entries.get(sid)
            val = _metric_value(e, ik, sid) if e else None
            if val is None:
                continue
            lvl = key_order[sid][0]
            per_sample_vals.append(val)
            if best_val is None or _is_better(val, best_val):
                best_val, best_level = val, lvl
            if worst_val is None or _is_worse(val, worst_val):
                worst_val, worst_level = val, lvl
        if not per_sample_vals:
            continue
        stats_by_method[disp] = {
            'mean': float(np.mean(per_sample_vals)),
            'std': float(np.std(per_sample_vals)),
            'min': float(np.min(per_sample_vals)),
            'max': float(np.max(per_sample_vals)),
            'runs': len(per_sample_vals),
            'best_level': best_level,
            'worst_level': worst_level,
        }

    if not stats_by_method:
        st.info("No data available for the selected metric.")
        return

    # Rows sorted by mean of the selected metric, best method at the top.
    method_names = sorted(stats_by_method, key=lambda m: stats_by_method[m]['mean'], reverse=True)

    # Flash cards: two facts a reader would otherwise have to hunt for
    # across 21 cells per method — most dependable method (lowest std,
    # i.e. score barely changes test to test) and the single hardest test
    # overall (lowest score anywhere in the grid, across every method).
    most_consistent = min(stats_by_method, key=lambda m: stats_by_method[m]['std'])
    worst_cell_method, worst_cell_sid, worst_cell_val = None, None, None
    for disp in method_names:
        ik = mm.get(disp)
        for sid in all_sample_ids:
            e = per_run.get(ik, {}).get(sid)
            val = _metric_value(e, ik, sid) if e else None
            if val is not None and (worst_cell_val is None or _is_worse(val, worst_cell_val)):
                worst_cell_val, worst_cell_method, worst_cell_sid = val, disp, sid
    dependable_tip = ("Ranked by standard deviation of this method's scores across every test in the "
                       "grid — the lower the standard deviation, the less that method's score moves "
                       "from test to test, meaning its performance is predictable rather than a "
                       "coin flip.")
    card_html = '<div class="kpi-row">'
    card_html += (
        f'<div class="kpi" style="--kc:var(--c2)">'
        f'<div class="kpi-n">{method_display_name(most_consistent)}</div>'
        f'<div class="kpi-l" style="display:flex;align-items:center">Most Dependable'
        f'<span class="info-tip" data-tip="{html.escape(dependable_tip)}">?</span></div>'
        f'<div class="kpi-s">score barely changes test to test</div></div>'
    )
    if worst_cell_method is not None:
        card_html += (
            f'<div class="kpi" style="--kc:var(--c5)">'
            f'<div class="kpi-n">{worst_cell_sid}</div>'
            f'<div class="kpi-l">Toughest Single Test</div>'
            f'<div class="kpi-s">{method_display_name(worst_cell_method)} scored {worst_cell_val:.2f} here, its worst</div></div>'
        )
    card_html += '</div>'
    st.markdown(card_html, unsafe_allow_html=True)

    # THE STATISTIC — same headline-number treatment as the Degradation tab:
    # not just "who has the best average" (already shown above) but "who
    # actually wins outright, cell by cell, head-to-head against whoever
    # else was tested on that exact sample".
    win_counts = {m: 0 for m in method_names}
    contested_cells = 0
    for sid in all_sample_ids:
        cell_vals = {}
        for disp in method_names:
            ik = mm.get(disp)
            e = per_run.get(ik, {}).get(sid) if ik else None
            v = _metric_value(e, ik, sid) if e else None
            if v is not None:
                cell_vals[disp] = v
        if len(cell_vals) >= 2:
            contested_cells += 1
            picker = min if lower_is_better else max
            win_counts[picker(cell_vals, key=cell_vals.get)] += 1

    if contested_cells and any(win_counts.values()):
        top_winner = max(win_counts, key=win_counts.get)
        win_pct = 100.0 * win_counts[top_winner] / contested_cells
        fair_share_pct = 100.0 / len(method_names)
        st.markdown(
            f'<div style="background:var(--warn-bg);border:1px solid var(--warn-bd);'
            f'border-radius:7px;padding:10px 14px;margin-bottom:14px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:600;'
            f'color:var(--amb);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">'
            f'The Statistic</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:var(--tx);line-height:1.6">'
            f'- <b>{method_display_name(top_winner)}</b> has the single best score, head-to-head, on '
            f'<b>{win_counts[top_winner]} of {contested_cells}</b> tests (<b>{win_pct:.0f}%</b>) — more '
            f'than any other method.<br>'
            f'- With {len(method_names)} methods competing, pure luck would win each one only about '
            f'1 test in {len(method_names)} (~{fair_share_pct:.0f}%). '
            f'{method_display_name(top_winner)} is winning {win_pct:.0f}% of the time — '
            f'far more often than chance alone would explain, so this is a real, consistent lead, not a fluke.'
            f'</div></div>',
            unsafe_allow_html=True)

    render_section_header(
        "SCORE GRID",
        "One row per method, one column per test (grouped into level blocks). Cell color follows "
        "the scale on the right: green = a strong score on that test, red = a poor one. Hover any "
        "cell for its exact value. The Mean/Std columns on the far right summarize each row: Mean "
        "is that method's average score across all tests shown; Std is the standard deviation of "
        "its scores across those tests — low Std means the row's colors stay similar shade to "
        "shade (consistent), high Std means the row swings between very green and very red cells "
        "(inconsistent).",
    )
    z, text, y_labels = [], [], []
    val_fmt = "{:.2f}s" if chosen_metric == "runtime_ms" else "{:.4f}"
    for disp in method_names:
        ik = mm.get(disp)
        entries = per_run.get(ik, {}) if ik else {}
        row, trow = [], []
        for sid in all_sample_ids:
            e = entries.get(sid)
            val = _metric_value(e, ik, sid) if e else None
            row.append(val if val is not None else float('nan'))
            trow.append(val_fmt.format(val) if val is not None else "-")
        z.append(row)
        text.append(trow)
        y_labels.append(disp)

    # RdYlGn is green-at-high-value by default (matches ktc/dice/iou, where
    # higher is better). Metrics in LOWER_IS_BETTER need the scale flipped
    # via reversescale rather than z-negation, so hover/text values stay the
    # real, non-negated numbers. Plotly auto-scales to the actual z min/max
    # here (no zmin/zmax set), so the colors are always relative to what's
    # actually on screen — a fast method's row won't read as "all red" just
    # because a slow outlier exists elsewhere in the grid.
    colorscale = 'RdYlGn'

    # When columns exceed 12 the cell text overlaps - rely on hover tooltips instead
    show_cell_text = len(all_sample_ids) <= 12

    # KTC score can go negative (worse than the water baseline); dice/iou
    # are bounded [0, 1] and never negative, so the baseline note below is
    # only shown when it's actually meaningful for the selected metric.
    # Tick labels are kept short/numeric (not "0.0 (baseline)" etc.) because
    # their rendered width is unpredictable and previously collided with
    # the Mean/Std columns placed just to the right of the colorbar.
    colorbar_kwargs = dict(
        thickness=10, len=0.9, x=1.02, xpad=4,
        tickfont=dict(family="JetBrains Mono,monospace", size=8, color="#848d97"),
        outlinecolor="#d0d7de", outlinewidth=1,
    )

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(s) for s in all_sample_ids],
        y=y_labels,
        text=text,
        texttemplate="%{text}" if show_cell_text else "",
        textfont=dict(family="JetBrains Mono,monospace", size=8),
        colorscale=colorscale,
        reversescale=lower_is_better,
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Sample: %{x}<br>Value: %{text}<extra></extra>",
        colorbar=colorbar_kwargs,
    ))

    # White/background gaps between level groups so L1 | L2 | ... | L7 read
    # as visually separate blocks instead of one continuous 21-column strip.
    level_seq = [key_order[sid][0] for sid in all_sample_ids]
    for i in range(1, len(level_seq)):
        if level_seq[i] != level_seq[i - 1]:
            fig.add_vline(x=i - 0.5, line_width=3, line_color=pb)

    fig.update_layout(
        height=max(220, len(method_names)*54 + 80),
        paper_bgcolor=pc.get('paper','rgba(0,0,0,0)'),
        plot_bgcolor=pb,
        font=dict(family="JetBrains Mono,monospace", color=pt, size=9),
        xaxis=dict(side='top', gridcolor=pb, linecolor=pg,
                   tickfont=dict(size=8, color=pt), title='Sample'),
        yaxis=dict(gridcolor=pb, linecolor=pg,
                   tickfont=dict(size=9, color=pt), autorange='reversed'),
        margin=dict(l=10, r=150, t=40, b=10),
    )

    # Mean / Std reference columns to the right of the heatmap — plain text,
    # not color-coded, so they read as reference numbers rather than one
    # more thing to visually compare against the color scale. Placed well
    # clear of the colorbar (fixed at x=1.02, thickness=10) so its tick
    # labels never collide with these columns.
    for disp in y_labels:
        s = stats_by_method[disp]
        fig.add_annotation(xref="paper", x=1.16, xanchor="left", yref="y", y=disp,
                            text=f"{s['mean']:.3f}", showarrow=False,
                            font=dict(family="JetBrains Mono,monospace", size=8, color=pt))
        fig.add_annotation(xref="paper", x=1.28, xanchor="left", yref="y", y=disp,
                            text=f"{s['std']:.3f}", showarrow=False,
                            font=dict(family="JetBrains Mono,monospace", size=8, color=pt))
    fig.add_annotation(xref="paper", x=1.16, xanchor="left", yref="paper", y=1.06,
                        text="Mean", showarrow=False,
                        font=dict(family="JetBrains Mono,monospace", size=8, color=pt))
    fig.add_annotation(xref="paper", x=1.28, xanchor="left", yref="paper", y=1.06,
                        text="Std", showarrow=False,
                        font=dict(family="JetBrains Mono,monospace", size=8, color=pt))

    if chosen_metric == 'ktc_score':
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            'color:var(--tx3);margin-bottom:4px">0.0 = water baseline (predicting nothing) '
            '&middot; negative = worse than that</div>',
            unsafe_allow_html=True,
        )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # -- Expandable per-method summary table ---------------------------------
    with st.expander("Per-Method Summary Table", expanded=False):
        st.caption(
            "Std is the standard deviation of each method's scores across every test shown above — "
            "a measure of consistency, not accuracy. Best_Level / Worst_Level show which difficulty "
            "level produced that method's single best and single worst score."
        )
        hm_rows = [
            {
                'Method': disp,
                'Mean': round(stats_by_method[disp]['mean'], 4),
                'Std': round(stats_by_method[disp]['std'], 4),
                'Min': round(stats_by_method[disp]['min'], 4),
                'Max': round(stats_by_method[disp]['max'], 4),
                'Best_Level': stats_by_method[disp]['best_level'],
                'Worst_Level': stats_by_method[disp]['worst_level'],
                'Runs': stats_by_method[disp]['runs'],
            }
            for disp in method_names
        ]
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
        figs_dir = run_dir / "figures"

        # Use pre-generated figures from the benchmark run - no matplotlib at export time.
        _CHART_NAMES = {
            "leaderboard":       "Leaderboard",
            "degradation_curve": "KTC Score vs Difficulty",
            "metrics_heatmap":   "Metrics Heatmap",
            "runtime_comparison":"Runtime Comparison",
            "failure_gallery":   "Failure Gallery",
        }

        def collect_recon_pngs() -> List[Path]:
            """Return up to 4 per-run panel PNGs from the figures directory."""
            if not figs_dir.exists():
                return []
            skip = set(_CHART_NAMES.keys())
            panels = sorted(
                p for p in figs_dir.glob("*.png")
                if not any(p.stem.startswith(s) for s in skip)
            )
            return panels[:4]

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
            [frame("Leaderboard", figs_dir / "leaderboard.png"),
             frame("Metrics Heatmap", figs_dir / "metrics_heatmap.png")],
            [frame("KTC Score vs Difficulty", figs_dir / "degradation_curve.png"),
             frame("Runtime Comparison", figs_dir / "runtime_comparison.png")],
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


def _write_dashboard_report_charts(scores: Dict, per_run: Dict, mm: Dict, export_figures: Path) -> None:
    """Save the same Plotly chart style used by the dashboard for report embedding."""
    if not scores:
        return

    df = build_leaderboard_df(scores, per_run, mm)
    fig = build_leaderboard_figure(scores, df)
    fig.write_image(str(export_figures / "leaderboard_dashboard.png"), scale=2)

    fig = go.Figure()
    stats = []
    chosen = [m for m in all_methods(scores) if m not in st.session_state.get("custom_methods", [])]
    all_levels = []
    for disp_name in chosen:
        ik = mm.get(disp_name, disp_name)
        if ik not in per_run:
            continue
        samps = per_run.get(ik, {})
        if not samps:
            continue
        levels = sorted({int(e["level"]) for e in samps.values()})
        by_level = {
            lv: [e.get("ktc_score", 0.0) for e in samps.values() if int(e.get("level", 1)) == lv]
            for lv in levels
        }
        ktc = [float(np.mean(by_level[lv])) for lv in levels]
        sds = [float(np.std(by_level[lv])) for lv in levels]
        c = get_method_color(disp_name)
        mu = float(np.mean(ktc))
        upper = [v + s for v, s in zip(ktc, sds)]
        lower = [max(0, v - s) for v, s in zip(ktc, sds)]
        all_levels.extend(levels)
        fig.add_trace(go.Scatter(
            x=levels + levels[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=hex_to_rgba(c, 0.10),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=levels,
            y=ktc,
            mode="lines+markers",
            name=disp_name,
            line=dict(width=2.5, color=c),
            marker=dict(size=7, color=c, line=dict(width=2, color="#ffffff")),
            hovertemplate=f"<b>{disp_name}</b><br>Level: %{{x}}<br>KTC: %{{y:.4f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[min(levels), max(levels)],
            y=[mu, mu],
            mode="lines",
            line=dict(width=1, color=c, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        stats.append(disp_name)
    if stats:
        min_lv = min(all_levels)
        max_lv = max(all_levels)
        fig.update_layout(
            title=f"KTC Score - Levels {min_lv}-{max_lv}",
            xaxis_title="Difficulty Level",
            yaxis_title="KTC Score (higher = better)",
            width=1040,
            height=430,
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f6f8fa",
            font=dict(family="JetBrains Mono,monospace", color="#848d97", size=11),
            xaxis=dict(gridcolor="#d0d7de", linecolor="#d0d7de", dtick=1),
            yaxis=dict(gridcolor="#d0d7de", linecolor="#d0d7de"),
            legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#d0d7de", borderwidth=1, font=dict(size=10)),
            margin=dict(l=52, r=18, t=48, b=54),
        )
        fig.write_image(str(export_figures / "degradation_dashboard.png"), scale=2)


def _write_dashboard_hull_charts(scores: Dict, per_run: Dict, mm: Dict, export_figures: Path) -> None:
    """Save hull-analysis charts for the report from the same data as the dashboard tab."""
    hull_rows = []
    for method in scores.keys():
        pr_key = mm.get(method, method)
        for entry in per_run.get(pr_key, {}).values():
            hull = entry.get("hull") or {}
            if not hull:
                continue
            hull_rows.append({
                "Method": method,
                "Level": int(entry.get("level", 1)),
                "KTC": entry.get("ktc_score", entry.get("metrics", {}).get("ktc_score", 0.0)),
                "Res Center Err": hull.get("hull_resistive_center_error"),
                "Res Area Err": hull.get("hull_resistive_area_error"),
                "Res Perim Err": hull.get("hull_resistive_perimeter_error"),
                "Con Center Err": hull.get("hull_conductive_center_error"),
                "Con Area Err": hull.get("hull_conductive_area_error"),
            })
    if not hull_rows:
        return

    df = pd.DataFrame(hull_rows)
    pc = st.session_state.get('_pcolors', {})
    pc = {
        "bg": pc.get("bg", "#f6f8fa"),
        "paper": pc.get("paper", "rgba(0,0,0,0)"),
        "grid": pc.get("grid", "#d0d7de"),
        "text": pc.get("text", "#848d97"),
        "legend": pc.get("legend", "rgba(255,255,255,.9)"),
    }
    selected_methods = list(scores.keys())

    avg_center = df.dropna(subset=["Res Center Err"]).groupby("Method")["Res Center Err"].mean().sort_values()
    if not avg_center.empty:
        fig = go.Figure()
        colors = [get_method_color(m) for m in avg_center.index]
        fig.add_trace(go.Bar(
            x=avg_center.index,
            y=avg_center.values,
            marker_color=colors,
            text=[f"{v:.1f}px" for v in avg_center.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(
            title="Resistive Region - Center Error by Method",
            yaxis_title="Center Error (px)",
            width=1040,
            height=300,
            margin=dict(l=52, r=18, t=48, b=62),
            plot_bgcolor=pc["paper"],
            paper_bgcolor=pc["paper"],
            font=dict(family="JetBrains Mono", size=11, color=pc["text"]),
            yaxis=dict(gridcolor=pc["grid"], zeroline=False),
            xaxis=dict(gridcolor=pc["grid"]),
        )
        fig.write_image(str(export_figures / "hull_center_error.png"), scale=2)

    avg_area = df.dropna(subset=["Res Area Err"]).groupby("Method")["Res Area Err"].mean().sort_values()
    if not avg_area.empty:
        fig = go.Figure()
        colors = [get_method_color(m) for m in avg_area.index]
        fig.add_trace(go.Bar(
            x=avg_area.index,
            y=avg_area.values,
            marker_color=colors,
            text=[f"{v:.0f}px2" for v in avg_area.values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(
            title="Resistive Region - Hull Area Error by Method",
            yaxis_title="Area Error (px2)",
            width=1040,
            height=300,
            margin=dict(l=52, r=18, t=48, b=62),
            plot_bgcolor=pc["paper"],
            paper_bgcolor=pc["paper"],
            font=dict(family="JetBrains Mono", size=11, color=pc["text"]),
            yaxis=dict(gridcolor=pc["grid"], zeroline=False),
            xaxis=dict(gridcolor=pc["grid"]),
        )
        fig.write_image(str(export_figures / "hull_area_error.png"), scale=2)

    scatter_df = df.dropna(subset=["Res Center Err", "KTC"])
    if not scatter_df.empty:
        fig = go.Figure()
        for method in selected_methods:
            mdf = scatter_df[scatter_df["Method"] == method]
            if mdf.empty:
                continue
            fig.add_trace(go.Scatter(
                x=mdf["KTC"],
                y=mdf["Res Center Err"],
                mode="markers",
                name=method.replace("Reconstruction", "Recon"),
                marker=dict(size=8, color=get_method_color(method)),
            ))
        fig.update_layout(
            title="KTC Score vs Center Error",
            xaxis_title="KTC Score",
            yaxis_title="Center Error (px)",
            width=1040,
            height=300,
            margin=dict(l=52, r=18, t=48, b=54),
            plot_bgcolor=pc["paper"],
            paper_bgcolor=pc["paper"],
            font=dict(family="JetBrains Mono", size=11, color=pc["text"]),
            xaxis=dict(gridcolor=pc["grid"]),
            yaxis=dict(gridcolor=pc["grid"]),
            legend=dict(bgcolor=pc["legend"], bordercolor=pc["grid"], borderwidth=1, font=dict(size=10)),
        )
        fig.write_image(str(export_figures / "hull_ktc_vs_center_error.png"), scale=2)

    deg_df = df.dropna(subset=["Res Center Err"])
    if not deg_df.empty:
        fig = go.Figure()
        for method in selected_methods:
            mdf = deg_df[deg_df["Method"] == method]
            if mdf.empty:
                continue
            by_level = mdf.groupby("Level")["Res Center Err"].mean().sort_index()
            fig.add_trace(go.Scatter(
                x=by_level.index,
                y=by_level.values,
                mode="lines+markers",
                name=method.replace("Reconstruction", "Recon"),
                line=dict(color=get_method_color(method), width=2.5),
                marker=dict(size=7),
            ))
        fig.update_layout(
            title="Hull Error Degradation by Level",
            xaxis_title="Difficulty Level",
            yaxis_title="Avg Center Error (px)",
            width=1040,
            height=300,
            margin=dict(l=52, r=18, t=48, b=54),
            plot_bgcolor=pc["paper"],
            paper_bgcolor=pc["paper"],
            font=dict(family="JetBrains Mono", size=11, color=pc["text"]),
            xaxis=dict(gridcolor=pc["grid"], dtick=1),
            yaxis=dict(gridcolor=pc["grid"]),
            legend=dict(bgcolor=pc["legend"], bordercolor=pc["grid"], borderwidth=1, font=dict(size=10)),
        )
        fig.write_image(str(export_figures / "hull_error_degradation.png"), scale=2)


def _render_html_report_export(scores:Dict, per_run:Dict, mm:Dict, run_name:str, target=None):
    """Generate/download the framework HTML report from filtered dashboard data."""
    target = target or st
    try:
        from ktc_framework.reporting.html_report import generate_html_report

        run_dir = Path("outputs") / run_name
        results = []
        selected_metric_labels = st.session_state.get("selected_metrics", ALL_METRICS_SIDEBAR.copy())
        selected_metric_keys = [
            key for label, key in METRIC_SPECS
            if label in selected_metric_labels
        ]
        metric_keys = list(dict.fromkeys(["ktc_score"] + selected_metric_keys))
        for display_method in scores.keys():
            ik = mm.get(display_method, display_method)
            for entry in per_run.get(ik, {}).values():
                level = int(entry.get("level", 1))
                sample = str(entry.get("sample", ""))
                dashboard_png = (
                    run_dir / "reconstructions" / f"level_{level}" /
                    f"sample_{sample}" / f"{ik}.png"
                )
                figure_png = run_dir / "figures" / f"{ik}_level{level}_sample{sample}.png"
                image_path = ""
                for candidate in [dashboard_png, figure_png, Path(str(entry.get("png_path", "")))]:
                    if candidate.exists():
                        image_path = str(candidate)
                        break
                metrics = {
                    key: float(entry.get(key, 0.0))
                    for key in metric_keys
                    if isinstance(entry.get(key, 0.0), (int, float))
                }
                results.append({
                    "method": display_method,
                    "level": level,
                    "sample": sample,
                    "metrics": metrics,
                    "hull": entry.get("hull") or {},
                    "image_path": image_path,
                    "runtime_ms": float(entry.get("runtime_ms", 0.0)),
                    "grade": entry.get("grade") or letter_grade(calculate_composite_score(metrics)),
                    "composite_score": entry.get("composite_score", calculate_composite_score(metrics)),
                    "gt_missing": entry.get("gt_missing", False),
                })

        if not results:
            target.warning("No filtered report data available to export.")
            return

        import shutil
        source_figures = run_dir / "figures"
        export_dir = run_dir / "report_export"
        export_figures = export_dir / "figures"
        if export_figures.exists():
            shutil.rmtree(export_figures)
        export_figures.mkdir(parents=True, exist_ok=True)
        # NOTE: we deliberately do NOT render the dashboard's Plotly charts to
        # PNG via kaleido here. On Windows each kaleido write_image spins up a
        # Chromium subprocess (6 charts => tens of seconds, and it can hang),
        # which is exactly what made the export appear frozen with no download
        # button. The HTML report has its own fast, self-contained SVG charts
        # (leaderboard / degradation / radar / hull) that render in a fraction
        # of a second and need no external process. Set the env var
        # KTC_REPORT_USE_KALEIDO=1 only if you specifically want pixel-identical
        # dashboard PNGs and are willing to wait.
        import os as _os
        if _os.environ.get("KTC_REPORT_USE_KALEIDO") == "1":
            for _writer, _label in (
                (_write_dashboard_report_charts, "Leaderboard/degradation"),
                (_write_dashboard_hull_charts, "Hull"),
            ):
                try:
                    _writer(scores, per_run, mm, export_figures)
                except Exception as chart_error:
                    target.warning(
                        f"{_label} PNG export failed; SVG fallback charts will be used. "
                        f"Details: {chart_error}"
                    )

        summary_names = set()
        wanted_names = set(summary_names)
        for method in sorted({r["method"] for r in results}):
            method_rows = [r for r in results if r["method"] == method]
            best = max(method_rows, key=lambda r: r["metrics"].get("ktc_score", 0.0), default=None)
            worst = min(method_rows, key=lambda r: r["metrics"].get("ktc_score", 0.0), default=None)
            for item in (best, worst):
                if item:
                    wanted_names.add(f"{item['method']}_level{item['level']}_sample{item['sample']}.png")
        if source_figures.exists():
            wanted_names.update(src.name for src in source_figures.glob("hull*.png"))
            for name in wanted_names:
                src = source_figures / name
                if src.exists():
                    shutil.copyfile(src, export_figures / name)

        with st.spinner("Building the report (embedding charts & reconstructions)…"):
            report_path = Path(generate_html_report(results, export_dir, {"_metric_keys": selected_metric_keys}))
        report_bytes = report_path.read_bytes()
        target.download_button(
            f"Download HTML Report ({len(report_bytes) / 1_000_000:.1f} MB)",
            data=report_bytes,
            file_name=f"eit_report_{run_name}.html",
            mime="text/html",
            key="html_report_download_btn",
            use_container_width=True,
        )
    except Exception as e:
        target.error(f"HTML report generation error: {e}")


# =========================================================
# MAIN
# =========================================================
# =========================================================
# VIEW 7 - HULL ANALYSIS
# =========================================================
@st.cache_data(show_spinner="Checking whether each reconstruction found the right shapes...")
def _compute_qualitative_detection(dataset_root: str, entries_key: tuple) -> dict:
    """Classify object detection per sample (detected / partially detected /
    missed) by loading the real saved prediction + ground truth and running
    HullAnalyzer fresh.

    Computed here rather than read from scores_nested.json's
    "_qualitative_summary" field (which experiment_runner.py can write) —
    that field is never actually present in any of this dashboard's real
    run directories, so relying on it would mean this feature silently
    shows nothing. Loading the saved .mat files directly is self-contained
    and doesn't depend on that separate, unpopulated pipeline.

    entries_key: tuple of (method, mat_path, level, sample) — passed as a
    plain tuple (not the per_run dict) so st.cache_data can hash it.
    """
    from ktc_framework.plugins.hull_plugin import HullAnalyzer
    from ktc_framework.metrics.qualitative_metrics import compute_qualitative_sample, aggregate_qualitative
    import scipy.io

    sample_to_num = {"A": "1", "B": "2", "C": "3"}

    def load_gt(level: int, sample: str):
        gt_dir = Path(dataset_root) / "GroundTruths"
        snum = sample_to_num.get(sample, sample)
        candidates = [
            gt_dir / f"level_{level}" / f"{snum}_true.mat",
            gt_dir / f"level{level}" / f"{snum}_true.mat",
        ]
        for path in candidates:
            if path.exists():
                mat = scipy.io.loadmat(str(path), squeeze_me=True)
                for key in ["truth", "Segmentation", "gt", "seg"]:
                    if key in mat:
                        arr = np.asarray(mat[key], dtype=np.uint8)
                        if arr.shape == (256, 256):
                            return arr
        return None

    analyzer = HullAnalyzer()
    per_method_samples: dict[str, list] = {}
    gt_cache: dict = {}

    for method, mat_path, level, sample in entries_key:
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True)
            pred = np.asarray(mat["reconstruction"], dtype=np.uint8)
        except Exception:
            continue
        if pred.shape != (256, 256):
            continue

        gt_key = (level, sample)
        if gt_key not in gt_cache:
            gt_cache[gt_key] = load_gt(level, sample)
        gt = gt_cache[gt_key]
        if gt is None or not np.any(gt):
            continue

        qual = compute_qualitative_sample(pred, gt, analyzer)
        qual['sample_id'] = f"L{level}_{sample}"
        per_method_samples.setdefault(method, []).append(qual)

    results = {}
    for method, samples in per_method_samples.items():
        agg = aggregate_qualitative(samples)
        # aggregate_qualitative() only reports detected/not — add the
        # "partially detected" nuance (some hull overlap, just below the
        # 0.3 IoU detection threshold) and a couple of concrete missed
        # examples to make the per-method story specific, not just a stat.
        partial = 0
        missed_examples: list[str] = []
        for s in samples:
            for cls in ('resistive', 'conductive'):
                if not s.get(f'{cls}_in_gt') or s.get(f'{cls}_detected'):
                    continue
                if s.get(f'{cls}_hull_iou', 0.0) > 0:
                    partial += 1
                elif len(missed_examples) < 3:
                    missed_examples.append(f"{s['sample_id']} ({cls})")
        total_objects = agg.get('resistive_gt_count', 0) + agg.get('conductive_gt_count', 0)
        total_detected = agg.get('resistive_detected_count', 0) + agg.get('conductive_detected_count', 0)
        results[method] = {
            **agg,
            'total_objects': total_objects,
            'total_detected': total_detected,
            'partial_count': partial,
            'missed_examples': missed_examples,
        }
    return results


@st.fragment
def view_hull_analysis(scores: Dict, per_run: Dict, mm: Dict, level_range: tuple = (1, 7)):
    """Object-detection quality: did each method find the right shapes,
    roughly the right size, in roughly the right place?"""

    pc = st.session_state.get('_pcolors', {})
    sel_methods = list(scores.keys())
    lvl_min, lvl_max = level_range

    # -- Collect hull data across all methods ------------------
    hull_rows = []
    entries_for_detection = []
    for method in sel_methods:
        pr_key = mm.get(method, method)
        entries = per_run.get(pr_key, {})
        for key, entry in entries.items():
            lv = int(entry.get("level", 1))
            if lv < lvl_min or lv > lvl_max:
                continue
            hull = entry.get("hull", {})
            if hull:
                hull_rows.append({
                    "Method": method,
                    "Level": lv,
                    "Sample": entry.get("sample", "?"),
                    "KTC": entry.get("ktc_score", entry.get("metrics", {}).get("ktc_score", 0.0)),
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
            mat_path = entry.get("mat_path", "")
            if mat_path and Path(mat_path).exists():
                entries_for_detection.append((method, mat_path, lv, entry.get("sample", "A")))

    if not hull_rows:
        st.info("No hull data available. Run the benchmark or `python scripts/compute_hull_data.py` to generate hull analysis.")
        return

    df = pd.DataFrame(hull_rows)

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx2);margin-bottom:10px">'
        'A high KTC score means the shape overlap was good on average — but did each method actually '
        'find the right number of objects, in roughly the right place?</div>',
        unsafe_allow_html=True)

    # -- Object-detection leaderboard ---------------------------
    render_section_header(
        "OBJECT DETECTION LEADERBOARD",
        "A reconstruction 'detects' an object when its predicted shape overlaps the true object "
        "by at least 30% (IoU >= 0.3) — a lower bar than typical image benchmarks, chosen because "
        "these reconstructions are inherently blurry by the nature of the physics involved. A "
        "'partial match' means some overlap was found but not enough to count as a full detection. "
        "Methods are ranked by an accuracy score, not raw detection count: a full detection is worth "
        "+1 point, a partial match +0.5, and a false positive (claiming an object that isn't there) "
        "costs -1 — the same as a miss — so confidently hallucinating objects is penalized, not free.",
    )
    if entries_for_detection:
        dataset_root = st.session_state.get("cfg_dataset_root", "EvaluationData")
        qual = _compute_qualitative_detection(dataset_root, tuple(entries_for_detection))
        if not qual:
            st.info("Could not load saved reconstructions/ground truth to check object detection for this run.")
        else:
            def _detection_score(q: dict) -> float:
                """Composite accuracy score, not a raw detection rate.

                +1.0 per full detection, +0.5 per partial match, -1.0 per
                false-positive sample (a hallucinated object is penalized
                as heavily as a missed one, not just left unscored) — all
                divided by the number of real objects tested. Can go
                negative if a method racks up more false positives than
                correct detections; that's intentional, not a display bug.
                """
                total = q['total_objects']
                if not total:
                    return 0.0
                return (
                    q['total_detected'] * 1.0
                    + q['partial_count'] * 0.5
                    - q.get('false_positive_count', 0) * 1.0
                ) / total

            ranked = [(m, q) for m, q in sorted(qual.items(), key=lambda kv: _detection_score(kv[1]), reverse=True)
                      if q['total_objects'] > 0]

            def _b(name: str) -> str:
                # Real <b> markup, not markdown "**" — st.markdown() does
                # not re-parse markdown syntax inside a raw HTML block.
                return f'<b style="color:var(--tx)">{method_display_name(name)}</b>'

            if ranked:
                # -- Field-wide stat strip --------------------------------
                n_objects = max(q['total_objects'] for _, q in ranked)
                field_avg_score = 100.0 * float(np.mean([_detection_score(q) for _, q in ranked]))
                total_fp = sum(q.get('false_positive_count', 0) for _, q in ranked)
                stat_cols = st.columns(4)
                for col, (val, lbl) in zip(stat_cols, [
                    (str(len(ranked)), "methods compared"),
                    (str(n_objects), "objects tested"),
                    (f"{field_avg_score:.0f}%", "field avg. accuracy score"),
                    (str(total_fp), "false positives, all methods"),
                ]):
                    col.markdown(
                        f'<div style="text-align:center;background:var(--sur);border:1px solid var(--bd);'
                        f'border-radius:6px;padding:10px 4px">'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;font-weight:700;'
                        f'color:var(--tx)">{val}</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        f'color:var(--tx3);text-transform:uppercase;letter-spacing:.04em">{lbl}</div>'
                        f'</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:var(--tx3);'
                    'margin:6px 0 14px">Accuracy score = (full detections + 0.5 &times; partial matches '
                    '&minus; false positives) &divide; objects tested. A false positive costs as much as a miss.</div>',
                    unsafe_allow_html=True)

                # -- Leader callout — driven by the same composite score
                # used to rank the cards below, so it can never contradict
                # the #1/#2 order.
                leader, leader_q = ranked[0]
                leader_score = _detection_score(leader_q)
                if len(ranked) > 1:
                    runner_up, ru_q = ranked[1]
                    ru_score = _detection_score(ru_q)
                    if abs(leader_score - ru_score) < 1e-9:
                        lead_story = (
                            f"{_b(leader)} and {_b(runner_up)} are tied at {leader_score * 100:.0f}% accuracy — "
                            f"identical detections, partial matches, and false positives."
                        )
                    else:
                        drivers = []
                        d_diff = leader_q['total_detected'] - ru_q['total_detected']
                        if d_diff:
                            drivers.append(
                                f"{'found ' + str(abs(d_diff)) + ' more full detection' + ('s' if abs(d_diff) != 1 else '') if d_diff > 0 else str(abs(d_diff)) + ' fewer full detections'}"
                            )
                        fp_l, fp_r = leader_q.get('false_positive_count', 0), ru_q.get('false_positive_count', 0)
                        if fp_l != fp_r:
                            drivers.append(f"{fp_l} false positive{'s' if fp_l != 1 else ''} vs {fp_r}")
                        detail = "; ".join(drivers) if drivers else "more partial-match credit"
                        lead_story = (
                            f"{_b(leader)} leads with a {leader_score * 100:.0f}% accuracy score vs "
                            f"{_b(runner_up)}'s {ru_score * 100:.0f}% — {detail}."
                        )
                else:
                    lead_story = f"{_b(leader)} scored {leader_score * 100:.0f}% accuracy."
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;line-height:1.6;'
                    f'color:var(--tx2);background:var(--sur);border:1px solid var(--bd);'
                    f'border-radius:6px;padding:12px 16px;margin-bottom:14px">{lead_story}</div>',
                    unsafe_allow_html=True)

            def _chip(value: str, label: str) -> str:
                return (
                    '<span style="display:inline-block;background:var(--bg);border:1px solid var(--bd);'
                    'border-radius:10px;padding:2px 9px;margin-right:6px;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:9px;color:var(--tx2);white-space:nowrap"><b style="color:var(--tx)">{value}</b> {label}</span>'
                )

            _RANK_COLORS = {1: "#9a6700", 2: "#57606a", 3: "#bf8700"}
            for i, (method, q) in enumerate(ranked, start=1):
                total_objects = q['total_objects']
                total_detected = q['total_detected']
                partial = q['partial_count']
                false_pos = q.get('false_positive_count', 0)
                score = _detection_score(q)
                score_pct = score * 100.0
                bar_pct = max(0.0, min(100.0, score_pct))
                label = method_display_name(method)

                chips = (
                    _chip(q.get('resistive_detected_str', '0/0'), "resistive")
                    + _chip(q.get('conductive_detected_str', '0/0'), "conductive")
                    + _chip(str(partial), f"partial match{'es' if partial != 1 else ''} (+0.5 pt each)")
                    + _chip(str(false_pos), f"false positive{'s' if false_pos != 1 else ''} (−1 pt each)")
                )
                missed = q.get('missed_examples') or []
                missed_line = (
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:var(--tx3);'
                    f'margin-top:4px">Missed entirely: {", ".join(missed)}'
                    f'{" ..." if total_objects - total_detected - partial > len(missed) else ""}</div>'
                    if missed else ""
                )

                bar_color = "#1a7f37" if score_pct >= 80 else "#9a6700" if score_pct >= 40 else "#cf222e"
                rank_color = _RANK_COLORS.get(i, "#848d97")
                rank_border = f"3px solid {rank_color}" if i <= 3 else "3px solid transparent"
                st.markdown(
                    f'<div style="background:var(--sur);border:1px solid var(--bd);border-left:{rank_border};'
                    f'border-radius:6px;padding:10px 14px;margin-bottom:8px">'
                    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
                    f'<div style="width:26px;flex-shrink:0;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:12px;font-weight:700;color:{rank_color}">#{i}</div>'
                    f'<div style="width:150px;flex-shrink:0;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:12px;font-weight:600;color:var(--tx);white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis">{label}</div>'
                    f'<div style="flex:1;background:var(--bg);border:1px solid var(--bd);border-radius:4px;height:16px;overflow:hidden">'
                    f'<div style="width:{bar_pct:.0f}%;background:{bar_color};height:100%"></div></div>'
                    f'<div style="width:64px;flex-shrink:0;text-align:right;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:11px;font-weight:700;color:{"var(--tx)" if score_pct >= 0 else "#cf222e"}">{score_pct:.0f}%</div>'
                    f'</div>'
                    f'<div style="padding-left:38px">{chips}</div>'
                    f'{missed_line}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Object-detection analysis needs the saved .mat reconstructions, which aren't available for this run.")

    # -- Detailed table (collapsed by default) -------------------
    with st.expander("Per-Run Hull Metrics", expanded=False):
        display_cols = ["Method", "Level", "Sample", "KTC",
                        "Res Center Err", "Res Area Err", "Res Perim Err",
                        "Res Pixels", "Con Pixels"]
        tbl = df[display_cols].copy()
        for c in ["KTC", "Res Center Err", "Res Area Err", "Res Perim Err"]:
            tbl[c] = tbl[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


def _render_first_run_wizard() -> None:
    """D2: Guided setup panel shown when no benchmark output exists yet."""
    import importlib as _il

    _d = Path(st.session_state.get("cfg_dataset_root", "EvaluationData"))
    _m = Path(st.session_state.get("cfg_mesh_path", "Codes_Matlab/Mesh_sparse.mat"))
    _eval_ok = any((_d / x).is_dir() for x in ["evaluation_datasets", "EvaluationData"])
    _mesh_ok = _m.exists()

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;font-weight:700;'
        'color:var(--tx);margin-bottom:6px">Getting started</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:var(--tx3);'
        'margin-bottom:20px">No benchmark output found yet. Complete the steps below to run '
        'your first evaluation.</div>',
        unsafe_allow_html=True)

    steps = [
        (
            "1. Set dataset path",
            _eval_ok,
            "Open **Dataset Settings** in the sidebar. Set **Dataset root** to the folder "
            "containing `evaluation_datasets/` and `GroundTruths/`, then click **Validate paths**.",
        ),
        (
            "2. Confirm mesh file",
            _mesh_ok,
            "Set **Mesh path** to your `Mesh_sparse.mat` (default: `Codes_Matlab/Mesh_sparse.mat`). "
            "Validate again to confirm it resolves.",
        ),
        (
            "3. Run the benchmark",
            False,
            "Click **Run all methods** in the sidebar. Progress updates every 2 s. "
            "The dashboard reloads automatically when the run completes.",
        ),
    ]

    for title, done, body in steps:
        icon_col = "#1a7f37" if done else "#9a6700"
        icon = "DONE" if done else "TODO"
        st.markdown(
            f'<div style="background:var(--sur);border:1px solid var(--bd);border-radius:8px;'
            f'padding:14px 16px;margin-bottom:10px">'
            f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:700;'
            f'color:{icon_col};border:1px solid {icon_col};border-radius:4px;padding:1px 5px">'
            f'{icon}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;font-weight:600;'
            f'color:var(--tx)">{title}</span>'
            f'</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx3);'
            f'line-height:1.6">{body}</div>'
            f'</div>',
            unsafe_allow_html=True)

    if _eval_ok and _mesh_ok:
        st.success("Paths look good — click **Run all methods** in the sidebar to begin.")


def view_raw_data(run_dir: Path) -> None:
    """C4: Read-only JSON viewer for scores, per_run_metrics, and failures."""
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
        'color:var(--tx3);margin-bottom:10px">Raw output files from the active run. '
        'Read-only — edit on disk if needed.</div>',
        unsafe_allow_html=True)
    files = [
        ("scores.json",           "Aggregated method scores"),
        ("per_run_metrics.json",  "Per-sample metrics for every run"),
        ("failures.json",         "Samples that errored during the run"),
    ]
    for fname, desc in files:
        fpath = run_dir / fname
        expanded = fname == "failures.json" and fpath.exists() and fpath.stat().st_size > 2
        with st.expander(f"{fname}  —  {desc}", expanded=expanded):
            if fpath.exists():
                try:
                    st.json(json.loads(fpath.read_text(encoding="utf-8")))
                except Exception as exc:
                    st.error(f"Could not parse {fname}: {exc}")
                    st.code(fpath.read_text(encoding="utf-8", errors="replace"), language="text")
            else:
                st.info(f"{fname} not present in this run.")


def main():
    # One-time-per-session auto-import: register every method already
    # sitting in external_methods/ (bundles + CLI scripts + class-based
    # plugins) without requiring a manual "Import from external_methods/"
    # click first. external_methods/ is git-tracked (unlike outputs/), so
    # this is what makes a method actually *runnable* immediately after a
    # teammate pulls the repo and opens the dashboard for the first time —
    # publish_method()'s baseline snapshot covers the *scores* half of that,
    # this covers the *runnable* half.
    if not st.session_state.get('_ext_methods_autoloaded'):
        try:
            from ktc_framework.registry import load_external_methods as _autoload_ext
            _autoload_ext(["external_methods"])
        except Exception:
            pass
        st.session_state['_ext_methods_autoloaded'] = True

    # Populate the available method list from the latest run BEFORE the sidebar
    # renders, so the methods section shows immediately instead of stalling on
    # "Loading methods..." on the first paint. (The sidebar reads this; the
    # main body refreshes it again after load_data below.)
    try:
        st.session_state['_available_methods'] = discover_available_methods()
    except Exception:
        st.session_state.setdefault('_available_methods', [])

    pdf_export_slot = render_sidebar()
    plot_bg     = '#f6f8fa'
    plot_paper  = 'rgba(0,0,0,0)'
    plot_grid   = '#d0d7de'
    plot_text   = '#848d97'
    plot_legend = 'rgba(255,255,255,.9)'
    st.session_state['_pcolors'] = dict(bg=plot_bg, paper=plot_paper,
                                        grid=plot_grid, text=plot_text, legend=plot_legend)

    header_bg = 'var(--sur)'

    # Header
    st.markdown("""
    <div class="dash-header">
      <div class="dash-title">EIT Reconstruction Dashboard</div>
      <div class="dash-sub">Electrical Impedance Tomography &mdash; Benchmarking &amp; Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress banner - visible whenever a benchmark subprocess is running
    render_bench_progress()

    try:
        latest_run = find_latest_run()
        scores, per_run, mm = load_data(str(latest_run.resolve()))
        scores, per_run, mm, published_fallback = _apply_published_baselines(scores, per_run, mm)
        removed_external = set(st.session_state.get('_removed_external_methods', []))
        if removed_external:
            scores = {k: v for k, v in scores.items() if k not in removed_external}
            per_run = {k: v for k, v in per_run.items() if k not in removed_external}
            mm = {k: v for k, v in mm.items() if k not in removed_external and v not in removed_external}

        # Active run label (+ badges for missing GT and failed samples)
        n_gt_missing = count_gt_missing(per_run)
        n_failures = len(_read_run_failures(latest_run))
        gt_badge = (
            f' &nbsp;|&nbsp; <span style="background:#ffebe9;border:1px solid #cf222e;'
            f'color:#cf222e;border-radius:5px;padding:1px 6px;font-weight:600">'
            f'{n_gt_missing} scored without GT</span>'
        ) if n_gt_missing else ''
        fail_badge = (
            f' &nbsp;|&nbsp; <span style="background:#ffebe9;border:1px solid #cf222e;'
            f'color:#cf222e;border-radius:5px;padding:1px 6px;font-weight:600">'
            f'{n_failures} failed</span>'
        ) if n_failures else ''
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
            f'color:var(--tx3);margin:-2px 0 10px;padding:0 2px;line-height:1.35">'
            f'Run: <span style="color:var(--grn)">{latest_run.name}</span> &nbsp;|&nbsp; '
            f'{len(scores)} method(s) &nbsp;|&nbsp; '
            f'{sum(len(v) for v in per_run.values()) if per_run else 0} total reconstructions'
            f'{gt_badge}{fail_badge}</div>',
            unsafe_allow_html=True)

        if published_fallback:
            st.info(
                "Showing a published baseline (not a local run) for: "
                + ", ".join(sorted(method_display_name(m) for m in published_fallback))
                + ". Click Run on that method to refresh it with a live result."
            )

        if not scores and not per_run:
            _render_first_run_wizard()
            return

        # Store available methods so sidebar checkboxes can render them
        st.session_state['_available_methods'] = discover_available_methods()
        # Selected metrics + methods + level range from sidebar
        sel_metrics = st.session_state.get('selected_metrics', ['KTC Score'])
        sel_methods = st.session_state.get('selected_methods', list(scores.keys()))
        level_range = st.session_state.get('level_range', (1, 7))
        sel_samples = st.session_state.get('selected_samples', ['A', 'B', 'C'])

        # Apply method, level, and sample filters once so every tab/report agrees.
        scores_f, per_run_f, mm_f = apply_dashboard_filters(
            scores, per_run, mm, sel_methods, level_range, sel_samples
        )
        if not scores_f:
            st.info("No dashboard data matches the selected sidebar filters.")
            return
        placeholder_methods = [m for m in scores_f if m not in scores]
        if placeholder_methods:
            st.warning(
                "Selected method(s) not present in the active run, shown as zero placeholders: "
                + ", ".join(placeholder_methods)
            )
        # Dataset info - inline KPI row (no expander = no icon issue)
        n_s = len(per_run_f.get(list(per_run_f.keys())[0], {})) if per_run_f else 0
        n_t = sum(len(v) for v in per_run_f.values()) if per_run_f else 0
        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:12px;align-items:stretch">'
            f'<div style="flex:0 0 112px;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:10px 12px;min-height:72px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:19px;font-weight:600;color:var(--tx);line-height:1">{len(scores_f)}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-top:6px">Methods</div>'
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

        with st.expander("Method Library"):
            _render_method_library_contents(st)

        # A1: Surface any per-sample failures recorded by the runner
        _render_failures_panel(latest_run)

        # PDF export - triggered from sidebar button
        if st.session_state.get('_trigger_pdf'):
            st.session_state['_trigger_pdf'] = False
            _render_html_report_export(scores_f, per_run_f, mm_f, latest_run.name, target=pdf_export_slot)

        # Tabs
        t1,t2,t3,t4,t5,t6 = st.tabs([
            "LEADERBOARD", "HEATMAP",
            "DEGRADATION", "RADAR",
            "RECONSTRUCTION", "HULL ANALYSIS"])

        with t1: view_leaderboard(scores_f, per_run_f, sel_metrics, mm_f, level_range)
        with t2: view_heatmap(scores_f, per_run_f, mm_f, level_range)
        with t3: view_degradation_curve(scores_f, per_run_f, mm_f, level_range)
        with t4: view_radar_chart(scores_f, per_run_f, sel_metrics)
        with t5: view_comparison(scores_f, per_run_f, mm_f, sel_metrics, level_range)
        with t6: view_hull_analysis(scores_f, per_run_f, mm_f, level_range)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# Run the app
main()

