"""Dashboard theme: CSS, color constants, and the hex_to_rgba utility."""
from __future__ import annotations

import streamlit as st
from matplotlib.colors import ListedColormap

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
.info-tip:focus-visible{outline:2px solid var(--c2);outline-offset:2px;}
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
.kpi{flex:1;background:var(--sur);border:1px solid var(--bd);border-radius:7px;padding:11px 13px;position:relative;overflow:visible;}
.kpi:hover,.kpi:focus-within{z-index:10;}
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

COLORS = {
    'water': '#1a3a5c', 'resistive': '#D85A30', 'conductive': '#1D9E75',
    'primary': '#0F172A', 'success': '#1D9E75', 'warning': '#F5A623', 'danger': '#D85A30',
    'teal': '#0F766E', 'method_bp': '#6366F1', 'method_gn': '#0EA5E9', 'method_un': '#10B981',
}
ALL_LEVELS = [1, 2, 3, 4, 5, 6, 7]
COLORMAP = ListedColormap([COLORS['water'], COLORS['resistive'], COLORS['conductive']])


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_theme() -> None:
    """Inject the dashboard CSS/JS into the current Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)
