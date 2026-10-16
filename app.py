"""CROWDVISION AI - Automated Crowd Density Estimation & Management System.
Streamlit dashboard (UI layer only; the AI pipeline in src/ is unchanged).

Run with:  streamlit run app.py
"""
import datetime
import html as html_lib
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

from src import config
from src.alerts import ALARM_SECONDS, COLUMNS as EVENT_COLUMNS, alarm_component_html, log_event, read_log, zone_names
from src.density_unet import UNetDensity
from src.detector_yolo import YoloCounter
from src.pipeline import Settings, analyze_frame
from src.preprocessing import BackgroundSubtractor

BLACK_MEAN = 8           # frames darker than this (mean pixel value) are skipped
ALARM_COOLDOWN_S = 30    # in video mode, do not re-ring the alarm more often than this
BATCH_FRAMES = 3         # video: analyzed frames per script run before the app reruns (Stop reacts between batches)
HISTORY_MAX = 6          # session-only list of recent analyses

PAGES = ["Home", "Crowd Analysis", "Event History", "About the Project"]
VIEW_MODES = ["Side by side", "AI overlay only", "Original only"]
PROJECT_TITLE = "Automated Crowd Density Estimation & Management System"
TEAM = "Group 8"
PROGRAMME = "B.Tech Computer Science and Engineering with Artificial Intelligence and Machine Learning (CSE AI & ML)"

st.set_page_config(page_title="CROWDVISION AI", page_icon="◈", layout="wide")


def esc(x):
    return html_lib.escape(str(x))


def do_rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


_toggle = getattr(st, "toggle", st.checkbox)

# ---------------- session defaults ----------------
for _k, _v in {"page": "Home", "history": [], "light_theme": False, "present_mode": False,
               "view_mode": VIEW_MODES[0], "alarm_n": 0, "view_n": 0}.items():
    st.session_state.setdefault(_k, _v)

# =====================================================================
# STYLING (presentation only)
# =====================================================================
THEMES = {
    "dark": {"bg": "#0B1220", "panel": "#111B2E", "panel2": "#16233B", "border": "#23344F", "text": "#E6EDF7",
             "muted": "#8FA1BD", "accent": "#22D3EE", "on-accent": "#04141A", "safe": "#14B8A6", "crit": "#EF4444",
             "crit-text": "#FCA5A5", "crit-bg": "rgba(239,68,68,.12)", "accent-bg": "rgba(34,211,238,.15)"},
    "light": {"bg": "#F3F6FB", "panel": "#FFFFFF", "panel2": "#EAF0F8", "border": "#CBD6E6", "text": "#0F1B2D",
              "muted": "#4B5E7A", "accent": "#0E7490", "on-accent": "#FFFFFF", "safe": "#0F9D8A", "crit": "#DC2626",
              "crit-text": "#B91C1C", "crit-bg": "rgba(220,38,38,.09)", "accent-bg": "rgba(14,116,144,.12)"},
}

STATIC_CSS = """
[data-testid="stAppViewContainer"] {background: var(--bg);}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: var(--panel); border-right: 1px solid var(--border);}
.block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1300px;}
footer {visibility: hidden;}
[data-stale="true"] {opacity: 1 !important; transition: none !important;}
h1, h2, h3, h4, p, label, li, span, div {color: var(--text);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color: var(--muted) !important;}
.stButton > button, .stDownloadButton > button {border-radius: 8px; font-weight: 600;}
.stButton > button[kind="secondary"], .stDownloadButton > button, [data-testid="stBaseButton-secondary"] {
  background: var(--panel); color: var(--text); border: 1px solid var(--border);}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--accent); border: 1px solid var(--accent);}
.stButton > button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {color: var(--on-accent) !important;}
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="select"] > div, textarea {
  background: var(--panel2) !important; color: var(--text) !important; border-color: var(--border) !important;}
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] {background: var(--panel) !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {color: var(--text) !important;}
[data-testid="stFileUploaderDropzone"] {background: var(--panel); border: 1px dashed var(--border);}
[data-testid="stExpander"] {border: 1px solid var(--border); border-radius: 10px; background: var(--panel);}
[data-testid="stMetric"] {background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: .6rem .9rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--border);}
[data-testid="stImage"] img {border-radius: 8px;}
[data-testid="stDataFrame"] {border: 1px solid var(--border); border-radius: 10px; overflow: hidden;}

.brand {display: flex; align-items: center; gap: .7rem;}
.logo {width: 2.2rem; height: 2.2rem; border: 2px solid var(--accent); border-radius: 8px; padding: 4px;
       display: grid; grid-template-columns: repeat(3, 1fr); gap: 3px; box-sizing: border-box;}
.logo i {display: block; border-radius: 2px; background: var(--accent-bg);}
.logo i.hot {background: var(--accent);}
.brand-name {font-size: 1.15rem; font-weight: 800; letter-spacing: .16em;}
.brand-sub {font-size: .74rem; color: var(--muted);}

.page-title {font-size: 1.6rem; font-weight: 700; margin: .6rem 0 .15rem;}
.page-desc {font-size: .92rem; color: var(--muted); margin-bottom: .9rem;}
.meta {display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .9rem;}
.chip {font-size: .78rem; padding: .22rem .7rem; border-radius: 999px; border: 1px solid var(--border); color: var(--muted);}
.chip b {color: var(--text); font-weight: 600;}
.panel-title {font-size: .85rem; font-weight: 700; margin: .2rem 0 .1rem;}
.panel-sub {font-size: .78rem; color: var(--muted); margin-bottom: .5rem;}
.ai-label {display: inline-block; font-size: .7rem; padding: .1rem .5rem; margin-left: .4rem; border-radius: 4px;
           background: var(--accent-bg); color: var(--accent); font-weight: 600;}

.hero-title {font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: 1.2rem 0 .6rem;}
.hero-sub {font-size: 1.1rem; color: var(--accent); margin-bottom: .9rem;}
.hero-text {font-size: .98rem; color: var(--muted); line-height: 1.65; max-width: 34rem; margin-bottom: 1.2rem;}
.hero-team {font-size: .85rem; color: var(--muted); margin-bottom: .9rem; line-height: 1.5;}
.hero-team b {color: var(--text);}
.heat {display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; padding: 14px;
       background: var(--panel); border: 1px solid var(--border); border-radius: 14px;}
.heat i {display: block; aspect-ratio: 1; border-radius: 3px;}
.heat-cap {font-size: .72rem; color: var(--muted); margin-top: .5rem; text-align: center;}
.feat {background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.1rem;
       height: 100%; transition: border-color .15s, transform .15s;}
.feat:hover {border-color: var(--accent); transform: translateY(-2px);}
.feat h4 {margin: 0 0 .3rem; font-size: .98rem; color: var(--accent);}
.feat p {margin: 0; font-size: .86rem; color: var(--muted);}
.foot {text-align: center; font-size: .78rem; color: var(--muted); margin-top: 2.2rem;
       border-top: 1px solid var(--border); padding-top: 1rem;}

.alert {border-radius: 12px; padding: 1rem 1.3rem; margin: .3rem 0 1rem; border: 1px solid var(--border);}
.alert.crit {background: var(--crit-bg); border-color: var(--crit); border-left: 5px solid var(--crit);}
.alert.safe {background: var(--panel); border-left: 5px solid var(--safe);}
.alert-title {font-weight: 800; letter-spacing: .04em; font-size: 1.02rem;}
.alert.crit .alert-title {color: var(--crit-text);}
.alert.safe .alert-title {color: var(--safe);}
.alert-body {font-size: .9rem; margin-top: .3rem;}
.alert-note {font-size: .78rem; color: var(--muted); margin-top: .4rem;}

.sum {display: grid; grid-template-columns: minmax(220px, 1fr) 2fr; gap: 1rem; margin: 1rem 0;}
@media (max-width: 760px) {.sum {grid-template-columns: 1fr;}}
.big {background: var(--panel); border: 1px solid var(--accent); border-radius: 14px; padding: 1.2rem 1.4rem;}
.big-label {font-size: .85rem; color: var(--muted);}
.big-num {font-size: 4rem; font-weight: 800; line-height: 1.05; color: var(--accent);}
.big-sub {font-size: .8rem; color: var(--muted);}
.stats {background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: .4rem 1.2rem;}
.stat {display: flex; align-items: center; gap: .9rem; padding: .6rem 0; border-bottom: 1px solid var(--border);}
.stat:last-child {border-bottom: none;}
.stat-ic {width: 2rem; text-align: center; font-size: 1.1rem; color: var(--accent);}
.stat-lbl {flex: 1; font-size: .88rem; color: var(--muted);}
.stat-val {font-size: 1.1rem; font-weight: 700;}

.zgrid {display: grid; gap: .55rem;}
.zone {border-radius: 10px; padding: .7rem .6rem; border: 1px solid var(--border); background: var(--panel);}
.zone.ok {border-top: 4px solid var(--safe);}
.zone.bad {border-top: 4px solid var(--crit); background: var(--crit-bg);}
.zone.sel {outline: 2px solid var(--accent); outline-offset: 1px;}
.z-id {font-size: .78rem; color: var(--muted); font-weight: 600;}
.z-people {font-size: 1.5rem; font-weight: 700; line-height: 1.3;}
.z-dens {font-size: .78rem; color: var(--muted);}
.z-stat {font-size: .76rem; font-weight: 700; margin-top: .25rem;}
.zone.ok .z-stat {color: var(--safe);} .zone.bad .z-stat {color: var(--crit-text);}
.legend {display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .7rem; font-size: .78rem; color: var(--muted);}
.dot {display: inline-block; width: .6rem; height: .6rem; border-radius: 3px; margin-right: .35rem;}

.rank {background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: .7rem 1rem;}
.rank-row {display: flex; align-items: center; gap: .6rem; padding: .28rem 0; font-size: .8rem;}
.rk-name {width: 3.2rem; font-weight: 600;}
.rk-track {position: relative; flex: 1; height: .65rem; background: var(--panel2); border-radius: 6px;}
.rk-fill {height: 100%; border-radius: 6px; background: var(--safe);}
.rk-fill.bad {background: var(--crit);}
.rk-lim {position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--text); opacity: .75;}
.rk-val {width: 7.6rem; text-align: right; color: var(--muted); font-size: .76rem;}
.rank-note {font-size: .72rem; color: var(--muted); margin-top: .4rem;}

.card {background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;}
.card h4 {margin: 0 0 .4rem; font-size: 1rem; color: var(--accent);}
.card p {font-size: .9rem; margin: 0;}
.flow {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .7rem; margin: .5rem 0 1rem;}
.step {background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: .8rem 1rem;}
.step b {color: var(--accent); font-size: .92rem;}
.step p {margin: .25rem 0 0; font-size: .8rem; color: var(--muted);}
.empty {border: 1px dashed var(--border); border-radius: 12px; padding: 1.4rem; text-align: center; color: var(--muted);}
.fileinfo {font-size: .85rem; color: var(--muted); margin-top: .4rem;}
.fileinfo b {color: var(--text);}
.kv {display: grid; grid-template-columns: 9rem 1fr; gap: .35rem .8rem; font-size: .88rem;}
.kv span:nth-child(odd) {color: var(--muted);}
"""

PRESENT_CSS = """
.block-container {max-width: 100% !important; padding-left: 2.5rem; padding-right: 2.5rem;}
.page-title {font-size: 2.3rem;} .page-desc {font-size: 1.1rem;}
.big-num {font-size: 6.5rem;} .stat-val {font-size: 1.5rem;} .stat-lbl {font-size: 1.1rem;}
.z-people {font-size: 2.3rem;} .z-id, .z-dens, .z-stat {font-size: 1rem;}
.alert-title {font-size: 1.5rem;} .alert-body {font-size: 1.15rem;}
.chip {font-size: 1rem;} .rank-row {font-size: 1rem;}
"""

LIGHT_EXTRA_CSS = """
[data-testid="stDataFrame"] {filter: invert(1) hue-rotate(180deg);}
"""


def theme_css():
    name = "light" if st.session_state["light_theme"] else "dark"
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in THEMES[name].items()) + "}"
    css = root + STATIC_CSS
    if st.session_state["present_mode"]:
        css += PRESENT_CSS
    if name == "light":
        css += LIGHT_EXTRA_CSS
    return f"<style>{css}</style>"


st.markdown(theme_css(), unsafe_allow_html=True)

# ---------------- navigation ----------------
page = st.session_state["page"]
present = st.session_state["present_mode"]

if page == "Home" or present:
    # Sidebar is hidden but its widgets still run, so settings keep their values.
    st.markdown('<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
                '[data-testid="collapsedControl"] {display: none;}</style>', unsafe_allow_html=True)


# =====================================================================
# VIDEO / ALARM STATE HELPERS
# =====================================================================
def _rm(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def stop_video(reason=None, level="info"):
    """Release the capture, delete the temp file, mark the run as stopped. Safe to call at any time."""
    vs = st.session_state.get("vs")
    if not vs:
        return
    was_running = vs.get("running", False)
    cap = vs.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    _rm(vs.get("path"))
    vs["cap"], vs["path"], vs["running"] = None, None, False
    if was_running:
        last = st.session_state.get("last_view")
        if last is not None and vs.get("done", 0) > 0:
            add_history(last)
        if reason:
            st.session_state["video_msg"] = (level, reason)


def stop_cb():
    vs = st.session_state.get("vs") or {}
    stop_video(f"Stopped by user. {vs.get('done', 0)} frames were analyzed; the last one stays on screen.")


def goto(p):
    stop_video(None)
    st.session_state["alarm_html"] = None   # the alarm iframe must not re-fire when a page is opened again
    st.session_state["page"] = p


def _alarm_nonce():
    st.session_state["alarm_n"] += 1
    return f"<!-- {st.session_state['alarm_n']} -->"


def fire_alarm(zones, count, kind="real", slot=None):
    """Real alarm (kind='real') or a harmless test (kind='test'). Neither one touches the event log here."""
    st.session_state["alarm_html"] = alarm_component_html(zones, count) + _alarm_nonce()
    st.session_state["alarm_info"] = {"kind": kind, "zones": ", ".join(zones), "t": time.monotonic(),
                                      "clock": datetime.datetime.now().strftime("%H:%M:%S")}
    if slot is not None:
        render_alarm(slot)


def test_alarm_cb():
    fire_alarm(["TEST ONLY - NOT A REAL EVENT"], 0, kind="test")


STOP_ALARM_JS = ("<script>(function(){var w=window.parent,d=w.document;"
                 "try{if(w.__crowdAudio){w.__crowdAudio.pause();w.__crowdAudio.currentTime=0;}}catch(e){}"
                 "var r=d.getElementById('crowd-alarm-root');if(r){r.remove();}})();</script>")


def stop_alarm_cb():
    st.session_state["alarm_html"] = STOP_ALARM_JS + _alarm_nonce()
    st.session_state["alarm_info"] = {"kind": "stopped", "zones": "", "t": time.monotonic(),
                                      "clock": datetime.datetime.now().strftime("%H:%M:%S")}


def render_alarm(slot):
    html = st.session_state.get("alarm_html")
    if html:
        with slot.container():
            st_html(html, height=0)
    else:
        slot.empty()


def alarm_status_text():
    info = st.session_state.get("alarm_info")
    if not info:
        return "Idle. No alarm has been triggered in this session."
    age = time.monotonic() - info["t"]
    if info["kind"] == "stopped":
        return f"Stopped from the dashboard at {info['clock']}."
    label = "Test alarm" if info["kind"] == "test" else "Alarm"
    if age < ALARM_SECONDS:
        return (f"{label} triggered at {info['clock']} ({info['zones']}). It stops by itself after "
                f"{ALARM_SECONDS} s, or press Stop alarm.")
    return f"Last alarm ({label.lower()}, {info['clock']}) has timed out."


# =====================================================================
# MODELS + SIDEBAR SETTINGS
# =====================================================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return YoloCounter(config.YOLO_WEIGHTS), UNetDensity(config.UNET_WEIGHTS)


try:
    yolo, unet = load_models()
except Exception as exc:  # missing weights / packages
    st.error(f"The AI models could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand-name" style="color:var(--accent);font-size:1rem">◈ CROWDVISION AI</div>',
                unsafe_allow_html=True)
    st.markdown("**Analysis settings**")
    st.caption("Changes apply to the next analysis. A running video keeps the settings it started with.")
    with st.expander("Camera and detection", expanded=True):
        camera_id = st.text_input("Camera ID", "CAM-01", help="Label stored with every logged event.")
        mode_label = st.selectbox(
            "Detection mode", ["Auto (smart switch)", "Force YOLO", "Force density map"],
            help="Auto uses YOLO for sparse scenes and the U-Net density map above the switch threshold.")
        threshold_n = st.slider("Switch threshold N (people)", 5, 200, config.SWITCH_THRESHOLD_N,
                                help="In Auto mode: above this headcount the density map is used instead of YOLO.")
    with st.expander("Grid and capacity", expanded=True):
        grid_label = st.selectbox("Grid", ["3x3", "4x4"], help="The frame is split into this many zones.")
        cell_area = st.number_input("Area covered by one grid cell (m²)", 0.5, 500.0, float(config.CELL_AREA_M2),
                                    help="Real-world floor area of ONE zone. Density = people / this area.")
        limit = st.number_input("Capacity limit (people/m²)", 0.5, 10.0, float(config.DENSITY_LIMIT),
                                help="Zones whose density reaches this limit are flagged Overcrowded.")
    with st.expander("YOLO", expanded=True):
        yolo.conf = st.slider("YOLO confidence", 0.05, 0.9, 0.2,
                              help="Minimum detection confidence. Lower finds more people but adds false hits.")
        tiles_label = st.selectbox("YOLO tiling (finds small/distant people)", ["Off (fast)", "2x2", "3x3"], index=1,
                                   help="Runs YOLO on image tiles so small, distant people are detected.")
    st.caption("U-Net weights: " + ("loaded" if unet.available else "not found (using fallback)"))

n = int(grid_label[0])
settings = Settings(
    threshold_n=threshold_n, rows=n, cols=n, cell_area_m2=cell_area, density_limit=limit,
    tiles={"Off (fast)": 1, "2x2": 2, "3x3": 3}[tiles_label],
    force={"Auto (smart switch)": "auto", "Force YOLO": "yolo", "Force density map": "density"}[mode_label],
)


# =====================================================================
# VIEW MODEL (everything shown comes from the real pipeline output)
# =====================================================================
def zname(r, c):
    return f"R{r + 1}C{c + 1}"


def build_view(frame_bgr, out, source_name, camera, s):
    res = out["result"]
    st.session_state["view_n"] += 1
    now = datetime.datetime.now().strftime("%H:%M:%S")
    return {
        "id": st.session_state["view_n"], "time": now, "source": source_name, "camera": camera,
        "mode": res.mode, "count": float(res.count),
        "orig": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        "overlay": cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB),
        "edges": out["edges"],
        "counts": np.array(out["counts"], dtype=float),
        "density": np.array(out["density"], dtype=float),
        "flags": np.array(out["flags"], dtype=bool),
        "limit": float(s.density_limit), "cell_area": float(s.cell_area_m2),
        "label": f"#{st.session_state['view_n']} · {now} · {source_name} · {int(round(res.count))} people",
    }


def add_history(view):
    hist = [h for h in st.session_state["history"] if h["id"] != view["id"]]
    st.session_state["history"] = ([view] + hist)[:HISTORY_MAX]


def zone_status(flag):
    return "Overcrowded" if flag else "Safe"


def zone_df(view):
    rows, cols = view["flags"].shape
    data = []
    for r in range(rows):
        for c in range(cols):
            d = float(view["density"][r, c])
            data.append({"Zone": zname(r, c), "People": round(float(view["counts"][r, c]), 1),
                         "People/m²": round(d, 2), "% of limit": round(100 * d / view["limit"]),
                         "Status": zone_status(bool(view["flags"][r, c]))})
    return pd.DataFrame(data)


def styled(df):
    def color(row):
        if row["Status"] == "Overcrowded":
            return ["background-color: rgba(239,68,68,.28); font-weight: 600"] * len(row)
        return [""] * len(row)
    return df.style.apply(color, axis=1)


# ---------------- HTML builders ----------------
def logo_html():
    return '<div class="logo">' + "".join(
        f'<i class="{"hot" if k in (1, 4, 5, 8) else ""}"></i>' for k in range(9)) + "</div>"


def header_html(view):
    return ('<div class="page-title">Crowd Intelligence Report</div>'
            '<div class="page-desc">Results from the AI pipeline for the analyzed image or video frame.</div>'
            f'<div class="meta"><span class="chip">File: <b>{esc(view["source"])}</b></span>'
            f'<span class="chip">Camera: <b>{esc(view["camera"])}</b></span>'
            f'<span class="chip">Detection method: <b>{esc(view["mode"])}</b></span>'
            f'<span class="chip">Analyzed at: <b>{esc(view["time"])}</b></span></div>')


def alert_html(view, over):
    if over:
        return ('<div class="alert crit"><div class="alert-title">⚠ OVER-CAPACITY ZONE DETECTED</div>'
                f'<div class="alert-body">Affected zones: <b>{esc(", ".join(over))}</b> · '
                f'Estimated people in frame: <b>{int(round(view["count"]))}</b></div>'
                f'<div class="alert-body">These zones reached the capacity limit of {view["limit"]:g} people/m². '
                'Review the highlighted zones below and take crowd-control action.</div>'
                '<div class="alert-note">The audio alarm plays when the browser allows it. If you hear nothing, the '
                'browser may have blocked autoplay: this panel is the alert.</div></div>')
    return ('<div class="alert safe"><div class="alert-title">✔ Safe</div>'
            '<div class="alert-body">No overcrowded zones detected in this analysis.</div></div>')


def summary_html(view, over):
    total = view["flags"].size
    overall, ocol = ("Overcrowded", "var(--crit-text)") if over else ("Safe", "var(--safe)")
    return (
        '<div class="sum"><div class="big"><div class="big-label">Estimated people</div>'
        f'<div class="big-num">{int(round(view["count"]))}</div><div class="big-sub">total in this frame</div></div>'
        '<div class="stats">'
        f'<div class="stat"><span class="stat-ic">◎</span><span class="stat-lbl">Detection method used</span>'
        f'<span class="stat-val">{esc(view["mode"])}</span></div>'
        f'<div class="stat"><span class="stat-ic">▦</span><span class="stat-lbl">Overcrowded zones</span>'
        f'<span class="stat-val">{len(over)} of {total}</span></div>'
        f'<div class="stat"><span class="stat-ic">◈</span><span class="stat-lbl">Overall density status</span>'
        f'<span class="stat-val" style="color:{ocol}">{overall}</span></div>'
        f'<div class="stat"><span class="stat-ic">≥</span><span class="stat-lbl">Capacity limit</span>'
        f'<span class="stat-val">{view["limit"]:g} people/m²</span></div>'
        '</div></div>')


def spatial_html(view, selected=None):
    rows, cols = view["flags"].shape
    cells = []
    for r in range(rows):
        for c in range(cols):
            d = float(view["density"][r, c])
            cnt = float(view["counts"][r, c])
            bad = bool(view["flags"][r, c])
            cls = ("bad" if bad else "ok") + (" sel" if selected == zname(r, c) else "")
            cells.append(f'<div class="zone {cls}"><div class="z-id">{zname(r, c)}</div>'
                         f'<div class="z-people">{cnt:.1f}</div><div class="z-dens">people · {d:.2f} per m²</div>'
                         f'<div class="z-stat">{zone_status(bad)}</div></div>')
    legend = ('<div class="legend"><span><span class="dot" style="background:var(--safe)"></span>Safe</span>'
              f'<span><span class="dot" style="background:var(--crit)"></span>Overcrowded (limit {view["limit"]:g} '
              'people/m²)</span></div>')
    return f'<div class="zgrid" style="grid-template-columns:repeat({cols},1fr)">{"".join(cells)}</div>{legend}'


def ranking_html(view):
    rows, cols = view["flags"].shape
    items = sorted(((zname(r, c), float(view["density"][r, c]), bool(view["flags"][r, c]))
                    for r in range(rows) for c in range(cols)), key=lambda t: -t[1])
    scale = max(view["limit"], max(d for _, d, _ in items)) * 1.05 or 1.0
    lim = view["limit"] / scale * 100
    out = ['<div class="rank">']
    for name, d, bad in items:
        out.append(f'<div class="rank-row"><span class="rk-name">{name}</span><div class="rk-track">'
                   f'<div class="rk-fill{" bad" if bad else ""}" style="width:{d / scale * 100:.1f}%"></div>'
                   f'<div class="rk-lim" style="left:{lim:.1f}%"></div></div>'
                   f'<span class="rk-val">{d:.2f}/m² · {zone_status(bad)}</span></div>')
    out.append(f'<div class="rank-note">Zones ranked by density. The vertical line marks the capacity limit '
               f'({view["limit"]:g} people/m²).</div></div>')
    return "".join(out)


def file_info(up):
    st.markdown(f'<div class="fileinfo">Selected file: <b>{esc(up.name)}</b><br>'
                f'Size: <b>{up.size / 1024:.1f} KB</b> · Type: <b>{esc(up.type or "unknown")}</b></div>',
                unsafe_allow_html=True)


def page_header(title, desc):
    st.markdown(f'<div class="page-title">{esc(title)}</div><div class="page-desc">{esc(desc)}</div>',
                unsafe_allow_html=True)


def top_bar():
    cols = st.columns([2.2, 1, 1.3, 1.3, 1.5])
    with cols[0]:
        st.markdown(f'<div class="brand">{logo_html()}<div><div class="brand-name">CROWDVISION AI</div>'
                    f'<div class="brand-sub">{TEAM} · Crowd density estimation</div></div></div>',
                    unsafe_allow_html=True)
    for col, p in zip(cols[1:], PAGES):
        with col:
            st.button(p, key=f"nav_{p}", on_click=goto, args=(p,), use_container_width=True,
                      type="primary" if p == page else "secondary")
    _, t1, t2 = st.columns([5, 1.6, 2])
    with t1:
        _toggle("Light theme", key="light_theme", help="Switch between the dark and light dashboard.")
    with t2:
        _toggle("Presentation mode", key="present_mode",
                help="Larger text and a wider layout for demonstrations. The settings sidebar is hidden while on.")


# =====================================================================
# REPORT RENDERING
# =====================================================================
def render_report(view, interactive):
    """interactive=False is used for live video frames (no widgets inside a loop)."""
    over = zone_names(view["flags"])
    st.markdown(header_html(view), unsafe_allow_html=True)
    st.markdown(alert_html(view, over), unsafe_allow_html=True)

    mode = st.session_state.get("view_mode", VIEW_MODES[0])
    if mode == VIEW_MODES[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="panel-title">Original input</div>'
                        '<div class="panel-sub">The uploaded image or current video frame.</div>',
                        unsafe_allow_html=True)
            st.image(view["orig"], use_container_width=True)
        with c2:
            st.markdown('<div class="panel-title">AI density analysis<span class="ai-label">AI-generated overlay'
                        '</span></div><div class="panel-sub">Heat map and zone grid produced by the model.</div>',
                        unsafe_allow_html=True)
            st.image(view["overlay"], use_container_width=True)
    elif mode == VIEW_MODES[1]:
        st.markdown('<div class="panel-title">AI density analysis<span class="ai-label">AI-generated overlay</span>'
                    '</div>', unsafe_allow_html=True)
        st.image(view["overlay"], use_container_width=True)
    else:
        st.markdown('<div class="panel-title">Original input</div>', unsafe_allow_html=True)
        st.image(view["orig"], use_container_width=True)

    st.markdown(summary_html(view, over), unsafe_allow_html=True)

    st.markdown('<div class="panel-title" style="font-size:1.05rem;margin-top:.6rem">Spatial density overview</div>'
                '<div class="panel-sub">People and density for each grid zone, and the zones ranked by density.</div>',
                unsafe_allow_html=True)
    rows, cols = view["flags"].shape
    names = [zname(r, c) for r in range(rows) for c in range(cols)]
    sel = None
    if interactive:
        if st.session_state.get("sel_zone") not in names:
            st.session_state["sel_zone"] = names[0]
        sel = st.selectbox("Select a zone to inspect", names, key="sel_zone")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(spatial_html(view, sel), unsafe_allow_html=True)
    with right:
        st.markdown(ranking_html(view), unsafe_allow_html=True)

    if not interactive:
        return
    i = names.index(sel)
    r, c = divmod(i, cols)
    d, cnt, bad = float(view["density"][r, c]), float(view["counts"][r, c]), bool(view["flags"][r, c])
    st.markdown(f'<div class="panel-title" style="margin-top:.8rem">Selected zone: {sel}</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("People", f"{cnt:.1f}")
    m2.metric("Density (people/m²)", f"{d:.2f}")
    m3.metric("Status", zone_status(bad))
    m4.metric("Share of capacity limit", f"{100 * d / view['limit']:.0f}%",
              help=f"Capacity limit: {view['limit']:g} people/m²")

    df = zone_df(view)
    with st.expander("View detailed zone data"):
        st.dataframe(styled(df), use_container_width=True, hide_index=True)
    with st.expander("Zone charts"):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="panel-sub">People per zone</div>', unsafe_allow_html=True)
            st.bar_chart(df.set_index("Zone")[["People"]])
        with g2:
            st.markdown(f'<div class="panel-sub">Density per zone (limit {view["limit"]:g} people/m²)</div>',
                        unsafe_allow_html=True)
            st.bar_chart(df.set_index("Zone")[["People/m²"]])
    with st.expander("Preprocessing preview (edge map)"):
        st.image(view["edges"], use_container_width=True)


# =====================================================================
# VIDEO PROCESSING (bounded batches + rerun, so Stop is real)
# =====================================================================
def start_video(up, params):
    stop_video(None)
    suffix = os.path.splitext(up.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(up.getvalue())
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    ok = cap.isOpened() and cap.read()[0]
    if not ok:
        cap.release()
        _rm(tmp.name)
        return "This video could not be opened or has no readable frames. Try another file or format."
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    st.session_state["vs"] = dict(params, running=True, cap=cap, path=tmp.name, idx=0, done=0, valid_in_pass=0,
                                  bg=BackgroundSubtractor(), prev_over=False, last_alarm=-1e9)
    st.session_state.pop("video_msg", None)
    return None


def run_video_batch(vs, live, status, alarm_slot):
    """Analyze up to BATCH_FRAMES frames, rendering each into `live`. Same trigger rules as before:
    log + alarm only when overcrowding STARTS, alarm cooldown ALARM_COOLDOWN_S."""
    yolo.conf = vs["conf"]
    processed = 0
    while vs["running"] and processed < BATCH_FRAMES:
        if not vs["loop"] and vs["done"] >= vs["max_frames"]:
            stop_video(f"Finished: {vs['done']} frames analyzed. The last analyzed frame stays on screen.", "success")
            return
        ok, frame = vs["cap"].read()
        if not ok:  # end of video
            if not vs["loop"] or vs["valid_in_pass"] == 0:
                if vs["done"] == 0:
                    stop_video("No usable frames were found (all frames too dark or unreadable).", "warning")
                else:
                    stop_video(f"Finished: {vs['done']} frames analyzed. The last analyzed frame stays on screen.",
                               "success")
                return
            vs["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)  # replay from the start
            vs["idx"] = vs["valid_in_pass"] = 0
            vs["bg"] = BackgroundSubtractor()
            vs["prev_over"] = False
            continue
        vs["idx"] += 1
        if vs["idx"] % vs["every_n"]:
            continue
        if frame.mean() < BLACK_MEAN:  # skip black / faded-out frames
            continue
        if frame.shape[1] > 960:
            sc = 960 / frame.shape[1]
            frame = cv2.resize(frame, (960, int(frame.shape[0] * sc)))

        out = analyze_frame(frame, yolo, unet, vs["settings"], vs["bg"])
        view = build_view(frame, out, vs["name"], vs["camera"], vs["settings"])
        st.session_state["last_view"] = view
        vs["done"] += 1
        vs["valid_in_pass"] += 1
        processed += 1

        now_over = bool(out["flags"].any())
        if now_over and not vs["prev_over"]:  # log + alarm only when overcrowding STARTS
            log_event(vs["camera"], vs["name"], out["result"].mode, out["result"].count, out["flags"], out["density"])
            if time.monotonic() - vs["last_alarm"] > ALARM_COOLDOWN_S:
                vs["last_alarm"] = time.monotonic()
                fire_alarm(zone_names(out["flags"]), out["result"].count, "real", alarm_slot)
        vs["prev_over"] = now_over

        status.caption(f"Analyzing: frame {vs['done']} of {vs['max_frames']}" if not vs["loop"]
                       else f"Analyzing continuously: {vs['done']} frames analyzed. Press Stop analysis to end.")
        with live.container():
            render_report(view, interactive=False)


# =====================================================================
# PAGE: HOME
# =====================================================================
def page_home():
    top_bar()
    left, right = st.columns([6, 5], gap="large")
    with left:
        st.markdown(
            f'<div class="hero-title">{esc(PROJECT_TITLE)}</div>'
            '<div class="hero-sub">See the crowd. Understand the density. Act before it becomes critical.</div>'
            f'<div class="hero-team"><b>{TEAM}</b><br>{esc(PROGRAMME)}</div>'
            '<div class="hero-text">Computer vision estimates how many people are in an uploaded image or video, '
            'measures density zone by zone to find the crowded areas, and raises an alert, with an audio alarm and '
            'an event log, when a zone reaches the capacity limit.</div>', unsafe_allow_html=True)
        st.button("Start Monitoring", type="primary", on_click=goto, args=("Crowd Analysis",))
    with right:
        # decorative illustration only: it is not data
        shades = ["#0E3A4A", "#0F5A63", "#14B8A6", "#F59E0B", "#EF4444"]
        pattern = [0, 0, 1, 1, 0, 0, 0, 0,
                   0, 1, 2, 2, 1, 0, 1, 0,
                   1, 2, 3, 4, 2, 1, 1, 0,
                   0, 2, 3, 3, 2, 2, 1, 0,
                   0, 1, 2, 2, 1, 1, 0, 0]
        cells = "".join(f'<i style="background:{shades[v]}"></i>' for v in pattern)
        st.markdown(f'<div class="heat" style="margin-top:1.5rem">{cells}</div>'
                    '<div class="heat-cap">Decorative illustration of a zone-wise heat map. Not real data.</div>',
                    unsafe_allow_html=True)
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    feats = [
        ("AI-powered crowd counting", "YOLOv8 counts individuals in sparse crowds and a U-Net density map handles dense ones."),
        ("Spatial density mapping", "The frame is split into a grid so you can see and rank the busiest zones."),
        ("Overcrowding alerts", "Zones at the limit trigger an alert, an audio alarm and a saved log entry."),
    ]
    for col, (t, d) in zip((f1, f2, f3), feats):
        with col:
            st.markdown(f'<div class="feat"><h4>{esc(t)}</h4><p>{esc(d)}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="foot">Academic demonstration project · analyses uploaded images and videos, '
                'not a live camera feed.</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE: CROWD ANALYSIS
# =====================================================================
def decode_image(up):
    arr = np.frombuffer(up.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


def page_analysis():
    top_bar()
    page_header("Crowd Analysis", "Upload an image or a video, run the analysis, then explore the report below.")
    alarm_slot = st.empty()
    render_alarm(alarm_slot)

    # ---- alarm panel ----
    with st.container(border=True):
        st.markdown('<div class="panel-title">Alarm</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns([4, 1.2, 1.2])
        with a1:
            st.caption(alarm_status_text())
        with a2:
            st.button("Test alarm", on_click=test_alarm_cb, use_container_width=True,
                      help="Plays the siren and banner with a clearly labelled test message. Nothing is logged.")
        with a3:
            st.button("Stop alarm", on_click=stop_alarm_cb, use_container_width=True,
                      help="Silences the siren and removes the banner.")
        st.caption("If you hear no sound, the browser blocked autoplay: click anywhere on the page once, "
                   "then press Test alarm again. Pressing STOP on the banner itself also works, but this panel "
                   "cannot detect that.")

    # ---- input panel ----
    analyze = False
    start = False
    with st.container(border=True):
        st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)
        source = st.radio("Input type", ["Image", "Video"], horizontal=True)
        frame = None
        up = None
        if source == "Image":
            stop_video("Video analysis was stopped because the input type was changed.")
            up = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])
            st.caption("Supported formats: JPG, JPEG, PNG.")
            if up is not None:
                file_info(up)
                frame = decode_image(up)
                if frame is None:
                    st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
                else:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=360,
                             caption=f"Preview · {frame.shape[1]}×{frame.shape[0]} px")
            analyze = st.button("Analyze Crowd", type="primary", disabled=(frame is None))
        else:
            up = st.file_uploader("Upload a crowd video", type=["mp4", "avi", "mov", "mkv"])
            st.caption("Supported formats: MP4, AVI, MOV, MKV.")
            if up is not None:
                file_info(up)
                if up.type in ("video/mp4", "video/quicktime", "video/webm"):
                    with st.expander("Preview video"):
                        st.video(up.getvalue())
            v1, v2 = st.columns(2)
            with v1:
                every_n = st.slider("Analyze every Nth frame", 1, 60, 10,
                                    help="Higher values are faster but check fewer frames.")
            with v2:
                max_frames = st.slider("Max frames to analyze (ignored when looping)", 5, 300, 60)
            loop_video = st.checkbox("Keep running: replay the video until I press Stop", value=True)
            vs0 = st.session_state.get("vs")
            if vs0 and vs0.get("running"):
                st.button("■ Stop analysis", type="primary", on_click=stop_cb, key="btn_stop",
                          help="Ends the run, releases the video and removes the temporary file.")
                st.caption("This run keeps the settings it started with. Changes apply to the next run.")
            else:
                start = st.button("Start analysis", type="primary", disabled=(up is None))

    # ---- actions ----
    if analyze and frame is not None:
        with st.spinner("Analyzing image..."):
            try:
                out = analyze_frame(frame, yolo, unet, settings)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                out = None
        if out is not None:
            view = build_view(frame, out, up.name, camera_id, settings)
            st.session_state["last_view"] = view
            add_history(view)
            st.session_state.pop("hist_pick", None)
            st.session_state.pop("video_msg", None)
            if out["flags"].any():
                log_event(camera_id, up.name, out["result"].mode, out["result"].count, out["flags"], out["density"])
                fire_alarm(zone_names(out["flags"]), out["result"].count, "real", alarm_slot)
            st.success("Analysis complete.")

    if start and up is not None:
        err = start_video(up, {"name": up.name, "camera": camera_id, "every_n": every_n, "max_frames": max_frames,
                               "loop": loop_video, "settings": settings, "conf": float(yolo.conf)})
        if err:
            st.error(err)
        else:
            st.session_state.pop("hist_pick", None)
            do_rerun()

    vs = st.session_state.get("vs")
    running = bool(vs and vs.get("running"))

    # ---- result controls ----
    hist = st.session_state["history"]
    chosen = None
    k1, k2 = st.columns([3, 3])
    with k1:
        st.radio("Image view", VIEW_MODES, horizontal=True, key="view_mode",
                 help="Original and AI overlay side by side, or either one full width for presentations.")
    with k2:
        if hist and not running:
            labels = [h["label"] for h in hist]
            pick = st.selectbox("Recent analyses (this session only)", ["Latest result"] + labels, key="hist_pick")
            if pick != "Latest result":
                chosen = hist[labels.index(pick)]

    # ---- results ----
    if running:
        status = st.empty()
        live = st.empty()
        last = st.session_state.get("last_view")
        if last is not None and vs["done"] > 0:
            with live.container():
                render_report(last, interactive=False)
        run_video_batch(vs, live, status, alarm_slot)
        do_rerun()  # next batch, or refresh the page once the run has ended
    else:
        msg = st.session_state.pop("video_msg", None)
        if msg:
            getattr(st, msg[0])(msg[1])
        view = chosen or st.session_state.get("last_view")
        if view is not None:
            if chosen is not None:
                st.caption("Showing an earlier analysis from this session.")
            render_report(view, interactive=True)
        else:
            st.markdown('<div class="empty">No analysis yet. Upload an image or video to see the Crowd '
                        'Intelligence Report.</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE: EVENT HISTORY
# =====================================================================
def load_events():
    try:
        raw = read_log()
    except Exception as exc:
        st.error(f"The event log could not be read: {exc}")
        return pd.DataFrame(columns=EVENT_COLUMNS + ["_zones_n"]), []
    df = raw.copy()
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ("total_count", "max_density_per_m2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("camera_id", "source", "mode", "overloaded_zones"):
        df[c] = df[c].fillna("").astype(str)
    df["_zones_n"] = df["overloaded_zones"].apply(lambda s: len([z for z in s.split(";") if z.strip()]))
    return df, missing


def page_history():
    top_bar()
    page_header("Event History", "Overcrowding events recorded by the system. Filtering never changes the log file.")
    df, missing = load_events()
    if len(df) == 0:
        st.markdown('<div class="empty">No overcrowding events recorded yet. Events appear here when a zone '
                    'reaches the capacity limit.</div>', unsafe_allow_html=True)
        return
    if missing:
        st.info("The log has no column for: " + ", ".join(missing) + ". Related filters and charts are limited.")

    mask = pd.Series(True, index=df.index)
    with st.container(border=True):
        st.markdown('<div class="panel-title">Search and filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            q = st.text_input("Search", placeholder="camera, file, zone, method...")
        cams = sorted(c for c in df["camera_id"].unique() if c)
        srcs = sorted(c for c in df["source"].unique() if c)
        with f2:
            cam_sel = st.multiselect("Camera", cams, default=[]) if cams else []
        with f3:
            src_sel = st.multiselect("Source file", srcs, default=[]) if srcs else []
        g1, g2, g3 = st.columns(3)
        with g1:
            max_z = int(df["_zones_n"].max())
            min_z = st.slider("Minimum overcrowded zones", 1, max_z, 1) if max_z > 1 else 1
        with g2:
            dmax = df["max_density_per_m2"].max()
            min_d = (st.slider("Minimum peak density (people/m²)", 0.0, float(round(dmax, 2)), 0.0)
                     if pd.notna(dmax) and dmax > 0 else 0.0)
        with g3:
            valid_ts = df["timestamp"].dropna()
            rng = None
            if len(valid_ts):
                lo, hi = valid_ts.min().date(), valid_ts.max().date()
                rng = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)
        h1, h2 = st.columns(2)
        sort_map = {"Time": "timestamp", "People": "total_count", "Peak density": "max_density_per_m2",
                    "Overcrowded zones": "_zones_n"}
        with h1:
            sort_label = st.selectbox("Sort by", list(sort_map))
        with h2:
            order = st.radio("Order", ["Highest / newest first", "Lowest / oldest first"], horizontal=True)

    if q:
        hay = df[["timestamp", "camera_id", "source", "mode", "overloaded_zones"]].astype(str).agg(" ".join, axis=1)
        mask &= hay.str.contains(q, case=False, regex=False)
    if cam_sel:
        mask &= df["camera_id"].isin(cam_sel)
    if src_sel:
        mask &= df["source"].isin(src_sel)
    mask &= df["_zones_n"] >= min_z
    if min_d > 0:
        mask &= df["max_density_per_m2"].fillna(0) >= min_d
    if rng is not None and isinstance(rng, (tuple, list)) and len(rng) == 2 and (rng[0], rng[1]) != (lo, hi):
        d = df["timestamp"].dt.date
        mask &= (d >= rng[0]) & (d <= rng[1])

    f = df[mask].sort_values(sort_map[sort_label], ascending=order.startswith("Lowest"), na_position="last")
    st.markdown(f'<div class="meta"><span class="chip">Matching events: <b>{len(f)}</b> of {len(df)}</span></div>',
                unsafe_allow_html=True)
    if len(f) == 0:
        st.markdown('<div class="empty">No events match these filters.</div>', unsafe_allow_html=True)
        return

    shown = f.drop(columns=["_zones_n"]).copy()
    shown["timestamp"] = shown["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    tab_t, tab_c = st.tabs(["Events", "Analytics"])
    with tab_t:
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download filtered log (CSV)", shown.to_csv(index=False), "events_filtered.csv", "text/csv")
        st.markdown('<div class="panel-title" style="margin-top:.8rem">Event details</div>', unsafe_allow_html=True)
        ids = f.index.tolist()
        pick = st.selectbox("Select an event", ids, format_func=lambda i: (
            f"{shown.loc[i, 'timestamp'] or 'unknown time'} · {shown.loc[i, 'source'] or 'unknown source'} · "
            f"{shown.loc[i, 'overloaded_zones'] or 'no zones recorded'}"))
        r = shown.loc[pick]
        pk = r["max_density_per_m2"]
        st.markdown(
            '<div class="card"><div class="kv">'
            f'<span>Time</span><span>{esc(r["timestamp"] or "n/a")}</span>'
            f'<span>Camera</span><span>{esc(r["camera_id"] or "n/a")}</span>'
            f'<span>Source</span><span>{esc(r["source"] or "n/a")}</span>'
            f'<span>Detection method</span><span>{esc(r["mode"] or "n/a")}</span>'
            f'<span>Estimated people</span><span>{"n/a" if pd.isna(r["total_count"]) else int(r["total_count"])}</span>'
            f'<span>Overcrowded zones</span><span>{esc(r["overloaded_zones"].replace(";", ", ") or "n/a")}</span>'
            f'<span>Peak density</span><span>{"n/a" if pd.isna(pk) else f"{pk:.2f} people/m²"}</span>'
            '</div></div>', unsafe_allow_html=True)
    with tab_c:
        st.caption("Charts use only the events that match the current filters.")
        c1, c2 = st.columns(2)
        ts = f["timestamp"].dropna()
        with c1:
            if len(ts):
                st.markdown('<div class="panel-sub">Events per day</div>', unsafe_allow_html=True)
                st.bar_chart(ts.dt.strftime("%Y-%m-%d").value_counts().sort_index())
            zones = f["overloaded_zones"].str.split(";").explode().str.strip()
            zones = zones[zones != ""]
            if len(zones):
                st.markdown('<div class="panel-sub">How often each zone was overcrowded</div>', unsafe_allow_html=True)
                st.bar_chart(zones.value_counts().sort_index())
        with c2:
            cams_ = f["camera_id"][f["camera_id"] != ""]
            if len(cams_):
                st.markdown('<div class="panel-sub">Events per camera</div>', unsafe_allow_html=True)
                st.bar_chart(cams_.value_counts())
            dens = f.dropna(subset=["timestamp", "max_density_per_m2"]).sort_values("timestamp")
            if len(dens):
                st.markdown('<div class="panel-sub">Peak density per event (people/m²)</div>', unsafe_allow_html=True)
                st.line_chart(dens.set_index("timestamp")["max_density_per_m2"])


# =====================================================================
# PAGE: ABOUT
# =====================================================================
def page_about():
    top_bar()
    page_header("About the Project", "How CROWDVISION AI turns an image or video into zone-wise crowd density.")
    st.markdown(
        f'<div class="card"><h4>{esc(PROJECT_TITLE)}</h4><p><b>{TEAM}</b> · {esc(PROGRAMME)}<br><br>'
        'An automated crowd density estimation and management system built as an academic demonstration. It analyses '
        'uploaded images and videos; it is not connected to a live camera or CCTV feed.</p></div>',
        unsafe_allow_html=True)
    steps = [
        ("Image / video input", "A JPG, PNG or video file is uploaded."),
        ("Preprocessing", "Gaussian blur and Canny edges; MOG2 background subtraction for video."),
        ("Smart Density Switch", "Auto mode picks the method using the switch threshold N."),
        ("YOLOv8 (sparse)", "Detects each person as a separate object; tiling finds small, distant people."),
        ("U-Net density map (dense)", "Predicts a density map whose sum is the estimated count."),
        ("Spatial grid", "The frame is split into a 3x3 or 4x4 grid; each zone gets a count and a density."),
        ("Alerts and event log", "Zones at the capacity limit raise an alert, an alarm and a CSV log entry."),
    ]
    st.markdown('<div class="flow">' + "".join(
        f'<div class="step"><b>{i + 1}. {esc(t)}</b><p>{esc(d)}</p></div>' for i, (t, d) in enumerate(steps)
    ) + '</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><h4>Sparse crowds: YOLOv8</h4><p>YOLOv8 finds each person as a separate '
                    'object and counts the detections. Tiling splits the image so small, distant people can be '
                    'found.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>Dense crowds: U-Net density map</h4><p>When people overlap, the U-Net '
                    'predicts a density map in which each pixel is a share of a person. Summing the map gives the '
                    'estimated count.</p></div>', unsafe_allow_html=True)


# =====================================================================
# ROUTER
# =====================================================================
if page == "Home":
    page_home()
elif page == "Crowd Analysis":
    page_analysis()
elif page == "Event History":
    page_history()
else:
    page_about()