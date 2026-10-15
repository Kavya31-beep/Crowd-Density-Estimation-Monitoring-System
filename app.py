"""Streamlit dashboard. Run with:  streamlit run app.py"""
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

from src import config
from src.alerts import alarm_component_html, log_event, read_log, zone_names
from src.density_unet import UNetDensity
from src.detector_yolo import YoloCounter
from src.pipeline import Settings, analyze_frame
from src.preprocessing import BackgroundSubtractor

BLACK_MEAN = 8          # frames darker than this (average pixel value) are skipped
ALARM_COOLDOWN_S = 30   # in loop mode, do not re-ring the alarm more often than this

st.set_page_config(page_title="Crowd Density Monitor", layout="wide")

# ---------------- styling + header (presentation only) ----------------
st.markdown(
    """
<style>
.block-container {padding-top: 4.5rem; padding-bottom: 2rem; max-width: 1400px;}
footer {visibility: hidden;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.25);}
[data-testid="stImage"] img {border-radius: 12px;}
[data-testid="stFileUploader"] section {border-radius: 12px; border: 1.5px dashed rgba(128,128,128,.5);}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 600; padding: .5rem 1.2rem;}
button[data-baseweb="tab"] {font-weight: 600; padding: .6rem 1.1rem;}
.hero {display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .8rem;
       padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem; color: #fff;
       background: linear-gradient(135deg, #1D4ED8 0%, #0EA5E9 100%);}
.hero h1 {margin: 0; padding: 0; font-size: 1.6rem; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .9; font-size: .95rem;}
.chips span {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); padding: .2rem .75rem;
             border-radius: 999px; font-size: .78rem; margin-left: .4rem; white-space: nowrap;}
.side-label {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7;
             margin: .9rem 0 .2rem; font-weight: 700;}
.section-title {font-size: 1.05rem; font-weight: 600; margin: 1rem 0 .5rem;}
.banner {padding: .8rem 1.2rem; border-radius: 12px; margin: .8rem 0; color: #fff; font-weight: 600;}
.kpi-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem; margin: .6rem 0;}
.kpi {background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
      padding: .8rem 1rem; transition: transform .15s, box-shadow .15s;}
.kpi:hover {transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.25);}
.kpi-label {font-size: .8rem; opacity: .75; text-transform: uppercase; letter-spacing: .05em;}
.kpi-value {font-size: 2rem; font-weight: 700; line-height: 1.25;}
.kpi-sub {font-size: .78rem; opacity: .7;}
.tile-grid {display: grid; gap: .5rem;}
.tile {border-radius: 10px; padding: .7rem .4rem; text-align: center; color: #fff;
       transition: transform .15s, box-shadow .15s;}
.tile:hover {transform: scale(1.04); box-shadow: 0 6px 14px rgba(0,0,0,.3);}
.tile.ok {background: #15803D;}
.tile.warn {background: #B45309;}
.tile.bad {background: #B91C1C;}
.tile-name {font-size: .75rem; opacity: .85; letter-spacing: .04em;}
.tile-val {font-size: 1.35rem; font-weight: 700; line-height: 1.3;}
.tile-sub {font-size: .72rem; opacity: .85;}
.legend {display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .7rem; font-size: .78rem; opacity: .85;}
.dot {display: inline-block; width: .6rem; height: .6rem; border-radius: 50%; margin-right: .35rem;}
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem; margin: .8rem 0;}
.step {background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
       padding: 1rem;}
.step-no {width: 1.8rem; height: 1.8rem; border-radius: 50%; background: #2563EB; color: #fff; display: flex;
          align-items: center; justify-content: center; font-weight: 700; margin-bottom: .5rem;}
.step-title {font-weight: 600; margin-bottom: .25rem;}
.step-text {font-size: .85rem; opacity: .8;}
</style>
<div class="hero">
<div><h1>Crowd Density Monitor</h1>
<p>Automated crowd density estimation &amp; management system · Group 8</p></div>
<div class="chips"><span>YOLOv8</span><span>U-Net</span><span>OpenCV</span><span>Streamlit</span></div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return YoloCounter(config.YOLO_WEIGHTS), UNetDensity(config.UNET_WEIGHTS)


yolo, unet = load_models()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    st.markdown('<div class="side-label">Camera</div>', unsafe_allow_html=True)
    camera_id = st.text_input("Camera ID", "CAM-01")
    st.markdown('<div class="side-label">Detection</div>', unsafe_allow_html=True)
    mode_label = st.selectbox("Detection mode", ["Auto (smart switch)", "Force YOLO", "Force density map"])
    threshold_n = st.slider("Switch threshold N (people)", 5, 200, config.SWITCH_THRESHOLD_N)
    st.markdown('<div class="side-label">Grid and capacity</div>', unsafe_allow_html=True)
    grid_label = st.selectbox("Grid", ["3x3", "4x4"])
    cell_area = st.number_input("Area covered by one grid cell (m²)", 0.5, 500.0, config.CELL_AREA_M2)
    limit = st.number_input("Capacity limit (people/m²)", 0.5, 10.0, config.DENSITY_LIMIT)
    st.markdown('<div class="side-label">YOLO</div>', unsafe_allow_html=True)
    yolo.conf = st.slider("YOLO confidence", 0.05, 0.9, 0.2)
    tiles_label = st.selectbox("YOLO tiling (finds small/distant people)", ["Off (fast)", "2x2", "3x3"], index=1)
    st.caption("U-Net weights: " + ("loaded" if unet.available else "not found (using fallback)"))

n = int(grid_label[0])
settings = Settings(
    threshold_n=threshold_n, rows=n, cols=n, cell_area_m2=cell_area, density_limit=limit,
    tiles={"Off (fast)": 1, "2x2": 2, "3x3": 3}[tiles_label],
    force={"Auto (smart switch)": "auto", "Force YOLO": "yolo", "Force density map": "density"}[mode_label],
)


# ---------------- display helpers (presentation only) ----------------
def zone_table(out):
    rows, cols = out["flags"].shape
    data = []
    for r in range(rows):
        for c in range(cols):
            data.append({"Zone": f"R{r + 1}C{c + 1}", "People": round(float(out["counts"][r, c]), 1),
                         "People/m²": round(float(out["density"][r, c]), 2),
                         "Status": "OVERCROWDED" if out["flags"][r, c] else "Safe"})
    return pd.DataFrame(data)


def styled(df):
    """Display-only: highlight overcrowded rows."""
    def color(row):
        if row["Status"] == "OVERCROWDED":
            return ["background-color: rgba(239,68,68,.28); font-weight: 600"] * len(row)
        return [""] * len(row)
    return df.style.apply(color, axis=1)


def banner_html(over):
    if over:
        bg = "linear-gradient(135deg,#B91C1C,#EF4444)"
        text = f"⚠ {len(over)} overcrowded zone(s): " + ", ".join(over)
    else:
        bg = "linear-gradient(135deg,#15803D,#22C55E)"
        text = "✔ All zones within the capacity limit"
    return f'<div class="banner" style="background:{bg}">{text}</div>'


def kpi_html(out, limit_value):
    res = out["result"]
    n_over = len(zone_names(out["flags"]))
    cards = [
        ("Estimated people", f"{int(round(res.count))}", "total in frame", "#2563EB"),
        ("Method used", str(res.mode), "detection method", "#0EA5E9"),
        ("Overcrowded zones", str(n_over), f"of {out['flags'].size} zones", "#EF4444" if n_over else "#22C55E"),
        ("Peak density", f"{float(out['density'].max()):.2f}", f"people/m² · limit {limit_value:g}", "#F59E0B"),
    ]
    items = "".join(
        f'<div class="kpi" style="border-top:4px solid {c}"><div class="kpi-label">{a}</div>'
        f'<div class="kpi-value">{b}</div><div class="kpi-sub">{s}</div></div>'
        for a, b, s, c in cards
    )
    return f'<div class="kpi-row">{items}</div>'


def tiles_html(out, limit_value):
    rows, cols = out["flags"].shape
    cells = []
    for r in range(rows):
        for c in range(cols):
            d = float(out["density"][r, c])
            cnt = float(out["counts"][r, c])
            if out["flags"][r, c]:
                cls = "bad"
            elif d >= 0.7 * limit_value:
                cls = "warn"
            else:
                cls = "ok"
            cells.append(f'<div class="tile {cls}"><div class="tile-name">R{r + 1}C{c + 1}</div>'
                         f'<div class="tile-val">{d:.2f}</div><div class="tile-sub">{cnt:.1f} people</div></div>')
    legend = ('<div class="legend">'
              '<span><span class="dot" style="background:#15803D"></span>Safe</span>'
              '<span><span class="dot" style="background:#B45309"></span>Near limit (70% or more)</span>'
              '<span><span class="dot" style="background:#B91C1C"></span>Overcrowded</span></div>')
    return f'<div class="tile-grid" style="grid-template-columns:repeat({cols},1fr)">{"".join(cells)}</div>{legend}'


def make_slots():
    """Create the result area in order: banner, KPI cards, zone map + zone table."""
    banner_t = st.empty()
    kpi_t = st.empty()
    left, right = st.columns([2, 3])
    with left:
        st.markdown('<div class="section-title">Zone map (people/m²)</div>', unsafe_allow_html=True)
        tiles_t = st.empty()
    with right:
        st.markdown('<div class="section-title">Zone details</div>', unsafe_allow_html=True)
        table_t = st.empty()
    return banner_t, kpi_t, tiles_t, table_t


def render_panels(out, banner_t, kpi_t, tiles_t, table_t):
    banner_t.markdown(banner_html(zone_names(out["flags"])), unsafe_allow_html=True)
    kpi_t.markdown(kpi_html(out, limit), unsafe_allow_html=True)
    tiles_t.markdown(tiles_html(out, limit), unsafe_allow_html=True)
    table_t.dataframe(styled(zone_table(out)), use_container_width=True, hide_index=True)


def show_image_result(frame, out):
    res = out["result"]
    over = zone_names(out["flags"])
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    c2.image(cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB), caption="Heat map + zones",
             use_container_width=True)
    render_panels(out, *make_slots())
    if over:
        st_html(alarm_component_html(over, res.count), height=0)


# ---------------- tabs ----------------
tab_run, tab_log, tab_about = st.tabs(["Analyze", "Event log", "About"])

with tab_run:
    source = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if source == "Image":
        up = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])
        if up is not None:
            frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Analyzing..."):
                out = analyze_frame(frame, yolo, unet, settings)
            show_image_result(frame, out)
            if out["flags"].any():
                log_event(camera_id, up.name, out["result"].mode, out["result"].count, out["flags"], out["density"])
            with st.expander("Preprocessing preview (edge map)"):
                st.image(out["edges"], use_container_width=True)
    else:
        up = st.file_uploader("Upload a crowd video", type=["mp4", "avi", "mov", "mkv"])
        every_n = st.slider("Analyze every Nth frame", 1, 60, 10)
        max_frames = st.slider("Max frames to analyze (ignored when looping)", 5, 300, 60)
        loop_video = st.checkbox("Keep running: replay the video until I stop it", value=True)

        if up is not None and st.button("Start analysis"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read())
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            bg = BackgroundSubtractor()
            img_ph = st.empty()
            slots = make_slots()
            alarm_ph = st.empty()
            idx = done = valid_in_pass = 0
            prev_over = False
            last_alarm = -1e9

            while loop_video or done < max_frames:
                ok, frame = cap.read()
                if not ok:  # end of video
                    if not loop_video or valid_in_pass == 0:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # replay from the start
                    idx = valid_in_pass = 0
                    bg = BackgroundSubtractor()
                    prev_over = False
                    continue
                idx += 1
                if idx % every_n:
                    continue
                if frame.mean() < BLACK_MEAN:  # skip black / faded-out frames
                    continue
                if frame.shape[1] > 960:
                    s = 960 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * s)))

                out = analyze_frame(frame, yolo, unet, settings, bg)
                overlay_rgb = cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB)
                img_ph.image(overlay_rgb, use_container_width=True)
                render_panels(out, *slots)

                # keep the latest good result so it stays visible after the run ends
                st.session_state["last_video2"] = {
                    "img": overlay_rgb,
                    "banner": banner_html(zone_names(out["flags"])),
                    "kpi": kpi_html(out, limit),
                    "tiles": tiles_html(out, limit),
                    "table": zone_table(out),
                }

                now_over = bool(out["flags"].any())
                if now_over and not prev_over:  # log + alarm only when overcrowding STARTS
                    log_event(camera_id, up.name, out["result"].mode, out["result"].count,
                              out["flags"], out["density"])
                    if time.monotonic() - last_alarm > ALARM_COOLDOWN_S:
                        last_alarm = time.monotonic()
                        with alarm_ph.container():
                            st_html(alarm_component_html(zone_names(out["flags"]), out["result"].count),
                                    height=0)
                prev_over = now_over

                done += 1
                valid_in_pass += 1

            cap.release()
            if not loop_video:
                st.success(f"Finished: {done} frames analyzed. The last analyzed frame stays on screen.")

        elif "last_video2" in st.session_state:
            # after a rerun (slider change etc.) show the last analyzed frame instead of a blank page
            lv = st.session_state["last_video2"]
            st.image(lv["img"], caption="Last analyzed frame", use_container_width=True)
            banner_t, kpi_t, tiles_t, table_t = make_slots()
            banner_t.markdown(lv["banner"], unsafe_allow_html=True)
            kpi_t.markdown(lv["kpi"], unsafe_allow_html=True)
            tiles_t.markdown(lv["tiles"], unsafe_allow_html=True)
            table_t.dataframe(styled(lv["table"]), use_container_width=True, hide_index=True)

with tab_log:
    df = read_log()
    st.markdown('<div class="section-title">Overcrowding events</div>', unsafe_allow_html=True)
    if len(df) == 0:
        st.info("No overcrowding events recorded yet.")
    else:
        st.caption(f"{len(df)} event(s) recorded")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(df):
        st.download_button("Download log (CSV)", df.to_csv(index=False), "events.csv", "text/csv")

with tab_about:
    st.markdown('<div class="section-title">How the system works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Input", "Image or video from a CCTV camera"),
        ("2", "Preprocessing", "Gaussian blur, Canny edges and MOG2 background subtraction"),
        ("3", "Smart density switch", "YOLOv8 for sparse crowds, U-Net density map for dense crowds"),
        ("4", "Spatial grid", "3x3 or 4x4 zones with people/m² for each zone"),
        ("5", "Alerts", "Dashboard banner, audio alarm and CSV event log"),
    ]
    cards = "".join(
        f'<div class="step"><div class="step-no">{a}</div><div class="step-title">{b}</div>'
        f'<div class="step-text">{c}</div></div>' for a, b, c in steps
    )
    st.markdown(f'<div class="steps">{cards}</div>', unsafe_allow_html=True)
    st.markdown(
        "**Reading the zone map:** a zone turns red when its density is above the capacity limit set in the "
        "sidebar (default 4 people/m²). Amber marks zones at 70% of the limit or more, as an early warning."
    )
    st.caption("Group 8 · BTech Computer Science · Automated Crowd Density Estimation & Management System")