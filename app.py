"""Streamlit dashboard. Run with:  streamlit run app.py"""
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.alerts import log_event, make_alarm_wav, read_log, zone_names
from src.density_unet import UNetDensity
from src.detector_yolo import YoloCounter
from src.pipeline import Settings, analyze_frame
from src.preprocessing import BackgroundSubtractor

st.set_page_config(page_title="Crowd Density Monitor", layout="wide")
st.title("Automated Crowd Density Estimation & Management System")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return YoloCounter(config.YOLO_WEIGHTS), UNetDensity(config.UNET_WEIGHTS)


yolo, unet = load_models()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    camera_id = st.text_input("Camera ID", "CAM-01")
    mode_label = st.selectbox("Detection mode", ["Auto (smart switch)", "Force YOLO", "Force density map"])
    threshold_n = st.slider("Switch threshold N (people)", 5, 200, config.SWITCH_THRESHOLD_N)
    grid_label = st.selectbox("Grid", ["3x3", "4x4"])
    cell_area = st.number_input("Area covered by one grid cell (m²)", 0.5, 500.0, config.CELL_AREA_M2)
    limit = st.number_input("Capacity limit (people/m²)", 0.5, 10.0, config.DENSITY_LIMIT)
    yolo.conf = st.slider("YOLO confidence", 0.05, 0.9, 0.2)
    tiles_label = st.selectbox("YOLO tiling (finds small/distant people)", ["Off (fast)", "2x2", "3x3"], index=1)
    st.caption("U-Net weights: " + ("loaded" if unet.available else "not found (using fallback)"))

n = int(grid_label[0])
settings = Settings(
    threshold_n=threshold_n, rows=n, cols=n, cell_area_m2=cell_area, density_limit=limit,
    tiles={"Off (fast)": 1, "2x2": 2, "3x3": 3}[tiles_label],
    force={"Auto (smart switch)": "auto", "Force YOLO": "yolo", "Force density map": "density"}[mode_label],
)


def zone_table(out):
    rows, cols = out["flags"].shape
    data = []
    for r in range(rows):
        for c in range(cols):
            data.append({"Zone": f"R{r + 1}C{c + 1}", "People": round(float(out["counts"][r, c]), 1),
                         "People/m²": round(float(out["density"][r, c]), 2),
                         "Status": "OVERCROWDED" if out["flags"][r, c] else "Safe"})
    return pd.DataFrame(data)


def show_result(frame, out, placeholders=None):
    res = out["result"]
    over = zone_names(out["flags"])
    if placeholders is None:
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        c2.image(cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB), caption="Heat map + zones",
                 use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated people", int(round(res.count)))
        m2.metric("Method used", res.mode)
        m3.metric("Overcrowded zones", len(over))
        st.dataframe(zone_table(out), use_container_width=True, hide_index=True)
        if over:
            st.error("ALERT: overcrowded zones " + ", ".join(over))
            st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)
    else:
        img_ph, info_ph, table_ph = placeholders
        img_ph.image(cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
        info_ph.write(f"**People:** {int(round(res.count))}  |  **Method:** {res.mode}  |  "
                      f"**Overcrowded zones:** {', '.join(over) if over else 'none'}")
        table_ph.dataframe(zone_table(out), use_container_width=True, hide_index=True)


tab_run, tab_log, tab_about = st.tabs(["Analyze", "Event log", "About"])

with tab_run:
    source = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if source == "Image":
        up = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])
        if up is not None:
            frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Analyzing..."):
                out = analyze_frame(frame, yolo, unet, settings)
            show_result(frame, out)
            if out["flags"].any():
                log_event(camera_id, up.name, out["result"].mode, out["result"].count, out["flags"], out["density"])
            with st.expander("Preprocessing preview (edge map)"):
                st.image(out["edges"], use_container_width=True)
    else:
        up = st.file_uploader("Upload a crowd video", type=["mp4", "avi", "mov", "mkv"])
        every_n = st.slider("Analyze every Nth frame", 1, 60, 10)
        max_frames = st.slider("Max frames to analyze", 5, 300, 60)
        if up is not None and st.button("Start analysis"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read())
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            bg = BackgroundSubtractor()
            img_ph, info_ph, table_ph, alarm_ph = st.empty(), st.empty(), st.empty(), st.empty()
            idx = done = 0
            alarm_played = False
            progress = st.progress(0.0)
            while cap.isOpened() and done < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % every_n:
                    continue
                if frame.shape[1] > 960:
                    s = 960 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * s)))
                out = analyze_frame(frame, yolo, unet, settings, bg)
                show_result(frame, out, (img_ph, info_ph, table_ph))
                if out["flags"].any():
                    log_event(camera_id, up.name, out["result"].mode, out["result"].count,
                              out["flags"], out["density"])
                    if not alarm_played:
                        alarm_ph.audio(make_alarm_wav(), format="audio/wav", autoplay=True)
                        alarm_played = True
                done += 1
                progress.progress(done / max_frames)
            cap.release()
            st.success(f"Finished: {done} frames analyzed.")

with tab_log:
    df = read_log()
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(df):
        st.download_button("Download log (CSV)", df.to_csv(index=False), "events.csv", "text/csv")

with tab_about:
    st.markdown(
        "**Pipeline:** Video/Image → Preprocessing → Smart Density Switch (YOLO or U-Net) → "
        "Spatial Grid → Dashboard + Alarm + Event log."
    )
