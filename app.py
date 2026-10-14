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


def info_text(out):
    res = out["result"]
    over = zone_names(out["flags"])
    return (f"**People:** {int(round(res.count))}  |  **Method:** {res.mode}  |  "
            f"**Overcrowded zones:** {', '.join(over) if over else 'none'}")


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
            st_html(alarm_component_html(over, res.count), height=0)
    else:
        img_ph, info_ph, table_ph = placeholders
        img_ph.image(cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
        info_ph.write(info_text(out))
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
        max_frames = st.slider("Max frames to analyze (ignored when looping)", 5, 300, 60)
        loop_video = st.checkbox("Keep running: replay the video until I stop it", value=True)

        if up is not None and st.button("Start analysis"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read())
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            bg = BackgroundSubtractor()
            img_ph, info_ph, table_ph, alarm_ph = st.empty(), st.empty(), st.empty(), st.empty()
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
                show_result(frame, out, (img_ph, info_ph, table_ph))

                # keep the latest good result so it stays visible after the run ends
                st.session_state["last_video"] = {
                    "img": cv2.cvtColor(out["overlay"], cv2.COLOR_BGR2RGB),
                    "info": info_text(out),
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

        elif "last_video" in st.session_state:
            # after a rerun (slider change etc.) show the last analyzed frame instead of a blank page
            lv = st.session_state["last_video"]
            st.image(lv["img"], caption="Last analyzed frame", use_container_width=True)
            st.write(lv["info"])
            st.dataframe(lv["table"], use_container_width=True, hide_index=True)

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
