"""
Smart Road Monitoring System
Real-time pothole detection using YOLO + OBS Virtual Camera
UI layout matches the reference dashboard screenshot.
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os
import time
import random
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Road Monitoring System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f0f2f6;
    color: #1a1a2e;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding-top: 0.5rem;
}

/* Sidebar section divider */
.sidebar-section {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 4px;
    border-bottom: 1.5px solid #e2e8f0;
}

/* Buttons */
.stButton > button {
    font-size: 0.82rem;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.45rem 1rem;
    width: 100%;
    cursor: pointer;
}

/* Panel title bar */
.panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #1e40af;
    border-left: 3px solid #3b82f6;
    padding-left: 8px;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Video info bar overlaid above the frame */
.video-info-bar {
    background: rgba(0,0,0,0.82);
    color: #fff;
    font-size: 0.82rem;
    font-family: 'Inter', monospace;
    padding: 6px 14px;
    border-radius: 6px 6px 0 0;
    display: flex;
    gap: 28px;
    align-items: center;
    font-weight: 500;
}
.info-fps   { color: #4ade80; font-weight: 700; }
.info-count { color: #fbbf24; font-weight: 700; }
.live-dot   { display:inline-block; width:9px; height:9px; background:#22c55e;
               border-radius:50%; margin-right:4px; animation: blink 1.2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Log entry card */
.log-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 5px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 0.75rem;
    line-height: 1.55;
    font-family: 'Inter', monospace;
}
.lc-time { color: #64748b; }
.lc-conf { color: #374151; font-weight: 600; }

/* Stat card */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.stat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}
.stat-value { font-size: 2rem; font-weight: 700; line-height: 1; }
.stat-high   { color: #ef4444; }
.stat-medium { color: #f59e0b; }
.stat-low    { color: #22c55e; }

.empty-log {
    font-size: 0.78rem;
    color: #94a3b8;
    padding: 10px 0;
    font-family: 'Inter', monospace;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
CSV_FILE    = "detections_log.csv"
CSV_COLUMNS = ["Timestamp", "Latitude", "Longitude", "Severity", "Confidence"]

GPS_LAT_RANGE = (18.4900, 18.5300)   # Simulated GPS — Pune, India
GPS_LON_RANGE = (73.8200, 73.8800)


# ──────────────────────────────────────────────
# MODULE: Model Loading
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str = "best.pt") -> YOLO:
    """Load and cache the YOLO model from disk."""
    return YOLO(model_path)


# ──────────────────────────────────────────────
# MODULE: Severity Classification
# ──────────────────────────────────────────────
def classify_severity(box_area: float, frame_area: float) -> str:
    """
    Classify pothole severity based on bounding-box area
    relative to total frame area.
        High   → > 3%
        Medium → 1–3%
        Low    → < 1%
    """
    ratio = box_area / frame_area if frame_area > 0 else 0
    if ratio > 0.03:
        return "High"
    elif ratio > 0.01:
        return "Medium"
    return "Low"


def severity_color(severity: str) -> tuple:
    """Return BGR color tuple for OpenCV drawing."""
    return {"High": (0, 0, 220), "Medium": (0, 165, 255), "Low": (0, 200, 80)}.get(severity, (200, 200, 200))


# ──────────────────────────────────────────────
# MODULE: GPS Simulation
# ──────────────────────────────────────────────
def simulate_gps() -> tuple:
    """Return a simulated GPS coordinate within the configured bounding box."""
    return (round(random.uniform(*GPS_LAT_RANGE), 6),
            round(random.uniform(*GPS_LON_RANGE), 6))


# ──────────────────────────────────────────────
# MODULE: CSV Logging
# ──────────────────────────────────────────────
def init_csv(filepath: str = CSV_FILE):
    """Create CSV with headers if it does not already exist."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def log_detection(severity: str, confidence: float, filepath: str = CSV_FILE) -> dict:
    """Append one detection row to the CSV and return the row dict."""
    lat, lon = simulate_gps()
    row = {
        "Timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Latitude":   lat,
        "Longitude":  lon,
        "Severity":   severity,
        "Confidence": round(confidence, 4),
    }
    with open(filepath, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
    return row


# ──────────────────────────────────────────────
# MODULE: Frame Annotation  (detection logic UNCHANGED)
# ──────────────────────────────────────────────
def annotate_frame(frame: np.ndarray, results, conf_threshold: float,
                   fps: float, show_labels: bool, show_overlay: bool) -> tuple:
    """
    Draw bounding boxes, severity labels, confidence, and FPS on frame.
    Returns (annotated_frame, list_of_detection_dicts).
    """
    h, w      = frame.shape[:2]
    frame_area = h * w
    detections = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)
        severity = classify_severity(box_area, frame_area)
        color    = severity_color(severity)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"{severity}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

        detections.append({"severity": severity, "confidence": conf})

    # FPS overlay — top-right
    if show_overlay:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)

    return frame, detections


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:

    st.markdown('<div class="sidebar-section">⚙️ &nbsp; System Configuration</div>', unsafe_allow_html=True)

    camera_index   = st.number_input("Camera Source", min_value=0, max_value=5, value=1,
                                      help="Video capture device index (0 = default webcam)")
    conf_threshold = st.slider("Detection Confidence", 0.10, 1.00, 0.40, 0.05,
                                help="Minimum confidence score to display a detection")

    st.markdown("")
    c1, c2 = st.columns(2)
    start = c1.button("▶ Start", use_container_width=True)
    stop  = c2.button("⏹ Stop",  use_container_width=True)

    st.markdown('<div class="sidebar-section">📊 &nbsp; Display Options</div>', unsafe_allow_html=True)
    show_labels  = st.checkbox("Show Detection Labels",     value=True)
    show_overlay = st.checkbox("Show Information Overlay",  value=True)
    show_stats   = st.checkbox("Show Real-time Statistics", value=True)

    st.markdown('<div class="sidebar-section">🎯 &nbsp; Detection Filters</div>', unsafe_allow_html=True)
    filter_high   = st.checkbox("High Severity",   value=True)
    filter_medium = st.checkbox("Medium Severity", value=True)
    filter_low    = st.checkbox("Low Severity",    value=True)

    allowed_severities = set()
    if filter_high:   allowed_severities.add("High")
    if filter_medium: allowed_severities.add("Medium")
    if filter_low:    allowed_severities.add("Low")

    st.markdown('<div class="sidebar-section">ℹ️ &nbsp; About</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.75rem;color:#64748b;line-height:1.6;">'
        'Smart Road Monitoring System<br>'
        'YOLO-based pothole detection<br>'
        'v1.0 · OBS Virtual Camera</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────
init_csv()

video_col, right_col = st.columns([3, 1], gap="medium")

# ── Left: Live Video Feed ──
with video_col:
    st.markdown('<div class="panel-title">📹 Live Video Feed</div>', unsafe_allow_html=True)
    info_bar_ph   = st.empty()
    frame_ph      = st.empty()

# ── Right: Log + Stats ──
with right_col:
    st.markdown('<div class="panel-title">📋 Detection Log</div>', unsafe_allow_html=True)
    log_ph = st.empty()

    st.markdown('<div class="panel-title" style="margin-top:1rem;">📊 Statistics</div>',
                unsafe_allow_html=True)
    stat_high_ph   = st.empty()
    stat_medium_ph = st.empty()
    stat_low_ph    = st.empty()


# ── Helpers ──
def render_stats(high: int, medium: int, low: int):
    stat_high_ph.markdown(
        f'<div class="stat-card"><div class="stat-label">High Severity</div>'
        f'<div class="stat-value stat-high">{high}</div></div>', unsafe_allow_html=True)
    stat_medium_ph.markdown(
        f'<div class="stat-card"><div class="stat-label">Medium Severity</div>'
        f'<div class="stat-value stat-medium">{medium}</div></div>', unsafe_allow_html=True)
    stat_low_ph.markdown(
        f'<div class="stat-card"><div class="stat-label">Low Severity</div>'
        f'<div class="stat-value stat-low">{low}</div></div>', unsafe_allow_html=True)


def render_log(recent_logs: list):
    if not recent_logs:
        log_ph.markdown('<div class="empty-log">No detections recorded yet</div>',
                        unsafe_allow_html=True)
        return
    color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
    html = ""
    for entry in recent_logs:
        c = color_map.get(entry["Severity"], "#94a3b8")
        html += (
            f'<div class="log-card">'
            f'<span style="font-weight:700;color:{c};">{entry["Severity"]}</span> &nbsp;'
            f'<span class="lc-conf">{entry["Confidence"]:.0%}</span><br>'
            f'<span class="lc-time">{entry["Timestamp"][11:]} · '
            f'{entry["Latitude"]}, {entry["Longitude"]}</span>'
            f'</div>'
        )
    log_ph.markdown(html, unsafe_allow_html=True)


def render_info_bar(now_str: str, fps: float, total: int, live: bool = True):
    dot   = '<span class="live-dot"></span>' if live else ''
    label = "LIVE" if live else "READY"
    info_bar_ph.markdown(
        f'<div class="video-info-bar">'
        f'<span>Time: {now_str}</span>'
        f'<span class="info-fps">FPS: {fps:.1f}</span>'
        f'<span class="info-count">Detected: {total}</span>'
        f'<span>{dot}{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Idle state on load ──
render_info_bar("--:--:--", 0.0, 0, live=False)
render_log([])
render_stats(0, 0, 0)


# ──────────────────────────────────────────────
# DETECTION LOOP
# ──────────────────────────────────────────────
if start:
    model = load_model("best.pt")
    cap   = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("❌ Camera not found. Adjust the Camera Source index in the sidebar.")
    else:
        high_count   = 0
        medium_count = 0
        low_count    = 0
        total_count  = 0
        recent_logs  = []       # keep last 15 for the log panel

        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame not received from camera.")
                break

            # ── FPS ──
            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # ── YOLO inference (UNCHANGED) ──
            results = model(frame)

            # ── Annotate ──
            annotated, detections = annotate_frame(
                frame, results, conf_threshold, fps, show_labels, show_overlay
            )

            # ── Process detections ──
            for det in detections:
                if det["severity"] not in allowed_severities:
                    continue
                total_count += 1
                if det["severity"] == "High":
                    high_count += 1
                elif det["severity"] == "Medium":
                    medium_count += 1
                else:
                    low_count += 1

                row = log_detection(det["severity"], det["confidence"])
                recent_logs.insert(0, row)
                recent_logs = recent_logs[:15]

            # ── Refresh UI ──
            now_str = datetime.now().strftime("%H:%M:%S")
            render_info_bar(now_str, fps, total_count, live=True)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, channels="RGB", use_container_width=True)

            render_log(recent_logs)
            if show_stats:
                render_stats(high_count, medium_count, low_count)

            if stop:
                break

        cap.release()
        st.success(f"✅ Session ended. Log saved to `{CSV_FILE}`.")