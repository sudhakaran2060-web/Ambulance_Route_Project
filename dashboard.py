
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Ambulance Route Monitoring",
    page_icon="🚑",
    layout="wide"
)

# ==========================================
# PROFESSIONAL DASHBOARD DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
    color: #172b4d;
}

.block-container {
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #102a43 !important;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: #123b67;
}

[data-testid="stDataFrame"] {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
}

.stButton > button[kind="primary"] {
    background-color: #123b67;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

.dashboard-header {
    background: linear-gradient(
        110deg, #102a43, #205b94
    );
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.dashboard-header h2 {
    color: white !important;
    margin: 0;
    font-size: 26px;
}

.dashboard-header p {
    color: #dbeafe;
    margin-top: 8px;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="dashboard-header">
    <h2>SMART AMBULANCE ROUTE MONITORING</h2>
    <p>
        Intelligent Traffic Analysis and
        Emergency Route Optimization
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Three-Road Monitoring | YOLO Detection | "
    "Streamlit Dashboard | Arduino Uno LCD"
)

# ==========================================
# CONFIGURATION
# ==========================================

VEHICLE_CLASSES = [2, 3, 5, 7]

ROADS = [
    {"number": 1, "name": "Road 1", "distance": 2.0},
    {"number": 2, "name": "Road 2", "distance": 2.5},
    {"number": 3, "name": "Road 3", "distance": 3.0}
]

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("System Settings")

    arduino_port = st.text_input(
        "Arduino COM Port",
        value="COM3"
    )

    frame_limit = st.slider(
        "Frames to analyze per road",
        min_value=10,
        max_value=150,
        value=90,
        step=10
    )

    st.caption(
        "Arduino USB serial communication "
        "is available only when running locally."
    )

# ==========================================
# LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# ==========================================
# VIDEO ANALYSIS
# ==========================================

def analyze_video(video_path, model, sample_limit):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    counts = []
    frame_number = 0

    while len(counts) < sample_limit:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        if frame_number % 10 != 0:
            continue

        result = model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=0.35,
            verbose=False
        )[0]

        counts.append(len(result.boxes))

    cap.release()

    if not counts:
        return None

    return {
        "average": round(float(np.mean(counts)), 1),
        "maximum": int(np.max(counts)),
        "samples": len(counts)
    }

# ==========================================
# TRAFFIC CLASSIFICATION
# ==========================================

def classify_traffic(count):

    if count < 5:
        return "LOW", 1.0

    elif count < 12:
        return "MODERATE", 1.5

    else:
        return "HIGH", 2.5

# ==========================================
# ESTIMATED TRAVEL TIME
# ==========================================

def calculate_time(distance, multiplier):

    assumed_speed = 40.0

    base_time = (
        distance / assumed_speed
    ) * 60

    return round(base_time * multiplier, 1)

# ==========================================
# ARDUINO COMMUNICATION
# ==========================================

def send_to_arduino(port, road_number, travel_time):

    try:
        import serial

        with serial.Serial(
            port=port,
            baudrate=9600,
            timeout=2,
            write_timeout=2
        ) as arduino:

            time.sleep(2)

            message = (
                f"{road_number},{travel_time:.1f}\n"
            )

            arduino.write(message.encode("ascii"))
            arduino.flush()

        return True, "Data sent to Arduino LCD."

    except Exception as error:
        return False, str(error)

# ==========================================
# SESSION STATE
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = None

# ==========================================
# VIDEO UPLOAD
# ==========================================

st.subheader("UPLOAD ROAD TRAFFIC VIDEOS")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        video1 = st.file_uploader(
            "Road 1 Video",
            type=["mp4"],
            key="video1"
        )

    with col2:
        video2 = st.file_uploader(
            "Road 2 Video",
            type=["mp4"],
            key="video2"
        )

    with col3:
        video3 = st.file_uploader(
            "Road 3 Video",
            type=["mp4"],
            key="video3"
        )

uploaded_videos = [video1, video2, video3]

# ==========================================
# ANALYZE BUTTON
# ==========================================

with st.container(border=True):

    st.subheader("TRAFFIC ANALYSIS CONTROL")

    if st.button(
        "ANALYZE ALL THREE ROADS",
        type="primary",
        width="stretch"
    ):

        if not all(uploaded_videos):
            st.error(
                "Please upload all three traffic videos."
            )
            st.stop()

        st.session_state.results = None
        results = []

        with st.spinner("Loading YOLO model..."):
            model = load_model()

        progress = st.progress(0)

        for index, (road, uploaded) in enumerate(
            zip(ROADS, uploaded_videos)
        ):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_file:

                    temp_file.write(uploaded.getvalue())
                    temp_path = temp_file.name

                with st.spinner(
                    f"Analyzing {road['name']}..."
                ):
                    analysis = analyze_video(
                        temp_path,
                        model,
                        frame_limit
                    )

                if analysis is None:
                    st.error(
                        f"Unable to analyze {road['name']}."
                    )
                    st.stop()

                traffic, multiplier = classify_traffic(
                    analysis["average"]
                )

                estimated_time = calculate_time(
                    road["distance"],
                    multiplier
                )

                results.append({
                    "number": road["number"],
                    "name": road["name"],
                    "count": analysis["average"],
                    "maximum": analysis["maximum"],
                    "samples": analysis["samples"],
                    "traffic": traffic,
                    "time": estimated_time
                })

                progress.progress(
                    (index + 1) / len(ROADS)
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        st.session_state.results = results

        st.success(
            "Traffic analysis completed successfully!"
        )

# ==========================================
# DISPLAY RESULTS
# ==========================================

results = st.session_state.results

if results:

    st.subheader("ROAD-WISE TRAFFIC MONITORING")

    columns = st.columns(3)

    for column, road in zip(columns, results):

        with column:

            with st.container(border=True):

                st.markdown(f"### {road['name']}")

                st.metric(
                    "Average Visible Vehicles",
                    f"{road['count']:.1f}"
                )

                if road["traffic"] == "LOW":
                    st.success("LOW TRAFFIC")

                elif road["traffic"] == "MODERATE":
                    st.warning("MODERATE TRAFFIC")

                else:
                    st.error("HIGH TRAFFIC")

                st.metric(
                    "Estimated Travel Time",
                    f"{road['time']:.1f} min"
                )

                st.caption(
                    f"Maximum visible vehicles: "
                    f"{road['maximum']}"
                )

    # ======================================
    # TRAFFIC COMPARISON
    # ======================================

    st.subheader("TRAFFIC COMPARISON")

    with st.container(border=True):

        table_data = []

        for road in results:

            table_data.append({
                "Road": road["name"],
                "Avg. Vehicles": road["count"],
                "Max. Vehicles": road["maximum"],
                "Traffic Level": road["traffic"],
                "Estimated Time (min)": road["time"]
            })

        st.dataframe(
            table_data,
            hide_index=True,
            width="stretch"
        )

    # ======================================
    # RECOMMENDED ROUTE
    # ======================================

    best_route = min(
        results,
        key=lambda road: road["time"]
    )

    st.subheader(
        "AMBULANCE ROUTE RECOMMENDATION"
    )

    with st.container(border=True):

        st.success(
            "RECOMMENDED ROUTE: "
            f"{best_route['name'].upper()}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Recommended Road",
                best_route["name"]
            )

        with col2:
            st.metric(
                "Estimated Travel Time",
                f"{best_route['time']:.1f} min"
            )

        with col3:
            st.metric(
                "Traffic Condition",
                best_route["traffic"]
            )

    # ======================================
    # ARDUINO LCD COMMUNICATION
    # ======================================

    st.subheader(
        "ARDUINO UNO LCD COMMUNICATION"
    )

    with st.container(border=True):

        st.info(
            f"LCD PREVIEW: BEST ROAD "
            f"{best_route['number']} | "
            f"TIME: {best_route['time']:.1f} MIN"
        )

        st.caption(
            "USB serial communication works only "
            "when Streamlit runs on the same "
            "computer as the Arduino."
        )

        if st.button(
            "SEND RECOMMENDED ROUTE TO LCD",
            width="stretch"
        ):

            success, message = send_to_arduino(
                arduino_port,
                best_route["number"],
                best_route["time"]
            )

            if success:
                st.success(message)

            else:
                st.error(
                    "Arduino communication failed: "
                    + message
                )

    st.divider()

    st.caption(
        "Academic demonstration only. Vehicle counts "
        "are averages per sampled frame, not unique "
        "vehicles. Travel times are illustrative "
        "and are not validated for emergency use."
    )

else:
    st.info(
        "Upload three MP4 videos and click "
        "ANALYZE ALL THREE ROADS."
    )
