
import streamlit as st
import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Smart Ambulance Route Monitoring",
    page_icon="🚑",
    layout="wide"
)

# ==========================================
# PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}

.block-container {
    max-width: 1300px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #102a43 !important;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
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
}

.dashboard-header p {
    color: #dbeafe;
    margin-bottom: 0;
}

.stButton > button[kind="primary"] {
    background: #123b67;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
    <h2>SMART AMBULANCE ROUTE MONITORING</h2>
    <p>
        Automatic Traffic Detection and
        Emergency Route Recommendation
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "YOLO Vehicle Detection | Three-Road Monitoring | "
    "Arduino Uno LCD"
)

# ==========================================
# CAMERA CONFIGURATION
# ==========================================

st.sidebar.header("Camera Settings")

st.sidebar.caption(
    "Use 0, 1, 2 for local webcams, "
    "or an ESP32-CAM/IP camera stream URL."
)

source1 = st.sidebar.text_input(
    "Road 1 Camera",
    value="0"
)

source2 = st.sidebar.text_input(
    "Road 2 Camera",
    value="1"
)

source3 = st.sidebar.text_input(
    "Road 3 Camera",
    value="2"
)

arduino_port = st.sidebar.text_input(
    "Arduino COM Port",
    value="COM3"
)

duration = st.sidebar.slider(
    "Analysis duration per road (seconds)",
    min_value=5,
    max_value=30,
    value=10
)

ROADS = [
    {"number": 1, "name": "Road 1", "distance": 2.0},
    {"number": 2, "name": "Road 2", "distance": 2.5},
    {"number": 3, "name": "Road 3", "distance": 3.0}
]

VEHICLE_CLASSES = [2, 3, 5, 7]

# ==========================================
# YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# ==========================================
# CAMERA SOURCE CONVERSION
# ==========================================

def parse_source(value):
    value = value.strip()

    if value.isdigit():
        return int(value)

    return value

# ==========================================
# LIVE CAMERA ANALYSIS
# ==========================================

def analyze_camera(source, model, duration_seconds):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return None

    counts = []
    start_time = time.monotonic()

    try:
        while (
            time.monotonic() - start_time
            < duration_seconds
        ):

            success, frame = cap.read()

            if not success:
                break

            result = model.predict(
                frame,
                classes=VEHICLE_CLASSES,
                conf=0.35,
                verbose=False
            )[0]

            counts.append(len(result.boxes))

    finally:
        cap.release()

    if not counts:
        return None

    return {
        "average": round(
            float(np.mean(counts)), 1
        ),
        "maximum": int(np.max(counts)),
        "samples": len(counts)
    }

# ==========================================
# TRAFFIC CLASSIFICATION
# ==========================================

def classify_traffic(count):

    # Demonstration thresholds
    if count < 5:
        return "LOW", 1.0

    elif count < 12:
        return "MODERATE", 1.5

    return "HIGH", 2.5

# ==========================================
# TRAVEL TIME ESTIMATION
# ==========================================

def calculate_time(distance, multiplier):

    assumed_speed = 40.0

    base_time = (
        distance / assumed_speed
    ) * 60

    return round(base_time * multiplier, 1)

# ==========================================
# ARDUINO SERIAL COMMUNICATION
# ==========================================

def send_to_arduino(port, road_number, travel_time):

    try:
        with serial.Serial(
            port=port,
            baudrate=9600,
            timeout=2,
            write_timeout=2
        ) as arduino:

            # Arduino Uno resets when port opens
            time.sleep(2)

            message = (
                f"{road_number},{travel_time:.1f}\n"
            )

            arduino.write(
                message.encode("ascii")
            )

            arduino.flush()

        return True, "Route sent to Arduino LCD."

    except (serial.SerialException, OSError) as error:
        return False, str(error)

# ==========================================
# SESSION STATE
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = None

# ==========================================
# TRAFFIC MONITORING CONTROL
# ==========================================

with st.container(border=True):

    st.subheader("AUTOMATIC TRAFFIC MONITORING")

    st.write(
        "Capture traffic directly from three cameras. "
        "No manual video upload is required."
    )

    if st.button(
        "START AUTOMATIC ANALYSIS",
        type="primary",
        width="stretch"
    ):

        st.session_state.results = None

        sources = [
            parse_source(source1),
            parse_source(source2),
            parse_source(source3)
        ]

        with st.spinner("Loading YOLO model..."):
            model = load_model()

        results = []
        progress = st.progress(0)

        for index, (road, source) in enumerate(
            zip(ROADS, sources)
        ):

            with st.spinner(
                f"Monitoring {road['name']}..."
            ):

                analysis = analyze_camera(
                    source,
                    model,
                    duration
                )

            if analysis is None:
                st.error(
                    f"Cannot access {road['name']} camera. "
                    "Check its source and connection."
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
                "traffic": traffic,
                "time": estimated_time
            })

            progress.progress(
                (index + 1) / 3
            )

        st.session_state.results = results

        st.success(
            "Automatic traffic analysis completed!"
        )

# ==========================================
# DASHBOARD RESULTS
# ==========================================

results = st.session_state.results

if results:

    st.subheader("ROAD-WISE TRAFFIC STATUS")

    columns = st.columns(3)

    for column, road in zip(columns, results):

        with column:

            with st.container(border=True):

                st.markdown(
                    f"### {road['name']}"
                )

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

    # ======================================
    # COMPARISON TABLE
    # ======================================

    st.subheader("TRAFFIC COMPARISON")

    with st.container(border=True):

        table_data = []

        for road in results:

            table_data.append({
                "Road": road["name"],
                "Avg. Vehicles": road["count"],
                "Traffic Level": road["traffic"],
                "Estimated Time (min)": road["time"]
            })

        st.dataframe(
            table_data,
            hide_index=True,
            width="stretch"
        )

    # ======================================
    # ROUTE RECOMMENDATION
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
    # ARDUINO LCD
    # ======================================

    st.subheader("ARDUINO UNO LCD COMMUNICATION")

    with st.container(border=True):

        st.info(
            f"LCD PREVIEW: BEST ROAD "
            f"{best_route['number']} | "
            f"TIME: {best_route['time']:.1f} MIN"
        )

        if st.button(
            "SEND ROUTE TO ARDUINO LCD",
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
                st.error(message)

    st.caption(
        "Academic prototype only. Vehicle detections "
        "are averages per frame. Travel-time estimates "
        "are simulated and not validated for "
        "real emergency navigation."
    )

else:

    st.info(
        "Configure your cameras and click "
        "START AUTOMATIC ANALYSIS."
    )
