
import streamlit as st
import cv2
import numpy as np
import serial
import time
import re
from pathlib import Path
from ultralytics import YOLO

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Ambulance Route Monitoring",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# PROFESSIONAL DASHBOARD STYLING
# =====================================================

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
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d5dfeb;
    border-radius: 10px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: #123b67;
}

[data-testid="stDataFrame"] {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    overflow: hidden;
}

.stButton > button {
    background-color: #123b67;
    color: white;
    border: 1px solid #123b67;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #205b94;
    border-color: #205b94;
    color: white;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
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

.status-low {
    color: #15803d;
    font-weight: 700;
}

.status-moderate {
    color: #b45309;
    font-weight: 700;
}

.status-high {
    color: #b91c1c;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="dashboard-header">
    <h2>SMART AMBULANCE ROUTE MONITORING SYSTEM</h2>
    <p>
        Intelligent Traffic Analysis and
        Emergency Route Optimization
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Three-Road Traffic Monitoring | "
    "YOLO Vehicle Detection | Arduino LCD Integration"
)

# =====================================================
# SETTINGS
# =====================================================

VIDEO_FOLDER = Path(__file__).resolve().parent / "videos"

VEHICLE_CLASSES = [2, 3, 5, 7]

ROADS = [
    {
        "number": 1,
        "name": "Road 1",
        "distance": 2.0,
        "video": "road1.mp4"
    },
    {
        "number": 2,
        "name": "Road 2",
        "distance": 2.5,
        "video": "road2.mp4"
    },
    {
        "number": 3,
        "name": "Road 3",
        "distance": 3.0,
        "video": "road3.mp4"
    }
]

# =====================================================
# SIDEBAR SETTINGS
# =====================================================

with st.sidebar:
    st.header("System Settings")

    arduino_port = st.text_input(
        "Arduino COM Port",
        value="COM3"
    )

    st.caption(
        "Check Arduino IDE > Tools > Port "
        "to find the correct COM port."
    )

    frame_limit = st.slider(
        "Frames to analyze per road",
        min_value=10,
        max_value=150,
        value=90,
        step=10
    )

    st.info(
        "Traffic thresholds and travel times "
        "are demonstration estimates."
    )

# =====================================================
# LOAD YOLO MODEL
# =====================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# =====================================================
# VEHICLE DETECTION
# =====================================================

def analyze_video(video_path, model, sample_limit=90):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    vehicle_counts = []
    frame_number = 0

    while len(vehicle_counts) < sample_limit:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Process every 10th frame
        if frame_number % 10 != 0:
            continue

        result = model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=0.35,
            verbose=False
        )[0]

        vehicle_count = len(result.boxes)

        vehicle_counts.append(vehicle_count)

    cap.release()

    if len(vehicle_counts) == 0:
        return None

    average_count = float(
        np.mean(vehicle_counts)
    )

    maximum_count = int(
        np.max(vehicle_counts)
    )

    return {
        "average": round(average_count, 1),
        "maximum": maximum_count,
        "samples": len(vehicle_counts)
    }

# =====================================================
# TRAFFIC CLASSIFICATION
# =====================================================

def classify_traffic(count):

    # Demonstration thresholds
    if count < 5:
        return "LOW", 1.0

    elif count < 12:
        return "MODERATE", 1.5

    else:
        return "HIGH", 2.5

# =====================================================
# ESTIMATED TRAVEL TIME
# =====================================================

def calculate_time(distance, multiplier):

    assumed_speed = 40.0  # km/h

    base_time = (
        distance / assumed_speed
    ) * 60

    return round(
        base_time * multiplier, 1
    )

# =====================================================
# ARDUINO SERIAL COMMUNICATION
# =====================================================

def send_to_arduino(port, road_number, travel_time):

    if not re.fullmatch(r"COM\d+", port.upper()):
        return False, "Invalid COM port."

    message = (
        f"{road_number},{travel_time:.1f}\n"
    )

    try:
        with serial.Serial(
            port=port,
            baudrate=9600,
            timeout=2,
            write_timeout=2
        ) as arduino:

            # Arduino Uno resets on serial connection
            time.sleep(2)

            arduino.write(
                message.encode("ascii")
            )

            arduino.flush()

        return True, "Route sent successfully to Arduino LCD."

    except (serial.SerialException, OSError) as error:
        return False, str(error)

# =====================================================
# SESSION STATE
# =====================================================

if "results" not in st.session_state:
    st.session_state.results = None

# =====================================================
# TRAFFIC ANALYSIS BUTTON
# =====================================================

with st.container(border=True):

    st.subheader("TRAFFIC ANALYSIS CONTROL")

    st.write(
        "Analyze three recorded traffic videos "
        "to estimate road congestion."
    )

    if st.button(
        "ANALYZE ALL THREE ROADS",
        width="stretch"
    ):

        st.session_state.results = None

        with st.spinner(
            "Loading vehicle detection model..."
        ):
            model = load_model()

        results = []

        progress = st.progress(0)

        for index, road in enumerate(ROADS):

            video_path = (
                VIDEO_FOLDER / road["video"]
            )

            if not video_path.is_file():
                st.error(
                    f"Video not found: {video_path}"
                )
                st.stop()

            with st.spinner(
                f"Analyzing {road['name']}..."
            ):
                analysis = analyze_video(
                    video_path,
                    model,
                    frame_limit
                )

            if analysis is None:
                st.error(
                    f"Unable to analyze {road['name']}"
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

        st.session_state.results = results

        st.success(
            "Traffic analysis completed successfully."
        )

# =====================================================
# DISPLAY RESULTS
# =====================================================

results = st.session_state.results

if results:

    st.markdown("### ROAD-WISE TRAFFIC MONITORING")

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

                st.caption(
                    f"Maximum visible: {road['maximum']}"
                )

    # =================================================
    # TRAFFIC COMPARISON TABLE
    # =================================================

    st.markdown("### TRAFFIC COMPARISON")

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

    # =================================================
    # BEST ROUTE SELECTION
    # =================================================

    best_route = min(
        results,
        key=lambda road: road["time"]
    )

    st.markdown(
        "### AMBULANCE ROUTE RECOMMENDATION"
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

        st.caption(
            "Recommendation based on illustrative "
            "travel-time estimates, assuming all "
            "three routes are open and suitable."
        )

    # =================================================
    # ARDUINO LCD SECTION
    # =================================================

    st.markdown(
        "### ARDUINO UNO LCD COMMUNICATION"
    )

    with st.container(border=True):

        st.write(
            "Send the recommended road number "
            "and estimated travel time to "
            "the Arduino Uno LCD."
        )

        st.info(
            "LCD Preview: "
            f"BEST: ROAD {best_route['number']} | "
            f"TIME: {best_route['time']:.1f} MIN"
        )

        if st.button(
            "SEND RECOMMENDED ROUTE TO LCD",
            width="stretch"
        ):

            with st.spinner(
                "Sending data to Arduino..."
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
        "Academic prototype | "
        "Recorded-video analysis | "
        "Not intended for real emergency navigation."
    )

else:

    st.info(
        "Click ANALYZE ALL THREE ROADS "
        "to display traffic monitoring data."
    )
