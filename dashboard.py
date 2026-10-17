
import streamlit as st
import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO

st.set_page_config(
    page_title="Ambulance Route Monitoring",
    layout="wide"
)

st.title("Smart Ambulance Route Monitoring")
st.caption("Three-road prototype | Recorded video analysis")

# Change COM3 to your Arduino's port
ARDUINO_PORT = "COM3"

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

ROADS = [
    {"name": "Road 1", "distance": 2.0},
    {"name": "Road 2", "distance": 2.5},
    {"name": "Road 3", "distance": 3.0}
]

def analyze_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    counts = []
    frame_number = 0

    while len(counts) < 90:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number % 10 != 0:
            continue

        result = model.predict(
            frame,
            classes=[2, 3, 5, 7],
            conf=0.35,
            verbose=False
        )[0]

        counts.append(len(result.boxes))

    cap.release()

    if not counts:
        return None

    return float(np.mean(counts))

def get_traffic(count):
    # Prototype thresholds only
    if count < 5:
        return "LOW", 1.0
    elif count < 12:
        return "MODERATE", 1.5
    return "HIGH", 2.5

def send_to_arduino(road_number, travel_time):
    message = f"{road_number},{travel_time:.1f}\n"

    try:
        with serial.Serial(
            ARDUINO_PORT,
            9600,
            timeout=2
        ) as arduino:
            # Uno resets when serial port opens
            time.sleep(2)
            arduino.write(message.encode())
            arduino.flush()

        return True, "Data sent to Arduino"

    except serial.SerialException as error:
        return False, str(error)

# Store results so button clicks do not
# repeatedly process all three videos
if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Analyze Traffic"):
    results = []

    with st.spinner("Analyzing three roads..."):
        for i, road in enumerate(ROADS):
            count = analyze_video(
                f"videos/road{i+1}.mp4"
            )

            if count is None:
                st.error(
                    f"Cannot read Road {i+1} video"
                )
                st.stop()

            traffic, multiplier = get_traffic(count)

            base_time = road["distance"] / 40 * 60
            travel_time = base_time * multiplier

            results.append({
                "name": road["name"],
                "number": i + 1,
                "count": round(count, 1),
                "traffic": traffic,
                "time": round(travel_time, 1)
            })

    st.session_state.results = results

results = st.session_state.results

if results:
    st.subheader("Traffic Monitoring Data")

    st.dataframe(
        [
            {
                "Road": r["name"],
                "Avg. Visible Vehicles": r["count"],
                "Traffic": r["traffic"],
                "Estimated Time (min)": r["time"]
            }
            for r in results
        ],
        hide_index=True,
        width="stretch"
    )

    best = min(results, key=lambda r: r["time"])

    st.subheader("Ambulance Route Recommendation")
    st.success(f"Recommended: {best['name']}")

    st.metric(
        "Estimated Travel Time",
        f"{best['time']:.1f} minutes"
    )

    if st.button("Send Route to Arduino LCD"):
        success, message = send_to_arduino(
            best["number"],
            best["time"]
        )

        if success:
            st.success(message)
        else:
            st.error(message)

    st.caption(
        "Travel times are simulated estimates, "
        "not validated ambulance arrival times."
    )
else:
    st.info("Click Analyze Traffic to begin.")
