
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="Smart Ambulance Route",
    layout="wide"
)

st.title("🚑 Smart Ambulance Route Optimization")
st.write("Three-Road Traffic Monitoring System")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# COCO vehicle class IDs:
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# Example road lengths and uncongested speeds.
# Replace with measured values for your road network.
ROAD_INFO = [
    {"name": "Road 1", "distance": 2.0, "speed": 40},
    {"name": "Road 2", "distance": 2.5, "speed": 40},
    {"name": "Road 3", "distance": 3.0, "speed": 40},
]

def traffic_level(count):
    # Demonstration thresholds only.
    if count < 5:
        return "LOW", 1.0
    elif count < 12:
        return "MODERATE", 1.5
    else:
        return "HIGH", 2.5

def analyze_video(path, frame_limit=90):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    # Process sampled frames from each video.
    samples = []
    preview = None
    frame_index = 0

    while len(samples) < frame_limit:
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # Sample every 10th frame
        if frame_index % 10 != 0:
            continue

        results = model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=0.35,
            verbose=False
        )

        # Number of visible vehicles in this frame
        count = len(results[0].boxes)
        samples.append(count)

        preview = results[0].plot()

    cap.release()

    if not samples:
        return None

    average_count = float(np.mean(samples))

    return {
        "average_count": average_count,
        "preview": preview
    }

results = []

for i, road in enumerate(ROAD_INFO):
    video_path = f"videos/road{i+1}.mp4"

    analysis = analyze_video(video_path)

    if analysis is None:
        st.error(f"Cannot read {video_path}")
        st.stop()

    level, multiplier = traffic_level(
        analysis["average_count"]
    )

    # Illustrative delay model
    base_time = (
        road["distance"] / road["speed"]
    ) * 60

    estimated_time = base_time * multiplier

    results.append({
        "name": road["name"],
        "count": analysis["average_count"],
        "level": level,
        "time": estimated_time,
        "preview": analysis["preview"]
    })

st.subheader("Traffic Monitoring")

columns = st.columns(3)

for col, road in zip(columns, results):
    with col:
        st.image(
            cv2.cvtColor(
                road["preview"],
                cv2.COLOR_BGR2RGB
            ),
            caption=road["name"],
            use_container_width=True
        )

        st.metric(
            "Average visible vehicles",
            f"{road['count']:.1f}"
        )

        st.write("Traffic:", road["level"])
        st.write(
            f"Estimated time: {road['time']:.1f} min"
        )

# Choose minimum estimated travel time
best_route = min(
    results,
    key=lambda road: road["time"]
)

st.divider()
st.subheader("Ambulance Route Recommendation")

st.success(
    f"Recommended Route: {best_route['name']}"
)

st.metric(
    "Estimated Travel Time",
    f"{best_route['time']:.1f} minutes"
)

st.caption(
    "Prototype simulation only. Congestion multipliers "
    "are illustrative and must be calibrated using "
    "measured road travel times."
)
