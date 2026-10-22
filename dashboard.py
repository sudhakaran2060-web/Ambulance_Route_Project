
import os
import time
import threading
from pathlib import Path

import cv2
import requests
import streamlit as st
from ultralytics import YOLO

# =====================================================
# STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart City Traffic Monitoring",
    page_icon="🚦",
    layout="wide"
)

# =====================================================
# ROAD SOURCES
# =====================================================

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "sudhakaran2060-web/Ambulance_Route_Project/main/"
)

# Replace this with the actual ESP32-CAM IP address.
ESP32_CAM_URL = "http://192.168.1.105:81/stream"

ROAD_SOURCES = {
    "Road 1": {
        "type": "video",
        "url": GITHUB_BASE + "road1.mp4",
    },
    "Road 2": {
        "type": "video",
        "url": GITHUB_BASE + "road2.mp4",
    },
    "Road 3": {
        "type": "video",
        "url": GITHUB_BASE + "road3.mp4",
    },
    "Road 4": {
        "type": "esp32cam",
        "url": ESP32_CAM_URL,
    },
}

VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
MAX_DENSITY_VEHICLES = 20
FRAME_WIDTH = 640

# =====================================================
# DOWNLOAD STORED GITHUB VIDEOS
# =====================================================

def download_video(road, url):

    filename = VIDEO_DIR / (
        road.lower().replace(" ", "") + ".mp4"
    )

    if filename.exists() and filename.stat().st_size > 0:
        return str(filename)

    temp_file = Path(str(filename) + ".part")

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=(15, 120)
        )
        response.raise_for_status()

        total_bytes = 0

        with open(temp_file, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)
                    total_bytes += len(chunk)

        if total_bytes == 0:
            raise ValueError("Downloaded file is empty")

        os.replace(temp_file, filename)
        return str(filename)

    except Exception as error:

        print(f"{road} download error: {error}")

        if temp_file.exists():
            temp_file.unlink()

        return None

# =====================================================
# INITIALIZE YOLO AND SHARED STATE
# =====================================================

@st.cache_resource
def initialize_system():

    model = YOLO(MODEL_PATH)

    state = {
        road: {
            "count": 0,
            "density": 0,
            "level": "Waiting",
            "status": "Initializing",
            "updated": "--",
            "frame": None,
        }
        for road in ROAD_SOURCES
    }

    state_lock = threading.RLock()
    inference_lock = threading.Lock()

    return model, state, state_lock, inference_lock

# =====================================================
# PROCESS EACH ROAD
# =====================================================

def process_road(
    road,
    config,
    model,
    state,
    state_lock,
    inference_lock
):

    source_type = config["type"]
    source_url = config["url"]

    while True:

        with state_lock:
            state[road]["status"] = "Connecting..."

        if source_type == "video":

            video_path = download_video(
                road,
                source_url
            )

            if not video_path:

                with state_lock:
                    state[road]["status"] = (
                        "GitHub video unavailable"
                    )
                    state[road]["level"] = "Unavailable"
                    state[road]["frame"] = None

                time.sleep(15)
                continue

            cap = cv2.VideoCapture(video_path)

        else:
            # Live ESP32-CAM MJPEG stream
            cap = cv2.VideoCapture(source_url)

        if not cap.isOpened():

            with state_lock:
                state[road]["status"] = (
                    "Cannot open video source"
                )
                state[road]["level"] = "Unavailable"
                state[road]["frame"] = None

            cap.release()
            time.sleep(5)
            continue

        if source_type == "esp32cam":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        with state_lock:
            state[road]["status"] = "Running"

        print(f"{road}: analysis started")

        while True:

            success, frame = cap.read()

            if not success:

                if source_type == "video":
                    # Replay stored GitHub video
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES, 0
                    )

                    success, frame = cap.read()

                    if not success:
                        break

                else:
                    # ESP32-CAM disconnected
                    print(
                        f"{road}: ESP32-CAM disconnected"
                    )
                    break

            height, width = frame.shape[:2]

            if width > FRAME_WIDTH:

                new_height = int(
                    height * FRAME_WIDTH / width
                )

                frame = cv2.resize(
                    frame,
                    (FRAME_WIDTH, new_height)
                )

            try:

                with inference_lock:
                    result = model.predict(
                        frame,
                        classes=VEHICLE_CLASS_IDS,
                        conf=CONFIDENCE,
                        verbose=False
                    )[0]

                count = (
                    len(result.boxes)
                    if result.boxes is not None
                    else 0
                )

                density = min(
                    100,
                    round(
                        count * 100 /
                        MAX_DENSITY_VEHICLES
                    )
                )

                if count <= 5:
                    level = "Low"

                elif count <= 12:
                    level = "Medium"

                else:
                    level = "High"

                annotated = result.plot()

                ok, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                with state_lock:

                    state[road].update({
                        "count": count,
                        "density": density,
                        "level": level,
                        "status": "Running",
                        "updated": time.strftime(
                            "%H:%M:%S"
                        ),
                        "frame": (
                            jpeg.tobytes()
                            if ok else None
                        ),
                    })

            except Exception as error:

                print(
                    f"{road} detection error: {error}"
                )

                with state_lock:
                    state[road]["status"] = (
                        "Detection error: "
                        + str(error)
                    )

                time.sleep(1)

            # Limit processing speed
            time.sleep(0.1)

        cap.release()

        with state_lock:
            state[road]["status"] = "Reconnecting..."
            state[road]["level"] = "Unavailable"
            state[road]["frame"] = None

        time.sleep(3)

# =====================================================
# START BACKGROUND WORKERS
# =====================================================

model, road_state, state_lock, inference_lock = (
    initialize_system()
)


@st.cache_resource
def start_workers():

    workers = []

    for road, config in ROAD_SOURCES.items():

        worker = threading.Thread(
            target=process_road,
            args=(
                road,
                config,
                model,
                road_state,
                state_lock,
                inference_lock,
            ),
            daemon=True,
            name=f"{road}-worker"
        )

        worker.start()
        workers.append(worker)

    return workers


workers = start_workers()

# =====================================================
# DASHBOARD UI
# =====================================================

st.title("🚦 Smart City Traffic Monitoring")

st.caption(
    "3 GitHub Traffic Videos + "
    "1 Live ESP32-CAM | YOLO Vehicle Detection"
)

st.info(
    "Roads 1–3 use prerecorded GitHub videos. "
    "Road 4 uses a live ESP32-CAM Wi-Fi stream."
)

# =====================================================
# AUTO-REFRESH DASHBOARD
# =====================================================

@st.fragment(run_every="2s")
def show_dashboard():

    with state_lock:
        snapshot = {
            road: data.copy()
            for road, data in road_state.items()
        }

    running_roads = sum(
        data["status"] == "Running"
        for data in snapshot.values()
    )

    total_vehicles = sum(
        data["count"]
        for data in snapshot.values()
        if data["status"] == "Running"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Roads",
        len(ROAD_SOURCES)
    )

    col2.metric(
        "Vehicles in Current Frames",
        total_vehicles
    )

    col3.metric(
        "Roads Running",
        f"{running_roads} / {len(ROAD_SOURCES)}"
    )

    st.divider()

    road_columns = st.columns(4)

    for column, (road, data) in zip(
        road_columns,
        snapshot.items()
    ):

        with column:

            with st.container(border=True):

                st.subheader(road)

                if road == "Road 4":
                    st.caption("ESP32-CAM — LIVE")
                else:
                    st.caption("GitHub — STORED VIDEO")

                if data["frame"] is not None:

                    st.image(
                        data["frame"],
                        caption=f"{road} YOLO Detection",
                        use_container_width=True
                    )

                else:
                    st.warning(
                        "Waiting for video frames..."
                    )

                st.metric(
                    "Vehicle Count",
                    data["count"]
                )

                st.metric(
                    "Traffic Density",
                    f'{data["density"]}%'
                )

                if data["level"] == "High":
                    st.error("HIGH TRAFFIC")

                elif data["level"] == "Medium":
                    st.warning("MEDIUM TRAFFIC")

                elif data["level"] == "Low":
                    st.success("LOW TRAFFIC")

                else:
                    st.info(data["level"])

                st.caption(
                    f'Status: {data["status"]}'
                )

                st.caption(
                    f'Updated: {data["updated"]}'
                )


show_dashboard()

st.divider()

st.caption(
    "Academic prototype: counts are per-frame "
    "vehicle detections, not unique vehicle totals. "
    "Traffic density is an illustrative score."
)
