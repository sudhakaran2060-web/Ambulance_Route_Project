
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

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "sudhakaran2060-web/Ambulance_Route_Project/main/"
)

ROAD_SOURCES = {
    "Road 1": GITHUB_BASE + "road1.mp4",
    "Road 2": GITHUB_BASE + "road2.mp4",
    "Road 3": GITHUB_BASE + "road3.mp4",
}

VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
MAX_DENSITY_VEHICLES = 20
FRAME_WIDTH = 640

# =====================================================
# DOWNLOAD VIDEOS FROM GITHUB
# =====================================================

def download_video(road, url):
    filename = VIDEO_DIR / f"{road.lower().replace(' ', '')}.mp4"

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
            for chunk in response.iter_content(chunk_size=1024 * 1024):
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
# SHARED YOLO MODEL AND ROAD STATUS
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
# PROCESS EACH ROAD IN A BACKGROUND THREAD
# =====================================================

def process_road(road, url, model, state, state_lock, inference_lock):
    while True:
        video_path = download_video(road, url)

        if not video_path:
            with state_lock:
                state[road]["status"] = (
                    "Video unavailable. Check GitHub URL and filename."
                )
                state[road]["level"] = "Unavailable"

            time.sleep(15)
            continue

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            with state_lock:
                state[road]["status"] = "Cannot open MP4 file"

            cap.release()
            time.sleep(10)
            continue

        with state_lock:
            state[road]["status"] = "Running"

        print(f"{road}: analysis started")

        while True:
            success, frame = cap.read()

            if not success:
                # Restart the stored video from the beginning.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()

                if not success:
                    with state_lock:
                        state[road]["status"] = (
                            "Cannot read video frames"
                        )
                    break

            height, width = frame.shape[:2]

            if width > FRAME_WIDTH:
                new_height = int(height * FRAME_WIDTH / width)
                frame = cv2.resize(frame, (FRAME_WIDTH, new_height))

            try:
                # One shared model; inference is protected by a lock.
                with inference_lock:
                    result = model.predict(
                        frame,
                        conf=CONFIDENCE,
                        verbose=False
                    )[0]

                count = 0

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        label = model.names[class_id]

                        if label in VEHICLE_CLASSES:
                            count += 1

                density = min(
                    100,
                    round(count * 100 / MAX_DENSITY_VEHICLES)
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
                        "updated": time.strftime("%H:%M:%S"),
                        "frame": jpeg.tobytes() if ok else None,
                    })

            except Exception as error:
                print(f"{road} detection error: {error}")

                with state_lock:
                    state[road]["status"] = (
                        "Detection error: " + str(error)
                    )

                time.sleep(1)

            # Limit processing speed and CPU usage.
            time.sleep(0.05)

        cap.release()
        time.sleep(3)


# =====================================================
# START BACKGROUND WORKERS ONLY ONCE
# =====================================================

model, road_state, state_lock, inference_lock = initialize_system()


@st.cache_resource
def start_workers():
    workers = []

    for road, url in ROAD_SOURCES.items():
        worker = threading.Thread(
            target=process_road,
            args=(
                road,
                url,
                model,
                road_state,
                state_lock,
                inference_lock,
            ),
            daemon=True,
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
    "Three-road vehicle detection using YOLO and GitHub-hosted videos"
)

st.info(
    "This dashboard analyzes stored MP4 videos. "
    "It is not a live ESP32-CAM stream."
)

# Automatically refresh dashboard while workers analyze the videos.
@st.fragment(run_every="2s")
def show_dashboard():

    with state_lock:
        snapshot = {
            road: data.copy()
            for road, data in road_state.items()
        }

    total_vehicles = sum(
        data["count"] for data in snapshot.values()
    )

    running_roads = sum(
        data["status"] == "Running"
        for data in snapshot.values()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Roads Configured", "3")
    col2.metric("Vehicles in Current Frames", total_vehicles)
    col3.metric("Roads Running", f"{running_roads} / 3")

    st.divider()

    road_columns = st.columns(3)

    for column, (road, data) in zip(road_columns, snapshot.items()):
        with column:
            st.subheader(road)

            if data["frame"] is not None:
                st.image(
                    data["frame"],
                    caption=f"{road} — YOLO detection",
                    use_container_width=True,
                )
            else:
                st.warning("Waiting for video frames...")

            st.metric("Vehicle Count", data["count"])
            st.metric("Traffic Density", f'{data["density"]}%')

            level = data["level"]

            if level == "High":
                st.error(f"Traffic Level: {level}")
            elif level == "Medium":
                st.warning(f"Traffic Level: {level}")
            elif level == "Low":
                st.success(f"Traffic Level: {level}")
            else:
                st.write(f"Traffic Level: {level}")

            st.caption(f'Status: {data["status"]}')
            st.caption(f'Last update: {data["updated"]}')


show_dashboard()

st.divider()
st.caption(
    "Vehicle count is for the current processed frame, "
    "not the total number of unique vehicles in the video."
)
