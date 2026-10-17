
import time
import threading
from pathlib import Path

import cv2
import requests
import streamlit as st
from ultralytics import YOLO


# ==========================================================
# SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Smart City Traffic Monitoring",
    page_icon="🚑",
    layout="wide"
)

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "sudhakaran2060-web/Ambulance_Route_Project/main/"
)

# Replace these example IP addresses with the actual ones
# shown in your ESP32-CAM and ESP32 Serial Monitors.

ESP32_CAM_URL = "http://192.168.1.110:81/stream"
ESP32_LCD_URL = "http://192.168.1.105/traffic"

SEND_TO_LCD = True

ROAD_SOURCES = {
    "Road 1": GITHUB_BASE + "road1.mp4",
    "Road 2": GITHUB_BASE + "road2.mp4",
    "Road 3": GITHUB_BASE + "road3.mp4",
    "Road 4": ESP32_CAM_URL
}

LIVE_ROADS = {"Road 4"}

VIDEO_DIR = Path("videos")

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35

MAX_VEHICLES = 20
FRAME_WIDTH = 640

PROCESS_DELAY = 0.15
LCD_SEND_INTERVAL = 10

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle"
}


# ==========================================================
# DOWNLOAD GITHUB VIDEOS
# ==========================================================

def download_video(road_name, url):
    VIDEO_DIR.mkdir(exist_ok=True)

    filename = road_name.lower().replace(" ", "") + ".mp4"
    video_path = VIDEO_DIR / filename

    if video_path.exists() and video_path.stat().st_size > 0:
        return str(video_path)

    temporary_path = video_path.with_suffix(".download")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 120)
        ) as response:

            response.raise_for_status()

            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        temporary_path.replace(video_path)
        return str(video_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


# ==========================================================
# INITIALIZE YOLO AND SHARED DATA
# ==========================================================

@st.cache_resource
def initialize_system():

    model = YOLO(MODEL_PATH)

    states = {}

    for road in ROAD_SOURCES:
        states[road] = {
            "count": 0,
            "density": 0,
            "level": "Waiting",
            "status": "Starting",
            "frame": None,
            "updated": 0
        }

    return {
        "model": model,
        "states": states,
        "lock": threading.RLock(),
        "inference_lock": threading.Lock(),
        "workers_started": False,
        "lcd_message": "",
        "lcd_sent_at": 0
    }


# ==========================================================
# TRAFFIC DENSITY
# ==========================================================

def calculate_traffic(count):

    density = min(
        100,
        round((count / MAX_VEHICLES) * 100)
    )

    if count <= 5:
        level = "Low"
    elif count <= 12:
        level = "Medium"
    else:
        level = "High"

    return density, level


# ==========================================================
# COUNT VEHICLES
# ==========================================================

def count_vehicles(result):

    total = 0

    if result.boxes is None:
        return 0

    for class_id in result.boxes.cls.tolist():

        class_name = result.names[int(class_id)]

        if class_name in VEHICLE_CLASSES:
            total += 1

    return total


# ==========================================================
# PROCESS ALL FOUR ROADS
# ==========================================================

def process_road(system, road_name, source):

    states = system["states"]
    is_live = road_name in LIVE_ROADS

    while True:

        capture = None

        try:

            with system["lock"]:
                states[road_name]["status"] = "Connecting"

            if is_live:
                # Road 4: Live ESP32-CAM MJPEG stream
                video_source = source

            else:
                # Roads 1-3: Downloaded MP4 files
                with system["lock"]:
                    states[road_name]["status"] = "Downloading"

                video_source = download_video(
                    road_name,
                    source
                )

            capture = cv2.VideoCapture(video_source)

            if not capture.isOpened():
                raise RuntimeError("Cannot open video source")

            # Helps reduce buffered frames on supported backends.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            with system["lock"]:
                states[road_name]["status"] = "Running"

            while True:

                success, frame = capture.read()

                if not success:

                    if is_live:
                        raise RuntimeError(
                            "ESP32-CAM disconnected"
                        )

                    # Loop stored MP4 files
                    capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )

                    success, frame = capture.read()

                    if not success:
                        raise RuntimeError(
                            "Cannot read video frame"
                        )

                height, width = frame.shape[:2]

                if width > FRAME_WIDTH:

                    new_height = int(
                        height * FRAME_WIDTH / width
                    )

                    frame = cv2.resize(
                        frame,
                        (FRAME_WIDTH, new_height)
                    )

                # Shared model: one inference at a time
                with system["inference_lock"]:

                    results = system["model"].predict(
                        frame,
                        conf=CONFIDENCE,
                        verbose=False
                    )

                result = results[0]

                vehicle_count = count_vehicles(result)

                density, level = calculate_traffic(
                    vehicle_count
                )

                annotated = result.plot()

                cv2.putText(
                    annotated,
                    f"{road_name} | Vehicles: {vehicle_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                if not ok:
                    continue

                with system["lock"]:

                    states[road_name].update({
                        "count": vehicle_count,
                        "density": density,
                        "level": level,
                        "status": "Running",
                        "frame": encoded.tobytes(),
                        "updated": time.time()
                    })

                time.sleep(PROCESS_DELAY)

        except Exception as error:

            with system["lock"]:

                states[road_name].update({
                    "status": f"Error: {str(error)[:100]}",
                    "frame": None,
                    "updated": 0
                })

            time.sleep(5)

        finally:
            if capture is not None:
                capture.release()


# ==========================================================
# START WORKERS
# ==========================================================

@st.cache_resource
def start_workers():

    system = initialize_system()

    if not system["workers_started"]:

        system["workers_started"] = True

        for road_name, source in ROAD_SOURCES.items():

            thread = threading.Thread(
                target=process_road,
                args=(system, road_name, source),
                daemon=True
            )

            thread.start()

    return system


# ==========================================================
# SUGGEST BEST ROAD
# ==========================================================

def suggest_best_road(snapshot):

    current_time = time.time()

    available = []

    for road, data in snapshot.items():

        # Exclude disconnected or stale camera data.
        if (
            data["status"] == "Running"
            and current_time - data["updated"] < 15
        ):
            available.append((road, data))

    if not available:

        return {
            "road": "Unavailable",
            "density": 0,
            "level": "Unknown",
            "message": "Waiting for road traffic data"
        }

    best_road, best_data = min(
        available,
        key=lambda item: (
            item[1]["count"],
            item[1]["density"]
        )
    )

    return {
        "road": best_road,
        "density": best_data["density"],
        "level": best_data["level"],
        "message": (
            f"Suggested Road: {best_road} | "
            f"Vehicles: {best_data['count']} | "
            f"Traffic: {best_data['level']} | "
            f"Density: {best_data['density']}%"
        )
    }


# ==========================================================
# SEND TO ESP32 LCD
# ==========================================================

def send_to_lcd(system, snapshot, suggestion):

    if not SEND_TO_LCD:
        return "LCD sending disabled"

    if suggestion["road"] == "Unavailable":
        return "Waiting for traffic data"

    current_time = time.time()
    message = suggestion["message"]

    with system["lock"]:

        if (
            message == system["lcd_message"]
            and current_time - system["lcd_sent_at"]
            < LCD_SEND_INTERVAL
        ):
            return "LCD data already sent"

        system["lcd_message"] = message
        system["lcd_sent_at"] = current_time

    payload = {
        "suggested_road": suggestion["road"],
        "density": suggestion["density"],
        "level": suggestion["level"],
        "roads": {}
    }

    for road, data in snapshot.items():

        payload["roads"][road] = {
            "count": data["count"],
            "density": data["density"],
            "level": data["level"],
            "status": data["status"]
        }

    try:

        response = requests.post(
            ESP32_LCD_URL,
            json=payload,
            timeout=2
        )

        response.raise_for_status()

        return "LCD updated successfully"

    except Exception as error:

        with system["lock"]:
            system["lcd_message"] = ""

        return f"LCD connection failed: {error}"


# ==========================================================
# DASHBOARD
# ==========================================================

st.title("🚦 Smart City Traffic Monitoring")

st.write(
    "YOLO11 Vehicle Detection | "
    "Four Roads | Live ESP32-CAM | "
    "Ambulance Route Suggestion"
)

st.info(
    "Roads 1-3 use GitHub videos. "
    "Road 4 uses the live ESP32-CAM stream."
)

try:
    system = start_workers()

except Exception as error:
    st.error(f"System initialization failed: {error}")
    st.stop()


@st.fragment(run_every="2s")
def show_dashboard():

    with system["lock"]:

        snapshot = {
            road: data.copy()
            for road, data in system["states"].items()
        }

    current_time = time.time()

    running = sum(
        data["status"] == "Running"
        and current_time - data["updated"] < 15
        for data in snapshot.values()
    )

    total_vehicles = sum(
        data["count"]
        for data in snapshot.values()
        if (
            data["status"] == "Running"
            and current_time - data["updated"] < 15
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Vehicles", total_vehicles)
    col2.metric(
        "Active Roads",
        f"{running}/{len(ROAD_SOURCES)}"
    )
    col3.metric("Live Cameras", "1")

    st.divider()

    # ------------------------------------------------------
    # AMBULANCE SUGGESTION
    # ------------------------------------------------------

    st.subheader("🚑 Ambulance Route Suggestion")

    suggestion = suggest_best_road(snapshot)

    if suggestion["road"] == "Unavailable":
        st.warning(suggestion["message"])
    else:
        st.success(suggestion["message"])

    lcd_status = send_to_lcd(
        system,
        snapshot,
        suggestion
    )

    st.caption("ESP32 LCD: " + lcd_status)

    st.divider()

    # ------------------------------------------------------
    # ROAD MONITORING
    # ------------------------------------------------------

    st.subheader("📹 Road Monitoring")

    road_items = list(snapshot.items())

    # Display four roads in a 2 x 2 layout
    for start in range(0, len(road_items), 2):

        columns = st.columns(2)

        for column, (road, data) in zip(
            columns,
            road_items[start:start + 2]
        ):

            with column:

                st.markdown(f"### {road}")

                if road in LIVE_ROADS:
                    st.caption("🔴 Live ESP32-CAM")
                else:
                    st.caption("📁 Stored GitHub video")

                if data["frame"] is not None:

                    st.image(
                        data["frame"],
                        use_container_width=True
                    )

                else:
                    st.info(
                        "Waiting for video frames..."
                    )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Vehicles",
                    data["count"]
                )

                m2.metric(
                    "Density",
                    f"{data['density']}%"
                )

                m3.metric(
                    "Traffic",
                    data["level"]
                )

                if (
                    data["status"] == "Running"
                    and current_time - data["updated"] < 15
                ):
                    st.success("Connected")
                else:
                    st.warning(data["status"])

                if data["updated"] > 0:
                    st.caption(
                        "Last updated: "
                        + time.strftime(
                            "%H:%M:%S",
                            time.localtime(data["updated"])
                        )
                    )


show_dashboard()
