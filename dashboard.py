
import time
import threading
from pathlib import Path

import cv2
import requests
import streamlit as st
from ultralytics import YOLO

try:
    import serial
except ImportError:
    serial = None


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Smart City Traffic Monitoring",
    page_icon="🚑",
    layout="wide"
)


# ============================================================
# USER SETTINGS - CHANGE THESE
# ============================================================

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "sudhakaran2060-web/Ambulance_Route_Project/main/"
)

# Enter your ESP32-CAM's actual IP address.
# Standard CameraWebServer MJPEG stream:
ESP32_CAM_URL = "http://192.168.1.110:81/stream"

# Arduino Uno USB port
ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 115200

ENABLE_ARDUINO_LCD = True

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
FRAME_WIDTH = 640
MAX_VEHICLES = 20

PROCESS_DELAY = 0.10
STALE_SECONDS = 15
LCD_INTERVAL = 5

VEHICLE_NAMES = {
    "car",
    "bus",
    "truck",
    "motorcycle"
}


# ============================================================
# DOWNLOAD STORED VIDEOS
# ============================================================

def download_video(road_name, url):

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    filename = road_name.lower().replace(" ", "") + ".mp4"
    video_path = VIDEO_DIR / filename

    if video_path.exists() and video_path.stat().st_size > 0:
        return str(video_path)

    temp_path = video_path.with_suffix(".download")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 120)
        ) as response:

            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        temp_path.replace(video_path)

        return str(video_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


# ============================================================
# INITIALIZE YOLO AND SYSTEM
# ============================================================

@st.cache_resource
def initialize_system():

    model = YOLO(MODEL_PATH)

    states = {}

    for road_name in ROAD_SOURCES:

        states[road_name] = {
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
        "serial_lock": threading.Lock(),
        "workers_started": False,
        "arduino": None,
        "lcd_message": "",
        "lcd_sent_at": 0
    }


# ============================================================
# TRAFFIC CALCULATION
# ============================================================

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


# ============================================================
# YOLO VEHICLE COUNT
# ============================================================

def count_vehicles(result):

    count = 0

    if result.boxes is None:
        return count

    for class_id in result.boxes.cls.tolist():

        name = result.names[int(class_id)]

        if name in VEHICLE_NAMES:
            count += 1

    return count


# ============================================================
# PROCESS EACH ROAD
# ============================================================

def process_road(system, road_name, source):

    is_live = road_name in LIVE_ROADS

    while True:

        capture = None

        try:

            with system["lock"]:
                system["states"][road_name]["status"] = (
                    "Connecting camera"
                    if is_live else
                    "Downloading video"
                )

            if is_live:
                video_source = source
            else:
                video_source = download_video(
                    road_name,
                    source
                )

            # Open live MJPEG stream or stored MP4
            capture = cv2.VideoCapture(video_source)

            if not capture.isOpened():

                if is_live:
                    raise RuntimeError(
                        "ESP32-CAM not reachable. "
                        "Check IP, WiFi and /stream URL."
                    )

                raise RuntimeError(
                    "Could not open MP4 video"
                )

            capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            with system["lock"]:
                system["states"][road_name]["status"] = "Running"

            while True:

                success, frame = capture.read()

                if not success:

                    if is_live:
                        raise RuntimeError(
                            "ESP32-CAM stream stopped"
                        )

                    # Restart stored videos
                    capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )

                    success, frame = capture.read()

                    if not success:
                        raise RuntimeError(
                            "Could not read MP4 frame"
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

                # YOLO inference
                with system["inference_lock"]:

                    results = system["model"].predict(
                        frame,
                        conf=CONFIDENCE,
                        verbose=False
                    )

                result = results[0]

                count = count_vehicles(result)

                density, level = calculate_traffic(count)

                annotated = result.plot()

                cv2.putText(
                    annotated,
                    f"{road_name} - Vehicles: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Traffic: {level} ({density}%)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                success_jpg, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                if not success_jpg:
                    continue

                with system["lock"]:

                    system["states"][road_name].update({
                        "count": count,
                        "density": density,
                        "level": level,
                        "status": "Running",
                        "frame": encoded.tobytes(),
                        "updated": time.time()
                    })

                time.sleep(PROCESS_DELAY)

        except Exception as error:

            with system["lock"]:

                system["states"][road_name].update({
                    "count": 0,
                    "density": 0,
                    "level": "Unavailable",
                    "status": f"Error: {str(error)[:150]}",
                    "frame": None,
                    "updated": 0
                })

            # Automatically reconnect after failure
            time.sleep(5)

        finally:

            if capture is not None:
                capture.release()


# ============================================================
# START FOUR ROAD THREADS
# ============================================================

@st.cache_resource
def start_workers():

    system = initialize_system()

    with system["lock"]:

        if not system["workers_started"]:

            system["workers_started"] = True

            for road_name, source in ROAD_SOURCES.items():

                thread = threading.Thread(
                    target=process_road,
                    args=(system, road_name, source),
                    daemon=True,
                    name=f"YOLO-{road_name}"
                )

                thread.start()

    return system


# ============================================================
# AMBULANCE ROUTE SUGGESTION
# ============================================================

def suggest_best_road(snapshot):

    now = time.time()

    available = []

    for road_name, data in snapshot.items():

        if (
            data["status"] == "Running"
            and data["updated"] > 0
            and now - data["updated"] < STALE_SECONDS
        ):

            available.append(
                (road_name, data)
            )

    if not available:

        return {
            "road": "Unavailable",
            "count": 0,
            "density": 0,
            "level": "Unknown"
        }

    # Lowest vehicle count wins.
    # This assumes the camera views are reasonably comparable.
    best_road, best_data = min(
        available,
        key=lambda item: (
            item[1]["count"],
            item[0]
        )
    )

    return {
        "road": best_road,
        "count": best_data["count"],
        "density": best_data["density"],
        "level": best_data["level"]
    }


# ============================================================
# ARDUINO UNO SERIAL LCD
# ============================================================

def send_to_arduino(system, suggestion):

    if not ENABLE_ARDUINO_LCD:
        return "LCD disabled"

    if serial is None:
        return "Install pyserial first"

    if suggestion["road"] == "Unavailable":
        return "Waiting for traffic data"

    road = suggestion["road"]
    traffic = suggestion["level"]

    message = f"{road},{traffic}\n"

    now = time.time()

    with system["serial_lock"]:

        if (
            message == system["lcd_message"]
            and now - system["lcd_sent_at"] < LCD_INTERVAL
        ):
            return "LCD connected / no change"

        try:

            connection = system["arduino"]

            if connection is None or not connection.is_open:

                connection = serial.Serial(
                    ARDUINO_PORT,
                    ARDUINO_BAUD,
                    timeout=1,
                    write_timeout=2
                )

                system["arduino"] = connection

                # Uno resets when serial connection opens.
                time.sleep(2)

            connection.write(
                message.encode("utf-8")
            )

            connection.flush()

            system["lcd_message"] = message
            system["lcd_sent_at"] = now

            return "Arduino Uno LCD updated"

        except Exception as error:

            if system["arduino"] is not None:
                try:
                    system["arduino"].close()
                except Exception:
                    pass

            system["arduino"] = None

            return f"Arduino error: {error}"


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("🚦 Smart City Traffic Monitoring")

st.write(
    "YOLO11 | 4 Roads | ESP32-CAM Live | "
    "Ambulance Route Suggestion | Arduino Uno LCD"
)

st.info(
    "Roads 1, 2 and 3 use stored GitHub videos. "
    "Road 4 uses a live ESP32-CAM MJPEG stream."
)

with st.expander("Camera and Arduino configuration"):

    st.write("ESP32-CAM stream:", ESP32_CAM_URL)

    st.write("Arduino Uno port:", ARDUINO_PORT)

    st.write(
        "ESP32-CAM and Windows PC must be on "
        "the same reachable network."
    )

    st.write(
        "Test the camera stream in a browser "
        "before running YOLO."
    )


# ============================================================
# START SYSTEM
# ============================================================

try:

    system = start_workers()

except Exception as error:

    st.error(
        f"Failed to initialize YOLO: {error}"
    )

    st.stop()


# ============================================================
# DASHBOARD AUTO REFRESH
# ============================================================

@st.fragment(run_every="2s")
def show_dashboard():

    with system["lock"]:

        snapshot = {
            road: data.copy()
            for road, data in system["states"].items()
        }

    now = time.time()

    active_roads = [
        road
        for road, data in snapshot.items()
        if (
            data["status"] == "Running"
            and data["updated"] > 0
            and now - data["updated"] < STALE_SECONDS
        )
    ]

    total_vehicles = sum(
        snapshot[road]["count"]
        for road in active_roads
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Vehicles",
        total_vehicles
    )

    col2.metric(
        "Active Roads",
        f"{len(active_roads)}/4"
    )

    col3.metric(
        "Stored Video Roads",
        "3"
    )

    col4.metric(
        "Live ESP32-CAM",
        "Online" if "Road 4" in active_roads else "Offline"
    )

    st.divider()

    # ========================================================
    # AMBULANCE ROUTE SUGGESTION
    # ========================================================

    st.subheader("🚑 Ambulance Route Suggestion")

    suggestion = suggest_best_road(snapshot)

    if suggestion["road"] == "Unavailable":

        st.warning(
            "No valid traffic data available."
        )

    else:

        st.success(
            f"Suggested Route: {suggestion['road']}"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Vehicle Count",
            suggestion["count"]
        )

        b.metric(
            "Traffic Density",
            f"{suggestion['density']}%"
        )

        c.metric(
            "Traffic Level",
            suggestion["level"]
        )

    # ========================================================
    # SEND TO ARDUINO UNO
    # ========================================================

    lcd_status = send_to_arduino(
        system,
        suggestion
    )

    st.caption("Arduino LCD: " + lcd_status)

    st.divider()

    # ========================================================
    # ROAD VIDEO DISPLAY
    # ========================================================

    st.subheader("📹 YOLO Road Monitoring")

    road_items = list(snapshot.items())

    for start in range(0, len(road_items), 2):

        columns = st.columns(2)

        for column, (road, data) in zip(
            columns,
            road_items[start:start + 2]
        ):

            with column:

                st.markdown(f"### {road}")

                if road in LIVE_ROADS:
                    st.caption("🔴 ESP32-CAM Live Stream")
                else:
                    st.caption("📁 GitHub Stored Video")

                if data["frame"] is not None:

                    st.image(
                        data["frame"],
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Waiting for video..."
                    )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Vehicles",
                    data["count"]
                )

                c2.metric(
                    "Density",
                    f"{data['density']}%"
                )

                c3.metric(
                    "Traffic",
                    data["level"]
                )

                if road in active_roads:

                    st.success("Connected")

                else:

                    st.error(data["status"])

                if data["updated"] > 0:

                    st.caption(
                        "Updated: "
                        + time.strftime(
                            "%H:%M:%S",
                            time.localtime(data["updated"])
                        )
                    )


show_dashboard()
