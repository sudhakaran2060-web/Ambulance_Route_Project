
import time
import threading
from pathlib import Path

import cv2
import requests
import streamlit as st
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart City Ambulance Route",
    page_icon="🚑",
    layout="wide",
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
MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.35
MAX_VEHICLES = 20
FRAME_WIDTH = 640
PROCESS_DELAY = 0.15

# Change this to the ESP32 IP address printed in Serial Monitor.
# Example only: replace it with your actual ESP32 address.
ESP32_URL = "http://192.168.1.100/traffic"

# LCD sending works directly when this dashboard and ESP32
# are on the same reachable network.
SEND_TO_LCD = True


# ============================================================
# VIDEO DOWNLOAD
# ============================================================

def download_video(road_name, url):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = road_name.lower().replace(" ", "")
    video_path = VIDEO_DIR / f"{safe_name}.mp4"

    # Reuse downloaded videos.
    if video_path.exists() and video_path.stat().st_size > 0:
        return str(video_path)

    temporary_path = video_path.with_suffix(".download")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 120),
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

    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()

        raise RuntimeError(
            f"Could not download {road_name}: {error}"
        )


# ============================================================
# SHARED YOLO MODEL AND ROAD STATES
# ============================================================

@st.cache_resource
def initialize_system():
    model = YOLO(MODEL_PATH)

    road_states = {
        road_name: {
            "count": 0,
            "density": 0,
            "level": "Waiting",
            "status": "Starting",
            "updated": time.time(),
            "frame": None,
        }
        for road_name in ROAD_SOURCES
    }

    return {
        "model": model,
        "states": road_states,
        "lock": threading.RLock(),
        "inference_lock": threading.Lock(),
        "workers_started": False,
        "lcd_last_message": "",
        "lcd_last_sent": 0.0,
    }


# ============================================================
# TRAFFIC CALCULATIONS
# ============================================================

def calculate_traffic(vehicle_count):
    density = min(
        100,
        round(vehicle_count / MAX_VEHICLES * 100),
    )

    if vehicle_count <= 5:
        level = "Low"
    elif vehicle_count <= 12:
        level = "Medium"
    else:
        level = "High"

    return density, level


def count_vehicles(result):
    vehicle_names = {
        "car",
        "bus",
        "truck",
        "motorcycle",
    }

    total = 0

    if result.boxes is None:
        return total

    for class_id in result.boxes.cls.tolist():
        class_name = result.names[int(class_id)]

        if class_name in vehicle_names:
            total += 1

    return total


# ============================================================
# PROCESS ONE ROAD VIDEO
# ============================================================

def process_road(system, road_name, video_url):
    states = system["states"]

    while True:
        capture = None

        try:
            with system["lock"]:
                states[road_name]["status"] = "Downloading video"

            video_path = download_video(road_name, video_url)

            capture = cv2.VideoCapture(video_path)

            if not capture.isOpened():
                raise RuntimeError(
                    "OpenCV could not open the video file."
                )

            with system["lock"]:
                states[road_name]["status"] = "Running"

            while True:
                success, frame = capture.read()

                # Restart the video after reaching the end.
                if not success:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = capture.read()

                if not success:
                    raise RuntimeError(
                        "Could not read a frame from the video."
                    )

                height, width = frame.shape[:2]

                if width > FRAME_WIDTH:
                    new_height = int(
                        height * FRAME_WIDTH / width
                    )
                    frame = cv2.resize(
                        frame,
                        (FRAME_WIDTH, new_height),
                    )

                # Only one thread performs inference at a time.
                with system["inference_lock"]:
                    results = system["model"].predict(
                        frame,
                        conf=CONFIDENCE,
                        verbose=False,
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
                    0.7,
                    (0, 255, 0),
                    2,
                )

                encode_ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75],
                )

                jpeg_bytes = (
                    encoded.tobytes() if encode_ok else None
                )

                with system["lock"]:
                    states[road_name].update({
                        "count": vehicle_count,
                        "density": density,
                        "level": level,
                        "status": "Running",
                        "updated": time.time(),
                        "frame": jpeg_bytes,
                    })

                time.sleep(PROCESS_DELAY)

        except Exception as error:
            with system["lock"]:
                states[road_name]["status"] = (
                    f"Error: {str(error)[:120]}"
                )
                states[road_name]["frame"] = None

            # Retry the road after a short delay.
            time.sleep(5)

        finally:
            if capture is not None:
                capture.release()


# ============================================================
# START BACKGROUND WORKERS
# ============================================================

@st.cache_resource
def start_workers():
    system = initialize_system()

    if not system["workers_started"]:
        system["workers_started"] = True

        for road_name, video_url in ROAD_SOURCES.items():
            thread = threading.Thread(
                target=process_road,
                args=(system, road_name, video_url),
                daemon=True,
                name=f"worker-{road_name}",
            )
            thread.start()

    return system


# ============================================================
# AMBULANCE ROUTE SUGGESTION
# ============================================================

def get_road_suggestion(snapshot):
    running_roads = [
        (road_name, data)
        for road_name, data in snapshot.items()
        if data["status"] == "Running"
    ]

    if not running_roads:
        return {
            "road": "Unavailable",
            "density": 100,
            "level": "Unknown",
            "message": "No road data is available yet.",
        }

    # Prefer lowest density; use vehicle count as a tie-breaker.
    best_road, best_data = min(
        running_roads,
        key=lambda item: (
            item[1]["density"],
            item[1]["count"],
        ),
    )

    return {
        "road": best_road,
        "density": best_data["density"],
        "level": best_data["level"],
        "message": (
            f"Suggested road: {best_road} | "
            f"Traffic: {best_data['level']} | "
            f"Density: {best_data['density']}%"
        ),
    }


# ============================================================
# SEND SUGGESTION TO ESP32 LCD
# ============================================================

def send_to_lcd(system, snapshot, suggestion):
    if not SEND_TO_LCD:
        return "LCD sending is disabled."

    if suggestion["road"] == "Unavailable":
        return "Waiting for traffic data."

    message = suggestion["message"]
    now = time.time()

    # Avoid repeatedly sending an unchanged message.
    with system["lock"]:
        if (
            message == system["lcd_last_message"]
            and now - system["lcd_last_sent"] < 10
        ):
            return "LCD update unchanged."

        # Reserve the send slot to reduce duplicate requests.
        system["lcd_last_message"] = message
        system["lcd_last_sent"] = now

    payload = {
        "suggested_road": suggestion["road"],
        "density": suggestion["density"],
        "level": suggestion["level"],
        "roads": {
            road_name: {
                "count": data["count"],
                "density": data["density"],
                "level": data["level"],
            }
            for road_name, data in snapshot.items()
        },
    }

    try:
        response = requests.post(
            ESP32_URL,
            json=payload,
            timeout=2,
        )
        response.raise_for_status()
        return "ESP32 LCD updated."

    except Exception as error:
        # Permit retry after a failed send.
        with system["lock"]:
            system["lcd_last_message"] = ""
        return f"LCD not reachable: {error}"


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("🚦 Smart City Traffic Monitoring")
st.caption(
    "YOLO vehicle detection | Three roads | "
    "Ambulance route suggestion | ESP32 LCD"
)

st.info(
    "This dashboard analyses stored MP4 videos, not live camera "
    "feeds. Route suggestions are based on detected vehicle counts "
    "and should be treated as decision support, not guaranteed "
    "real-world fastest routes."
)

try:
    system = start_workers()
except Exception as error:
    st.error(f"Could not start the YOLO system: {error}")
    st.stop()


@st.fragment(run_every="2s")
def show_dashboard():
    with system["lock"]:
        snapshot = {
            road_name: data.copy()
            for road_name, data in system["states"].items()
        }

    total_vehicles = sum(
        data["count"] for data in snapshot.values()
    )

    running_count = sum(
        data["status"] == "Running"
        for data in snapshot.values()
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Detected vehicles", total_vehicles)
    metric2.metric("Roads running", f"{running_count}/3")
    metric3.metric("Roads monitored", len(snapshot))

    st.divider()
    st.subheader("🚑 Ambulance Route Suggestion")

    suggestion = get_road_suggestion(snapshot)

    if suggestion["road"] == "Unavailable":
        st.warning(suggestion["message"])
    else:
        st.success(suggestion["message"])
        st.caption(
            "The suggestion selects the available road with the "
            "lowest estimated traffic density."
        )

    lcd_status = send_to_lcd(
        system,
        snapshot,
        suggestion,
    )

    st.caption(f"LCD connection: {lcd_status}")

    st.divider()
    st.subheader("Live YOLO Analysis of Stored Videos")

    columns = st.columns(3)

    for column, (road_name, data) in zip(
        columns,
        snapshot.items(),
    ):
        with column:
            st.markdown(f"### {road_name}")

            if data["frame"] is not None:
                st.image(
                    data["frame"],
                    use_container_width=True,
                )
            else:
                st.info("Waiting for video processing...")

            st.metric("Vehicles", data["count"])
            st.metric("Traffic density", f"{data['density']}%")
            st.metric("Traffic level", data["level"])

            if data["status"] == "Running":
                st.success("Running")
            else:
                st.warning(data["status"])

            st.caption(
                "Updated "
                + time.strftime(
                    "%H:%M:%S",
                    time.localtime(data["updated"]),
                )
            )


show_dashboard()
