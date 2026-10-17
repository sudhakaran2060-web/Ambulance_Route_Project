from flask import Flask, render_template_string, jsonify, Response
from ultralytics import YOLO
import cv2
import requests
import threading
import time
import os

app = Flask(__name__)

# ---------------- CONFIGURATION ----------------

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "sudhakaran2060-web/Ambulance_Route_Project/main/"
)

ROADS = {
    "Road 1": GITHUB_BASE + "road1.mp4",
    "Road 2": GITHUB_BASE + "road2.mp4",
    "Road 3": GITHUB_BASE + "road3.mp4",
}

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
MAX_DENSITY_VEHICLES = 20

os.makedirs("videos", exist_ok=True)

# ---------------- YOLO MODEL ----------------

model = YOLO(MODEL_PATH)

lock = threading.RLock()

road_data = {
    road: {
        "vehicle_count": 0,
        "density": 0,
        "traffic_level": "Waiting",
        "status": "Initializing",
        "updated": "",
        "frame": None,
    }
    for road in ROADS
}


# ---------------- DOWNLOAD VIDEO ----------------

def download_video(road, url):
    filename = os.path.join(
        "videos", road.lower().replace(" ", "") + ".mp4"
    )

    # Reuse the downloaded file
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return filename

    try:
        print(f"[{road}] Downloading video...")

        response = requests.get(
            url,
            stream=True,
            timeout=(15, 120)
        )
        response.raise_for_status()

        temp_file = filename + ".part"
        total_bytes = 0

        with open(temp_file, "wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    file.write(chunk)
                    total_bytes += len(chunk)

        if total_bytes == 0:
            raise ValueError("Downloaded video is empty")

        os.replace(temp_file, filename)

        print(f"[{road}] Download complete")
        return filename

    except Exception as error:
        print(f"[{road}] Download failed: {error}")

        if os.path.exists(filename + ".part"):
            os.remove(filename + ".part")

        return None


# ---------------- TRAFFIC ANALYSIS ----------------

def process_road(road, url):
    while True:
        filename = download_video(road, url)

        if not filename:
            with lock:
                road_data[road]["status"] = (
                    "Cannot access video. Check GitHub URL."
                )
                road_data[road]["traffic_level"] = "Unavailable"

            time.sleep(15)
            continue

        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            with lock:
                road_data[road]["status"] = (
                    "OpenCV cannot open video"
                )

            cap.release()
            time.sleep(10)
            continue

        print(f"[{road}] YOLO analysis started")

        with lock:
            road_data[road]["status"] = "Running"

        while True:
            success, frame = cap.read()

            # Replay video when it reaches the end
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()

                if not success:
                    break

            # Resize to reduce CPU usage
            height, width = frame.shape[:2]

            if width > 640:
                new_height = int(height * 640 / width)
                frame = cv2.resize(frame, (640, new_height))

            try:
                # Use one shared model safely across all roads
                with lock:
                    result = model.predict(
                        frame,
                        conf=CONFIDENCE,
                        verbose=False
                    )[0]

                vehicle_count = 0

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        label = model.names[class_id]

                        if label in VEHICLE_CLASSES:
                            vehicle_count += 1

                # Density estimate based on the current frame
                density = min(
                    100,
                    round(
                        vehicle_count
                        * 100
                        / MAX_DENSITY_VEHICLES
                    )
                )

                if vehicle_count <= 5:
                    traffic_level = "Low"
                elif vehicle_count <= 12:
                    traffic_level = "Medium"
                else:
                    traffic_level = "High"

                annotated_frame = result.plot()

                # Encode annotated frame for the dashboard
                success_jpg, encoded = cv2.imencode(
                    ".jpg", annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                with lock:
                    road_data[road].update({
                        "vehicle_count": vehicle_count,
                        "density": density,
                        "traffic_level": traffic_level,
                        "status": "Running",
                        "updated": time.strftime("%H:%M:%S"),
                        "frame": (
                            encoded.tobytes()
                            if success_jpg else None
                        ),
                    })

            except Exception as error:
                print(f"[{road}] Detection error: {error}")

                with lock:
                    road_data[road]["status"] = (
                        "Detection error: " + str(error)
                    )

                time.sleep(1)

            time.sleep(0.03)

        cap.release()

        with lock:
            road_data[road]["status"] = (
                "Video ended or cannot read frames"
            )

        time.sleep(3)


# ---------------- DASHBOARD HTML ----------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart City Traffic Monitoring</title>

<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #101827;
    color: white;
}
header {
    padding: 24px;
    text-align: center;
    background: #17253a;
}
header h1 {
    margin: 0 0 8px;
}
header p {
    color: #bdcbe0;
}
.container {
    padding: 20px;
    max-width: 1500px;
    margin: auto;
}
.summary, .roads {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
}
.card, .road {
    background: #1b2a40;
    border: 1px solid #34465e;
    border-radius: 12px;
    overflow: hidden;
}
.card {
    padding: 18px;
}
.card h3 {
    color: #c0cee1;
    font-size: 15px;
}
.value {
    font-size: 30px;
    font-weight: bold;
}
.road h2 {
    padding: 16px;
    margin: 0;
}
.video {
    width: 100%;
    aspect-ratio: 16/9;
    background: #05080d;
    object-fit: contain;
    display: block;
}
.metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    padding: 14px;
}
.metric {
    background: #101b2b;
    padding: 12px;
    border-radius: 8px;
}
.metric span {
    display: block;
    font-size: 12px;
    color: #bdcbe0;
    margin-bottom: 7px;
}
.metric strong {
    font-size: 20px;
}
.status {
    padding: 0 14px 14px;
    color: #bdcbe0;
    overflow-wrap: anywhere;
    font-size: 13px;
}
.low { color: #4ade80; }
.medium { color: #facc15; }
.high { color: #fb7185; }

@media(max-width: 950px) {
    .roads, .summary {
        grid-template-columns: 1fr 1fr;
    }
}
@media(max-width: 600px) {
    .roads, .summary {
        grid-template-columns: 1fr;
    }
}
</style>
</head>

<body>

<header>
    <h1>Smart City Traffic Monitoring</h1>
    <p>YOLO Vehicle Detection | Three Road Monitoring</p>
</header>

<div class="container">

<section class="summary">
    <div class="card">
        <h3>Roads Configured</h3>
        <div class="value">3</div>
    </div>

    <div class="card">
        <h3>Total Vehicles in Current Frames</h3>
        <div class="value" id="total">0</div>
    </div>

    <div class="card">
        <h3>Roads Running</h3>
        <div class="value" id="running">0 / 3</div>
    </div>
</section>

<br>

<section class="roads">
{% for road in roads %}
<div class="road">
    <h2>{{ road }}</h2>

    <img class="video"
         src="/video/{{ road }}"
         alt="{{ road }} video feed">

    <div class="metrics">
        <div class="metric">
            <span>Vehicle Count</span>
            <strong id="{{ road }}_count">0</strong>
        </div>

        <div class="metric">
            <span>Traffic Density</span>
            <strong id="{{ road }}_density">0%</strong>
        </div>

        <div class="metric">
            <span>Traffic Level</span>
            <strong id="{{ road }}_level">Waiting</strong>
        </div>

        <div class="metric">
            <span>Last Update</span>
            <strong id="{{ road }}_updated">--</strong>
        </div>
    </div>

    <div class="status" id="{{ road }}_status">
        Connecting to video...
    </div>
</div>
{% endfor %}
</section>

</div>

<script>
async function refreshDashboard() {
    try {
        const response = await fetch("/data", {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("Unable to read dashboard data");
        }

        const data = await response.json();

        let total = 0;
        let running = 0;

        for (const [road, item] of Object.entries(data)) {
            document.getElementById(road + "_count").textContent =
                item.vehicle_count;

            document.getElementById(road + "_density").textContent =
                item.density + "%";

            const level = document.getElementById(road + "_level");
            level.textContent = item.traffic_level;
            level.className = "";

            if (item.traffic_level === "Low") {
                level.classList.add("low");
            } else if (item.traffic_level === "Medium") {
                level.classList.add("medium");
            } else if (item.traffic_level === "High") {
                level.classList.add("high");
            }

            document.getElementById(road + "_updated").textContent =
                item.updated || "--";

            document.getElementById(road + "_status").textContent =
                item.status;

            total += Number(item.vehicle_count || 0);

            if (item.status === "Running") {
                running++;
            }
        }

        document.getElementById("total").textContent = total;
        document.getElementById("running").textContent =
            running + " / 3";

    } catch (error) {
        console.error(error);
    }
}

refreshDashboard();
setInterval(refreshDashboard, 2000);
</script>

</body>
</html>
"""


# ---------------- FLASK ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(
        HTML_PAGE,
        roads=list(ROADS.keys())
    )


@app.route("/data")
def get_data():
    with lock:
        return jsonify({
            road: {
                key: value
                for key, value in data.items()
                if key != "frame"
            }
            for road, data in road_data.items()
        })


@app.route("/video/<path:road>")
def video_feed(road):
    if road not in ROADS:
        return "Unknown road", 404

    def generate():
        while True:
            with lock:
                frame = road_data[road]["frame"]

            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame + b"\r\n"
                )

            time.sleep(0.08)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "roads": list(ROADS.keys())
    })


# ---------------- START APPLICATION ----------------

if __name__ == "__main__":

    for road, url in ROADS.items():
        thread = threading.Thread(
            target=process_road,
            args=(road, url),
            daemon=True
        )
        thread.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
