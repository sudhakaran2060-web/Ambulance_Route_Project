
import os
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import pandas as pd
import requests
import streamlit as st
from ultralytics import YOLO

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart City AI Traffic Command Center",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# VIDEO SOURCES
# ==========================================================

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "sudhakaran2060-web/"
    "Ambulance_Route_Project/main/"
)

ROAD_SOURCES = {
    "Road 1": GITHUB_BASE + "road1.mp4",
    "Road 2": GITHUB_BASE + "road2.mp4",
    "Road 3": GITHUB_BASE + "road3.mp4",
}

VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

REFERENCE_VEHICLES = 20

# ==========================================================
# PROFESSIONAL DARK THEME
# ==========================================================

st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(145deg, #07101f, #0c1830);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #0b1628;
    border-right: 1px solid #26374f;
}

.block-container {
    max-width: 1650px;
    padding-top: 1.4rem;
}

[data-testid="stMetric"] {
    background: #101f35;
    border: 1px solid #263c57;
    border-radius: 15px;
    padding: 17px;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #334155;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #26374f !important;
    border-radius: 15px !important;
}

.hero {
    padding: 25px;
    background: linear-gradient(
        110deg, #102d52, #102037
    );
    border: 1px solid #23527b;
    border-radius: 18px;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 29px;
    font-weight: 800;
    color: #ffffff;
}

.hero-subtitle {
    color: #a7c5df;
    font-size: 14px;
    margin-top: 6px;
}

.route-card {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        115deg, #064e3b, #0f352d
    );
    border: 1px solid #10b981;
    margin: 15px 0;
}

.route-label {
    color: #a7f3d0;
    font-size: 13px;
    letter-spacing: 1px;
    font-weight: 700;
}

.route-name {
    color: #ffffff;
    font-size: 38px;
    font-weight: 800;
    margin: 6px 0;
}

.route-detail {
    color: #d1fae5;
    font-size: 16px;
}

.alert-card {
    background: #3a1923;
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 15px;
    color: #fecaca;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# DOWNLOAD VIDEOS FROM GITHUB
# ==========================================================

def download_video(road, url):

    filename = VIDEO_DIR / (
        road.lower().replace(" ", "") + ".mp4"
    )

    if filename.exists() and filename.stat().st_size > 0:
        return str(filename)

    temp = filename.with_suffix(".part")

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=(15, 120)
        )
        response.raise_for_status()

        with open(temp, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

        if temp.stat().st_size == 0:
            raise ValueError("Video is empty")

        os.replace(temp, filename)
        return str(filename)

    except Exception as error:
        print(f"{road} download error: {error}")

        if temp.exists():
            temp.unlink()

        return None

# ==========================================================
# INITIALIZE YOLO AND ROAD DATA
# ==========================================================

@st.cache_resource
def initialize_system():

    model = YOLO("yolo11n.pt")

    state = {}

    for road in ROAD_SOURCES:

        state[road] = {
            "frame": None,
            "count": 0,
            "density": 0,
            "level": "WAITING",
            "status": "Initializing",
            "classes": {
                name: 0
                for name in VEHICLE_CLASSES.values()
            },
            "fps": 0.0,
            "updated": "--",
            "history": deque(maxlen=60)
        }

    return {
        "model": model,
        "state": state,
        "lock": threading.RLock(),
        "inference_lock": threading.Lock(),
        "workers_started": False
    }


system = initialize_system()

# ==========================================================
# TRAFFIC LEVEL
# ==========================================================

def get_traffic_level(count):

    if count <= 5:
        return "LOW"

    if count <= 12:
        return "MODERATE"

    return "HIGH"

# ==========================================================
# YOLO VIDEO PROCESSING
# ==========================================================

def process_road(road, url):

    while True:

        with system["lock"]:
            system["state"][road]["status"] = (
                "Loading video"
            )

        path = download_video(road, url)

        if not path:

            with system["lock"]:
                system["state"][road]["status"] = (
                    "Source error"
                )
                system["state"][road]["frame"] = None

            time.sleep(15)
            continue

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():

            with system["lock"]:
                system["state"][road]["status"] = (
                    "Video error"
                )
                system["state"][road]["frame"] = None

            cap.release()
            time.sleep(5)
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if video_fps <= 0 or video_fps > 120:
            video_fps = 20

        frame_interval = 1.0 / video_fps

        while True:

            start = time.monotonic()

            success, frame = cap.read()

            if not success:

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()

                if not success:
                    break

            height, width = frame.shape[:2]

            if width > 640:

                frame = cv2.resize(
                    frame,
                    (640, int(height * 640 / width))
                )

            try:

                with system["inference_lock"]:

                    result = system["model"].predict(
                        frame,
                        classes=list(VEHICLE_CLASSES),
                        conf=0.35,
                        imgsz=640,
                        verbose=False
                    )[0]

                class_counts = {
                    name: 0
                    for name in VEHICLE_CLASSES.values()
                }

                if result.boxes is not None:

                    for class_id in result.boxes.cls:

                        label = VEHICLE_CLASSES.get(
                            int(class_id.item())
                        )

                        if label:
                            class_counts[label] += 1

                count = sum(class_counts.values())

                density = min(
                    100,
                    round(
                        count * 100 /
                        REFERENCE_VEHICLES
                    )
                )

                level = get_traffic_level(count)

                annotated = result.plot()

                ok, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                elapsed = time.monotonic() - start

                fps = (
                    round(1 / elapsed, 1)
                    if elapsed > 0 else 0
                )

                with system["lock"]:

                    data = system["state"][road]

                    data["frame"] = (
                        jpeg.tobytes()
                        if ok else None
                    )

                    data["count"] = count
                    data["density"] = density
                    data["level"] = level
                    data["classes"] = class_counts
                    data["fps"] = fps
                    data["status"] = "Monitoring"
                    data["updated"] = time.strftime(
                        "%H:%M:%S"
                    )

                    data["history"].append({
                        "time": time.time(),
                        "vehicles": count,
                        "density": density
                    })

            except Exception as error:

                print(f"{road}: {error}")

                with system["lock"]:
                    system["state"][road]["status"] = (
                        "Analysis error"
                    )

                time.sleep(1)

            elapsed = time.monotonic() - start

            time.sleep(
                max(0, frame_interval - elapsed)
            )

        cap.release()

        with system["lock"]:
            system["state"][road]["status"] = (
                "Reconnecting"
            )
            system["state"][road]["frame"] = None

        time.sleep(2)

# ==========================================================
# START VIDEO WORKERS
# ==========================================================

with system["lock"]:

    if not system["workers_started"]:

        for road, url in ROAD_SOURCES.items():

            worker = threading.Thread(
                target=process_road,
                args=(road, url),
                daemon=True
            )

            worker.start()

        system["workers_started"] = True

# ==========================================================
# AMBULANCE ROUTE ALGORITHM
# ==========================================================

def calculate_routes(
    snapshot,
    distances,
    free_flow_speed,
    congestion_weight
):

    routes = []

    for road, data in snapshot.items():

        if data["status"] != "Monitoring":
            continue

        distance = distances[road]
        density = data["density"]

        slowdown = (
            1
            + congestion_weight * density / 100
        )

        estimated_speed = (
            free_flow_speed / slowdown
        )

        travel_time = (
            distance / estimated_speed
        ) * 60

        normal_time = (
            distance / free_flow_speed
        ) * 60

        delay = travel_time - normal_time

        routes.append({
            "Road": road,
            "Distance (km)": round(distance, 2),
            "Vehicles": data["count"],
            "Traffic Index (%)": density,
            "Traffic Level": data["level"],
            "Estimated Time (min)": round(
                travel_time, 1
            ),
            "Estimated Delay (min)": round(
                delay, 1
            ),
            "Estimated Speed (km/h)": round(
                estimated_speed, 1
            )
        })

    routes.sort(
        key=lambda item: (
            item["Estimated Time (min)"],
            item["Traffic Index (%)"]
        )
    )

    return routes

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

with st.sidebar:

    st.title("🚦 CITY CONTROL")

    st.caption("AI TRAFFIC INTELLIGENCE")

    st.divider()

    st.subheader("📹 Video Sources")

    for road in ROAD_SOURCES:
        st.write(f"● {road} — GitHub MP4")

    st.divider()

    st.subheader("🚑 Ambulance Simulation")

    st.caption(
        "Configure three alternative routes "
        "between the same origin and destination."
    )

    distances = {
        "Road 1": st.number_input(
            "Road 1 Distance (km)",
            min_value=0.1,
            value=4.0,
            step=0.5
        ),
        "Road 2": st.number_input(
            "Road 2 Distance (km)",
            min_value=0.1,
            value=5.0,
            step=0.5
        ),
        "Road 3": st.number_input(
            "Road 3 Distance (km)",
            min_value=0.1,
            value=3.5,
            step=0.5
        )
    }

    free_flow_speed = st.slider(
        "Free-flow Speed (km/h)",
        min_value=10,
        max_value=60,
        value=35,
        step=5
    )

    congestion_weight = st.slider(
        "Congestion Sensitivity",
        min_value=0.2,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    st.divider()

    st.subheader("🧠 AI Model")

    st.write("YOLO11 Nano")
    st.write("Vehicle Detection")
    st.write("Confidence: 35%")

    st.divider()

    st.info(
        "Video feeds are prerecorded and replayed. "
        "Route times are simulation estimates."
    )

# ==========================================================
# MAIN DASHBOARD
# ==========================================================

@st.fragment(run_every="2s")
def dashboard():

    with system["lock"]:

        snapshot = {}

        for road, data in system["state"].items():

            snapshot[road] = {
                **{
                    key: value
                    for key, value in data.items()
                    if key != "history"
                },
                "classes": data["classes"].copy(),
                "history": list(data["history"])
            }

    active = {
        road: data
        for road, data in snapshot.items()
        if data["status"] == "Monitoring"
    }

    total_vehicles = sum(
        data["count"]
        for data in active.values()
    )

    avg_density = (
        round(
            sum(
                data["density"]
                for data in active.values()
            ) / len(active)
        )
        if active else 0
    )

    high_roads = sum(
        data["level"] == "HIGH"
        for data in active.values()
    )

    routes = calculate_routes(
        snapshot,
        distances,
        free_flow_speed,
        congestion_weight
    )

    # ======================================================
    # HEADER
    # ======================================================

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🚑 SMART CITY TRAFFIC COMMAND CENTER
        </div>
        <div class="hero-subtitle">
            AI-Powered Traffic Monitoring |
            Ambulance Route Intelligence |
            YOLO11 Vehicle Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # TOP METRICS
    # ======================================================

    a, b, c, d, e = st.columns(5)

    a.metric("MONITORED ROADS", "03")

    b.metric(
        "ACTIVE SOURCES",
        f"{len(active)}/3"
    )

    c.metric(
        "VEHICLES IN FRAMES",
        total_vehicles
    )

    d.metric(
        "TRAFFIC INDEX",
        f"{avg_density}%"
    )

    e.metric(
        "HIGH TRAFFIC ROADS",
        high_roads
    )

    st.divider()

    # ======================================================
    # ATTRACTIVE AMBULANCE ROUTE PANEL
    # ======================================================

    st.subheader(
        "🚑 AI-Assisted Ambulance Route Recommendation"
    )

    if routes:

        best = routes[0]

        st.markdown(f"""
        <div class="route-card">
            <div class="route-label">
                🏆 PREFERRED ROUTE — SIMULATION
            </div>
            <div class="route-name">
                🚑 {best['Road']}
            </div>
            <div class="route-detail">
                Estimated Journey:
                <b>{best['Estimated Time (min)']} min</b>
                &nbsp; | &nbsp;
                Distance:
                <b>{best['Distance (km)']} km</b>
                &nbsp; | &nbsp;
                Traffic:
                <b>{best['Traffic Level']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        x, y, z, w = st.columns(4)

        x.metric(
            "PREFERRED ROAD",
            best["Road"]
        )

        y.metric(
            "ESTIMATED ETA",
            f"{best['Estimated Time (min)']} min"
        )

        z.metric(
            "SIMULATED DELAY",
            f"{best['Estimated Delay (min)']} min"
        )

        w.metric(
            "TRAFFIC SCORE",
            f"{best['Traffic Index (%)']}%"
        )

        st.write("")

        st.markdown("#### 🏁 Route Ranking")

        ranking_columns = st.columns(
            len(routes)
        )

        medals = ["🥇", "🥈", "🥉"]

        for index, (column, route) in enumerate(
            zip(ranking_columns, routes)
        ):

            with column:

                with st.container(border=True):

                    st.markdown(
                        f"### {medals[index]} "
                        f"{route['Road']}"
                    )

                    st.metric(
                        "Estimated Time",
                        f"{route['Estimated Time (min)']} min"
                    )

                    st.write(
                        f"**Distance:** "
                        f"{route['Distance (km)']} km"
                    )

                    st.write(
                        f"**Vehicles:** "
                        f"{route['Vehicles']}"
                    )

                    st.write(
                        f"**Traffic Index:** "
                        f"{route['Traffic Index (%)']}%"
                    )

                    if route["Traffic Level"] == "LOW":
                        st.success("LOW TRAFFIC")

                    elif route["Traffic Level"] == "MODERATE":
                        st.warning("MODERATE TRAFFIC")

                    else:
                        st.error("HIGH TRAFFIC")

        st.markdown(
            "#### ⏱️ Estimated Journey Comparison"
        )

        route_df = pd.DataFrame(routes)

        st.bar_chart(
            route_df.set_index("Road")[
                "Estimated Time (min)"
            ],
            color="#38bdf8"
        )

        if len(routes) > 1:

            slowest = routes[-1]

            saving = (
                slowest["Estimated Time (min)"]
                - best["Estimated Time (min)"]
            )

            st.info(
                f"💡 {best['Road']} has the lowest "
                f"simulated journey time. "
                f"The estimated difference from "
                f"{slowest['Road']} is "
                f"{saving:.1f} minutes."
            )

    else:

        st.warning(
            "Waiting for traffic analysis "
            "to calculate route recommendations."
        )

    st.divider()

    # ======================================================
    # VIDEO MONITORING
    # ======================================================

    st.subheader("📹 AI Traffic Surveillance")

    columns = st.columns(3)

    for column, (road, data) in zip(
        columns,
        snapshot.items()
    ):

        with column:

            with st.container(border=True):

                st.markdown(f"### 📍 {road}")

                st.caption(
                    "GITHUB VIDEO | YOLO11 ANALYSIS"
                )

                if data["frame"] is not None:

                    st.image(
                        data["frame"],
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Waiting for video..."
                    )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Vehicles",
                    data["count"]
                )

                c2.metric(
                    "Traffic Index",
                    f'{data["density"]}%'
                )

                if data["status"] != "Monitoring":
                    st.warning(data["status"])

                elif data["level"] == "LOW":
                    st.success("🟢 LOW TRAFFIC")

                elif data["level"] == "MODERATE":
                    st.warning("🟡 MODERATE TRAFFIC")

                else:
                    st.error("🔴 HIGH TRAFFIC")

                st.caption(
                    f"Processing: {data['fps']} FPS"
                )

                st.caption(
                    f"Updated: {data['updated']}"
                )

    st.divider()

    # ======================================================
    # TRAFFIC ANALYTICS
    # ======================================================

    st.subheader("📊 Traffic Analytics")

    left, right = st.columns([3, 2])

    with left:

        st.markdown("#### Vehicle Count by Road")

        comparison = pd.DataFrame([
            {
                "Road": road,
                "Vehicles": data["count"]
            }
            for road, data in active.items()
        ])

        if not comparison.empty:

            st.bar_chart(
                comparison.set_index("Road"),
                color="#38bdf8"
            )

        else:
            st.info("Waiting for data")

    with right:

        st.markdown("#### Vehicle Classification")

        totals = {
            name: sum(
                data["classes"][name]
                for data in active.values()
            )
            for name in VEHICLE_CLASSES.values()
        }

        classification = pd.DataFrame([
            {
                "Vehicle Type": name,
                "Detected": count
            }
            for name, count in totals.items()
        ])

        st.dataframe(
            classification,
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # TRAFFIC HISTORY
    # ======================================================

    st.subheader("📈 Recent Traffic Trends")

    history_rows = []

    for road, data in snapshot.items():

        for point in data["history"]:

            history_rows.append({
                "Time": pd.to_datetime(
                    point["time"],
                    unit="s"
                ),
                "Road": road,
                "Vehicles": point["vehicles"]
            })

    if history_rows:

        history_df = pd.DataFrame(history_rows)

        history_df = history_df.pivot_table(
            index="Time",
            columns="Road",
            values="Vehicles",
            aggfunc="mean"
        ).sort_index()

        st.line_chart(history_df)

    else:

        st.info(
            "Collecting traffic trend data..."
        )

    st.divider()

    # ======================================================
    # TRAFFIC ALERTS
    # ======================================================

    st.subheader("🚨 Congestion & System Alerts")

    alerts_found = False

    for road, data in snapshot.items():

        if data["status"] != "Monitoring":

            st.warning(
                f"{road}: {data['status']}"
            )

            alerts_found = True

        elif data["level"] == "HIGH":

            st.error(
                f"🔴 {road}: HIGH TRAFFIC — "
                f"{data['count']} vehicles detected."
            )

            alerts_found = True

        elif data["level"] == "MODERATE":

            st.warning(
                f"🟡 {road}: Moderate traffic — "
                f"{data['count']} vehicles detected."
            )

            alerts_found = True

    if not alerts_found:

        st.success(
            "All monitored roads currently "
            "show LOW traffic classification."
        )

    st.divider()

    # ======================================================
    # ROAD STATUS TABLE
    # ======================================================

    st.subheader("🛣️ Road Monitoring Report")

    summary = pd.DataFrame([
        {
            "Road": road,
            "Video Source": "GitHub MP4",
            "Vehicles": data["count"],
            "Traffic Index (%)": data["density"],
            "Traffic Level": data["level"],
            "Status": data["status"],
            "Processing FPS": data["fps"],
            "Last Updated": data["updated"]
        }
        for road, data in snapshot.items()
    ])

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    # ======================================================
    # DOWNLOAD REPORTS
    # ======================================================

    st.subheader("📄 Export Analysis")

    report_col1, report_col2 = st.columns(2)

    with report_col1:

        st.download_button(
            "⬇️ Download Traffic Report",
            data=summary.to_csv(index=False),
            file_name="traffic_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with report_col2:

        if routes:

            st.download_button(
                "⬇️ Download Route Analysis",
                data=pd.DataFrame(
                    routes
                ).to_csv(index=False),
                file_name="ambulance_route_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.divider()

    st.caption(
        "ACADEMIC PROJECT DEMONSTRATION | "
        "Video feeds are prerecorded. "
        "Vehicle counts represent detections "
        "in individual analyzed frames. "
        "Traffic index and route ETA are "
        "illustrative simulation values, "
        "not validated real-time road conditions."
    )


dashboard()
