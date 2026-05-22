#!/usr/bin/env python3
"""
MEET & GREET KPI - Configuration File (V4)
All algorithm-driving parameters and camera configurations
"""

from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent


def _resolve_model_path() -> str:
    """Resolve the YOLO weights file from the workspace directory."""
    candidate_names = (
        "new_staff.pt",
        "security_latest 2.pt",
    )

    for candidate_name in candidate_names:
        candidate_path = BASE_DIR / candidate_name
        if candidate_path.exists():
            return str(candidate_path)

    raise FileNotFoundError(
        "No model weights file found. Expected one of: "
        + ", ".join(candidate_names)
    )

# ===================== CAMERA CONFIGURATIONS =====================
RTSP_CAMERAS = [
    # index: 0
    # ── SOMAJIGUDA — rectangle zones (ratios → rect)
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/301?rtsp_transport=tcp",
        "camera_id": "GF-1-CAM-3",
        "site_id": "1",
        "site_name": "somajiguda",
        "zone_mode": "rect",
        "entry_zone_1_ratios": [
            [0.3719, 0.3547],
            [0.5400, 0.3547],
            [0.5400, 0.4105],
            [0.3719, 0.4105],
        ],
        "entry_zone_2_ratios": [
            [0.3100, 0.4802],
            [0.5875, 0.4802],
            [0.5875, 0.8070],
            [0.3100, 0.8070],
        ],
    },

    # index: 1
    # ── JUBILEE HILLS — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.121.130:8001/Streaming/Channels/301?rtsp_transport=tcp&fflags=discardcorrupt&flags=low_delay&fflags=nobuffer",
        "camera_id": "GF-2-CAM-4",
        "site_id": "2",
        "site_name": "jubilee_hills",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0048, 0.5954], [0.0301, 0.8921], [0.115, 0.8487], [0.0751, 0.5711]],
        "entry_zone_2_ratios": [[0.115, 0.5151], [0.3906, 0.3487], [0.5517, 0.5829], [0.2009, 0.9197]],
    },

    # index: 2
    # ── VIZAG — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.103:8002/Streaming/Channels/401?rtsp_transport=tcp&fflags=discardcorrupt&flags=low_delay&fflags=nobuffer",
        "camera_id": "FF-3-CAM-4",
        "site_id": "3",
        "site_name": "vizag",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6552, 0.2097], [0.6211, 0.2463], [0.6982, 0.3574], [0.7305, 0.3019]],
        "entry_zone_2_ratios": [[0.576, 0.2861], [0.3758, 0.4981], [0.7401, 0.906], [0.9109, 0.5537]],
    },

    # index: 3
    # ── VIJAYAWADA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.76.17:8001/Streaming/Channels/401?rtsp_transport=tcp&fflags=discardcorrupt&flags=low_delay&fflags=nobuffer",
        "camera_id": "GF-4-CAM-4",
        "site_id": "4",
        "site_name": "vijayawada",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.7031, 0.3227], [0.6669, 0.3782], [0.8427, 0.5], [0.8573, 0.4287]],
        "entry_zone_2_ratios": [[0.5867, 0.3833], [0.3289, 0.6958], [0.7557, 0.9861], [0.8672, 0.538]],
    },

    # index: 4
    # ── JAYANAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.37.109:8001/Streaming/Channels/401?rtsp_transport=tcp&fflags=discardcorrupt&flags=low_delay&fflags=nobuffer",
        "camera_id": "GF-5-CAM-4",
        "site_id": "5",
        "site_name": "jayanagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2461, 0.6806], [0.2977, 0.762], [0.1542, 0.9287], [0.1268, 0.8019]],
        "entry_zone_2_ratios": [[0.1055, 0.9806], [0.5185, 0.5569], [0.7508, 0.7532], [0.4081, 0.9944]],
    },
]

# ===================== BUSINESS HOURS & TIMEZONE =====================
IST = ZoneInfo("Asia/Kolkata")
BUSINESS_START = dtime(10, 0)  # 10:00 AM
BUSINESS_END = dtime(21, 0)    # 09:00 PM

# ===================== ENTRY ZONE (Zone-1) Parameters =====================
ENTRY_MARGIN = 5
ENTRY_DEBOUNCE_FRAMES = 2

# ===================== GREET ZONE (Zone-2) Confirmation Parameters =====================
RECT2_CONFIRM_FRAMES = 2
RECT2_ABSENCE_ABORT_SEC = 20

# ===================== GREET HIT COUNTING Parameters =====================
GREET_HIT_THRESHOLD = 4
GREET_GAP_TOLERANCE = 0.5
SESSION_MAX_SEC = 40

# ===================== COOLDOWN & RECOVERY =====================
COOLDOWN_SEC = 30

# ===================== DETECTION & FILTERING =====================
CONF_THRESHOLD = 0.30
MIN_W, MIN_H = 40, 100
CUSTOMER_LABEL = "customers"
GREEN_STAFF_LABELS = {"sec1", "sec2", "sec3"}

# ===================== MODEL =====================
MODEL_PATH = _resolve_model_path()

# ===================== FRAME PROCESSING =====================
MAX_FPS = 25
FRAME_INTERVAL = 1.0 / MAX_FPS
PREVIEW_WINDOW_W = 1280
PREVIEW_WINDOW_H = 720

# ===================== DISPLAY & HEADLESS MODE =====================
HEADLESS = False
PANEL_W = 300

# ===================== DEVICE CONFIGURATION =====================
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

# ===================== OUTPUT & LOGGING =====================
# Use a repository-relative output directory by default
OUTPUT_BASE_DIR = "outputs"
LOG_DIR = "logs"
LOG_FILE = "meet_greet_log.txt"
LOG_BACKUP_COUNT = 7
