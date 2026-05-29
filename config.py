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

    # index: 5
    # ── CHANDANAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.100.131:8001/Streaming/Channels/501",
        "camera_id": "GF-11-CAM-5",
        "site_id": "11",
        "site_name": "chandanagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.5493119266, 0.3590535973], [0.5447247706, 0.4503138580], [0.6628440367, 0.4847899565], [0.6628440367, 0.3996137132]],
        "entry_zone_2_ratios": [[0.6330275229, 0.9897633993], [0.1387614679, 0.9857073877], [0.4128440367, 0.5294060840], [0.6628440367, 0.6531144375]],
    },

    # index: 6
    # ── HIMAYATHNAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.97.83:8001/Streaming/Channels/201",
        "camera_id": "GF-6-CAM-2",
        "site_id": "6",
        "site_name": "himayathnagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1961009174, 0.5621814475], [0.2316513761, 0.6743119266], [0.3405963303, 0.5805300714], [0.2866972477, 0.4969418960]],
        "entry_zone_2_ratios": [[0.1158256881, 0.1034658512], [0.9988532110, 0.0647298675], [0.9919724771, 0.6539245668], [0.1272935780, 0.9006116208]],
    },

    # index: 7
    # ── KARMINAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@45.117.66.58:8001/Streaming/Channel/801",
        "camera_id": "GF-21-CAM-8",
        "site_id": "21",
        "site_name": "karminagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2958715596, 0.5334620956], [0.3061926606, 0.8437469821], [0.3956422018, 0.8356349590], [0.3807339450, 0.5192660550]],
        "entry_zone_2_ratios": [[0.5057339450, 0.0183486239], [0.9701834862, 0.0366006760], [0.9587155963, 0.9106711733], [0.4655963303, 0.9066151618]],
    },

    # index: 8
    # ── KHAMMAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.113.18:8001/Streaming/Channels/401",
        "camera_id": "GF-30-CAM-4",
        "site_id": "30",
        "site_name": "khammam",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.4793577982, 0.2461773700], [0.6192660550, 0.2563710499], [0.6181192661, 0.3154943935], [0.4782110092, 0.3053007136]],
        "entry_zone_2_ratios": [[0.3704128440, 0.0015290520], [0.1181192661, 0.7660550459], [0.8405963303, 0.8190621814], [0.7133027523, 0.0035677880]],
    },

    # index: 9
    # ── KOKAPET — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.112.217:8001/Streaming/Channels/601",
        "camera_id": "GF-12-CAM-6",
        "site_id": "12",
        "site_name": "kokapet",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0045871560, 0.5517141478], [0.0011467890, 0.7200386287], [0.0814220183, 0.9958474167], [0.1525229358, 0.9938194109]],
        "entry_zone_2_ratios": [[0.2694954128, 0.0163206181], [0.7694954128, 0.0082085949], [0.7812655010, 0.8003472222], [0.2878440367, 0.8011588605], [0.2167431193, 0.3793336552]],
    },

    # index: 10
    # ── KOMPALLY — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.106.41:8001/Streaming/Channels/401",
        "camera_id": "GF-10-CAM-4",
        "site_id": "10",
        "site_name": "kompally",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1548165138, 0.6409464027], [0.2029816514, 0.8478029937], [0.2591743119, 0.8031868662], [0.2155963303, 0.5841622405]],
        "entry_zone_2_ratios": [[0.1662844037, 0.2475132786], [0.5665137615, 0.0568807339], [0.8126094571, 0.4365102774], [0.2889908257, 0.9045871560]],
    },

    # Miryalaguda zones were provided, but the RTSP URL was not included in the attachment.

    # index: 11
    # ── NIZAMBAD — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@43.249.216.149:8001/Streaming/Channels/401",
        "camera_id": "GF-31-CAM-4",
        "site_id": "31",
        "site_name": "nizambad",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.7259174312, 0.3976287012], [0.7568807339, 0.5200035871], [0.8107798165, 0.5035300448], [0.7763761468, 0.3835085221]],
        "entry_zone_2_ratios": [[0.5779816514, 0.0587444019], [0.9724770642, 0.1058116657], [0.9759174312, 0.9247820557], [0.5470183486, 0.9271354189]],
    },

    # index: 12
    # ── WARANGAL — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.89.55.147:8001/Streaming/Channels/701",
        "camera_id": "GF-15-CAM-7",
        "site_id": "15",
        "site_name": "warangal",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6606501795, 0.2160616939], [0.6582568807, 0.2726809378], [0.7121559633, 0.2747196738], [0.7137075013, 0.2215626312]],
        "entry_zone_2_ratios": [[0.2809633028, 0.0891946993], [0.2293577982, 0.6620795107], [0.8715596330, 0.6478083588], [0.7236238532, 0.0728848114]],
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
