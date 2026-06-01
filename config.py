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
    """Resolve the YOLO weights file from the workspace directory.

    This project now uses a single canonical weights file named
    `security_latest 2.pt`. Return its absolute path if present,
    otherwise raise FileNotFoundError.
    """
    expected_name = "security_latest 2.pt"
    expected_path = BASE_DIR / expected_name
    if expected_path.exists():
        return str(expected_path)
    raise FileNotFoundError(f"No model weights file found. Expected: {expected_name}")

# ===================== CAMERA CONFIGURATIONS =====================
RTSP_CAMERAS = [

    #Telangana Cameras Below
    # index: 0
    # ── SOMAJIGUDA — rectangle zones (ratios → rect)
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/301",
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
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.121.130:8001/Streaming/Channels/301",
        "camera_id": "GF-2-CAM-3",
        "site_id": "2",
        "site_name": "jubilee_hills",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0048, 0.5954], [0.0301, 0.8921], [0.115, 0.8487], [0.0751, 0.5711]],
        "entry_zone_2_ratios": [[0.115, 0.5151], [0.3906, 0.3487], [0.5517, 0.5829], [0.2009, 0.9197]],
    },

    # index: 2
    # ── JAYANAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.37.109:8001/Streaming/Channels/401",
        "camera_id": "GF-5-CAM-4",
        "site_id": "5",
        "site_name": "jayanagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2461, 0.6806], [0.2977, 0.762], [0.1542, 0.9287], [0.1268, 0.8019]],
        "entry_zone_2_ratios": [[0.1055, 0.9806], [0.5185, 0.5569], [0.7508, 0.7532], [0.4081, 0.9944]],
    },

    # index: 3
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

    # index: 4
    # ── HIMAYATHNAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.97.83:8001/Streaming/Channels/201",
        "camera_id": "GF-6-CAM-2",
        "site_id": "6",
        "site_name": "himayatnagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1961009174, 0.5621814475], [0.2316513761, 0.6743119266], [0.3405963303, 0.5805300714], [0.2866972477, 0.4969418960]],
        "entry_zone_2_ratios": [[0.1158256881, 0.1034658512], [0.9988532110, 0.0647298675], [0.9919724771, 0.6539245668], [0.1272935780, 0.9006116208]],
    },

    # index: 5
    # ── KARMINAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@45.117.66.58:8001/Streaming/Channel/801",
        "camera_id": "GF-21-CAM-8",
        "site_id": "21",
        "site_name": "karimnagar-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2958715596, 0.5334620956], [0.3061926606, 0.8437469821], [0.3956422018, 0.8356349590], [0.3807339450, 0.5192660550]],
        "entry_zone_2_ratios": [[0.5057339450, 0.0183486239], [0.9701834862, 0.0366006760], [0.9587155963, 0.9106711733], [0.4655963303, 0.9066151618]],
    },

    # index: 6
    # ── KHAMMAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.113.18:8001/Streaming/Channels/401",
        "camera_id": "GF-30-CAM-4",
        "site_id": "30",
        "site_name": "khammam-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.4793577982, 0.2461773700], [0.6192660550, 0.2563710499], [0.6181192661, 0.3154943935], [0.4782110092, 0.3053007136]],
        "entry_zone_2_ratios": [[0.3704128440, 0.0015290520], [0.1181192661, 0.7660550459], [0.8405963303, 0.8190621814], [0.7133027523, 0.0035677880]],
    },

    # index: 7
    # ── KOKAPET — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.112.217:8001/Streaming/Channels/601",
        "camera_id": "GF-12-CAM-6",
        "site_id": "12",
        "site_name": "kokapet-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0045871560, 0.5517141478], [0.0011467890, 0.7200386287], [0.0814220183, 0.9958474167], [0.1525229358, 0.9938194109]],
        "entry_zone_2_ratios": [[0.2694954128, 0.0163206181], [0.7694954128, 0.0082085949], [0.7812655010, 0.8003472222], [0.2878440367, 0.8011588605], [0.2167431193, 0.3793336552]],
    },

    # index: 8
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
    # index: 9
    # ── MIRYALAGUDA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@154.210.236.163:8001/Streaming/Channel/3001",
        "camera_id": "GF-18-CAM-30",
        "site_id": "18",    
        "site_name": "miryalaguda-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6339754816, 0.6161397364], [0.7810858144, 0.6594985713], [0.7810858144, 0.6594985713], [0.7600700525, 0.7926721357], [0.6094570928, 0.7462162411]],
        "entry_zone_2_ratios": [[0.8178633975, 0.1484837312], [0.7583187391, 1.0], [0.1541155867, 0.9908839524], [0.6147110333, 0.1391925523]],
    },
    # index: 10
    # ── NIZAMABAD — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@43.249.216.149:8001/Streaming/Channels/401",
        "camera_id": "GF-31-CAM-4",
        "site_id": "31",
        "site_name": "nizamabad",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.7259174312, 0.3976287012], [0.7568807339, 0.5200035871], [0.8107798165, 0.5035300448], [0.7763761468, 0.3835085221]],
        "entry_zone_2_ratios": [[0.5779816514, 0.0587444019], [0.9724770642, 0.1058116657], [0.9759174312, 0.9247820557], [0.5470183486, 0.9271354189]],
    },

    # index: 11
    # ── WARANGAL — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.89.55.147:8001/Streaming/Channels/701",
        "camera_id": "GF-15-CAM-7",
        "site_id": "15",
        "site_name": "warangal-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6606501795, 0.2160616939], [0.6582568807, 0.2726809378], [0.7121559633, 0.2747196738], [0.7137075013, 0.2215626312]],
        "entry_zone_2_ratios": [[0.2809633028, 0.0891946993], [0.2293577982, 0.6620795107], [0.8715596330, 0.6478083588], [0.7236238532, 0.0728848114]],
    },


    # Andhra Pradesh Cameras Below

    # index: 12
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

    # index: 13
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

    # index: 14
    # ── ANANTHAPUR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.248.210.164:8001/Streaming/Channels/301",
        "camera_id": "GF-26-CAM-3",
        "site_id": "26",
        "site_name": "ananthapur-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.5917431193, 0.4036697248], [0.5791284404, 0.5131820377], [0.7041284404, 0.5638821825], [0.7155963303, 0.4523418638]],
        "entry_zone_2_ratios": [[0.3520642202, 0.4280057943], [0.7110091743, 0.5922742636], [0.4323394495, 1.0], [0.0722477064, 0.9877353935]],
    },

    # index: 15
    # ── BHIMAVARAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.94.133:8001/Streaming/Channels/301",
        "camera_id": "GF-25-CAM-3",
        "site_id": "25",
        "site_name": "bhimavaram-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2626146789, 0.5922742636], [0.3291284404, 0.5537421535], [0.3314220183, 0.6024142926], [0.2683486239, 0.6450024143]],
        "entry_zone_2_ratios": [[0.6594036697, 0.9836793819], [0.3291284404, 0.3002414293], [0.1456422018, 0.3773056494], [0.126146789, 0.9877353935]],
    },

    # index: 16
    # ── ELURU — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.205.164.51:8001/Streaming/Channels/1201",
        "camera_id": "GF-13-CAM-12",
        "site_id": "13",
        "site_name": "eluru-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6685779817, 0.3752776437], [0.754587156, 0.4949299855], [0.7305045872, 0.5760502173], [0.6433486239, 0.4482858522]],
        "entry_zone_2_ratios": [[0.7201834862, 0.1724770642], [0.8658256881, 0.332689522], [0.6525229358, 0.9796233704], [0.1158256881, 0.9857073877], [0.121559633, 0.1968131338]],
    },

    # index: 17
    # ── GUNTUR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.205.164.154:8001/Streaming/Channels/601",
        "camera_id": "FF-19-CAM-6",
        "site_id": "19",
        "site_name": "guntur-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.7970183486, 0.5071355759], [0.7649082569, 0.5316004077], [0.8394495413, 0.6376146789], [0.8543577982, 0.5886850153]],
        "entry_zone_2_ratios": [[0.751146789, 0.2359836901], [0.745412844, 0.9617737003], [0.3233944954, 0.9658511723], [0.3222477064, 0.2196738022]],
    },

    # index: 18
    # ── KADAPA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.171.190.6:8001/Streaming/Channels/301",
        "camera_id": "GF-17-CAM-3",
        "site_id": "17",
        "site_name": "kadapa-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0910683012, 0.7288382954], [0.1313485114, 0.9965946682], [0.3099824869, 0.9965946682], [0.238178634, 0.6105273399]],
        "entry_zone_2_ratios": [[0.8178633975, 0.0096322242], [0.8178633975, 0.6665693715], [0.3852889667, 0.9903677758], [0.3082311734, 0.0127456704]],
    },

    # index: 19
    # ── KAKINADA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.145.50:8001/Streaming/Channels/301",
        "camera_id": "GF-22-CAM-3",
        "site_id": "22",
        "site_name": "kakinada-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.8061926606, 0.3419979613], [0.7924311927, 0.4031600408], [0.8497706422, 0.4520897044], [0.8646788991, 0.378695209]],
        "entry_zone_2_ratios": [[0.7110091743, 0.0790010194], [0.9002293578, 0.1890927625], [0.8027522936, 0.8251783894], [0.4506880734, 0.4520897044]],
    },
]

# Ensure all RTSP URLs include recommended FFmpeg/OpenCV capture options
# By default append a conservative set of options that improves stability:
#   rtsp_transport=tcp          -> use TCP
#   fflags=discardcorrupt+genpts -> drop corrupt frames + regenerate PTS
#   flags=low_delay             -> reduce latency
#   reorder_queue_size=0        -> avoid internal reorder buffering
# Do NOT add `fflags=nobuffer` by default (can cause visible glitches on WAN).
# If a URL already contains ffmpeg/rtsp flags (e.g. contains 'fflags=',
# 'rtsp_transport=' or 'flags='), we assume it's intentionally configured
# and we will NOT append the defaults.
RTSP_SUFFIX = (
    "rtsp_transport=tcp"
    "&fflags=discardcorrupt+genpts"
    "&flags=low_delay"
    "&reorder_queue_size=0"
)

for cam in RTSP_CAMERAS:
    url = cam.get("rtsp_url")
    if not url:
        continue

    lower = url.lower()

    if any(
        k in lower
        for k in (
            "fflags=",
            "rtsp_transport=",
            "flags=",
            "genpts",
            "reorder_queue_size",
        )
    ):
        continue

    cam["rtsp_url"] = (
        url + "&" + RTSP_SUFFIX
        if "?" in url
        else url + "?" + RTSP_SUFFIX
    )

# ===================== BUSINESS HOURS & TIMEZONE =====================
IST = ZoneInfo("Asia/Kolkata")
BUSINESS_START = dtime(10, 0)  # 10:00 AM
BUSINESS_END = dtime(21, 0)    # 09:00 PM

# ===================== ENTRY ZONE (Zone-1) Parameters =====================
ENTRY_MARGIN = 5
ENTRY_DEBOUNCE_FRAMES = 2

# Time-based entry confirmation (alternative to frame-count debouncing)
# If `USE_TIME_BASED_ENTRY` is True, the system will require the customer
# to remain in the entry zone for at least `ENTRY_DEBOUNCE_SEC` seconds
# before starting a session. This avoids relying purely on frame counts
# and can be more robust on low-framerate or lossy streams.
USE_TIME_BASED_ENTRY = True
ENTRY_DEBOUNCE_SEC = 0.5

# ===================== GREET ZONE (Zone-2) Confirmation Parameters =====================
RECT2_CONFIRM_FRAMES = 2
RECT2_ABSENCE_ABORT_SEC = 20

# Time-based greet-zone confirmation (alternative to frame-count debouncing)
# If `USE_TIME_BASED_RECT2` is True, the system will require the customer
# + staff to both be present inside the greet zone for at least
# `RECT2_CONFIRM_SEC` seconds before marking the greet zone as confirmed.
USE_TIME_BASED_RECT2 = True
RECT2_CONFIRM_SEC = 0.5

# ===================== GREET COEXISTENCE Parameters =====================
# Coexistence: When customer + staff are both in zone 2 for ≥2 seconds, save event.
# Gap tolerance: If they separate for >5s, restart the coexistence timer.
GREET_GAP_TOLERANCE = 5.0   # Max absence before restarting coexistence timer (seconds)
SESSION_MAX_SEC = 40        # Max session duration before auto-abort (seconds)

# ===================== COOLDOWN & RECOVERY =====================
COOLDOWN_SEC = 30

# After a greet is saved, use a short post-save cooldown before allowing
# the next coexistence window to start. This prevents immediate duplicate
# saves while allowing a fast restart when a new customer arrives.
POST_SAVE_COOLDOWN_SEC = 2.0
# ===================== DETECTION & FILTERING =====================
CONF_THRESHOLD = 0.30
MIN_W, MIN_H = 40, 100
CUSTOMER_LABEL = "customers"
GREEN_STAFF_LABELS = {"sec1", "sec2", "sec3"}
# Per-label confidence thresholds to reduce misclassification
# Detections classified as `CUSTOMER_LABEL` must meet this confidence
# to be considered a customer in downstream logic.
# Per-label thresholds removed — using the single global `CONF_THRESHOLD`
# for both customer and staff detections to keep behavior consistent.

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
OUTPUT_BASE_DIR = "meet_and_greet"
LOG_DIR = "logs"
LOG_FILE = "meet_greet_log.txt"
LOG_BACKUP_COUNT = 5