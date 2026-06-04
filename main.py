#!/usr/bin/env python3
"""
MEET & GREET KPI - V4 MULTI-CAMERA PRODUCTION
Unified logic for ALL cameras:
  - entry_zone_1 (rect or poly)  → session trigger (customer walks in)
  - entry_zone_2 (rect or poly)  → greet zone     (same as rect-2 in somajiguda)
  - All zones defined via ratios, converted to pixels at runtime
  - All cameras run in parallel threads
"""
 
import os

# Configure FFmpeg/RTSP transport + timeouts BEFORE cv2 is imported.
# stimeout is in microseconds. Prevents indefinite hangs in cap.grab()/open.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5",
)

import time
import cv2
from ultralytics import YOLO
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import logging
import json
import gc
from logging.handlers import TimedRotatingFileHandler
import threading
from threading import Thread, Event
 
# ===================== CONFIG (imported) =====================
from config import (
    RTSP_CAMERAS, IST, BUSINESS_START, BUSINESS_END, HEADLESS,
    DEVICE, MODEL_PATH,
    ENTRY_MARGIN, ENTRY_DEBOUNCE_FRAMES, RECT2_CONFIRM_FRAMES, SESSION_MAX_SEC,
    GREET_GAP_TOLERANCE, RECT2_ABSENCE_ABORT_SEC, COOLDOWN_SEC, POST_SAVE_COOLDOWN_SEC,
    CONF_THRESHOLD, GREEN_STAFF_LABELS, CUSTOMER_LABEL,
    MAX_FPS, FRAME_INTERVAL, MIN_W, MIN_H,
    PANEL_W, PREVIEW_WINDOW_W, PREVIEW_WINDOW_H,
    OUTPUT_BASE_DIR, LOG_DIR, LOG_FILE, LOG_BACKUP_COUNT,
    # time-based options
    USE_TIME_BASED_ENTRY, ENTRY_DEBOUNCE_SEC,
    USE_TIME_BASED_RECT2, RECT2_CONFIRM_SEC,
)
import torch
import argparse
import sys
import collections
import queue as _queue
from frame_buffer import FrameBuffer
from inference_scheduler import InferenceResult, InferenceScheduler, compute_batch_size
 
# ===================== LOGGING =====================
os.makedirs(LOG_DIR, exist_ok=True)
logger    = logging.getLogger("MeetGreet")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, LOG_FILE), when="midnight", interval=1, backupCount=LOG_BACKUP_COUNT
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
 
# ===================== GLOBALS =====================
created_dirs    = set()
last_frame_time = {}
restart_events  = {}   # signal: break inner loop -> reconnect RTSP
stop_events     = {}   # signal: exit outer loop -> terminate thread

# Model will be loaded/unloaded dynamically based on business hours
model = None

# Shared inference pipeline structures (initialised in load_model_on_gpu)
zones_dict: dict = {}                    # cam_key → zone pixel coords (built on first decode frame)
result_queues: dict = {}                 # cam_key → Queue[InferenceResult]
frame_buffer: FrameBuffer | None = None  # single-slot per camera
scheduler_stop_event: threading.Event = threading.Event()
_scheduler: InferenceScheduler | None = None
 
# ===================== PATH BUILDER =====================
def get_output_path(event_type, site_name, camera_id, site_id, dur_str):
    now      = datetime.now(IST)
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H-%M-%S")
    dir_path = os.path.join(OUTPUT_BASE_DIR, site_name, camera_id, date_str)
    if dir_path not in created_dirs:
        os.makedirs(dir_path, exist_ok=True)
        created_dirs.add(dir_path)
    filename = f"{event_type}_{site_id}_{camera_id}_{date_str}_{time_str}_{dur_str}.png"
    return os.path.join(dir_path, filename)
 
# ===================== HELPERS =====================
def foot(b):
    """Returns foot center point (bottom-center) of a bounding box."""
    return ((b[0] + b[2]) // 2, b[3])
 
def is_business_hours():
    return BUSINESS_START <= datetime.now(IST).time() <= BUSINESS_END
 
def make_fresh_cap(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Defensive: some OpenCV builds expose these props; ignore if not supported.
    for prop_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            try:
                cap.set(prop, 5000)
            except Exception:
                pass
    return cap
 
def ratios_to_poly(ratios, frame_w, frame_h):
    """Convert [[x_ratio, y_ratio], ...] → pixel [(x, y), ...] points."""
    return [(int(r[0] * frame_w), int(r[1] * frame_h)) for r in ratios]
 
def ratios_to_rect(ratios, frame_w, frame_h):
    """
    Convert 4-point ratio list to axis-aligned bounding rect (x1,y1,x2,y2).
    Works for both rect and poly zones when you need a simple rect fallback.
    """
    xs = [int(r[0] * frame_w) for r in ratios]
    ys = [int(r[1] * frame_h) for r in ratios]
    return (min(xs), min(ys), max(xs), max(ys))
 
# ===================== UNIFIED ZONE CONTAINMENT =====================
def point_in_zone(px, py, zone_pts, mode):
    """
    Check if point (px, py) is inside zone.
    - rect mode : axis-aligned rectangle with ENTRY_MARGIN
    - poly mode : polygon using ray-casting (cv2.pointPolygonTest)
    zone_pts: list of (x, y) pixel tuples (4 points for both modes)
    """
    pts = np.array(zone_pts, dtype=np.float32)
    if mode == "rect":
        # Use bounding box of the 4 points with margin
        x1 = int(pts[:, 0].min()) + ENTRY_MARGIN
        y1 = int(pts[:, 1].min()) + ENTRY_MARGIN
        x2 = int(pts[:, 0].max()) - ENTRY_MARGIN
        y2 = int(pts[:, 1].max()) - ENTRY_MARGIN
        return x1 <= px <= x2 and y1 <= py <= y2
    else:
        # Polygon containment
        result = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
        return result >= 0
 
def box_in_zone(b, zone_pts, mode):
    """
    Check if bounding box b overlaps zone.
    Tests foot center + box corners against the zone.
    """
    cx = (b[0] + b[2]) // 2
    test_points = [
        (cx, b[3]),               # foot center
        (b[0], b[3]),             # bottom-left
        (b[2], b[3]),             # bottom-right
        (cx, (b[1] + b[3]) // 2), # body center
    ]
    return any(point_in_zone(px, py, zone_pts, mode) for px, py in test_points)
 
def customer_foot_in_zone(customers, zone_pts, mode):
    """Returns list of customers whose foot center is inside the zone."""
    return [c for c in customers if point_in_zone(*foot(c), zone_pts, mode)]
 
def staff_intersects_zone(green_staff, zone_pts, mode):
    """Returns list of staff whose bounding box overlaps the zone."""
    return [s for s in green_staff if box_in_zone(s, zone_pts, mode)]
 
# ===================== ANNOTATE HELPERS =====================
def annotate_banner(img, text):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 2.0
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w = img.shape[:2]
    bx1, by1 = w - tw - 20, 10
    bx2, by2 = w - 10, 10 + th + 20
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
    cv2.putText(img, text, (bx1 + 10, by1 + th + 10),
                font, scale, (255, 255, 255), thickness)
 
# ===================== DRAW ZONES ON PREVIEW =====================
def draw_zones_on_preview(preview, zones, rect2_confirmed, mode):
    """Draw entry_zone_1 (cyan) and entry_zone_2/greet zone (orange/green)."""
    z1_pts = np.array(zones["zone1_px"], dtype=np.int32)
    z2_pts = np.array(zones["zone2_px"], dtype=np.int32)
 
    # entry_zone_1 — cyan
    if mode == "rect":
        x1,y1,x2,y2 = zones["zone1_rect"]
        cv2.rectangle(preview, (x1,y1), (x2,y2), (0, 220, 220), 2)
        cv2.putText(preview, "ENTRY", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)
    else:
        cv2.polylines(preview, [z1_pts], isClosed=True, color=(0, 220, 220), thickness=2)
        cv2.putText(preview, "ENTRY", zones["zone1_px"][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)
 
    # entry_zone_2 / greet zone — orange → green when confirmed
    g_color = (0, 255, 80) if rect2_confirmed else (0, 165, 255)
    g_thick = 3 if rect2_confirmed else 2
    if mode == "rect":
        x1,y1,x2,y2 = zones["zone2_rect"]
        cv2.rectangle(preview, (x1,y1), (x2,y2), g_color, g_thick)
        cv2.putText(preview, "GREET ZONE", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color, 1, cv2.LINE_AA)
    else:
        cv2.polylines(preview, [z2_pts], isClosed=True, color=g_color, thickness=g_thick)
        cv2.putText(preview, "GREET ZONE", zones["zone2_px"][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color, 1, cv2.LINE_AA)
 
# ===================== PREVIEW HUD =====================
def draw_hud(frame, state, cfg, now_ts):
    h, w = frame.shape[:2]
    hud = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    hud[:] = (20, 20, 20)
    cv2.line(hud, (0, 0), (0, h), (70, 70, 70), 1)
 
    font = cv2.FONT_HERSHEY_SIMPLEX
    y, lh, px = 28, 26, 12
 
    def txt(text, color=(210, 210, 210), scale=0.5, bold=False):
        nonlocal y
        cv2.putText(hud, text, (px, y), font, scale, color,
                    2 if bold else 1, cv2.LINE_AA)
        y += lh
 
    def sep():
        nonlocal y
        cv2.line(hud, (8, y), (PANEL_W - 8, y), (65, 65, 65), 1)
        y += 8
 
    txt("MEET & GREET KPI", (0, 200, 255), scale=0.56, bold=True)
    txt(datetime.now(IST).strftime("%d-%m-%Y  %H:%M:%S"), (140, 140, 140), scale=0.42)
    txt(f"Site : {cfg['site_name']}",  (160, 160, 160), scale=0.43)
    txt(f"Cam  : {cfg['camera_id']}", (160, 160, 160), scale=0.43)
    sep()
 
    biz = is_business_hours()
    txt("● OPEN" if biz else "● CLOSED", (0,200,80) if biz else (0,0,200), scale=0.5, bold=True)
    sep()
 
    session         = state.get("session")
    cooldown_until  = state.get("cooldown_until", 0)
    rect2_confirmed = state.get("rect2_confirmed", False)
    in_cooldown     = now_ts < cooldown_until if isinstance(cooldown_until, (int, float)) else False
 
    if in_cooldown:
        txt(f"COOLDOWN  {cooldown_until - now_ts:.0f}s", (0,130,255), scale=0.52, bold=True)
    elif session and not session.get("done"):
        txt("● SESSION ACTIVE", (0,220,80), scale=0.52, bold=True)
    else:
        txt("● IDLE", (120,120,120), scale=0.52)
    sep()
 
    r2_color = (0,220,0) if rect2_confirmed else (80,80,80)
    txt("GREET ZONE CONFIRMED" if rect2_confirmed else "GREET ZONE PENDING",
        r2_color, scale=0.44)
    sep()
 
    txt("COEXISTENCE DURATION", (200,200,200), scale=0.46, bold=True)
    if session and session.get("coexistence_start_time"):
        coexist_duration = now_ts - session["coexistence_start_time"]
    else:
        coexist_duration = 0.0
    bar_x1, bar_y = px, y
    bar_w, bar_h  = PANEL_W - px * 2, 18
    filled = int(bar_w * min(coexist_duration, 2.0) / 2.0)  # 2s full bar
    cv2.rectangle(hud, (bar_x1, bar_y), (bar_x1+bar_w, bar_y+bar_h), (50,50,50), -1)
    bar_color = (0,200,80) if coexist_duration >= 2.0 else (0,130,255)
    if filled > 0:
        cv2.rectangle(hud, (bar_x1, bar_y), (bar_x1+filled, bar_y+bar_h), bar_color, -1)
    cv2.rectangle(hud, (bar_x1, bar_y), (bar_x1+bar_w, bar_y+bar_h), (100,100,100), 1)
    cv2.putText(hud, f"{coexist_duration:.1f}/2.0s",
                (bar_x1+bar_w//2-28, bar_y+14), font, 0.42, (255,255,255), 1, cv2.LINE_AA)
    y += bar_h + 8
    sep()
 
    if session:
        elapsed = now_ts - session["start"]
        t_col = (0,200,80) if elapsed < SESSION_MAX_SEC * 0.75 else (0,80,255)
        txt(f"Session : {elapsed:.1f}s / {SESSION_MAX_SEC}s", t_col, scale=0.44)
    else:
        txt("Session : --", (80,80,80), scale=0.44)
    sep()
 
    txt("DETECTIONS", (200,200,200), scale=0.46, bold=True)
    txt(f"Customers : {state.get('customer_count', 0)}", (0,220,0),   scale=0.44)
    txt(f"Staff     : {state.get('staff_count', 0)}",    (255,80,80), scale=0.44)
    sep()
 
    txt("LEGEND", (200,200,200), scale=0.44, bold=True)
    for color, label in [
        ((0,255,0),   " Customer"),
        ((255,80,0),  " Staff"),
        ((0,255,255), " Group BBox"),
        ((0,220,220), " Entry Zone-1"),
        ((0,165,255), " Greet Zone-2"),
    ]:
        cv2.rectangle(hud, (px, y), (px+14, y+14), color, -1)
        cv2.putText(hud, label, (px+18, y+12), font, 0.4, (180,180,180), 1)
        y += 20
 
    cv2.putText(hud, "ESC to quit", (px, h-12), font, 0.38, (70,70,70), 1)
   
    return np.hstack((frame, hud))
 
# ===================== DECODE THREAD =====================
def decode_loop(cfg: dict) -> None:
    """
    RTSP decode only — no inference.
    Writes latest frame to shared frame_buffer.
    Builds zone pixel coords on first frame and stores in zones_dict.
    """
    cam_key  = cfg["camera_id"]
    rtsp_url = cfg["rtsp_url"]
    mode     = cfg["zone_mode"]

    last_frame_time[cam_key] = time.time()
    _last_throttle_key = cam_key + "_throttle"

    cap = None
    while not stop_events[cam_key].is_set():
        try:
            cap = make_fresh_cap(rtsp_url)
            logger.info(f"🎥 STARTED → {cam_key} ({cfg['site_name']}) | mode={mode}")

            first_frame_logged = False
            first_frame_start  = time.time()
            failed_reads       = 0
            buffer_flush_count = 4   # updated after first frame using actual stream FPS

            while not stop_events[cam_key].is_set():
                # ── Watchdog restart ─────────────────────────────────────────
                if restart_events[cam_key].is_set():
                    logger.warning(f"🔄 Watchdog restart → {cam_key}")
                    restart_events[cam_key].clear()
                    break

                # ── Drain stale frames dynamically ────────────────────────────
                for _ in range(buffer_flush_count):
                    cap.grab()
                ret, frame = cap.retrieve()

                if not ret:
                    failed_reads += 1
                    if failed_reads >= 30:
                        logger.warning(f"🔄 30 failed reads, reconnecting → {cam_key}")
                        break
                    time.sleep(0.1)
                    continue
                failed_reads = 0

                frame_timestamp = time.time()
                last_frame_time[cam_key] = frame_timestamp

                frame = cv2.resize(frame, (PREVIEW_WINDOW_W, PREVIEW_WINDOW_H))

                # ── First frame: build zones + calibrate flush count ──────────
                if not first_frame_logged:
                    if time.time() - first_frame_start > 10:
                        logger.warning(f"⏳ First frame took >10s → {cam_key}")
                    frame_h, frame_w = frame.shape[:2]
                    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    logger.info(f"📊 {cam_key} Resolution: {frame_w}x{frame_h} @ {fps_cam:.1f} FPS")

                    # Flush enough frames to cover scheduler worst-case latency.
                    # Use min(fps*0.5, 15) as safe heuristic; never below 2.
                    buffer_flush_count = max(2, min(int(fps_cam * 0.5), 15))

                    # Build zone pixel coords once; store for state machine
                    z1_px = ratios_to_poly(cfg["entry_zone_1_ratios"], frame_w, frame_h)
                    z2_px = ratios_to_poly(cfg["entry_zone_2_ratios"], frame_w, frame_h)
                    zones_dict[cam_key] = {
                        "zone1_px"   : z1_px,
                        "zone2_px"   : z2_px,
                        "zone1_rect" : ratios_to_rect(cfg["entry_zone_1_ratios"], frame_w, frame_h),
                        "zone2_rect" : ratios_to_rect(cfg["entry_zone_2_ratios"], frame_w, frame_h),
                    }
                    first_frame_logged = True

                # ── FPS throttle ──────────────────────────────────────────────
                if frame_timestamp - last_frame_time.get(_last_throttle_key, 0) < FRAME_INTERVAL:
                    continue
                last_frame_time[_last_throttle_key] = frame_timestamp

                # ── Write to shared frame buffer ──────────────────────────────
                if frame_buffer is not None:
                    frame_buffer.write(cam_key, frame, frame_timestamp)

        except Exception as e:
            logger.error(f"❌ Decode crash in {cam_key}: {str(e)}", exc_info=True)
            stop_events[cam_key].wait(timeout=5)
        finally:
            if cap is not None:
                try:
                    cap.release()
                    logger.info(f"🔌 cap released → {cam_key}")
                except Exception:
                    pass
                cap = None

        if stop_events[cam_key].is_set():
            break
        stop_events[cam_key].wait(timeout=3)

    logger.info(f"🛑 DECODE EXITED → {cam_key}")


# ===================== CAMERA THREAD (legacy — replaced by decode_loop + state_machine_loop) =====================
def run_camera(cfg):
    cam_key  = cfg["camera_id"]
    rtsp_url = cfg["rtsp_url"]
    mode     = cfg["zone_mode"]
 
    state = {
        "session":              None,
        "rect2_hits":           0,
        "rect2_confirmed":      False,
        "rect2_last_seen_time": None,
        "cooldown_until":       0,
        "entry_candidate_hits": 0,
        "entry_candidate_start_time": None,
        "rect2_confirm_start_time": None,
        "customer_count":       0,
        "staff_count":          0,
    }
 
    last_frame_time[cam_key] = time.time()
    zones = {}  # filled after first frame resolves
 
    win_name = f"Meet & Greet — {cam_key}"
    if not HEADLESS:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    cap = None
    while not stop_events[cam_key].is_set():  # outer self-restart loop
        try:
            cap = make_fresh_cap(rtsp_url)
            logger.info(f"🎥 STARTED → {cam_key} ({cfg['site_name']}) | mode={mode}")

            first_frame_logged = False
            first_frame_start  = time.time()
            failed_reads       = 0

            while not stop_events[cam_key].is_set():
                # ── Watchdog restart ─────────────────────────────────────────
                if restart_events[cam_key].is_set():
                    logger.warning(f"🔄 Watchdog restart → {cam_key}")
                    restart_events[cam_key].clear()
                    break

                # ── Flush stale buffer ────────────────────────────────────────
                cap.grab()
                cap.grab()
                ret, frame = cap.retrieve()

                if not ret:
                    failed_reads += 1
                    if failed_reads >= 30:
                        logger.warning(f"🔄 30 failed reads, reconnecting → {cam_key}")
                        break
                    time.sleep(0.1)
                    continue
                failed_reads = 0
               
                # ── Capture frame timestamp BEFORE processing to decouple from inference latency ──
                frame_timestamp = time.time()
                
                frame = cv2.resize(frame, (PREVIEW_WINDOW_W, PREVIEW_WINDOW_H))
 
                # ── First frame: build zones from ratios ──────────────────────
                if not first_frame_logged:
                    if time.time() - first_frame_start > 10:
                        logger.warning(f"⏳ First frame took >10s → {cam_key}")
                    frame_h, frame_w = frame.shape[:2]
                    fps_cam = cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"📊 {cam_key} Resolution: {frame_w}x{frame_h} @ {fps_cam:.1f} FPS")
 
                    # Convert ratios → pixel points (works for both rect and poly)
                    z1_px = ratios_to_poly(cfg["entry_zone_1_ratios"], frame_w, frame_h)
                    z2_px = ratios_to_poly(cfg["entry_zone_2_ratios"], frame_w, frame_h)
 
                    zones = {
                        "zone1_px"   : z1_px,   # entry zone pixel points
                        "zone2_px"   : z2_px,   # greet zone pixel points
                        # rect fallback (used only for rect mode drawing)
                        "zone1_rect" : ratios_to_rect(cfg["entry_zone_1_ratios"], frame_w, frame_h),
                        "zone2_rect" : ratios_to_rect(cfg["entry_zone_2_ratios"], frame_w, frame_h),
                    }
                    first_frame_logged = True

                # Use frame_timestamp for all state machine decisions (not wall-clock now)
                # This ensures KPI accuracy: events timestamped at frame capture, not post-inference
                if frame_timestamp - last_frame_time[cam_key] < FRAME_INTERVAL:
                    continue
                last_frame_time[cam_key] = frame_timestamp
 
                # ── Session timeout ───────────────────────────────────────────
                if state["session"] and frame_timestamp - state["session"]["start"] >= SESSION_MAX_SEC:
                    logger.info(f"⏰ Session timed out @ {frame_timestamp} → {cam_key}")
                    state["session"]             = None
                    state["rect2_hits"]          = 0
                    state["rect2_confirmed"]      = False
                    state["rect2_last_seen_time"] = None
                    state["entry_candidate_hits"] = 0
 
                # ── YOLO Detection ────────────────────────────────────────────
                # Guard: model may be unloaded during business-hours-end transition.
                local_model = model
                if local_model is None:
                    time.sleep(0.1)
                    continue
                results = local_model.predict(
                    frame, conf=CONF_THRESHOLD, device=DEVICE,
                    agnostic_nms=True, verbose=False
                )[0]
 
                green_staff = []
                customers   = []
 
                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                    clss  = results.boxes.cls.cpu().numpy().astype(int)
                    confs = results.boxes.conf.cpu().numpy()
                    for b, c, conf in zip(boxes, clss, confs):
                        if c in staff_cls_ids:
                            if conf >= CONF_THRESHOLD:
                                green_staff.append(b)
                        elif c in customer_cls_ids:
                            if (conf >= CONF_THRESHOLD and
                                    (b[2]-b[0]) >= MIN_W and (b[3]-b[1]) >= MIN_H):
                                customers.append(b)
 
                state["customer_count"] = len(customers)
                state["staff_count"]    = len(green_staff)
 
                # ── ENTRY ZONE (zone1) — session trigger ──────────────────────
                # foot center of customer must be inside entry_zone_1
                cust_in_entry = customer_foot_in_zone(customers, zones["zone1_px"], mode)
 
                if cust_in_entry and state["session"] is None:
                    # Instant trigger: no debounce, start session immediately upon detection
                    state["session"] = {
                        "start":                   frame_timestamp,
                        "coexistence_start_time":  None,  # When customer + staff first together in zone 2
                        "last_coexistence_time":   None,  # Last frame when both were together
                        "done":                    False
                    }
                    state["entry_candidate_hits"] = 0
                    state["entry_candidate_start_time"] = None
                    logger.info(f"👤 SESSION STARTED (instant) → {cam_key}")
                else:
                    if state["session"] is None:
                        state["entry_candidate_hits"] = 0
                        state["entry_candidate_start_time"] = None
 
                # ── GREET ZONE (zone2) — rect-2 confirmation ──────────────────
                # Both customer and staff: any part of bbox overlaps greet zone
                if state["session"] and not state["rect2_confirmed"]:
                    cust_in_greet  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
                    staff_in_greet = staff_intersects_zone(green_staff, zones["zone2_px"], mode)

                    if cust_in_greet and staff_in_greet:
                        if USE_TIME_BASED_RECT2:
                            if state["rect2_confirm_start_time"] is None:
                                state["rect2_confirm_start_time"] = frame_timestamp
                            elif frame_timestamp - state["rect2_confirm_start_time"] >= RECT2_CONFIRM_SEC:
                                state["rect2_confirmed"]      = True
                                state["rect2_last_seen_time"] = frame_timestamp
                                state["rect2_confirm_start_time"] = None
                                logger.info(f"✅ GREET ZONE CONFIRMED → {cam_key}")
                            else:
                                state["rect2_last_seen_time"] = frame_timestamp
                        else:
                            state["rect2_hits"] += 1
                            state["rect2_last_seen_time"] = frame_timestamp
                            if state["rect2_hits"] >= RECT2_CONFIRM_FRAMES:
                                state["rect2_confirmed"]      = True
                                state["rect2_last_seen_time"] = frame_timestamp
                                logger.info(f"✅ GREET ZONE CONFIRMED → {cam_key}")
                    else:
                        state["rect2_hits"] = 0
                        state["rect2_confirm_start_time"] = None
 
                # ── ABSENCE ABORT ─────────────────────────────────────────────
                if (state["session"] and state["rect2_confirmed"] and
                        state["rect2_last_seen_time"] and
                        frame_timestamp - state["rect2_last_seen_time"] > RECT2_ABSENCE_ABORT_SEC):
                    logger.info(f"🚫 Greet zone absence abort → {cam_key}")
                    state["session"]             = None
                    state["rect2_confirmed"]      = False
                    state["rect2_hits"]           = 0
                    state["rect2_last_seen_time"] = None
                    state["entry_candidate_hits"] = 0
 
                # ── COMPLETED INTERACTION: avoid duplicate screenshots ────────
                # After one greet snapshot is saved, keep the session closed while
                # the same continuous customer+staff co-presence remains in zone 2.
                # Re-arm only after that co-presence has been gone long enough.
                if (state["session"] and state["rect2_confirmed"] and
                        state["session"].get("done")):
                    cust_in_greet_done  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
                    staff_in_greet_done = staff_intersects_zone(green_staff, zones["zone2_px"], mode)

                    if cust_in_greet_done and staff_in_greet_done:
                        state["rect2_last_seen_time"] = frame_timestamp
                    elif (state["rect2_last_seen_time"] and
                            frame_timestamp - state["rect2_last_seen_time"] > GREET_GAP_TOLERANCE):
                        logger.info(f"✅ Completed greet interaction reset → {cam_key}")
                        state["session"]             = None
                        state["rect2_confirmed"]      = False
                        state["rect2_hits"]           = 0
                        state["rect2_last_seen_time"] = None
                        state["entry_candidate_hits"] = 0
                        state["rect2_confirm_start_time"] = None

                # ── MEET & GREET hit counting ─────────────────────────────────
                if (state["session"] and state["rect2_confirmed"] and
                        not state["session"]["done"] and frame_timestamp >= state["cooldown_until"]):
 
                    cust_in_greet  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
                    staff_in_greet = staff_intersects_zone(green_staff, zones["zone2_px"], mode)
 
                    if cust_in_greet and staff_in_greet:
                        # Track coexistence time (when both are together in zone 2)
                        state["rect2_last_seen_time"] = frame_timestamp
                        
                        # Initialize coexistence tracking on first co-presence
                        if state["session"]["coexistence_start_time"] is None:
                            state["session"]["coexistence_start_time"] = frame_timestamp
                            state["session"]["last_coexistence_time"] = frame_timestamp
                            logger.info(f"🤝 Customer + Staff coexistence started → {cam_key}")
                        else:
                            # Check for gap since last time both were together
                            gap_since_last = frame_timestamp - state["session"]["last_coexistence_time"]
                            if gap_since_last > GREET_GAP_TOLERANCE:
                                # Gap exceeded 5s, reset coexistence timer
                                state["session"]["coexistence_start_time"] = frame_timestamp
                                logger.info(f"⏸️ Coexistence gap reset ({gap_since_last:.1f}s > {GREET_GAP_TOLERANCE}s) → {cam_key}")
                            
                            # Update last coexistence timestamp
                            state["session"]["last_coexistence_time"] = frame_timestamp
                        
                        coexistence_duration = frame_timestamp - state["session"]["coexistence_start_time"]

                        # ── Save snapshot when 2+ seconds of coexistence ──────────────
                        if coexistence_duration >= 2.0:
                                dur_str = (f"{int(coexistence_duration)}sec"
                                           if coexistence_duration < 60
                                           else f"{round(coexistence_duration/60,2)}mins")
 
                                snap = frame.copy()
                                for b in cust_in_greet:
                                    cv2.rectangle(snap, (b[0],b[1]), (b[2],b[3]), (0,255,0), 4)
                                    cv2.putText(snap, "CUSTOMER", (b[0], b[1]-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                                for b in staff_in_greet:
                                    cv2.rectangle(snap, (b[0],b[1]), (b[2],b[3]), (0,0,255), 4)
                                    cv2.putText(snap, "STAFF", (b[0], b[1]-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
 
                                all_boxes = cust_in_greet + staff_in_greet
                                gx1 = min(b[0] for b in all_boxes)
                                gy1 = min(b[1] for b in all_boxes)
                                gx2 = max(b[2] for b in all_boxes)
                                gy2 = max(b[3] for b in all_boxes)
                                cv2.rectangle(snap, (gx1,gy1), (gx2,gy2), (0,255,255), 4)
                                annotate_banner(snap, f"MEET & GREET ({dur_str})")
 
                                greet_path = get_output_path(
                                    event_type="greet",
                                    site_name=cfg["site_name"],
                                    camera_id=cfg["camera_id"],
                                    site_id=cfg["site_id"],
                                    dur_str=dur_str
                                )
                                cv2.imwrite(greet_path, snap)
                                # Also write a JSON metadata file alongside the image
                                try:
                                    meta = {}
                                    entry_ts = state["session"]["start"]
                                    coexist_start_ts = state["session"]["coexistence_start_time"]
                                    greet_ts = frame_timestamp
                                    meta["event_type"] = "greet"
                                    meta["site_name"] = cfg.get("site_name")
                                    meta["site_id"] = cfg.get("site_id")
                                    meta["camera_id"] = cfg.get("camera_id")
                                    meta["entry_time_epoch"] = entry_ts
                                    meta["coexistence_start_epoch"] = coexist_start_ts
                                    meta["greet_time_epoch"] = greet_ts
                                    meta["entry_to_coexistence_sec"] = coexist_start_ts - entry_ts
                                    meta["coexistence_duration_sec"] = coexistence_duration
                                    meta["duration_str"] = dur_str
                                    # (removed frame dimensions and per-box lists by user request)
                                    meta["model_path"] = MODEL_PATH if "MODEL_PATH" in globals() else None
                                    # friendly timestamps in ISO
                                    try:
                                        meta["entry_time"] = datetime.fromtimestamp(entry_ts, IST).isoformat()
                                        meta["coexistence_start_time"] = datetime.fromtimestamp(coexist_start_ts, IST).isoformat()
                                        meta["greet_time"] = datetime.fromtimestamp(greet_ts, IST).isoformat()
                                    except Exception:
                                        meta["entry_time"] = entry_ts
                                        meta["coexistence_start_time"] = coexist_start_ts
                                        meta["greet_time"] = greet_ts

                                    json_path = os.path.splitext(greet_path)[0] + ".json"
                                    with open(json_path, "w") as jf:
                                        json.dump(meta, jf, indent=2)
                                except Exception as je:
                                    logger.warning(f"Failed to write greet metadata JSON: {je}")
                                logger.info(f"📸 GREET SAVED → {greet_path}")

                                # Mark this session complete so the same continuous
                                # interaction cannot generate repeated snapshots.
                                # The completed-interaction block re-arms after
                                # co-presence leaves zone 2 for long enough.
                                state["rect2_last_seen_time"] = frame_timestamp
                                state["session"]["coexistence_start_time"] = None
                                state["session"]["last_coexistence_time"] = frame_timestamp
                                state["session"]["done"] = True
                                # Use a short cooldown after save (configurable)
                                try:
                                    post_cd = POST_SAVE_COOLDOWN_SEC
                                except Exception:
                                    post_cd = 2.0
                                state["cooldown_until"] = frame_timestamp + post_cd
 
                # ── PREVIEW RENDERING ─────────────────────────────────────────
                if not HEADLESS:
                    preview = frame.copy()
 
                    if zones:
                        draw_zones_on_preview(
                            preview, zones, state["rect2_confirmed"], mode
                        )
 
                    for b in customers:
                        cv2.rectangle(preview, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)
                        cv2.putText(preview, "CUSTOMER", (b[0], b[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)
 
                    for b in green_staff:
                        cv2.rectangle(preview, (b[0],b[1]), (b[2],b[3]), (255,80,0), 2)
                        cv2.putText(preview, "STAFF", (b[0], b[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,80,0), 1, cv2.LINE_AA)
 
                    # Group bbox when both in greet zone
                    c_greet = customer_foot_in_zone(customers,   zones["zone2_px"], mode)
                    s_greet = staff_intersects_zone(green_staff, zones["zone2_px"], mode)
                    if c_greet and s_greet:
                        all_b = c_greet + s_greet
                        cv2.rectangle(preview,
                                      (min(b[0] for b in all_b), min(b[1] for b in all_b)),
                                      (max(b[2] for b in all_b), max(b[3] for b in all_b)),
                                      (0,255,255), 3)
 
                    preview = draw_hud(preview, state, cfg, frame_timestamp)
                    cv2.imshow(win_name, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        logger.info(f"🛑 Quit key pressed → {cam_key}")
                        return
 
        except Exception as e:
            logger.error(f"❌ Crash in {cam_key}: {str(e)}", exc_info=True)
            # Short backoff, but exit immediately if shutdown requested.
            stop_events[cam_key].wait(timeout=5)

        finally:
            if cap is not None:
                try:
                    cap.release()
                    logger.info(f"🔌 cap released → {cam_key}")
                except Exception:
                    pass
                cap = None

        if stop_events[cam_key].is_set():
            break
        # Backoff before reconnect, but break out fast on shutdown.
        stop_events[cam_key].wait(timeout=3)

    if not HEADLESS:
        try:
            cv2.destroyWindow(win_name)
        except Exception:
            pass
    logger.info(f"🛑 EXITED → {cam_key}")
 
# ===================== BUSINESS HOURS HELPER =====================
def get_next_business_start():
    """
    Return the next datetime when business hours will start (IST).
    If already in business hours, return None (start immediately).
    Otherwise return the datetime of when business hours begin.
    """
    now = datetime.now(IST)
    now_time = now.time()
    
    # If currently in business hours, return None
    if BUSINESS_START <= now_time <= BUSINESS_END:
        return None
    
    # If before business start today, return today's start time
    if now_time < BUSINESS_START:
        return now.replace(hour=BUSINESS_START.hour, minute=BUSINESS_START.minute, second=0, microsecond=0)
    
    # After business end today, return tomorrow's start time
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=BUSINESS_START.hour, minute=BUSINESS_START.minute, second=0, microsecond=0)

# ===================== MODEL LOADER (handles dynamic loading/unloading) =====================
customer_cls_ids = []
staff_cls_ids = []

def load_model_on_gpu():
    """Load model to GPU when business hours start."""
    global model, customer_cls_ids, staff_cls_ids
    if model is not None:
        return  # Already loaded
    
    logger.info(f"🧠 Loading model to {DEVICE.upper()} (CUDA Available: {torch.cuda.is_available()})")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    
    customer_cls_ids = []
    staff_cls_ids = []
    for idx, name in model.names.items():
        name_low = name.lower()
        if CUSTOMER_LABEL in name_low:
            customer_cls_ids.append(idx)
        if any(s in name_low for s in GREEN_STAFF_LABELS):
            staff_cls_ids.append(idx)
    
    logger.info(f"✅ Model loaded | Customer IDs: {customer_cls_ids} | Staff IDs: {staff_cls_ids}")

def unload_model_from_gpu():
    """Unload model from GPU when business hours end.
    Caller MUST ensure all camera threads have exited before calling this,
    otherwise they may try to invoke .predict() on a freed reference."""
    global model
    if model is None:
        return  # Already unloaded

    logger.info("🧹 Unloading model from GPU...")
    model = None
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("✅ Model unloaded, GPU memory freed")

# ===================== WATCHDOG =====================
def watchdog_thread():
    while True:
        time.sleep(60)
        now = time.time()
        for cam in RTSP_CAMERAS:
            cam_key = cam["camera_id"]
            # Skip cams that are being shut down (off-hours).
            ev = stop_events.get(cam_key)
            if ev is None or ev.is_set():
                continue
            if cam_key in last_frame_time:
                gap = now - last_frame_time[cam_key]
                if gap > 60:
                    logger.error(f"💀 Watchdog: {cam_key} frozen ({gap:.0f}s). Requesting restart.")
                    if cam_key in restart_events:
                        restart_events[cam_key].set()
                        last_frame_time[cam_key] = now
 
# ===================== MAIN =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meet & Greet runner")
    parser.add_argument("video", nargs="?", help="Optional local video file to run against a camera config")
    parser.add_argument("--stream-index", type=int, help="Index of camera in config to run (used with a local video or to select a single camera)")
    parser.add_argument("--stream-indices", help="Comma-separated camera indices to run (e.g. 0,2,3)")
    parser.add_argument("--show", action="store_true", help="Force showing preview windows")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and show windows")
    args = parser.parse_args()

    # Wire CLI flags into module globals
    if args.show:
        HEADLESS = False
    if args.debug:
        HEADLESS = False
        logger.setLevel(logging.DEBUG)

    # Auto-detect display availability on headless servers
    disp = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not disp:
        if not HEADLESS:
            logger.warning("No display detected (DISPLAY/WAYLAND_DISPLAY unset) — forcing headless mode. GUI will be disabled.")
        HEADLESS = True

    # Select cameras to run
    selected = []
    if args.video:
        # Running a single camera config but with a local video file
        if args.stream_index is None:
            logger.error("When supplying a local video file, --stream-index must be provided to pick a camera config.")
            sys.exit(2)
        si = args.stream_index
        if si < 0 or si >= len(RTSP_CAMERAS):
            logger.error(f"stream-index {si} out of range (0..{len(RTSP_CAMERAS)-1})")
            sys.exit(2)
        base = dict(RTSP_CAMERAS[si])
        base["rtsp_url"] = args.video
        selected = [base]
    elif args.stream_indices:
        try:
            idxs = [int(x.strip()) for x in args.stream_indices.split(",") if x.strip()!='']
        except Exception:
            logger.error("Failed to parse --stream-indices, expected comma-separated integers")
            sys.exit(2)
        for i in idxs:
            if i < 0 or i >= len(RTSP_CAMERAS):
                logger.error(f"stream-index {i} out of range (0..{len(RTSP_CAMERAS)-1})")
                sys.exit(2)
            selected.append(RTSP_CAMERAS[i])
    elif args.stream_index is not None:
        si = args.stream_index
        if si < 0 or si >= len(RTSP_CAMERAS):
            logger.error(f"stream-index {si} out of range (0..{len(RTSP_CAMERAS)-1})")
            sys.exit(2)
        selected = [RTSP_CAMERAS[si]]
    else:
        selected = list(RTSP_CAMERAS)

    # Replace module RTSP_CAMERAS with the selected subset so watchdog and other code operate on it
    RTSP_CAMERAS = selected

    logger.info("🚀 MEET & GREET SYSTEM STARTED (V4 - MULTI-CAMERA PARALLEL)")
    logger.info(f"📷 Total cameras : {len(RTSP_CAMERAS)}")
    logger.info(f"⏰ Business hours: {BUSINESS_START.strftime('%H:%M')} → {BUSINESS_END.strftime('%H:%M')} IST")
    logger.info(f"📋 Threads will START/STOP automatically based on business hours")

    # ── 24/7 Main Loop ──────────────────────────────────────────────────────────
    # Check business hours, start/stop threads accordingly
    active_threads = []
    watchdog_thread_obj = None

    def _start_camera_threads():
        """Spin up one thread per camera. Reset events + frame timestamps."""
        global watchdog_thread_obj
        load_model_on_gpu()
        logger.info(f"🎬 Starting {len(RTSP_CAMERAS)} camera thread(s)...")

        for cam in RTSP_CAMERAS:
            cam_key = cam["camera_id"]
            # Fresh events every cycle so previous-cycle .set() doesn't leak.
            restart_events[cam_key] = Event()
            stop_events[cam_key] = Event()
            last_frame_time[cam_key] = time.time()

        if watchdog_thread_obj is None or not watchdog_thread_obj.is_alive():
            watchdog_thread_obj = Thread(target=watchdog_thread, daemon=True, name="watchdog")
            watchdog_thread_obj.start()

        threads = []
        for cam in RTSP_CAMERAS:
            t = Thread(target=run_camera, args=(cam,), daemon=True, name=cam["camera_id"])
            t.start()
            threads.append(t)
            logger.info(f"  🎬 → {cam['camera_id']} ({cam['site_name']}) | mode={cam['zone_mode']}")
        return threads

    def _stop_camera_threads(threads):
        """Signal stop, join with timeout, then unload model only if all exited."""
        logger.info(f"🛑 Stopping {len(threads)} camera thread(s)...")
        # Signal stop FIRST, then nudge inner loop via restart event.
        for cam in RTSP_CAMERAS:
            cam_key = cam["camera_id"]
            if cam_key in stop_events:
                stop_events[cam_key].set()
            if cam_key in restart_events:
                restart_events[cam_key].set()

        for t in threads:
            t.join(timeout=15)

        alive = [t for t in threads if t.is_alive()]
        if alive:
            # Stuck threads (likely blocked in cv2.VideoCapture open). Leave them
            # as daemons — they'll die with the process. Do NOT unload the model;
            # they may still call .predict() and segfault on freed CUDA memory.
            logger.warning(
                f"⚠️  {len(alive)} thread(s) did not exit in time: "
                f"{[t.name for t in alive]} — skipping model unload"
            )
            return False

        logger.info("✅ All threads stopped")
        unload_model_from_gpu()
        return True

    last_idle_log_min = None
    try:
        while True:
            now = datetime.now(IST)
            in_business = is_business_hours()

            if in_business and not active_threads:
                # Business hours STARTED — load model and start camera threads
                logger.info(f"✅ BUSINESS HOURS ACTIVE ({now.strftime('%H:%M:%S IST')})")
                active_threads = _start_camera_threads()

            elif not in_business and active_threads:
                # Business hours ENDED — stop threads and unload model
                logger.warning(f"⏸️  BUSINESS HOURS ENDED ({now.strftime('%H:%M:%S IST')})")
                _stop_camera_threads(active_threads)
                active_threads = []

            elif not in_business and not active_threads:
                # Still outside business hours — wait and check periodically
                next_start = get_next_business_start()
                if next_start:
                    wait_min = int((next_start - now).total_seconds() // 60)
                    bucket = wait_min // 60  # log once per hour bucket
                    if bucket != last_idle_log_min:
                        logger.info(f"⏳ Waiting for business hours ({wait_min} minutes remaining)")
                        last_idle_log_min = bucket
                time.sleep(30)
                continue

            # In business hours: reset idle tracker, also reap dead threads.
            last_idle_log_min = None
            # Detect a thread that exited unexpectedly (e.g. RTSP died and outer
            # loop honored stop, or fatal crash). Restart it without reloading
            # the model.
            for i, t in enumerate(list(active_threads)):
                if not t.is_alive():
                    cam = next((c for c in RTSP_CAMERAS if c["camera_id"] == t.name), None)
                    if cam is None:
                        active_threads.remove(t)
                        continue
                    cam_key = cam["camera_id"]
                    logger.warning(f"♻️  Thread {cam_key} not alive — respawning")
                    stop_events[cam_key] = Event()
                    restart_events[cam_key] = Event()
                    last_frame_time[cam_key] = time.time()
                    nt = Thread(target=run_camera, args=(cam,), daemon=True, name=cam_key)
                    nt.start()
                    active_threads[i] = nt

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt, shutting down...")
        if active_threads:
            _stop_camera_threads(active_threads)
        logger.info("✅ Shutdown complete")
 