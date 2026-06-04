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
    NUM_INFERENCE_WORKERS,
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

# Models loaded/unloaded dynamically based on business hours.
# One model instance per inference worker (NUM_INFERENCE_WORKERS).
models: list = []

# Shared inference pipeline structures (initialised in load_model_on_gpu)
zones_dict: dict = {}                    # cam_key → zone pixel coords (built on first decode frame)
result_queues: dict = {}                 # cam_key → Queue[InferenceResult]
frame_buffer: FrameBuffer | None = None  # single-slot per camera
scheduler_stop_event: threading.Event = threading.Event()
_schedulers: list = []                   # one InferenceScheduler per worker
 
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
    if session and session.get("activation_time"):
        coexist_duration = (session.get("last_coexist_time") or now_ts) - session["activation_time"]
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


# ===================== STATE MACHINE THREAD =====================
def state_machine_loop(cfg: dict) -> None:
    """
    Greet KPI logic — reads InferenceResult from per-camera result queue.
    All greet detection, coexistence tracking, and snapshot saving unchanged.
    No GPU access — pure CPU state machine.
    """
    cam_key = cfg["camera_id"]
    mode    = cfg["zone_mode"]

    state = {
        "session":                    None,
        "rect2_hits":                 0,
        "rect2_confirmed":            False,
        "rect2_last_seen_time":       None,
        "cooldown_until":             0,
        "entry_candidate_hits":       0,
        "entry_candidate_start_time": None,
        "rect2_confirm_start_time":   None,
        "customer_count":             0,
        "staff_count":                0,
    }

    win_name = f"Meet & Greet — {cam_key}"
    if not HEADLESS:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    q = result_queues.get(cam_key)
    if q is None:
        logger.error(f"state_machine_loop: no result queue for {cam_key}")
        return

    def _finalize_greet(sess: dict, reason: str) -> None:
        """
        Save the last frame where customer+staff coexisted, labelled with the
        FULL coexistence duration (activation → last coexist). Only saves if
        total duration >= 2.0s. Called on customer-legs-exit or session timeout.
        """
        if sess is None:
            return
        activation = sess.get("activation_time")
        last_ts    = sess.get("last_coexist_time")
        snap_frame = sess.get("last_coexist_frame")
        boxes      = sess.get("last_coexist_boxes")
        if activation is None or last_ts is None or snap_frame is None:
            return  # never activated — nothing to save

        total = last_ts - activation
        if total < 2.0:
            logger.info(f"⏱️ Greet below 2s min ({total:.1f}s, {reason}) — discarded → {cam_key}")
            return

        dur_str = f"{int(total)}sec" if total < 60 else f"{round(total / 60, 2)}mins"
        cust_boxes, staff_boxes = boxes if boxes else ([], [])

        snap = snap_frame.copy()
        for b in cust_boxes:
            cv2.rectangle(snap, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
            cv2.putText(snap, "CUSTOMER", (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for b in staff_boxes:
            cv2.rectangle(snap, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
            cv2.putText(snap, "STAFF", (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if cust_boxes or staff_boxes:
            all_boxes = list(cust_boxes) + list(staff_boxes)
            gx1 = min(b[0] for b in all_boxes)
            gy1 = min(b[1] for b in all_boxes)
            gx2 = max(b[2] for b in all_boxes)
            gy2 = max(b[3] for b in all_boxes)
            cv2.rectangle(snap, (gx1, gy1), (gx2, gy2), (0, 255, 255), 4)
        annotate_banner(snap, f"MEET & GREET ({dur_str})")

        greet_path = get_output_path(
            event_type="greet",
            site_name=cfg["site_name"],
            camera_id=cfg["camera_id"],
            site_id=cfg["site_id"],
            dur_str=dur_str,
        )
        cv2.imwrite(greet_path, snap)

        try:
            meta: dict = {}
            entry_ts = sess["start"]
            meta["event_type"]               = "greet"
            meta["site_name"]                = cfg.get("site_name")
            meta["site_id"]                  = cfg.get("site_id")
            meta["camera_id"]                = cfg.get("camera_id")
            meta["entry_time_epoch"]         = entry_ts
            meta["coexistence_start_epoch"]  = activation
            meta["greet_time_epoch"]         = last_ts
            meta["entry_to_coexistence_sec"] = activation - entry_ts
            meta["coexistence_duration_sec"] = total
            meta["duration_str"]             = dur_str
            meta["exit_reason"]              = reason
            meta["model_path"]               = MODEL_PATH
            try:
                meta["entry_time"]             = datetime.fromtimestamp(entry_ts, IST).isoformat()
                meta["coexistence_start_time"] = datetime.fromtimestamp(activation, IST).isoformat()
                meta["greet_time"]             = datetime.fromtimestamp(last_ts, IST).isoformat()
            except Exception:
                meta["entry_time"]             = entry_ts
                meta["coexistence_start_time"] = activation
                meta["greet_time"]             = last_ts
            json_path = os.path.splitext(greet_path)[0] + ".json"
            with open(json_path, "w") as jf:
                json.dump(meta, jf, indent=2)
        except Exception as je:
            logger.warning(f"Failed to write greet metadata JSON: {je}")

        logger.info(f"📸 GREET SAVED ({dur_str}, {reason}) → {greet_path}")

    while not stop_events[cam_key].is_set():
        # ── Get next inference result ─────────────────────────────────────
        try:
            inf_result: InferenceResult = q.get(timeout=1.0)
        except _queue.Empty:
            # No inference arrived — check session timeout via wall clock
            if state["session"]:
                now = time.time()
                if now - state["session"]["start"] >= SESSION_MAX_SEC:
                    logger.info(f"⏰ Session timed out @ {now} → {cam_key}")
                    _finalize_greet(state["session"], "session_timeout")
                    state["session"]             = None
                    state["rect2_hits"]          = 0
                    state["rect2_confirmed"]      = False
                    state["rect2_last_seen_time"] = None
                    state["entry_candidate_hits"] = 0
            continue

        frame           = inf_result.frame
        frame_timestamp = inf_result.frame_timestamp
        customers       = inf_result.customers
        green_staff     = inf_result.green_staff

        # ── Wait for zones to be built by decode thread ───────────────────
        zones = zones_dict.get(cam_key)
        if not zones:
            continue

        state["customer_count"] = len(customers)
        state["staff_count"]    = len(green_staff)

        # ── Session timeout ───────────────────────────────────────────────
        if state["session"] and frame_timestamp - state["session"]["start"] >= SESSION_MAX_SEC:
            logger.info(f"⏰ Session timed out @ {frame_timestamp} → {cam_key}")
            _finalize_greet(state["session"], "session_timeout")
            state["session"]             = None
            state["rect2_hits"]          = 0
            state["rect2_confirmed"]      = False
            state["rect2_last_seen_time"] = None
            state["entry_candidate_hits"] = 0

        # ── ENTRY ZONE (zone1) — session trigger ──────────────────────────
        cust_in_entry = customer_foot_in_zone(customers, zones["zone1_px"], mode)

        if cust_in_entry and state["session"] is None:
            state["session"] = {
                "start":              frame_timestamp,
                "activation_time":    None,   # when both boxes first coexist in greet zone
                "last_coexist_time":  None,   # latest frame both coexisted
                "last_coexist_frame": None,   # that frame (saved as snapshot)
                "last_coexist_boxes": None,   # (cust_boxes, staff_boxes) for annotation
                "legs_out_since":     None,   # when customer FOOT left greet zone (exit timer)
                "done":               False,
            }
            state["entry_candidate_hits"]       = 0
            state["entry_candidate_start_time"] = None
            logger.info(f"👤 SESSION STARTED (instant) → {cam_key}")
        else:
            if state["session"] is None:
                state["entry_candidate_hits"]       = 0
                state["entry_candidate_start_time"] = None

        # ── GREET ZONE (zone2) — confirmation ────────────────────────────
        if state["session"] and not state["rect2_confirmed"]:
            cust_in_greet  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
            staff_in_greet = staff_intersects_zone(green_staff, zones["zone2_px"], mode)

            if cust_in_greet and staff_in_greet:
                if USE_TIME_BASED_RECT2:
                    if state["rect2_confirm_start_time"] is None:
                        state["rect2_confirm_start_time"] = frame_timestamp
                    elif frame_timestamp - state["rect2_confirm_start_time"] >= RECT2_CONFIRM_SEC:
                        state["rect2_confirmed"]          = True
                        state["rect2_last_seen_time"]     = frame_timestamp
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
                state["rect2_hits"]               = 0
                state["rect2_confirm_start_time"] = None

        # ── COEXISTENCE TRACKING + LEGS-EXIT SAVE ─────────────────────────
        # Activation/continue = customer BOX + staff BOX overlap greet zone.
        # Exit = customer FOOT leaves greet zone for >= GREET_GAP_TOLERANCE.
        # Brief foot-out (< tolerance) is bridged — keeps the timer running.
        if state["session"] and state["rect2_confirmed"]:
            sess = state["session"]
            cust_box_in_greet  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
            staff_in_greet     = staff_intersects_zone(green_staff, zones["zone2_px"], mode)
            cust_foot_in_greet = customer_foot_in_zone(customers, zones["zone2_px"], mode)

            both_coexist = bool(cust_box_in_greet) and bool(staff_in_greet)

            if both_coexist:
                # Activate timer on first coexistence frame
                if sess["activation_time"] is None:
                    sess["activation_time"] = frame_timestamp
                    logger.info(f"🤝 Customer + Staff coexistence started → {cam_key}")
                # Update last-coexist frame/time/boxes (this frame is the snapshot candidate)
                sess["last_coexist_time"]  = frame_timestamp
                sess["last_coexist_frame"] = frame
                sess["last_coexist_boxes"] = (cust_box_in_greet, staff_in_greet)
                sess["legs_out_since"]     = None   # customer present — reset exit timer

            # Exit detection — only after activation, based on customer FOOT
            if sess["activation_time"] is not None and not cust_foot_in_greet:
                if sess["legs_out_since"] is None:
                    sess["legs_out_since"] = frame_timestamp
                elif frame_timestamp - sess["legs_out_since"] >= GREET_GAP_TOLERANCE:
                    # Customer legs left greet zone long enough — conclude M&G
                    total = (sess["last_coexist_time"] or sess["activation_time"]) - sess["activation_time"]
                    logger.info(
                        f"🚶 Customer legs exited greet zone ({total:.1f}s coexist) → {cam_key}"
                    )
                    _finalize_greet(sess, "legs_exit")
                    state["session"]                  = None
                    state["rect2_confirmed"]           = False
                    state["rect2_hits"]                = 0
                    state["rect2_last_seen_time"]      = None
                    state["entry_candidate_hits"]      = 0
                    state["rect2_confirm_start_time"]  = None
                    state["cooldown_until"]            = frame_timestamp + POST_SAVE_COOLDOWN_SEC

        # ── PREVIEW RENDERING ─────────────────────────────────────────────
        if not HEADLESS:
            preview = frame.copy()
            if zones:
                draw_zones_on_preview(preview, zones, state["rect2_confirmed"], mode)
            for b in customers:
                cv2.rectangle(preview, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(preview, "CUSTOMER", (b[0], b[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            for b in green_staff:
                cv2.rectangle(preview, (b[0], b[1]), (b[2], b[3]), (255, 80, 0), 2)
                cv2.putText(preview, "STAFF", (b[0], b[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 1, cv2.LINE_AA)
            c_greet = customer_foot_in_zone(customers,   zones["zone2_px"], mode)
            s_greet = staff_intersects_zone(green_staff, zones["zone2_px"], mode)
            if c_greet and s_greet:
                all_b = c_greet + s_greet
                cv2.rectangle(
                    preview,
                    (min(b[0] for b in all_b), min(b[1] for b in all_b)),
                    (max(b[2] for b in all_b), max(b[3] for b in all_b)),
                    (0, 255, 255), 3,
                )
            preview = draw_hud(preview, state, cfg, frame_timestamp)
            cv2.imshow(win_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info(f"🛑 Quit key pressed → {cam_key}")
                return

    if not HEADLESS:
        try:
            cv2.destroyWindow(win_name)
        except Exception:
            pass
    logger.info(f"🛑 STATE MACHINE EXITED → {cam_key}")


def run_camera(cfg):
    """Legacy stub — kept only to avoid NameError if referenced elsewhere. Not called."""
    raise RuntimeError(
        "run_camera() has been replaced by decode_loop() + state_machine_loop(). "
        "Do not call run_camera() directly."
    )
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

# ===================== MODEL LOADER =====================
customer_cls_ids: list = []
staff_cls_ids: list    = []

def load_model_on_gpu() -> None:
    """Load N model instances (one per worker), init FrameBuffer + result queues."""
    global models, customer_cls_ids, staff_cls_ids, frame_buffer, result_queues, scheduler_stop_event

    if models:
        return  # already loaded

    n_workers = max(1, NUM_INFERENCE_WORKERS)
    logger.info(
        f"🧠 Loading {n_workers} model instance(s) to {DEVICE.upper()} "
        f"(CUDA Available: {torch.cuda.is_available()})"
    )

    models = []
    for i in range(n_workers):
        m = YOLO(MODEL_PATH)
        m.to(DEVICE)
        models.append(m)
        logger.info(f"  🧠 model instance {i + 1}/{n_workers} loaded")

    # Resolve class IDs from first instance (all share the same weights file)
    customer_cls_ids = []
    staff_cls_ids    = []
    for idx, name in models[0].names.items():
        name_low = name.lower()
        if CUSTOMER_LABEL in name_low:
            customer_cls_ids.append(idx)
        if any(s in name_low for s in GREEN_STAFF_LABELS):
            staff_cls_ids.append(idx)

    if not customer_cls_ids:
        raise RuntimeError(
            f"Model class names do not contain '{CUSTOMER_LABEL}'. "
            f"Available: {list(models[0].names.values())}. "
            f"Fix CUSTOMER_LABEL in config.py."
        )
    if not staff_cls_ids:
        raise RuntimeError(
            f"Model class names do not match any of {GREEN_STAFF_LABELS}. "
            f"Available: {list(models[0].names.values())}. "
            f"Fix GREEN_STAFF_LABELS in config.py."
        )

    logger.info(f"✅ Models loaded | Customer IDs: {customer_cls_ids} | Staff IDs: {staff_cls_ids}")

    cam_keys      = [c["camera_id"] for c in RTSP_CAMERAS]
    frame_buffer  = FrameBuffer(cam_keys)
    result_queues = {k: _queue.Queue(maxsize=2) for k in cam_keys}
    scheduler_stop_event = threading.Event()
    logger.info(
        f"🗂️  FrameBuffer + result_queues initialised | cameras={len(cam_keys)}"
    )


def unload_model_from_gpu() -> None:
    """Stop all InferenceScheduler workers, unload models, free GPU memory."""
    global models, _schedulers

    if _schedulers:
        logger.info(f"⏹️  Stopping {len(_schedulers)} InferenceScheduler worker(s)...")
        scheduler_stop_event.set()
        for s in _schedulers:
            s.join(timeout=10)
            if s.is_alive():
                logger.warning(f"⚠️  {s.name} did not stop in time")
        _schedulers = []

    if not models:
        return

    logger.info("🧹 Unloading models from GPU...")
    models = []
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("✅ Models unloaded, GPU memory freed")

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

    def _start_camera_threads() -> list:
        """Start N InferenceScheduler workers + decode + state_machine threads."""
        global _schedulers, scheduler_stop_event, watchdog_thread_obj
        load_model_on_gpu()

        cam_keys   = [c["camera_id"] for c in RTSP_CAMERAS]
        n_workers  = len(models)
        batch_size = compute_batch_size(models[0], len(cam_keys), n_workers)

        # Fresh stop event for this business-hours cycle
        scheduler_stop_event = threading.Event()

        # Spawn one scheduler per worker, each owns an interleaved camera slice.
        _schedulers = []
        for i in range(n_workers):
            worker_cams = cam_keys[i::n_workers]   # interleaved split
            if not worker_cams:
                continue
            sched = InferenceScheduler(
                model=models[i],
                frame_buffer=frame_buffer,
                result_queues=result_queues,
                batch_size=batch_size,
                cam_keys=worker_cams,
                customer_cls_ids=customer_cls_ids,
                staff_cls_ids=staff_cls_ids,
                stop_event=scheduler_stop_event,
                conf_threshold=CONF_THRESHOLD,
                device=DEVICE,
                min_w=MIN_W,
                min_h=MIN_H,
            )
            sched.name = f"InferenceScheduler-{i}"
            sched.start()
            _schedulers.append(sched)
            logger.info(f"  ⚙️  worker {i} → {len(worker_cams)} cams, batch={batch_size}")

        logger.info(f"🎬 Starting {len(RTSP_CAMERAS)} camera thread(s)...")

        threads = []
        for cam in RTSP_CAMERAS:
            cam_key = cam["camera_id"]
            restart_events[cam_key] = Event()
            stop_events[cam_key]    = Event()
            last_frame_time[cam_key] = time.time()

            dt = Thread(
                target=decode_loop, args=(cam,),
                daemon=True, name=f"decode-{cam_key}",
            )
            st = Thread(
                target=state_machine_loop, args=(cam,),
                daemon=True, name=f"state-{cam_key}",
            )
            dt.start()
            st.start()
            threads.extend([dt, st])
            logger.info(f"  🎬 → {cam_key} ({cam['site_name']}) | mode={cam['zone_mode']}")

        if watchdog_thread_obj is None or not watchdog_thread_obj.is_alive():
            watchdog_thread_obj = Thread(target=watchdog_thread, daemon=True, name="watchdog")
            watchdog_thread_obj.start()

        return threads

    def _stop_camera_threads(threads: list) -> bool:
        """Signal stop for scheduler + all decode/state threads, then join."""
        logger.info(f"🛑 Stopping {len(threads) // 2} camera(s)...")

        # Stop InferenceScheduler first — no more GPU calls
        scheduler_stop_event.set()

        for cam in RTSP_CAMERAS:
            cam_key = cam["camera_id"]
            if cam_key in stop_events:
                stop_events[cam_key].set()
            if cam_key in restart_events:
                restart_events[cam_key].set()

        for t in threads:
            t.join(timeout=15)

        for s in _schedulers:
            if s.is_alive():
                s.join(timeout=10)

        alive = [t for t in threads if t.is_alive()]
        if alive:
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
            # Detect decode/state threads that exited unexpectedly and respawn
            # them independently. Thread names are "decode-<cam>" / "state-<cam>".
            for i, t in enumerate(list(active_threads)):
                if t.is_alive():
                    continue
                name = t.name
                if name.startswith("decode-"):
                    cam_key = name[len("decode-"):]
                    target  = decode_loop
                    is_decode = True
                elif name.startswith("state-"):
                    cam_key = name[len("state-"):]
                    target  = state_machine_loop
                    is_decode = False
                else:
                    active_threads.remove(t)
                    continue

                cam = next((c for c in RTSP_CAMERAS if c["camera_id"] == cam_key), None)
                if cam is None:
                    active_threads.remove(t)
                    continue

                logger.warning(f"♻️  Thread {name} not alive — respawning")
                # Only the decode thread owns stop/restart events + frame timestamp.
                if is_decode:
                    stop_events[cam_key]    = Event()
                    restart_events[cam_key] = Event()
                    last_frame_time[cam_key] = time.time()
                nt = Thread(target=target, args=(cam,), daemon=True, name=name)
                nt.start()
                active_threads[i] = nt

            # Respawn any InferenceScheduler worker that died unexpectedly.
            if _schedulers and not scheduler_stop_event.is_set():
                cam_keys  = [c["camera_id"] for c in RTSP_CAMERAS]
                n_workers = len(models)
                for i, s in enumerate(list(_schedulers)):
                    if s.is_alive():
                        continue
                    logger.error(f"💀 {s.name} died — respawning")
                    worker_cams = cam_keys[i::n_workers]
                    batch_size  = compute_batch_size(models[i], len(cam_keys), n_workers)
                    ns = InferenceScheduler(
                        model=models[i],
                        frame_buffer=frame_buffer,
                        result_queues=result_queues,
                        batch_size=batch_size,
                        cam_keys=worker_cams,
                        customer_cls_ids=customer_cls_ids,
                        staff_cls_ids=staff_cls_ids,
                        stop_event=scheduler_stop_event,
                        conf_threshold=CONF_THRESHOLD,
                        device=DEVICE,
                        min_w=MIN_W,
                        min_h=MIN_H,
                    )
                    ns.name = f"InferenceScheduler-{i}"
                    ns.start()
                    _schedulers[i] = ns

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt, shutting down...")
        if active_threads:
            _stop_camera_threads(active_threads)
        logger.info("✅ Shutdown complete")
 