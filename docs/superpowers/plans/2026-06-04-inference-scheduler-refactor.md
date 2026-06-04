# Inference Scheduler Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate GPU starvation across 28 camera streams by decoupling RTSP decode from inference, routing all inference through a single round-robin batch scheduler that guarantees every camera gets equal GPU time — zero missed greet events.

**Architecture:** One `DecodeThread` per camera writes the latest frame to a `FrameBuffer` slot. A single `InferenceScheduler` thread polls the buffer in round-robin order, batches `BATCH_SIZE` frames (computed from free VRAM at startup), calls `model.predict()` once per batch, then dispatches `InferenceResult` objects into per-camera `result_queues`. One `StateMachineThread` per camera reads from its queue and runs all existing greet logic unchanged.

**Tech Stack:** Python 3.12, Ultralytics YOLO, PyTorch CUDA, OpenCV, threading, queue, collections.deque

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `frame_buffer.py` | **CREATE** | `FrameSlot` dataclass + `FrameBuffer` class — thread-safe single-slot per camera |
| `inference_scheduler.py` | **CREATE** | `InferenceResult` dataclass + `InferenceScheduler` thread + `compute_batch_size()` |
| `main.py` | **MODIFY** | Replace `run_camera()` with `decode_loop()` + `state_machine_loop()`; wire new threading model into `_start_camera_threads()` / `_stop_camera_threads()` |

---

## Task 1: Create `frame_buffer.py`

**Files:**
- Create: `frame_buffer.py`

- [ ] **Step 1: Create the file**

```python
# frame_buffer.py
"""
Thread-safe single-slot frame buffer.
One slot per camera. Decode thread always overwrites with the latest frame.
InferenceScheduler reads and marks consumed (fresh=False).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameSlot:
    frame: np.ndarray | None = None
    frame_timestamp: float = 0.0
    fresh: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class FrameBuffer:
    """Holds one FrameSlot per camera. Decode threads write; scheduler reads."""

    def __init__(self, cam_keys: list[str]) -> None:
        self._slots: dict[str, FrameSlot] = {k: FrameSlot() for k in cam_keys}

    # --- decode thread calls this -------------------------------------------

    def write(self, cam_key: str, frame: np.ndarray, timestamp: float) -> None:
        """Overwrite slot with latest frame. Always succeeds — never blocks."""
        slot = self._slots[cam_key]
        with slot._lock:
            slot.frame = frame
            slot.frame_timestamp = timestamp
            slot.fresh = True

    # --- scheduler calls this ------------------------------------------------

    def read_and_consume(self, cam_key: str) -> tuple[np.ndarray | None, float]:
        """
        Return (frame, timestamp) if slot is fresh, else (None, 0.0).
        Marks slot as consumed so the same frame is never inferred twice.
        """
        slot = self._slots[cam_key]
        with slot._lock:
            if not slot.fresh or slot.frame is None:
                return None, 0.0
            frame = slot.frame
            ts = slot.frame_timestamp
            slot.fresh = False
            return frame, ts

    def cam_keys(self) -> list[str]:
        return list(self._slots.keys())
```

- [ ] **Step 2: Commit**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
git add frame_buffer.py
git commit -m "feat: add FrameBuffer — thread-safe single-slot per camera"
```

---

## Task 2: Create `inference_scheduler.py`

**Files:**
- Create: `inference_scheduler.py`

- [ ] **Step 1: Create the file**

```python
# inference_scheduler.py
"""
Round-robin batch InferenceScheduler.

Guarantees every camera gets equal GPU time — no starvation.
Single thread owns all model.predict() calls.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger("MeetGreet")

# Imported at call-site from config to avoid circular imports
_CONF_THRESHOLD: float = 0.30
_DEVICE: str = "cuda"
_MIN_W: int = 40
_MIN_H: int = 100


@dataclass
class InferenceResult:
    """Everything the state machine needs from one inference run on one camera."""
    cam_key: str
    frame: np.ndarray          # original resized frame (for snapshot saving)
    frame_timestamp: float
    customers: list            # list of np.ndarray [x1,y1,x2,y2] int
    green_staff: list          # list of np.ndarray [x1,y1,x2,y2] int


def compute_batch_size(model, num_cameras: int) -> int:
    """
    Compute batch size from currently free VRAM.
    Conservative: 8 MB per frame slot (640×640 RGB FP32 + YOLO activations).
    Caps at min(free_budget, num_cameras, 16).
    Falls back to 1 on CPU.
    """
    if not torch.cuda.is_available():
        logger.info("compute_batch_size: no CUDA → batch_size=1")
        return 1

    free_vram, _ = torch.cuda.mem_get_info()
    # 8 MB conservative estimate per frame in batch
    frame_vram_bytes = 8 * 1024 * 1024
    # Keep 25% headroom for other processes sharing the GPU
    usable = free_vram * 0.75
    batch_size = max(1, min(
        int(usable / frame_vram_bytes),
        num_cameras,
        16,
    ))
    logger.info(
        f"compute_batch_size: free_vram={free_vram / 1e9:.2f}GB "
        f"usable={usable / 1e9:.2f}GB "
        f"batch_size={batch_size} "
        f"(capped at min(budget={int(usable/frame_vram_bytes)}, "
        f"cams={num_cameras}, max=16))"
    )
    return batch_size


class InferenceScheduler(threading.Thread):
    """
    Single thread that owns all GPU inference.

    Round-robin: rotates through every camera in order, collects up to
    BATCH_SIZE fresh frames, calls model.predict() once, then dispatches
    InferenceResult objects into per-camera result queues.

    Fairness invariant: the rotation pointer advances on every camera slot
    regardless of whether that camera had a fresh frame. No camera can be
    skipped more than len(cameras) consecutive cycles.
    """

    def __init__(
        self,
        model,
        frame_buffer,           # FrameBuffer
        result_queues: dict,    # cam_key -> queue.Queue
        batch_size: int,
        cam_keys: list[str],
        customer_cls_ids: list[int],
        staff_cls_ids: list[int],
        stop_event: threading.Event,
        conf_threshold: float,
        device: str,
        min_w: int,
        min_h: int,
    ) -> None:
        super().__init__(daemon=True, name="InferenceScheduler")
        self.model = model
        self.frame_buffer = frame_buffer
        self.result_queues = result_queues
        self.batch_size = batch_size
        self.rotation: collections.deque[str] = collections.deque(cam_keys)
        self.customer_cls_ids = set(customer_cls_ids)
        self.staff_cls_ids = set(staff_cls_ids)
        self.stop_event = stop_event
        self.conf_threshold = conf_threshold
        self.device = device
        self.min_w = min_w
        self.min_h = min_h
        # Telemetry
        self._inference_count: dict[str, int] = {k: 0 for k in cam_keys}
        self._last_telemetry_ts: float = time.time()

    # -------------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            f"🚀 InferenceScheduler started | "
            f"cameras={len(self.rotation)} batch_size={self.batch_size}"
        )
        while not self.stop_event.is_set():
            batch_frames: list[np.ndarray] = []
            batch_cam_keys: list[str] = []
            batch_timestamps: list[float] = []

            # Round-robin fill — advance pointer unconditionally for fairness
            num_cams = len(self.rotation)
            checked = 0
            while len(batch_frames) < self.batch_size and checked < num_cams:
                cam_key = self.rotation[0]
                self.rotation.rotate(-1)   # advance BEFORE checking freshness
                checked += 1

                frame, ts = self.frame_buffer.read_and_consume(cam_key)
                if frame is None:
                    continue
                batch_frames.append(frame)
                batch_cam_keys.append(cam_key)
                batch_timestamps.append(ts)

            if not batch_frames:
                time.sleep(0.001)   # brief yield — avoid busy-spin
                continue

            # Single GPU call for entire batch
            try:
                results_list = self.model.predict(
                    batch_frames,
                    conf=self.conf_threshold,
                    device=self.device,
                    agnostic_nms=True,
                    verbose=False,
                )
            except Exception as exc:
                logger.error(f"InferenceScheduler predict error: {exc}", exc_info=True)
                continue

            # Parse and dispatch results
            for cam_key, result, frame, ts in zip(
                batch_cam_keys, results_list, batch_frames, batch_timestamps
            ):
                customers, green_staff = self._parse_boxes(result)
                inf_result = InferenceResult(
                    cam_key=cam_key,
                    frame=frame,
                    frame_timestamp=ts,
                    customers=customers,
                    green_staff=green_staff,
                )
                q = self.result_queues.get(cam_key)
                if q is not None:
                    self._put_result(q, inf_result)
                self._inference_count[cam_key] = self._inference_count.get(cam_key, 0) + 1

            self._log_telemetry()

    # -------------------------------------------------------------------------

    def _parse_boxes(self, result) -> tuple[list, list]:
        customers: list = []
        green_staff: list = []
        if result.boxes is None:
            return customers, green_staff
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for b, c, conf in zip(boxes, clss, confs):
            if conf < self.conf_threshold:
                continue
            if c in self.staff_cls_ids:
                green_staff.append(b)
            elif c in self.customer_cls_ids:
                if (b[2] - b[0]) >= self.min_w and (b[3] - b[1]) >= self.min_h:
                    customers.append(b)
        return customers, green_staff

    def _put_result(self, q: queue.Queue, result: InferenceResult) -> None:
        """Non-blocking put. If full, drop oldest and insert newest."""
        try:
            q.put_nowait(result)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(result)
            except queue.Full:
                pass

    def _log_telemetry(self) -> None:
        now = time.time()
        if now - self._last_telemetry_ts < 60:
            return
        elapsed = now - self._last_telemetry_ts
        rates = {k: round(v / elapsed, 2) for k, v in self._inference_count.items()}
        zero_rate_cams = [k for k, v in rates.items() if v == 0]
        if zero_rate_cams:
            logger.warning(f"⚠️  Zero inference rate: {zero_rate_cams}")
        else:
            min_rate = min(rates.values())
            logger.info(f"📊 Inference rates (fps): min={min_rate} | {rates}")
        # Reset counters
        self._inference_count = {k: 0 for k in self._inference_count}
        self._last_telemetry_ts = now
```

- [ ] **Step 2: Commit**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
git add inference_scheduler.py
git commit -m "feat: add InferenceScheduler — round-robin batch GPU inference, equal fairness"
```

---

## Task 3: Refactor `main.py` — add `decode_loop()`

**Files:**
- Modify: `main.py`

Replace the existing `run_camera()` function body with two focused functions.
`decode_loop()` handles only RTSP + frame writing. Insert it BEFORE the existing `run_camera()` definition. Keep `run_camera()` in place until Task 4 removes it.

- [ ] **Step 1: Add imports at the top of `main.py`**

Find the imports block (lines 1–48). Add after the existing imports:

```python
import collections
import queue as _queue
from frame_buffer import FrameBuffer
from inference_scheduler import InferenceResult, InferenceScheduler, compute_batch_size
```

- [ ] **Step 2: Add `zones_dict` and `result_queues` globals after existing globals (after line 71)**

```python
zones_dict: dict[str, dict] = {}       # cam_key → zone pixel coords (built on first decode frame)
result_queues: dict[str, _queue.Queue] = {}  # cam_key → Queue[InferenceResult]
frame_buffer: FrameBuffer | None = None
scheduler_stop_event: threading.Event = threading.Event()
```

- [ ] **Step 3: Add `decode_loop()` function — insert before existing `run_camera()` (before line 304)**

```python
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
    _last_infer_key = cam_key + "_throttle"

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

                # ── Drain stale frames ────────────────────────────────────────
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

                    # Flush enough frames to cover scheduler's worst-case latency.
                    # Worst case: batch_size * ~30ms per inference / frame_interval.
                    # Use min(fps*0.5, 15) as a safe heuristic; never below 2.
                    buffer_flush_count = max(2, min(int(fps_cam * 0.5), 15))

                    # Build zone pixel coords once and store in shared dict
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
                if frame_timestamp - last_frame_time.get(_last_infer_key, 0) < FRAME_INTERVAL:
                    continue
                last_frame_time[_last_infer_key] = frame_timestamp

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
```

- [ ] **Step 4: Commit**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
git add main.py
git commit -m "feat(main): add decode_loop() — RTSP decode separated from inference"
```

---

## Task 4: Refactor `main.py` — add `state_machine_loop()`, remove `run_camera()`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `state_machine_loop()` immediately after `decode_loop()`**

```python
# ===================== STATE MACHINE THREAD =====================
def state_machine_loop(cfg: dict) -> None:
    """
    Greet KPI logic — unchanged from original run_camera().
    Reads InferenceResult from per-camera result queue.
    No GPU access — pure CPU state machine.
    """
    cam_key = cfg["camera_id"]
    mode    = cfg["zone_mode"]

    state = {
        "session":              None,
        "rect2_hits":           0,
        "rect2_confirmed":      False,
        "rect2_last_seen_time": None,
        "cooldown_until":       0,
        "entry_candidate_hits": 0,
        "entry_candidate_start_time": None,
        "rect2_confirm_start_time":   None,
        "customer_count":       0,
        "staff_count":          0,
    }

    win_name = f"Meet & Greet — {cam_key}"
    if not HEADLESS:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    q = result_queues.get(cam_key)
    if q is None:
        logger.error(f"state_machine_loop: no result queue for {cam_key}")
        return

    while not stop_events[cam_key].is_set():
        # ── Get next inference result (blocks up to 1s) ───────────────────
        try:
            inf_result: InferenceResult = q.get(timeout=1.0)
        except _queue.Empty:
            # No inference arrived — check for session timeout using wall clock
            if state["session"]:
                now = time.time()
                if now - state["session"]["start"] >= SESSION_MAX_SEC:
                    logger.info(f"⏰ Session timed out @ {now} → {cam_key}")
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
            state["session"]             = None
            state["rect2_hits"]          = 0
            state["rect2_confirmed"]      = False
            state["rect2_last_seen_time"] = None
            state["entry_candidate_hits"] = 0

        # ── ENTRY ZONE (zone1) — session trigger ──────────────────────────
        cust_in_entry = customer_foot_in_zone(customers, zones["zone1_px"], mode)

        if cust_in_entry and state["session"] is None:
            state["session"] = {
                "start":                  frame_timestamp,
                "coexistence_start_time": None,
                "last_coexistence_time":  None,
                "done":                   False,
            }
            state["entry_candidate_hits"] = 0
            state["entry_candidate_start_time"] = None
            logger.info(f"👤 SESSION STARTED (instant) → {cam_key}")
        else:
            if state["session"] is None:
                state["entry_candidate_hits"] = 0
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
                state["rect2_hits"] = 0
                state["rect2_confirm_start_time"] = None

        # ── ABSENCE ABORT ─────────────────────────────────────────────────
        if (state["session"] and state["rect2_confirmed"] and
                state["rect2_last_seen_time"] and
                frame_timestamp - state["rect2_last_seen_time"] > RECT2_ABSENCE_ABORT_SEC):
            logger.info(f"🚫 Greet zone absence abort → {cam_key}")
            state["session"]             = None
            state["rect2_confirmed"]      = False
            state["rect2_hits"]           = 0
            state["rect2_last_seen_time"] = None
            state["entry_candidate_hits"] = 0

        # ── COMPLETED INTERACTION guard ───────────────────────────────────
        if (state["session"] and state["rect2_confirmed"] and
                state["session"].get("done")):
            cust_in_greet_done  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
            staff_in_greet_done = staff_intersects_zone(green_staff, zones["zone2_px"], mode)

            if cust_in_greet_done and staff_in_greet_done:
                state["rect2_last_seen_time"] = frame_timestamp
            elif (state["rect2_last_seen_time"] and
                    frame_timestamp - state["rect2_last_seen_time"] > GREET_GAP_TOLERANCE):
                logger.info(f"✅ Completed greet interaction reset → {cam_key}")
                state["session"]                  = None
                state["rect2_confirmed"]           = False
                state["rect2_hits"]                = 0
                state["rect2_last_seen_time"]      = None
                state["entry_candidate_hits"]      = 0
                state["rect2_confirm_start_time"]  = None

        # ── MEET & GREET hit counting + snapshot ──────────────────────────
        if (state["session"] and state["rect2_confirmed"] and
                not state["session"]["done"] and frame_timestamp >= state["cooldown_until"]):

            cust_in_greet  = [c for c in customers if box_in_zone(c, zones["zone2_px"], mode)]
            staff_in_greet = staff_intersects_zone(green_staff, zones["zone2_px"], mode)

            if cust_in_greet and staff_in_greet:
                state["rect2_last_seen_time"] = frame_timestamp

                if state["session"]["coexistence_start_time"] is None:
                    state["session"]["coexistence_start_time"] = frame_timestamp
                    state["session"]["last_coexistence_time"]  = frame_timestamp
                    logger.info(f"🤝 Customer + Staff coexistence started → {cam_key}")
                else:
                    gap_since_last = frame_timestamp - state["session"]["last_coexistence_time"]
                    if gap_since_last > GREET_GAP_TOLERANCE:
                        state["session"]["coexistence_start_time"] = frame_timestamp
                        logger.info(
                            f"⏸️ Coexistence gap reset ({gap_since_last:.1f}s > "
                            f"{GREET_GAP_TOLERANCE}s) → {cam_key}"
                        )
                    state["session"]["last_coexistence_time"] = frame_timestamp

                coexistence_duration = frame_timestamp - state["session"]["coexistence_start_time"]

                if coexistence_duration >= 2.0:
                    dur_str = (
                        f"{int(coexistence_duration)}sec"
                        if coexistence_duration < 60
                        else f"{round(coexistence_duration / 60, 2)}mins"
                    )

                    snap = frame.copy()
                    for b in cust_in_greet:
                        cv2.rectangle(snap, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
                        cv2.putText(snap, "CUSTOMER", (b[0], b[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    for b in staff_in_greet:
                        cv2.rectangle(snap, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
                        cv2.putText(snap, "STAFF", (b[0], b[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    all_boxes = cust_in_greet + staff_in_greet
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
                        entry_ts         = state["session"]["start"]
                        coexist_start_ts = state["session"]["coexistence_start_time"]
                        greet_ts         = frame_timestamp
                        meta["event_type"]                = "greet"
                        meta["site_name"]                 = cfg.get("site_name")
                        meta["site_id"]                   = cfg.get("site_id")
                        meta["camera_id"]                 = cfg.get("camera_id")
                        meta["entry_time_epoch"]          = entry_ts
                        meta["coexistence_start_epoch"]   = coexist_start_ts
                        meta["greet_time_epoch"]          = greet_ts
                        meta["entry_to_coexistence_sec"]  = coexist_start_ts - entry_ts
                        meta["coexistence_duration_sec"]  = coexistence_duration
                        meta["duration_str"]              = dur_str
                        meta["model_path"]                = MODEL_PATH
                        try:
                            meta["entry_time"]            = datetime.fromtimestamp(entry_ts, IST).isoformat()
                            meta["coexistence_start_time"] = datetime.fromtimestamp(coexist_start_ts, IST).isoformat()
                            meta["greet_time"]            = datetime.fromtimestamp(greet_ts, IST).isoformat()
                        except Exception:
                            meta["entry_time"]            = entry_ts
                            meta["coexistence_start_time"] = coexist_start_ts
                            meta["greet_time"]            = greet_ts

                        json_path = os.path.splitext(greet_path)[0] + ".json"
                        with open(json_path, "w") as jf:
                            import json as _json
                            _json.dump(meta, jf, indent=2)
                    except Exception as je:
                        logger.warning(f"Failed to write greet metadata JSON: {je}")

                    logger.info(f"📸 GREET SAVED → {greet_path}")

                    state["rect2_last_seen_time"]              = frame_timestamp
                    state["session"]["coexistence_start_time"] = None
                    state["session"]["last_coexistence_time"]  = frame_timestamp
                    state["session"]["done"]                   = True
                    state["cooldown_until"]                    = frame_timestamp + POST_SAVE_COOLDOWN_SEC

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
```

- [ ] **Step 2: Delete the old `run_camera()` function**

Remove lines from `# ===================== CAMERA THREAD =====================` through the closing `logger.info(f"🛑 EXITED → {cam_key}")` line (the entire `run_camera` function body including the enclosing function definition).

- [ ] **Step 3: Commit**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
git add main.py
git commit -m "feat(main): add state_machine_loop(), remove run_camera() — greet logic unchanged"
```

---

## Task 5: Refactor `main.py` — wire new threading model into start/stop/main

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace `load_model_on_gpu()` to also compute batch size and init shared structures**

Find the existing `load_model_on_gpu()` function (around line 710). Replace it entirely:

```python
# ===================== MODEL LOADER =====================
customer_cls_ids: list[int] = []
staff_cls_ids: list[int]    = []
_scheduler: InferenceScheduler | None = None

def load_model_on_gpu() -> None:
    """Load model, compute batch size, initialise FrameBuffer + result queues."""
    global model, customer_cls_ids, staff_cls_ids, frame_buffer, result_queues, scheduler_stop_event

    if model is not None:
        return  # already loaded

    logger.info(f"🧠 Loading model to {DEVICE.upper()} (CUDA Available: {torch.cuda.is_available()})")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    customer_cls_ids = []
    staff_cls_ids    = []
    for idx, name in model.names.items():
        name_low = name.lower()
        if CUSTOMER_LABEL in name_low:
            customer_cls_ids.append(idx)
        if any(s in name_low for s in GREEN_STAFF_LABELS):
            staff_cls_ids.append(idx)

    if not customer_cls_ids:
        raise RuntimeError(
            f"Model class names do not contain '{CUSTOMER_LABEL}'. "
            f"Available: {list(model.names.values())}. "
            f"Fix CUSTOMER_LABEL in config.py."
        )
    if not staff_cls_ids:
        raise RuntimeError(
            f"Model class names do not match any of {GREEN_STAFF_LABELS}. "
            f"Available: {list(model.names.values())}. "
            f"Fix GREEN_STAFF_LABELS in config.py."
        )

    logger.info(f"✅ Model loaded | Customer IDs: {customer_cls_ids} | Staff IDs: {staff_cls_ids}")

    cam_keys   = [c["camera_id"] for c in RTSP_CAMERAS]
    batch_size = compute_batch_size(model, len(cam_keys))

    frame_buffer   = FrameBuffer(cam_keys)
    result_queues  = {k: _queue.Queue(maxsize=2) for k in cam_keys}
    scheduler_stop_event = threading.Event()

    logger.info(f"🗂️  FrameBuffer + result_queues initialised for {len(cam_keys)} cameras | batch_size={batch_size}")
```

- [ ] **Step 2: Replace `unload_model_from_gpu()` to also stop the scheduler**

Find the existing `unload_model_from_gpu()` (around line 731). Replace it entirely:

```python
def unload_model_from_gpu() -> None:
    """Stop InferenceScheduler, unload model, free GPU memory."""
    global model, frame_buffer, result_queues, _scheduler

    if _scheduler is not None and _scheduler.is_alive():
        logger.info("⏹️  Stopping InferenceScheduler...")
        scheduler_stop_event.set()
        _scheduler.join(timeout=10)
        if _scheduler.is_alive():
            logger.warning("⚠️  InferenceScheduler did not stop in time")
        _scheduler = None

    if model is None:
        return

    logger.info("🧹 Unloading model from GPU...")
    model = None
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("✅ Model unloaded, GPU memory freed")
```

- [ ] **Step 3: Replace `_start_camera_threads()` inside `__main__` block**

Find `def _start_camera_threads():` (around line 838). Replace it entirely:

```python
def _start_camera_threads() -> list[threading.Thread]:
    """
    Start InferenceScheduler + decode + state_machine threads.
    Returns flat list of all threads (for join/monitor in main loop).
    """
    global _scheduler, scheduler_stop_event

    load_model_on_gpu()

    cam_keys   = [c["camera_id"] for c in RTSP_CAMERAS]
    batch_size = compute_batch_size(model, len(cam_keys))

    # Fresh stop event for this business-hours cycle
    scheduler_stop_event = threading.Event()

    # Start InferenceScheduler (single thread — owns all GPU calls)
    _scheduler = InferenceScheduler(
        model=model,
        frame_buffer=frame_buffer,
        result_queues=result_queues,
        batch_size=batch_size,
        cam_keys=cam_keys,
        customer_cls_ids=customer_cls_ids,
        staff_cls_ids=staff_cls_ids,
        stop_event=scheduler_stop_event,
        conf_threshold=CONF_THRESHOLD,
        device=DEVICE,
        min_w=MIN_W,
        min_h=MIN_H,
    )
    _scheduler.start()
    logger.info(f"🎬 Starting {len(RTSP_CAMERAS)} camera thread(s)...")

    threads: list[threading.Thread] = []
    for cam in RTSP_CAMERAS:
        cam_key = cam["camera_id"]
        restart_events[cam_key] = threading.Event()
        stop_events[cam_key]    = threading.Event()
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

    # Watchdog (unchanged)
    global watchdog_thread_obj
    if watchdog_thread_obj is None or not watchdog_thread_obj.is_alive():
        watchdog_thread_obj = Thread(target=watchdog_thread, daemon=True, name="watchdog")
        watchdog_thread_obj.start()

    return threads
```

- [ ] **Step 4: Replace `_stop_camera_threads()` inside `__main__` block**

Find `def _stop_camera_threads(threads):` (around line 863). Replace it entirely:

```python
def _stop_camera_threads(threads: list[threading.Thread]) -> bool:
    """Signal stop for all decode+state threads and InferenceScheduler, then join."""
    logger.info(f"🛑 Stopping {len(threads) // 2} camera(s)...")

    # Stop InferenceScheduler first — no more GPU calls
    scheduler_stop_event.set()

    # Signal all camera decode+state threads
    for cam in RTSP_CAMERAS:
        cam_key = cam["camera_id"]
        if cam_key in stop_events:
            stop_events[cam_key].set()
        if cam_key in restart_events:
            restart_events[cam_key].set()

    for t in threads:
        t.join(timeout=15)

    if _scheduler is not None and _scheduler.is_alive():
        _scheduler.join(timeout=10)

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
```

- [ ] **Step 5: Update the dead-thread respawn logic inside the main `while True` loop**

Find the block that detects dead threads and respawns them (around line 926). Replace:

```python
            # Detect dead threads — respawn decode or state machine independently
            for i, t in enumerate(list(active_threads)):
                if not t.is_alive():
                    # Thread name is "decode-<cam_key>" or "state-<cam_key>"
                    name = t.name
                    if name.startswith("decode-"):
                        cam_key = name[len("decode-"):]
                        target  = decode_loop
                        prefix  = "decode"
                    elif name.startswith("state-"):
                        cam_key = name[len("state-"):]
                        target  = state_machine_loop
                        prefix  = "state"
                    else:
                        active_threads.remove(t)
                        continue

                    cam = next((c for c in RTSP_CAMERAS if c["camera_id"] == cam_key), None)
                    if cam is None:
                        active_threads.remove(t)
                        continue

                    logger.warning(f"♻️  Thread {name} not alive — respawning")
                    if prefix == "decode":
                        stop_events[cam_key]    = threading.Event()
                        restart_events[cam_key] = threading.Event()
                        last_frame_time[cam_key] = time.time()
                    nt = Thread(
                        target=target, args=(cam,),
                        daemon=True, name=name,
                    )
                    nt.start()
                    active_threads[i] = nt

            # Respawn InferenceScheduler if it died unexpectedly
            if _scheduler is not None and not _scheduler.is_alive():
                logger.error("💀 InferenceScheduler died — respawning")
                scheduler_stop_event.clear()
                cam_keys   = [c["camera_id"] for c in RTSP_CAMERAS]
                batch_size = compute_batch_size(model, len(cam_keys))
                _scheduler = InferenceScheduler(
                    model=model,
                    frame_buffer=frame_buffer,
                    result_queues=result_queues,
                    batch_size=batch_size,
                    cam_keys=cam_keys,
                    customer_cls_ids=customer_cls_ids,
                    staff_cls_ids=staff_cls_ids,
                    stop_event=scheduler_stop_event,
                    conf_threshold=CONF_THRESHOLD,
                    device=DEVICE,
                    min_w=MIN_W,
                    min_h=MIN_H,
                )
                _scheduler.start()
```

- [ ] **Step 6: Commit**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
git add main.py
git commit -m "feat(main): wire InferenceScheduler into start/stop/respawn — full pipeline refactor"
```

---

## Task 6: Smoke-test end-to-end

**Files:**
- No file changes — validation only

- [ ] **Step 1: Verify imports load cleanly**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
source .venv/bin/activate
python -c "from frame_buffer import FrameBuffer; from inference_scheduler import InferenceScheduler, compute_batch_size; print('imports OK')"
```

Expected output:
```
imports OK
```

- [ ] **Step 2: Run against a single camera stream for 60 seconds, confirm scheduler log lines appear**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
source .venv/bin/activate
timeout 65 python main.py --stream-index 0 2>&1 | tee /tmp/smoke_test.log
grep -E "(InferenceScheduler|batch_size|compute_batch_size|VRAM|DECODE EXITED|STATE MACHINE)" /tmp/smoke_test.log
```

Expected to see lines like:
```
compute_batch_size: free_vram=X.XXB usable=X.XXB batch_size=N
🚀 InferenceScheduler started | cameras=1 batch_size=N
🎥 STARTED → GF-1-CAM-3 ...
📊 GF-1-CAM-3 Resolution: 1280x720 @ 25.0 FPS
```

- [ ] **Step 3: Run all streams for 90 seconds, confirm every camera gets inference (telemetry log)**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
source .venv/bin/activate
timeout 95 python main.py 2>&1 | tee /tmp/full_test.log
# After 60s the scheduler logs inference rates — verify no zeros
grep -E "(inference rates|Zero inference rate|⚠️)" /tmp/full_test.log
```

Expected: `📊 Inference rates (fps): min=X.XX` with no `Zero inference rate` lines.

- [ ] **Step 4: Verify no regressions — greet saves still work**

```bash
grep -E "(GREET SAVED|SESSION STARTED|GREET ZONE CONFIRMED)" /tmp/full_test.log | head -20
```

Expected: same log pattern as before refactor.

- [ ] **Step 5: Final commit**

```bash
cd "/home/ashish-ratna/PMJ/Meet&Greet"
git add docs/
git commit -m "docs: add inference-scheduler refactor implementation plan"
```

---

## Known edge cases handled

| Case | Handling |
|------|----------|
| Camera disconnects mid-session | `state_machine_loop` detects session timeout via wall clock on `queue.Empty` |
| VRAM too small (batch=1) | `compute_batch_size` returns `max(1,...)` — always at least sequential |
| Scheduler dies unexpectedly | Main loop respawns it with same `frame_buffer` + `result_queues` |
| Class name mismatch in model | `load_model_on_gpu` raises `RuntimeError` at startup — loud fail, not silent zero |
| Camera added/removed | Restart business-hours cycle — `load_model_on_gpu` reinitialises all structures |
| GPU not available | `compute_batch_size` returns 1, `DEVICE="cpu"` from config |
