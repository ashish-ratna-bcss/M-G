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
    Conservative: 8 MB per frame slot (640x640 RGB FP32 + YOLO activations).
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
        f"(capped at min(budget={int(usable / frame_vram_bytes)}, "
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
        frame_buffer,            # FrameBuffer
        result_queues: dict,     # cam_key -> queue.Queue
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
