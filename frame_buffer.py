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
