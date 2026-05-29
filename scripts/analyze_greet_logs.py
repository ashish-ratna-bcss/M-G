#!/usr/bin/env python3
"""
Simple analyser for meet_greet logs to compute per-camera greet timings.
Parses `logs/meet_greet_log.txt` (or a path you provide) and reports
statistics for the time between "SESSION STARTED" and "GREET SAVED".

Usage:
  python3 scripts/analyze_greet_logs.py [path/to/meet_greet_log.txt]

This helps tune `SESSION_MAX_SEC`, `GREET_HIT_THRESHOLD`, and
to verify whether time-based confirmation is feasible for your streams.
"""
import sys
import re
from datetime import datetime
from collections import defaultdict, deque
import statistics

LOG_RE_SESSION = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - .* - .*SESSION STARTED → (\S+)")
LOG_RE_GREET = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .* - .*GREET SAVED → (.+)")
LOG_RE_GREET_CAM = re.compile(r"greet_\d+_([^_/]+)_(?:\d{2}-\d{2}-\d{4})_\d{2}-\d{2}-\d{2}_.+\.png")

DEFAULT_LOG = "logs/meet_greet_log.txt"


def parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")


def analyze(log_path):
    starts = defaultdict(deque)  # cam -> deque of start datetimes (unmatched)
    durations = defaultdict(list)  # cam -> list of seconds between start and greet
    lines = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for L in lines:
        L = L.strip()
        m = LOG_RE_SESSION.search(L)
        if m:
            # extract timestamp at start of line
            ts = L[:23]
            cam = m.group(1)
            try:
                starts[cam].append(parse_ts(ts))
            except Exception:
                continue
            continue

        mg = LOG_RE_GREET.search(L)
        if mg:
            ts_str, path = mg.group(1), mg.group(2)
            try:
                greet_ts = parse_ts(ts_str)
            except Exception:
                continue
            # try to infer camera id from saved filename
            mc = LOG_RE_GREET_CAM.search(path)
            cam = mc.group(1) if mc else None
            if cam and starts.get(cam):
                start_ts = starts[cam].popleft()
                delta = (greet_ts - start_ts).total_seconds()
                durations[cam].append(delta)
            # else: unmatched greet saved — record under 'unknown' key
    # report
    if not durations:
        print("No matched session->greet events found in logs.")
        return

    print(f"Analyzed log: {log_path}\n")
    print("Reporting MINIMUM session->greet time per camera (useful for conservative timer tuning):\n")
    for cam, ds in sorted(durations.items(), key=lambda x: -len(x[1])):
        cnt = len(ds)
        mn = min(ds)
        print(f"Camera: {cam}")
        print(f"  Samples: {cnt}")
        print(f"  Min: {mn:.2f}s\n")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    try:
        analyze(path)
    except FileNotFoundError:
        print(f"Log file not found: {path}")
        sys.exit(2)
