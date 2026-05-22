# Meet & Greet (M&G) — KPI Detector

Meet & Greet is a multi-camera production-ready system to detect customer greeting events at retail entrances.

**This repository** contains the V4 multi-camera parallel runner that uses Ultralytics YOLO for detections, draws a preview HUD, and saves snapshots when configured greet conditions are met.

**Quick features**
- Multi-camera parallel threads
- Config-driven zones (entry + greet) using ratio coordinates
- Snapshot saving for meet & greet events
- Optional preview HUD and debug overlays
- Can run with RTSP streams or local video files for testing

**Requirements**
- Linux or macOS (display required for GUI windows)
- Python 3.9+
- GPU recommended for real-time inference (CUDA)

Install dependencies (example using pip):

```bash
python3 -m pip install -r requirements.txt
```

If you don't have a `requirements.txt`, at minimum install:

```bash
python3 -m pip install ultralytics opencv-python-headless torch numpy
```

Note: if you want GUI windows (`--show`/`--debug`), install `opencv-python` (non-headless) and ensure a display server is available.

**Configuration**
All runtime configuration is in `config.py` (camera list `RTSP_CAMERAS`, thresholds, model path, output dirs, etc.). Edit `config.py` to point to your model and camera(s).

**Usage**
Run all configured cameras (default):

```bash
python3 main.py
```

Run a single camera by index (index from 0):

```bash
python3 main.py --stream-index 0
```

Run multiple specific cameras by indices:

```bash
python3 main.py --stream-indices 0,2,3
```

Test with a local video file but use a camera's configuration (zones, site name, etc.). You must provide `--stream-index` to select which camera config to use with the local file:

```bash
python3 main.py /path/to/28-E.mp4 --stream-index 0
```

Force preview windows and enable debug logging/overlays:

```bash
python3 main.py --stream-indices 0,3,4 --show --debug
python3 main.py /path/to/28-E.mp4 --stream-index 0 --show --debug
```

Flags summary
- `video` (positional): optional local video file to run (requires `--stream-index`)
- `--stream-index`: integer index of camera in `RTSP_CAMERAS` to run
- `--stream-indices`: comma-separated list of indices to run (e.g. `0,1,2`)
- `--show`: force preview windows (overrides headless)
- `--debug`: enable debug logging and show windows

**How `--show` and `--debug` behave**
- `--show` forces display of the preview windows (if a display server is available).
- `--debug` sets logging to DEBUG and also forces preview windows. It can produce more console output and draw additional overlays when implemented.

**Outputs**
- Snapshots for confirmed meet & greet events are written under the `OUTPUT_BASE_DIR` configured in `config.py`, organized by site/date/camera.
- Logs are written to the `LOG_DIR` defined in `config.py`.

**Testing tips**
- To test layout and thresholds, run a local video with `--show --debug` and tune zone ratios in `config.py`.

**Development**
- The main program is `main.py`. Camera loop logic lives in `run_camera()`.
- Zones are defined as ratio coordinates and converted to pixel points on the first frame.

If you want, I can also add a `requirements.txt` and a minimal example `config.py` you can use for quick testing.

---
Updated: May 22, 2026
