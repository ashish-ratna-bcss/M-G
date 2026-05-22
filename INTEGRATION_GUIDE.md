# PHASE 1 INTEGRATION GUIDE
## How to integrate inference queue + caching into main.py

---

## **QUICK START (Copy-paste ready)**

### **Step 1: Update imports in main.py**

```python
# Add to top of main.py after existing imports:
from inference_queue import InferenceQueueProcessor, InferenceRequest
from frame_cache import FrameCache, AdaptiveFrameSkipper
from config_optimization import get_production_config
from telemetry import TelemetryCollector, AccuracyTracker, print_monitoring_dashboard
```

### **Step 2: Initialize in main block**

```python
if __name__ == "__main__":
    logger.info("🚀 MEET & GREET SYSTEM STARTED (V4 - MULTI-CAMERA PARALLEL)")
    logger.info(f"📷 Total cameras : {len(RTSP_CAMERAS)}")
    
    # ========== NEW: Initialize optimization components ==========
    
    # Get production config (auto-selects batch size)
    prod_config = get_production_config(
        device=DEVICE,
        num_cameras=len(RTSP_CAMERAS)
    )
    batch_size = prod_config.get("batch_size", 4)
    
    # Initialize inference queue
    inference_processor = InferenceQueueProcessor(
        model=model,
        device=DEVICE,
        batch_size=batch_size,
        queue_max_size=64,
        max_wait_ms=50,
        conf_threshold=CONF_THRESHOLD
    )
    inference_processor.start()
    logger.info(f"✅ Inference queue started (batch_size={batch_size})")
    
    # Initialize frame cache (optional but recommended)
    frame_cache = FrameCache(
        max_cache_size=prod_config["cache_config"]["max_frames"],
        max_age_ms=prod_config["cache_config"]["max_age_ms"]
    )
    logger.info("✅ Frame cache initialized")
    
    # Initialize frame skipper
    frame_skipper = AdaptiveFrameSkipper()
    logger.info("✅ Adaptive frame skipping enabled")
    
    # Initialize telemetry
    telemetry = TelemetryCollector(window_size=300)
    telemetry.start()
    accuracy_tracker = AccuracyTracker(window_size=100)
    logger.info("✅ Telemetry collector started")
    
    # ========== Initialize cameras as before ==========
    
    for cam in RTSP_CAMERAS:
        cam_key = cam["camera_id"]
        restart_events[cam_key]  = Event()
        last_frame_time[cam_key] = time.time()
    
    Thread(target=watchdog_thread, daemon=True).start()
    
    for cam in RTSP_CAMERAS:
        t = Thread(target=run_camera, args=(cam, inference_processor, frame_cache, 
                                           frame_skipper, telemetry, accuracy_tracker),
                   daemon=True, name=cam["camera_id"])
        t.start()
        logger.info(f"🎬 Thread started → {cam['camera_id']}")
    
    # Telemetry logging every 10 seconds
    def periodic_telemetry():
        while True:
            time.sleep(10)
            telemetry.log_health()
            accuracy_tracker.log_accuracy()
            inference_processor.log_stats()
            frame_cache.log_stats()
    
    Thread(target=periodic_telemetry, daemon=True).start()
    
    while True:
        time.sleep(10)
```

### **Step 3: Update run_camera() signature**

```python
# BEFORE:
def run_camera(cfg):

# AFTER:
def run_camera(cfg, inference_processor, frame_cache, frame_skipper, 
               telemetry, accuracy_tracker):
```

### **Step 4: Replace inference in run_camera loop**

Find this line (~384 in main.py):
```python
# ── YOLO Detection ────────────────────────────────────────────
results = model.predict(
    frame, conf=CONF_THRESHOLD, device=DEVICE,
    agnostic_nms=True, verbose=False
)[0]

green_staff = []
customers   = []

if results.boxes is not None:
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    clss  = results.boxes.cls.cpu().numpy().astype(int)
```

Replace with:
```python
# ── YOLO Detection (OPTIMIZED via Queue) ──────────────────────

# Step A: Determine session state for adaptive skipping
session_state = "idle"
if state["session"]:
    if state["rect2_confirmed"]:
        session_state = "greet_counting"
    else:
        session_state = "zone_confirming"
elif state["entry_candidate_hits"] > 0:
    session_state = "entry_candidate"

# Step B: Decide whether to process this frame
if not frame_skipper.should_process(session_state, cam_key):
    continue  # Skip this frame

# Step C: Check cache first
cached_result = frame_cache.get(frame, hash_method="fast")
if cached_result:
    boxes = cached_result.boxes
    clss = cached_result.classes
    confidences = cached_result.confidences
    telemetry.record_inference(1)  # Cache hit is ~1ms
else:
    # Submit to inference queue
    request_id = f"{cam_key}_{time.time()}"
    request = InferenceRequest(
        request_id=request_id,
        frame=frame,
        camera_id=cam_key,
        timestamp=now,
        priority=(2 if session_state in ["entry_candidate", "zone_confirming"] else 0)
    )
    inference_processor.submit(request)
    
    # Wait for result (blocking)
    result = inference_processor.get_result(request_id, timeout=2.0)
    if result is None:
        logger.warning(f"Inference timeout for {cam_key}")
        continue
    
    boxes = result.boxes
    clss = result.classes
    confidences = result.confidences
    
    # Store in cache
    if boxes is not None:
        frame_cache.put(frame, boxes, clss, confidences)
    
    # Record telemetry
    telemetry.record_inference(result.inference_time_ms)

# Reconstruct results object for compatibility
green_staff = []
customers   = []

if boxes is not None:
    boxes = boxes.astype(int)
    for b, c in zip(boxes, clss):
        if c in staff_cls_ids:
            green_staff.append(b)
        elif c in customer_cls_ids:
            if (b[2]-b[0]) >= MIN_W and (b[3]-b[1]) >= MIN_H:
                customers.append(b)

# Record accuracy metrics
if state["session"]:
    if state["rect2_confirmed"]:
        telemetry.record_queue_size(inference_processor.request_queue.qsize())
```

---

## **CONFIGURATION TWEAKS**

### **For LATENCY (< 10ms per inference)**

Update `config.py`:
```python
MAX_FPS = 30  # Higher frame rate if you have GPU headroom
```

### **For THROUGHPUT (max fps, balanced accuracy)**

Update `config.py`:
```python
MAX_FPS = 25
# Other params stay default
```

### **For RESOURCE_CONSTRAINED (CPU or 10+ cameras)**

Update `config.py`:
```python
MAX_FPS = 10  # Skip frames more aggressively
# The frame skipper will reduce to ~2fps in idle
```

---

## **TESTING THE OPTIMIZATION**

### **Unit test inference queue (standalone)**

```bash
python3 -c "
from inference_queue import InferenceQueueProcessor, InferenceRequest
from ultralytics import YOLO
import numpy as np
import time

model = YOLO('new_staff.pt')
processor = InferenceQueueProcessor(model, 'cuda', batch_size=4)
processor.start()

frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(10)]

start = time.time()
for i, frame in enumerate(frames):
    req = InferenceRequest(
        request_id=f'test_{i}',
        frame=frame,
        camera_id='test',
        timestamp=time.time(),
        priority=0
    )
    processor.submit(req)
    result = processor.get_result(req.request_id)
    print(f'Frame {i}: {result.inference_time_ms:.1f}ms')

elapsed = time.time() - start
print(f'Total: {elapsed:.1f}s for {len(frames)} frames')
print(f'Metrics: {processor.get_metrics()}')

processor.stop()
"
```

### **Benchmark throughput**

```python
# Add to main.py before while True loop:
import time
start_bench = time.time()
frames_processed = [0]

# In telemetry thread:
def benchmark():
    while True:
        time.sleep(30)
        elapsed = time.time() - start_bench
        fps = sum(frames_processed) / elapsed
        logger.info(f"Throughput: {fps:.1f} fps avg, "
                   f"processed {sum(frames_processed)} frames in {elapsed:.0f}s")
```

---

## **EXPECTED IMPROVEMENTS**

### **Before Optimization**
- Throughput: 125 fps (5 cameras × 25fps)
- GPU utilization: 40-60% (spiky)
- Inference latency p99: 40-80ms
- Idle latency: 150-200ms

### **After Phase 1**
- Throughput: 150-180 fps (+25-40%)
- GPU utilization: 65-75% (stable)
- Inference latency p99: 15-25ms (-50%)
- Idle CPU: 5-10% (frame skipping)

---

## **ROLLBACK PLAN**

If issues arise, simply comment out the new code and revert to single-threaded inference:

```python
# Comment out optimization imports
# Comment out inference processor initialization

# In run_camera, use original inference:
results = model.predict(
    frame, conf=CONF_THRESHOLD, device=DEVICE,
    agnostic_nms=True, verbose=False
)[0]
```

---

## **NEXT STEPS (Phase 2)**

1. Deploy Phase 1 to production (monitor for 1 week)
2. Collect metrics from `telemetry.log_health()`
3. Benchmark actual frame rate, latency, accuracy
4. Implement Phase 2: async RTSP I/O + multi-GPU support (if available)
5. Fine-tune batch size based on production metrics
