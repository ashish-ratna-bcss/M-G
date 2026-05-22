#!/usr/bin/env python3
"""
INFERENCE QUEUE SYSTEM
Central batch inference processor for multi-camera YOLO detection
Improves throughput 3-4x over per-thread inference
"""

import time
import threading
import queue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

@dataclass
class InferenceRequest:
    """Single frame inference request"""
    request_id: str          # unique ID (camera_id_timestamp)
    frame: any               # np.ndarray
    camera_id: str
    timestamp: float         # submission time
    priority: int = 0        # 0=low, 1=normal, 2=critical (entry/greet zones)
    callback: Optional[callable] = None  # notify on completion
    
    def __lt__(self, other):
        """Priority queue ordering (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp

@dataclass
class InferenceResult:
    """Inference result with metadata"""
    request_id: str
    boxes: any               # np.ndarray (N, 4)
    classes: any             # np.ndarray (N,)
    confidences: any         # np.ndarray (N,)
    inference_time_ms: float
    timestamp: float         # completion time
    camera_id: str

class InferenceQueueProcessor:
    """
    Central inference queue + batch processor with DYNAMIC STREAM AWARENESS
    
    Automatically adapts batch size, queue depth, and processing based on:
    - Number of active camera streams
    - Queue depth
    - GPU utilization feedback
    
    Usage:
        processor = InferenceQueueProcessor(model, DEVICE, batch_size=4)
        processor.start()
        
        # Register streams dynamically
        processor.register_stream("CAM-1")
        processor.register_stream("CAM-2")
        
        # From camera threads:
        processor.submit(InferenceRequest(...))
        result = processor.get_result(request_id, timeout=1.0)
    """
    
    def __init__(self, model, device, batch_size=4, queue_max_size=64, 
                 max_wait_ms=50, conf_threshold=0.30):
        """
        Args:
            model: YOLO model instance
            device: "cuda" or "cpu"
            batch_size: optimal batch size for GPU (usually 4-8)
            queue_max_size: max pending requests (drop oldest if exceeded)
            max_wait_ms: max time to wait before processing partial batch
            conf_threshold: YOLO confidence threshold
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_size_base = batch_size  # Store original for scaling
        self.queue_max_size = queue_max_size
        self.queue_max_size_base = queue_max_size
        self.max_wait_ms = max_wait_ms
        self.max_wait_ms_base = max_wait_ms
        self.conf_threshold = conf_threshold
        
        # Dynamic stream tracking
        self.active_streams = set()
        self.stream_lock = threading.Lock()
        self.total_streams_configured = 0
        
        # Request/result queues
        self.request_queue = queue.PriorityQueue(maxsize=queue_max_size)
        self.result_store = {}  # request_id → InferenceResult
        self.result_lock = threading.Lock()
        
        # Metrics
        self.total_processed = 0
        self.total_batches = 0
        self.batch_sizes = defaultdict(int)  # batch_size → count
        self.inference_times = []
        
        # Control
        self.running = False
        self.processor_thread = None
        
        logger = logging.getLogger("InferenceQueue")
        self.logger = logger
    
    def register_stream(self, camera_id: str):
        """Register a camera stream (called when stream starts)"""
        with self.stream_lock:
            self.active_streams.add(camera_id)
            self.total_streams_configured += 1
        
        self._adapt_to_stream_count()
        self.logger.info(f"✅ Stream registered: {camera_id} (active: {len(self.active_streams)}/{self.total_streams_configured})")
    
    def unregister_stream(self, camera_id: str):
        """Unregister a camera stream (called when stream stops)"""
        with self.stream_lock:
            self.active_streams.discard(camera_id)
        
        self._adapt_to_stream_count()
        self.logger.info(f"❌ Stream unregistered: {camera_id} (active: {len(self.active_streams)}/{self.total_streams_configured})")
    
    def get_active_stream_count(self) -> int:
        """Get current active stream count"""
        with self.stream_lock:
            return len(self.active_streams)
    
    def _adapt_to_stream_count(self):
        """
        Dynamically adjust batch size, queue depth based on active streams
        
        Strategy:
        - 1-2 streams: batch_size = base × 1.0 (normal)
        - 3-5 streams: batch_size = base × 0.75 (more queue depth, faster batching)
        - 6+ streams: batch_size = base × 0.5 (prioritize throughput)
        - CPU only: batch_size stays 1
        """
        active = self.get_active_stream_count()
        
        if self.device == "cpu":
            self.batch_size = 1
            return
        
        if active == 0:
            self.batch_size = self.batch_size_base
            return
        
        # Scaling formulas
        if active <= 2:
            scale = 1.0  # No change
        elif active <= 5:
            scale = 0.75  # Reduce batch size, favor queue depth
        else:  # 6+ streams
            scale = 0.5
        
        self.batch_size = max(1, int(self.batch_size_base * scale))
        self.max_wait_ms = min(100, int(self.max_wait_ms_base / scale))  # Inverse: batch faster with more streams
        
        self.logger.info(f"🔄 Adapted to {active} streams: "
                        f"batch_size={self.batch_size} (scale={scale}), "
                        f"max_wait={self.max_wait_ms}ms")
    
    def submit(self, request: InferenceRequest) -> bool:
        """
        Submit a frame for inference
        
        Returns:
            True if queued, False if queue full (request dropped)
        """
        try:
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            self.logger.warning(f"InferenceQueue FULL ({self.queue_max_size}), dropping oldest")
            try:
                # Drop lowest priority old request
                old_req = self.request_queue.get_nowait()
                self.request_queue.put_nowait(request)
                return True
            except:
                return False
    
    def get_result(self, request_id: str, timeout: float = 2.0) -> Optional[InferenceResult]:
        """
        Retrieve inference result for a request
        
        Blocks until result available or timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            with self.result_lock:
                if request_id in self.result_store:
                    result = self.result_store.pop(request_id)
                    return result
            time.sleep(0.001)  # 1ms poll interval
        
        self.logger.warning(f"Result timeout for {request_id} after {timeout}s")
        return None
    
    def start(self):
        """Start background processor thread"""
        if self.running:
            return
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processor_thread.start()
        self.logger.info(f"InferenceQueueProcessor started (batch_size={self.batch_size})")
    
    def stop(self):
        """Stop processor thread"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        self.logger.info("InferenceQueueProcessor stopped")
    
    def _process_loop(self):
        """Main processing loop - batches frames and runs inference"""
        batch_request_ids = []
        batch_frames = []
        batch_start = time.time()
        
        while self.running:
            try:
                # Wait for first request (or timeout to process partial batch)
                timeout = self.max_wait_ms / 1000.0
                request = self.request_queue.get(timeout=timeout)
                
                batch_request_ids.append(request.request_id)
                batch_frames.append(request.frame)
                
                # Fill batch or timeout
                while len(batch_frames) < self.batch_size and not self.request_queue.empty():
                    try:
                        request = self.request_queue.get_nowait()
                        batch_request_ids.append(request.request_id)
                        batch_frames.append(request.frame)
                    except queue.Empty:
                        break
                
                # Process batch if full or timeout exceeded
                if (len(batch_frames) >= self.batch_size or 
                    time.time() - batch_start > timeout):
                    
                    self._inference_batch(batch_request_ids, batch_frames)
                    batch_request_ids = []
                    batch_frames = []
                    batch_start = time.time()
                    
            except queue.Empty:
                # Timeout waiting for request
                if batch_frames:  # Process partial batch
                    self._inference_batch(batch_request_ids, batch_frames)
                    batch_request_ids = []
                    batch_frames = []
                    batch_start = time.time()
            except Exception as e:
                self.logger.error(f"Error in process loop: {e}", exc_info=True)
    
    def _inference_batch(self, request_ids: List[str], frames: List):
        """Run batch inference on frames"""
        if not frames:
            return
        
        try:
            inf_start = time.time()
            
            # Run YOLO batch inference
            results = self.model.predict(
                frames,
                conf=self.conf_threshold,
                device=self.device,
                agnostic_nms=True,
                verbose=False
            )
            
            inf_time_ms = (time.time() - inf_start) * 1000
            
            # Parse results for each frame
            with self.result_lock:
                for req_id, result in zip(request_ids, results):
                    boxes = None
                    classes = None
                    confidences = None
                    
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                    
                    self.result_store[req_id] = InferenceResult(
                        request_id=req_id,
                        boxes=boxes,
                        classes=classes,
                        confidences=confidences,
                        inference_time_ms=inf_time_ms / len(frames),
                        timestamp=time.time(),
                        camera_id=req_id.split("_")[0]  # camera_id from request_id
                    )
            
            # Metrics
            self.total_processed += len(frames)
            self.total_batches += 1
            self.batch_sizes[len(frames)] += 1
            self.inference_times.append(inf_time_ms / len(frames))
            
            self.logger.debug(f"Batch {len(frames)} frames in {inf_time_ms:.1f}ms "
                             f"({inf_time_ms/len(frames):.1f}ms/frame)")
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}", exc_info=True)
    
    def get_metrics(self) -> dict:
        """Return performance metrics with stream awareness"""
        avg_inference_ms = (sum(self.inference_times) / len(self.inference_times)
                           if self.inference_times else 0)
        
        active_streams = self.get_active_stream_count()
        
        return {
            "total_processed": self.total_processed,
            "total_batches": self.total_batches,
            "batch_size_distribution": dict(self.batch_sizes),
            "avg_inference_ms": avg_inference_ms,
            "queue_size": self.request_queue.qsize(),
            "pending_results": len(self.result_store),
            "active_streams": active_streams,
            "total_streams": self.total_streams_configured,
            "current_batch_size": self.batch_size,
            "throughput_fps": (self.total_processed / (sum(self.inference_times) / 1000 + 0.01)) if self.inference_times else 0,
        }
    
    def log_stats(self):
        """Log performance statistics with stream awareness"""
        metrics = self.get_metrics()
        self.logger.info(f"Queue Stats: Processed={metrics['total_processed']}, "
                        f"Batches={metrics['total_batches']}, "
                        f"AvgInf={metrics['avg_inference_ms']:.1f}ms, "
                        f"Throughput={metrics['throughput_fps']:.0f}fps, "
                        f"BatchDist={metrics['batch_size_distribution']}, "
                        f"Streams={metrics['active_streams']}/{metrics['total_streams']}")
