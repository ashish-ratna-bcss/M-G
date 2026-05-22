#!/usr/bin/env python3
"""
TELEMETRY & MONITORING
Real-time GPU/CPU metrics, latency tracking, accuracy monitoring
"""

import time
import threading
import psutil
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class SystemMetrics:
    """Snapshot of system performance"""
    timestamp: float
    gpu_used_mb: float
    gpu_total_mb: float
    gpu_util_pct: float
    cpu_util_pct: float
    memory_used_mb: float
    memory_total_mb: float
    inference_latency_ms: float
    queue_size: int
    cache_hit_rate_pct: float

class TelemetryCollector:
    """
    Real-time performance monitoring with STREAM-AWARENESS
    
    Tracks:
    - Per-stream inference latency
    - Queue depth per stream
    - Stream load distribution
    - Dynamic resource adaptation
    """
    
    def __init__(self, window_size: int = 300):
        """
        Args:
            window_size: collect last N datapoints (~5min at 1Hz)
        """
        self.window_size = window_size
        self.metrics_window = deque(maxlen=window_size)
        self.inference_latencies = deque(maxlen=1000)
        self.inference_latencies_per_stream = {}  # stream_id → deque of latencies
        self.queue_sizes = deque(maxlen=100)
        
        self.running = False
        self.collector_thread = None
        
        # Stream tracking
        self.stream_count_history = deque(maxlen=window_size)
        self.active_streams = set()
        self.stream_lock = threading.Lock()
        
        self.logger = logging.getLogger("Telemetry")
    
    def record_inference(self, latency_ms: float):
        """Record inference latency"""
        self.inference_latencies.append(latency_ms)
    
    def record_inference_per_stream(self, stream_id: str, latency_ms: float):
        """Record inference latency for specific stream"""
        if stream_id not in self.inference_latencies_per_stream:
            self.inference_latencies_per_stream[stream_id] = deque(maxlen=500)
        self.inference_latencies_per_stream[stream_id].append(latency_ms)
    
    def register_stream(self, stream_id: str):
        """Register stream for tracking"""
        with self.stream_lock:
            self.active_streams.add(stream_id)
    
    def unregister_stream(self, stream_id: str):
        """Unregister stream"""
        with self.stream_lock:
            self.active_streams.discard(stream_id)
    
    def start(self):
        """Start background telemetry collector"""
        if self.running:
            return
        self.running = True
        self.collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.collector_thread.start()
        self.logger.info("Telemetry collector started")
    
    def stop(self):
        """Stop collector"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
    
    def _collect_loop(self):
        """Periodic metrics collection (1Hz)"""
        while self.running:
            try:
                metrics = self._capture_metrics()
                self.metrics_window.append(metrics)
                
                # Track stream count history
                with self.stream_lock:
                    stream_count = len(self.active_streams)
                self.stream_count_history.append(stream_count)
                
                time.sleep(1.0)  # Collect every second
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
    
    def _capture_metrics(self) -> SystemMetrics:
        """Capture current system state with per-stream stats"""
        gpu_used_mb = 0
        gpu_total_mb = 0
        gpu_util_pct = 0
        
        try:
            if torch.cuda.is_available():
                gpu_used_mb = torch.cuda.memory_allocated() / (1024**2)
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                gpu_util_pct = (gpu_used_mb / gpu_total_mb) * 100 if gpu_total_mb > 0 else 0
        except:
            pass
        
        cpu_util_pct = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024**2)
        memory_total_mb = memory.total / (1024**2)
        
        # Inference latency percentile (aggregate)
        inference_latency_ms = 0
        if self.inference_latencies:
            inference_latency_ms = sum(self.inference_latencies) / len(self.inference_latencies)
        
        # Per-stream latency stats (logged separately)
        with self.stream_lock:
            per_stream_stats = {}
            for stream_id, latencies in self.inference_latencies_per_stream.items():
                if latencies:
                    data = sorted(list(latencies))
                    per_stream_stats[stream_id] = {
                        'p50': data[len(data) // 2],
                        'p99': data[int(len(data) * 0.99)] if len(data) > 1 else data[0],
                        'mean': sum(data) / len(data),
                        'max': data[-1],
                        'count': len(data)
                    }
        
        # Queue size average
        queue_size = 0
        if self.queue_sizes:
            queue_size = sum(self.queue_sizes) / len(self.queue_sizes)
        
        # Store per-stream stats as custom attribute for logging
        metrics = SystemMetrics(
            timestamp=time.time(),
            gpu_used_mb=gpu_used_mb,
            gpu_total_mb=gpu_total_mb,
            gpu_util_pct=gpu_util_pct,
            cpu_util_pct=cpu_util_pct,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            inference_latency_ms=inference_latency_ms,
            queue_size=queue_size,
            cache_hit_rate_pct=0,  # Will be set by caller
        )
        
        # Attach per-stream stats for logging (not part of dataclass)
        metrics.per_stream_stats = per_stream_stats
        metrics.active_stream_count = len(self.active_streams)
        
        return metrics
    
    def get_stats(self) -> Dict:
        """Return aggregated statistics"""
        if not self.metrics_window:
            return {}
        
        metrics_list = list(self.metrics_window)
        
        gpu_utils = [m.gpu_util_pct for m in metrics_list]
        cpu_utils = [m.cpu_util_pct for m in metrics_list]
        inference_lats = [m.inference_latency_ms for m in metrics_list]
        
        # Percentiles for inference latency
        sorted_infs = sorted(self.inference_latencies)
        p50_inf = sorted_infs[len(sorted_infs) // 2] if sorted_infs else 0
        p99_inf = sorted_infs[int(len(sorted_infs) * 0.99)] if sorted_infs else 0
        
        return {
            "gpu_util_pct": {
                "avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                "max": max(gpu_utils) if gpu_utils else 0,
                "min": min(gpu_utils) if gpu_utils else 0,
            },
            "cpu_util_pct": {
                "avg": sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0,
                "max": max(cpu_utils) if cpu_utils else 0,
            },
            "inference_latency_ms": {
                "avg": sum(inference_lats) / len(inference_lats) if inference_lats else 0,
                "p50": p50_inf,
                "p99": p99_inf,
                "max": max(self.inference_latencies) if self.inference_latencies else 0,
            },
            "queue_size_avg": sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0,
        }
    
    def check_slo(self, slo_dict: Dict) -> Dict[str, bool]:
        """
        Check if metrics comply with SLO (stream-aware)
        
        Args:
            slo_dict: {
                "p50_inference_ms": 8, 
                "p99_inference_ms": 15, 
                "gpu_max_util_pct": 85,
                "per_stream_p99_ms": 20  # Per-stream SLO threshold
            }
        
        Returns:
            {
                "p50_ok": bool, 
                "p99_ok": bool, 
                "gpu_ok": bool,
                "per_stream_ok": dict  # stream_id -> bool
            }
        """
        stats = self.get_stats()
        infs = stats.get("inference_latency_ms", {})
        
        # Aggregate compliance
        compliance = {
            "p50_ok": infs.get("p50", 999) <= slo_dict.get("p50_inference_ms", 999),
            "p99_ok": infs.get("p99", 999) <= slo_dict.get("p99_inference_ms", 999),
            "gpu_ok": stats.get("gpu_util_pct", {}).get("avg", 100) <= slo_dict.get("gpu_max_util_pct", 100),
        }
        
        # Per-stream compliance
        per_stream_ok = {}
        if hasattr(self, '_current_metrics') and hasattr(self._current_metrics, 'per_stream_stats'):
            per_stream_p99_threshold = slo_dict.get("per_stream_p99_ms", 25)
            for stream_id, stats_dict in self._current_metrics.per_stream_stats.items():
                per_stream_ok[stream_id] = stats_dict.get('p99', 999) <= per_stream_p99_threshold
        
        compliance["per_stream_ok"] = per_stream_ok
        return compliance
    
    def log_health(self):
        """Log system health including per-stream metrics"""
        stats = self.get_stats()
        
        # Aggregate health
        info_msg = (f"Health: GPU={stats.get('gpu_util_pct', {}).get('avg', 0):.1f}% "
                   f"CPU={stats.get('cpu_util_pct', {}).get('avg', 0):.1f}% "
                   f"InfLat_p50={stats.get('inference_latency_ms', {}).get('p50', 0):.1f}ms "
                   f"InfLat_p99={stats.get('inference_latency_ms', {}).get('p99', 0):.1f}ms "
                   f"Queue={stats.get('queue_size_avg', 0):.0f} "
                   f"Streams={len(self.active_streams)}/total")
        
        self.logger.info(info_msg)
        
        # Per-stream health
        if self.inference_latencies_per_stream:
            for stream_id, latencies in self.inference_latencies_per_stream.items():
                if latencies:
                    data = sorted(list(latencies))
                    p50 = data[len(data) // 2]
                    p99 = data[int(len(data) * 0.99)] if len(data) > 1 else data[0]
                    mean = sum(data) / len(data)
                    self.logger.info(f"  Stream[{stream_id}] p50={p50:.1f}ms p99={p99:.1f}ms mean={mean:.1f}ms samples={len(data)}")

class AccuracyTracker:
    """
    Track detection accuracy & false positive rates
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: track last N detections per zone
        """
        self.detections_per_zone = {
            "entry_zone": deque(maxlen=window_size),
            "greet_zone": deque(maxlen=window_size),
        }
        
        self.false_positives = 0
        self.true_positives = 0
        self.false_negatives = 0
        
        self.logger = logging.getLogger("Accuracy")
    
    def record_entry_detection(self, num_customers: int, num_staff: int):
        """Record entry zone detection"""
        self.detections_per_zone["entry_zone"].append({
            "customers": num_customers,
            "staff": num_staff,
            "timestamp": time.time()
        })
    
    def record_greet_detection(self, num_customers: int, num_staff: int):
        """Record greet zone detection"""
        self.detections_per_zone["greet_zone"].append({
            "customers": num_customers,
            "staff": num_staff,
            "timestamp": time.time()
        })
    
    def get_detection_rate(self, zone: str) -> float:
        """Get percentage of frames with ≥1 detection"""
        detections = self.detections_per_zone.get(zone, [])
        if not detections:
            return 0.0
        
        with_detection = sum(1 for d in detections if d["customers"] > 0 or d["staff"] > 0)
        return (with_detection / len(detections)) * 100
    
    def log_accuracy(self):
        """Log accuracy metrics"""
        entry_rate = self.get_detection_rate("entry_zone")
        greet_rate = self.get_detection_rate("greet_zone")
        
        self.logger.info(f"Accuracy: Entry zone detection rate={entry_rate:.1f}% "
                        f"Greet zone detection rate={greet_rate:.1f}%")

# ===================== PRODUCTION MONITORING DASHBOARD =====================

def print_monitoring_dashboard(telemetry: TelemetryCollector, accuracy: AccuracyTracker):
    """
    Print formatted monitoring dashboard
    """
    stats = telemetry.get_stats()
    
    print("\n" + "="*70)
    print(" PRODUCTION MONITORING DASHBOARD")
    print("="*70)
    
    print(f"\n🖥️  GPU UTILIZATION:")
    print(f"   Avg: {stats.get('gpu_util_pct', {}).get('avg', 0):.1f}%  "
          f"Max: {stats.get('gpu_util_pct', {}).get('max', 0):.1f}%")
    
    print(f"\n⚡ CPU UTILIZATION:")
    print(f"   Avg: {stats.get('cpu_util_pct', {}).get('avg', 0):.1f}%  "
          f"Max: {stats.get('cpu_util_pct', {}).get('max', 0):.1f}%")
    
    inference_stats = stats.get('inference_latency_ms', {})
    print(f"\n🎯 INFERENCE LATENCY:")
    print(f"   Avg: {inference_stats.get('avg', 0):.1f}ms  "
          f"p50: {inference_stats.get('p50', 0):.1f}ms  "
          f"p99: {inference_stats.get('p99', 0):.1f}ms")
    
    print(f"\n📊 QUEUE DEPTH:")
    print(f"   Avg: {stats.get('queue_size_avg', 0):.0f}")
    
    print(f"\n✅ DETECTION RATES:")
    print(f"   Entry zone: {accuracy.get_detection_rate('entry_zone'):.1f}%  "
          f"Greet zone: {accuracy.get_detection_rate('greet_zone'):.1f}%")
    
    print("\n" + "="*70)
