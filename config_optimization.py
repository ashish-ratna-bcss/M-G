#!/usr/bin/env python3
"""
PRODUCTION OPTIMIZATION CONFIG
Tuning parameters for industrial-grade resource utilization
Auto-scales batch size based on GPU memory
"""

import torch
import logging

# ===================== STREAM-AWARE RESOURCE ALLOCATION =====================

class StreamAwareResourceAllocator:
    """
    Dynamic resource allocation based on active stream count
    
    Automatically scales:
    - Batch size (down as streams increase)
    - Queue depth (up as streams increase)
    - Cache size (up to handle more unique frames)
    - Frame skip ratios (up to reduce load)
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ResourceAllocator")
    
    def compute_batch_size_for_streams(self, num_streams: int, device: str, 
                                       base_batch_size: int = 4) -> int:
        """
        Compute optimal batch size based on number of active streams
        
        Logic:
        - More streams = smaller batches (favor throughput over latency)
        - 1-2 streams: batch_size = base
        - 3-5 streams: batch_size = base × 0.75
        - 6+ streams: batch_size = base × 0.5
        """
        if device == "cpu":
            return 1
        
        if num_streams <= 2:
            return base_batch_size
        elif num_streams <= 5:
            return max(2, int(base_batch_size * 0.75))
        else:
            return max(1, int(base_batch_size * 0.5))
    
    def compute_queue_size_for_streams(self, num_streams: int, 
                                       base_queue_size: int = 64) -> int:
        """
        Compute optimal queue size based on stream count
        
        More streams = larger queue (buffer more frames while processing)
        """
        if num_streams <= 2:
            return base_queue_size
        elif num_streams <= 5:
            return int(base_queue_size * 1.5)  # +50%
        else:
            return int(base_queue_size * 2.0)  # +100%
    
    def compute_cache_size_for_streams(self, num_streams: int, 
                                       base_cache_size: int = 1000) -> int:
        """
        Compute optimal cache size based on stream count
        
        More streams = larger cache (more unique frames in parallel)
        """
        if num_streams <= 2:
            return base_cache_size
        elif num_streams <= 5:
            return int(base_cache_size * 1.5)  # +50%
        else:
            return int(base_cache_size * 2.0)  # +100%
    
    def get_stream_aware_config(self, device: str, num_streams: int) -> dict:
        """
        Generate complete stream-aware resource configuration
        """
        base_batch = 4 if device == "cuda" else 1
        
        config = {
            "num_streams": num_streams,
            "device": device,
            "batch_size": self.compute_batch_size_for_streams(num_streams, device, base_batch),
            "queue_size": self.compute_queue_size_for_streams(num_streams, 64),
            "cache_size": self.compute_cache_size_for_streams(num_streams, 1000),
            "frame_skip_scale": 1.0 if num_streams <= 2 else (1.5 if num_streams <= 5 else 2.0),
        }
        
        self.logger.info(f"Stream-aware config: {num_streams} streams → "
                        f"batch={config['batch_size']}, "
                        f"queue={config['queue_size']}, "
                        f"cache={config['cache_size']}")
        
        return config

# ===================== GPU BATCH OPTIMIZATION =====================

def get_optimal_batch_size(device: str, model_size_mb: float = 500,
                           target_gpu_util_pct: float = 70) -> int:
    """
    Auto-compute optimal batch size based on GPU memory
    
    Args:
        device: "cuda" or "cpu"
        model_size_mb: model weights in MB (YOLOv8-S ~100MB, Med ~200MB, L ~500MB)
        target_gpu_util_pct: target GPU utilization (70% is safe for production)
    
    Returns:
        optimal batch size (4, 6, or 8)
    """
    logger = logging.getLogger("OptConfig")
    
    if device == "cpu":
        logger.info("CPU device: using batch_size=1 (no parallel inference)")
        return 1
    
    try:
        gpu_mem_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem_reserved_mb = 500  # Reserve for OS/other
        gpu_mem_available_mb = (gpu_mem_total_gb * 1024) - gpu_mem_reserved_mb
        
        # Overhead per inference batch
        inference_mem_per_frame_mb = 50  # Activation buffers
        
        # Calculate batch size to hit target utilization
        model_allocation_mb = model_size_mb + (model_size_mb * 0.2)  # +20% headroom
        available_for_batch_mb = gpu_mem_available_mb - model_allocation_mb
        
        batch_size = max(1, int(available_for_batch_mb / inference_mem_per_frame_mb))
        
        # Cap at reasonable max
        batch_size = min(batch_size, 32)
        
        # Prefer powers of 2
        if batch_size >= 8:
            batch_size = 8
        elif batch_size >= 6:
            batch_size = 6
        elif batch_size >= 4:
            batch_size = 4
        else:
            batch_size = 1
        
        logger.info(f"GPU Memory: {gpu_mem_total_gb:.1f}GB total, "
                   f"Available: {available_for_batch_mb:.0f}MB → "
                   f"Optimal batch_size: {batch_size}")
        
        return batch_size
        
    except Exception as e:
        logger.warning(f"Could not determine batch size: {e}, defaulting to 4")
        return 4

# ===================== PRODUCTION TUNING PROFILE =====================

class ProductionOptimizationConfig:
    """
    Production-grade tuning parameters
    Choose profile based on your deployment scenario
    """
    
    # Profile: LATENCY (< 10ms per inference, max accuracy)
    LATENCY_OPTIMIZED = {
        "inference": {
            "batch_size": 1,
            "enable_batching": False,
            "enable_caching": False,
            "enable_frame_skipping": False,
        },
        "queue": {
            "max_size": 128,
            "max_wait_ms": 10,  # Process quickly
        },
        "accuracy": {
            "conf_threshold": 0.35,  # Strict
            "nms_iou": 0.45,
        },
        "use_case": "Real-time alerts, low tolerance for missing events",
    }
    
    # Profile: THROUGHPUT (max fps, balanced accuracy)
    THROUGHPUT_OPTIMIZED = {
        "inference": {
            "batch_size": 6,
            "enable_batching": True,
            "enable_caching": True,
            "enable_frame_skipping": True,
        },
        "queue": {
            "max_size": 64,
            "max_wait_ms": 50,  # Batch up requests
        },
        "accuracy": {
            "conf_threshold": 0.30,
            "nms_iou": 0.50,
        },
        "use_case": "Typical production (5 cameras, 25fps each)",
    }
    
    # Profile: RESOURCE_CONSTRAINED (CPU-only, many cameras)
    RESOURCE_CONSTRAINED = {
        "inference": {
            "batch_size": 1,
            "enable_batching": False,
            "enable_caching": True,  # Cache is important for CPU
            "enable_frame_skipping": True,  # Skip 80% of frames in idle
        },
        "queue": {
            "max_size": 32,
            "max_wait_ms": 100,  # More aggressive batching
        },
        "accuracy": {
            "conf_threshold": 0.25,  # More lenient
            "nms_iou": 0.60,
        },
        "use_case": "10+ cameras on CPU, prioritize skip/cache",
    }
    
    # Profile: ACCURACY_FIRST (Multi-GPU, cost no object)
    ACCURACY_FIRST = {
        "inference": {
            "batch_size": 4,
            "enable_batching": True,
            "enable_caching": False,
            "enable_frame_skipping": False,  # Process every frame
        },
        "queue": {
            "max_size": 256,
            "max_wait_ms": 20,
        },
        "accuracy": {
            "conf_threshold": 0.40,  # Highest threshold (fewer false positives)
            "nms_iou": 0.40,         # Stricter NMS
        },
        "use_case": "High-security, no false negatives acceptable, multi-GPU",
    }
    
    @staticmethod
    def get_profile(device: str, num_cameras: int) -> dict:
        """
        Auto-select profile based on deployment
        
        Args:
            device: "cuda" or "cpu"
            num_cameras: number of streams
        
        Returns:
            config dict
        """
        logger = logging.getLogger("ProductionConfig")
        
        if device == "cuda":
            if num_cameras <= 5:
                profile = ProductionOptimizationConfig.THROUGHPUT_OPTIMIZED
                logger.info(f"Selected: THROUGHPUT_OPTIMIZED (GPU, {num_cameras} cameras)")
            else:
                profile = ProductionOptimizationConfig.ACCURACY_FIRST
                logger.info(f"Selected: ACCURACY_FIRST (GPU, {num_cameras} cameras)")
        else:  # CPU
            profile = ProductionOptimizationConfig.RESOURCE_CONSTRAINED
            logger.info(f"Selected: RESOURCE_CONSTRAINED (CPU, {num_cameras} cameras)")
        
        return profile

# ===================== FRAME SKIPPING STRATEGIES =====================

FRAME_SKIP_STRATEGY = {
    "idle": {
        "skip_every_n_frames": 5,    # 5fps instead of 25fps (80% savings)
        "reason": "No active session"
    },
    "entry_detecting": {
        "skip_every_n_frames": 1,    # No skip, 100% processing
        "reason": "Critical: entry zone detection"
    },
    "greet_confirming": {
        "skip_every_n_frames": 1,    # No skip, 100% processing
        "reason": "Critical: greet zone confirmation"
    },
    "greet_counting": {
        "skip_every_n_frames": 2,    # 50% processing (can afford to skip)
        "reason": "Active session, some tolerance"
    },
}

# ===================== QUEUE PARAMETERS =====================

QUEUE_CONFIG = {
    "max_queue_size": 64,        # Drop old requests if exceeded
    "max_wait_before_batch_ms": 50,  # Max wait for batch to fill
    "batch_increment": 2,         # Gradually increase batch as queue grows
    "batch_min": 1,
    "batch_max": 32,
}

# ===================== CACHE PARAMETERS =====================

CACHE_CONFIG = {
    "enabled": True,
    "max_frames": 1000,           # LRU cache size
    "max_age_ms": 100,            # 100ms cache validity (2-3 frames at 25fps)
    "hash_method": "fast",        # "fast" or "strict" (SHA256)
    "expected_hit_rate_pct": 25,  # Typical hit rate during static scenes
}

# ===================== MEMORY LIMITS =====================

MEMORY_CONFIG = {
    "gpu_memory_limit_gb": 8,     # Leave headroom on VRAM
    "model_weights_mb": 500,      # ~500MB for YOLOv8-L
    "activation_buffer_per_frame_mb": 50,
    "cache_size_limit_mb": 512,   # ~1000 frames cached
}

# ===================== ACCURACY TUNING =====================

ACCURACY_CONFIG = {
    "conf_threshold": 0.30,       # YOLO confidence (lower = more detections)
    "nms_iou": 0.50,              # Non-max suppression IoU threshold
    "zone_debounce_frames": 2,    # Avoid false positives on flickering detections
    "min_person_width": 40,       # Reject tiny boxes (likely false+)
    "min_person_height": 100,     # Reject tall/thin boxes
}

# ===================== LATENCY TARGETS (SLO) =====================

LATENCY_SLO = {
    "p50_inference_ms": 8,        # Median inference time
    "p99_inference_ms": 15,       # 99th percentile acceptable
    "p99_queue_wait_ms": 5,       # Queue wait time
    "max_e2e_latency_ms": 100,    # End-to-end (capture to detection)
}

# ===================== HELPER: Get auto-tuned config =====================

def get_production_config(device: str, num_cameras: int, gpu_model: str = "unknown"):
    """
    Generate production config based on deployment
    
    Returns:
        Combined config dict ready for inference pipeline
    """
    profile = ProductionOptimizationConfig.get_profile(device, num_cameras)
    batch_size = get_optimal_batch_size(device) if device == "cuda" else 1
    
    config = {
        **profile,
        "device": device,
        "num_cameras": num_cameras,
        "batch_size": batch_size,
        "queue_config": QUEUE_CONFIG,
        "cache_config": CACHE_CONFIG,
        "memory_config": MEMORY_CONFIG,
        "accuracy_config": ACCURACY_CONFIG,
        "latency_slo": LATENCY_SLO,
    }
    
    logger = logging.getLogger("ProdConfig")
    logger.info(f"Production config: {num_cameras} cameras, device={device}, batch_size={batch_size}")
    
    return config
