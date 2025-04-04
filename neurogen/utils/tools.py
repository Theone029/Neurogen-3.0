import time
import hashlib
import inspect
import json
import logging
import os
import sys
import uuid
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from collections import deque
import threading
import functools

# ==========================================
# Hashing and Identifiers
# ==========================================

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a short, unique identifier.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random portion
        
    Returns:
        Generated ID
    """
    random_part = uuid.uuid4().hex[:length]
    return f"{prefix}{random_part}" if prefix else random_part

def hash_content(content: Any, algorithm: str = "sha256") -> str:
    """
    Create a hash of any content.
    
    Args:
        content: Content to hash
        algorithm: Hash algorithm to use (md5, sha1, sha256)
        
    Returns:
        Content hash
    """
    hash_func = getattr(hashlib, algorithm)()
    
    # Handle different content types
    if isinstance(content, str):
        hash_func.update(content.encode('utf-8'))
    elif isinstance(content, bytes):
        hash_func.update(content)
    elif isinstance(content, (int, float, bool)):
        hash_func.update(str(content).encode('utf-8'))
    elif isinstance(content, (list, tuple)):
        for item in content:
            hash_func.update(hash_content(item, algorithm).encode('utf-8'))
    elif isinstance(content, dict):
        # Sort keys for consistent hashing
        for key in sorted(content.keys()):
            hash_func.update(hash_content(key, algorithm).encode('utf-8'))
            hash_func.update(hash_content(content[key], algorithm).encode('utf-8'))
    else:
        # Fall back to string representation
        hash_func.update(str(content).encode('utf-8'))
        
    return hash_func.hexdigest()

def compare_hashes(hash1: str, hash2: str, tolerance: float = 0.0) -> bool:
    """
    Compare two hashes with optional tolerance for similarity.
    
    Args:
        hash1: First hash
        hash2: Second hash
        tolerance: Similarity tolerance (0.0-1.0)
        
    Returns:
        True if hashes match within tolerance
    """
    if not hash1 or not hash2:
        return False
        
    if tolerance <= 0:
        # Exact match required
        return hash1 == hash2
    else:
        # Calculate similarity
        min_len = min(len(hash1), len(hash2))
        if min_len == 0:
            return False
            
        matching_chars = sum(1 for i in range(min_len) if hash1[i] == hash2[i])
        similarity = matching_chars / min_len
        
        return similarity >= (1 - tolerance)

# ==========================================
# Timing and Performance
# ==========================================

class Timer:
    """High-precision timer for performance analysis."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self.elapsed = 0
        self.is_running = False
        self.laps = []
    
    def start(self) -> float:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.is_running = True
        return self.start_time
        
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if not self.is_running:
            return self.elapsed
            
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.is_running = False
        return self.elapsed
        
    def lap(self, label: str = "") -> float:
        """Record a lap time."""
        if not self.is_running:
            return 0
            
        current = time.perf_counter()
        lap_time = current - self.start_time
        
        if not self.laps:
            # First lap
            interval = lap_time
        else:
            # Interval since last lap
            interval = lap_time - self.laps[-1]["time"]
            
        self.laps.append({
            "label": label,
            "time": lap_time,
            "interval": interval
        })
        
        return lap_time
        
    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = 0
        self.end_time = 0
        self.elapsed = 0
        self.is_running = False
        self.laps = []
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()

def timing_decorator(func):
    """Decorator for timing function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(func.__name__)
        timer.start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        logging.debug(f"Function '{func.__name__}' took {elapsed:.6f} seconds")
        return result
    return wrapper

class PerformanceTracker:
    """
    Tracks performance metrics across multiple function calls.
    """
    
    def __init__(self, window_size: int = 100):
        self.stats = {}
        self.window_size = window_size
        self.lock = threading.RLock()
        
    def track(self, name: str, value: float) -> Dict[str, float]:
        """
        Track a performance metric.
        
        Args:
            name: Metric name
            value: Metric value (typically time in seconds)
            
        Returns:
            Current statistics for this metric
        """
        with self.lock:
            if name not in self.stats:
                self.stats[name] = {
                    "values": deque(maxlen=self.window_size),
                    "count": 0,
                    "sum": 0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
                
            stat = self.stats[name]
            stat["values"].append(value)
            stat["count"] += 1
            stat["sum"] += value
            stat["min"] = min(stat["min"], value)
            stat["max"] = max(stat["max"], value)
            
            # Calculate current stats
            values = list(stat["values"])
            return {
                "mean": stat["sum"] / stat["count"],
                "median": sorted(values)[len(values) // 2] if values else 0,
                "min": stat["min"],
                "max": stat["max"],
                "count": stat["count"],
                "recent_mean": sum(values) / len(values) if values else 0
            }
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Args:
            name: Optional metric name to get specific stats
            
        Returns:
            Current statistics
        """
        with self.lock:
            if name:
                if name not in self.stats:
                    return {}
                    
                stat = self.stats[name]
                values = list(stat["values"])
                
                return {
                    "mean": stat["sum"] / stat["count"] if stat["count"] > 0 else 0,
                    "median": sorted(values)[len(values) // 2] if values
