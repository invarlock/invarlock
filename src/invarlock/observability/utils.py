"""
Monitoring utilities and helper functions.
"""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import psutil


@dataclass
class TimingContext:
    """Context for timing operations."""

    start_time: float
    end_time: float | None = None
    duration: float | None = None
    operation: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def finish(self):
        """Mark timing as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration


class Timer:
    """Simple timer for measuring operation durations."""

    def __init__(self, name: str = "", auto_log: bool = False):
        self.name = name
        self.auto_log = auto_log
        self.logger = logging.getLogger(__name__)
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.duration: float | None = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        """Stop the timer and return duration."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        if self.auto_log:
            operation = self.name or "operation"
            self.logger.info(f"{operation} completed in {self.duration:.3f}s")

        return self.duration

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextmanager
def timed_operation(
    operation_name: str,
    metadata: dict[str, Any] | None = None,
    callback: Callable[[TimingContext], None] | None = None,
):
    """Context manager for timing operations with callback support."""
    context = TimingContext(
        start_time=time.time(), operation=operation_name, metadata=metadata or {}
    )

    try:
        yield context
    finally:
        context.finish()
        if callback:
            callback(context)


def timing_decorator(
    operation_name: str | None = None,
    auto_log: bool = True,
    callback: Callable[[TimingContext], None] | None = None,
):
    """Decorator for timing function execution."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"

            def timing_callback(context):
                if auto_log:
                    logger = logging.getLogger(__name__)
                    logger.info(f"{name} completed in {context.duration:.3f}s")
                if callback:
                    callback(context)

            with timed_operation(name, callback=timing_callback):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimiter:
    """Simple rate limiter for monitoring operations."""

    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def is_allowed(self) -> bool:
        """Check if operation is allowed within rate limit."""
        now = time.time()

        with self.lock:
            # Remove old calls outside the window
            self.calls = [
                call_time
                for call_time in self.calls
                if now - call_time < self.window_seconds
            ]

            # Check if we're under the limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True

            return False

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()

        with self.lock:
            recent_calls = [
                call_time
                for call_time in self.calls
                if now - call_time < self.window_seconds
            ]

            return {
                "current_calls": len(recent_calls),
                "max_calls": self.max_calls,
                "window_seconds": self.window_seconds,
                "utilization": len(recent_calls) / self.max_calls,
                "next_available": min(recent_calls) + self.window_seconds
                if recent_calls
                else now,
            }


class CircularBuffer:
    """Circular buffer for storing recent metrics."""

    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()

    def append(self, item):
        """Add item to buffer."""
        with self.lock:
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get_all(self) -> list[Any]:
        """Get all items in chronological order."""
        with self.lock:
            if self.count == 0:
                return []

            if self.count < self.size:
                return [item for item in self.buffer[: self.count] if item is not None]
            else:
                return self.buffer[self.head :] + self.buffer[: self.head]

    def get_recent(self, n: int) -> list[Any]:
        """Get n most recent items."""
        all_items = self.get_all()
        return all_items[-n:] if n <= len(all_items) else all_items

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer = [None] * self.size
            self.head = 0
            self.count = 0

    def __len__(self):
        return self.count


class MovingAverage:
    """Calculate moving average of values."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = CircularBuffer(window_size)
        self.sum = 0.0
        self.lock = threading.Lock()

    def add(self, value: float):
        """Add a value to the moving average."""
        with self.lock:
            old_values = self.values.get_all()

            # If buffer is full, subtract the oldest value
            if len(old_values) == self.window_size:
                oldest = old_values[0] if old_values else 0
                self.sum -= oldest

            # Add new value
            self.values.append(value)
            self.sum += value

    def get_average(self) -> float:
        """Get current moving average."""
        with self.lock:
            count = len(self.values)
            return self.sum / count if count > 0 else 0

    def get_stats(self) -> dict[str, float]:
        """Get statistics about the moving average."""
        with self.lock:
            values = self.values.get_all()
            if not values:
                return {"average": 0, "min": 0, "max": 0, "count": 0}

            return {
                "average": self.sum / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }


class PercentileCalculator:
    """Calculate percentiles from a stream of values."""

    def __init__(self, window_size: int = 1000):
        self.values = CircularBuffer(window_size)

    def add(self, value: float):
        """Add a value."""
        self.values.append(value)

    def get_percentile(self, percentile: float) -> float:
        """Get the specified percentile (0-100)."""
        values = self.values.get_all()
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int((percentile / 100) * (len(sorted_values) - 1))
        return float(sorted_values[index])

    def get_percentiles(self, percentiles: list[float]) -> dict[float, float]:
        """Get multiple percentiles at once."""
        values = self.values.get_all()
        if not values:
            return dict.fromkeys(percentiles, 0)

        sorted_values = sorted(values)
        result = {}

        for percentile in percentiles:
            index = int((percentile / 100) * (len(sorted_values) - 1))
            result[percentile] = sorted_values[index]

        return result


def get_system_info() -> dict[str, Any]:
    """Get comprehensive system information."""
    try:
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Get GPU info if available
        gpu_info = {}
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_names": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ],
                    "cuda_version": torch.version.cuda,
                }
            else:
                gpu_info = {"gpu_available": False}
        except ImportError:
            gpu_info = {"gpu_available": False, "torch_available": False}

        return {
            "cpu": {
                "count_physical": cpu_count,
                "count_logical": cpu_count_logical,
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            },
            "gpu": gpu_info,
            "python_version": getattr(psutil, "sys", {}).get("version", "unknown"),
            "platform": getattr(psutil, "os", {}).get("name", "unknown"),
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get system info: {e}")
        return {"error": str(e)}


def format_bytes(bytes_value: int | float) -> str:
    """Format bytes into human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration into human readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculate exponential backoff delay."""
    delay = base_delay * (2**attempt)
    return float(min(delay, max_delay))


class DebounceTimer:
    """Debounce timer for limiting rapid operations."""

    def __init__(self, delay: float):
        self.delay = delay
        self.last_call = 0.0
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Call function with debouncing."""
        with self.lock:
            current_time = time.time()

            # Cancel existing timer
            if self.timer:
                self.timer.cancel()

            # Schedule new call
            time_since_last = current_time - self.last_call
            if time_since_last >= self.delay:
                # Call immediately
                self.last_call = current_time
                func(*args, **kwargs)
            else:
                # Schedule for later
                remaining_delay = self.delay - time_since_last
                self.timer = threading.Timer(
                    remaining_delay, self._delayed_call, args=[func, args, kwargs]
                )
                self.timer.start()

    def _delayed_call(self, func: Callable, args: tuple, kwargs: dict):
        """Execute delayed call."""
        with self.lock:
            self.last_call = time.time()
            self.timer = None  # Clear the timer reference
            func(*args, **kwargs)


class ThresholdMonitor:
    """Monitor values against thresholds with hysteresis."""

    def __init__(self, threshold: float, hysteresis: float = 0.1):
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.triggered = False
        self.last_value: float | None = None
        self.trigger_count = 0
        self.last_trigger_time: float | None = None

    def check(self, value: float) -> bool:
        """Check value against threshold. Returns True if threshold is crossed."""
        self.last_value = value
        current_time = time.time()

        if not self.triggered:
            # Check for threshold breach
            if value > self.threshold:
                self.triggered = True
                self.trigger_count += 1
                self.last_trigger_time = float(current_time)
                return True
        else:
            # Check for recovery (with hysteresis)
            if value < (self.threshold - self.hysteresis):
                self.triggered = False

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get threshold monitor statistics."""
        return {
            "threshold": self.threshold,
            "hysteresis": self.hysteresis,
            "triggered": self.triggered,
            "last_value": self.last_value,
            "trigger_count": self.trigger_count,
            "last_trigger_time": self.last_trigger_time,
        }


# Error handling utilities
class MonitoringError(Exception):
    """Base exception for monitoring operations."""

    pass


class MetricsCollectionError(MonitoringError):
    """Error during metrics collection."""

    pass


class ExportError(MonitoringError):
    """Error during metrics export."""

    pass


class HealthCheckError(MonitoringError):
    """Error during health check."""

    pass


def retry_with_backoff(
    max_attempts: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,)
):
    """Decorator for retrying operations with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = exponential_backoff(attempt, base_delay)
                        time.sleep(delay)
                    else:
                        break

            if last_exception:
                raise last_exception
            raise RuntimeError("Failed after all retry attempts")

        return wrapper

    return decorator


def log_exceptions(
    logger: logging.Logger | None = None,
    level: int = logging.ERROR,
    reraise: bool = True,
):
    """Decorator for logging exceptions."""
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(level, f"Exception in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                return None

        return wrapper

    return decorator
