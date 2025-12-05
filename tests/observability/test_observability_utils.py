"""
Tests for observability utility functions and helper classes.

This module tests:
- TimingContext and Timer classes
- timed_operation context manager
- timing_decorator
- RateLimiter
- CircularBuffer
- MovingAverage
- PercentileCalculator
- ThresholdMonitor
- DebounceTimer
- Utility functions (format_bytes, format_duration, etc.)
- Error handling utilities and decorators
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# =============================================================================
# TimingContext Tests
# =============================================================================


@pytest.mark.unit
class TestTimingContext:
    """Tests for TimingContext dataclass."""

    def test_creation(self):
        """Test creating a timing context."""
        from invarlock.observability.utils import TimingContext

        ctx = TimingContext(
            start_time=1000.0, operation="test_op", metadata={"key": "value"}
        )

        assert ctx.start_time == 1000.0
        assert ctx.operation == "test_op"
        assert ctx.metadata == {"key": "value"}
        assert ctx.end_time is None
        assert ctx.duration is None

    def test_finish(self):
        """Test finishing a timing context."""
        from invarlock.observability.utils import TimingContext

        start = time.time()
        ctx = TimingContext(start_time=start)

        time.sleep(0.1)
        duration = ctx.finish()

        assert ctx.end_time is not None
        assert ctx.duration is not None
        assert duration >= 0.1
        assert ctx.duration == duration

    def test_default_metadata(self):
        """Test default metadata is empty dict."""
        from invarlock.observability.utils import TimingContext

        ctx = TimingContext(start_time=time.time())
        assert ctx.metadata == {}


# =============================================================================
# Timer Tests
# =============================================================================


@pytest.mark.unit
class TestTimer:
    """Tests for Timer class."""

    def test_start_stop(self):
        """Test basic start/stop functionality."""
        from invarlock.observability.utils import Timer

        timer = Timer("test_timer")

        timer.start()
        time.sleep(0.05)
        duration = timer.stop()

        assert duration >= 0.05
        assert timer.duration == duration

    def test_stop_without_start_raises(self):
        """Test stopping timer without starting raises error."""
        from invarlock.observability.utils import Timer

        timer = Timer()

        with pytest.raises(ValueError, match="Timer not started"):
            timer.stop()

    def test_context_manager(self):
        """Test using timer as context manager."""
        from invarlock.observability.utils import Timer

        timer = Timer("test_timer")

        with timer:
            time.sleep(0.05)

        assert timer.duration is not None
        assert timer.duration >= 0.05

    def test_auto_log(self):
        """Test auto-logging on stop."""
        from invarlock.observability.utils import Timer

        timer = Timer("test_op", auto_log=True)

        with timer:
            pass

        # No exception means logging worked


# =============================================================================
# timed_operation Tests
# =============================================================================


@pytest.mark.unit
class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_basic_usage(self):
        """Test basic timed_operation usage."""
        from invarlock.observability.utils import timed_operation

        with timed_operation("test_op") as ctx:
            time.sleep(0.05)

        assert ctx.operation == "test_op"
        assert ctx.duration is not None
        assert ctx.duration >= 0.05

    def test_with_metadata(self):
        """Test timed_operation with metadata."""
        from invarlock.observability.utils import timed_operation

        with timed_operation("test_op", metadata={"key": "value"}) as ctx:
            pass

        assert ctx.metadata == {"key": "value"}

    def test_with_callback(self):
        """Test timed_operation with callback."""
        from invarlock.observability.utils import timed_operation

        callback_called = []

        def callback(ctx):
            callback_called.append(ctx)

        with timed_operation("test_op", callback=callback):
            pass

        assert len(callback_called) == 1
        assert callback_called[0].operation == "test_op"


# =============================================================================
# timing_decorator Tests
# =============================================================================


@pytest.mark.unit
class TestTimingDecorator:
    """Tests for timing_decorator."""

    def test_basic_usage(self):
        """Test basic decorator usage."""
        from invarlock.observability.utils import timing_decorator

        @timing_decorator(auto_log=False)
        def test_function():
            time.sleep(0.05)
            return "result"

        result = test_function()

        assert result == "result"

    def test_custom_operation_name(self):
        """Test decorator with custom operation name."""
        from invarlock.observability.utils import timing_decorator

        contexts = []

        def capture_callback(ctx):
            contexts.append(ctx)

        @timing_decorator(
            operation_name="custom_op", auto_log=False, callback=capture_callback
        )
        def test_function():
            pass

        test_function()

        assert len(contexts) == 1
        assert contexts[0].operation == "custom_op"

    def test_preserves_function_metadata(self):
        """Test decorator preserves function metadata."""
        from invarlock.observability.utils import timing_decorator

        @timing_decorator(auto_log=False)
        def documented_function():
            """This is a docstring."""
            pass

        assert documented_function.__doc__ == "This is a docstring."


# =============================================================================
# RateLimiter Tests
# =============================================================================


@pytest.mark.unit
class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_within_limit(self):
        """Test rate limiter allows calls within limit."""
        from invarlock.observability.utils import RateLimiter

        limiter = RateLimiter(max_calls=5, window_seconds=60)

        for _ in range(5):
            assert limiter.is_allowed() is True

    def test_blocks_over_limit(self):
        """Test rate limiter blocks calls over limit."""
        from invarlock.observability.utils import RateLimiter

        limiter = RateLimiter(max_calls=3, window_seconds=60)

        for _ in range(3):
            assert limiter.is_allowed() is True

        assert limiter.is_allowed() is False

    def test_allows_after_window(self):
        """Test rate limiter allows after window expires."""
        from invarlock.observability.utils import RateLimiter

        limiter = RateLimiter(max_calls=1, window_seconds=0.1)

        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False

        time.sleep(0.15)

        assert limiter.is_allowed() is True

    def test_get_stats(self):
        """Test getting rate limiter statistics."""
        from invarlock.observability.utils import RateLimiter

        limiter = RateLimiter(max_calls=10, window_seconds=60)

        for _ in range(5):
            limiter.is_allowed()

        stats = limiter.get_stats()

        assert stats["current_calls"] == 5
        assert stats["max_calls"] == 10
        assert stats["window_seconds"] == 60
        assert stats["utilization"] == 0.5

    def test_thread_safety(self):
        """Test rate limiter is thread-safe."""
        from invarlock.observability.utils import RateLimiter

        limiter = RateLimiter(max_calls=100, window_seconds=60)
        allowed_count = []

        def try_calls():
            count = 0
            for _ in range(20):
                if limiter.is_allowed():
                    count += 1
            allowed_count.append(count)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_calls) for _ in range(10)]
            for f in futures:
                f.result()

        # Total allowed should be exactly 100
        assert sum(allowed_count) == 100


# =============================================================================
# CircularBuffer Tests
# =============================================================================


@pytest.mark.unit
class TestCircularBuffer:
    """Tests for CircularBuffer."""

    def test_append_and_get(self):
        """Test basic append and retrieval."""
        from invarlock.observability.utils import CircularBuffer

        buffer = CircularBuffer(size=5)

        buffer.append(1)
        buffer.append(2)
        buffer.append(3)

        assert buffer.get_all() == [1, 2, 3]

    def test_overflow(self):
        """Test buffer overflow behavior."""
        from invarlock.observability.utils import CircularBuffer

        buffer = CircularBuffer(size=3)

        for i in range(5):
            buffer.append(i)

        # Should keep last 3 items
        assert buffer.get_all() == [2, 3, 4]

    def test_get_recent(self):
        """Test getting recent items."""
        from invarlock.observability.utils import CircularBuffer

        buffer = CircularBuffer(size=10)

        for i in range(10):
            buffer.append(i)

        recent = buffer.get_recent(3)
        assert recent == [7, 8, 9]

    def test_clear(self):
        """Test clearing buffer."""
        from invarlock.observability.utils import CircularBuffer

        buffer = CircularBuffer(size=5)

        for i in range(5):
            buffer.append(i)

        buffer.clear()

        assert buffer.get_all() == []
        assert len(buffer) == 0

    def test_len(self):
        """Test length tracking."""
        from invarlock.observability.utils import CircularBuffer

        buffer = CircularBuffer(size=10)

        assert len(buffer) == 0

        for i in range(5):
            buffer.append(i)

        assert len(buffer) == 5

        for i in range(10):
            buffer.append(i)

        assert len(buffer) == 10  # Capped at size


# =============================================================================
# MovingAverage Tests
# =============================================================================


@pytest.mark.unit
class TestMovingAverage:
    """Tests for MovingAverage."""

    def test_basic_average(self):
        """Test basic moving average calculation."""
        from invarlock.observability.utils import MovingAverage

        ma = MovingAverage(window_size=5)

        for i in [1, 2, 3, 4, 5]:
            ma.add(float(i))

        assert ma.get_average() == 3.0

    def test_window_slides(self):
        """Test window slides correctly."""
        from invarlock.observability.utils import MovingAverage

        ma = MovingAverage(window_size=3)

        for i in [1, 2, 3]:
            ma.add(float(i))

        assert ma.get_average() == 2.0  # (1+2+3)/3

        ma.add(4)  # Now window is [2, 3, 4]
        assert ma.get_average() == 3.0  # (2+3+4)/3

    def test_empty_average(self):
        """Test average on empty buffer."""
        from invarlock.observability.utils import MovingAverage

        ma = MovingAverage(window_size=5)

        assert ma.get_average() == 0

    def test_get_stats(self):
        """Test getting statistics."""
        from invarlock.observability.utils import MovingAverage

        ma = MovingAverage(window_size=5)

        for i in [1, 2, 3, 4, 5]:
            ma.add(float(i))

        stats = ma.get_stats()

        assert stats["average"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5


# =============================================================================
# PercentileCalculator Tests
# =============================================================================


@pytest.mark.unit
class TestPercentileCalculator:
    """Tests for PercentileCalculator."""

    def test_basic_percentile(self):
        """Test basic percentile calculation."""
        from invarlock.observability.utils import PercentileCalculator

        calc = PercentileCalculator()

        for i in range(1, 101):
            calc.add(float(i))

        assert calc.get_percentile(50) == 50.0
        assert calc.get_percentile(90) == 90.0

    def test_empty_percentile(self):
        """Test percentile on empty calculator."""
        from invarlock.observability.utils import PercentileCalculator

        calc = PercentileCalculator()

        assert calc.get_percentile(50) == 0

    def test_get_multiple_percentiles(self):
        """Test getting multiple percentiles at once."""
        from invarlock.observability.utils import PercentileCalculator

        calc = PercentileCalculator()

        for i in range(1, 101):
            calc.add(float(i))

        percentiles = calc.get_percentiles([50, 90, 95, 99])

        assert percentiles[50] == 50.0
        assert percentiles[90] == 90.0


# =============================================================================
# ThresholdMonitor Tests
# =============================================================================


@pytest.mark.unit
class TestThresholdMonitor:
    """Tests for ThresholdMonitor."""

    def test_triggers_on_threshold_breach(self):
        """Test threshold triggers when breached."""
        from invarlock.observability.utils import ThresholdMonitor

        monitor = ThresholdMonitor(threshold=80.0)

        assert monitor.check(50.0) is False
        assert monitor.check(90.0) is True
        assert monitor.triggered is True

    def test_hysteresis(self):
        """Test hysteresis prevents rapid toggling."""
        from invarlock.observability.utils import ThresholdMonitor

        monitor = ThresholdMonitor(threshold=80.0, hysteresis=10.0)

        monitor.check(85.0)  # Trigger
        assert monitor.triggered is True

        monitor.check(75.0)  # Still above hysteresis threshold
        assert monitor.triggered is True

        monitor.check(65.0)  # Below hysteresis threshold
        assert monitor.triggered is False

    def test_no_duplicate_triggers(self):
        """Test already-triggered threshold doesn't re-trigger."""
        from invarlock.observability.utils import ThresholdMonitor

        monitor = ThresholdMonitor(threshold=80.0)

        assert monitor.check(90.0) is True  # First trigger
        assert monitor.check(95.0) is False  # Already triggered

    def test_get_stats(self):
        """Test getting monitor statistics."""
        from invarlock.observability.utils import ThresholdMonitor

        monitor = ThresholdMonitor(threshold=80.0, hysteresis=5.0)

        monitor.check(50.0)
        monitor.check(90.0)

        stats = monitor.get_stats()

        assert stats["threshold"] == 80.0
        assert stats["hysteresis"] == 5.0
        assert stats["triggered"] is True
        assert stats["last_value"] == 90.0
        assert stats["trigger_count"] == 1


# =============================================================================
# DebounceTimer Tests
# =============================================================================


@pytest.mark.unit
class TestDebounceTimer:
    """Tests for DebounceTimer."""

    def test_immediate_call(self):
        """Test first call executes immediately."""
        from invarlock.observability.utils import DebounceTimer

        debounce = DebounceTimer(delay=0.5)
        call_count = [0]

        def callback():
            call_count[0] += 1

        debounce.call(callback)

        assert call_count[0] == 1

    def test_debounces_rapid_calls(self):
        """Test rapid calls are debounced."""
        from invarlock.observability.utils import DebounceTimer

        debounce = DebounceTimer(delay=0.2)
        call_count = [0]

        def callback():
            call_count[0] += 1

        debounce.call(callback)  # Immediate
        debounce.call(callback)  # Debounced
        debounce.call(callback)  # Debounced

        time.sleep(0.01)  # Short wait
        assert call_count[0] == 1  # Only first call executed

    def test_delayed_call_executes(self):
        """Test delayed call eventually executes."""
        from invarlock.observability.utils import DebounceTimer

        debounce = DebounceTimer(delay=0.1)
        call_count = [0]

        def callback():
            call_count[0] += 1

        debounce.call(callback)  # Immediate
        debounce.call(callback)  # Scheduled for later

        time.sleep(0.15)  # Wait for debounce delay

        assert call_count[0] == 2


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.unit
class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_format_bytes(self):
        """Test byte formatting."""
        from invarlock.observability.utils import format_bytes

        assert format_bytes(500) == "500.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_format_duration(self):
        """Test duration formatting."""
        from invarlock.observability.utils import format_duration

        assert format_duration(30.5) == "30.50s"
        assert format_duration(90) == "1.5m"
        assert format_duration(3600) == "1.0h"
        assert format_duration(86400) == "1.0d"

    def test_safe_divide(self):
        """Test safe division."""
        from invarlock.observability.utils import safe_divide

        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, default=-1) == -1

    def test_clamp(self):
        """Test value clamping."""
        from invarlock.observability.utils import clamp

        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        from invarlock.observability.utils import exponential_backoff

        assert exponential_backoff(0, base_delay=1.0) == 1.0
        assert exponential_backoff(1, base_delay=1.0) == 2.0
        assert exponential_backoff(2, base_delay=1.0) == 4.0
        assert exponential_backoff(10, base_delay=1.0, max_delay=60.0) == 60.0


# =============================================================================
# Error Handling Utilities Tests
# =============================================================================


@pytest.mark.unit
class TestErrorHandlingUtilities:
    """Tests for error handling utilities."""

    def test_monitoring_error_hierarchy(self):
        """Test monitoring error hierarchy."""
        from invarlock.observability.utils import (
            ExportError,
            HealthCheckError,
            MetricsCollectionError,
            MonitoringError,
        )

        assert issubclass(MetricsCollectionError, MonitoringError)
        assert issubclass(ExportError, MonitoringError)
        assert issubclass(HealthCheckError, MonitoringError)

    def test_retry_with_backoff_success(self):
        """Test retry decorator with successful call."""
        from invarlock.observability.utils import retry_with_backoff

        @retry_with_backoff(max_attempts=3)
        def succeeds():
            return "success"

        assert succeeds() == "success"

    def test_retry_with_backoff_eventual_success(self):
        """Test retry decorator with eventual success."""
        from invarlock.observability.utils import retry_with_backoff

        call_count = [0]

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("Temporary failure")
            return "success"

        assert fails_twice() == "success"
        assert call_count[0] == 3

    def test_retry_with_backoff_exhausted(self):
        """Test retry decorator exhausts attempts."""
        from invarlock.observability.utils import retry_with_backoff

        @retry_with_backoff(max_attempts=2, base_delay=0.01)
        def always_fails():
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError, match="Always fails"):
            always_fails()

    def test_log_exceptions_reraises(self):
        """Test log_exceptions decorator re-raises by default."""
        from invarlock.observability.utils import log_exceptions

        @log_exceptions()
        def raises_error():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            raises_error()

    def test_log_exceptions_no_reraise(self):
        """Test log_exceptions decorator can suppress re-raise."""
        from invarlock.observability.utils import log_exceptions

        @log_exceptions(reraise=False)
        def raises_error():
            raise ValueError("Test error")

        result = raises_error()
        assert result is None


# =============================================================================
# get_system_info Tests
# =============================================================================


@pytest.mark.unit
class TestGetSystemInfo:
    """Tests for get_system_info function."""

    def test_returns_dict(self):
        """Test function returns a dict."""
        from invarlock.observability.utils import get_system_info

        info = get_system_info()

        assert isinstance(info, dict)
        # On some systems (e.g., macOS), psutil may fail with certain sysctls
        # In that case, we get an error dict back which is still valid behavior
        if "error" not in info:
            assert "cpu" in info
            assert "memory" in info
            assert "disk" in info
            assert "gpu" in info

    def test_cpu_info(self):
        """Test CPU info is populated when available."""
        from invarlock.observability.utils import get_system_info

        info = get_system_info()

        # Skip if psutil failed on this system
        if "error" in info:
            pytest.skip("psutil failed to gather system info on this platform")

        assert "count_physical" in info["cpu"]
        assert "count_logical" in info["cpu"]

    def test_memory_info(self):
        """Test memory info is populated when available."""
        from invarlock.observability.utils import get_system_info

        info = get_system_info()

        # Skip if psutil failed on this system
        if "error" in info:
            pytest.skip("psutil failed to gather system info on this platform")

        assert "total" in info["memory"]
        assert "available" in info["memory"]
        assert "percent" in info["memory"]
