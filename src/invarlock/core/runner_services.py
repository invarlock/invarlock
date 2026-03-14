from __future__ import annotations

import time
from typing import Any

from invarlock.observability.metrics import capture_memory_snapshot

from .checkpoint import CheckpointManager
from .events import EventLogger


def initialize_services(
    runner: Any,
    config: Any,
    *,
    event_logger_factory: Any = EventLogger,
    checkpoint_factory: Any = CheckpointManager,
) -> None:
    """Initialize event logging and checkpoint services."""
    if config.event_path:
        run_id = None
        if isinstance(config.context, dict):
            run_id = config.context.get("run_id")
        runner.event_logger = event_logger_factory(config.event_path, run_id=run_id)

    if config.checkpoint_interval > 0:
        runner.checkpoint_manager = checkpoint_factory()


def cleanup_services(runner: Any) -> None:
    """Clean up event logging and checkpoint services."""
    if runner.event_logger:
        runner.event_logger.close()
        runner.event_logger = None

    if runner.checkpoint_manager:
        runner.checkpoint_manager.cleanup()
        runner.checkpoint_manager = None


def record_timing(
    timings: dict[str, float],
    key: str,
    start: float,
    *,
    perf_counter: Any = time.perf_counter,
) -> None:
    timings[key] = max(0.0, float(perf_counter() - start))


def capture_memory(
    memory_snapshots: list[dict[str, Any]],
    phase: str,
    *,
    capture_fn: Any = capture_memory_snapshot,
) -> None:
    snapshot = capture_fn(phase)
    if snapshot:
        memory_snapshots.append(snapshot)


__all__ = [
    "capture_memory",
    "cleanup_services",
    "initialize_services",
    "record_timing",
]
