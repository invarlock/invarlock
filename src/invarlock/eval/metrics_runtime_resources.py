"""Runtime metric resource helpers."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from invarlock.core.exceptions import ValidationError


class _MemoryProcess(Protocol):
    def memory_info(self) -> Any: ...


def latency_validation_error(
    reason: str, details: dict[str, object]
) -> ValidationError:
    return ValidationError(
        code="E402",
        message="METRICS-VALIDATION-FAILED",
        details={"reason": reason, **details},
    )


def memory_validation_error(reason: str, details: dict[str, object]) -> ValidationError:
    return ValidationError(
        code="E402",
        message="METRICS-VALIDATION-FAILED",
        details={"reason": reason, **details},
    )


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def memory_measurement_baseline(
    device: torch.device,
) -> tuple[float, _MemoryProcess | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        return baseline_memory, None

    import psutil

    process = psutil.Process()
    baseline_memory = float(process.memory_info().rss) / (1024 * 1024)
    return baseline_memory, process


def current_memory_mb(device: torch.device, process: _MemoryProcess | None) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 * 1024)
    assert process is not None
    return float(process.memory_info().rss) / (1024 * 1024)


def cleanup_memory_measurement_failure(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


__all__ = [
    "cleanup_memory_measurement_failure",
    "current_memory_mb",
    "latency_validation_error",
    "maybe_cuda_synchronize",
    "memory_measurement_baseline",
    "memory_validation_error",
]
