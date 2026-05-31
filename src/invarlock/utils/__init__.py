"""Common utility functions used across InvarLock modules."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import psutil

if TYPE_CHECKING:
    import torch

_ENC = "utf-8"
_TORCH_UNSET = object()
_torch: Any = _TORCH_UNSET
_TORCH_CUDA_QUERY_ERRORS = (
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_FAST_MEAN_STATISTICS = {np.mean, np.nanmean}


class _HashLike(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self, /) -> str: ...


def _get_torch() -> Any | None:
    global _torch
    if _torch is _TORCH_UNSET:
        try:  # pragma: no cover - exercised when torch is missing
            _torch = importlib.import_module("torch")
        except (
            ModuleNotFoundError
        ):  # pragma: no cover - exercised when torch is missing
            _torch = None
    return None if _torch is _TORCH_UNSET else _torch


def _require_torch() -> Any:
    torch_mod = _get_torch()
    if torch_mod is None:
        raise ModuleNotFoundError(
            "torch is required for invarlock.utils tensor helpers"
        )
    return torch_mod


def _h() -> _HashLike:
    return hashlib.blake2s(digest_size=32)


def hash_bytes(b: bytes, *, salt: bytes | None = None) -> str:
    h = _h()
    if salt:
        h.update(salt)
    h.update(b)
    return h.hexdigest()


def hash_json(obj: Any, *, salt: str | None = None) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hash_bytes(s.encode(_ENC), salt=salt.encode(_ENC) if salt else None)


def hash_int_array(arr: Any, *, salt: str | None = None) -> str:
    a = np.asarray(arr, dtype=np.int32, order="C")
    return hash_bytes(a.tobytes(order="C"), salt=salt.encode(_ENC) if salt else None)


def _is_mean_statistic(statistic: Callable[[Any], Any] | None) -> bool:
    if statistic is None or statistic in _FAST_MEAN_STATISTICS:
        return True
    name = getattr(statistic, "__name__", "")
    return bool(name in {"mean", "nanmean"})


def bootstrap_mean_statistics(
    data: np.ndarray,
    *,
    n_bootstrap: int,
    random_state: np.random.Generator,
    max_resample_elements: int = 1_000_000,
) -> np.ndarray:
    """Return bootstrap resample means for a 1D array using chunked vectorization."""
    if data.ndim != 1:
        raise ValueError("bootstrap_mean_statistics requires 1D input")
    if n_bootstrap <= 0:
        return np.empty(0, dtype=float)

    data = np.asarray(data, dtype=float)
    sample_size = int(data.size)
    if sample_size <= 0:
        return np.empty(0, dtype=float)

    chunk_rows = max(1, int(max_resample_elements) // max(sample_size, 1))
    chunk_rows = min(chunk_rows, int(n_bootstrap))

    stats = np.empty(int(n_bootstrap), dtype=float)
    for start in range(0, int(n_bootstrap), chunk_rows):
        stop = min(start + chunk_rows, int(n_bootstrap))
        indices = random_state.integers(
            0, sample_size, size=(stop - start, sample_size)
        )
        stats[start:stop] = data[indices].mean(axis=1, dtype=float)
    return stats


def bootstrap_statistics(
    data: np.ndarray,
    *,
    n_bootstrap: int,
    random_state: np.random.Generator,
    statistic: Callable[[Any], Any] | None = None,
) -> np.ndarray:
    """Return bootstrap statistics for a 1D array with a fast path for sample means."""
    data = np.asarray(data, dtype=float)
    if _is_mean_statistic(statistic):
        return bootstrap_mean_statistics(
            data,
            n_bootstrap=int(n_bootstrap),
            random_state=random_state,
        )

    stats = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sample_idx = random_state.integers(0, data.size, size=data.size)
        stats[index] = float(cast(Callable[[Any], Any], statistic)(data[sample_idx]))
    return stats


def percentile_interval_from_statistics(
    statistics: np.ndarray, *, alpha: float
) -> tuple[float, float]:
    """Return the two-sided percentile interval for bootstrap statistics."""
    lower = float(np.percentile(statistics, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(statistics, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def extract_input_ids(
    batch: Any, device: str | None = None, strict: bool = False
) -> torch.Tensor:
    """
    Extract input_ids from various batch formats.

    Args:
        batch: Input batch (tensor, dict, or other format)
        device: Target device for tensor
        strict: Whether to raise errors on format issues

    Returns:
        Extracted input_ids tensor
    """
    torch_mod = _require_torch()

    if isinstance(batch, torch_mod.Tensor):
        input_ids = batch
    elif isinstance(batch, dict):
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        elif "inputs" in batch:
            input_ids = batch["inputs"]
        else:
            if strict:
                raise ValueError(
                    f"Dict batch missing 'input_ids' or 'inputs' keys: {list(batch.keys())}"
                )
            # Try first tensor value
            for value in batch.values():
                if isinstance(value, torch_mod.Tensor):
                    input_ids = value
                    break
            else:
                raise ValueError("No tensor found in batch dict")
    elif hasattr(batch, "input_ids"):
        input_ids = batch.input_ids
    else:
        if strict:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        # Try to convert directly
        input_ids = torch_mod.tensor(batch)

    # Move to device if specified
    if device is not None:
        input_ids = input_ids.to(device)

    return input_ids


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model."""
    _require_torch()
    return next(model.parameters()).device


def ensure_tensor(data: Any, device: torch.device | None = None) -> torch.Tensor:
    """Ensure data is a tensor on the correct device."""
    torch_mod = _require_torch()
    if not isinstance(data, torch_mod.Tensor):
        data = torch_mod.tensor(data)

    if device is not None:
        data = data.to(device)

    return data


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def dict_to_device(
    data: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move all tensors in a dictionary to the specified device."""
    torch_mod = _require_torch()
    return {
        key: value.to(device) if isinstance(value, torch_mod.Tensor) else value
        for key, value in data.items()
    }


def format_number(num: float, precision: int = 3) -> str:
    """Format a number for display."""
    if abs(num) < 1e-3:
        return f"{num:.2e}"
    elif abs(num) < 1:
        return f"{num:.{precision + 1}f}"
    else:
        return f"{num:.{precision}f}"


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage in MB."""
    import gc

    # Force garbage collection
    gc.collect()

    # Get process memory
    process = psutil.Process()
    memory_info = process.memory_info()

    result = {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }

    # Add CUDA memory if available
    torch_mod = _get_torch()
    try:
        if (
            torch_mod is not None
            and hasattr(torch_mod, "cuda")
            and torch_mod.cuda.is_available()
        ):
            result["cuda_allocated_mb"] = (
                torch_mod.cuda.memory_allocated() / 1024 / 1024
            )
            result["cuda_reserved_mb"] = torch_mod.cuda.memory_reserved() / 1024 / 1024
    except _TORCH_CUDA_QUERY_ERRORS:
        # If torch is unavailable or querying CUDA fails, fall back to CPU-only stats.
        pass

    return result


__all__ = [
    "bootstrap_mean_statistics",
    "bootstrap_statistics",
    "percentile_interval_from_statistics",
    "hash_bytes",
    "hash_json",
    "hash_int_array",
    "extract_input_ids",
    "get_model_device",
    "ensure_tensor",
    "safe_divide",
    "dict_to_device",
    "format_number",
    "get_memory_usage",
]
