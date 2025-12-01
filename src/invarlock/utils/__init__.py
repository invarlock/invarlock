"""
InvarLock Utilities
===============

Common utility functions used across InvarLock modules.

This package also exposes submodules such as `invarlock.utils.digest` for
hashing and provenance utilities.
"""

from __future__ import annotations

from typing import Any

import psutil

try:  # Torch is optional; utils can be imported without it.
    import torch  # type: ignore[import]
except Exception:  # pragma: no cover - exercised when torch is missing
    torch = None  # type: ignore[assignment]


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
    if isinstance(batch, torch.Tensor):
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
                if isinstance(value, torch.Tensor):
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
        input_ids = torch.tensor(batch)

    # Move to device if specified
    if device is not None:
        input_ids = input_ids.to(device)

    return input_ids


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model."""
    return next(model.parameters()).device


def ensure_tensor(data: Any, device: torch.device | None = None) -> torch.Tensor:
    """Ensure data is a tensor on the correct device."""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

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
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
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
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    except Exception:
        # If torch is unavailable or querying CUDA fails, fall back to CPU-only stats.
        pass

    return result


__all__ = [
    "extract_input_ids",
    "get_model_device",
    "ensure_tensor",
    "safe_divide",
    "dict_to_device",
    "format_number",
    "get_memory_usage",
]
