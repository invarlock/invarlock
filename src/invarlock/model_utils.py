"""
InvarLock Model Utilities
=====================

Model-specific utility functions for InvarLock.
Contains device handling, seeding, and model manipulation helpers.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # utils may be used without torch
    TORCH_AVAILABLE = False

__all__ = [
    "set_seed",
    "get_device",
    "time_block",
    "json_save",
    "json_load",
    "dump_df",
    "deterministic",
    "extract_input_ids",
]


# ------------------------------------------------------------------#
# Reproducibility
# ------------------------------------------------------------------#


def set_seed(seed: int = 42):
    """Deterministic seeds for Python, NumPy and Torch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------#
# Device helper
# ------------------------------------------------------------------#


def get_device(prefer_gpu: bool = True):
    """Return cuda → mps → cpu in that order of preference."""
    if prefer_gpu and TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu") if TORCH_AVAILABLE else "cpu"


# ------------------------------------------------------------------#
# Timing context manager
# ------------------------------------------------------------------#


@contextlib.contextmanager
def time_block(label: str = "", stream=print):
    t0 = time.time()
    yield
    stream(f"[{label}] {time.time() - t0:6.2f}s")


# ------------------------------------------------------------------#
# Tiny JSON wrappers
# ------------------------------------------------------------------#


def json_save(obj: dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2)


def json_load(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return json.load(fh)


# ------------------------------------------------------------------#
# DataFrame dumping
# ------------------------------------------------------------------#


def dump_df(df, path: str | Path, fmt: str | None = None):
    """
    Save a Pandas DataFrame; format inferred from extension or `fmt`.
    """

    path = Path(path)
    fmt = fmt or path.suffix.lstrip(".").lower() or "csv"
    if fmt == "csv":
        df.to_csv(path.with_suffix(".csv"), index=False)
    elif fmt in {"parquet", "pq"}:
        df.to_parquet(path.with_suffix(".parquet"))
    else:
        raise ValueError(f"Unsupported DataFrame format: {fmt}")


# ------------------------------------------------------------------#
# Deterministic execution context manager
# ------------------------------------------------------------------#


@contextlib.contextmanager
def deterministic(seed: int = 13):
    """
    A context manager to ensure deterministic execution for a block of code.
    Sets seeds for torch, random, and numpy, and handles GPU cache clearing.
    """
    # Store original states
    random_state = random.getstate()
    np_state = np.random.get_state()

    torch_state = None
    cuda_state = None
    if TORCH_AVAILABLE:
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()

    # Set new seeds
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        # Restore original states
        random.setstate(random_state)
        np.random.set_state(np_state)
        if TORCH_AVAILABLE and torch_state is not None:
            torch.set_rng_state(torch_state)
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

            # Clean up memory
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()


# ------------------------------------------------------------------#
# Batch utilities
# ------------------------------------------------------------------#


def extract_input_ids(batch, device=None, strict=False):
    """
    Extract input_ids from various batch formats.

    Args:
        batch: Batch data (dict, tensor, or other format)
        device: Target device (optional)
        strict: Whether to be strict about format (optional)

    Returns:
        torch.Tensor: input_ids tensor or None if not found
    """
    if batch is None:
        return None

    # Handle dictionary format
    if isinstance(batch, dict):
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        elif "inputs" in batch:
            input_ids = batch["inputs"]
        else:
            if strict:
                raise ValueError(
                    f"No input_ids found in batch dict with keys: {list(batch.keys())}"
                )
            return None
    # Handle tensor format
    elif TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
        input_ids = batch
    # Handle list/tuple format
    elif isinstance(batch, list | tuple) and len(batch) > 0:
        input_ids = batch[0]
    else:
        if strict:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        return None

    # Convert to tensor if needed
    if TORCH_AVAILABLE and not isinstance(input_ids, torch.Tensor):
        try:
            input_ids = torch.tensor(input_ids)
        except Exception as e:
            if strict:
                raise ValueError(f"Could not convert to tensor: {e}") from e
            return None

    # Move to device if specified
    if device is not None and TORCH_AVAILABLE and hasattr(input_ids, "to"):
        input_ids = input_ids.to(device)

    return input_ids
