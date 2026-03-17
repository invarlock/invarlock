"""Model utilities used by CLI/runtime code."""

import random
from typing import Any

import numpy as np

_TORCH_UNSET = object()
_torch_module: Any = _TORCH_UNSET

__all__ = ["set_seed"]


def _get_torch() -> Any:
    global _torch_module
    if _torch_module is _TORCH_UNSET:
        try:
            import torch as _torch
        except ModuleNotFoundError:  # utils may be used without torch
            _torch_module = None
        else:
            _torch_module = _torch
    return None if _torch_module is _TORCH_UNSET else _torch_module


def set_seed(seed: int = 42):
    """Deterministic seeds for Python, NumPy and Torch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch = _get_torch()
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
