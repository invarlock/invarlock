"""Model utilities used by CLI/runtime code."""

import random

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # utils may be used without torch
    TORCH_AVAILABLE = False

__all__ = ["set_seed"]


def set_seed(seed: int = 42):
    """Deterministic seeds for Python, NumPy and Torch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
