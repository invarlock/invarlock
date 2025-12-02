"""
InvarLock Sparsity Utilities
========================

Utilities for working with sparse models and sparsity patterns.
Helper functions for analyzing and manipulating model sparsity.
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    "calculate_sparsity",
    "get_zero_mask",
    "apply_mask",
    "count_parameters",
    "get_sparsity_stats",
    "create_structured_mask",
    "validate_sparsity_target",
]


def calculate_sparsity(tensor) -> float:
    """
    Calculate sparsity ratio of a tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Sparsity ratio (fraction of zero elements)
    """
    if not TORCH_AVAILABLE:
        return 0.0

    if not isinstance(tensor, torch.Tensor):
        return 0.0

    total_elements = tensor.numel()
    if total_elements == 0:
        return 0.0

    zero_elements = (tensor == 0).sum().item()
    return zero_elements / total_elements


def get_zero_mask(tensor, threshold: float = 1e-8):
    """
    Get boolean mask of effectively zero elements.

    Args:
        tensor: PyTorch tensor
        threshold: Threshold below which values are considered zero

    Returns:
        Boolean mask where True indicates zero/near-zero elements
    """
    if not TORCH_AVAILABLE:
        return None

    if not isinstance(tensor, torch.Tensor):
        return None

    return torch.abs(tensor) <= threshold


def apply_mask(tensor, mask, fill_value: float = 0.0):
    """
    Apply a boolean mask to zero out elements.

    Args:
        tensor: PyTorch tensor to modify
        mask: Boolean mask (True = zero out)
        fill_value: Value to fill masked positions

    Returns:
        Modified tensor
    """
    if not TORCH_AVAILABLE:
        return tensor

    if not isinstance(tensor, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return tensor

    result = tensor.clone()
    result[mask] = fill_value
    return result


def count_parameters(module, only_trainable: bool = True) -> dict[str, int]:
    """
    Count parameters in a module.

    Args:
        module: PyTorch module
        only_trainable: Count only trainable parameters

    Returns:
        Dictionary with parameter counts
    """
    if not TORCH_AVAILABLE or not isinstance(module, nn.Module):
        return {"total": 0, "trainable": 0, "non_trainable": 0}

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in module.parameters():
        param_count = param.numel()
        total_params += param_count

        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def get_sparsity_stats(module) -> dict[str, Any]:
    """
    Get comprehensive sparsity statistics for a module.

    Args:
        module: PyTorch module

    Returns:
        Dictionary with sparsity statistics
    """
    if not TORCH_AVAILABLE or not isinstance(module, nn.Module):
        return {}

    stats = {
        "overall_sparsity": 0.0,
        "layer_sparsities": {},
        "parameter_counts": count_parameters(module),
        "sparse_layers": 0,
        "dense_layers": 0,
    }

    total_elements = 0
    total_zeros = 0

    for name, param in module.named_parameters():
        if param.numel() == 0:
            continue

        layer_sparsity = calculate_sparsity(param)
        stats["layer_sparsities"][name] = layer_sparsity

        # Count for overall statistics
        elements = param.numel()
        zeros = (param == 0).sum().item()

        total_elements += elements
        total_zeros += zeros

        # Classify layer as sparse or dense
        if layer_sparsity > 0.1:  # 10% threshold for "sparse"
            stats["sparse_layers"] += 1
        else:
            stats["dense_layers"] += 1

    # Calculate overall sparsity
    if total_elements > 0:
        stats["overall_sparsity"] = total_zeros / total_elements

    return stats


def create_structured_mask(
    shape: tuple[int, ...], pattern: str = "block", block_size: int = 4
) -> torch.Tensor | None:
    """
    Create block/group sparsity masks.

    Args:
        shape: Shape of the tensor
        pattern: Sparsity pattern ('block', 'column', 'row')
        block_size: Size of blocks for block sparsity

    Returns:
        Boolean mask tensor
    """
    if not TORCH_AVAILABLE:
        return None

    mask = torch.zeros(shape, dtype=torch.bool)

    if pattern == "block" and len(shape) >= 2:
        # Block sparsity pattern
        for i in range(0, shape[0], block_size):
            for j in range(0, shape[1], block_size):
                # Create checkerboard pattern of blocks
                if (i // block_size + j // block_size) % 2 == 0:
                    end_i = min(i + block_size, shape[0])
                    end_j = min(j + block_size, shape[1])
                    mask[i:end_i, j:end_j] = True

    elif pattern == "column" and len(shape) >= 2:
        # Column sparsity - zero out every other column
        mask[:, ::2] = True

    elif pattern == "row" and len(shape) >= 2:
        # Row sparsity - zero out every other row
        mask[::2, :] = True

    else:
        warnings.warn(
            f"Unsupported sparsity pattern '{pattern}' for shape {shape}", stacklevel=2
        )

    return mask


def validate_sparsity_target(target_sparsity: Any) -> bool:
    """
    Validate sparsity target value.

    Args:
        target_sparsity: Target sparsity ratio (0.0 to 1.0)

    Returns:
        True if valid
    """
    if not isinstance(target_sparsity, int | float):
        return False
    return 0.0 <= target_sparsity <= 1.0


def get_magnitude_mask(tensor, sparsity_ratio: float) -> torch.Tensor | None:
    """
    Create magnitude-based sparsity mask.

    Args:
        tensor: PyTorch tensor
        sparsity_ratio: Target sparsity fraction

    Returns:
        Boolean mask where True indicates weights to zero
    """
    if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
        return None

    if not validate_sparsity_target(sparsity_ratio):
        raise ValueError(f"Invalid sparsity ratio: {sparsity_ratio}")

    # Flatten tensor for magnitude ranking
    flat_tensor = tensor.view(-1)

    # Find threshold for zeroing
    num_to_zero = int(sparsity_ratio * flat_tensor.numel())
    if num_to_zero == 0:
        return torch.zeros_like(tensor, dtype=torch.bool)

    # Get magnitude and find threshold
    magnitudes = torch.abs(flat_tensor)
    threshold_value = torch.kthvalue(magnitudes, num_to_zero).values

    # Create mask
    mask = torch.abs(tensor) <= threshold_value

    return mask


def analyze_sparsity_impact(original_tensor, edited_tensor) -> dict[str, Any]:
    """
    Analyze the impact of applied sparsity on a tensor.

    Args:
        original_tensor: Original tensor before changes
        edited_tensor: Tensor after applying sparsity

    Returns:
        Dictionary with impact analysis
    """
    if not TORCH_AVAILABLE:
        return {}

    if not (
        isinstance(original_tensor, torch.Tensor)
        and isinstance(edited_tensor, torch.Tensor)
    ):
        return {}

    # Calculate basic statistics
    original_sparsity = calculate_sparsity(original_tensor)
    final_sparsity = calculate_sparsity(edited_tensor)

    # Calculate magnitude changes
    magnitude_change = torch.norm(edited_tensor - original_tensor).item()
    relative_change = (
        magnitude_change / torch.norm(original_tensor).item()
        if torch.norm(original_tensor) > 0
        else 0.0
    )

    analysis = {
        "original_sparsity": original_sparsity,
        "final_sparsity": final_sparsity,
        "sparsity_increase": final_sparsity - original_sparsity,
        "magnitude_change": magnitude_change,
        "relative_change": relative_change,
        "compression_ratio": final_sparsity / original_sparsity
        if original_sparsity > 0
        else float("inf"),
    }

    return analysis
