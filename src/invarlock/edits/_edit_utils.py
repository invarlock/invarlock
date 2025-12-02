"""
InvarLock Edit Utilities
====================

Shared helper functions for edit implementations.
Common functionality used across multiple edit types.
"""

from __future__ import annotations

import warnings
from typing import Any

try:
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    "validate_model_structure",
    "calculate_compression_ratio",
    "get_layer_info",
    "validate_edit_parameters",
    "create_edit_metadata",
]


def validate_model_structure(
    model_desc: dict[str, Any], required_fields: list[str]
) -> bool:
    """
    Validate that a model description has required fields for an edit.

    Args:
        model_desc: Model description from adapter
        required_fields: List of required field names

    Returns:
        True if all required fields are present
    """
    missing_fields = []

    for field in required_fields:
        if field not in model_desc:
            missing_fields.append(field)

    if missing_fields:
        warnings.warn(
            f"Model missing required fields for edit: {missing_fields}", stacklevel=2
        )
        return False

    return True


def calculate_compression_ratio(original_params: int, final_params: int) -> float:
    """
    Calculate parameter compression ratio.

    Args:
        original_params: Original parameter count
        final_params: Final parameter count after edit

    Returns:
        Compression ratio (final/original)
    """
    if original_params == 0:
        return 1.0

    return final_params / original_params


def get_layer_info(model_desc: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    """
    Extract information for a specific layer.

    Args:
        model_desc: Model description from adapter
        layer_idx: Index of the layer

    Returns:
        Dictionary with layer-specific information
    """
    n_layers = model_desc.get("n_layer", 0)

    if layer_idx >= n_layers:
        raise IndexError(f"Layer index {layer_idx} >= {n_layers}")

    heads_per_layer = model_desc.get("heads_per_layer", [])
    mlp_dims = model_desc.get("mlp_dims", [])

    layer_info = {
        "layer_idx": layer_idx,
        "n_heads": heads_per_layer[layer_idx]
        if layer_idx < len(heads_per_layer)
        else None,
        "mlp_dim": mlp_dims[layer_idx] if layer_idx < len(mlp_dims) else None,
        "hidden_size": model_desc.get("hidden_size"),
    }

    return layer_info


def validate_edit_parameters(
    params: dict[str, Any],
    required_params: list[str],
    optional_params: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Validate edit parameters.

    Args:
        params: Parameters to validate
        required_params: List of required parameter names
        optional_params: Dict of optional parameters with default values

    Returns:
        (is_valid, error_message)
    """
    # Check required parameters
    missing_required = []
    for param in required_params:
        if param not in params:
            missing_required.append(param)

    if missing_required:
        return False, f"Missing required parameters: {missing_required}"

    # Add default values for optional parameters
    if optional_params:
        for param, default_value in optional_params.items():
            if param not in params:
                params[param] = default_value

    return True, "Parameters valid"


def create_edit_metadata(
    edit_name: str,
    model_desc: dict[str, Any],
    parameters: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Create standardized edit metadata.

    Args:
        edit_name: Name of the edit operation
        model_desc: Original model description
        parameters: Edit parameters used
        result: Edit operation results

    Returns:
        Standardized metadata dictionary
    """
    original_params = model_desc.get("total_params", 0)
    final_params = result.get("final_params", original_params)

    metadata: dict[str, Any] = {
        "name": edit_name,
        "parameters": parameters.copy(),
        "original_model": {
            "n_layer": model_desc.get("n_layer", 0),
            "total_params": original_params,
            "model_type": model_desc.get("model_type", "unknown"),
        },
        "results": {
            "final_params": final_params,
            "compression_ratio": calculate_compression_ratio(
                original_params, final_params
            ),
            "layers_modified": result.get("layers_modified", []),
            "success": result.get("success", True),
        },
    }

    # Add edit-specific results
    results_dict = metadata["results"]
    assert isinstance(results_dict, dict)
    for key, value in result.items():
        if key not in results_dict:
            results_dict[key] = value

    return metadata


def get_module_parameter_count(module) -> int:
    """
    Count parameters in a module (if torch is available).

    Args:
        module: PyTorch module

    Returns:
        Number of parameters
    """
    if not TORCH_AVAILABLE:
        return 0

    if not isinstance(module, nn.Module):
        return 0

    return sum(p.numel() for p in module.parameters())


def validate_compression_target(
    target_ratio: Any, min_ratio: float = 0.1, max_ratio: float = 1.0
) -> bool:
    """
    Validate compression target ratio.

    Args:
        target_ratio: Desired compression ratio
        min_ratio: Minimum allowed ratio
        max_ratio: Maximum allowed ratio

    Returns:
        True if ratio is valid
    """
    if not isinstance(target_ratio, int | float):
        return False

    return min_ratio <= target_ratio <= max_ratio


def create_layer_mask(
    n_layers: int, layers_to_edit: list[int] | None = None
) -> list[bool]:
    """
    Create a boolean mask for layers to edit.

    Args:
        n_layers: Total number of layers
        layers_to_edit: List of layer indices to edit (None = all layers)

    Returns:
        Boolean mask where True indicates layer should be edited
    """
    if layers_to_edit is None:
        return [True] * n_layers

    mask = [False] * n_layers
    for layer_idx in layers_to_edit:
        if 0 <= layer_idx < n_layers:
            mask[layer_idx] = True

    return mask
