"""
External Utilities for Edit Operations
=====================================

Utilities for integrating with external edit backends and guard chains.
Provides common functionality for model snapshots, validation, and guard policies.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn


class ExternalBackend:
    """Base class for external edit backends."""

    def apply_edit(self, model: nn.Module, config: dict, calib: Any) -> dict:
        """Apply edit to model."""
        raise NotImplementedError()

    def get_edit_info(self, model: nn.Module, config: dict) -> dict:
        """Get edit information without applying."""
        raise NotImplementedError()


class ExternalEditWrapper:
    """Wrapper for external edit implementations."""

    def __init__(self, backend: ExternalBackend):
        self.backend = backend

    def preview(self, model: nn.Module, adapter: Any, calib: Any) -> dict:
        """Preview edit operation."""
        return {"plan": {}, "estimated_sparsity": {}, "preview_metrics": {}}

    def apply(self, model: nn.Module, adapter: Any, plan: dict) -> dict:
        """Apply edit operation."""
        return {"actual_sparsity": {}, "modified_layers": [], "metrics": {}}


def check_dependencies(edit_type: str) -> bool:
    """
    Check if dependencies for edit type are available.

    Args:
        edit_type: Type of edit (e.g., 'quant')

    Returns:
        True if dependencies are available
    """
    if edit_type == "svd":
        return False
    elif edit_type == "quant":
        # Quantization uses standard PyTorch
        return True
    else:
        return False


def validate_edit_config(config: dict, edit_type: str) -> dict:
    """
    Validate and normalize edit configuration.

    Args:
        config: Configuration dictionary
        edit_type: Type of edit operation

    Returns:
        Validated configuration dictionary
    """
    validated_config = config.copy()

    if edit_type == "svd":
        raise ValueError("Unsupported edit type: svd")

    elif edit_type == "quant":
        # Validate quantization specific config
        bits = validated_config.get("bits", 8)
        if bits not in [4, 8, 16]:
            warnings.warn(
                f"Unusual bit width {bits}, common values are 4, 8, or 16", stacklevel=2
            )

    return validated_config


def compute_edit_metrics(
    model_before: nn.Module, model_after: nn.Module, config: dict
) -> dict:
    """
    Compute metrics comparing model before and after edit.

    Args:
        model_before: Model state before edit
        model_after: Model state after edit
        config: Edit configuration

    Returns:
        Dictionary of computed metrics
    """
    # Count parameters
    params_before = sum(p.numel() for p in model_before.parameters())
    params_after = sum(p.numel() for p in model_after.parameters())

    # Count non-zero parameters (for sparsity)
    nonzero_before = sum((p != 0).sum().item() for p in model_before.parameters())
    nonzero_after = sum((p != 0).sum().item() for p in model_after.parameters())

    # Calculate compression ratios
    param_ratio = params_after / params_before if params_before > 0 else 1.0
    sparsity_ratio = (
        1.0 - (nonzero_after / nonzero_before) if nonzero_before > 0 else 0.0
    )

    return {
        "params_before": params_before,
        "params_after": params_after,
        "nonzero_before": nonzero_before,
        "nonzero_after": nonzero_after,
        "param_compression_ratio": param_ratio,
        "sparsity_achieved": sparsity_ratio,
        "memory_saved_mb": (params_before - params_after) * 4 / (1024 * 1024),
    }


def prepare_calibration_data(calib_data: Any, config: dict) -> Any:
    """
    Prepare calibration data for edit operations.

    Args:
        calib_data: Raw calibration data
        config: Edit configuration

    Returns:
        Prepared calibration data
    """
    if calib_data is None:
        return None

    # If it's already prepared (has __iter__ and __len__), return as-is
    if hasattr(calib_data, "__iter__") and hasattr(calib_data, "__len__"):
        return calib_data

    # If it's a list, return as-is
    if isinstance(calib_data, list):
        return calib_data

    # If it's a tensor, wrap in a list
    if isinstance(calib_data, torch.Tensor):
        return [calib_data]

    return calib_data


def safe_model_snapshot(model: nn.Module, adapter: Any) -> dict:
    """
    Create a safe snapshot of model state for potential rollback.

    Args:
        model: Model to snapshot
        adapter: Model adapter for model-specific operations

    Returns:
        Model snapshot dictionary
    """
    try:
        # Create deep copy of state dict for safety
        state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.clone()

        # Store model configuration if available through adapter
        model_config = {}
        if hasattr(adapter, "get_config"):
            try:
                model_config = adapter.get_config(model)
            except Exception:
                pass

        return {
            "state_dict": state_dict,
            "config": model_config,
            "snapshot_time": torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available()
            else None,
        }
    except Exception as e:
        warnings.warn(f"Failed to create model snapshot: {e}", stacklevel=2)
        return {}


def safe_model_restore(model: nn.Module, snapshot: dict, adapter: Any) -> bool:
    """
    Safely restore model from snapshot.

    Args:
        model: Model to restore
        snapshot: Snapshot from safe_model_snapshot()
        adapter: Model adapter for model-specific operations

    Returns:
        True if restore was successful
    """
    try:
        if "state_dict" not in snapshot:
            warnings.warn("Snapshot missing state_dict, cannot restore", stacklevel=2)
            return False

        # Restore parameters
        state_dict = snapshot["state_dict"]
        for name, param in model.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])

        return True
    except Exception as e:
        warnings.warn(f"Failed to restore model from snapshot: {e}", stacklevel=2)
        return False


def create_baseline_guard_policy(edit_type: str) -> dict:
    """
    Create baseline guard policy for edit type.

    Args:
        edit_type: Type of edit operation

    Returns:
        Guard policy configuration
    """
    baseline_policies = {
        "quant": {
            "spectral_monitoring": False,  # Quantization may change spectral properties
            "rmt_detection": True,
            "weight_change_threshold": 0.5,  # Higher tolerance for quantization
            "activation_change_threshold": 0.3,
            "max_spectral_norm_increase": 3.0,
        },
    }

    return baseline_policies.get(
        edit_type,
        {
            "spectral_monitoring": True,
            "rmt_detection": True,
            "weight_change_threshold": 0.1,
            "activation_change_threshold": 0.1,
            "max_spectral_norm_increase": 2.0,
        },
    )


__all__ = [
    "ExternalBackend",
    "ExternalEditWrapper",
    "check_dependencies",
    "validate_edit_config",
    "compute_edit_metrics",
    "prepare_calibration_data",
    "safe_model_snapshot",
    "safe_model_restore",
    "create_baseline_guard_policy",
]
