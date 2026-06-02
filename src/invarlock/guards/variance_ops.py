from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .quantized_weights import is_packed_quantized_module, is_quantized_weight

_VARIANCE_OPERATION_ERRORS = (
    ArithmeticError,
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_quantized_mutation_unsupported(
    guard: Any, *, module_name: str, weight: Any, operation: str
) -> None:
    entry = {
        "module": module_name,
        "dtype": str(getattr(weight, "dtype", "unknown")),
        "operation": operation,
        "reason": "packed_quantized_weight_mutation_unsupported",
        "assurance_blocking": True,
    }
    guard._stats.setdefault("quantized_mutation_unsupported", []).append(entry)


def _quantized_mutation_marker(module: Any) -> Any | None:
    weight = getattr(module, "weight", None)
    if weight is not None and is_quantized_weight(weight):
        return weight
    if is_packed_quantized_module(module):
        for attr in ("weight", "qweight", "packed_weight", "packed_weights"):
            try:
                marker = getattr(module, attr, None)
            except (RuntimeError, TypeError, ValueError):
                return module
            if marker is not None:
                return marker
        return module
    return None


def _target_module_for_scale(guard: Any, scale_name: str) -> Any | None:
    for target_name, target_module in guard._target_modules.items():
        if scale_name == target_name:
            return target_module
        if guard._scale_matches_target(scale_name, target_name):
            return target_module
    return None


def push_checkpoint(guard: Any, model: nn.Module) -> None:
    """Push current target-module weights to the checkpoint stack."""
    if not guard._target_modules:
        return

    checkpoint: dict[str, torch.Tensor] = {}
    for name, module in guard._target_modules.items():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if is_quantized_weight(weight) or is_packed_quantized_module(module):
            continue
        checkpoint[name] = weight.data.clone().detach()

    guard._checkpoint_stack.append(checkpoint)
    guard._log_event(
        "checkpoint_pushed",
        message=f"Pushed checkpoint for {len(checkpoint)} modules",
        modules_count=len(checkpoint),
        stack_depth=len(guard._checkpoint_stack),
    )


def pop_checkpoint(guard: Any, model: nn.Module) -> bool:
    """Pop and restore the most recent checkpoint."""
    if not guard._checkpoint_stack:
        guard._log_event(
            "checkpoint_pop_failed",
            level="WARN",
            message="No checkpoint available for rollback",
        )
        return False

    checkpoint = guard._checkpoint_stack.pop()
    restored_count = 0
    for name, saved_weight in checkpoint.items():
        if name in guard._target_modules:
            module = guard._target_modules[name]
            if hasattr(module, "weight"):
                module.weight.data.copy_(saved_weight)
                restored_count += 1

    guard._log_event(
        "checkpoint_popped",
        message=f"Restored checkpoint for {restored_count}/{len(checkpoint)} modules",
        restored_count=restored_count,
        stack_depth=len(guard._checkpoint_stack),
    )
    return True


def commit_checkpoint(guard: Any) -> None:
    """Commit current state by removing the most recent checkpoint."""
    if guard._checkpoint_stack:
        guard._checkpoint_stack.pop()
        guard._log_event(
            "checkpoint_committed",
            message="Committed current state, removed checkpoint",
            stack_depth=len(guard._checkpoint_stack),
        )


def enable_guard(guard: Any, model: nn.Module, adapter=None) -> bool:
    """Enable VE with checkpoint discipline and idempotent operation."""
    _ = adapter
    guard._enable_attempt_count += 1

    if guard._monitor_only:
        guard._log_event(
            "enable_skipped_monitor_only",
            level="INFO",
            message="Monitor-only mode: VE enable skipped",
            attempt_count=guard._enable_attempt_count,
        )
        guard._enabled = False
        return False

    if not guard._prepared or not guard._scales:
        guard._log_event(
            "enable_skipped",
            level="WARN",
            message="Cannot enable VE: not prepared or no scales computed",
            attempt_count=guard._enable_attempt_count,
        )
        return False

    if guard._enabled:
        guard._log_event(
            "enable_idempotent",
            message="VE already enabled, verifying state",
            attempt_count=guard._enable_attempt_count,
        )
        return True

    quantized_unsupported: list[str] = []
    for scale_name in guard._scales:
        module = _target_module_for_scale(guard, scale_name)
        if module is None:
            continue
        quantized_marker = _quantized_mutation_marker(module)
        if quantized_marker is None:
            continue
        _record_quantized_mutation_unsupported(
            guard,
            module_name=scale_name,
            weight=quantized_marker,
            operation="enable",
        )
        quantized_unsupported.append(scale_name)
        guard._log_event(
            "scale_unsupported_quantized",
            level="WARN",
            message=(
                "Cannot apply variance scaling to packed quantized "
                f"weights in {scale_name}"
            ),
            module_name=scale_name,
            dtype=str(getattr(quantized_marker, "dtype", "unknown")),
            assurance_blocking=True,
        )

    if quantized_unsupported:
        guard._log_event(
            "enable_failed_quantized_unsupported",
            level="ERROR",
            message=(
                "Packed quantized variance targets are not safely mutable; "
                "skipping checkpoint and scale application"
            ),
            quantized_modules=quantized_unsupported,
            assurance_blocking=True,
        )
        guard._enabled = False
        return False

    push_checkpoint(guard, model)
    guard._log_event(
        "enable_start",
        message=f"Enabling VE with {len(guard._scales)} scale factors",
        attempt_count=guard._enable_attempt_count,
    )

    try:
        applied_count = 0
        failed_modules: list[str] = []
        late_quantized_unsupported: list[str] = []

        for scale_name, scale_factor in guard._scales.items():
            try:
                module = _target_module_for_scale(guard, scale_name)

                if module is not None:
                    quantized_marker = _quantized_mutation_marker(module)
                    if quantized_marker is not None:
                        _record_quantized_mutation_unsupported(
                            guard,
                            module_name=scale_name,
                            weight=quantized_marker,
                            operation="enable",
                        )
                        late_quantized_unsupported.append(scale_name)
                        failed_modules.append(scale_name)
                        guard._log_event(
                            "scale_unsupported_quantized",
                            level="WARN",
                            message=(
                                "Cannot apply variance scaling to packed quantized "
                                f"weights in {scale_name}"
                            ),
                            module_name=scale_name,
                            dtype=str(getattr(quantized_marker, "dtype", "unknown")),
                            assurance_blocking=True,
                        )
                        continue

                if module is not None and hasattr(module, "weight"):
                    if scale_name not in guard._original_scales:
                        guard._original_scales[scale_name] = 1.0

                    with torch.no_grad():
                        original_device = module.weight.device
                        original_dtype = module.weight.dtype
                        if str(original_device).startswith("mps"):
                            module.weight.data = module.weight.data * scale_factor
                        else:
                            scale_tensor = module.weight.new_tensor(scale_factor).to(
                                dtype=original_dtype, device=original_device
                            )
                            module.weight.mul_(scale_tensor)

                    applied_count += 1
                    guard._log_event(
                        "scale_applied",
                        message=f"Applied scale {scale_factor:.3f} to {scale_name}",
                        module_name=scale_name,
                        scale_factor=scale_factor,
                    )
                else:
                    failed_modules.append(scale_name)
            except _VARIANCE_OPERATION_ERRORS as error:
                failed_modules.append(scale_name)
                guard._log_event(
                    "scale_apply_error",
                    level="ERROR",
                    message=f"Failed to apply scale to {scale_name}: {str(error)}",
                    module_name=scale_name,
                    error=str(error),
                )

        if late_quantized_unsupported:
            pop_checkpoint(guard, model)
            guard._log_event(
                "enable_failed_quantized_unsupported",
                level="ERROR",
                message=(
                    "Packed quantized variance targets are not safely mutable; "
                    "rolling back"
                ),
                failed_modules=failed_modules,
                quantized_modules=late_quantized_unsupported,
                assurance_blocking=True,
            )
            guard._enabled = False
            return False

        if applied_count == 0:
            pop_checkpoint(guard, model)
            guard._log_event(
                "enable_failed",
                level="ERROR",
                message="No modules were successfully scaled, rolling back",
                failed_modules=failed_modules,
            )
            return False

        if failed_modules:
            guard._log_event(
                "enable_partial",
                level="WARN",
                message=(
                    f"Partial success: {applied_count} succeeded, {len(failed_modules)} failed"
                ),
                applied_count=applied_count,
                failed_modules=failed_modules,
            )

        commit_checkpoint(guard)
        guard._enabled = True
        guard._log_event(
            "enable_complete",
            message=f"Enabled VE on {applied_count}/{len(guard._scales)} modules",
            applied_count=applied_count,
            total_scales=len(guard._scales),
            attempt_count=guard._enable_attempt_count,
        )
        return True
    except _VARIANCE_OPERATION_ERRORS as error:
        pop_checkpoint(guard, model)
        guard._log_event(
            "enable_catastrophic_failure",
            level="ERROR",
            message=f"Catastrophic failure during enable: {str(error)}",
            error=str(error),
            attempt_count=guard._enable_attempt_count,
        )
        return False


def disable_guard(guard: Any, model: nn.Module, adapter=None) -> bool:
    """Disable VE with idempotent operation and exact restoration."""
    _ = adapter
    guard._disable_attempt_count += 1

    if not guard._enabled:
        guard._log_event(
            "disable_idempotent",
            message="VE already disabled",
            attempt_count=guard._disable_attempt_count,
        )
        return True

    guard._log_event(
        "disable_start",
        message="Disabling VE by reverting to exact previous state",
        attempt_count=guard._disable_attempt_count,
    )

    try:
        if guard._checkpoint_stack:
            success = pop_checkpoint(guard, model)
            if success:
                guard._enabled = False
                guard._log_event(
                    "disable_checkpoint_complete",
                    message="Disabled VE using checkpoint restoration",
                    attempt_count=guard._disable_attempt_count,
                )
                return True
            guard._log_event(
                "disable_checkpoint_failed",
                level="WARN",
                message="Checkpoint restoration failed, falling back to inverse scaling",
            )

        reverted_count = 0
        failed_modules: list[str] = []
        for scale_name, scale_factor in guard._scales.items():
            try:
                module = None
                for target_name, target_module in guard._target_modules.items():
                    if scale_name == target_name:
                        module = target_module
                        break
                    if guard._scale_matches_target(scale_name, target_name):
                        module = target_module
                        break

                if module is not None:
                    quantized_marker = _quantized_mutation_marker(module)
                    if quantized_marker is not None:
                        _record_quantized_mutation_unsupported(
                            guard,
                            module_name=scale_name,
                            weight=quantized_marker,
                            operation="disable",
                        )
                        guard._log_event(
                            "revert_unsupported_quantized",
                            level="WARN",
                            message=(
                                "Cannot revert variance scaling for packed "
                                f"quantized weights in {scale_name}"
                            ),
                            module_name=scale_name,
                            dtype=str(getattr(quantized_marker, "dtype", "unknown")),
                            assurance_blocking=True,
                        )
                        continue

                if module is not None and hasattr(module, "weight"):
                    revert_factor = 1.0 / scale_factor
                    with torch.no_grad():
                        original_device = module.weight.device
                        original_dtype = module.weight.dtype
                        if str(original_device).startswith("mps"):
                            module.weight.data = module.weight.data * revert_factor
                        else:
                            revert_tensor = module.weight.new_tensor(revert_factor).to(
                                dtype=original_dtype, device=original_device
                            )
                            module.weight.mul_(revert_tensor)

                    reverted_count += 1
                    guard._log_event(
                        "scale_reverted",
                        message=(
                            f"Reverted scale {scale_factor:.3f} from {scale_name} "
                            f"(factor: {revert_factor:.3f})"
                        ),
                        module_name=scale_name,
                        original_scale=scale_factor,
                        revert_factor=revert_factor,
                    )
                else:
                    failed_modules.append(scale_name)
            except _VARIANCE_OPERATION_ERRORS as error:
                failed_modules.append(scale_name)
                guard._log_event(
                    "scale_revert_error",
                    level="ERROR",
                    message=f"Failed to revert scale from {scale_name}: {str(error)}",
                    module_name=scale_name,
                    error=str(error),
                )

        if reverted_count == 0 and guard._scales:
            guard._log_event(
                "disable_failed",
                level="ERROR",
                message="No modules were successfully reverted",
                failed_modules=failed_modules,
            )
            return False

        if failed_modules:
            guard._log_event(
                "disable_partial",
                level="WARN",
                message=(
                    f"Partial success: {reverted_count} reverted, {len(failed_modules)} failed"
                ),
                reverted_count=reverted_count,
                failed_modules=failed_modules,
            )

        guard._enabled = False
        guard._log_event(
            "disable_complete",
            message=f"Disabled VE on {reverted_count}/{len(guard._scales)} modules",
            reverted_count=reverted_count,
            attempt_count=guard._disable_attempt_count,
        )
        return True
    except _VARIANCE_OPERATION_ERRORS as error:
        guard._log_event(
            "disable_catastrophic_failure",
            level="ERROR",
            message=f"Catastrophic failure during disable: {str(error)}",
            error=str(error),
            attempt_count=guard._disable_attempt_count,
        )
        return False


__all__ = [
    "commit_checkpoint",
    "disable_guard",
    "enable_guard",
    "pop_checkpoint",
    "push_checkpoint",
]
