from __future__ import annotations

from typing import Any

import torch

_SPECTRAL_CONTROL_ERRORS = (
    ArithmeticError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def apply_weight_rescale(
    model: Any,
    scale_factor: float = 1.0,
    scope: str = "all",
    *,
    should_process_module_fn: Any | None = None,
) -> dict[str, Any]:
    """Apply weight rescaling to model parameters."""
    if should_process_module_fn is None:
        from . import spectral_detection as _spectral_detection

        should_process_module_fn = _spectral_detection.should_process_module
    try:
        rescaled_modules = []
        failed_modules = []

        for name, module in model.named_modules():
            if not should_process_module_fn(name, module, scope):
                continue
            try:
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                    if hasattr(weight, "dtype") and weight.dtype in [torch.int8]:
                        continue
                    with torch.no_grad():
                        weight.mul_(scale_factor)
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.mul_(scale_factor)
                    rescaled_modules.append(name)
            except _SPECTRAL_CONTROL_ERRORS as error:
                failed_modules.append((name, str(error)))

        return {
            "applied": len(rescaled_modules) > 0,
            "scale_factor": scale_factor,
            "rescaled_modules": rescaled_modules,
            "failed_modules": failed_modules,
            "message": f"Rescaled {len(rescaled_modules)} modules with factor {scale_factor}",
        }
    except _SPECTRAL_CONTROL_ERRORS as error:
        return {
            "applied": False,
            "error": str(error),
            "message": f"Weight rescaling failed: {error}",
        }


def apply_relative_spectral_cap(
    model: Any,
    cap_ratio: float = 2.0,
    scope: str = "all",
    baseline_sigmas: dict[str, float] | None = None,
    *,
    should_process_module_fn: Any | None = None,
    capture_baseline_sigmas_fn: Any | None = None,
    compute_sigma_max_fn: Any | None = None,
) -> dict[str, Any]:
    """Apply relative spectral capping to model weights."""
    if should_process_module_fn is None:
        from . import spectral_detection as _spectral_detection

        should_process_module_fn = _spectral_detection.should_process_module
    if capture_baseline_sigmas_fn is None:
        from . import spectral_measurement as _spectral_measurement

        capture_baseline_sigmas_fn = _spectral_measurement.capture_baseline_sigmas
    if compute_sigma_max_fn is None:
        from . import spectral_measurement as _spectral_measurement

        compute_sigma_max_fn = _spectral_measurement.compute_sigma_max
    try:
        if baseline_sigmas is None:
            baseline_sigmas = capture_baseline_sigmas_fn(
                model, scope=scope, should_process_module_fn=should_process_module_fn
            )

        capped_modules = []
        failed_modules = []
        for name, module in model.named_modules():
            if not should_process_module_fn(name, module, scope):
                continue
            try:
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                    if hasattr(weight, "dtype") and weight.dtype in [torch.int8]:
                        continue
                    current_sigma = compute_sigma_max_fn(weight)
                    baseline_sigma = baseline_sigmas.get(name, current_sigma)
                    max_allowed = baseline_sigma * cap_ratio
                    if current_sigma > max_allowed:
                        scale_factor = max_allowed / current_sigma
                        with torch.no_grad():
                            weight.mul_(scale_factor)
                        capped_modules.append(
                            {
                                "module": name,
                                "original_sigma": current_sigma,
                                "capped_sigma": max_allowed,
                                "scale_factor": scale_factor,
                            }
                        )
            except _SPECTRAL_CONTROL_ERRORS as error:
                failed_modules.append((name, str(error)))

        return {
            "applied": len(capped_modules) > 0,
            "cap_ratio": cap_ratio,
            "capped_modules": capped_modules,
            "failed_modules": failed_modules,
            "message": f"Applied spectral capping to {len(capped_modules)} modules",
        }
    except _SPECTRAL_CONTROL_ERRORS as error:
        return {
            "applied": False,
            "error": str(error),
            "message": f"Spectral capping failed: {error}",
        }


def apply_spectral_control(
    model: Any,
    policy: dict[str, Any],
    *,
    apply_relative_spectral_cap_fn: Any | None = None,
    apply_weight_rescale_fn: Any | None = None,
) -> dict[str, Any]:
    """Apply spectral control based on policy."""
    if apply_relative_spectral_cap_fn is None:
        apply_relative_spectral_cap_fn = apply_relative_spectral_cap
    if apply_weight_rescale_fn is None:
        apply_weight_rescale_fn = apply_weight_rescale
    try:
        results: dict[str, Any] = {
            "rescaling_applied": False,
            "capping_applied": False,
            "modules_processed": 0,
            "corrections": [],
        }

        scope = policy.get("scope", "all")
        baseline_sigmas = policy.get("baseline_sigmas")
        cap_ratio = policy.get("cap_ratio", 2.0)
        cap_result = apply_relative_spectral_cap_fn(
            model, cap_ratio=cap_ratio, scope=scope, baseline_sigmas=baseline_sigmas
        )
        if cap_result["applied"]:
            results["capping_applied"] = True
            results["corrections"].extend(cap_result["capped_modules"])

        if "rescale_factor" in policy:
            rescale_result = apply_weight_rescale_fn(
                model, scale_factor=policy["rescale_factor"], scope=scope
            )
            if rescale_result["applied"]:
                results["rescaling_applied"] = True
                results["modules_processed"] += len(rescale_result["rescaled_modules"])

        results["applied"] = results["rescaling_applied"] or results["capping_applied"]
        results["policy"] = policy
        results["message"] = (
            f"Spectral control applied: capping={results['capping_applied']}, "
            f"rescaling={results['rescaling_applied']}"
        )
        return results
    except _SPECTRAL_CONTROL_ERRORS as error:
        return {
            "applied": False,
            "error": str(error),
            "policy": policy,
            "message": f"Spectral control failed: {error}",
        }


__all__ = [
    "apply_relative_spectral_cap",
    "apply_spectral_control",
    "apply_weight_rescale",
]
