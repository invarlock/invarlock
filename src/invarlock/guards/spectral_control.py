from __future__ import annotations

from typing import Any

import torch

from .quantized_weights import is_quantized_weight

_SPECTRAL_CONTROL_ERRORS = (
    ArithmeticError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _is_matrix_weight(weight: Any) -> bool:
    if weight is None:
        return False
    try:
        return int(getattr(weight, "ndim", 0) or 0) == 2
    except (TypeError, ValueError):
        # guard-fallback-ok: malformed weight metadata is classified as not a matrix.
        return False


def _spectral_norm(weight: torch.Tensor) -> float:
    """Compute the spectral norm (largest singular value) of ``weight``."""
    if weight.ndim != 2:
        weight = weight.view(weight.shape[0], -1)
    try:
        s = torch.linalg.svdvals(weight)
    except RuntimeError:
        s = torch.linalg.svdvals(weight.cpu()).to(weight.device)
    return float(s.max().item())


def enforce_relative_spectral_cap(
    weight: torch.Tensor, baseline_sigma: float | torch.Tensor, cap_ratio: float
) -> torch.Tensor:
    """Clamp the spectral norm of ``weight`` to ``cap_ratio * baseline_sigma``."""
    baseline_value = float(baseline_sigma)
    if not torch.isfinite(torch.tensor(baseline_value)) or baseline_value <= 0:
        return weight
    with torch.no_grad():
        sigma = _spectral_norm(weight)
        limit = baseline_value * cap_ratio
        if sigma > limit and sigma > 0:
            safe_limit = limit * (1.0 - 1e-6)
            if safe_limit < 0:
                safe_limit = 0.0
            weight.mul_(safe_limit / sigma)
    return weight


def enforce_weight_energy_bound(
    approx: torch.Tensor, exact: torch.Tensor, max_relative_error: float
) -> torch.Tensor:
    """Return ``approx`` if the relative error against ``exact`` is within bounds."""
    denom = torch.norm(exact).clamp_min(1e-12)
    rel_err = torch.norm(approx - exact) / denom
    if rel_err <= max_relative_error:
        return approx
    return exact


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
                if not _is_matrix_weight(weight):
                    continue
                if is_quantized_weight(weight):
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
    selected_modules: set[str] | list[str] | tuple[str, ...] | None = None,
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

        selected_names = (
            {str(name) for name in selected_modules}
            if selected_modules is not None
            else None
        )
        capped_modules = []
        failed_modules = []
        for name, module in model.named_modules():
            if selected_names is not None and name not in selected_names:
                continue
            if not should_process_module_fn(name, module, scope):
                continue
            try:
                weight = getattr(module, "weight", None)
                if not _is_matrix_weight(weight):
                    continue
                if is_quantized_weight(weight):
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
            model,
            cap_ratio=cap_ratio,
            scope=scope,
            baseline_sigmas=baseline_sigmas,
            selected_modules=policy.get("selected_modules"),
        )
        results["cap_result"] = cap_result
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
