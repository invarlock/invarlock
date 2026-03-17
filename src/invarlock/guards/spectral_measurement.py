from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._estimators import power_iter_sigma_max


def compute_sigma_max(
    weight_matrix: Any,
    *,
    iters: int = 4,
    init: str = "ones",
    power_iter_sigma_max_fn: Any | None = None,
) -> float:
    """Compute maximum singular value of a weight matrix."""
    if power_iter_sigma_max_fn is None:
        power_iter_sigma_max_fn = power_iter_sigma_max
    try:
        iters_i = int(iters)
    except Exception:
        iters_i = 4
    if iters_i < 1:
        iters_i = 1
    init_s = str(init or "ones").strip().lower()
    if init_s not in {"ones", "e0"}:
        init_s = "ones"

    if not isinstance(weight_matrix, torch.Tensor):
        return 1.0
    if weight_matrix.dtype in {torch.int8, torch.uint8}:
        return 1.0
    if weight_matrix.numel() == 0 or weight_matrix.ndim != 2:
        return 0.0

    try:
        return float(power_iter_sigma_max_fn(weight_matrix, iters=iters_i, init=init_s))
    except Exception:
        return 1.0


def auto_sigma_target(
    model: Any,
    percentile: float = 0.95,
    *,
    compute_sigma_max_fn: Any = compute_sigma_max,
) -> float:
    """Automatically determine sigma target for a model."""
    try:
        spectral_norms = []
        for _name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.ndim == 2:
                sigma = compute_sigma_max_fn(module.weight)
                if sigma > 0:
                    spectral_norms.append(sigma)
        if spectral_norms:
            return float(np.percentile(spectral_norms, percentile * 100))
        return percentile
    except Exception:
        return percentile


def capture_baseline_sigmas(
    model: Any,
    scope: str = "all",
    *,
    should_process_module_fn: Any | None = None,
    compute_sigma_max_fn: Any = compute_sigma_max,
    modules: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] | None = None,
) -> dict[str, float]:
    """Capture baseline singular values for model layers."""
    if should_process_module_fn is None:
        from .spectral_detection import (
            should_process_module as should_process_module_fn,
        )

    try:
        baseline_sigmas = {}
        module_iter = modules
        if module_iter is None:
            module_iter = tuple(model.named_modules())
        for name, module in module_iter:
            if not should_process_module_fn(name, module, scope):
                continue
            if hasattr(module, "weight") and module.weight.ndim == 2:
                baseline_sigmas[name] = compute_sigma_max_fn(module.weight)
        return baseline_sigmas
    except Exception:
        return {}


def scan_model_gains(
    model: Any,
    scope: str = "all",
    *,
    should_process_module_fn: Any | None = None,
    compute_sigma_max_fn: Any = compute_sigma_max,
    modules: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Scan model for gain values and spectral statistics."""
    if should_process_module_fn is None:
        from .spectral_detection import (
            should_process_module as should_process_module_fn,
        )

    results: dict[str, Any] = {
        "total_layers": 0,
        "scanned_modules": 0,
        "spectral_norms": [],
        "weight_statistics": {},
    }

    try:
        module_iter = modules
        if module_iter is None:
            module_iter = tuple(model.named_modules())
        for name, module in module_iter:
            results["total_layers"] += 1
            if should_process_module_fn(name, module, scope):
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    results["scanned_modules"] += 1
                    sigma_max = compute_sigma_max_fn(module.weight)
                    results["spectral_norms"].append(sigma_max)
                    try:
                        results["weight_statistics"][name] = {
                            "mean": module.weight.mean().item(),
                            "std": module.weight.std().item(),
                            "min": module.weight.min().item(),
                            "max": module.weight.max().item(),
                        }
                    except Exception:
                        pass

        if results["spectral_norms"]:
            results["mean_spectral_norm"] = np.mean(results["spectral_norms"])
            results["max_spectral_norm"] = np.max(results["spectral_norms"])
            results["min_spectral_norm"] = np.min(results["spectral_norms"])

        results["message"] = (
            f"Scanned {results['scanned_modules']} modules out of {results['total_layers']} total layers"
        )
        return results
    except Exception as error:
        return {
            "total_layers": int(results.get("total_layers", 0)),
            "scanned_modules": 0,
            "error": str(error),
            "message": f"Model scanning failed: {error}",
        }


def capture_sigmas(
    guard: Any,
    model: Any,
    *,
    phase: str,
    power_iter_sigma_max_fn: Any | None = None,
) -> dict[str, float]:
    """Capture σ̂max for each in-scope module under the measurement contract."""
    _ = phase
    if power_iter_sigma_max_fn is None:
        power_iter_sigma_max_fn = power_iter_sigma_max
    sigmas: dict[str, float] = {}
    try:
        iters = int((guard.estimator or {}).get("iters", 4) or 4)
    except Exception:
        iters = 4
    if iters < 1:
        iters = 1
    init = str((guard.estimator or {}).get("init", "ones") or "ones").strip().lower()
    if init not in {"ones", "e0"}:
        init = "ones"

    if hasattr(guard, "_get_scoped_modules"):
        module_iter = guard._get_scoped_modules(model)
    else:
        module_iter = tuple(
            (name, module)
            for name, module in model.named_modules()
            if guard._should_check_module(name, module)
        )

    for name, module in module_iter:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if weight.dtype in {torch.int8, torch.uint8}:
            sigmas[name] = 1.0
            continue
        try:
            sigmas[name] = float(
                power_iter_sigma_max_fn(weight, iters=iters, init=init)
            )
        except Exception:
            sigmas[name] = 1.0
    return sigmas


def compute_spectral_norms(model: Any, scope: str = "all") -> dict[str, float]:
    """Compatibility helper returning per-module spectral norms for a model."""
    from .spectral_detection import should_process_module

    return capture_baseline_sigmas(
        model,
        scope=scope,
        should_process_module_fn=should_process_module,
    )


__all__ = [
    "auto_sigma_target",
    "capture_baseline_sigmas",
    "capture_sigmas",
    "compute_spectral_norms",
    "compute_sigma_max",
    "scan_model_gains",
]
