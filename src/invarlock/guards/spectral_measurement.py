from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ._estimators import power_iter_sigma_max

_SPECTRAL_MEASUREMENT_ERRORS = (RuntimeError, TypeError, ValueError)


def _is_real_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_matrix_weight(weight: Any) -> bool:
    if weight is None:
        return False
    try:
        return int(getattr(weight, "ndim", 0) or 0) == 2
    except (TypeError, ValueError):
        return False


def _is_quantized_weight(weight: Any) -> bool:
    return getattr(weight, "dtype", None) in {torch.int8, torch.uint8}


def _scalarize_stat(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _append_measurement_diagnostic(
    diagnostics: list[dict[str, Any]] | None,
    *,
    kind: str,
    severity: str,
    message: str,
    module_name: str | None = None,
    fallback_value: float,
    **details: Any,
) -> None:
    if diagnostics is None:
        return
    payload: dict[str, Any] = {
        "kind": kind,
        "severity": severity,
        "message": message,
        "fallback_value": fallback_value,
    }
    if module_name is not None:
        payload["module"] = module_name
    payload.update(details)
    diagnostics.append(payload)


def _record_guard_measurement_diagnostics(
    guard: Any,
    diagnostics: list[dict[str, Any]],
    *,
    phase: str,
) -> None:
    if not diagnostics:
        return
    store = getattr(guard, "_measurement_diagnostics", None)
    for diagnostic in diagnostics:
        record = dict(diagnostic)
        record["phase"] = phase
        if isinstance(store, list):
            store.append(record)
        if hasattr(guard, "_log_event"):
            severity = str(record.get("severity", "warning")).lower()
            level = "ERROR" if severity == "error" else "WARN"
            if severity in {"info", "debug"}:
                level = severity.upper()
            details = {
                key: value
                for key, value in record.items()
                if key not in {"kind", "severity", "message"}
            }
            guard._log_event(
                str(record.get("kind", "spectral_measurement_fallback")),
                level=level,
                message=str(record.get("message", "")),
                **details,
            )


def _compute_sigma_with_optional_diagnostics(
    compute_sigma_max_fn: Any,
    weight: Any,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    module_name: str | None = None,
) -> float:
    if compute_sigma_max_fn is compute_sigma_max:
        return float(
            compute_sigma_max_fn(
                weight,
                diagnostics=diagnostics,
                module_name=module_name,
            )
        )
    try:
        return float(compute_sigma_max_fn(weight))
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_custom_estimator_error",
            severity="error",
            message="Custom spectral sigma measurement failed; using neutral fallback.",
            module_name=module_name,
            fallback_value=1.0,
            error=str(exc),
        )
        return 1.0


def compute_sigma_max(
    weight_matrix: Any,
    *,
    iters: int = 4,
    init: str = "ones",
    power_iter_sigma_max_fn: Any | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    module_name: str | None = None,
) -> float:
    """Compute maximum singular value of a weight matrix."""
    if power_iter_sigma_max_fn is None:
        power_iter_sigma_max_fn = power_iter_sigma_max
    try:
        iters_i = int(iters)
    except (TypeError, ValueError):
        iters_i = 4
    if iters_i < 1:
        iters_i = 1
    init_s = str(init or "ones").strip().lower()
    if init_s not in {"ones", "e0"}:
        init_s = "ones"

    if not isinstance(weight_matrix, torch.Tensor):
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_non_tensor",
            severity="error",
            message="Spectral sigma measurement received a non-tensor weight.",
            module_name=module_name,
            fallback_value=1.0,
            observed_type=type(weight_matrix).__name__,
        )
        return 1.0
    if weight_matrix.dtype in {torch.int8, torch.uint8}:
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_quantized_weight",
            severity="warning",
            message="Spectral sigma measurement skipped a quantized weight.",
            module_name=module_name,
            fallback_value=1.0,
            dtype=str(weight_matrix.dtype),
        )
        return 1.0
    if weight_matrix.numel() == 0 or weight_matrix.ndim != 2:
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_invalid_shape",
            severity="warning",
            message="Spectral sigma measurement received an empty or non-matrix weight.",
            module_name=module_name,
            fallback_value=0.0,
            shape=tuple(weight_matrix.shape),
        )
        return 0.0
    try:
        finite = bool(torch.isfinite(weight_matrix).all().item())
    except _SPECTRAL_MEASUREMENT_ERRORS:
        finite = True
    if not finite:
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_non_finite_weight",
            severity="error",
            message="Spectral sigma measurement found non-finite weight values.",
            module_name=module_name,
            fallback_value=0.0,
            dtype=str(weight_matrix.dtype),
            shape=tuple(weight_matrix.shape),
        )
        return 0.0

    try:
        sigma = float(
            power_iter_sigma_max_fn(weight_matrix, iters=iters_i, init=init_s)
        )
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_estimator_error",
            severity="error",
            message="Spectral sigma estimator failed; using neutral fallback.",
            module_name=module_name,
            fallback_value=1.0,
            error=str(exc),
        )
        return 1.0
    if not math.isfinite(sigma):
        _append_measurement_diagnostic(
            diagnostics,
            kind="spectral_sigma_fallback_non_finite_estimate",
            severity="error",
            message="Spectral sigma estimator returned a non-finite value.",
            module_name=module_name,
            fallback_value=1.0,
            observed_value=str(sigma),
        )
        return 1.0
    return sigma


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
            weight = getattr(module, "weight", None)
            if not _is_matrix_weight(weight) or _is_quantized_weight(weight):
                continue
            sigma = compute_sigma_max_fn(weight)
            if _is_real_number(sigma) and sigma > 0:
                spectral_norms.append(float(sigma))
        if spectral_norms:
            try:
                return float(np.percentile(spectral_norms, percentile * 100))
            except _SPECTRAL_MEASUREMENT_ERRORS:
                return percentile
        return percentile
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
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
        from .spectral_detection import should_process_module

        should_process = should_process_module
    else:
        should_process = should_process_module_fn

    try:
        baseline_sigmas = {}
        module_iter = modules
        if module_iter is None:
            module_iter = tuple(model.named_modules())
        for name, module in module_iter:
            if not should_process(name, module, scope):
                continue
            weight = getattr(module, "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or not _is_matrix_weight(weight)
                or _is_quantized_weight(weight)
            ):
                continue
            baseline_sigmas[name] = _compute_sigma_with_optional_diagnostics(
                compute_sigma_max_fn,
                weight,
                module_name=name,
            )
        return baseline_sigmas
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
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
        from .spectral_detection import should_process_module

        should_process = should_process_module
    else:
        should_process = should_process_module_fn

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
            if should_process(name, module, scope):
                weight = getattr(module, "weight", None)
                if (
                    not isinstance(weight, torch.Tensor)
                    or not _is_matrix_weight(weight)
                    or _is_quantized_weight(weight)
                ):
                    continue
                results["scanned_modules"] += 1
                diagnostics: list[dict[str, Any]] = []
                sigma_max = _compute_sigma_with_optional_diagnostics(
                    compute_sigma_max_fn,
                    weight,
                    diagnostics=diagnostics,
                    module_name=name,
                )
                results["spectral_norms"].append(sigma_max)
                if diagnostics:
                    results.setdefault("diagnostics", []).extend(diagnostics)
                try:
                    results["weight_statistics"][name] = {
                        "mean": _scalarize_stat(weight.mean()),
                        "std": _scalarize_stat(weight.std()),
                        "min": _scalarize_stat(weight.min()),
                        "max": _scalarize_stat(weight.max()),
                    }
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

        if results["spectral_norms"]:
            results["mean_spectral_norm"] = np.mean(results["spectral_norms"])
            results["max_spectral_norm"] = np.max(results["spectral_norms"])
            results["min_spectral_norm"] = np.min(results["spectral_norms"])

        results["message"] = (
            f"Scanned {results['scanned_modules']} modules out of {results['total_layers']} total layers"
        )
        return results
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
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
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
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
        diagnostics: list[dict[str, Any]] = []
        sigmas[name] = compute_sigma_max(
            weight,
            iters=iters,
            init=init,
            power_iter_sigma_max_fn=power_iter_sigma_max_fn,
            diagnostics=diagnostics,
            module_name=name,
        )
        _record_guard_measurement_diagnostics(guard, diagnostics, phase=phase)
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
