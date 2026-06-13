from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from ._estimators import frobenius_norm_sq, row_col_norm_extrema
from .quantized_weights import is_quantized_weight
from .spectral_measurement import compute_sigma_max
from .spectral_policy import default_family_caps

_SPECTRAL_CHECK_ERRORS = (
    ArithmeticError,
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def should_process_module(name: str, module: Any, scope: str) -> bool:
    """Determine if a module should be processed based on scope."""
    weight = getattr(module, "weight", None)
    if getattr(weight, "ndim", None) != 2:
        return False
    if scope == "all":
        return True
    if scope == "attn":
        return any(
            keyword in name.lower()
            for keyword in ["attn", "attention", "self_attn", "c_attn", "c_proj"]
        )
    if scope == "ffn":
        return any(
            keyword in name.lower()
            for keyword in ["mlp", "ffn", "feed_forward", "fc", "c_fc"]
        )
    if scope == "ffn+proj":
        lname = name.lower()
        return any(
            keyword in lname
            for keyword in [
                "mlp",
                "ffn",
                "feed_forward",
                "fc",
                "c_fc",
                "c_proj",
                "projection",
            ]
        )
    return True


def classify_module_family(name: str, module: Any) -> str:
    """Classify module into a spectral family for policy purposes."""
    lname = name.lower()
    if any(
        tok in lname
        for tok in (
            "gate_up_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "c_fc",
            "fc1",
            "fc2",
        )
    ):
        return "ffn"
    if any(
        tok in lname
        for tok in ("router", "routing", "gate", "gating", "dispatch", "switch")
    ):
        return "router"
    if any(tok in lname for tok in ("experts", "expert", "moe", "mixture_of_experts")):
        return "expert_ffn"
    if "mlp" in lname or "ffn" in lname or "feed_forward" in lname:
        return "ffn"
    if (
        "attn" in lname
        or "attention" in lname
        or any(
            token in lname
            for token in ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn"]
        )
    ):
        return "attn"
    if "embed" in lname or "wte" in lname or "embedding" in lname:
        return "embed"

    module_type = module.__class__.__name__.lower()
    if "embedding" in module_type:
        return "embed"
    return "other"


def classify_model_families(
    model: Any,
    scope: str = "all",
    existing: dict[str, str] | None = None,
    *,
    modules: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] | None = None,
    should_process_module_fn: Any = should_process_module,
    classify_module_family_fn: Any = classify_module_family,
) -> dict[str, str]:
    """Build or update a module→family map for the provided model."""
    family_map = dict(existing) if existing else {}
    module_iter = modules
    if module_iter is None:
        module_iter = tuple(model.named_modules())
    for name, module in module_iter:
        if should_process_module_fn(name, module, scope):
            family_map[name] = classify_module_family_fn(name, module)
    return family_map


def summarize_sigmas(sigmas: dict[str, float]) -> dict[str, float]:
    """Compute summary statistics for a sigma dictionary."""
    if not sigmas:
        return {
            "max_spectral_norm": 0.0,
            "mean_spectral_norm": 0.0,
            "min_spectral_norm": 0.0,
        }
    values = np.array(list(sigmas.values()), dtype=float)
    return {
        "max_spectral_norm": float(values.max()),
        "mean_spectral_norm": float(values.mean()),
        "min_spectral_norm": float(values.min()),
    }


def compute_z_score_for_value(
    sigma: float,
    family_stats: dict[str, float],
    fallback_value: float,
    deadband: float,
) -> float:
    """Compute per-family z-score for a spectral norm with sensible fallbacks."""
    mean = float(family_stats.get("mean", 0.0) or 0.0)
    std = float(family_stats.get("std", 0.0) or 0.0)
    if std > 0:
        return float((sigma - mean) / std)

    denom = fallback_value if fallback_value > 0 else 1.0
    rel_change = (sigma / denom) - 1.0
    if abs(rel_change) <= deadband:
        return 0.0
    scale = deadband if deadband > 0 else 1.0
    return float(rel_change / scale)


def compute_z_scores(
    metrics: dict[str, float],
    baseline_family_stats: dict[str, dict[str, float]],
    module_family_map: dict[str, str],
    baseline_sigmas: dict[str, float],
    deadband: float,
) -> dict[str, float]:
    """Compute z-scores for all modules given baseline family stats."""
    z_scores: dict[str, float] = {}
    for name, sigma in metrics.items():
        family = module_family_map.get(name, "other")
        family_stats = baseline_family_stats.get(family, {})
        fallback_value = baseline_sigmas.get(name, family_stats.get("mean", sigma))
        z_scores[name] = compute_z_score_for_value(
            float(sigma), family_stats, float(fallback_value), deadband=deadband
        )
    return z_scores


def summarize_family_z_scores(
    z_scores: dict[str, float],
    module_family_map: dict[str, str],
    family_caps: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Summarize z-scores per family, including violation counts."""
    family_values: dict[str, list[float]] = defaultdict(list)
    for name, z_score in z_scores.items():
        family = module_family_map.get(name, "other")
        family_values[family].append(float(z_score))

    summary: dict[str, dict[str, float]] = {}
    for family, values in family_values.items():
        arr = np.array(values, dtype=float)
        cap = family_caps.get(family, {}).get("kappa")
        violations = int(np.sum(arr > float(cap))) if cap is not None else 0
        summary[family] = {
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "count": len(values),
            "violations": violations,
        }
        if cap is not None:
            summary[family]["kappa"] = float(cap)
    return summary


def compute_family_stats(
    sigmas: dict[str, float], family_map: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Compute per-family statistics (mean/std/min/max/count)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for name, sigma in sigmas.items():
        family = family_map.get(name, "other")
        buckets[family].append(float(sigma))

    stats: dict[str, dict[str, float]] = {}
    for family, values in buckets.items():
        arr = np.array(values, dtype=float)
        stats[family] = {
            "count": len(values),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return stats


def should_check_module(guard: Any, name: str, module: Any) -> bool:
    """Determine if a module should be checked based on the guard scope."""
    weight = getattr(module, "weight", None)
    if getattr(weight, "ndim", None) != 2:
        return False
    if guard.scope == "all":
        return True
    if guard.scope == "attn":
        return any(
            keyword in name.lower() for keyword in ["attn", "attention", "self_attn"]
        )
    if guard.scope == "ffn":
        return any(
            keyword in name.lower() for keyword in ["mlp", "ffn", "feed_forward", "fc"]
        )
    return True


def detect_spectral_violations(
    guard: Any,
    model: Any,
    metrics: dict[str, float],
    phase: str = "finalize",
    *,
    compute_sigma_max_fn: Any | None = None,
    classify_module_family_fn: Any | None = None,
    compute_z_score_for_value_fn: Any | None = None,
    default_family_caps_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """Detect spectral property violations using per-family z-score caps."""
    if compute_sigma_max_fn is None:
        compute_sigma_max_fn = compute_sigma_max
    if classify_module_family_fn is None:
        classify_module_family_fn = classify_module_family
    if compute_z_score_for_value_fn is None:
        compute_z_score_for_value_fn = compute_z_score_for_value
    if default_family_caps_fn is None:
        default_family_caps_fn = default_family_caps

    violations: list[dict[str, Any]] = []
    latest_z: dict[str, float] = {}

    for name, module in guard._get_scoped_modules(model):
        try:
            weight = getattr(module, "weight", None)
            if getattr(weight, "ndim", None) == 2:
                sigma_max = metrics.get(name)
                if sigma_max is None:
                    if is_quantized_weight(weight):
                        guard._log_event(
                            "spectral_quantized_weight_unmeasurable",
                            level="WARN",
                            message=(
                                "Skipping spectral z-score enforcement for "
                                "quantized weight without a dense matrix view"
                            ),
                            module=name,
                            phase=phase,
                            dtype=str(getattr(weight, "dtype", "unknown")),
                        )
                        continue
                    sigma_max = compute_sigma_max_fn(weight)

                baseline_sigma = guard.baseline_sigmas.get(name, guard.target_sigma)
                family = guard.module_family_map.get(name)
                if family is None:
                    family = classify_module_family_fn(name, module)
                    guard.module_family_map[name] = family

                family_stats = guard.baseline_family_stats.get(family, {})
                cap_config = guard.family_caps.get(family, {})
                kappa_raw = (
                    cap_config.get("kappa") if isinstance(cap_config, dict) else None
                )
                if not (
                    isinstance(kappa_raw, int | float)
                    and math.isfinite(float(kappa_raw))
                ):
                    other_cfg = guard.family_caps.get("other", {})
                    kappa_raw = (
                        other_cfg.get("kappa") if isinstance(other_cfg, dict) else None
                    )
                if not (
                    isinstance(kappa_raw, int | float)
                    and math.isfinite(float(kappa_raw))
                ):
                    kappa_raw = (
                        default_family_caps_fn().get("other", {}).get("kappa", 3.0)
                    )
                kappa_cap = float(kappa_raw)

                z_score = compute_z_score_for_value_fn(
                    sigma_max,
                    family_stats,
                    fallback_value=baseline_sigma,
                    deadband=guard.deadband,
                )
                latest_z[name] = z_score

                if guard.ignore_preview_inflation and phase == "after_edit":
                    continue

                if abs(z_score) > kappa_cap:
                    violations.append(
                        {
                            "type": "family_z_cap",
                            "severity": "budgeted",
                            "module": name,
                            "family": family,
                            "z_score": float(z_score),
                            "kappa": kappa_cap,
                            "sigma": float(sigma_max),
                            "baseline_sigma": float(baseline_sigma),
                            "message": f"Family '{family}' z-score {z_score:.2f} exceeds cap {kappa_cap:.2f}",
                        }
                    )

                if (
                    guard.max_spectral_norm is not None
                    and sigma_max > guard.max_spectral_norm
                ):
                    threshold = float(guard.max_spectral_norm)
                    violations.append(
                        {
                            "type": "max_spectral_norm",
                            "severity": "fatal",
                            "module": name,
                            "family": family,
                            "current_sigma": float(sigma_max),
                            "threshold": threshold,
                            "message": f"Spectral norm {sigma_max:.3f} exceeds maximum {threshold}",
                        }
                    )

                if bool((guard.degeneracy or {}).get("enabled")):
                    base = guard.baseline_degeneracy.get(name) or {}
                    base_sr = base.get("stable_rank")
                    base_nc = base.get("norm_collapse")
                    eps = 1e-12
                    try:
                        sr_cfg = (
                            (guard.degeneracy.get("stable_rank") or {})
                            if isinstance(guard.degeneracy, dict)
                            else {}
                        )
                        nc_cfg = (
                            (guard.degeneracy.get("norm_collapse") or {})
                            if isinstance(guard.degeneracy, dict)
                            else {}
                        )
                        sr_warn = float(sr_cfg.get("warn_ratio", 0.5))
                        sr_fatal = float(sr_cfg.get("fatal_ratio", 0.25))
                        nc_warn = float(nc_cfg.get("warn_ratio", 0.25))
                        nc_fatal = float(nc_cfg.get("fatal_ratio", 0.10))
                    except _SPECTRAL_CHECK_ERRORS:
                        sr_warn, sr_fatal, nc_warn, nc_fatal = 0.5, 0.25, 0.25, 0.10

                    if (
                        isinstance(base_sr, int | float)
                        and math.isfinite(float(base_sr))
                        and float(base_sr) > 0
                    ):
                        try:
                            sr_cur = frobenius_norm_sq(module.weight) / max(
                                float(sigma_max) ** 2, eps
                            )
                            sr_ratio = float(sr_cur) / max(float(base_sr), eps)
                            if math.isfinite(sr_ratio) and sr_ratio < sr_warn:
                                violations.append(
                                    {
                                        "type": "degeneracy_stable_rank_drop",
                                        "severity": "fatal"
                                        if sr_ratio < sr_fatal
                                        else "budgeted",
                                        "module": name,
                                        "family": family,
                                        "stable_rank_base": float(base_sr),
                                        "stable_rank_cur": float(sr_cur),
                                        "ratio": float(sr_ratio),
                                        "warn_ratio": float(sr_warn),
                                        "fatal_ratio": float(sr_fatal),
                                        "message": (
                                            f"Stable-rank ratio {sr_ratio:.3f} below {sr_warn:.3f} "
                                            f"(base={float(base_sr):.3f}, cur={float(sr_cur):.3f})"
                                        ),
                                    }
                                )
                        except _SPECTRAL_CHECK_ERRORS:
                            pass

                    if (
                        isinstance(base_nc, int | float)
                        and math.isfinite(float(base_nc))
                        and float(base_nc) > 0
                    ):
                        try:
                            norms = row_col_norm_extrema(module.weight, eps=eps)
                            row_med = max(float(norms.get("row_median", 0.0)), eps)
                            col_med = max(float(norms.get("col_median", 0.0)), eps)
                            nc_cur = min(
                                float(norms.get("row_min", 0.0)) / row_med,
                                float(norms.get("col_min", 0.0)) / col_med,
                            )
                            nc_ratio = float(nc_cur) / max(float(base_nc), eps)
                            if math.isfinite(nc_ratio) and nc_ratio < nc_warn:
                                violations.append(
                                    {
                                        "type": "degeneracy_norm_collapse",
                                        "severity": "fatal"
                                        if nc_ratio < nc_fatal
                                        else "budgeted",
                                        "module": name,
                                        "family": family,
                                        "norm_collapse_base": float(base_nc),
                                        "norm_collapse_cur": float(nc_cur),
                                        "ratio": float(nc_ratio),
                                        "warn_ratio": float(nc_warn),
                                        "fatal_ratio": float(nc_fatal),
                                        "message": (
                                            f"Norm-collapse ratio {nc_ratio:.3f} below {nc_warn:.3f} "
                                            f"(base={float(base_nc):.3e}, cur={float(nc_cur):.3e})"
                                        ),
                                    }
                                )
                        except _SPECTRAL_CHECK_ERRORS:
                            pass
        except _SPECTRAL_CHECK_ERRORS as error:
            guard._log_event(
                "violation_check_error",
                level="WARN",
                message=f"Failed to check module {name}: {str(error)}",
                module=name,
                error=str(error),
            )

    guard.latest_z_scores = latest_z
    return violations


__all__ = [
    "classify_model_families",
    "classify_module_family",
    "compute_family_stats",
    "compute_z_score_for_value",
    "compute_z_scores",
    "detect_spectral_violations",
    "should_check_module",
    "should_process_module",
    "summarize_family_z_scores",
    "summarize_sigmas",
]
