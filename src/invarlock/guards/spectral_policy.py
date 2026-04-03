from __future__ import annotations

import math
from typing import Any

from invarlock.core.exceptions import ValidationError


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def default_family_caps() -> dict[str, dict[str, float]]:
    """Default per-family spectral z-score caps."""
    return {
        "ffn": {"kappa": 2.5},
        "attn": {"kappa": 2.8},
        "embed": {"kappa": 3.0},
        "other": {"kappa": 3.0},
    }


def normalize_family_caps(
    caps: Any, *, default: bool = True
) -> dict[str, dict[str, float]]:
    """Normalize family cap configuration into canonical mapping."""
    if not isinstance(caps, dict) or not caps:
        return default_family_caps() if default else {}

    normalized: dict[str, dict[str, float]] = {}
    for family, values in caps.items():
        entry: dict[str, float] = {}
        if isinstance(values, dict):
            for key, val in values.items():
                if _is_non_bool_number(val) and math.isfinite(float(val)):
                    entry[str(key)] = float(val)
        elif _is_non_bool_number(values) and math.isfinite(float(values)):
            entry["kappa"] = float(values)
        if entry:
            normalized[str(family)] = entry

    if normalized:
        return normalized
    return default_family_caps() if default else {}


def _policy_invalid(
    param: str, reason: str, *, value: Any | None = None
) -> ValidationError:
    details: dict[str, Any] = {"param": param, "reason": reason}
    if value is not None:
        details["value"] = value
    return ValidationError(
        code="E501",
        message="POLICY-PARAM-INVALID",
        details=details,
    )


def _require_policy_mapping(param: str, value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _policy_invalid(param, "must be a mapping")
    return value


def _require_policy_float(
    param: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise _policy_invalid(param, "must be a finite float", value=value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise _policy_invalid(param, "must be a finite float", value=value) from exc
    if not math.isfinite(numeric):
        raise _policy_invalid(param, "must be a finite float", value=value)
    if minimum is not None and numeric < minimum:
        raise _policy_invalid(param, f"must be >= {minimum}", value=value)
    if maximum is not None and numeric > maximum:
        raise _policy_invalid(param, f"must be <= {maximum}", value=value)
    return numeric


def _require_policy_int(param: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise _policy_invalid(param, "must be an integer", value=value)
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise _policy_invalid(param, "must be an integer", value=value) from exc
    if minimum is not None and integer < minimum:
        raise _policy_invalid(param, f"must be >= {minimum}", value=value)
    return integer


def normalize_multiple_testing_config(value: Any | None) -> dict[str, Any]:
    mt_policy = _require_policy_mapping("multiple_testing", value)
    method = str(mt_policy.get("method", "bh") or "bh").strip().lower()
    if method not in {"bh", "bonferroni"}:
        raise _policy_invalid("multiple_testing.method", "must be 'bh' or 'bonferroni'")
    alpha_value = mt_policy.get("alpha", 0.05)
    alpha = _require_policy_float(
        "multiple_testing.alpha", alpha_value, minimum=0.0, maximum=1.0
    )
    if alpha <= 0.0:
        raise _policy_invalid(
            "multiple_testing.alpha", "must be > 0", value=alpha_value
        )
    m_value = mt_policy.get("m", 4)
    m = _require_policy_int("multiple_testing.m", m_value, minimum=1)
    return {"method": method, "alpha": alpha, "m": m}


def multiple_testing_alpha(value: Any | None) -> float:
    return float(normalize_multiple_testing_config(value).get("alpha", 0.05))


def normalize_estimator_config(value: Any | None) -> dict[str, Any]:
    estimator_policy = _require_policy_mapping("estimator", value)
    iters = _require_policy_int(
        "estimator.iters", estimator_policy.get("iters", 4), minimum=1
    )
    init = str(estimator_policy.get("init", "ones") or "ones").strip().lower()
    if init not in {"ones", "e0"}:
        raise _policy_invalid("estimator.init", "must be 'ones' or 'e0'", value=init)
    return {"type": "power_iter", "iters": iters, "init": init}


def normalize_degeneracy_config(value: Any | None) -> dict[str, Any]:
    degeneracy_policy = _require_policy_mapping("degeneracy", value)
    stable_rank_cfg = _require_policy_mapping(
        "degeneracy.stable_rank", degeneracy_policy.get("stable_rank")
    )
    norm_collapse_cfg = _require_policy_mapping(
        "degeneracy.norm_collapse", degeneracy_policy.get("norm_collapse")
    )
    return {
        "enabled": bool(degeneracy_policy.get("enabled", True)),
        "stable_rank": {
            "warn_ratio": _require_policy_float(
                "degeneracy.stable_rank.warn_ratio",
                stable_rank_cfg.get("warn_ratio", 0.5),
                minimum=0.0,
            ),
            "fatal_ratio": _require_policy_float(
                "degeneracy.stable_rank.fatal_ratio",
                stable_rank_cfg.get("fatal_ratio", 0.25),
                minimum=0.0,
            ),
        },
        "norm_collapse": {
            "warn_ratio": _require_policy_float(
                "degeneracy.norm_collapse.warn_ratio",
                norm_collapse_cfg.get("warn_ratio", 0.25),
                minimum=0.0,
            ),
            "fatal_ratio": _require_policy_float(
                "degeneracy.norm_collapse.fatal_ratio",
                norm_collapse_cfg.get("fatal_ratio", 0.10),
                minimum=0.0,
            ),
        },
    }


def serialize_policy(guard: Any) -> dict[str, Any]:
    """Snapshot current guard policy for report serialization."""
    return {
        "scope": guard.scope,
        "sigma_quantile": float(guard.sigma_quantile),
        "deadband": float(guard.deadband),
        "max_caps": int(guard.max_caps),
        "max_spectral_norm": (
            float(guard.max_spectral_norm)
            if guard.max_spectral_norm is not None
            else None
        ),
        "family_caps": guard.family_caps,
        "multiple_testing": guard.multiple_testing,
        "estimator": guard.estimator,
        "degeneracy": guard.degeneracy,
        "correction_enabled": bool(guard.correction_enabled),
        "ignore_preview_inflation": bool(guard.ignore_preview_inflation),
    }


def apply_policy_overrides(guard: Any, policy: dict[str, Any]) -> None:
    """Hydrate a spectral guard from an override policy block."""
    sigma_value = policy.get("sigma_quantile")
    if "contraction" in policy or "kappa" in policy:
        raise ValueError(
            "Spectral policy keys 'contraction'/'kappa' are not supported; "
            "use 'sigma_quantile'."
        )
    if sigma_value is not None:
        guard.sigma_quantile = _require_policy_float(
            "sigma_quantile", sigma_value, minimum=0.0, maximum=1.0
        )
        policy["sigma_quantile"] = guard.sigma_quantile
    guard.config["sigma_quantile"] = guard.sigma_quantile

    if "deadband" in policy:
        guard.deadband = _require_policy_float(
            "deadband", policy["deadband"], minimum=0.0
        )
        guard.config["deadband"] = guard.deadband

    if "scope" in policy:
        guard.scope = policy["scope"]
        guard.config["scope"] = guard.scope

    if "max_spectral_norm" in policy:
        max_spectral_norm = policy["max_spectral_norm"]
        if max_spectral_norm is None:
            guard.max_spectral_norm = None
        else:
            guard.max_spectral_norm = _require_policy_float(
                "max_spectral_norm", max_spectral_norm
            )
        guard.config["max_spectral_norm"] = guard.max_spectral_norm

    if "correction_enabled" in policy:
        guard.correction_enabled = bool(policy["correction_enabled"])
        guard.config["correction_enabled"] = guard.correction_enabled

    if "max_caps" in policy:
        guard.max_caps = _require_policy_int("max_caps", policy["max_caps"], minimum=0)
        guard.config["max_caps"] = guard.max_caps

    if "family_caps" in policy:
        guard.family_caps = normalize_family_caps(policy["family_caps"], default=True)
        guard.config["family_caps"] = guard.family_caps

    if "ignore_preview_inflation" in policy:
        guard.ignore_preview_inflation = bool(policy["ignore_preview_inflation"])
        guard.config["ignore_preview_inflation"] = guard.ignore_preview_inflation

    if "baseline_family_stats" in policy and isinstance(
        policy["baseline_family_stats"], dict
    ):
        guard.baseline_family_stats = {
            family: stats.copy()
            for family, stats in policy["baseline_family_stats"].items()
            if isinstance(stats, dict)
        }
        guard.config["baseline_family_stats"] = guard.baseline_family_stats

    if "multipletesting" in policy:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={
                "param": "multipletesting",
                "hint": "Use spectral.multiple_testing instead.",
            },
        )

    mt_policy = policy.get("multiple_testing")
    if mt_policy is not None:
        guard.multiple_testing = normalize_multiple_testing_config(mt_policy)
        policy["multiple_testing"] = guard.multiple_testing
        guard.config["multiple_testing"] = guard.multiple_testing

    estimator_policy = policy.get("estimator")
    if estimator_policy is not None:
        guard.estimator = normalize_estimator_config(estimator_policy)
        guard.config["estimator"] = guard.estimator

    degeneracy_policy = policy.get("degeneracy")
    if degeneracy_policy is not None:
        guard.degeneracy = normalize_degeneracy_config(degeneracy_policy)
        guard.config["degeneracy"] = guard.degeneracy


__all__ = [
    "apply_policy_overrides",
    "default_family_caps",
    "multiple_testing_alpha",
    "normalize_degeneracy_config",
    "normalize_estimator_config",
    "normalize_family_caps",
    "normalize_multiple_testing_config",
    "serialize_policy",
]
