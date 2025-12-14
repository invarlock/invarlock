# mypy: ignore-errors
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from invarlock.core.auto_tuning import get_tier_policies, resolve_tier_policies

from .report_types import RunReport


def _compute_variance_policy_digest(policy: dict[str, Any]) -> str:
    """Compute a stable digest for the variance guard policy knobs."""
    canonical_keys = [
        "deadband",
        "min_abs_adjust",
        "max_scale_step",
        "min_effect_lognll",
        "predictive_one_sided",
        "topk_backstop",
        "max_adjusted_modules",
    ]
    canonical_payload = {
        key: policy.get(key) for key in canonical_keys if key in policy
    }
    if not canonical_payload:
        return ""
    serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _compute_thresholds_payload(
    tier: str, resolved_policy: dict[str, Any]
) -> dict[str, Any]:
    """Build canonical thresholds payload for digest stability."""
    from .certificate import TIER_RATIO_LIMITS  # local to avoid cycles

    tier_lc = (tier or "balanced").lower()
    metrics_policy = (
        resolved_policy.get("metrics", {}) if isinstance(resolved_policy, dict) else {}
    )
    if not isinstance(metrics_policy, dict):
        metrics_policy = {}

    pm_policy = metrics_policy.get("pm_ratio", {})
    if not isinstance(pm_policy, dict):
        pm_policy = {}

    acc_policy = metrics_policy.get("accuracy", {})
    if not isinstance(acc_policy, dict):
        acc_policy = {}

    ratio_limit_base = pm_policy.get("ratio_limit_base")
    try:
        if ratio_limit_base is not None:
            ratio_limit_base = float(ratio_limit_base)
    except Exception:
        ratio_limit_base = None
    if ratio_limit_base is None:
        tier_defaults = get_tier_policies().get(tier_lc, {})
        fallback_pm = (
            (tier_defaults.get("metrics") or {}).get("pm_ratio")
            if isinstance(tier_defaults, dict)
            else {}
        )
        ratio_limit_base = float(
            (fallback_pm or {}).get(
                "ratio_limit_base", TIER_RATIO_LIMITS.get(tier_lc, 1.10)
            )
            if isinstance(fallback_pm, dict)
            else TIER_RATIO_LIMITS.get(tier_lc, 1.10)
        )
    variance_policy = (
        resolved_policy.get("variance", {}) if isinstance(resolved_policy, dict) else {}
    )

    payload = {
        "tier": tier_lc,
        "pm_ratio": {
            "ratio_limit_base": ratio_limit_base,
            "min_tokens": int(pm_policy.get("min_tokens", 0) or 0),
            "min_token_fraction": float(
                pm_policy.get("min_token_fraction", 0.0) or 0.0
            ),
            "hysteresis_ratio": float(pm_policy.get("hysteresis_ratio", 0.0) or 0.0),
        },
        "accuracy": {
            "delta_min_pp": float(acc_policy.get("delta_min_pp", -1.0) or -1.0),
            "min_examples": int(acc_policy.get("min_examples", 200) or 200),
            "min_examples_fraction": float(
                acc_policy.get("min_examples_fraction", 0.0) or 0.0
            ),
            "hysteresis_delta_pp": float(
                acc_policy.get("hysteresis_delta_pp", 0.0) or 0.0
            ),
        },
        "variance": {
            "min_effect_lognll": float(
                variance_policy.get("min_effect_lognll", 0.0) or 0.0
            )
        },
    }
    return payload


def _compute_thresholds_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _promote_legacy_multiple_testing_key(payload: dict[str, Any]) -> None:
    """Promote legacy 'multipletesting' to 'multiple_testing' in-place if present."""
    try:
        legacy_mt = payload.pop("multipletesting", None)
        if legacy_mt is not None and "multiple_testing" not in payload:
            payload["multiple_testing"] = legacy_mt
    except Exception:
        pass


def _resolve_policy_tier(report: RunReport) -> str:
    """Resolve the policy tier from report metadata or context."""
    tier: Any = None
    try:
        meta = report.get("meta", {}) if isinstance(report, dict) else {}
        auto_cfg: dict[str, Any] | None = (
            meta.get("auto", {}) if isinstance(meta, dict) else {}
        )
        tier = auto_cfg.get("tier") or meta.get("policy_tier")
        if not tier:
            context = report.get("context", {}) if isinstance(report, dict) else {}
            if isinstance(context, dict):
                tier = context.get("policy_tier") or (
                    context.get("auto", {})
                    if isinstance(context.get("auto"), dict)
                    else {}
                ).get("tier")
        if not tier:
            tier = "balanced"
        return str(tier).lower()
    except Exception:
        return "balanced"


def _format_family_caps(caps: Any) -> dict[str, dict[str, float]]:
    formatted: dict[str, dict[str, float]] = {}
    if isinstance(caps, dict):
        for family, data in caps.items():
            family_name = str(family)
            if isinstance(data, dict):
                kappa_val = data.get("kappa")
                if isinstance(kappa_val, int | float):
                    try:
                        formatted[family_name] = {"kappa": float(kappa_val)}
                    except Exception:
                        pass
            elif isinstance(data, int | float):
                try:
                    formatted[family_name] = {"kappa": float(data)}
                except Exception:
                    pass
    return formatted


def _format_epsilon_map(epsilon_map: Any) -> dict[str, float]:
    formatted: dict[str, float] = {}
    if isinstance(epsilon_map, dict):
        for family, value in epsilon_map.items():
            if isinstance(value, int | float):
                try:
                    formatted[str(family)] = float(value)
                except Exception:
                    pass
    return formatted


def _build_resolved_policies(
    tier: str,
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    variance: dict[str, Any],
    *,
    profile: str | None = None,
    explicit_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge tier defaults with observed policies to surface the resolved configuration."""
    tier_key = (tier or "balanced").lower()
    if tier_key == "none":
        tier_key = "balanced"
    base = resolve_tier_policies(
        tier_key, edit_name=None, explicit_overrides=explicit_overrides, profile=profile
    )

    resolved: dict[str, Any] = {}

    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    # Spectral guard
    base_spectral = (
        copy.deepcopy(base.get("spectral", {})) if isinstance(base, dict) else {}
    )
    spectral_resolved: dict[str, Any] = {}
    if isinstance(base_spectral, dict):
        spectral_resolved.update(base_spectral)
    spectral_caps = spectral.get("family_caps") or spectral_resolved.get("family_caps")
    from .policy_utils import _format_family_caps as _ffc  # self import safe

    spectral_resolved["family_caps"] = _ffc(spectral_caps)
    # Prefer observed policy sigma_quantile (accepting legacy aliases), then fallback
    pol_sq = None
    try:
        pol_sq = (spectral.get("policy", {}) or {}).get("sigma_quantile")
        if pol_sq is None:
            # Legacy aliases
            pol_sq = (spectral.get("policy", {}) or {}).get("contraction") or (
                spectral.get("policy", {}) or {}
            ).get("kappa")
    except Exception:
        pol_sq = None
    spectral_resolved["sigma_quantile"] = _safe_float(
        pol_sq
        if pol_sq is not None
        else spectral.get(
            "sigma_quantile", spectral_resolved.get("sigma_quantile", 0.95)
        ),
        0.95,
    )
    spectral_resolved["deadband"] = _safe_float(
        spectral.get("deadband", spectral_resolved.get("deadband", 0.1)), 0.1
    )
    max_caps_val = spectral.get("max_caps", spectral_resolved.get("max_caps", 5))
    spectral_resolved["max_caps"] = _safe_int(
        max_caps_val, base_spectral.get("max_caps", 5) or 5
    )
    if "ignore_preview_inflation" in base_spectral:
        spectral_resolved["ignore_preview_inflation"] = base_spectral[
            "ignore_preview_inflation"
        ]
    spectral_resolved["scope"] = (
        spectral.get("policy", {}).get("scope")
        or spectral_resolved.get("scope")
        or "all"
    )
    mt_source = spectral.get("multiple_testing") or spectral_resolved.get(
        "multiple_testing"
    )
    if isinstance(mt_source, dict):
        mt_entry = {
            "method": mt_source.get("method"),
            "alpha": _safe_float(mt_source.get("alpha", 0.05), 0.05),
        }
        m_val = mt_source.get("m")
        mt_entry["m"] = _safe_int(m_val, len(spectral_resolved["family_caps"] or {}))
        spectral_resolved["multiple_testing"] = mt_entry
    correction_flag = spectral.get("policy", {}).get(
        "correction_enabled",
        spectral_resolved.get("correction_enabled", False),
    )
    spectral_resolved["correction_enabled"] = bool(correction_flag)
    if tier_key == "balanced":
        spectral_resolved["correction_enabled"] = False
        spectral_resolved["max_spectral_norm"] = None
    else:
        spectral_resolved["max_spectral_norm"] = spectral.get("policy", {}).get(
            "max_spectral_norm", spectral_resolved.get("max_spectral_norm")
        )
    resolved["spectral"] = spectral_resolved

    # RMT guard
    base_rmt = copy.deepcopy(base.get("rmt", {})) if isinstance(base, dict) else {}
    rmt_resolved: dict[str, Any] = {}
    if isinstance(base_rmt, dict):
        rmt_resolved.update(base_rmt)
    rmt_resolved["margin"] = _safe_float(
        rmt.get("margin", rmt_resolved.get("margin", 1.5)), 1.5
    )
    rmt_resolved["deadband"] = _safe_float(
        rmt.get("deadband", rmt_resolved.get("deadband", 0.1)), 0.1
    )
    epsilon_default_val = rmt.get(
        "epsilon_default", rmt_resolved.get("epsilon_default", 0.1)
    )
    rmt_resolved["epsilon_default"] = _safe_float(epsilon_default_val, 0.1)
    from .policy_utils import _format_epsilon_map as _fem

    epsilon_map = _fem(rmt.get("epsilon_by_family") or rmt_resolved.pop("epsilon", {}))
    if epsilon_map:
        rmt_resolved["epsilon_by_family"] = epsilon_map
    else:
        rmt_resolved.pop("epsilon", None)
    if "epsilon" in rmt_resolved:
        rmt_resolved.pop("epsilon", None)
    if "correct" in rmt_resolved:
        rmt_resolved["correct"] = bool(rmt_resolved["correct"])
    resolved["rmt"] = rmt_resolved

    # Variance guard
    base_variance = (
        copy.deepcopy(base.get("variance", {})) if isinstance(base, dict) else {}
    )
    variance_resolved: dict[str, Any] = {}
    if isinstance(base_variance, dict):
        variance_resolved.update(base_variance)

    observed_variance_policy = (
        variance.get("policy") if isinstance(variance, dict) else None
    )
    if isinstance(observed_variance_policy, dict) and observed_variance_policy:
        for key in (
            "deadband",
            "min_abs_adjust",
            "max_scale_step",
            "min_effect_lognll",
            "predictive_one_sided",
            "topk_backstop",
            "max_adjusted_modules",
            "tap",
            "predictive_gate",
            "scope",
            "clamp",
            "min_gain",
            "min_rel_gain",
            "max_calib",
            "seed",
            "mode",
            "alpha",
            "tie_breaker_deadband",
            "calibration",
        ):
            if (
                key in observed_variance_policy
                and observed_variance_policy.get(key) is not None
            ):
                variance_resolved[key] = observed_variance_policy.get(key)
    predictive_gate = variance.get("predictive_gate", {})
    predictive_one_sided = variance_resolved.get("predictive_one_sided")
    if isinstance(predictive_gate, dict) and "sided" in predictive_gate:
        predictive_one_sided = predictive_gate.get("sided") in (True, "one_sided")
    variance_resolved["predictive_one_sided"] = bool(
        predictive_one_sided if predictive_one_sided is not None else True
    )
    variance_resolved["min_effect_lognll"] = _safe_float(
        variance_resolved.get("min_effect_lognll", 0.0), 0.0
    )
    if "topk_backstop" in variance_resolved:
        variance_resolved["topk_backstop"] = _safe_int(
            variance_resolved.get("topk_backstop", 0), 0
        )
    variance_resolved["max_adjusted_modules"] = _safe_int(
        variance_resolved.get("max_adjusted_modules", 0), 0
    )
    if "deadband" in variance_resolved:
        variance_resolved["deadband"] = _safe_float(
            variance_resolved.get("deadband", 0.0), 0.0
        )
    if "min_abs_adjust" in variance_resolved:
        variance_resolved["min_abs_adjust"] = _safe_float(
            variance_resolved.get("min_abs_adjust", 0.0), 0.0
        )
    if "max_scale_step" in variance_resolved:
        variance_resolved["max_scale_step"] = _safe_float(
            variance_resolved.get("max_scale_step", 0.0), 0.0
        )
    resolved["variance"] = variance_resolved

    # Metric gates (PM ratio, accuracy, confidence, etc.)
    try:
        metrics = base.get("metrics", {}) if isinstance(base, dict) else {}
        if isinstance(metrics, dict) and metrics:
            resolved["metrics"] = copy.deepcopy(metrics)
    except Exception:
        pass

    # Confidence thresholds (optional policy knobs)
    try:
        conf = None
        metrics = (
            resolved.get("metrics")
            if isinstance(resolved.get("metrics"), dict)
            else None
        )
        if isinstance(metrics, dict):
            conf = metrics.get("confidence")
        if isinstance(conf, dict) and conf:
            resolved["confidence"] = {}
            if "ppl_ratio_width_max" in conf:
                try:
                    resolved["confidence"]["ppl_ratio_width_max"] = float(
                        conf.get("ppl_ratio_width_max")
                    )
                except Exception:
                    pass
            if "accuracy_delta_pp_width_max" in conf:
                try:
                    resolved["confidence"]["accuracy_delta_pp_width_max"] = float(
                        conf.get("accuracy_delta_pp_width_max")
                    )
                except Exception:
                    pass
    except Exception:
        pass

    return resolved


def _extract_effective_policies(report: RunReport) -> dict[str, Any]:
    """Extract the effective policies that were applied during the run."""
    policies: dict[str, Any] = {}

    guard_entries: list[dict[str, Any]] = report.get("guards", [])
    for guard in guard_entries:
        guard_name = guard.get("name", "").lower()
        guard_policy = guard.get("policy", {})
        original_policy = dict(guard_policy) if isinstance(guard_policy, dict) else {}
        guard_metrics = guard.get("metrics", {})

        if not guard_policy and guard_metrics:
            if guard_name == "rmt":
                guard_policy = {
                    "deadband": guard_metrics.get("deadband_used", 0.1),
                    "margin": guard_metrics.get("margin_used", 1.5),
                    "detection_threshold": guard_metrics.get(
                        "detection_threshold", 1.65
                    ),
                    "q_method": guard_metrics.get("q_used", "auto"),
                    "epsilon_default": guard_metrics.get("epsilon_default"),
                    "epsilon_by_family": guard_metrics.get("epsilon_by_family"),
                }
            elif guard_name == "spectral":
                sigma_quantile = guard_metrics.get(
                    "sigma_quantile",
                    guard_metrics.get("contraction", guard_metrics.get("kappa", 0.95)),
                )
                multiple_testing = guard_metrics.get("multiple_testing") or (
                    guard_metrics.get("multipletesting")
                    if isinstance(guard_metrics.get("multipletesting"), dict)
                    else None
                )
                guard_policy = {
                    "max_spectral_norm": guard_metrics.get("max_spectral_norm"),
                    "stability_score": guard_metrics.get("stability_score", 0.95),
                    "caps_applied": guard_metrics.get("caps_applied", 0),
                    "sigma_quantile": sigma_quantile,
                    "deadband": guard_metrics.get("deadband", 0.1),
                    "max_caps": guard_metrics.get("max_caps", 5),
                }
                if isinstance(multiple_testing, dict) and multiple_testing:
                    guard_policy["multiple_testing"] = multiple_testing
            elif guard_name == "variance":
                guard_policy = {
                    "scope": guard_metrics.get("scope", "both"),
                    "min_gain_threshold": guard_metrics.get("min_gain_threshold", 0.3),
                    "target_modules": guard_metrics.get("target_modules", 0),
                    "ve_enabled": guard_metrics.get("ve_enabled", False),
                }
            elif guard_name == "invariants":
                guard_policy = {
                    "checks_performed": guard_metrics.get("checks_performed", 0),
                    "violations_found": guard_metrics.get("violations_found", 0),
                }

        if guard_policy:
            if guard_name == "spectral":
                sigma_quantile = guard_policy.get("sigma_quantile")
                if sigma_quantile is None:
                    sigma_quantile = guard_policy.get("contraction")
                if sigma_quantile is None and "kappa" in guard_policy:
                    sigma_quantile = guard_policy["kappa"]
                sanitized_policy = dict(guard_policy)
                if sigma_quantile is not None:
                    try:
                        sanitized_policy["sigma_quantile"] = float(sigma_quantile)
                    except (TypeError, ValueError):
                        pass
                _promote_legacy_multiple_testing_key(sanitized_policy)
                sanitized_policy.pop("contraction", None)
                sanitized_policy.pop("kappa", None)
                if sanitized_policy.get("max_spectral_norm") in (None, 0):
                    sanitized_policy["max_spectral_norm"] = None
                guard_policy = sanitized_policy
            elif guard_name == "rmt" and isinstance(guard_metrics, dict):
                eps_default = guard_metrics.get("epsilon_default")
                if eps_default is not None and "epsilon_default" not in guard_policy:
                    guard_policy = dict(guard_policy)
                    guard_policy["epsilon_default"] = eps_default
            elif guard_name == "variance" and original_policy:
                guard_policy = dict(guard_policy)
                for key in (
                    "deadband",
                    "min_abs_adjust",
                    "max_scale_step",
                    "min_effect_lognll",
                    "predictive_one_sided",
                    "topk_backstop",
                    "max_adjusted_modules",
                ):
                    if key in original_policy and key not in guard_policy:
                        guard_policy[key] = original_policy[key]
            policies[guard_name] = dict(guard_policy)

    tier_defaults = get_tier_policies().get(_resolve_policy_tier(report), {})

    def _merge_defaults(target: dict[str, Any], defaults: dict[str, Any]) -> None:
        for key, value in defaults.items():
            if isinstance(value, dict):
                target_value = target.get(key)
                if not isinstance(target_value, dict):
                    target[key] = copy.deepcopy(value)
                else:
                    _merge_defaults(target_value, value)
            else:
                if target.get(key) in (None, "", [], {}):
                    target[key] = copy.deepcopy(value)

    for guard_name, defaults in tier_defaults.items():
        if not isinstance(defaults, dict):
            continue
        if guard_name not in policies:
            policies[guard_name] = copy.deepcopy(defaults)
        else:
            _merge_defaults(policies[guard_name], defaults)

    if not policies:
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
        if "spectral" in metrics:
            policies["spectral"] = {"status": "default_config"}
        if "rmt" in metrics:
            policies["rmt"] = {"status": "default_config"}

    return policies


def _normalize_override_entry(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if item is not None]
    return []


def _extract_policy_overrides(report: RunReport) -> list[str]:
    """Return ordered list of policy override paths referenced by the run."""
    overrides: list[str] = []
    meta = report.get("meta", {})
    config = report.get("config", {})
    candidate_sources: list[str] = []
    if isinstance(meta, dict):
        candidate_sources.extend(
            _normalize_override_entry(meta.get("policy_overrides"))
        )
        candidate_sources.extend(_normalize_override_entry(meta.get("overrides")))
        auto_meta = meta.get("auto")
        if isinstance(auto_meta, dict):
            candidate_sources.extend(
                _normalize_override_entry(auto_meta.get("overrides"))
            )
    if isinstance(config, dict):
        candidate_sources.extend(_normalize_override_entry(config.get("overrides")))
    seen: set[str] = set()
    for entry in candidate_sources:
        if entry and entry not in seen:
            overrides.append(entry)
            seen.add(entry)
    return overrides


def _compute_policy_digest(policy: dict[str, Any]) -> str:
    canonical = json.dumps(
        policy, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


__all__ = [
    "_compute_variance_policy_digest",
    "_compute_thresholds_payload",
    "_compute_thresholds_hash",
    "_promote_legacy_multiple_testing_key",
    "_resolve_policy_tier",
    "_build_resolved_policies",
    "_extract_effective_policies",
    "_normalize_override_entry",
    "_extract_policy_overrides",
    "_format_family_caps",
    "_format_epsilon_map",
    "_compute_policy_digest",
]
