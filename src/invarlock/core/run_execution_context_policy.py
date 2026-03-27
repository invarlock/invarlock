from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

ToSerialisableDictFn = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class RunExecutionConfigPayloads:
    auto_config: dict[str, Any]
    edit_config: dict[str, Any]


def _normalize_profile_checks(existing_checks: Any) -> list[str]:
    if isinstance(existing_checks, list | tuple | set):
        return [str(item) for item in existing_checks]
    if existing_checks:
        return [str(existing_checks)]
    return []


def _baseline_eval_windows(
    baseline_report_data: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(baseline_report_data, Mapping):
        return None
    evaluation_windows = baseline_report_data.get("evaluation_windows")
    if not isinstance(evaluation_windows, Mapping):
        return None
    final = evaluation_windows.get("final")
    if not isinstance(final, Mapping):
        return None
    window_ids = final.get("window_ids")
    logloss = final.get("logloss")
    if not isinstance(window_ids, list) or not isinstance(logloss, list):
        return None
    payload: dict[str, Any] = {
        "final": {
            "window_ids": list(window_ids or []),
            "logloss": list(logloss or []),
        }
    }
    token_counts = final.get("token_counts")
    if isinstance(token_counts, list):
        payload["final"]["token_counts"] = list(token_counts or [])
    return payload


def build_run_context_payload(
    *,
    cfg: Any,
    profile: str | None,
    pairing_schedule: dict[str, Any] | None,
    seed_bundle: Mapping[str, Any],
    plugin_provenance: Mapping[str, Any],
    run_id: str,
    baseline_report_data: Mapping[str, Any] | None,
    pm_acceptance_range: tuple[float, float] | None,
    pm_drift_band: tuple[float, float] | None,
    guard_overhead_threshold: float,
    model_profile: Any,
    resolved_loss_type: str,
    tiny_relax_enabled: bool,
    to_serialisable_dict_fn: ToSerialisableDictFn,
) -> dict[str, Any]:
    guard_overrides = {
        "spectral": to_serialisable_dict_fn(getattr(cfg.guards, "spectral", {})),
        "rmt": to_serialisable_dict_fn(getattr(cfg.guards, "rmt", {})),
        "variance": to_serialisable_dict_fn(getattr(cfg.guards, "variance", {})),
        "invariants": to_serialisable_dict_fn(getattr(cfg.guards, "invariants", {})),
    }

    if getattr(model_profile, "invariants", None):
        invariants_policy = guard_overrides.setdefault("invariants", {})
        checks_list = _normalize_profile_checks(
            invariants_policy.get("profile_checks", [])
        )
        for invariant in model_profile.invariants:
            invariant_name = str(invariant)
            if invariant_name not in checks_list:
                checks_list.append(invariant_name)
        invariants_policy["profile_checks"] = checks_list

    run_context = {
        "eval": to_serialisable_dict_fn(cfg.eval),
        "dataset": to_serialisable_dict_fn(cfg.dataset),
        "guards": guard_overrides,
        "profile": profile if profile else "",
        "pairing_baseline": pairing_schedule,
        "seeds": dict(seed_bundle),
        "plugins": dict(plugin_provenance),
        "run_id": run_id,
    }
    if tiny_relax_enabled:
        run_context.setdefault("run", {})["tiny_relax"] = True

    baseline_eval = _baseline_eval_windows(baseline_report_data)
    if baseline_eval is not None:
        run_context["baseline_eval_windows"] = baseline_eval

    run_context.setdefault("primary_metric", {})["acceptance_range"] = (
        pm_acceptance_range
    )
    run_context["pm_acceptance_range"] = pm_acceptance_range
    if pm_drift_band:
        run_context.setdefault("primary_metric", {})["drift_band"] = pm_drift_band
        run_context["pm_drift_band"] = pm_drift_band
    run_context.setdefault("primary_metric", {})["overhead_threshold"] = (
        guard_overhead_threshold
    )
    run_context["guard_overhead_threshold"] = guard_overhead_threshold
    run_context["model_profile"] = {
        "family": getattr(model_profile, "family", ""),
        "default_loss": getattr(model_profile, "default_loss", ""),
        "module_selectors": getattr(model_profile, "module_selectors", {}),
        "invariants": getattr(model_profile, "invariants", []),
        "cert_lints": getattr(model_profile, "cert_lints", []),
    }
    extra_context = to_serialisable_dict_fn(getattr(cfg, "context", {}))
    if isinstance(extra_context, dict):
        run_context.update(extra_context)
    try:
        run_context.setdefault("eval", {}).setdefault("loss", {})[
            "resolved_type"
        ] = resolved_loss_type
    except (AttributeError, TypeError):
        pass

    return run_context


def _normalize_edit_plan(plan_obj: Any) -> dict[str, Any]:
    if isinstance(plan_obj, dict):
        return dict(plan_obj)
    plan_data = getattr(plan_obj, "_data", None)
    if isinstance(plan_data, dict):
        return dict(plan_data)
    if hasattr(plan_obj, "items"):
        try:
            return dict(plan_obj)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return {}
    return {}


def build_run_execution_config_payloads(
    *,
    cfg: Any,
    model_profile: Any,
) -> RunExecutionConfigPayloads:
    try:
        auto_enabled = bool(cfg.auto.enabled)
    except (AttributeError, TypeError, ValueError):
        auto_enabled = False
    try:
        auto_tier = cfg.auto.tier
    except (AttributeError, TypeError, ValueError):
        auto_tier = "balanced"
    try:
        auto_probes = int(cfg.auto.probes)
    except (AttributeError, TypeError, ValueError):
        auto_probes = 0
    try:
        auto_target_ratio = float(cfg.auto.target_pm_ratio)
    except (AttributeError, TypeError, ValueError):
        auto_target_ratio = 2.0

    auto_config = {
        "enabled": auto_enabled,
        "tier": auto_tier,
        "probes": auto_probes,
        "target_pm_ratio": auto_target_ratio,
    }

    edit_config: dict[str, Any] = {}
    try:
        plan_obj = getattr(cfg.edit, "plan", {})
        if plan_obj:
            edit_config = _normalize_edit_plan(plan_obj)
    except (AttributeError, TypeError):
        edit_config = {}

    module_selectors = getattr(model_profile, "module_selectors", None)
    if (
        isinstance(module_selectors, dict)
        and module_selectors
        and "module_selectors" not in edit_config
    ):
        edit_config["module_selectors"] = {
            key: list(values) for key, values in module_selectors.items()
        }

    return RunExecutionConfigPayloads(
        auto_config=auto_config,
        edit_config=edit_config,
    )
