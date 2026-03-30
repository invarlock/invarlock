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


def _section_dict(cfg: Any, name: str) -> dict[str, Any]:
    section_fn = getattr(cfg, "section", None)
    if callable(section_fn):
        try:
            section = section_fn(name)
        except (AttributeError, KeyError, TypeError):
            section = None
        if isinstance(section, dict):
            return section
    try:
        value = getattr(cfg, name)
    except (AttributeError, KeyError, TypeError):
        value = None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
    return {}


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
    guards_section = _section_dict(cfg, "guards")
    eval_section = _section_dict(cfg, "eval")
    guard_overrides = {
        "spectral": to_serialisable_dict_fn(guards_section.get("spectral", {})),
        "rmt": to_serialisable_dict_fn(guards_section.get("rmt", {})),
        "variance": to_serialisable_dict_fn(guards_section.get("variance", {})),
        "invariants": to_serialisable_dict_fn(guards_section.get("invariants", {})),
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

    run_context: dict[str, Any] = {
        "eval": to_serialisable_dict_fn(eval_section),
        "dataset": to_serialisable_dict_fn(cfg.dataset),
        "guards": guard_overrides,
        "profile": profile if profile else "",
        "pairing_baseline": pairing_schedule,
        "seeds": dict(seed_bundle),
        "plugins": dict(plugin_provenance),
        "run_id": run_id,
    }
    if tiny_relax_enabled:
        run_section = run_context.setdefault("run", {})
        if isinstance(run_section, dict):
            run_section["tiny_relax"] = True

    baseline_eval = _baseline_eval_windows(baseline_report_data)
    if baseline_eval is not None:
        run_context["baseline_eval_windows"] = baseline_eval

    primary_metric = run_context.setdefault("primary_metric", {})
    if not isinstance(primary_metric, dict):
        primary_metric = {}
        run_context["primary_metric"] = primary_metric
    primary_metric["acceptance_range"] = pm_acceptance_range
    run_context["pm_acceptance_range"] = pm_acceptance_range
    if pm_drift_band:
        primary_metric["drift_band"] = pm_drift_band
        run_context["pm_drift_band"] = pm_drift_band
    primary_metric["overhead_threshold"] = guard_overhead_threshold
    run_context["guard_overhead_threshold"] = guard_overhead_threshold
    run_context["model_profile"] = {
        "family": getattr(model_profile, "family", ""),
        "default_loss": getattr(model_profile, "default_loss", ""),
        "module_selectors": getattr(model_profile, "module_selectors", {}),
        "invariants": getattr(model_profile, "invariants", []),
        "cert_lints": getattr(model_profile, "cert_lints", []),
    }
    extra_context = to_serialisable_dict_fn(_section_dict(cfg, "context"))
    if isinstance(extra_context, dict):
        run_context.update(extra_context)
    eval_context = run_context.get("eval")
    if isinstance(eval_context, dict):
        loss_context = eval_context.setdefault("loss", {})
        if isinstance(loss_context, dict):
            loss_context["resolved_type"] = resolved_loss_type

    return run_context


def _normalize_edit_plan(plan_obj: Any) -> dict[str, Any]:
    if isinstance(plan_obj, dict):
        return dict(plan_obj)
    plan_data = getattr(plan_obj, "_data", None)
    if isinstance(plan_data, dict):
        return dict(plan_data)
    if hasattr(plan_obj, "items"):
        try:
            return dict(plan_obj)
        except (TypeError, ValueError):
            return {}
    return {}


def build_run_execution_config_payloads(
    *,
    cfg: Any,
    model_profile: Any,
) -> RunExecutionConfigPayloads:
    auto_section = _section_dict(cfg, "auto")
    try:
        auto_enabled = bool(auto_section.get("enabled"))
    except (AttributeError, TypeError, ValueError):
        auto_enabled = False
    try:
        auto_tier = auto_section.get("tier") or "balanced"
    except (AttributeError, TypeError, ValueError):
        auto_tier = "balanced"
    try:
        auto_probes = int(auto_section.get("probes", 0))
    except (AttributeError, TypeError, ValueError):
        auto_probes = 0
    try:
        auto_target_ratio = float(auto_section.get("target_pm_ratio", 2.0))
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
