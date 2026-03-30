from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .api import Guard, GuardWithContext, GuardWithPrepare, RunConfig, RunReport
from .exceptions import GuardError, InvarlockError
from .auto_tuning import resolve_tier_policies
from .types import GuardDiagnostic, GuardValidationResult, LogLevel

ResolveTierPoliciesFn = Callable[
    [str, str | None, dict[str, Any]], dict[str, dict[str, Any]]
]


def _coerce_diagnostics(raw: Any) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, (list, tuple)) else ():
        if isinstance(item, GuardDiagnostic):
            diagnostics.append(
                {
                    "kind": item.kind,
                    "severity": item.severity,
                    "message": item.message,
                    "details": dict(item.details),
                }
            )
        elif isinstance(item, dict):
            diagnostics.append(
                {
                    "kind": str(item.get("kind", "guard_diagnostic")),
                    "severity": str(item.get("severity", "info")),
                    "message": str(item.get("message", "")),
                    "details": {
                        str(key): value
                        for key, value in item.items()
                        if key not in {"kind", "severity", "message"}
                    },
                }
            )
    return diagnostics


def _normalize_guard_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, GuardValidationResult):
        return {
            "passed": bool(raw.passed),
            "decision": str(raw.decision),
            "metrics": dict(raw.metrics),
            "diagnostics": _coerce_diagnostics(raw.diagnostics),
            "policy": dict(raw.policy),
            "details": dict(raw.details),
            "violations": [dict(item) for item in raw.violations],
            **dict(raw.extras),
        }

    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported guard result type: {type(raw)!r}")

    diagnostics = _coerce_diagnostics(raw.get("diagnostics"))
    decision = raw.get("decision")
    if not isinstance(decision, str) or not decision:
        decision = "allow" if bool(raw.get("passed", False)) else "block"
    normalized = {
        "passed": bool(raw.get("passed", False)),
        "decision": str(decision),
        "metrics": dict(raw.get("metrics", {})),
        "diagnostics": diagnostics,
        "policy": dict(raw.get("policy", {})),
        "details": dict(raw.get("details", {})),
        "violations": [
            dict(item) if isinstance(item, dict) else {"message": str(item)}
            for item in raw.get("violations", [])
            if isinstance(raw.get("violations", []), list | tuple)
        ],
    }
    for extra_key in ("final_z_scores", "module_family_map", "baseline_metrics", "final_metrics"):
        if extra_key in raw:
            normalized[extra_key] = raw[extra_key]
    return normalized


def resolve_guard_policies(
    runner: Any,
    report: RunReport,
    auto_config: dict[str, Any] | None = None,
    *,
    resolver: ResolveTierPoliciesFn = resolve_tier_policies,
) -> dict[str, dict[str, Any]]:
    """Resolve tier-based guard policies from configuration."""
    auto_cfg: dict[str, Any] | None = auto_config
    if auto_cfg is None:
        config_meta = report.meta.get("config") or {}
        auto_cfg = report.__dict__.get("auto_config")
        if auto_cfg is None and isinstance(config_meta, dict) and "auto" in config_meta:
            auto_cfg = config_meta["auto"]
        elif auto_cfg is None:
            auto_cfg = {"tier": "balanced", "enabled": True}

    if not isinstance(auto_cfg, dict):
        auto_cfg = {"tier": "balanced", "enabled": True}

    tier = auto_cfg.get("tier", "balanced")
    edit_name = None
    if hasattr(report, "edit") and report.edit:
        edit_name = report.edit.get("name")
    if not edit_name and "edit_name" in report.meta:
        edit_name = report.meta["edit_name"]

    config_meta = report.meta.get("config") or {}
    explicit_overrides = (
        config_meta.get("guards", {}) if isinstance(config_meta, dict) else {}
    )

    try:
        policies = resolver(tier, edit_name, explicit_overrides)
    except Exception as error:
        runner._log_event(
            "auto_tuning",
            "tier_resolution_failed",
            LogLevel.ERROR,
            {"tier": tier, "error": str(error)},
        )
        raise

    runner._log_event(
        "auto_tuning",
        "tier_resolved",
        LogLevel.INFO,
        {"tier": tier, "edit": edit_name, "policies_count": len(policies)},
    )
    return policies


def apply_guard_policy(runner: Any, guard: Guard, policy: dict[str, Any]) -> None:
    """Apply resolved policy parameters to a guard instance."""
    try:
        guard_config = getattr(guard, "config", None)
        guard_policy = getattr(guard, "policy", None)
        for param_name, param_value in policy.items():
            if hasattr(guard, param_name):
                setattr(guard, param_name, param_value)
            elif isinstance(guard_config, dict):
                guard_config[param_name] = param_value
            elif isinstance(guard_policy, dict):
                guard_policy[param_name] = param_value
            else:
                setattr(guard, param_name, param_value)
    except Exception as error:
        runner._log_event(
            "auto_tuning",
            "policy_application_failed",
            LogLevel.WARNING,
            {"guard": guard.name, "policy": policy, "error": str(error)},
        )
        raise


def _set_guard_run_context(guard: Guard, report: RunReport) -> None:
    if not isinstance(guard, GuardWithContext):
        return
    try:
        guard.set_run_context(report)
    except Exception as exc:
        raise RuntimeError(
            f"Guard '{guard.name}' run context setup failed: {exc}"
        ) from exc


def prepare_guards_phase(
    runner: Any,
    model: Any,
    adapter: Any,
    guards: list[Guard],
    calibration_data: Any,
    report: RunReport,
    auto_config: dict[str, Any] | None = None,
    config: RunConfig | None = None,
) -> None:
    """Prepare safety guards with tier-resolved policies."""
    runner._log_event("guards_prepare", "start", LogLevel.INFO, {"count": len(guards)})
    policy_flags = runner._resolve_policy_flags(config)
    strict_guard_prepare = policy_flags["strict_guard_prepare"]
    tier_policies = runner._resolve_guard_policies(report, auto_config)

    for guard in guards:
        runner._log_event(
            "guard_prepare", "start", LogLevel.INFO, {"guard": guard.name}
        )
        try:
            guard_policy: dict[str, Any] = tier_policies.get(guard.name, {})
            if guard_policy:
                apply_guard_policy(runner, guard, guard_policy)
                runner._log_event(
                    "guard_prepare",
                    "policy_applied",
                    LogLevel.INFO,
                    {"guard": guard.name, "policy": guard_policy},
                )

            _set_guard_run_context(guard, report)

            if isinstance(guard, GuardWithPrepare):
                prepare_result = guard.prepare(
                    model, adapter, calibration_data, guard_policy
                )
                runner._log_event(
                    "guard_prepare",
                    "complete",
                    LogLevel.INFO,
                    {"guard": guard.name, "ready": prepare_result.get("ready", False)},
                )
            else:
                runner._log_event(
                    "guard_prepare",
                    "skipped",
                    LogLevel.INFO,
                    {"guard": guard.name, "reason": "no_prepare_method"},
                )
        except Exception as error:
            runner._log_event(
                "guard_prepare",
                "error",
                LogLevel.ERROR,
                {"guard": guard.name, "error": str(error)},
            )
            report.meta.setdefault("guard_prepare_failures", []).append(
                {"guard": guard.name, "error": str(error)}
            )
            if strict_guard_prepare or not isinstance(error, InvarlockError):
                raise RuntimeError(
                    f"Guard '{guard.name}' prepare failed: {error}"
                ) from error

    report.meta["tier_policies"] = tier_policies
    runner._log_event(
        "guards_prepare", "complete", LogLevel.INFO, {"count": len(guards)}
    )


def guard_phase(
    runner: Any,
    model: Any,
    adapter: Any,
    guards: list[Guard],
    report: RunReport,
    *,
    guard_timings: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run safety guards."""
    runner._log_event("guards", "start", LogLevel.INFO, {"count": len(guards)})
    guard_results: dict[str, dict[str, Any]] = {}

    for guard in guards:
        runner._log_event("guard", "start", LogLevel.INFO, {"guard": guard.name})
        guard_start = time.perf_counter()

        _set_guard_run_context(guard, report)

        try:
            result = guard.validate(model, adapter, report.context)
            normalized_result = _normalize_guard_result(result)
            guard_results[guard.name] = normalized_result
            status = "passed" if normalized_result.get("passed", False) else "failed"
            runner._log_event(
                "guard",
                "complete",
                LogLevel.INFO,
                {"guard": guard.name, "status": status},
            )
        except Exception as error:
            runner._log_event(
                "guard",
                "error",
                LogLevel.ERROR,
                {"guard": guard.name, "error": str(error)},
            )
            raise
        finally:
            if guard_timings is not None:
                guard_timings[guard.name] = max(
                    0.0, float(time.perf_counter() - guard_start)
                )

    report.guards = guard_results
    passed_guards = sum(
        1 for result in guard_results.values() if result.get("passed", False)
    )
    runner._log_event(
        "guards",
        "complete",
        LogLevel.INFO,
        {"total": len(guards), "passed": passed_guards},
    )
    return guard_results


__all__ = [
    "apply_guard_policy",
    "guard_phase",
    "prepare_guards_phase",
    "resolve_guard_policies",
]
