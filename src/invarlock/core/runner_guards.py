from __future__ import annotations

import time
from typing import Any

from .api import Guard, GuardWithContext, GuardWithPrepare, RunConfig, RunReport
from .auto_tuning import resolve_tier_policies
from .types import LogLevel


def resolve_guard_policies(
    runner: Any,
    report: RunReport,
    auto_config: dict[str, Any] | None = None,
    *,
    resolver: Any = resolve_tier_policies,
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
        runner._log_event(
            "auto_tuning",
            "tier_resolved",
            LogLevel.INFO,
            {"tier": tier, "edit": edit_name, "policies_count": len(policies)},
        )
        return policies
    except Exception as error:
        runner._log_event(
            "auto_tuning",
            "tier_resolution_failed",
            LogLevel.ERROR,
            {"tier": tier, "error": str(error)},
        )
        return {}


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

            if isinstance(guard, GuardWithContext):
                try:
                    guard.set_run_context(report)
                except Exception as exc:
                    runner._log_event(
                        "guard_prepare",
                        "context_error",
                        LogLevel.WARNING,
                        {"guard": guard.name, "error": str(exc)},
                    )

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
            if strict_guard_prepare:
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

        if isinstance(guard, GuardWithContext):
            try:
                guard.set_run_context(report)
            except Exception as exc:  # pragma: no cover - defensive
                runner._log_event(
                    "guard",
                    "context_error",
                    LogLevel.WARNING,
                    {"guard": guard.name, "error": str(exc)},
                )

        try:
            result = guard.validate(model, adapter, report.context)
            guard_results[guard.name] = result
            status = "passed" if result.get("passed", False) else "failed"
            runner._log_event(
                "guard",
                "complete",
                LogLevel.INFO,
                {"guard": guard.name, "status": status},
            )
        except Exception as error:
            guard_results[guard.name] = {"passed": False, "error": str(error)}
            runner._log_event(
                "guard",
                "error",
                LogLevel.ERROR,
                {"guard": guard.name, "error": str(error)},
            )
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
