from __future__ import annotations

import time
from typing import Any, Protocol

from ..api import Guard, GuardWithContext, GuardWithPrepare, RunConfig, RunReport
from ..auto_tuning import resolve_tier_policies
from ..exceptions import InvarlockError
from ..types import GuardDiagnostic, GuardValidationResult, LogLevel


class ResolveTierPoliciesFn(Protocol):
    def __call__(
        self,
        tier: str,
        edit_name: str | None,
        explicit_overrides: dict[str, dict[str, Any]] | None,
        *,
        profile: str | None,
    ) -> dict[str, dict[str, Any]]: ...


_HANDLED_GUARD_ERRORS = (
    AttributeError,
    InvarlockError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


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
    for extra_key in (
        "final_z_scores",
        "module_family_map",
        "baseline_metrics",
        "final_metrics",
        "final_degeneracy",
        "measurement_inventory",
        "correction_ledger",
        "supported",
        "reason",
        "assurance_blocking",
        "status",
        "stage",
    ):
        if extra_key in raw:
            normalized[extra_key] = raw[extra_key]
    return normalized


def _load_external_guard_baseline(
    runner: Any,
    guard: Guard,
    report: RunReport,
) -> None:
    """Load paired-run guard measurements after local coverage preparation."""

    context = report.context if isinstance(report.context, dict) else {}
    if not context.get("baseline_guard_evidence_required", False):
        return
    if str(guard.name).lower() not in {"spectral", "rmt"}:
        return
    loader = getattr(guard, "load_external_baseline_evidence", None)
    if not callable(loader):
        raise RuntimeError(
            f"Guard '{guard.name}' cannot consume required baseline evidence"
        )
    outcome = loader()
    if not isinstance(outcome, dict):
        raise TypeError(
            f"Guard '{guard.name}' returned invalid baseline evidence outcome"
        )
    report.meta.setdefault("baseline_guard_evidence", {})[guard.name] = dict(outcome)
    runner._log_event(
        "guard_prepare",
        "external_baseline_loaded"
        if outcome.get("ready", False)
        else "external_baseline_unavailable",
        LogLevel.INFO if outcome.get("ready", False) else LogLevel.ERROR,
        {"guard": guard.name, **dict(outcome)},
    )


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
    raw_overrides = (
        config_meta.get("guards", {}) if isinstance(config_meta, dict) else {}
    )
    if not isinstance(raw_overrides, dict) or any(
        not isinstance(value, dict) for value in raw_overrides.values()
    ):
        raise TypeError("guard policy overrides must be a mapping of guard mappings")
    explicit_overrides = {
        str(name): dict(value) for name, value in raw_overrides.items()
    }

    raw_report_context = getattr(report, "context", None)
    report_context = raw_report_context if isinstance(raw_report_context, dict) else {}
    raw_profile = report_context.get("profile")
    if not isinstance(raw_profile, str) or not raw_profile.strip():
        runtime_context = report_context.get("runtime")
        raw_profile = (
            runtime_context.get("profile")
            if isinstance(runtime_context, dict)
            else None
        )
    profile = (
        raw_profile.strip().lower()
        if isinstance(raw_profile, str) and raw_profile.strip()
        else None
    )

    try:
        policies = resolver(
            tier,
            edit_name,
            explicit_overrides,
            profile=profile,
        )
    except _HANDLED_GUARD_ERRORS as error:
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
        {
            "tier": tier,
            "edit": edit_name,
            "profile": profile,
            "policies_count": len(policies),
        },
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
    except _HANDLED_GUARD_ERRORS as error:
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
    except _HANDLED_GUARD_ERRORS as exc:
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
                _load_external_guard_baseline(runner, guard, report)
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
        except _HANDLED_GUARD_ERRORS as error:
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
    result_keys: list[str] | None = None,
    result_stages: list[str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run safety guards."""
    runner._log_event("guards", "start", LogLevel.INFO, {"count": len(guards)})
    guard_results: dict[str, dict[str, Any]] = {}

    if result_keys is not None and len(result_keys) != len(guards):
        raise ValueError("result_keys must align with guards")
    if result_stages is not None and len(result_stages) != len(guards):
        raise ValueError("result_stages must align with guards")

    for index, guard in enumerate(guards):
        result_key = result_keys[index] if result_keys is not None else guard.name
        result_stage = result_stages[index] if result_stages is not None else None
        runner._log_event("guard", "start", LogLevel.INFO, {"guard": guard.name})
        guard_start = time.perf_counter()

        _set_guard_run_context(guard, report)

        try:
            result = guard.validate(model, adapter, report.context)
            normalized_result = _normalize_guard_result(result)
            if result_stage:
                normalized_result["stage"] = result_stage
            guard_results[result_key] = normalized_result
            status = "passed" if normalized_result.get("passed", False) else "failed"
            runner._log_event(
                "guard",
                "complete",
                LogLevel.INFO,
                {"guard": guard.name, "status": status},
            )
        except _HANDLED_GUARD_ERRORS as error:
            runner._log_event(
                "guard",
                "error",
                LogLevel.ERROR,
                {"guard": guard.name, "error": str(error)},
            )
            raise
        finally:
            if guard_timings is not None:
                guard_timings[result_key] = max(
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
