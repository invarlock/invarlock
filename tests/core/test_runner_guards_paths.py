from __future__ import annotations

import pytest

from invarlock.core.api import Guard, RunReport
from invarlock.core.exceptions import InvarlockError
from invarlock.core.runner_guards import (
    _coerce_diagnostics,
    _normalize_guard_result,
    guard_phase,
    prepare_guards_phase,
    resolve_guard_policies,
)
from invarlock.core.types import GuardDiagnostic, GuardValidationResult


class _RunnerStub:
    def __init__(
        self,
        *,
        strict_guard_prepare: bool = False,
        tier_policies: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.events: list[tuple[str, str, str, dict[str, object]]] = []
        self.strict_guard_prepare = strict_guard_prepare
        self.tier_policies = tier_policies or {}

    def _log_event(
        self, category: str, event: str, level: str, details: dict[str, object]
    ) -> None:
        self.events.append((category, event, level, details))

    def _resolve_policy_flags(self, _config: object) -> dict[str, bool]:
        return {"strict_guard_prepare": self.strict_guard_prepare}

    def _resolve_guard_policies(
        self, _report: RunReport, _auto_config: dict[str, object] | None
    ) -> dict[str, dict[str, object]]:
        return self.tier_policies


class _ContextPrepareGuard(Guard):
    def __init__(self, name: str, result: GuardValidationResult | dict[str, object]):
        self.name = name
        self.result = result
        self.received_context: RunReport | None = None
        self.prepare_calls: list[tuple[object, object, object, dict[str, object]]] = []

    def set_run_context(self, report: RunReport) -> None:
        self.received_context = report

    def prepare(
        self,
        model: object,
        adapter: object,
        calib: object,
        policy_config: dict[str, object],
    ) -> dict[str, object]:
        self.prepare_calls.append((model, adapter, calib, dict(policy_config)))
        return {"ready": True}

    def validate(
        self, model: object, adapter: object, context: dict[str, object]
    ) -> GuardValidationResult | dict[str, object]:
        _ = model, adapter, context
        return self.result


class _PrepareFailureGuard(Guard):
    name = "broken"

    def set_run_context(self, report: RunReport) -> None:
        report.meta["broken_seen"] = True

    def prepare(
        self,
        model: object,
        adapter: object,
        calib: object,
        policy_config: dict[str, object],
    ) -> dict[str, object]:
        _ = model, adapter, calib, policy_config
        raise InvarlockError(code="E999", message="prepare failed")

    def validate(
        self, model: object, adapter: object, context: dict[str, object]
    ) -> dict[str, object]:
        _ = model, adapter, context
        return {"passed": True}


def test_coerce_and_normalize_guard_results_cover_typed_and_raw_paths() -> None:
    typed = _coerce_diagnostics(
        [
            GuardDiagnostic(
                kind="typed",
                severity="warning",
                message="typed message",
                details={"source": "guard"},
            ),
            {"kind": "raw", "severity": "info", "message": "raw message", "extra": 3},
        ]
    )

    assert typed == [
        {
            "kind": "typed",
            "severity": "warning",
            "message": "typed message",
            "details": {"source": "guard"},
        },
        {
            "kind": "raw",
            "severity": "info",
            "message": "raw message",
            "details": {"extra": 3},
        },
    ]

    normalized_typed = _normalize_guard_result(
        GuardValidationResult(
            passed=False,
            decision="monitor",
            metrics={"score": 0.5},
            diagnostics=(
                GuardDiagnostic(
                    kind="typed",
                    severity="warning",
                    message="typed message",
                    details={"source": "guard"},
                ),
            ),
            policy={"deadband": 0.1},
            details={"checked": True},
            violations=({"message": "violation"},),
            extras={"baseline_metrics": {"sigma": 1.2}},
        )
    )

    assert normalized_typed["decision"] == "monitor"
    assert normalized_typed["baseline_metrics"] == {"sigma": 1.2}
    assert normalized_typed["diagnostics"][0]["kind"] == "typed"

    normalized_raw = _normalize_guard_result(
        {
            "passed": True,
            "decision": "",
            "diagnostics": [{"message": "from raw", "family": "ffn"}],
            "violations": ["string violation"],
            "baseline_metrics": {"sigma": 0.9},
        }
    )

    assert normalized_raw["decision"] == "allow"
    assert normalized_raw["violations"] == [{"message": "string violation"}]
    assert normalized_raw["diagnostics"][0]["details"] == {"family": "ffn"}
    assert normalized_raw["baseline_metrics"] == {"sigma": 0.9}


def test_normalize_guard_result_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="Unsupported guard result type"):
        _normalize_guard_result("not-a-result")


def test_resolve_guard_policies_defaults_for_missing_or_invalid_auto_config() -> None:
    runner = _RunnerStub()
    seen: dict[str, object] = {}

    def fake_resolver(
        tier: str, edit_name: str | None, overrides: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        seen["tier"] = tier
        seen["edit_name"] = edit_name
        seen["overrides"] = dict(overrides)
        return {"spectral": {"deadband": 0.1}}

    report = RunReport(meta={"config": "not-a-dict", "edit_name": "quant"})
    policies = resolve_guard_policies(runner, report, None, resolver=fake_resolver)

    assert policies == {"spectral": {"deadband": 0.1}}
    assert seen == {"tier": "balanced", "edit_name": "quant", "overrides": {}}

    seen.clear()
    report = RunReport(meta={"config": {"guards": {"spectral": {"deadband": 0.2}}}})
    resolve_guard_policies(runner, report, "bad-auto-config", resolver=fake_resolver)
    assert seen["tier"] == "balanced"
    assert seen["overrides"] == {"spectral": {"deadband": 0.2}}


def test_resolve_guard_policies_defaults_when_config_meta_has_no_auto_key() -> None:
    runner = _RunnerStub()
    seen: dict[str, object] = {}

    def fake_resolver(
        tier: str, edit_name: str | None, overrides: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        seen["tier"] = tier
        seen["edit_name"] = edit_name
        seen["overrides"] = dict(overrides)
        return {"spectral": {"deadband": 0.1}}

    report = RunReport(meta={"config": {}, "edit_name": "quant"})
    policies = resolve_guard_policies(runner, report, None, resolver=fake_resolver)

    assert policies == {"spectral": {"deadband": 0.1}}
    assert seen == {"tier": "balanced", "edit_name": "quant", "overrides": {}}


def test_prepare_guards_phase_non_strict_invarlock_errors_are_recorded_and_skipped() -> (
    None
):
    runner = _RunnerStub(
        strict_guard_prepare=False,
        tier_policies={"steady": {"deadband": 0.2}},
    )
    report = RunReport(meta={"config": {"guards": {}}})
    steady = _ContextPrepareGuard(
        "steady",
        GuardValidationResult(passed=True, decision="allow"),
    )

    prepare_guards_phase(
        runner,
        model=object(),
        adapter=object(),
        guards=[_PrepareFailureGuard(), steady],
        calibration_data={"rows": 2},
        report=report,
        auto_config=None,
        config=None,
    )

    failures = report.meta.get("guard_prepare_failures", [])
    assert failures == [{"guard": "broken", "error": "[INVARLOCK:E999] prepare failed"}]
    assert report.meta["tier_policies"] == {"steady": {"deadband": 0.2}}
    assert steady.received_context is report
    assert steady.prepare_calls[0][3] == {"deadband": 0.2}


def test_guard_phase_sets_context_and_skips_timings_when_not_requested() -> None:
    runner = _RunnerStub()
    report = RunReport(context={"baseline": "ok"})
    guard = _ContextPrepareGuard(
        "spectral",
        GuardValidationResult(
            passed=True,
            decision="allow",
            diagnostics=(
                GuardDiagnostic(
                    kind="spectral_ok",
                    severity="info",
                    message="all clear",
                ),
            ),
            extras={"final_metrics": {"layer": 1.0}},
        ),
    )

    results = guard_phase(
        runner,
        model=object(),
        adapter=object(),
        guards=[guard],
        report=report,
    )

    assert guard.received_context is report
    assert report.guards == results
    assert results["spectral"]["passed"] is True
    assert results["spectral"]["final_metrics"] == {"layer": 1.0}
