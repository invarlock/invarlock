from invarlock.core.types import (
    GuardDiagnostic,
    GuardOutcome,
    GuardValidationResult,
    PolicyConfig,
    decision_to_action,
    get_worst_action,
    get_worst_decision,
    normalize_guard_decision,
)


def test_policy_config_action_resolution():
    # Default on_violation used when requested is 'none'
    cfg = PolicyConfig(
        on_violation="warn", guard_overrides=None, enable_auto_rollback=False
    )
    assert cfg.get_action_for_guard("spectral", "none") == "warn"

    # Requested action other than none takes precedence
    assert cfg.get_action_for_guard("spectral", "rollback") == "rollback"

    # Guard-specific override takes highest precedence
    cfg.guard_overrides = {"spectral": "abort"}
    assert cfg.get_action_for_guard("spectral", "warn") == "abort"


def test_get_worst_action_and_guard_outcome_defaults():
    # Worst action across a set
    actions = ["none", "warn", "rollback", "abort"]
    assert get_worst_action(actions) == "abort"

    # Empty list falls back to none
    assert get_worst_action([]) == "none"

    # Unknown actions retain their label but lowest priority
    assert get_worst_action(["foo"]) == "foo"
    assert get_worst_decision(["allow", "monitor", "rollback", "block"]) == "block"
    assert normalize_guard_decision("warn") == "monitor"
    assert decision_to_action("block") == "abort"

    # GuardOutcome defaults populated
    o = GuardOutcome("g", True)
    assert isinstance(o.violations, list) and o.violations == []
    assert isinstance(o.metrics, dict) and o.metrics == {}
    assert o.decision == "allow"
    assert o.action == "none"


def test_guard_validation_result_properties_and_extras() -> None:
    result = GuardValidationResult(
        passed=False,
        decision="monitor",
        metrics={"score": 0.1},
        diagnostics=(
            GuardDiagnostic(
                kind="guard_warning",
                severity="warning",
                message="warn",
                details={"field": "value"},
            ),
        ),
        policy={"threshold": 0.5},
        details={"reason": "policy"},
        violations=({"kind": "threshold"},),
        extras={"raw": True},
    )

    assert result.passed is False
    assert result.decision == "monitor"
    assert result.action == "warn"
    assert result.metrics == {"score": 0.1}
    assert result.policy == {"threshold": 0.5}
    assert result.details == {"reason": "policy"}
    assert result.violations == ({"kind": "threshold"},)
    assert result.diagnostics == (
        GuardDiagnostic(
            kind="guard_warning",
            severity="warning",
            message="warn",
            details={"field": "value"},
        ),
    )
    assert result.extras == {"raw": True}


def test_guard_validation_result_property_fallbacks_filter_invalid_records() -> None:
    result = GuardValidationResult(passed=True, decision="allow")
    result.pop("action", None)
    result["diagnostics"] = [
        "ignore-me",
        {"kind": "diag", "severity": "info", "message": "kept", "details": {"x": 1}},
    ]
    result["violations"] = "bad"
    result["metrics"] = {"value": 1}
    result["policy"] = {"mode": "strict"}
    result["details"] = {"detail": "kept"}

    assert result.action == "none"
    assert result.diagnostics == (
        GuardDiagnostic(
            kind="diag",
            severity="info",
            message="kept",
            details={"x": 1},
        ),
    )
    assert result.violations == ()
    assert result.metrics == {"value": 1}
    assert result.policy == {"mode": "strict"}
    assert result.details == {"detail": "kept"}


def test_decision_normalization_and_priority_edges() -> None:
    assert normalize_guard_decision(None, fallback_action="reject") == "block"
    assert normalize_guard_decision(None, passed=False) == "block"
    assert decision_to_action("monitor") == "warn"
    assert get_worst_decision([]) == "allow"
    assert get_worst_decision(["monitor", "allow", "rollback"]) == "rollback"


def test_guard_outcome_and_policy_config_cover_explicit_branches() -> None:
    override_cfg = PolicyConfig(
        on_violation="warn",
        guard_overrides={"variance": "rollback"},
        enable_auto_rollback=False,
    )
    assert override_cfg.guard_overrides == {"variance": "rollback"}
    assert override_cfg.get_decision_for_guard("variance", "allow") == "rollback"
    assert override_cfg.get_action_for_guard("variance", "allow") == "rollback"

    by_action = GuardOutcome("variance", False, action="warn")
    assert by_action.decision == "monitor"
    assert by_action.action == "warn"

    by_decision = GuardOutcome("spectral", False, decision="rollback")
    assert by_decision.decision == "rollback"
    assert by_decision.action == "rollback"
