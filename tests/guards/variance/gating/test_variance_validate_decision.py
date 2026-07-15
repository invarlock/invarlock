from types import SimpleNamespace

import invarlock.guards.variance_runtime as runtime_variance
from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_runtime import before_edit_guard


def _prep(policy):
    g = VarianceGuard(policy=policy)
    g._prepared = True
    g._post_edit_evaluated = True
    return g


def test_validate_monitor_decision_when_monitor_only_and_fail():
    g = _prep(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "monitor_only": True,
            "scope": "ffn",
            "max_calib": 100,
        }
    )
    # Set metrics to cause failure and ensure finalize flags errors
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 100.0
    g._ratio_ci = (1.0, 1.0)
    # Cause PPL rise when VE disabled to produce an error
    g._final_ppl = 101.0
    result = g.validate(object(), adapter=None, context={})
    assert result["passed"] is False and result["decision"] == "monitor"


def test_validate_passes_when_aligned_and_no_warnings():
    g = _prep(
        {
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.01,
            "scope": "ffn",
            "max_calib": 100,
        }
    )
    # Configure should_enable False (ratio_ci hi > 1 - min_rel_gain), with matching enabled_after_ab False
    g._predictive_gate_state.update({"evaluated": True, "passed": True})
    g._ab_gain = 0.0
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 100.0
    g._ratio_ci = (0.99, 0.995)
    result = g.validate(object(), adapter=None, context={})
    # Some warnings may be produced by ancillary checks; ensure passed True
    assert result["passed"] is True


def test_variance_runtime_decision_and_before_edit_skip_paths() -> None:
    recorded: list[tuple[str, dict[str, object]]] = []
    guard = SimpleNamespace(
        _prepared=False,
        _log_event=lambda event, **kwargs: recorded.append((event, kwargs)),
    )

    before_edit_guard(guard, model=object())
    assert recorded == []


def test_variance_runtime_before_and_after_edit_logging_paths() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    guard = SimpleNamespace(
        _prepared=True,
        _post_edit_evaluated=False,
        _scales={"layer": 1.0},
        _log_event=lambda event, **kwargs: events.append((event, kwargs)),
        _refresh_after_edit_metrics=lambda model: events.append(
            ("refresh", {"model": model})
        ),
    )

    before_edit_guard(guard, model=object())
    runtime_variance.after_edit_guard(guard, model=object())

    skipped = SimpleNamespace(
        _prepared=False,
        _log_event=lambda event, **kwargs: events.append((event, kwargs)),
    )
    runtime_variance.after_edit_guard(skipped, model=object())

    assert [event for event, _kwargs in events] == [
        "before_edit",
        "refresh",
        "after_edit",
        "after_edit_skipped",
    ]


def test_variance_validate_guard_decision_matrix() -> None:
    guard = SimpleNamespace(
        _monitor_only=False,
        _policy={"seed": 123},
        finalize=lambda _model: {
            "passed": True,
            "warnings": ["warn"],
            "errors": [],
            "metrics": {"score": 1.0},
            "details": {"policy": {"scope": "ffn"}},
        },
    )

    warning_result = runtime_variance.validate_guard(
        guard, model=object(), adapter=None, context={}
    )
    assert warning_result.passed is True
    assert warning_result.decision == "monitor"
    assert warning_result.diagnostics[0].severity == "warning"

    guard._monitor_only = True
    guard.finalize = lambda _model: {
        "passed": False,
        "warnings": [],
        "errors": ["boom"],
        "metrics": {"score": 0.0},
        "details": {"policy": {"scope": "ffn"}},
    }
    blocked_result = runtime_variance.validate_guard(
        guard, model=object(), adapter=None, context={}
    )
    assert blocked_result.passed is False
    assert blocked_result.decision == "monitor"
    assert blocked_result.violations[0]["message"] == "boom"


def test_variance_finalize_guard_covers_early_return_and_gate_transitions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_variance,
        "evaluate_finalize_state",
        lambda **kwargs: {"passed": True, "warnings": [], "errors": []},
    )
    monkeypatch.setattr(
        runtime_variance,
        "build_finalize_metrics",
        lambda **kwargs: {
            "from_metrics": True,
            "enabled_after_ab": kwargs["enabled_after_ab"],
        },
    )
    monkeypatch.setattr(
        runtime_variance,
        "build_finalize_result",
        lambda **kwargs: {
            "passed": kwargs["passed"],
            "metrics": kwargs["metrics"],
            "warnings": kwargs["warnings"],
            "errors": kwargs["errors"],
        },
    )

    early_events: list[tuple[str, dict[str, object]]] = []
    not_prepared = SimpleNamespace(
        _prepared=False,
        _log_event=lambda event, **kwargs: early_events.append((event, kwargs)),
        name="variance",
    )
    early_result = runtime_variance.finalize_guard(not_prepared, model=object())
    assert early_result["passed"] is False
    assert early_result["errors"] == ["Preparation failed or no target modules found"]
    assert early_events[0][0] == "finalize_failed"

    refresh_events: list[str] = []
    refresh_guard = SimpleNamespace(
        _prepared=True,
        _post_edit_evaluated=False,
        _monitor_only=False,
        _enabled=False,
        _scales={"layer": 1.0},
        _target_modules=["layer"],
        _stats={},
        _focus_modules=[],
        _ab_windows_used=0,
        _ab_seed_used=0,
        _ab_gain=0.0,
        _policy={"min_gain": 0.0, "seed": 123},
        _ppl_no_ve=10.0,
        _ppl_with_ve=10.0,
        _ratio_ci=(0.99, 1.0),
        _calibration_stats={"status": "ok"},
        _predictive_gate_state={},
        _raw_scales_pre_edit={},
        _raw_scales_post_edit={},
        _enable_attempt_count=0,
        _disable_attempt_count=0,
        _checkpoint_stack=[],
        ABSOLUTE_FLOOR=0.0,
        TIE_BREAKER_DEADBAND=0.0,
        _log_event=lambda *_a, **_k: None,
        _refresh_after_edit_metrics=lambda *_a, **_k: refresh_events.append("refresh"),
        _evaluate_ab_gate=lambda: (False, "refresh"),
        enable=lambda _model: None,
        disable=lambda _model: None,
    )
    refresh_result = runtime_variance.finalize_guard(refresh_guard, model=object())
    assert refresh_events == ["refresh"]
    assert refresh_result["decision"] == "allow"

    enable_events: list[str] = []
    enable_guard = SimpleNamespace(
        _prepared=True,
        _post_edit_evaluated=True,
        _monitor_only=True,
        _enabled=True,
        _scales={"layer": 1.0},
        _target_modules=["layer"],
        _stats={},
        _focus_modules=[],
        _ab_windows_used=0,
        _ab_seed_used=0,
        _ab_gain=0.0,
        _policy={"min_gain": 0.0, "seed": 123},
        _ppl_no_ve=10.0,
        _ppl_with_ve=10.0,
        _ratio_ci=(0.99, 1.0),
        _calibration_stats={"status": "ok"},
        _predictive_gate_state={},
        _raw_scales_pre_edit={},
        _raw_scales_post_edit={},
        _enable_attempt_count=0,
        _disable_attempt_count=0,
        _checkpoint_stack=[],
        ABSOLUTE_FLOOR=0.0,
        TIE_BREAKER_DEADBAND=0.0,
        _log_event=lambda *_a, **_k: None,
        _refresh_after_edit_metrics=lambda *_a, **_k: None,
        _evaluate_ab_gate=lambda: (True, "enable"),
        enable=lambda _model: (
            enable_events.append("enable")
            or setattr(enable_guard, "_enabled", True)
            or False
        ),
        disable=lambda _model: (
            enable_events.append("disable")
            or setattr(enable_guard, "_enabled", False)
            or True
        ),
    )

    enable_result = runtime_variance.finalize_guard(enable_guard, model=object())
    assert enable_guard._enabled is False
    assert enable_guard._scales == {}
    assert enable_result["decision"] == "monitor"
    assert enable_result["metrics"]["enabled_after_ab"] is True
    assert enable_result["metrics"]["ve_enabled"] is False
    assert enable_result["metrics"]["subject_restored_after_ab"] is True
    assert enable_events == ["enable", "disable"]

    disable_events: list[str] = []
    disable_guard = SimpleNamespace(
        _prepared=True,
        _post_edit_evaluated=True,
        _monitor_only=False,
        _enabled=True,
        _scales={"layer": 1.0},
        _target_modules=["layer"],
        _stats={},
        _focus_modules=[],
        _ab_windows_used=0,
        _ab_seed_used=0,
        _ab_gain=0.0,
        _policy={"min_gain": 0.0, "seed": 123},
        _ppl_no_ve=10.0,
        _ppl_with_ve=10.0,
        _ratio_ci=(0.99, 1.0),
        _calibration_stats={"status": "ok"},
        _predictive_gate_state={},
        _raw_scales_pre_edit={},
        _raw_scales_post_edit={},
        _enable_attempt_count=0,
        _disable_attempt_count=0,
        _checkpoint_stack=[],
        ABSOLUTE_FLOOR=0.0,
        TIE_BREAKER_DEADBAND=0.0,
        _log_event=lambda *_a, **_k: None,
        _refresh_after_edit_metrics=lambda *_a, **_k: None,
        _evaluate_ab_gate=lambda: (False, "disable"),
        enable=lambda _model: disable_events.append("enable"),
        disable=lambda _model: (
            disable_events.append("disable")
            or setattr(disable_guard, "_enabled", False)
            or None
        ),
    )

    disable_result = runtime_variance.finalize_guard(disable_guard, model=object())
    assert disable_guard._enabled is False
    assert disable_events == ["disable"]
    assert disable_result["decision"] == "allow"
