from __future__ import annotations

from invarlock.guards.variance_results import (
    build_finalize_metrics,
    build_finalize_result,
    build_prepare_result,
    build_scale_statistics,
    evaluate_finalize_state,
)


def test_build_scale_statistics_defaults_and_values() -> None:
    assert build_scale_statistics({}) == {
        "mean_scale": 1.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
    }
    stats = build_scale_statistics({"a": 1.2, "b": 0.8})
    assert stats["mean_scale"] == 1.0
    assert stats["min_scale"] == 0.8
    assert stats["max_scale"] == 1.2


def test_build_prepare_result_ready_and_failure_paths() -> None:
    ready = build_prepare_result(
        policy={"scope": "ffn", "min_gain": 0.0},
        target_modules={"m": object()},
        scales={"m": 1.1},
        calibration_stats={"status": "complete"},
        preparation_time=1.5,
        ready=True,
    )
    assert ready["ready"] is True
    assert ready["baseline_metrics"]["scale_statistics"]["max_scale"] == 1.1

    failed = build_prepare_result(
        policy={"scope": "ffn", "min_gain": 0.0},
        target_modules={},
        scales={},
        calibration_stats={"status": "insufficient"},
        preparation_time=0.5,
        ready=False,
        warning="none",
        error="boom",
    )
    assert failed["ready"] is False
    assert failed["warning"] == "none"
    assert failed["error"] == "boom"
    assert failed["baseline_metrics"] == {}


def test_evaluate_finalize_state_collects_errors_and_warnings() -> None:
    state = evaluate_finalize_state(
        should_enable=False,
        enabled_after_ab=True,
        gate_reason="gate_failed",
        ppl_no_ve=2.0,
        ppl_with_ve=1.98,
        final_ppl=2.7,
        ab_windows_used=4,
        ab_seed_used=999,
        expected_seed=123,
        enable_attempt_count=4,
        disable_attempt_count=5,
        checkpoint_depth=2,
        ab_gain=0.001,
        required_gain_with_deadband=0.01,
        absolute_floor=0.05,
        calibration_status="insufficient",
    )
    assert state["passed"] is False
    assert any("A/B gate rejection" in err for err in state["errors"])
    assert any("tie-breaker deadband" in err for err in state["errors"])
    assert any("unexpected seed" in warn for warn in state["warnings"])
    assert any("operating in monitor mode" in warn for warn in state["warnings"])


def test_evaluate_finalize_state_handles_disabled_ve_rise_and_conservative_warning() -> (
    None
):
    state = evaluate_finalize_state(
        should_enable=True,
        enabled_after_ab=False,
        gate_reason="approved",
        ppl_no_ve=2.0,
        ppl_with_ve=1.5,
        final_ppl=2.6,
        ab_windows_used=None,
        ab_seed_used=None,
        expected_seed=123,
        enable_attempt_count=0,
        disable_attempt_count=0,
        checkpoint_depth=0,
        ab_gain=0.25,
        required_gain_with_deadband=0.1,
        absolute_floor=0.05,
        calibration_status="complete",
    )
    assert state["passed"] is False
    assert any(
        "disabled despite A/B gate approval" in warn for warn in state["warnings"]
    )
    assert any("> 0.5 when VE disabled" in err for err in state["errors"])


def test_evaluate_finalize_state_accepts_disabled_clean_path() -> None:
    state = evaluate_finalize_state(
        should_enable=False,
        enabled_after_ab=False,
        gate_reason="not_needed",
        ppl_no_ve=2.0,
        ppl_with_ve=1.8,
        final_ppl=2.4,
        ab_windows_used=None,
        ab_seed_used=None,
        expected_seed=123,
        enable_attempt_count=0,
        disable_attempt_count=0,
        checkpoint_depth=0,
        ab_gain=None,
        required_gain_with_deadband=0.1,
        absolute_floor=0.05,
        calibration_status="complete",
    )
    assert state == {"passed": True, "warnings": [], "errors": []}


def test_evaluate_finalize_state_accepts_enabled_clean_path() -> None:
    state = evaluate_finalize_state(
        should_enable=True,
        enabled_after_ab=True,
        gate_reason="approved",
        ppl_no_ve=2.0,
        ppl_with_ve=1.7,
        final_ppl=1.7,
        ab_windows_used=4,
        ab_seed_used=123,
        expected_seed=123,
        enable_attempt_count=1,
        disable_attempt_count=0,
        checkpoint_depth=0,
        ab_gain=0.3,
        required_gain_with_deadband=0.1,
        absolute_floor=0.1,
        calibration_status="complete",
    )
    assert state == {"passed": True, "warnings": [], "errors": []}


def test_build_finalize_metrics_and_result_copy_payloads() -> None:
    stats = {
        "target_module_names": ["m"],
        "tap": ["transformer.h.*.mlp.c_proj"],
        "ab_provenance": {"condition_a": {"tag": "pre"}},
        "ab_point_estimates": {"tag": "pre", "coverage": 4},
        "proposed_scales_pre_edit": {"m": 1.1},
        "proposed_scales_post_edit": {"m": 1.2},
    }
    metrics = build_finalize_metrics(
        scales={"m": 1.1},
        target_modules={"m": object()},
        stats=stats,
        focus_modules={"m"},
        enabled_after_ab=True,
        should_enable=True,
        ab_gain=0.2,
        ab_windows_used=4,
        ab_seed_used=123,
        monitor_only=False,
        policy={
            "min_gain": 0.1,
            "scope": "ffn",
            "max_calib": 16,
            "mode": "ci",
            "min_rel_gain": 0.0,
            "alpha": 0.05,
        },
        ppl_no_ve=2.0,
        ppl_with_ve=1.6,
        ratio_ci=(0.7, 0.9),
        calibration_stats={"status": "complete"},
        predictive_gate_state={"passed": True},
        raw_scales_pre_edit={"m": 1.1},
        raw_scales_post_edit={"m": 1.2},
    )
    assert metrics["met_threshold"] is True
    assert metrics["focus_modules"] == ["m"]
    assert metrics["ab_provenance"]["condition_a"]["tag"] == "pre"

    result = build_finalize_result(
        passed=True,
        metrics=metrics,
        warnings=[],
        errors=[],
        finalize_time=0.5,
        events=[{"operation": "finalize_complete"}],
        enabled_after_ab=True,
        ppl_no_ve=2.0,
        scales={"m": 1.1},
        stats=stats,
        policy={"scope": "ffn"},
    )
    assert result["passed"] is True
    assert result["details"]["ve_applied"] is True
    assert result["details"]["ab_test_performed"] is True
