from typing import Any

import pytest

from invarlock.reporting.validate import (
    validate_against_baseline,
    validate_drift_gate,
    validate_guard_metric_impact,
)


def _pm(
    kind: str,
    ratio: float | None = None,
    preview: float | None = None,
    final: float | None = None,
) -> dict[str, Any]:
    pm = {"kind": kind}
    if ratio is not None:
        field = "delta_vs_baseline_pp" if kind == "accuracy" else "ratio_vs_baseline"
        pm[field] = ratio
    if preview is not None:
        pm["preview"] = preview
    if final is not None:
        pm["final"] = final
    return pm


def test_validate_against_baseline_ppl_like_pass():
    run = {
        "metrics": {
            "primary_metric": _pm("ppl_causal", ratio=1.28),
            "invariants_passed": True,
        },
        "param_reduction_ratio": 0.022,
        "layers_modified": 1,
        "metrics_extra": {},
    }
    baseline = {
        "ratio_vs_baseline": 1.27,
        "param_reduction_ratio": 0.022,
        "layers_modified": 1,
    }

    result = validate_against_baseline(
        run, baseline, tol_ratio=0.02, tol_param_ratio=0.05
    )
    assert result.passed
    assert result.checks.get("ratio_tolerance") is True
    assert result.checks.get("param_ratio_tolerance") is True
    assert result.checks.get("ratio_bounds") is True


def test_validate_against_baseline_ppl_like_fail_bounds_and_param():
    run = {
        "metrics": {"primary_metric": _pm("ppl_seq2seq", ratio=1.40)},
        "param_reduction_ratio": 0.10,
    }
    baseline = {"ratio_vs_baseline": 1.25, "param_reduction_ratio": 0.02}

    result = validate_against_baseline(
        run, baseline, ratio_bounds=(1.0, 1.32), tol_param_ratio=0.01
    )
    assert result.passed is False
    assert result.checks.get("ratio_bounds") is False
    assert result.checks.get("param_ratio_tolerance") is False
    assert any("outside acceptable bounds" in m for m in result.messages)


def test_validate_against_baseline_accuracy_delta_pp_bounds():
    run = {"metrics": {"primary_metric": _pm("accuracy", ratio=+0.4)}}
    baseline = {"param_reduction_ratio": 0.0}

    ok = validate_against_baseline(run, baseline, delta_bounds_pp=(-1.0, +1.0))
    assert ok.checks.get("delta_bounds_pp") is True

    bad = validate_against_baseline(run, baseline, delta_bounds_pp=(-0.2, +0.2))
    assert bad.checks.get("delta_bounds_pp") is False
    assert bad.passed is False


def test_validate_drift_gate_pass_and_fail():
    ok = validate_drift_gate(
        {"metrics": {"primary_metric": _pm("ppl_causal", preview=100.0, final=101.0)}}
    )
    assert ok.passed and ok.checks.get("drift_gate") is True

    bad = validate_drift_gate(
        {"metrics": {"primary_metric": _pm("ppl_causal", preview=100.0, final=120.0)}}
    )
    assert bad.passed is False and bad.checks.get("drift_gate") is False

    missing = validate_drift_gate({"metrics": {}})
    assert missing.passed is False


def test_validate_guard_metric_impact_pass_fail_missing():
    bare = {"metrics": {"primary_metric": _pm("ppl_causal", final=100.0)}}
    guarded_ok = {"metrics": {"primary_metric": _pm("ppl_causal", final=100.5)}}
    guarded_bad = {"metrics": {"primary_metric": _pm("ppl_causal", final=105.0)}}

    r_ok = validate_guard_metric_impact(bare, guarded_ok, degradation_limit=0.01)
    assert r_ok.passed and r_ok.checks.get("guard_metric_impact") is True
    assert r_ok.metrics == {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "bare_value": 100.0,
        "guarded_value": 100.5,
        "degradation_basis": "relative_increase",
        "degradation": pytest.approx(0.005),
        "display_value": pytest.approx(0.5),
        "display_unit": "percent",
    }

    r_bad = validate_guard_metric_impact(bare, guarded_bad, degradation_limit=0.01)
    assert r_bad.passed is False and r_bad.checks.get("guard_metric_impact") is False

    r_missing = validate_guard_metric_impact({}, {})
    assert (
        r_missing.passed is False
        and r_missing.checks.get("guard_metric_impact") is False
    )


def test_validate_guard_metric_impact_rejects_invalid_numeric_and_metric_semantics():
    ppl = {"metrics": {"primary_metric": _pm("ppl_causal", final=100.0)}}
    accuracy = {"metrics": {"primary_metric": _pm("accuracy", final=0.9)}}

    for invalid_threshold in (True, -0.01, float("nan"), float("inf")):
        result = validate_guard_metric_impact(
            ppl,
            ppl,
            degradation_limit=invalid_threshold,
        )
        assert result.passed is False
        assert result.checks["guard_metric_impact"] is False

    for invalid_final in (True, -1.0, 0.0, float("nan"), float("inf")):
        invalid = {
            "metrics": {
                "primary_metric": _pm("ppl_causal", final=invalid_final),
            }
        }
        assert validate_guard_metric_impact(ppl, invalid).passed is False

    assert validate_guard_metric_impact(accuracy, accuracy).passed is True
    assert validate_guard_metric_impact(ppl, accuracy).passed is False


def test_validate_guard_metric_impact_is_direction_aware_for_accuracy() -> None:
    bare = {"metrics": {"primary_metric": _pm("accuracy", final=0.8)}}
    guarded_ok = {"metrics": {"primary_metric": _pm("accuracy", final=0.792)}}
    guarded_bad = {"metrics": {"primary_metric": _pm("accuracy", final=0.78)}}
    guarded_better = {"metrics": {"primary_metric": _pm("accuracy", final=0.82)}}

    result = validate_guard_metric_impact(bare, guarded_ok, degradation_limit=0.01)

    assert result.passed is True
    assert result.metrics == {
        "metric_kind": "accuracy",
        "direction": "higher",
        "bare_value": 0.8,
        "guarded_value": 0.792,
        "degradation_basis": "absolute_drop",
        "degradation": pytest.approx(0.008),
        "display_value": pytest.approx(0.8),
        "display_unit": "percentage_points",
    }
    assert (
        validate_guard_metric_impact(bare, guarded_bad, degradation_limit=0.01).passed
        is False
    )
    assert (
        validate_guard_metric_impact(
            bare, guarded_better, degradation_limit=0.01
        ).passed
        is True
    )
