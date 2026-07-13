from __future__ import annotations

import math

import pytest

from invarlock.core import (
    assurance_guard_validation_variance_measurements as measurements_module,
)
from invarlock.core.assurance_guard_validation_raw import (
    _guard_metric_impact_raw_errors,
    _invariants_raw_errors,
    _rmt_raw_errors,
)
from invarlock.core.assurance_guard_validation_variance_measurement_parsing import (
    _arm_values,
    _float_list,
    _pair_close,
    _token_count_list,
    _window_errors,
)
from invarlock.core.assurance_guard_validation_variance_measurements import (
    _aggregate_errors,
    _variance_measurement_errors,
)
from invarlock.core.assurance_guard_validation_variance_scale_selection import (
    _producer_scale_replay_errors,
    _replay_producer_scale_filter,
)
from invarlock.eval.guard_metric_impact import (
    build_guard_metric_bare_report,
    extract_guard_metric_arm_facts,
    guard_metric_schedule_digest,
)


def _rmt_entry() -> dict:
    return {
        "name": "rmt",
        "passed": True,
        "policy": {"epsilon_default": 0.1},
        "metrics": {
            "measurement_contract": {"schema": "rmt-v1"},
            "edge_risk_by_family_base": {"ffn": 1.0},
            "edge_risk_by_family": {"ffn": 1.05},
            "edge_risk_by_module_base": {"layer": 1.0},
            "edge_risk_by_module": {"layer": 1.05},
            "module_family_map": {"layer": "ffn"},
            "epsilon_by_family": {"ffn": 0.1},
            "stable": True,
        },
    }


def _rmt_errors(entry: dict, report: dict | None = None) -> list[str]:
    return _rmt_raw_errors(report or {}, [("rmt", entry, "guards[1]")])


def test_rmt_replay_requires_one_complete_raw_record() -> None:
    assert "exactly one" in _rmt_raw_errors({}, [])[0]
    missing = _rmt_entry()
    missing.pop("metrics")
    assert "metrics must be a non-empty object" in "\n".join(_rmt_errors(missing))


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ("empty-family", "non-empty baseline/current family maps"),
        ("empty-module", "requires module-level baseline/current risks"),
        ("module-inventory", "module-level RMT inventories must match"),
        ("invalid-module", "invalid module evidence"),
        ("base-mirror", "family baseline disagrees"),
        ("current-mirror", "family current value disagrees"),
        ("family-inventory", "baseline/current family sets must match"),
        ("epsilon-default", "non-negative epsilon_default"),
        ("invalid-family", "invalid RMT values"),
        ("stable", "metrics.stable disagrees"),
        ("passed", ".passed disagrees"),
    ],
)
def test_rmt_replay_rejects_forged_measurement_relationships(
    mutation: str, fragment: str
) -> None:
    entry = _rmt_entry()
    metrics = entry["metrics"]
    if mutation == "empty-family":
        metrics["edge_risk_by_family_base"] = {}
    elif mutation == "empty-module":
        metrics["edge_risk_by_module_base"] = {}
    elif mutation == "module-inventory":
        metrics["edge_risk_by_module"] = {"other": 1.05}
    elif mutation == "invalid-module":
        metrics["module_family_map"]["layer"] = ""
    elif mutation == "base-mirror":
        metrics["edge_risk_by_family_base"]["ffn"] = 9.0
    elif mutation == "current-mirror":
        metrics["edge_risk_by_family"]["ffn"] = 9.0
    elif mutation == "family-inventory":
        metrics["edge_risk_by_family"]["other"] = 0.0
    elif mutation == "epsilon-default":
        entry["policy"]["epsilon_default"] = -1.0
    elif mutation == "invalid-family":
        metrics["edge_risk_by_family_base"]["ffn"] = "bad"
    elif mutation == "stable":
        metrics["stable"] = False
    elif mutation == "passed":
        entry["passed"] = False

    assert fragment in "\n".join(_rmt_errors(entry))


def test_rmt_replay_handles_zero_baseline_and_reconciles_report_mirrors() -> None:
    entry = _rmt_entry()
    metrics = entry["metrics"]
    metrics["edge_risk_by_family_base"]["ffn"] = 0.0
    metrics["edge_risk_by_module_base"]["layer"] = 0.0
    metrics["stable"] = False
    entry["passed"] = False
    report = {
        "rmt": {
            "edge_risk_by_family_base": {"forged": 0.0},
            "edge_risk_by_family": {"forged": 0.0},
            "epsilon_by_family": {"forged": 0.0},
        }
    }

    errors = _rmt_errors(entry, report)

    assert sum("disagrees with the raw rmt record" in error for error in errors) == 3


def _invariant_entry(stage: str) -> tuple[str, dict, str]:
    return (
        "invariants",
        {
            "name": "invariants",
            "stage": stage,
            "policy": {"strict_mode": True, "on_fail": "block"},
            "metrics": {"checks_performed": 1, "violations_found": 0},
            "details": {
                "baseline_checks": {"finite": True},
                "current_checks": {"finite": True},
            },
            "violations": [],
        },
        f"guards.{stage}",
    )


def test_invariant_replay_rejects_changed_counts_violations_and_open_policy() -> None:
    assert "pre and post" in _invariants_raw_errors([])[0]
    pre = _invariant_entry("pre")
    post = _invariant_entry("post")
    post[1]["details"]["current_checks"]["finite"] = False
    post[1]["metrics"] = {"checks_performed": 2, "violations_found": 0}
    post[1]["violations"] = "not-an-array"
    post[1]["policy"] = {"strict_mode": False, "on_fail": "warn"}

    text = "\n".join(_invariants_raw_errors([pre, post]))
    assert "observations changed" in text
    assert "checks_performed disagrees" in text
    assert "violations_found disagrees" in text
    assert "not fail-closed" in text


def test_invariant_replay_requires_nonempty_check_evidence() -> None:
    pre = _invariant_entry("pre")
    post = _invariant_entry("post")
    pre[1]["details"]["baseline_checks"] = {}

    assert "requires non-empty baseline_checks" in "\n".join(
        _invariants_raw_errors([pre, post])
    )


def test_guard_metric_impact_rejects_tampered_derived_values() -> None:
    final_ids = [7, 8]
    bare_source = {
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {
            "final": {
                "window_ids": final_ids,
                "logloss": [math.log(10.0), math.log(10.0)],
                "token_counts": [1, 1],
            }
        },
    }
    report = {
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 11.0}},
        "evaluation_windows": {
            "final": {
                "window_ids": final_ids,
                "logloss": [math.log(11.0), math.log(11.0)],
                "token_counts": [1, 1],
            }
        },
    }
    bare_facts = extract_guard_metric_arm_facts(bare_source, "ppl_causal")
    guarded_facts = extract_guard_metric_arm_facts(report, "ppl_causal")
    bare_report = build_guard_metric_bare_report(bare_source, "ppl_causal")
    schedule_digest = guard_metric_schedule_digest(report, "ppl_causal")
    assert bare_facts is not None
    assert guarded_facts is not None
    assert bare_report is not None
    assert schedule_digest is not None
    report["guard_metric_impact"] = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 10.0,
        "guarded_value": 11.0,
        "degradation": 0.2,
        "degradation_limit": 0.5,
        "display_value": 20.0,
        "display_unit": "percent",
        "bare_facts": bare_facts,
        "guarded_facts": guarded_facts,
        "bare_report": bare_report,
        "schedule_digest": schedule_digest,
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "diagnostics": [],
        "source": "paired_control",
    }
    errors = _guard_metric_impact_raw_errors(report)
    assert {
        "guard_metric_impact degradation disagrees with retained measurements.",
        "guard_metric_impact display_value disagrees with retained measurements.",
        "guard metric impact degradation mismatch",
        "guard metric impact display_value mismatch",
    } <= set(errors)


def test_guard_metric_impact_raw_replay_rejects_every_contract_surface() -> None:
    report = {
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 11.0}},
        "evaluation_windows": {
            "final": {
                "window_ids": ["w0"],
                "logloss": [math.log(11.0)],
                "token_counts": [1],
            }
        },
        "guard_metric_impact": {
            "metric_kind": "ppl_causal",
            "direction": "higher",
            "degradation_basis": "absolute_drop",
            "bare_value": 10.0,
            "guarded_value": 11.0,
            "bare_facts": {"weighted_logloss_sum": math.log(9.0), "token_count": 1},
            "guarded_facts": {
                "weighted_logloss_sum": math.log(12.0),
                "token_count": 1,
            },
            "degradation": 0.1,
            "degradation_limit": 0.2,
            "display_value": 10.0,
            "display_unit": "percentage_points",
            "evaluated": True,
            "passed": True,
            "checks": {"measurements_valid": False},
            "bare_ppl": 10.0,
        },
    }

    text = "\n".join(_guard_metric_impact_raw_errors(report))

    for fragment in (
        "bare_ppl is not part",
        "arm facts do not replay",
        "direction disagrees",
        "degradation_basis disagrees",
        "display_unit disagrees",
        "checks must contain only passing booleans",
    ):
        assert fragment in text


def test_guard_metric_impact_raw_replay_requires_measurements_and_checks() -> None:
    errors = _guard_metric_impact_raw_errors(
        {
            "guard_metric_impact": {
                "metric_kind": "unsupported",
                "bare_value": True,
                "guarded_value": object(),
                "checks": {},
            }
        }
    )

    text = "\n".join(errors)
    assert "requires valid retained primary-metric measurements" in text
    assert "checks must retain measured consistency checks" in text


def test_variance_measurement_parsers_reject_invalid_window_values() -> None:
    assert _float_list([], positive=False) is None
    assert _float_list([True], positive=False) is None
    assert _float_list([0.0], positive=True) is None
    assert _token_count_list([]) is None
    assert _token_count_list([0]) is None
    assert _pair_close([1.0], [1.0, 2.0]) is False

    errors, values = _arm_values({}, "condition_a", 1, source="ab")
    assert values is None
    assert errors == ["ab.condition_a is required."]


def test_variance_arm_rejects_cardinality_and_exp_log_loss_mismatch() -> None:
    measurements = {
        "condition_a": {
            "ppl": [3.0],
            "log_loss": [0.0],
            "token_counts": [10],
        }
    }
    errors, values = _arm_values(measurements, "condition_a", 1, source="ab")
    assert values is not None
    assert errors == ["ab.condition_a.ppl[0] must equal exp(log_loss[0])."]

    errors, values = _arm_values(measurements, "condition_a", 2, source="ab")
    assert values is None
    assert len(errors) == 3


def test_variance_window_ids_must_be_unique_and_provenance_bound() -> None:
    assert (
        "unique identifier"
        in _window_errors({"window_ids": ["same", "same"]}, {}, 2, source="ab")[0]
    )
    errors = _window_errors(
        {"window_ids": ["w1"]},
        {"ab_provenance": {"condition_a": {}, "condition_b": None}},
        1,
        source="ab",
    )
    assert len(errors) == 2


def test_variance_aggregate_replay_rejects_pairing_and_all_derived_claims() -> None:
    condition_a = ([2.0], [0.6931471805599453], [10])
    condition_b = ([1.0], [0.0], [20])
    errors = _aggregate_errors(
        {
            "ppl_no_ve": 9.0,
            "ppl_with_ve": 9.0,
            "ab_gain": 9.0,
            "ratio_ci": [9.0, 9.0],
            "predictive_gate": {
                "mean_delta": 9.0,
                "delta_ci": [9.0, 9.0],
                "gain_ci": [9.0, 9.0],
                "passed": True,
                "reason": "forged",
            },
        },
        {"ratio_ci": [9.0, 9.0], "delta_log_ci": [9.0, 9.0]},
        condition_a,
        condition_b,
        (0.4, 0.6),
        (-0.8, -0.6),
        {"min_effect_lognll": 0.01, "predictive_one_sided": True},
        source="ab",
        no_adjustment=False,
    )
    text = "\n".join(errors)
    for fragment in (
        "token counts must match",
        "ppl_no_ve",
        "ppl_with_ve",
        "ab_gain",
        "ratio_ci",
        "delta_log_ci",
        "mean_delta",
        "delta_ci",
        "gain_ci",
        "predictive decision",
    ):
        assert fragment in text

    no_adjustment_errors = _aggregate_errors(
        {},
        {},
        condition_a,
        condition_b,
        (0.4, 0.6),
        (-0.8, -0.6),
        {},
        source="ab",
        no_adjustment=True,
    )
    assert any(
        "virtual no-adjustment condition B must equal A" in error
        for error in no_adjustment_errors
    )


def test_variance_measurement_replay_fails_closed_before_bootstrap() -> None:
    assert (
        "coverage must be positive"
        in _variance_measurement_errors(
            {}, {}, {}, 0, {}, source="variance", no_adjustment=False
        )[0]
    )
    assert (
        "ab_measurements is required"
        in _variance_measurement_errors(
            {}, {}, {}, 1, {}, source="variance", no_adjustment=False
        )[0]
    )


def test_variance_measurement_replay_reports_bootstrap_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "window_ids": ["w1"],
        "condition_a": {"ppl": [1.0], "log_loss": [0.0], "token_counts": [10]},
        "condition_b": {"ppl": [1.0], "log_loss": [0.0], "token_counts": [10]},
        "ratio_bootstrap": {
            "method": "percentile_mean_ppl_ratio",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 1,
        },
        "delta_log_bootstrap": {
            "method": "bca_paired_delta_log",
            "replicates": 500,
            "alpha": 0.05,
            "seed": 212,
            "weights": "condition_a_token_counts",
        },
    }
    metrics = {
        "ab_measurements": raw,
        "ab_seed_used": 1,
        "ab_provenance": {
            "condition_a": {"window_ids": ["w1"]},
            "condition_b": {"window_ids": ["w1"]},
        },
    }
    entry = {"details": {"stats": {"ab_measurements": raw}}}
    variance = {"ab_test": {"measurements": raw}}

    def _fail(*_args, **_kwargs):
        raise ArithmeticError("degenerate bootstrap")

    monkeypatch.setattr(measurements_module, "_replay_intervals", _fail)
    errors = _variance_measurement_errors(
        variance,
        entry,
        metrics,
        1,
        {"alpha": 0.05},
        source="variance",
        no_adjustment=False,
    )

    assert errors == [
        "variance.metrics.ab_measurements cannot be replayed: degenerate bootstrap"
    ]


def test_scale_filter_replay_rejects_bad_values_and_enforces_deterministic_cap() -> (
    None
):
    assert (
        _replay_producer_scale_filter(
            {"layer": True},
            max_step=0.1,
            min_abs=0.01,
            topk=1,
            deadband=0.01,
            max_adjusted=1,
        )
        is None
    )
    filtered = _replay_producer_scale_filter(
        {"b": 0.8, "a": 1.2},
        max_step=0.1,
        min_abs=0.01,
        topk=0,
        deadband=0.01,
        max_adjusted=1,
    )
    assert filtered == {"a": 1.1}


def test_scale_replay_requires_exact_keys_and_values() -> None:
    kwargs = {
        "label": "final",
        "max_step": 0.1,
        "min_abs": 0.01,
        "topk": 0,
        "deadband": 0.01,
        "max_adjusted": 2,
        "source": "variance",
    }
    assert (
        "keys must exactly replay"
        in _producer_scale_replay_errors({}, {"layer": 1.2}, **kwargs)[0]
    )
    assert (
        ".layer must exactly replay"
        in _producer_scale_replay_errors({"layer": 1.0}, {"layer": 1.2}, **kwargs)[0]
    )
