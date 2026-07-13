from __future__ import annotations

import math

import pytest

from invarlock.core import assurance_guard_validation as facade
from invarlock.core import assurance_guard_validation_common as common
from invarlock.core import assurance_guard_validation_matrix as matrix
from invarlock.core import assurance_guard_validation_runtime as runtime


def _messages(errors: list[str]) -> str:
    return "\n".join(errors)


def test_validation_mirrors_fail_closed_on_missing_malformed_and_false_values() -> None:
    assert facade._validation_errors({}, require_complete=False) == []
    assert facade._validation_errors({"validation": {}}, require_complete=False) == []
    assert facade._validation_errors({}, require_complete=True) == [
        "strict assurance requires a validation object."
    ]
    errors = facade._validation_errors(
        {
            "validation": {
                "invariants_pass": "yes",
                "spectral_stable": False,
                "rmt_stable": True,
                "guard_metric_impact_acceptable": True,
                "guard_warning_policy_acceptable": 1,
                "primary_metric_tail_acceptable": False,
            }
        },
        require_complete=True,
    )

    text = _messages(errors)
    assert "validation.preview_final_drift_acceptable is required" in text
    assert "validation.primary_metric_acceptable is required" in text
    assert "validation.invariants_pass must be a boolean" in text
    assert "validation.spectral_stable is false" in text
    assert "validation.guard_warning_policy_acceptable must be a boolean" in text
    assert "validation.primary_metric_tail_acceptable is false" in text

    assert (
        facade._validation_errors(
            {
                "validation": {
                    "guard_warning_policy_acceptable": True,
                    "primary_metric_tail_acceptable": True,
                }
            },
            require_complete=False,
        )
        == []
    )


def test_guard_chain_parser_rejects_non_arrays_and_unknown_guards() -> None:
    assert facade._chain_from_sequence("invariants") is None
    assert facade._chain_from_sequence([{"name": "unknown"}]) is None

    errors = facade.strict_guard_chain_errors(
        {
            "assurance": {
                "canonical_guard_chain": "invalid",
                "guard_chain_observed": None,
            },
            "plugins": {"guards": ["unknown"]},
            "guards": "invalid",
            "context": {},
        },
        canonical_chain=("invariants", "spectral", "rmt", "invariants"),
    )
    text = _messages(errors)
    assert (
        "assurance.canonical_guard_chain must be an ordered guard chain array" in text
    )
    assert "requires assurance.guard_chain_observed" in text
    assert "plugins.guards must be an ordered guard chain array" in text
    assert "guards must be an ordered guard chain array" in text
    assert "requires context.guard_chain_observed" in text


def test_guard_chain_requires_pre_and_post_stage_bindings() -> None:
    chain = [
        {"name": "invariants", "stage": "wrong"},
        {"name": "spectral"},
        {"name": "rmt"},
        {"name": "invariants", "stage": "wrong"},
    ]
    report = {
        "assurance": {
            "canonical_guard_chain": chain,
            "guard_chain_observed": chain,
        },
        "plugins": {"guards": chain},
        "guards": chain,
        "context": {"guard_chain_observed": chain},
    }

    errors = facade.strict_guard_chain_errors(
        report,
        canonical_chain=("invariants", "spectral", "rmt", "invariants"),
    )

    assert "guards[0].stage must be pre for strict assurance." in errors
    assert "the final invariants guard stage must be post." in errors


def test_common_parsers_and_deduplication_reject_malformed_values() -> None:
    assert common._dedupe(["a", "a", "b"]) == ["a", "b"]
    assert common._nonnegative_int(True) is None
    assert common._nonnegative_int(-1) is None
    assert common._guard_base_name(7) is None
    assert common._guard_base_name("invariants_post") == "invariants"
    assert common._guard_inventory(
        {"guards": ["bad", {"name": "unknown"}, {"name": "rmt"}]}
    ) == [("rmt", {"name": "rmt"}, "guards[2]")]

    assert common._validate_diagnostics("guard", "bad") == [
        "guard.diagnostics must be an array."
    ]
    diagnostics = common._validate_diagnostics(
        "guard",
        ["bad", {}, {"severity": 3}, {"severity": "ERROR"}],
    )
    text = _messages(diagnostics)
    assert "diagnostics[0] must be an object" in text
    assert "diagnostics[1].severity is required" in text
    assert "diagnostics[2].severity must be a string" in text
    assert "diagnostics[3] records a blocking ERROR event" in text
    unsupported = common._validate_diagnostics("guard", [{"severity": "unknown"}])
    assert "severity is unsupported: unknown" in _messages(unsupported)


def test_guard_outcome_rejects_each_explicit_failure_surface() -> None:
    errors = common._validate_guard_outcome(
        "spectral",
        {
            "supported": True,
            "passed": False,
            "decision": "unknown",
            "status": "unknown",
            "violations": "bad",
            "failures": ["failure"],
            "assurance_blocking": "yes",
        },
        source="spectral",
        require_complete=True,
    )
    text = _messages(errors)
    assert "spectral.passed is false" in text
    assert "decision must be an allow/pass decision" in text
    assert "status must be a canonical passing status" in text
    assert "violations must be an array" in text
    assert "failures must be empty" in text
    assert "assurance_blocking must be a boolean" in text

    empty_fields = common._validate_guard_outcome(
        "rmt",
        {
            "supported": True,
            "passed": True,
            "decision": "",
            "status": "",
            "violations": [],
            "failures": "bad",
        },
        source="rmt",
        require_complete=True,
    )
    assert "decision must be a non-empty string" in _messages(empty_fields)
    assert "status must be a non-empty string" in _messages(empty_fields)
    assert "failures must be an array" in _messages(empty_fields)


def test_consistency_helpers_reject_untyped_and_conflicting_values() -> None:
    bool_errors: list[str] = []
    assert common._consistent_bool(bool_errors, [("a", 1)], required_path="a") is None
    assert "a must be a boolean." in bool_errors

    bool_errors = []
    assert common._consistent_bool(bool_errors, [("a", True), ("b", False)]) is None
    assert "a, b disagree" in _messages(bool_errors)

    int_errors: list[str] = []
    assert common._consistent_nonnegative_int(int_errors, [("a", -1)]) is None
    assert "a must be a non-negative integer." in int_errors

    int_errors = []
    assert common._consistent_nonnegative_int(int_errors, [("a", 1), ("b", 2)]) is None
    assert "a, b disagree" in _messages(int_errors)


@pytest.mark.parametrize(
    ("spectral", "inventory", "expected"),
    [
        ({"evaluated": "yes"}, [], "spectral.evaluated must be a boolean"),
        ({"evaluated": False}, [], "spectral.evaluated must be true"),
        (
            {
                "evaluated": True,
                "caps_applied": 1,
                "max_caps": "bad",
                "caps_exceeded": False,
                "summary": {"modules_checked": 1, "status": "capped"},
            },
            [],
            "caps cannot be accepted without a typed max_caps limit",
        ),
        (
            {
                "evaluated": True,
                "caps_applied": 0,
                "max_caps": 1,
                "caps_exceeded": False,
                "summary": {},
            },
            [],
            "summary.modules_checked is required",
        ),
        (
            {
                "evaluated": True,
                "caps_applied": 0,
                "max_caps": 1,
                "caps_exceeded": False,
                "summary": {"modules_checked": "one", "status": 1},
            },
            [],
            "summary.modules_checked must be an integer",
        ),
        (
            {
                "evaluated": True,
                "caps_applied": 0,
                "max_caps": 1,
                "caps_exceeded": False,
                "summary": {"modules_checked": 0, "status": "unstable"},
            },
            [],
            "summary.modules_checked must be positive",
        ),
        (
            {
                "evaluated": True,
                "caps_applied": 1,
                "max_caps": 2,
                "caps_exceeded": False,
                "summary": {"modules_checked": 1, "status": "stable"},
                "policy": {"max_caps": 2},
            },
            [
                (
                    "spectral",
                    {"policy": {"max_caps": 2}},
                    "guards[0]",
                )
            ],
            "status=stable contradicts caps_applied",
        ),
    ],
)
def test_spectral_reconciliation_rejects_malformed_or_contradictory_evidence(
    spectral, inventory, expected
) -> None:
    errors = matrix._spectral_errors(
        {"spectral": spectral}, inventory, require_complete=True
    )
    assert expected in _messages(errors)


def test_numeric_maps_reject_empty_bad_keys_and_nonfinite_values() -> None:
    errors: list[str] = []
    assert matrix._numeric_map(errors, [], path="values", require_nonempty=True) is None
    assert "values must be a non-empty object." in errors

    errors: list[str] = []
    assert matrix._numeric_map(errors, {}, path="values", require_nonempty=True) == {}
    assert "values must be a non-empty object." in errors

    errors = []
    result = matrix._numeric_map(
        errors,
        {"": 1.0, "ok": float("nan")},
        path="values",
        require_nonempty=True,
    )
    assert result == {}
    assert "keys must be non-empty strings" in _messages(errors)
    assert "values.ok must be a finite number" in _messages(errors)


def test_rmt_reconciliation_rejects_inconsistent_maps_and_family_details() -> None:
    rmt = {
        "evaluated": "yes",
        "stable": False,
        "epsilon_violations": "bad",
        "edge_risk_by_family_base": {"zero": 0.0, "missing": 1.0, "extra": 1.0},
        "edge_risk_by_family": {"zero": 0.1, "missing": 1.0, "other": 1.0},
        "epsilon_by_family": {"zero": 0.1},
        "epsilon_default": -1.0,
        "families": {
            "zero": "bad",
            "absent": {},
            "missing": {"edge_base": "bad", "edge_cur": 9.0},
        },
    }
    errors = matrix._rmt_errors({"rmt": rmt}, [], require_complete=True)
    text = _messages(errors)
    assert "rmt.evaluated must be a boolean" in text
    assert "rmt.stable is false" in text
    assert "rmt.epsilon_violations must be an array" in text
    assert "epsilon_default must be a finite non-negative number" in text
    assert "family sets must match exactly" in text
    assert "epsilon for family 'missing' is unavailable" in text
    assert "rmt.families.zero must be an object" in text
    assert "rmt.families.absent is absent from the edge-risk maps" in text
    assert "rmt.families.missing.edge_base must be finite" in text
    assert "rmt.families.missing.edge_cur disagrees with recomputation" in text


def test_matrix_reconciliation_ignores_absent_blocks_and_reads_inventory_metrics() -> (
    None
):
    assert matrix._spectral_errors({}, [], require_complete=True) == []
    assert matrix._rmt_errors({}, [], require_complete=True) == []

    spectral_errors = matrix._spectral_errors(
        {
            "spectral": {
                "evaluated": True,
                "caps_applied": 0,
                "max_caps": 1,
                "caps_exceeded": False,
                "summary": {"modules_checked": 1},
            }
        },
        [
            (
                "spectral",
                {"metrics": {"caps_applied": 1}, "policy": {"max_caps": 2}},
                "guards[0]",
            )
        ],
        require_complete=True,
    )
    assert "caps_applied disagree" in _messages(spectral_errors)

    rmt_errors = matrix._rmt_errors(
        {
            "rmt": {
                "evaluated": True,
                "stable": True,
                "epsilon_violations": [],
                "edge_risk_by_family_base": {"ffn": 1.0},
                "edge_risk_by_family": {"ffn": 1.0},
                "epsilon_by_family": {"ffn": 0.1},
                "epsilon_default": 0.1,
                "families": {},
            }
        },
        [("rmt", {"metrics": {"stable": False}}, "guards[0]")],
        require_complete=True,
    )
    assert "stable disagree" in _messages(rmt_errors)


def test_rmt_reconciliation_rejects_false_evaluation_violations_and_zero_basis() -> (
    None
):
    errors = matrix._rmt_errors(
        {
            "rmt": {
                "evaluated": False,
                "stable": True,
                "epsilon_violations": ["ffn"],
                "edge_risk_by_family_base": {"ffn": 0.0},
                "edge_risk_by_family": {"ffn": 0.0},
                "epsilon_by_family": {},
                "epsilon_default": 0.1,
                "families": {},
            }
        },
        [],
        require_complete=True,
    )
    text = _messages(errors)
    assert "rmt.evaluated must be true" in text
    assert "epsilon violations were recorded" in text
    assert "at least one positive baseline" in text


def test_invariant_variance_and_metric_impact_reconciliation_reject_bad_evidence() -> (
    None
):
    invariant_errors = runtime._invariants_errors(
        {
            "invariants": {
                "pre": 1,
                "post": "failed",
                "failures": "bad",
                "summary": {
                    "violations_found": -1,
                    "fatal_violations": 1,
                    "warning_violations": 0,
                    "checks_performed": "two",
                },
            }
        },
        require_complete=True,
    )
    text = _messages(invariant_errors)
    assert "invariants.pre must be a passing status" in text
    assert "invariants.post must be a passing status" in text
    assert "invariants.failures must be an array" in text
    assert "violations_found must be an integer" in text
    assert "fatal_violations must be zero" in text
    assert "checks_performed must be an integer" in text

    variance_errors = runtime._variance_errors(
        {
            "variance": {
                "enabled": "yes",
                "monitor_only": True,
                "predictive_gate": {"evaluated": False, "passed": False},
            }
        },
        require_complete=True,
    )
    text = _messages(variance_errors)
    assert "variance.enabled must be a boolean" in text
    assert "variance.monitor_only cannot pass" in text
    assert "predictive_gate.evaluated must be true" in text
    assert "predictive_gate.passed is false" in text

    impact_errors = runtime._guard_metric_impact_errors(
        {
            "guard_metric_impact": {
                "evaluated": "yes",
                "passed": False,
                "skipped": True,
                "mode": "SKIPPED",
                "degradation": 1.3,
                "degradation_limit": 0.1,
                "display_value": 1.0,
                "bare_value": -1.0,
                "guarded_value": 2.0,
                "checks": {"typed": "yes", "within_limit": False},
                "diagnostics": [{"severity": "fatal"}],
            }
        },
        require_complete=True,
    )
    text = _messages(impact_errors)
    assert "guard_metric_impact.evaluated must be a boolean" in text
    assert "guard_metric_impact.passed must be true" in text
    assert "requires measured guard_metric_impact evidence" in text
    assert "retained measurements are invalid or unsupported" in text
    assert "checks.typed must be a boolean" in text
    assert "checks.within_limit is false" in text
    assert "records a blocking fatal event" in text


def test_missing_strict_runtime_evidence_is_reported() -> None:
    assert runtime._invariants_errors({}, require_complete=True) == []
    assert runtime._variance_errors({}, require_complete=True) == []

    invariant_errors = runtime._invariants_errors(
        {"invariants": {"summary": {}}}, require_complete=True
    )
    assert "invariants.pre is required" in _messages(invariant_errors)
    assert "invariants.summary.checks_performed is required" in _messages(
        invariant_errors
    )

    variance_errors = runtime._variance_errors({"variance": {}}, require_complete=True)
    assert "variance.enabled is required" in _messages(variance_errors)
    assert "variance.predictive_gate is required" in _messages(variance_errors)

    assert runtime._guard_metric_impact_errors({}, require_complete=True) == [
        "strict assurance missing guard_metric_impact evidence."
    ]


def test_metric_impact_reconciliation_rejects_invalid_numeric_and_check_shapes() -> (
    None
):
    errors = runtime._guard_metric_impact_errors(
        {
            "guard_metric_impact": {
                "evaluated": True,
                "passed": True,
                "skipped": "no",
                "degradation": 0.0,
                "degradation_limit": -1.0,
                "display_value": float("nan"),
                "bare_value": 1.0,
                "guarded_value": 2.0,
                "checks": [],
            }
        },
        require_complete=False,
    )
    text = _messages(errors)
    assert "guard_metric_impact.skipped must be a boolean" in text
    assert "retained measurements are invalid or unsupported" in text
    assert "degradation_limit must be finite and non-negative" in text
    assert "guard_metric_impact.checks must be an object" in text


def test_runtime_reconciliation_covers_remaining_fail_closed_shapes() -> None:
    invariant_errors = runtime._invariants_errors(
        {
            "invariants": {
                "pre": "pass",
                "post": "pass",
                "failures": ["failure"],
                "summary": {"checks_performed": 0},
            }
        },
        require_complete=True,
    )
    text = _messages(invariant_errors)
    assert "invariants.failures must be empty" in text
    assert "invariants.summary.violations_found is required" in text
    assert "checks_performed must be positive" in text

    variance_errors = runtime._variance_errors(
        {
            "variance": {
                "enabled": True,
                "monitor_only": "no",
                "predictive_gate": {"evaluated": "yes", "passed": "yes"},
            }
        },
        require_complete=True,
    )
    text = _messages(variance_errors)
    assert "variance.monitor_only must be a boolean" in text
    assert "predictive_gate.evaluated must be a boolean" in text
    assert "predictive_gate.passed must be a boolean" in text

    missing_errors = runtime._guard_metric_impact_errors(
        {"guard_metric_impact": {"checks": {}}}, require_complete=True
    )
    text = _messages(missing_errors)
    assert "guard_metric_impact.evaluated is required" in text
    assert "guard_metric_impact.passed is required" in text
    assert "guard_metric_impact.degradation is required" in text
    assert "guard_metric_impact.degradation_limit is required" in text

    optional_errors = runtime._guard_metric_impact_errors(
        {
            "guard_metric_impact": {
                "evaluated": True,
                "passed": False,
                "degradation": 1.0,
                "degradation_limit": 1.0,
                "display_value": float("nan"),
                "bare_value": 1.0,
                "guarded_value": 0.0,
                "checks": {"failed": False},
            }
        },
        require_complete=False,
    )
    text = _messages(optional_errors)
    assert "guard_metric_impact.passed is false" in text
    assert "retained measurements are invalid or unsupported" in text
    assert "guard_metric_impact.checks.failed is false" in text


def test_metric_impact_reconciliation_recomputes_degradation_from_ppl() -> None:
    errors = runtime._guard_metric_impact_errors(
        {
            "guard_metric_impact": {
                "evaluated": True,
                "passed": True,
                "metric_kind": "ppl_causal",
                "direction": "lower",
                "degradation": 1.0,
                "degradation_basis": "relative_increase",
                "degradation_limit": 1.0,
                "display_value": 100.0,
                "display_unit": "percent",
                "bare_value": 2.0,
                "guarded_value": 3.0,
                "bare_facts": {
                    "weighted_logloss_sum": math.log(2.0),
                    "token_count": 1,
                },
                "guarded_facts": {
                    "weighted_logloss_sum": math.log(3.0),
                    "token_count": 1,
                },
            }
        },
        require_complete=False,
    )
    assert "degradation disagrees with retained measurements" in _messages(errors)


def test_variance_gain_reconciliation_rejects_incomplete_or_forged_claims() -> None:
    errors = runtime._variance_errors(
        {
            "variance": {
                "enabled": True,
                "monitor_only": False,
                "predictive_gate": {
                    "evaluated": True,
                    "passed": True,
                    "reason": "ci_gain_met",
                    "delta_ci": [0.2, -0.2],
                    "gain_ci": [9.0, 8.0],
                    "mean_delta": 0.1,
                },
                "policy": {
                    "min_effect_lognll": -1.0,
                    "predictive_one_sided": "yes",
                },
            }
        },
        require_complete=True,
    )
    errors.extend(
        runtime._variance_gain_errors(
            {"policy": {"min_effect_lognll": 0.0, "predictive_one_sided": True}},
            {
                "delta_ci": [-0.1, 0.2],
                "gain_ci": [-0.2, 0.1],
                "mean_delta": -0.01,
            },
            enabled=False,
        )
    )
    errors.extend(
        runtime._variance_gain_errors(
            {
                "policy": {"min_effect_lognll": 0.0, "predictive_one_sided": True},
                "calibration": {"status": "partial", "coverage": 0, "min_coverage": 1},
            },
            {"delta_ci": None, "gain_ci": None, "mean_delta": None},
            enabled=False,
        )
    )

    text = _messages(errors)
    for fragment in (
        "final variance.enabled=false",
        "non-negative finite policy.min_effect_lognll",
        "boolean policy.predictive_one_sided",
        "delta_ci must be ordered",
        "delta_ci strictly below zero",
        "negative mean_delta",
        "exact inverse of delta_ci",
        "requires variance.calibration evidence",
        "finite two-value delta_ci",
        "finite two-value gain_ci",
        "finite mean_delta",
        "calibration.status=complete",
        "adequate variance calibration coverage",
    ):
        assert fragment in text


def test_metric_impact_reconciliation_rejects_forged_canonical_metadata() -> None:
    errors = runtime._guard_metric_impact_errors(
        {
            "guard_metric_impact": {
                "evaluated": True,
                "passed": True,
                "metric_kind": "ppl_causal",
                "direction": "higher",
                "degradation_basis": "absolute_drop",
                "bare_value": 2.0,
                "guarded_value": 3.0,
                "bare_facts": {
                    "weighted_logloss_sum": math.log(2.0),
                    "token_count": 1,
                },
                "guarded_facts": {
                    "weighted_logloss_sum": math.log(3.0),
                    "token_count": 1,
                },
                "degradation": "not-a-number",
                "degradation_limit": 1.0,
                "display_value": 50.0,
                "display_unit": "percentage_points",
                "checks": {},
            }
        },
        require_complete=False,
    )

    text = _messages(errors)
    assert "direction disagrees with metric_kind" in text
    assert "degradation_basis disagrees with metric_kind" in text
    assert "degradation must be finite" in text
    assert "display_unit disagrees with metric_kind" in text
