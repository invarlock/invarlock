from __future__ import annotations

import copy

import pytest

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from invarlock.core.runner_runtime import guard_acceptance
from invarlock.core.types import RunStatus
from tests.core._support_spectral_replay import _bounded_report


def _observed_spectral_result() -> dict[str, object]:
    result = copy.deepcopy(_bounded_report()["guards"][0])
    result.update({"passed": False, "decision": "block"})
    result["policy"]["max_caps"] = 0
    result["metrics"].update(
        {
            "max_caps": 0,
            "caps_exceeded": True,
            "cap_budget_exceeded": True,
            "measurement_contract": {
                "estimator": {"type": "power_iter", "iters": 4, "init": "ones"},
                "degeneracy": copy.deepcopy(result["policy"]["degeneracy"]),
            },
        }
    )
    return result


def _observed_rmt_result() -> dict[str, object]:
    finding = {
        "family": "ffn",
        "edge_base": 1.0,
        "edge_cur": 1.02,
        "allowed": 1.01,
        "epsilon": 0.01,
        "delta": 0.02,
    }
    return {
        "passed": False,
        "decision": "block",
        "policy": {
            "epsilon_default": 0.01,
            "epsilon_by_family": {"ffn": 0.01},
        },
        "metrics": {
            "prepared": True,
            "stable": False,
            "edge_risk_by_family_base": {"ffn": 1.0},
            "edge_risk_by_family": {"ffn": 1.02},
            "edge_risk_by_module_base": {"layer.0.mlp": 1.0},
            "edge_risk_by_module": {"layer.0.mlp": 1.02},
            "module_family_map": {"layer.0.mlp": "ffn"},
            "epsilon_by_family": {"ffn": 0.01},
            "epsilon_violations": [copy.deepcopy(finding)],
            "measurement_contract": {"estimator": {"type": "power_iter"}},
        },
        "violations": [
            {
                "type": "epsilon_band",
                "severity": "error",
                **finding,
            }
        ],
    }


def _all_observe_authority() -> dict[str, str]:
    return {
        "spectral": "observe",
        "rmt": "observe",
        "variance": "observe",
    }


def _observed_variance_result() -> dict[str, object]:
    return {
        "passed": False,
        "decision": "block",
        "metrics": {
            "monitor_only": False,
            "predictive_gate": {
                "evaluated": True,
                "passed": False,
                "reason": "gain_below_threshold",
            },
            "calibration": {
                "status": "complete",
                "coverage": 8,
                "min_coverage": 6,
            },
        },
    }


@pytest.mark.parametrize(
    "contradiction",
    [
        {"supported": False},
        {"assurance_blocking": True},
        {"status": "degraded"},
        {"status": "unsupported"},
        {"errors": ["measurement failed"]},
        {"diagnostics": [{"severity": "error", "kind": "measurement_failed"}]},
    ],
)
def test_finalize_rejects_contradictory_passing_guard_evidence(
    contradiction: dict[str, object],
) -> None:
    runner = CoreRunner()
    report = RunReport()
    result = {"passed": True, **contradiction}

    status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": result},
        {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}},
        RunConfig(),
        report,
    )

    assert status == RunStatus.ROLLBACK.value


def test_finalize_observe_allows_finding_but_not_degraded_evidence() -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {
        "guard_authority": {
            "spectral": "observe",
            "rmt": "enforce",
            "variance": "enforce",
        }
    }
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}}

    status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": _observed_spectral_result()},
        metrics,
        RunConfig(),
        report,
    )
    assert status == RunStatus.SUCCESS.value

    status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": {"passed": False, "decision": "block", "violations": []}},
        metrics,
        RunConfig(),
        report,
    )
    assert status == RunStatus.ROLLBACK.value

    incomplete = _observed_spectral_result()
    incomplete["metrics"].pop("measurement_contract")
    status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": incomplete},
        metrics,
        RunConfig(),
        report,
    )
    assert status == RunStatus.ROLLBACK.value

    status = runner._finalize_phase(
        object(),
        object(),
        {
            "spectral": {
                "passed": False,
                "decision": "block",
                "supported": False,
                "status": "unsupported",
            }
        },
        metrics,
        RunConfig(),
        report,
    )
    assert status == RunStatus.ROLLBACK.value


@pytest.mark.parametrize(
    ("guard_name", "mutation"),
    [
        ("spectral", "missing-raw-section"),
        ("spectral", "fatal-finding"),
        ("spectral", "cap-facts-disagree"),
        ("rmt", "missing-raw-section"),
        ("rmt", "epsilon-facts-disagree"),
        ("rmt", "family-aggregate-disagrees"),
    ],
)
def test_finalize_observe_rejects_non_replayable_guard_findings(
    guard_name: str,
    mutation: str,
) -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {
        "guard_authority": {
            "spectral": "observe",
            "rmt": "observe",
            "variance": "enforce",
        }
    }
    result = (
        _observed_spectral_result()
        if guard_name == "spectral"
        else _observed_rmt_result()
    )
    if mutation == "missing-raw-section":
        if guard_name == "spectral":
            result.pop("final_z_scores")
        else:
            result["metrics"].pop("edge_risk_by_module")
    elif mutation == "fatal-finding":
        result["violations"][0]["severity"] = "fatal"
        result["metrics"]["fatal_violations"] = 1
    elif mutation == "cap-facts-disagree":
        result["metrics"]["caps_exceeded"] = False
    elif mutation == "epsilon-facts-disagree":
        result["violations"][0]["allowed"] = 1.0
    elif mutation == "family-aggregate-disagrees":
        result["metrics"]["edge_risk_by_family"]["ffn"] = 1.01

    status = runner._finalize_phase(
        object(),
        object(),
        {guard_name: result},
        {"primary_metric": {"kind": "accuracy", "final": 1.0}},
        RunConfig(),
        report,
    )

    assert status == RunStatus.ROLLBACK.value


def test_finalize_observe_accepts_complete_replayed_rmt_finding() -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {
        "guard_authority": {
            "spectral": "enforce",
            "rmt": "observe",
            "variance": "enforce",
        }
    }

    status = runner._finalize_phase(
        object(),
        object(),
        {"rmt": _observed_rmt_result()},
        {"primary_metric": {"kind": "accuracy", "final": 1.0}},
        RunConfig(),
        report,
    )

    assert status == RunStatus.SUCCESS.value


def test_finalize_observe_never_applies_to_invariants() -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {
        "guard_authority": {
            "spectral": "observe",
            "rmt": "observe",
            "variance": "observe",
        }
    }
    status = runner._finalize_phase(
        object(),
        object(),
        {"invariants": {"passed": False, "decision": "block"}},
        {"primary_metric": {"kind": "accuracy", "final": 1.0}},
        RunConfig(),
        report,
    )
    assert status == RunStatus.ROLLBACK.value


@pytest.mark.parametrize(
    "authority",
    [
        {"rmt": "observe"},
        {
            "spectral": "enforce",
            "rmt": "enforce",
            "variance": "enforce",
            "unknown": "observe",
        },
        {
            "spectral": "ignore",
            "rmt": "enforce",
            "variance": "enforce",
        },
    ],
)
def test_finalize_rejects_malformed_explicit_guard_authority(
    authority: dict[str, str],
) -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {"guard_authority": authority}

    status = runner._finalize_phase(
        object(),
        object(),
        {"rmt": {"passed": True}},
        {"primary_metric": {"kind": "accuracy", "final": 1.0}},
        RunConfig(),
        report,
    )

    assert status == RunStatus.ROLLBACK.value


def test_finalize_observed_variance_requires_complete_negative_measurement() -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {
        "guard_authority": {
            "spectral": "enforce",
            "rmt": "enforce",
            "variance": "observe",
        }
    }
    metrics = {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.0}}
    observed = {
        "passed": False,
        "decision": "block",
        "metrics": {
            "monitor_only": False,
            "predictive_gate": {
                "evaluated": True,
                "passed": False,
                "reason": "gain_below_threshold",
            },
            "calibration": {
                "status": "complete",
                "coverage": 8,
                "min_coverage": 6,
            },
        },
    }

    assert (
        runner._finalize_phase(
            object(),
            object(),
            {"variance": observed},
            metrics,
            RunConfig(),
            report,
        )
        == RunStatus.SUCCESS.value
    )

    observed["metrics"]["calibration"]["coverage"] = 0
    assert (
        runner._finalize_phase(
            object(),
            object(),
            {"variance": observed},
            metrics,
            RunConfig(),
            report,
        )
        == RunStatus.ROLLBACK.value
    )


@pytest.mark.parametrize("value", [True, "1", None, float("inf"), float("nan")])
def test_guard_acceptance_rejects_non_finite_or_non_numeric_facts(
    value: object,
) -> None:
    assert guard_acceptance._finite_number(value) is None


@pytest.mark.parametrize("value", [True, -1, 1.0, "1"])
def test_guard_acceptance_rejects_non_count_facts(value: object) -> None:
    assert guard_acceptance._nonnegative_int(value) is None


@pytest.mark.parametrize(
    "value",
    [
        {1: 1.0},
        {"": 1.0},
        {"layer": float("nan")},
    ],
)
def test_guard_acceptance_rejects_malformed_numeric_maps(value: object) -> None:
    assert guard_acceptance._numeric_map(value) is None


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("module-set", {"layer.0": "ffn"}),
        ("family-caps", {}),
        ("selected-families", "ffn"),
    ],
)
def test_observed_spectral_rejects_incomplete_outer_replay_contract(
    mutation: str,
    value: object,
) -> None:
    result = _observed_spectral_result()
    if mutation == "module-set":
        result["module_family_map"] = value
    elif mutation == "family-caps":
        result["policy"]["family_caps"] = value
    else:
        result["metrics"]["multiple_testing_selection"]["families_selected"] = value

    assert not guard_acceptance.guard_result_is_acceptable(
        "spectral", result, _all_observe_authority()
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("severity", "fatal"),
        ("module", "missing.layer"),
        ("z_score", None),
        ("type", "unknown_budgeted_finding"),
        ("family_pvalue", 0.5),
    ],
)
def test_observed_spectral_rejects_non_replayable_finding_facts(
    mutation: str,
    value: object,
) -> None:
    result = _observed_spectral_result()
    if mutation == "family_pvalue":
        result["metrics"]["multiple_testing_selection"]["family_pvalues"]["ffn"] = value
    else:
        result["violations"][0][mutation] = value

    assert not guard_acceptance.guard_result_is_acceptable(
        "spectral", result, _all_observe_authority()
    )


@pytest.mark.parametrize(
    ("finding_type", "metric"),
    [
        ("degeneracy_stable_rank_drop", "stable_rank"),
        ("degeneracy_norm_collapse", "norm_collapse"),
    ],
)
def test_observed_spectral_accepts_complete_degeneracy_finding(
    finding_type: str,
    metric: str,
) -> None:
    result = _observed_spectral_result()
    result["baseline_metrics"]["baseline_degeneracy"] = {"layer.0": {metric: 4.0}}
    result["final_degeneracy"] = {"layer.0": {metric: 1.6}}
    result["violations"] = [
        {
            "type": finding_type,
            "severity": "budgeted",
            "module": "layer.0",
            "family": "ffn",
            "selected": True,
            f"{metric}_base": 4.0,
            f"{metric}_cur": 1.6,
            "ratio": 0.4,
            "warn_ratio": 0.5,
            "fatal_ratio": 0.25,
        }
    ]

    assert guard_acceptance.guard_result_is_acceptable(
        "spectral", result, _all_observe_authority()
    )

    result["violations"][0]["ratio"] = 0.45
    assert not guard_acceptance.guard_result_is_acceptable(
        "spectral", result, _all_observe_authority()
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("initial-contract", {}),
        ("module-family", ""),
        ("policy-default", -0.01),
        ("policy-family", 0.02),
        ("metric-finding", None),
    ],
)
def test_observed_rmt_rejects_malformed_replay_contract(
    mutation: str,
    value: object,
) -> None:
    result = _observed_rmt_result()
    if mutation == "initial-contract":
        result["metrics"] = value
    elif mutation == "module-family":
        result["metrics"]["module_family_map"]["layer.0.mlp"] = value
    elif mutation == "policy-default":
        result["policy"]["epsilon_default"] = value
    elif mutation == "policy-family":
        result["policy"]["epsilon_by_family"]["ffn"] = value
    else:
        result["metrics"]["epsilon_violations"][0] = value

    assert not guard_acceptance.guard_result_is_acceptable(
        "rmt", result, _all_observe_authority()
    )


def test_rmt_replay_does_not_invent_a_violation_for_zero_baseline() -> None:
    assert (
        guard_acceptance._expected_rmt_violation_families(
            {"ffn": 0.0},
            {"ffn": 100.0},
            {"ffn": 0.01},
            {"ffn": 0.01},
            0.01,
        )
        == set()
    )


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"passed": "yes"},
        {"passed": True, "status": "monitor_only"},
        {"passed": True, "supported": False},
        {"passed": True, "assurance_blocking": True},
    ],
)
def test_guard_acceptance_fails_closed_on_malformed_or_degraded_results(
    result: object,
) -> None:
    assert not guard_acceptance.guard_result_is_acceptable(
        "spectral", result, _all_observe_authority()
    )


def test_guard_acceptance_allows_clean_passing_unknown_guard_only() -> None:
    authority = _all_observe_authority()
    assert guard_acceptance.guard_result_is_acceptable(
        "custom_guard", {"passed": True}, authority
    )
    assert not guard_acceptance.guard_result_is_acceptable(
        "custom_guard",
        {"passed": False, "decision": "block"},
        authority,
    )
    assert not guard_acceptance.guard_result_is_acceptable(
        "guard_metric_impact",
        {"passed": False, "decision": "block"},
        authority,
    )


def test_observe_does_not_accept_variance_monitor_only_or_missing_facts() -> None:
    authority = _all_observe_authority()
    observed = _observed_variance_result()
    assert guard_acceptance.guard_result_is_acceptable("variance", observed, authority)

    observed["metrics"]["monitor_only"] = True
    assert not guard_acceptance.guard_result_is_acceptable(
        "variance", observed, authority
    )

    for malformed in (
        {"passed": False, "decision": "block"},
        {"passed": False, "decision": "block", "metrics": {}},
        {
            "passed": False,
            "decision": "allow",
            "metrics": _observed_variance_result()["metrics"],
        },
    ):
        assert not guard_acceptance.guard_result_is_acceptable(
            "variance", malformed, authority
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("predictive_gate", "evaluated", False),
        ("predictive_gate", "passed", True),
        ("predictive_gate", "reason", "unsupported_reason"),
        ("calibration", "status", "partial"),
        ("calibration", "coverage", True),
        ("calibration", "min_coverage", True),
        ("calibration", "min_coverage", 0),
    ],
)
def test_observed_variance_rejects_incomplete_or_degraded_measurements(
    section: str,
    field: str,
    value: object,
) -> None:
    observed = _observed_variance_result()
    observed["metrics"][section][field] = value

    assert not guard_acceptance.guard_result_is_acceptable(
        "variance", observed, _all_observe_authority()
    )


def test_observe_guard_finding_never_overrides_primary_metric_drift() -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {"guard_authority": _all_observe_authority()}

    status = runner._finalize_phase(
        object(),
        object(),
        {"spectral": _observed_spectral_result()},
        {"primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 3.0}},
        RunConfig(),
        report,
    )

    assert status == RunStatus.ROLLBACK.value
    assert str(report.meta["rollback_reason"]).startswith("catastrophic_ppl_spike")


def test_observe_guard_finding_never_overrides_invalid_primary_metric() -> None:
    runner = CoreRunner()
    report = RunReport()
    report.meta["tier_policies"] = {"guard_authority": _all_observe_authority()}

    status = runner._finalize_phase(
        object(),
        object(),
        {"rmt": _observed_rmt_result()},
        {"primary_metric": {"kind": "accuracy", "final": 1.0, "invalid": True}},
        RunConfig(),
        report,
    )

    assert status == RunStatus.ROLLBACK.value
    assert report.meta["rollback_reason"] == "primary_metric_invalid"
