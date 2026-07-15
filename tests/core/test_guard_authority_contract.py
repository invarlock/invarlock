from __future__ import annotations

import copy

from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET,
    ASSURANCE_CLAIM_SET_V2,
    LEGACY_ASSURANCE_CLAIM_SET,
    build_assurance_section,
    strict_report_policy_errors,
)
from invarlock.core.assurance_guard_validation import guard_evidence_policy_errors
from tests.core._support_assurance_contract import (
    _sync_variance_guard_metrics,
    strict_report,
    strict_variance_gain_report,
)

ALL_ENFORCE = {
    "spectral": "enforce",
    "rmt": "enforce",
    "variance": "enforce",
}


def _with_authority(report: dict, **overrides: str) -> dict:
    report = copy.deepcopy(report)
    authority = {**ALL_ENFORCE, **overrides}
    report.setdefault("resolved_policy", {})["guard_authority"] = authority
    report["assurance"] = build_assurance_section(report)
    return report


def test_v2_claim_binds_authority_while_legacy_is_enforce_all() -> None:
    assert ASSURANCE_CLAIM_SET == ASSURANCE_CLAIM_SET_V2

    legacy = strict_report()
    legacy["assurance"] = build_assurance_section(legacy)
    assert legacy["assurance"]["claim_set"] == LEGACY_ASSURANCE_CLAIM_SET
    assert "guard_authority" not in legacy["assurance"]

    for explicit_authority in (copy.deepcopy(ALL_ENFORCE), None):
        malformed_legacy = copy.deepcopy(legacy)
        malformed_legacy["assurance"]["guard_authority"] = explicit_authority
        assert any(
            "legacy strict assurance cannot declare guard_authority" in error
            for error in strict_report_policy_errors(
                malformed_legacy, require_strict=True
            )
        )

    current = _with_authority(strict_report(), spectral="observe")
    assert current["assurance"]["claim_set"] == ASSURANCE_CLAIM_SET_V2
    assert current["assurance"]["guard_authority"]["spectral"] == "observe"

    current["assurance"]["guard_authority"]["spectral"] = "enforce"
    assert any(
        "guard_authority" in error
        for error in strict_report_policy_errors(current, require_strict=True)
    )


def test_observe_waives_only_replayed_rmt_threshold_outcome() -> None:
    report = strict_report()
    report["rmt"].update(
        {
            "passed": False,
            "decision": "block",
            "status": "unstable",
            "stable": False,
            "epsilon_violations": [
                {
                    "family": "ffn",
                    "edge_base": 1.0,
                    "edge_cur": 1.02,
                    "allowed": 1.01,
                    "epsilon": 0.01,
                    "delta": 0.02,
                }
            ],
            "edge_risk_by_family": {"ffn": 1.02},
        }
    )
    report["rmt"]["families"]["ffn"].update(
        {"edge_cur": 1.02, "ratio": 1.02, "delta": 0.02}
    )
    rmt_guard = next(entry for entry in report["guards"] if entry["name"] == "rmt")
    rmt_guard.update(
        {
            "passed": False,
            "decision": "block",
            "violations": [
                {
                    "type": "epsilon_band",
                    "severity": "error",
                    "family": "ffn",
                }
            ],
            "diagnostics": [
                {
                    "kind": "epsilon_band",
                    "severity": "error",
                    "message": "measured epsilon violation",
                }
            ],
        }
    )
    rmt_guard["metrics"]["stable"] = False
    rmt_guard["metrics"]["edge_risk_by_family"] = {"ffn": 1.02}
    rmt_guard["metrics"]["edge_risk_by_module"] = {"layer.0.mlp": 1.02}
    rmt_guard["metrics"]["epsilon_violations"] = copy.deepcopy(
        report["rmt"]["epsilon_violations"]
    )
    report["validation"]["rmt_stable"] = False

    enforced = _with_authority(report)
    assert any(
        "rmt" in error
        for error in guard_evidence_policy_errors(enforced, require_complete=True)
    )

    observed = _with_authority(report, rmt="observe")
    assert guard_evidence_policy_errors(observed, require_complete=True) == []

    observed["rmt"]["supported"] = False
    assert any(
        "unsupported" in error
        for error in guard_evidence_policy_errors(observed, require_complete=True)
    )


def test_observe_accepts_complete_variance_negative_but_not_insufficient_coverage() -> (
    None
):
    report = strict_variance_gain_report()
    report["variance"]["policy"]["min_effect_lognll"] = 0.03
    report["variance"].update(
        {
            "passed": False,
            "decision": "block",
            "met_threshold": False,
        }
    )
    report["variance"]["predictive_gate"].update(
        {"passed": False, "reason": "gain_below_threshold"}
    )
    variance_guard = next(
        entry for entry in report["guards"] if entry["name"] == "variance"
    )
    variance_guard.update(
        {
            "passed": False,
            "decision": "block",
            "violations": [
                {
                    "type": "variance_error",
                    "severity": "error",
                    "message": "predictive gain below threshold",
                }
            ],
            "diagnostics": [
                {
                    "kind": "variance_error",
                    "severity": "error",
                    "message": "predictive gain below threshold",
                }
            ],
        }
    )
    _sync_variance_guard_metrics(report)

    observed = _with_authority(report, variance="observe")
    assert guard_evidence_policy_errors(observed, require_complete=True) == []

    observed["variance"]["calibration"]["coverage"] = 0
    _sync_variance_guard_metrics(observed)
    observed["assurance"] = build_assurance_section(observed)
    assert any(
        "coverage" in error or "monitor" in error
        for error in guard_evidence_policy_errors(observed, require_complete=True)
    )


def test_observe_never_waives_invariants_or_guard_metric_impact() -> None:
    report = _with_authority(strict_report(), spectral="observe", rmt="observe")
    report["validation"]["invariants_pass"] = False
    assert any(
        "invariants_pass" in error
        for error in guard_evidence_policy_errors(report, require_complete=True)
    )

    report = _with_authority(strict_report(), spectral="observe", rmt="observe")
    report["validation"]["guard_metric_impact_acceptable"] = False
    assert any(
        "guard_metric_impact_acceptable" in error
        for error in guard_evidence_policy_errors(report, require_complete=True)
    )
