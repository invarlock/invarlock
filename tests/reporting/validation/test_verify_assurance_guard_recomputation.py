from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest

from invarlock.core.assurance_contract import (
    CANONICAL_GUARD_CHAIN,
)
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports
from tests.cli.verify._support_runtime_provenance import bind_runtime_policy_receipt
from tests.reporting.validation._support_verify_assurance_guard_chain import (
    _report,
    _run_strict,
    _verified_runtime,
    _write_report,
)


@pytest.mark.parametrize(
    ("guard_name", "replacement", "diagnostic_fragment"),
    [
        pytest.param(
            "spectral",
            {"supported": True, "status": "pass", "caps_exceeded": True},
            "caps_exceeded",
            id="spectral-caps-exceeded",
        ),
        pytest.param(
            "spectral",
            {
                "supported": True,
                "status": "pass",
                "caps_applied": 999,
                "max_caps": 5,
            },
            "max_caps",
            id="spectral-over-cap-limit",
        ),
        pytest.param(
            "spectral",
            {"supported": True, "status": "fail"},
            "spectral",
            id="spectral-failing-status",
        ),
        pytest.param(
            "rmt",
            {"supported": True, "status": "pass", "stable": False},
            "stable",
            id="rmt-explicitly-unstable",
        ),
        pytest.param(
            "rmt",
            {
                "supported": True,
                "status": "pass",
                "stable": True,
                "epsilon_violations": [{"family": "ffn"}],
            },
            "epsilon_violations",
            id="rmt-epsilon-violations",
        ),
        pytest.param(
            "variance",
            {"supported": True},
            "variance",
            id="variance-supported-only",
        ),
        pytest.param(
            "invariants",
            {"supported": True},
            "invariants",
            id="invariants-supported-only",
        ),
        pytest.param(
            "variance",
            {
                "supported": "false",
                "passed": "false",
                "decision": "BLOCK",
                "status": "PASS",
            },
            "variance",
            id="truthy-string-outcome-values",
        ),
        pytest.param(
            "invariants",
            {
                "supported": True,
                "status": "pass",
                "passed": True,
                "decision": "allow",
                "violations": [{"check": "finite_parameters"}],
            },
            "violations",
            id="invariant-violation-hidden-by-pass",
        ),
    ],
)
def test_verify_assurance_strict_recomputes_guard_outcomes_from_raw_evidence(
    tmp_path: Path,
    monkeypatch,
    guard_name: str,
    replacement: dict,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload[guard_name] = copy.deepcopy(replacement)
    validation_key = {
        "spectral": "spectral_stable",
        "rmt": "rmt_stable",
        "invariants": "invariants_pass",
    }.get(guard_name)
    if validation_key is not None:
        payload["validation"][validation_key] = True
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize(
    ("mutation", "diagnostic_fragment"),
    [
        pytest.param(
            "spectral-summary-disagrees",
            "disagree across guard evidence",
            id="spectral-summary-cannot-hide-cap-exhaustion",
        ),
        pytest.param(
            "rmt-inequality-fails",
            "rmt acceptance inequality failed",
            id="rmt-stable-label-cannot-hide-edge-growth",
        ),
        pytest.param(
            "invariant-summary-violation",
            "invariants.summary.violations_found must be zero",
            id="invariant-pass-label-cannot-hide-summary-violation",
        ),
        pytest.param(
            "variance-predictive-failure",
            "variance.predictive_gate.passed is false",
            id="variance-pass-label-cannot-hide-predictive-failure",
        ),
        pytest.param(
            "inventory-missing-passed",
            "guards[1].passed is required for strict assurance",
            id="inventory-outcome-must-be-complete",
        ),
        pytest.param(
            "inventory-string-passed",
            "guards[1].passed must be a boolean",
            id="inventory-outcome-must-be-typed",
        ),
        pytest.param(
            "validation-false",
            "validation.primary_metric_acceptable is false",
            id="validation-mirror-cannot-be-false",
        ),
        pytest.param(
            "assurance-blocking",
            "spectral.assurance_blocking is true",
            id="blocking-outcome-cannot-pass",
        ),
        pytest.param(
            "missing-invariant-stages",
            "guards[0].stage must be pre",
            id="duplicate-invariants-must-be-stage-bound",
        ),
    ],
)
def test_verify_assurance_strict_rejects_cross_surface_guard_contradictions(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    if mutation == "spectral-summary-disagrees":
        payload["spectral"]["summary"]["caps_exceeded"] = True
    elif mutation == "rmt-inequality-fails":
        payload["rmt"]["edge_risk_by_family"]["ffn"] = 1.02
        payload["rmt"]["families"]["ffn"].update(
            {
                "edge_cur": 1.02,
                "ratio": 1.02,
                "delta": 0.02,
            }
        )
    elif mutation == "invariant-summary-violation":
        payload["invariants"]["summary"]["violations_found"] = 1
    elif mutation == "variance-predictive-failure":
        payload["variance"]["predictive_gate"]["passed"] = False
    elif mutation == "inventory-missing-passed":
        payload["guards"][1].pop("passed")
    elif mutation == "inventory-string-passed":
        payload["guards"][1]["passed"] = "true"
    elif mutation == "validation-false":
        payload["validation"]["primary_metric_acceptable"] = False
    elif mutation == "assurance-blocking":
        payload["spectral"]["assurance_blocking"] = True
    elif mutation == "missing-invariant-stages":
        payload["guards"][0].pop("stage")
        payload["guards"][-1].pop("stage")
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize(
    ("guard_index", "field", "diagnostic_fragment"),
    [
        pytest.param(1, "metrics", "guards[1].metrics", id="spectral-metrics"),
        pytest.param(
            1, "baseline_metrics", "guards[1].baseline_metrics", id="spectral-baseline"
        ),
        pytest.param(2, "policy", "guards[2].policy", id="rmt-policy"),
        pytest.param(0, "details", "guards[0].details", id="invariants-details"),
    ],
)
def test_verify_assurance_strict_requires_raw_guard_observations(
    tmp_path: Path,
    monkeypatch,
    guard_index: int,
    field: str,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["guards"][guard_index].pop(field)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


def test_verify_assurance_strict_accepts_replayed_bounded_spectral_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    p_value = math.erfc(2.0 / math.sqrt(2.0))
    violation = {
        "type": "family_z_cap",
        "severity": "budgeted",
        "module": "layer.0",
        "family": "ffn",
        "z_score": 2.0,
        "kappa": 1.0,
        "sigma": 1.2,
        "baseline_sigma": 1.0,
        "p_value": p_value,
        "selected": True,
        "message": "bounded spectral correction",
    }
    raw = payload["guards"][1]
    raw.update({"passed": True, "decision": "monitor", "violations": [violation]})
    raw["policy"]["family_caps"]["ffn"]["kappa"] = 1.0
    raw["policy"]["multiple_testing"] = {
        "method": "bonferroni",
        "alpha": 0.05,
        "m": 1,
    }
    raw["policy"]["correction_enabled"] = False
    raw["final_metrics"]["layer.0"] = 1.2
    raw["final_z_scores"]["layer.0"] = 2.0
    raw["metrics"].update(
        {
            "violations_detected": 1,
            "candidate_violations_detected": 1,
            "candidate_budgeted_violations": 1,
            "budgeted_violations": 1,
            "caps_applied": 1,
            "caps_exceeded": False,
            "max_spectral_norm": 1.2,
            "mean_spectral_norm": 1.1,
            "family_caps": copy.deepcopy(raw["policy"]["family_caps"]),
            "multiple_testing": copy.deepcopy(raw["policy"]["multiple_testing"]),
            "multiple_testing_selection": {
                "method": "bonferroni",
                "alpha": 0.05,
                "m": 1,
                "families_tested": ["ffn"],
                "families_selected": ["ffn"],
                "family_pvalues": {"ffn": p_value},
                "family_max_abs_z": {"ffn": 2.0},
                "family_violation_counts": {"ffn": 1},
                "default_selected_without_pvalue": 0,
            },
            "selected_budgeted_findings": 1,
            "cap_budget_exceeded": False,
            "corrections_attempted": 0,
            "corrections_applied": 0,
            "correction_policy_result": "correction_disabled",
            "identity_changed_modules": [],
            "measurement_exclusions": [],
            "discovery_errors": [],
        }
    )
    finding = copy.deepcopy(violation)
    finding["finding_id"] = "finding-0001:family_z_cap:layer.0"
    raw["correction_ledger"] = {
        "schema_version": 1,
        "phase": "validate",
        "correction_enabled": False,
        "correction_cap_ratio": 2.0,
        "pre_correction_metrics": {"layer.0": 1.2, "layer.1": 1.0},
        "pre_correction_z_scores": {"layer.0": 2.0, "layer.1": 0.0},
        "pre_correction_degeneracy": {},
        "multiple_testing_selection": copy.deepcopy(
            raw["metrics"]["multiple_testing_selection"]
        ),
        "selected_findings": [finding],
        "corrections": [
            {
                "correction_id": "correction-0001:layer.0",
                "finding_ids": [finding["finding_id"]],
                "module": "layer.0",
                "operation": "none",
                "attempted": False,
                "mutation_applied": False,
                "outcome": "not_attempted_policy_disabled",
                "pre_sigma": 1.2,
                "baseline_sigma": 1.0,
                "post_sigma": 1.2,
                "scale_factor": 1.0,
                "pre_weight_digest": "a" * 64,
                "post_weight_digest": "a" * 64,
            }
        ],
        "policy_result": "correction_disabled",
        "post_correction_metrics": {"layer.0": 1.2, "layer.1": 1.0},
    }
    payload["spectral"].update(
        {
            "passed": True,
            "decision": "monitor",
            "status": "capped",
            "caps_applied": 1,
            "violations": [violation],
        }
    )
    payload["spectral"]["summary"].update(
        {"status": "capped", "caps_applied": 1, "caps_exceeded": False}
    )
    bind_runtime_policy_receipt(payload)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.OK, "\n".join(
        item.message for item in result.diagnostics
    )


def test_verify_assurance_strict_rebuilds_rmt_family_risk_from_modules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["guards"][2]["metrics"]["edge_risk_by_module"]["layer.0.mlp"] = 2.0
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "family current value disagrees with module evidence" in diagnostics


@pytest.mark.parametrize(
    ("replacement", "diagnostic_fragment", "expected_outcome"),
    [
        pytest.param(
            {
                "evaluated": True,
                "degradation": 1.0,
                "passed": False,
            },
            "guard_metric_impact.passed",
            VerifyOutcome.POLICY_FAIL,
            id="explicit-fail",
        ),
        pytest.param(
            {
                "evaluated": True,
                "degradation": 1.50,
                "degradation_limit": 0.01,
                "passed": True,
            },
            "degradation_limit",
            VerifyOutcome.POLICY_FAIL,
            id="ratio-over-limit",
        ),
        pytest.param(
            {
                "evaluated": "true",
                "degradation": 1.0,
                "passed": "true",
            },
            "guard_metric_impact.evaluated",
            VerifyOutcome.MALFORMED,
            id="truthy-strings",
        ),
    ],
)
def test_verify_assurance_strict_recomputes_guard_metric_impact_outcome(
    tmp_path: Path,
    monkeypatch,
    replacement: dict,
    diagnostic_fragment: str,
    expected_outcome: VerifyOutcome,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["guard_metric_impact"].update(copy.deepcopy(replacement))
    payload["validation"]["guard_metric_impact_acceptable"] = True
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == expected_outcome
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize(
    ("field", "value", "delete", "diagnostic_fragment"),
    [
        pytest.param(
            "verdict",
            "pass",
            False,
            "assurance.verdict",
            id="premature-pass-verdict",
        ),
        pytest.param(
            "report_local_verdict",
            "fail",
            False,
            "report_local_verdict",
            id="local-fail-hidden-by-pass",
        ),
        pytest.param(
            "verified_assurance_verdict",
            "fail",
            False,
            "verified_assurance_verdict",
            id="verified-fail-hidden-by-pass",
        ),
        pytest.param(
            "fallback_fields_used",
            None,
            True,
            "fallback_fields_used",
            id="missing-fallback-state",
        ),
        pytest.param(
            "report_local_verdict",
            None,
            True,
            "report_local_verdict",
            id="missing-local-state",
        ),
        pytest.param(
            "verified_assurance_verdict",
            None,
            True,
            "verified_assurance_verdict",
            id="missing-verified-state",
        ),
        pytest.param(
            "runtime_provenance_verified",
            True,
            False,
            "runtime_provenance_verified",
            id="submitted-runtime-preverified",
        ),
        pytest.param(
            "runtime_provenance_verification_status",
            "verified",
            False,
            "runtime_provenance_verification_status",
            id="submitted-runtime-status-verified",
        ),
    ],
)
def test_verify_assurance_strict_requires_canonical_pre_verifier_state(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
    delete: bool,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    if delete:
        payload["assurance"].pop(field)
    else:
        payload["assurance"][field] = value
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize(
    "mutation",
    [
        "delete-canonical",
        "delete-observed",
        "delete-plugin-chain",
        "delete-inventory",
        "reverse-plugin-chain",
        "reverse-inventory",
        "divergent-observed",
        "divergent-context",
    ],
)
def test_verify_assurance_strict_reconciles_every_guard_chain_representation(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["context"]["guard_chain_observed"] = list(CANONICAL_GUARD_CHAIN)
    if mutation == "delete-canonical":
        payload["assurance"].pop("canonical_guard_chain")
    elif mutation == "delete-observed":
        payload["assurance"].pop("guard_chain_observed")
    elif mutation == "delete-plugin-chain":
        payload["plugins"].pop("guards")
    elif mutation == "delete-inventory":
        payload.pop("guards")
    elif mutation == "reverse-plugin-chain":
        payload["plugins"]["guards"] = list(reversed(CANONICAL_GUARD_CHAIN))
    elif mutation == "reverse-inventory":
        payload["guards"] = list(reversed(payload["guards"]))
    elif mutation == "divergent-observed":
        payload["assurance"]["guard_chain_observed"] = ["invariants", "rmt"]
    elif mutation == "divergent-context":
        payload["context"]["guard_chain_observed"] = ["spectral"]
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "guard chain" in diagnostics.lower()


@pytest.mark.parametrize(
    ("caller_profile", "diagnostic_fragment"),
    [
        pytest.param(
            "dev",
            "caller profile must be ci or release",
            id="dev-cannot-downgrade-ci-report",
        ),
        pytest.param(
            None,
            "caller profile must be ci or release",
            id="missing-profile-cannot-downgrade-ci-report",
        ),
        pytest.param(
            "release",
            "caller profile must exactly match assurance.profile",
            id="release-cannot-mismatch-ci-report",
        ),
    ],
)
def test_verify_assurance_strict_rejects_caller_profile_downgrade_or_mismatch(
    tmp_path: Path,
    monkeypatch,
    caller_profile: str | None,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, _report(list(CANONICAL_GUARD_CHAIN)))

    result = _run_strict(report_path, profile=caller_profile)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize(
    ("mutation", "diagnostic_fragment"),
    [
        pytest.param(
            "low-windows",
            "canonical strict window floor",
            id="two-windows-cannot-self-declare-sufficient",
        ),
        pytest.param(
            "one-replicate",
            "canonical strict bootstrap replicate floor",
            id="one-replicate-cannot-self-declare-sufficient",
        ),
        pytest.param(
            "bad-alpha",
            "strict bootstrap alpha must equal 0.05",
            id="alpha-99-percent-is-not-approved",
        ),
    ],
)
def test_verify_assurance_strict_derives_evidence_volume_policy_independently(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    stats = payload["dataset"]["windows"]["stats"]
    if mutation == "low-windows":
        for arm in ("preview", "final"):
            section = payload["evaluation_windows"][arm]
            for field in ("window_ids", "logloss", "token_counts"):
                section[field] = section[field][:2]
            payload["dataset"]["windows"][arm] = 2
            stats[f"actual_{arm}"] = 2
            stats["coverage"][arm] = {"used": 2, "required": 2, "ok": True}
        stats["paired_windows"] = 2
        digest = hashlib.blake2s(digest_size=16)
        for window_id in payload["evaluation_windows"]["final"]["window_ids"]:
            digest.update(window_id.to_bytes(8, "little", signed=True))
        schedule_digest = digest.hexdigest()
        payload["provenance"]["window_ids_digest"] = schedule_digest
        payload["provenance"]["window_plan_digest"] = schedule_digest
        payload["guard_metric_impact"]["schedule_digest"] = schedule_digest
    elif mutation == "one-replicate":
        stats["bootstrap"]["replicates"] = 1
        stats["coverage"]["replicates"] = {
            "used": 1,
            "required": 1,
            "ok": True,
        }
    elif mutation == "bad-alpha":
        stats["bootstrap"]["alpha"] = 0.99
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize(
    ("mutation", "diagnostic_fragment"),
    [
        pytest.param(
            "legacy-only",
            "schema validation failed",
            id="legacy-paired-label-cannot-pass-strict",
        ),
        pytest.param(
            "paired-basis",
            "basis must be independent_disjoint_slices",
            id="disjoint-slices-cannot-claim-pairing",
        ),
        pytest.param(
            "ci-mismatch",
            "does not match independent two-slice bootstrap replay",
            id="slice-ci-is-replayed-independently",
        ),
        pytest.param(
            "degenerate-mismatch",
            "degenerate does not match the replayed independent bootstrap",
            id="slice-degeneracy-is-replayed-independently",
        ),
    ],
)
def test_verify_assurance_strict_requires_independent_preview_final_summary(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    diagnostic_fragment: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    stats = payload["dataset"]["windows"]["stats"]
    summary = stats["preview_final_slice_delta_summary"]
    if mutation == "legacy-only":
        stats["paired_delta_summary"] = stats.pop("preview_final_slice_delta_summary")
    elif mutation == "paired-basis":
        summary["basis"] = "paired_preview_final"
        summary["paired"] = True
    elif mutation == "ci-mismatch":
        summary["ci"] = [-100.0, 100.0]
    elif mutation == "degenerate-mismatch":
        summary["degenerate"] = False
        summary["degenerate_reason"] = None
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    diagnostics = "\n".join(item.message for item in result.diagnostics)
    if mutation in {"legacy-only", "paired-basis"}:
        assert result.outcome == VerifyOutcome.MALFORMED
        assert "schema validation failed" in diagnostics
    else:
        assert result.outcome == VerifyOutcome.POLICY_FAIL
        assert diagnostic_fragment in diagnostics


@pytest.mark.parametrize("profile", ["ci", "release"])
def test_ci_and_release_profiles_reject_explicit_raw_guard_failure_without_strict(
    tmp_path: Path,
    monkeypatch,
    profile: str,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["spectral"]["status"] = "fail"
    payload["validation"]["spectral_stable"] = True
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = run_verify_reports(
        [report_path],
        profile=profile,
        assurance_mode="off",
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "spectral.status='fail' is not passing" in diagnostics
