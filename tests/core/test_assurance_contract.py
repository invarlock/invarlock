from __future__ import annotations

import pytest

from invarlock.core.assurance_contract import (
    CANONICAL_GUARD_CHAIN,
    LEGACY_ASSURANCE_CLAIM_SET,
    build_assurance_section,
    normalize_assurance_mode,
    normalize_verify_assurance_mode,
    observed_guard_chain_from_report,
    report_build_has_blocking_evidence_events,
    resolve_report_assurance_mode,
    strict_evaluate_policy_errors,
    strict_report_policy_errors,
)
from tests.core._support_assurance_contract import strict_report as _strict_report


def test_build_assurance_section_passes_for_canonical_strict_report() -> None:
    report = _strict_report()

    section = build_assurance_section(report)

    assert section["mode"] == "strict"
    assert section["claim_set"] == LEGACY_ASSURANCE_CLAIM_SET
    assert section["guard_chain_observed"] == list(CANONICAL_GUARD_CHAIN)
    assert section["canonical_guard_chain_enforced"] is True
    assert section["runtime_provenance_verified"] is False
    assert section["runtime_provenance_declared"] == "container"
    assert section["runtime_provenance_verification_status"] == "pending"
    assert section["verdict"] == "pending_verifier"
    assert section["report_local_verdict"] == "pass"
    assert section["verified_assurance_verdict"] == "pending"
    assert section["blocking_reasons"] == []


def test_observed_guard_chain_prefers_context_chain_with_duplicates() -> None:
    report = _strict_report()
    report["guards"] = [
        {"name": name} for name in ("invariants", "spectral", "rmt", "variance")
    ]
    report["context"]["guard_chain_observed"] = list(CANONICAL_GUARD_CHAIN)

    assert observed_guard_chain_from_report(report) == list(CANONICAL_GUARD_CHAIN)


def test_build_assurance_section_rejects_invalid_runtime_status() -> None:
    report = _strict_report()

    section = build_assurance_section(
        report,
        runtime_provenance_verification_status="failed",
    )

    assert section["runtime_provenance_verification_status"] == "failed"
    assert section["verdict"] == "fail"
    assert (
        "strict assurance requires report/manifest binding plus a "
        "independently supplied runtime image digest." in section["blocking_reasons"]
    )


def test_strict_report_policy_rejects_wrong_guard_chain() -> None:
    report = _strict_report()
    report["assurance"] = build_assurance_section(report)
    report["assurance"]["guard_chain_observed"] = ["invariants", "spectral"]
    report["assurance"]["canonical_guard_chain_enforced"] = False

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any("canonical guard chain" in error for error in errors)


def test_strict_report_policy_rejects_unsupported_blocking_guard() -> None:
    report = _strict_report()
    report["rmt"] = {
        "supported": False,
        "reason": "no_supported_rmt_modules",
        "assurance_blocking": True,
    }
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any("no_supported_rmt_modules" in error for error in errors)


def test_strict_report_policy_rejects_fallback_fields() -> None:
    report = _strict_report()
    report["assurance"] = build_assurance_section(
        report,
        fallback_fields_used=True,
    )

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any("repaired fields" in error for error in errors)


def test_strict_report_policy_accepts_pending_report_when_verifier_confirms() -> None:
    report = _strict_report()
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(
        report,
        require_strict=True,
        runtime_provenance_verified=True,
    )

    assert errors == []


def test_build_assurance_section_uses_unknown_runtime_when_not_declared() -> None:
    report = _strict_report()
    report["context"].pop("runtime")

    section = build_assurance_section(report)

    assert section["runtime_provenance_declared"] == "unknown"
    assert section["verdict"] == "fail"
    assert (
        "strict assurance requires container execution mode and fail-closed "
        "runtime provenance checks." in section["blocking_reasons"]
    )


def test_build_assurance_section_derives_runtime_from_context() -> None:
    report = _strict_report()
    report["context"]["runtime"] = {"execution_mode": " HOST "}

    section = build_assurance_section(report, mode="off")

    assert section["runtime_provenance_declared"] == "host"


def test_build_assurance_section_derives_runtime_from_provenance_runtime() -> None:
    report = _strict_report()
    report.pop("context")
    report["provenance"] = {"runtime": {"execution_mode": "container"}}

    section = build_assurance_section(report, mode="strict")

    assert section["runtime_provenance_declared"] == "container"


def test_strict_report_policy_rejects_pending_report_when_verifier_fails() -> None:
    report = _strict_report()
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(
        report,
        require_strict=True,
        runtime_provenance_verified=False,
    )

    assert (
        "strict assurance requires report/manifest binding plus a "
        "independently supplied runtime image digest." in errors
    )


def test_strict_report_policy_rejects_missing_guard_evidence() -> None:
    report = _strict_report()
    report.pop("rmt")
    report["assurance"] = build_assurance_section(report)

    errors = strict_report_policy_errors(
        report,
        require_strict=True,
        runtime_provenance_verified=True,
    )

    assert "strict assurance missing rmt guard evidence." in errors


def test_strict_report_policy_rejects_failed_report_local_pending_verdict() -> None:
    report = _strict_report()
    report["assurance"] = build_assurance_section(report)
    report["assurance"]["report_local_verdict"] = "fail"

    errors = strict_report_policy_errors(
        report,
        require_strict=True,
        runtime_provenance_verified=True,
    )

    assert "strict assurance.report_local_verdict must be pass." in errors


def test_strict_report_policy_rejects_structured_report_build_events() -> None:
    report = _strict_report()
    report["report_build"] = {
        "synthesized_fields": [
            {
                "field": "primary_metric.display_ci",
                "reason": "default_estimated_interval",
                "source": "test",
            }
        ],
        "repaired_fields": [],
        "fallback_fields": [],
    }
    report["assurance"] = build_assurance_section(report)

    assert report["assurance"]["fallback_fields_used"] is True
    assert report["assurance"]["verdict"] == "fail"

    errors = strict_report_policy_errors(report, require_strict=True)

    assert any("repaired fields" in error for error in errors)


def test_strict_report_policy_allows_display_ci_computed_from_ci_event() -> None:
    report = _strict_report()
    report["report_build"] = {
        "synthesized_fields": [
            {
                "field": "primary_metric.display_ci",
                "reason": "computed_from_primary_metric_ci",
                "source": "primary_metric_utils._attach_primary_metric_from_report",
            }
        ],
        "repaired_fields": [],
        "fallback_fields": [],
    }
    report["assurance"] = build_assurance_section(report)

    assert report_build_has_blocking_evidence_events(report) is False
    assert report["assurance"]["fallback_fields_used"] is False
    assert report["assurance"]["verdict"] == "pending_verifier"
    assert not strict_report_policy_errors(
        report,
        require_strict=True,
        runtime_provenance_verified=True,
    )


def test_report_build_event_helper_ignores_non_lists_and_flags_non_objects() -> None:
    assert (
        report_build_has_blocking_evidence_events(
            {
                "report_build": {
                    "synthesized_fields": "not-a-list",
                    "repaired_fields": [],
                    "fallback_fields": [],
                }
            }
        )
        is False
    )
    assert (
        report_build_has_blocking_evidence_events(
            {
                "report_build": {
                    "synthesized_fields": [None],
                    "repaired_fields": [],
                    "fallback_fields": [],
                }
            }
        )
        is True
    )


def test_assurance_mode_normalizers_reject_unknown_modes() -> None:
    assert normalize_assurance_mode(None, default="off") == "off"
    assert normalize_verify_assurance_mode(None) == "report"

    with pytest.raises(ValueError, match="strict, off"):
        normalize_assurance_mode("permissive")
    with pytest.raises(ValueError, match="report, strict, off"):
        normalize_verify_assurance_mode("permissive")


def test_strict_evaluate_policy_reports_all_blockers() -> None:
    errors = strict_evaluate_policy_errors(
        assurance_mode="strict",
        profile="dev",
        tier="aggressive",
        guards_order=["invariants", "spectral"],
        execution_mode="host",
        allow_unverified_provenance=True,
    )

    assert errors == [
        "strict assurance requires profile ci or release.",
        "strict assurance requires tier balanced or conservative.",
        "strict assurance requires the canonical guard chain.",
        "strict assurance requires container execution mode and fail-closed "
        "runtime provenance checks.",
    ]


def test_strict_evaluate_policy_is_noop_when_assurance_off() -> None:
    assert (
        strict_evaluate_policy_errors(
            assurance_mode="off",
            profile="dev",
            tier="aggressive",
            guards_order=[],
            execution_mode="host",
            allow_unverified_provenance=True,
        )
        == []
    )


def test_report_assurance_mode_resolution_uses_context_fallback() -> None:
    assert (
        resolve_report_assurance_mode({"assurance": {"mode": " STRICT "}}) == "strict"
    )
    assert (
        resolve_report_assurance_mode(
            {"assurance": {"mode": ""}, "context": {"assurance": {"mode": "off"}}}
        )
        == "off"
    )
    assert (
        resolve_report_assurance_mode({"context": {"assurance": {"mode": " strict "}}})
        == "strict"
    )
    assert resolve_report_assurance_mode({"context": {"assurance": {}}}) == "off"
    assert resolve_report_assurance_mode({}) == "off"


def test_observed_guard_chain_resolution_prefers_valid_sources() -> None:
    assert observed_guard_chain_from_report(
        {"assurance": {"guard_chain_observed": ["invariants"]}}
    ) == ["invariants"]
    assert observed_guard_chain_from_report(
        {
            "assurance": {"guard_chain_observed": ["invariants", 7]},
            "guards": [{"name": "spectral"}],
        }
    ) == ["spectral"]
    assert observed_guard_chain_from_report(
        {"guards": [{"name": "spectral"}, {"name": 7}, "bad"]}
    ) == ["spectral"]
    assert observed_guard_chain_from_report({"plugins": {"guards": ["rmt"]}}) == ["rmt"]
    assert observed_guard_chain_from_report({"plugins": {"guards": ["rmt", 7]}}) == []


def test_build_assurance_section_reads_tier_fallbacks() -> None:
    base = {
        "context": {
            "profile": "release",
            "auto": {"tier": "conservative"},
            "runtime": {"execution_mode": "container"},
        },
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
    }
    assert build_assurance_section(base, mode="strict")["tier"] == "conservative"

    report = {
        "context": {
            "profile": "release",
            "tier": "balanced",
            "runtime": {"execution_mode": "container"},
        },
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
    }
    assert build_assurance_section(report, mode="strict")["tier"] == "balanced"

    report = {
        "context": {
            "profile": "release",
            "tier": 7,
            "runtime": {"execution_mode": "container"},
        },
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
    }
    section = build_assurance_section(report, mode="strict")
    assert section["tier"] == ""
    assert any(
        "tier balanced or conservative" in item for item in section["blocking_reasons"]
    )

    report = {
        "context": {
            "profile": "release",
            "runtime": {"execution_mode": "container"},
        },
        "auto": {"tier": "balanced"},
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
        "spectral": {
            "supported": False,
            "reason": "optional-family",
            "assurance_blocking": False,
        },
    }
    section = build_assurance_section(report, mode="strict")
    assert section["tier"] == "balanced"
    assert section["verdict"] == "fail"
    assert any("unsupported" in item for item in section["blocking_reasons"])


def test_strict_report_policy_collects_report_level_failures() -> None:
    report = {
        "assurance": {
            "claim_set": "unknown",
            "mode": "off",
            "verdict": "fail",
            "canonical_guard_chain_enforced": False,
            "fallback_fields_used": True,
            "runtime_provenance_verified": False,
            "blocking_reasons": ["explicit blocker"],
            "guard_chain_observed": ["variance"],
        },
        "context": {"profile": "dev", "tier": "aggressive"},
        "variance": {"status": "monitor-only"},
    }

    errors = strict_report_policy_errors(
        report,
        require_strict=True,
        fallback_fields_used=True,
        runtime_provenance_verified=False,
    )

    assert "strict assurance report has unknown claim_set." in errors
    assert "strict assurance report mode must be strict." in errors
    assert (
        "strict assurance.verdict must be pending_verifier in submitted evidence."
        in errors
    )
    assert "strict assurance requires canonical_guard_chain_enforced=true." in errors
    assert "explicit blocker" in errors
    assert "variance.status='monitor-only' is not passing." in errors


def test_strict_report_policy_returns_empty_when_not_required() -> None:
    assert strict_report_policy_errors({}, require_strict=False) == []


def test_strict_report_policy_rejects_report_controlled_tiny_relax() -> None:
    report = _strict_report()
    report["context"]["run"] = {"tiny_relax": True}

    errors = strict_report_policy_errors(report, require_strict=True)

    assert "strict assurance forbids development-only tiny_relax policy." in errors


def test_non_strict_report_policy_preserves_tiny_relax_for_development() -> None:
    report = {"context": {"run": {"tiny_relax": True}}}

    assert strict_report_policy_errors(report, require_strict=False) == []


def test_strict_report_policy_rejects_missing_assurance_section() -> None:
    errors = strict_report_policy_errors({}, require_strict=True)

    assert "strict assurance report missing assurance section." in errors
