from __future__ import annotations

import pytest

from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET,
    CANONICAL_GUARD_CHAIN,
    build_assurance_section,
    normalize_assurance_mode,
    normalize_verify_assurance_mode,
    observed_guard_chain_from_report,
    resolve_report_assurance_mode,
    strict_evaluate_policy_errors,
    strict_report_policy_errors,
)


def _strict_report() -> dict:
    return {
        "context": {"profile": "ci", "assurance": {"mode": "strict"}},
        "auto": {"tier": "balanced"},
        "guards": [{"name": name} for name in CANONICAL_GUARD_CHAIN],
        "spectral": {"supported": True},
        "rmt": {"supported": True},
        "variance": {"enabled": False, "supported": True},
        "invariants": {"supported": True},
    }


def test_build_assurance_section_passes_for_canonical_strict_report() -> None:
    report = _strict_report()

    section = build_assurance_section(report)

    assert section["mode"] == "strict"
    assert section["claim_set"] == ASSURANCE_CLAIM_SET
    assert section["guard_chain_observed"] == list(CANONICAL_GUARD_CHAIN)
    assert section["canonical_guard_chain_enforced"] is True
    assert section["verdict"] == "pass"
    assert section["blocking_reasons"] == []


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
        "strict assurance requires verified container provenance.",
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
        "context": {"profile": "release", "auto": {"tier": "conservative"}},
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
    }
    assert build_assurance_section(base, mode="strict")["tier"] == "conservative"

    report = {
        "context": {"profile": "release", "tier": "balanced"},
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
    }
    assert build_assurance_section(report, mode="strict")["tier"] == "balanced"

    report = {
        "context": {"profile": "release", "tier": 7},
        "plugins": {"guards": list(CANONICAL_GUARD_CHAIN)},
    }
    section = build_assurance_section(report, mode="strict")
    assert section["tier"] == ""
    assert any(
        "tier balanced or conservative" in item for item in section["blocking_reasons"]
    )

    report = {
        "context": {"profile": "release"},
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
    assert section["verdict"] == "pass"


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
    assert "strict assurance verdict must be pass." in errors
    assert "strict assurance requires canonical_guard_chain_enforced=true." in errors
    assert "explicit blocker" in errors
    assert "variance is degraded/monitor-only under strict assurance." in errors


def test_strict_report_policy_returns_empty_when_not_required() -> None:
    assert strict_report_policy_errors({}, require_strict=False) == []


def test_strict_report_policy_rejects_missing_assurance_section() -> None:
    errors = strict_report_policy_errors({}, require_strict=True)

    assert "strict assurance report missing assurance section." in errors
