from __future__ import annotations

from invarlock.core.guard_evidence import GuardEvidence


def test_guard_evidence_normalizes_report_entry_and_blocking_reasons() -> None:
    evidence = GuardEvidence.from_result(
        "rmt",
        {
            "passed": False,
            "policy": {"activation_required": True},
            "metrics": {"edge": 1.0},
            "diagnostics": ["missing activations"],
            "violations": ["activation_required"],
            "details": {"layer": "attn"},
            "supported": False,
            "reason": "activation_required",
            "assurance_blocking": True,
            "status": "degraded",
        },
    )

    assert evidence is not None
    entry = evidence.as_report_entry()
    assert entry["name"] == "rmt"
    assert entry["decision"] == "block"
    assert entry["supported"] is False
    assert entry["assurance_blocking"] is True
    assert "rmt unsupported for strict assurance" in " ".join(
        evidence.strict_blocking_reasons()
    )
    assert "rmt did not pass." in evidence.strict_blocking_reasons()


def test_guard_evidence_skips_non_mapping_results() -> None:
    assert GuardEvidence.from_result("spectral", object()) is None


def test_guard_evidence_clean_pass_has_no_blocking_reason() -> None:
    evidence = GuardEvidence.from_result(
        "spectral",
        {
            "passed": True,
            "decision": "allow",
            "final_z_scores": {"linear": 0.1},
            "module_family_map": {"layer": "linear"},
        },
    )

    assert evidence is not None
    entry = evidence.as_report_entry()
    assert entry["final_z_scores"] == {"linear": 0.1}
    assert entry["module_family_map"] == {"layer": "linear"}
    assert evidence.strict_blocking_reasons() == ()


def test_guard_evidence_blocks_monitor_only_status_spellings() -> None:
    hyphenated = GuardEvidence.from_report_block("variance", {"status": "monitor-only"})
    underscored = GuardEvidence.from_report_block(
        "variance", {"status": "monitor_only"}
    )

    assert hyphenated is not None
    assert underscored is not None
    assert "variance status monitor-only is not strict-assurance passing." in (
        hyphenated.strict_blocking_reasons()
    )
    assert "variance status monitor_only is not strict-assurance passing." in (
        underscored.strict_blocking_reasons()
    )


def test_guard_evidence_from_report_block_preserves_explicit_pass_decision() -> None:
    evidence = GuardEvidence.from_report_block(
        "spectral",
        {"passed": True, "metrics": {"sigma": 1.0}},
    )

    assert evidence is not None
    assert evidence.decision == "allow"
