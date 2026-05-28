from __future__ import annotations

import json
from pathlib import Path

import invarlock.reporting.verify_contract as verify_mod
from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET,
    CANONICAL_GUARD_CHAIN,
)
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports
from invarlock.runtime_provenance import RuntimeProvenanceResult


def _report(guard_chain: list[str]) -> dict:
    return {
        "schema_version": "v1",
        "run_id": "strict-test",
        "artifacts": {},
        "plugins": {"guards": guard_chain, "adapters": [], "edits": []},
        "guards": [{"name": name} for name in guard_chain],
        "spectral": {"supported": True, "status": "pass"},
        "rmt": {"supported": True, "status": "pass"},
        "variance": {"supported": True, "status": "pass"},
        "invariants": {"supported": True, "status": "pass"},
        "meta": {"profile": "ci"},
        "context": {
            "profile": "ci",
            "runtime": {"execution_mode": "container"},
        },
        "auto": {"tier": "balanced"},
        "provenance": {"provider_digest": {"ids_sha256": "subject-ids"}},
        "dataset": {
            "provider": "local_jsonl",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "paired_windows": 2,
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "window_pairing_reason": None,
                    "coverage": {
                        "preview": {"used": 2},
                        "final": {"used": 2},
                    },
                },
            },
        },
        "baseline_ref": {"primary_metric": {"final": 2.0}},
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 2.0,
            "final": 2.0,
            "ratio_vs_baseline": 1.0,
            "ci": [0.0, 0.0],
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {
            "final": {
                "logloss": [0.6931471805599453, 0.6931471805599453],
                "token_counts": [10, 10],
            }
        },
        "validation": {},
        "assurance": {
            "mode": "strict",
            "profile": "ci",
            "tier": "balanced",
            "claim_set": ASSURANCE_CLAIM_SET,
            "canonical_guard_chain": list(CANONICAL_GUARD_CHAIN),
            "guard_chain_observed": guard_chain,
            "canonical_guard_chain_enforced": guard_chain
            == list(CANONICAL_GUARD_CHAIN),
            "fallback_fields_used": False,
            "runtime_provenance_verified": False,
            "runtime_provenance_declared": "container",
            "runtime_provenance_verification_status": "pending",
            "verdict": "pending_verifier",
            "report_local_verdict": "pass",
            "verified_assurance_verdict": "pending",
            "blocking_reasons": [],
        },
    }


def _write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _verified_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        verify_mod,
        "verify_runtime_provenance",
        lambda *args, **kwargs: RuntimeProvenanceResult(
            verified=True,
            skipped=False,
        ),
        raising=True,
    )


def _run_strict(path: Path, *, profile: str = "dev"):
    return run_verify_reports(
        [path],
        profile=profile,
        assurance_mode="strict",
    )


def test_verify_assurance_strict_rejects_wrong_guard_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, _report(["invariants", "spectral", "variance", "rmt"]))

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "canonical guard chain" in diagnostics


def test_verify_assurance_strict_rejects_missing_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload.pop("assurance")
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "requires report assurance.mode=strict" in diagnostics


def test_verify_assurance_strict_accepts_pending_report_when_manifest_verifies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, _report(list(CANONICAL_GUARD_CHAIN)))

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.OK


def test_verify_assurance_strict_rejects_unverified_override(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, _report(list(CANONICAL_GUARD_CHAIN)))

    result = run_verify_reports(
        [report_path],
        profile="dev",
        allow_unverified_provenance=True,
        assurance_mode="strict",
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "verified runtime provenance" in diagnostics


def test_verify_assurance_strict_rejects_manipulated_display_ci(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["primary_metric"]["display_ci"] = [1.0, 1.5]
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "display_ci mismatch" in diagnostics


def test_verify_assurance_strict_rejects_manipulated_ratio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["primary_metric"]["ratio_vs_baseline"] = 1.25
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "Primary metric ratio mismatch" in diagnostics


def test_verify_assurance_strict_rejects_missing_final_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["evaluation_windows"].pop("final")
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path, profile="ci")

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "evaluation_windows.final missing" in diagnostics


def test_verify_assurance_strict_rejects_mismatched_final_window_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["evaluation_windows"]["final"]["window_ids"] = [1]
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path, profile="ci")

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "window_ids length differs" in diagnostics


def test_verify_assurance_strict_rejects_duplicated_final_window_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["evaluation_windows"]["final"]["window_ids"] = [1, 1]
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path, profile="ci")

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "window_ids contains duplicates" in diagnostics


def test_verify_assurance_strict_rejects_duplicate_or_extra_guard_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    guard_chain = list(CANONICAL_GUARD_CHAIN) + ["spectral"]
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, _report(guard_chain))

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "canonical guard chain" in diagnostics


def test_verify_assurance_strict_rejects_missing_top_level_guard_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload.pop("rmt")
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "missing rmt guard evidence" in diagnostics


def test_verify_assurance_strict_rejects_empty_spectral_guard_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["spectral"] = {}
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "missing spectral guard evidence" in diagnostics


def test_verify_assurance_strict_rejects_spectral_without_pass_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["spectral"] = {"metrics": {"sigma": 1.0}}
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "spectral missing strict guard pass evidence." in diagnostics


def test_verify_assurance_strict_rejects_empty_invariants_guard_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["invariants"] = {}
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "missing invariants guard evidence" in diagnostics


def test_verify_assurance_strict_rejects_missing_second_invariants_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    guard_chain = ["invariants", "spectral", "rmt", "variance"]
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, _report(guard_chain))

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "canonical guard chain" in diagnostics


def test_verify_assurance_strict_rejects_unsupported_blocking_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["rmt"] = {
        "supported": False,
        "reason": "no_supported_rmt_modules",
        "assurance_blocking": True,
    }
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "no_supported_rmt_modules" in diagnostics


def test_verify_assurance_strict_rejects_monitor_only_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["variance"]["status"] = "monitor-only"
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "monitor-only" in diagnostics


def test_verify_assurance_strict_rejects_degraded_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["rmt"]["status"] = "degraded"
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "rmt status degraded is not strict-assurance passing." in diagnostics


def test_verify_assurance_strict_rejects_tokenizer_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["meta"]["tokenizer_hash"] = "edited"
    payload["baseline_ref"]["tokenizer_hash"] = "baseline"
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "Tokenizer hash mismatch" in diagnostics


def test_verify_assurance_strict_rejects_provider_parity_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["provenance"]["provider_digest"] = {
        "ids_sha256": "window-ids",
        "tokenizer_sha256": "subject-tokenizer",
    }
    baseline = {
        "schema_version": "v1",
        "provenance": {
            "provider_digest": {
                "ids_sha256": "window-ids",
                "tokenizer_sha256": "baseline-tokenizer",
            }
        },
    }
    baseline_path = tmp_path / "baseline.report.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = run_verify_reports(
        [report_path],
        baseline=baseline_path,
        profile="ci",
        assurance_mode="strict",
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "TOKENIZER-DIGEST-MISMATCH" in diagnostics


def test_verify_assurance_strict_rejects_structured_report_build_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _verified_runtime(monkeypatch)
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload["report_build"] = {
        "synthesized_fields": [
            {
                "field": "primary_metric.display_ci",
                "reason": "test_mutation",
                "source": "test",
            }
        ],
        "repaired_fields": [],
        "fallback_fields": [],
    }
    payload["assurance"]["fallback_fields_used"] = False
    report_path = tmp_path / "evaluation.report.json"
    _write_report(report_path, payload)

    result = _run_strict(report_path)

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "repaired fields" in diagnostics
