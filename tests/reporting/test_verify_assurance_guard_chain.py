from __future__ import annotations

import json
from pathlib import Path

from invarlock.core.assurance_contract import (
    ASSURANCE_CLAIM_SET,
    CANONICAL_GUARD_CHAIN,
)
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports


def _report(guard_chain: list[str]) -> dict:
    return {
        "schema_version": "v1",
        "run_id": "strict-test",
        "artifacts": {},
        "plugins": {"guards": guard_chain, "adapters": [], "edits": []},
        "guards": [{"name": name} for name in guard_chain],
        "meta": {"profile": "ci"},
        "context": {"profile": "ci"},
        "auto": {"tier": "balanced"},
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
            "final": {"logloss": [0.1, 0.2], "token_counts": [10, 10]}
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
            "runtime_provenance_verified": True,
            "verdict": "pass",
            "blocking_reasons": [],
        },
    }


def test_verify_assurance_strict_rejects_wrong_guard_order(tmp_path: Path) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(
        json.dumps(_report(["invariants", "spectral", "variance", "rmt"])),
        encoding="utf-8",
    )

    result = run_verify_reports(
        [report_path],
        profile="dev",
        allow_unverified_provenance=True,
        assurance_mode="strict",
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "canonical guard chain" in diagnostics


def test_verify_assurance_strict_rejects_missing_claim(tmp_path: Path) -> None:
    payload = _report(list(CANONICAL_GUARD_CHAIN))
    payload.pop("assurance")
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_verify_reports(
        [report_path],
        profile="dev",
        allow_unverified_provenance=True,
        assurance_mode="strict",
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    diagnostics = "\n".join(item.message for item in result.diagnostics)
    assert "requires report assurance.mode=strict" in diagnostics
