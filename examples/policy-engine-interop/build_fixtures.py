#!/usr/bin/env python3
"""Regenerate the six deterministic OPA/CUE policy fixtures."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from verify_envelope import build_policy_input, canonical_bytes

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT.parent / "acceptance-handoff" / "golden"
FIXTURES = ROOT / "fixtures"
SUBJECT_NAME = "producer.example/subject"
SUBJECT_SHA256 = "a9fcf5a7cb042b0f4db67dead3d64fad8c3775d7ea25c91ee6759b019b5603cb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = build_policy_input(
        envelope_path=GOLDEN / "acceptance.dsse.json",
        envelope_key_path=GOLDEN / "producer.public.pem",
        recipient_policy_path=GOLDEN / "recipient-policy.json",
        expected_subject_name=SUBJECT_NAME,
        expected_subject_sha256=SUBJECT_SHA256,
        now="2026-07-25T12:30:00Z",
    )
    scenarios: dict[str, dict[str, object]] = {"positive": positive}

    policy_rejected = copy.deepcopy(positive)
    policy_rejected["recipient"]["required_technical_verdict"] = "fail"
    scenarios["policy-rejected"] = policy_rejected

    tampered_subject = copy.deepcopy(positive)
    tampered_subject["statement"]["subject"][0]["digest"]["sha256"] = "0" * 64
    tampered_subject["authentication"]["envelope_signature"] = False
    tampered_subject["authentication"]["projection_consistent"] = False
    scenarios["tampered-subject"] = tampered_subject

    untrusted_signer = copy.deepcopy(positive)
    untrusted_signer["recipient"]["trusted_signers"] = [
        {
            "fingerprint": f"sha256:{'1' * 64}",
            "identity": "recipient.example/unknown",
            "status": "active",
        }
    ]
    scenarios["untrusted-signer"] = untrusted_signer

    stale_evidence = copy.deepcopy(positive)
    stale_evidence["verified"]["now_unix"] = (
        stale_evidence["verified"]["attestation_issued_at_unix"]
        + stale_evidence["recipient"]["max_attestation_age_seconds"]
        + 1
    )
    scenarios["stale-evidence"] = stale_evidence

    unsupported_contract = copy.deepcopy(positive)
    unsupported_contract["recipient"]["allowed_contract_versions"] = ["0.14.0"]
    scenarios["unsupported-contract"] = unsupported_contract

    generated: dict[Path, bytes] = {}
    for name, value in scenarios.items():
        generated[FIXTURES / f"{name}.json"] = canonical_bytes(value)
    reasons = {
        "policy-rejected": ["technical_verdict_rejected"],
        "positive": [],
        "stale-evidence": ["stale_evidence"],
        "tampered-subject": ["authentication_failed", "subject_rejected"],
        "unsupported-contract": ["unsupported_contract"],
        "untrusted-signer": ["untrusted_signer"],
    }
    expectations = {
        "format": "invarlock/policy-engine-expectations-v1",
        "scenarios": {
            name: {"allow": name == "positive", "reasons": reasons[name]}
            for name in sorted(scenarios)
        },
    }
    generated[FIXTURES / "expectations.json"] = canonical_bytes(expectations)
    if args.check:
        stale = [
            path.name
            for path, payload in generated.items()
            if not path.exists() or path.read_bytes() != payload
        ]
        if stale:
            raise SystemExit(f"policy fixtures are stale: {', '.join(sorted(stale))}")
        return
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for path, payload in generated.items():
        path.write_bytes(payload)


if __name__ == "__main__":
    main()
