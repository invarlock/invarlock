"""Authenticate historical receipts with caller-owned, family-specific inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from invarlock.captured_contracts import read_file, sha
from invarlock.evidence_pack_json import parse_json_bytes

FORMAT = "invarlock/report-verification-inputs-v1"
_COMMON = {
    "policy_path",
    "expected_pack_signer_fingerprint",
    "expected_verifier_identity",
    "expected_verifier_fingerprint",
}
_OPTIONAL = {"expected_trust_profile_digest", "expected_request_digest"}
_REQUIRED = {
    "runtime": _COMMON
    | {
        "expected_artifact_digests",
        "expected_schedule_digest",
        "expected_runtime_digests",
    },
    "captured": _COMMON | {"expected_run_digests", "expected_request_digest"},
    "judge": {
        "recipient_policy_path",
        "expected_verifier_identity",
        "expected_verifier_fingerprint",
    },
    "evidence_set": {"recipient_policy_path"},
}


def authenticate_receipt(
    receipt: Path,
    evidence: Path,
    *,
    inputs: Path,
    kind: str,
    digest: str,
    submitted_evidence: Path,
) -> dict[str, Any]:
    """Never accept booleans from attachments or discover trust in submitted evidence."""
    raw = read_file(inputs, 64 * 1024)
    value = parse_json_bytes(raw, label="report verification inputs")
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "kind", "expectations"}
        or value["format"] != FORMAT
        or value["kind"] != kind
    ):
        raise ValueError("invalid report verification inputs or evidence family")
    args = value["expectations"]
    required = _REQUIRED[kind]
    optional = _OPTIONAL if kind in {"runtime", "captured"} else set()
    if (
        not isinstance(args, dict)
        or not required <= args.keys()
        or args.keys() - required - optional
    ):
        raise ValueError("missing or unsupported independent receipt expectations")
    args = dict(args)
    policy_key = (
        "policy_path" if kind in {"runtime", "captured"} else "recipient_policy_path"
    )
    policy = Path(args[policy_key])
    if not policy.is_absolute():
        policy = inputs.absolute().parent / policy
    if any(
        p.resolve().is_relative_to(root.resolve())
        for p in (policy, inputs)
        for root in (evidence, submitted_evidence)
    ):
        raise ValueError(
            "receipt trust inputs must be supplied outside submitted evidence"
        )
    args[policy_key] = policy
    policy_digest = sha(read_file(policy, 8 * 1024**2))
    mode = (
        "authenticated historical receipt; no current authorization or freshness claim"
    )
    if kind in {"runtime", "captured"}:
        from invarlock.evidence_receipt import verify_signed_verification_receipt

        result = verify_signed_verification_receipt(receipt, evidence, **args)
        if not result.ok or result.statement is None:
            raise ValueError(
                "receipt authentication failed: " + "; ".join(result.errors)
            )
        verdict = dict(result.statement["verdict"])
        accepted = result.statement["verdict"]["ok"]
        verifier = result.statement["verifier"]["identity"]
        verifier_fingerprint = result.verifier_fingerprint
    elif kind == "judge":
        from invarlock.judge_measurements.acceptance import (
            load_judge_recipient_policy,
            verify_signed_judge_verification_receipt,
        )
        from invarlock.judge_measurements.evidence import object_sha256

        policy_value = load_judge_recipient_policy(policy, evidence_path=evidence)
        result_j = verify_signed_judge_verification_receipt(
            receipt,
            expected_recipient_policy_sha256=object_sha256(policy_value),
            expected_verifier_identity=args["expected_verifier_identity"],
            expected_verifier_fingerprint=args["expected_verifier_fingerprint"],
        )
        if (
            not result_j.ok
            or result_j.result is None
            or result_j.result.envelope_sha256 != digest
        ):
            raise ValueError("judge receipt authentication or evidence binding failed")
        verdict = {
            "authenticated": result_j.result.authenticated,
            "replayed": result_j.result.replayed,
            "verified": result_j.result.verified,
            "accepted": result_j.result.accepted,
            "decision": result_j.result.decision,
        }
        accepted = result_j.result.accepted
        verifier = args["expected_verifier_identity"]
        verifier_fingerprint = result_j.verifier_fingerprint
    else:
        from invarlock.evidence_sets.verification import (
            verify_stored_evidence_set_result,
        )

        combined = verify_stored_evidence_set_result(
            receipt, evidence, recipient_policy=policy
        )
        if combined.payload["errors"]:
            raise ValueError(
                "combined stored result differs from independent component verification"
            )
        # The existing contract rechecks component receipts and conjunction; no new joint signature.
        verdict = {
            name: {
                k: member.get(k)
                for k in (
                    "authenticated",
                    "replayed",
                    "verified",
                    "accepted",
                    "decision",
                )
            }
            for name, member in combined.payload["members"].items()
        }
        accepted = combined.accepted
        verifier = "Component verifier identities retained in verification.receipt.json"
        verifier_fingerprint = None
        mode = "fresh component verification and exact stored-result comparison; no signed joint attestation"
    if (
        read_file(inputs, 64 * 1024) != raw
        or sha(read_file(policy, 8 * 1024**2)) != policy_digest
    ):
        raise ValueError("independent receipt inputs changed during authentication")
    return {
        "format": "invarlock/report-verification-snapshot-v1",
        "state": "authenticated_acceptance" if accepted else "authenticated_rejection",
        "authentic": True,
        "recorded_verdict": verdict,
        "accepted": accepted,
        "mode": mode,
        "evidence_statement_sha256": digest,
        "receipt_sha256": sha(read_file(receipt, 8 * 1024**2)),
        "independent_inputs_sha256": sha(raw),
        "policy_file_sha256": policy_digest,
        "verifier_identity": verifier,
        "verifier_fingerprint": verifier_fingerprint,
        "receipt_check_time": datetime.now(UTC).isoformat(),
        "original_verification_time": None,
        "current_authorization": "not established",
        "freshness": "not established",
    }


def verification_facts(snapshot: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    label = (
        "Authenticated historical acceptance"
        if snapshot["accepted"]
        else "Authenticated historical rejection"
    )
    if snapshot["mode"].startswith("fresh"):
        label = (
            "Fresh component acceptance"
            if snapshot["accepted"]
            else "Fresh component rejection"
        )
    import json

    return (
        (
            "Recipient result details",
            json.dumps(snapshot["recorded_verdict"], sort_keys=True),
        ),
        ("Recipient verification", label + ". " + snapshot["mode"] + "."),
        ("Verifier identity", str(snapshot["verifier_identity"])),
        ("Receipt authentication time", snapshot["receipt_check_time"]),
        (
            "Verification time",
            "Original verification time unavailable in this receipt contract.",
        ),
        (
            "Trust snapshot",
            "Independent input identity: "
            + snapshot["independent_inputs_sha256"]
            + "; policy file identity: "
            + snapshot["policy_file_sha256"]
            + ". Current authorization and freshness are not established.",
        ),
    )
