"""Offline same-answer evidence composition; fixture scores are synthetic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.captured_contracts import atomic_write, sha
from invarlock.captured_evidence_publication import publish_captured_evidence
from invarlock.captured_normalization import (
    captured_request_digest,
    normalize_captured_request,
)
from invarlock.evaluation_comparison.comparison import compare_runs
from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_sets.contracts import (
    SCOPE,
    STATISTICAL_SCOPE,
    write_evidence_set_index,
)
from invarlock.evidence_sets.verification import verify_evidence_set
from invarlock.judge_measurements.analysis import (
    analyze_measurements,
    decode_analysis_policy,
)
from invarlock.judge_measurements.evidence import object_sha256, publish_judge_evidence


def write(path: Path, value: object) -> None:
    atomic_write(path, canonical_json_bytes(value))


def build(output: Path) -> Path:
    output = output.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1] / "judge-measurements"
    values = {
        name: json.loads((fixtures / f"{name}.json").read_text())
        for name in (
            "plan",
            "measurements",
            "baseline_run",
            "subject_run",
            "analysis_policy",
        )
    }
    baseline, subject = values["baseline_run"], values["subject_run"]
    captured_policy = {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "exact_match",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 1,
                "maximum_interval_width": 2,
            }
        ],
        "slices": [],
    }
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    normalized = normalize_captured_request(
        request, baseline=baseline, subject=subject, policy=captured_policy
    )
    request_digest = captured_request_digest(normalized)
    signer, verifier = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    producer, recipient = output / "producer", output / "recipient"
    producer.mkdir(mode=0o700)
    recipient.mkdir(mode=0o700)
    for path, key in (
        (producer / "signer.pem", signer),
        (recipient / "verifier.pem", verifier),
    ):
        atomic_write(
            path,
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
    # This one-process demo controls both roles. Real recipients independently
    # review inputs/policies and obtain the producer key fingerprint externally.
    case_set = {
        "format": "invarlock/evaluation-case-set-v1",
        "cases": [
            {key: row[key] for key in ("id", "input", "expected", "metadata")}
            for row in baseline["records"]
        ],
    }
    shared = {
        "baseline_run_sha256": run_digest(baseline),
        "subject_run_sha256": run_digest(subject),
        "case_set_sha256": case_set_digest(case_set),
        "subject_artifact_sha256": subject["artifact_digest"],
    }
    analysis = analyze_measurements(
        values["plan"],
        values["measurements"],
        decode_analysis_policy(values["analysis_policy"], plan=values["plan"]),
        baseline_run=baseline,
        subject_run=subject,
    )
    judge_policy = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": "bounded-judge-fixed-benchmark-v1",
        "intended_subject": shared["subject_artifact_sha256"],
        "required_metric_name": values["analysis_policy"]["metric_name"],
        "trusted_signer": {
            "identity": "example-producer",
            "public_key_sha256": public_key_fingerprint(signer.public_key()),
        },
        "bindings": {
            key: shared[key]
            for key in ("baseline_run_sha256", "subject_run_sha256", "case_set_sha256")
        },
        "required_decision": "pass",
    }
    judge_policy["bindings"].update(
        plan_sha256=object_sha256(values["plan"]),
        measurements_sha256=object_sha256(values["measurements"]),
        analysis_policy_sha256=object_sha256(values["analysis_policy"]),
        analysis_result_sha256=object_sha256(analysis.to_dict()),
    )
    write(recipient / "judge.json", judge_policy)
    write(recipient / "deterministic-policy.json", captured_policy)
    write(
        recipient / "captured.json",
        {
            "format": "invarlock/trust-inputs-v2",
            "kind": "captured",
            "policy": {"path": "deterministic-policy.json"},
            "anchors": {
                "baseline_run_digest": shared["baseline_run_sha256"],
                "subject_run_digest": shared["subject_run_sha256"],
                "request_digest": request_digest,
                "evidence_signer_fingerprint": public_key_fingerprint(
                    signer.public_key()
                ),
            },
            "verifier": {
                "identity": "example-recipient",
                "signing_key_path": "verifier.pem",
            },
        },
    )
    evidence = output / "evidence"
    evidence.mkdir(mode=0o700)
    publish_captured_evidence(
        evidence / "deterministic",
        baseline=baseline,
        subject=subject,
        policy=captured_policy,
        comparison=compare_runs(baseline, subject, captured_policy),
        request_digest=request_digest,
        signing_key_path=producer / "signer.pem",
        unsigned=False,
        normalized_request=normalized,
    )
    publish_judge_evidence(
        evidence / "judge",
        plan=values["plan"],
        measurements=values["measurements"],
        baseline_run=baseline,
        subject_run=subject,
        analysis_policy=values["analysis_policy"],
        signing_key=signer,
        signer_identity="example-producer",
    )
    index = write_evidence_set_index(
        evidence, deterministic="deterministic", judge="judge"
    )
    # The transport index is approved only after independently binding both
    # components above; an index never supplies the child authorization policy.
    write(
        recipient / "composition.json",
        {
            "format": "invarlock/evidence-set-recipient-policy-v1",
            "scope": SCOPE,
            "index_sha256": sha(index.read_bytes()),
            "shared_inputs": shared,
            "members": {
                name: {
                    "kind": kind,
                    "trust_profile": filename,
                    "trust_profile_sha256": sha((recipient / filename).read_bytes()),
                    "role": "required",
                }
                for name, kind, filename in [
                    ("deterministic", "captured", "captured.json"),
                    ("judge", "judge", "judge.json"),
                ]
            },
            "decision_rule": "all-required-components-pass",
            "statistical_scope": STATISTICAL_SCOPE,
        },
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New working directory."
    )
    args = parser.parse_args()
    evidence = build(args.output)
    result = verify_evidence_set(
        evidence,
        recipient_policy=evidence.parent / "recipient/composition.json",
        receipt=evidence.parent / "verification.json",
    )
    print(result.as_json(), end="")
    # An inconclusive synthetic example is expected; integrity must still verify.
    raise SystemExit(0 if result.payload["verified"] else result.exit_code)


if __name__ == "__main__":
    main()
