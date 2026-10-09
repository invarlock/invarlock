"""Offline A -> declared incident -> B rehearsal using synthetic hosted captures.

No service is called. Fixed observation times and answers illustrate distinct
campaigns; signatures authenticate retained inputs, not their execution or time.
The scenario index is explanatory context, not an evidence or incident contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.engine import (
    EvidenceVerificationError,
    capture_evaluator_run,
    captured_request_digest,
    digest,
    evaluate_request_file,
    normalize_captured_request,
    run_digest,
    verify_evidence,
)


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): sha(path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def keypair(directory, name):
    key = Ed25519PrivateKey.generate()
    private = directory / f"{name}.pem"
    with private.open("xb") as stream:
        private.chmod(0o600)
        stream.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    (directory / f"{name}.public.pem").write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return private, sha(key.public_key().public_bytes_raw())


def captured_run(name, day, answer):
    configuration = {"temperature": 0, "purpose": "synthetic reassessment rehearsal"}
    return capture_evaluator_run(
        [
            {
                "id": f"case-{index}",
                "input": f"question-{index}",
                "expected": "yes",
                "output": answer,
            }
            for index in range(32)
        ],
        source={"name": "reassessment-fixture", "version": "1"},
        run_id=name,
        artifact_digest=None,
        service_identity={
            "kind": "hosted_service",
            "provider": "example-provider",
            "service": "example-service",
            "deployment": "example-deployment",
            "requested_model": "example-alias",
            "observed_model": None,
            "exposed_revision": None,
            "configuration": configuration,
            "configuration_digest": digest(configuration),
            "harness": {
                "name": "reassessment-fixture",
                "version": "1",
                "source_digest": sha(Path(__file__).read_bytes()),
            },
            "observation_window": {
                "started_at": f"2026-09-{day:02d}T12:00:00Z",
                "ended_at": f"2026-09-{day:02d}T12:01:00Z",
            },
        },
    )


def assess(directory, *, baseline, subject, policy, signer, verifier):
    directory.mkdir()
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "policy": "comparison-policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    # Freeze expectations from reviewed inputs before publishing the evidence.
    anchors = {
        "baseline_run_digest": run_digest(baseline),
        "subject_run_digest": run_digest(subject),
        "request_digest": captured_request_digest(
            normalize_captured_request(
                request, baseline=baseline, subject=subject, policy=policy
            )
        ),
        "evidence_signer_fingerprint": signer[1],
    }
    for filename, value in (
        ("baseline.json", baseline),
        ("subject.json", subject),
        ("comparison-policy.json", policy),
        ("request.json", request),
        ("anchors.json", anchors),
    ):
        write(directory / filename, value)
    write(
        directory / "trust.json",
        {
            "format": "invarlock/trust-inputs-v2",
            "kind": "captured",
            "policy": {"path": "comparison-policy.json"},
            "anchors": anchors,
            "verifier": {
                "identity": "reassessment-example-recipient",
                "signing_key_path": "../keys/verifier.pem",
            },
        },
    )
    evaluate_request_file(directory / "request.json", signing_key_path=signer[0])
    receipt = directory / "verification.receipt.json"
    try:
        verified = verify_evidence(
            directory / "evidence",
            policy_path=directory / "comparison-policy.json",
            expected_baseline_run=anchors["baseline_run_digest"],
            expected_subject_run=anchors["subject_run_digest"],
            expected_request_digest=anchors["request_digest"],
            expected_signer=signer[1],
            receipt_path=receipt,
            verifier_signing_key_path=verifier[0],
            verifier_identity="reassessment-example-recipient",
        ).payload
    except EvidenceVerificationError as exc:
        if exc.exit_code != 7 or exc.payload.get("integrity_ok") is not True:
            raise
        verified = exc.payload
    write(directory / "verification.json", verified)
    return {
        "anchors": anchors,
        "decision": verified["decision"],
        "observation_window": subject["service_identity"]["observation_window"],
        "evidence": f"{directory.name}/evidence",
        "receipt": f"{directory.name}/verification.receipt.json",
        "receipt_sha256": sha(receipt.read_bytes()),
    }


def rehearse(output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    keys = output / "keys"
    keys.mkdir(mode=0o700)
    signer, verifier = keypair(keys, "evidence-signer"), keypair(keys, "verifier")
    baseline = captured_run("approved-frozen-baseline", 20, "yes")
    policy = example_policy()
    first = assess(
        output / "a",
        baseline=baseline,
        subject=captured_run("assessment-a", 21, "yes"),
        policy=policy,
        signer=signer,
        verifier=verifier,
    )
    historical = snapshot(output / "a")
    trigger = {
        "reference": "example-incident-001",
        "declared_at": "2026-09-22T12:00:00Z",
        "reason": "Reported answer changes prompt a new bounded comparison.",
    }
    write(output / "trigger.json", trigger)
    later = assess(
        output / "b",
        baseline=baseline,
        subject=captured_run("assessment-b", 23, "no"),
        policy=policy,
        signer=signer,
        verifier=verifier,
    )
    if snapshot(output / "a") != historical:
        raise ValueError("historical assessment changed")
    if (first["decision"], later["decision"]) != ("pass", "regression"):
        raise ValueError("unexpected rehearsal decisions")
    result = {
        "qualification": "synthetic_integration_fixture",
        "trigger": trigger,
        "assessments": {"a": first, "b": later},
        "historical_files_sha256": historical,
        "verifier_fingerprint": verifier[1],
        "relationship": "B reassesses later observations; A is not corrected or rewritten.",
    }
    write(output / "scenario.json", result)
    return result


def example_policy():
    return {
        "format": "invarlock/comparison-policy-v1",
        "slices": [],
        "metrics": [
            {
                "name": "quality",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 32,
                "maximum_regression": 0.25,
                "maximum_interval_width": 1,
            }
        ],
    }


def correction_impact(graph, corrected):
    """Find downstream consumers in this supplied graph, not a global registry."""
    if corrected not in graph or any(
        dep not in graph for deps in graph.values() for dep in deps
    ):
        raise ValueError("dependency graph contains an unknown node")
    visited, active = set(), set()

    def visit(node):
        if node in active:
            raise ValueError("dependency graph contains a cycle")
        if node in visited:
            return
        active.add(node)
        for dependency in graph[node]:
            visit(dependency)
        active.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    affected = {corrected}
    while True:
        expanded = affected | {
            node for node, deps in graph.items() if set(deps) & affected
        }
        if expanded == affected:
            return sorted(affected - {corrected})
        affected = expanded


def rehearse_correction(output):
    """Correct a synthetic source-mapping error for the same observed window."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    keys = output / "keys"
    keys.mkdir(mode=0o700)
    signer, verifier = keypair(keys, "evidence-signer"), keypair(keys, "verifier")
    baseline = captured_run("approved-frozen-baseline", 20, "yes")
    policy = example_policy()
    # The retained source says no. A deliberately erroneous first import says
    # yes; signatures cannot establish that an importer copied the source well.
    source = [{"id": f"case-{i}", "output": "no"} for i in range(32)]
    write(output / "source-export.json", source)
    first = assess(
        output / "a",
        baseline=baseline,
        subject=captured_run("erroneous-import", 21, "yes"),
        policy=policy,
        signer=signer,
        verifier=verifier,
    )
    historical = snapshot(output / "a")
    corrected_subject = captured_run("corrected-import", 21, "no")
    if [(r["id"], r["output"]) for r in corrected_subject["records"]] != [
        (r["id"], r["output"]) for r in source
    ]:
        raise ValueError("replacement does not match retained source")
    replacement = assess(
        output / "replacement",
        baseline=baseline,
        subject=corrected_subject,
        policy=policy,
        signer=signer,
        verifier=verifier,
    )
    if snapshot(output / "a") != historical:
        raise ValueError("historical assessment changed")
    if (first["decision"], replacement["decision"]) != ("pass", "regression"):
        raise ValueError("unexpected correction decisions")
    graph = {
        "assessment-a": [],
        "release-review": ["assessment-a"],
        "deployment-approval": ["release-review"],
        "unrelated-review": [],
    }
    correction = {
        "status": "advisory_requires_recipient_review",
        "error": "The synthetic first import mapped all subject outputs from no to yes.",
        "affected_claim": "Reliance on assessment A's pass is unsupported by the retained source outputs.",
        "source_export_sha256": sha((output / "source-export.json").read_bytes()),
        "original_receipt_sha256": first["receipt_sha256"],
        "replacement_receipt_sha256": replacement["receipt_sha256"],
        "replacement_evidence": replacement["evidence"],
        "dependency_scope": "supplied_graph_only",
        "requires_review": correction_impact(graph, "assessment-a"),
        "unassessed_consumers": "unknown",
        "authority": "Explanatory fixture record; recipient approval, correction authenticity and completeness require separate handling.",
    }
    result = {
        "qualification": "synthetic_correction_fixture",
        "assessments": {"a": first, "replacement": replacement},
        "historical_files_sha256": historical,
        "verifier_fingerprint": verifier[1],
        "dependency_graph": graph,
        "correction": correction,
    }
    write(output / "correction.json", correction)
    write(output / "scenario.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=("later-observations", "correction"),
        default="later-observations",
    )
    args = parser.parse_args()
    if args.scenario == "correction":
        result = rehearse_correction(args.output)
        print(
            json.dumps(
                {
                    "original": result["assessments"]["a"]["decision"],
                    "replacement": result["assessments"]["replacement"]["decision"],
                    "requires_review": result["correction"]["requires_review"],
                    "qualification": result["qualification"],
                }
            )
        )
        return
    result = rehearse(args.output)
    print(
        json.dumps(
            {
                "a": result["assessments"]["a"]["decision"],
                "b": result["assessments"]["b"]["decision"],
                "historical_bytes_unchanged": True,
                "qualification": result["qualification"],
            }
        )
    )


if __name__ == "__main__":
    main()
