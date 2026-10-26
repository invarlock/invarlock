"""Retained workload authoring through the fixed installed Python reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fixtures import make_fixture, write_json
from .worker import runtime_identity, sha_file


def captured(root, count, scenario):
    make_fixture(root, count, scenario)
    pins = json.loads((root / "trust-pins.json").read_bytes())
    config = {
        key: value
        for key, value in pins.items()
        if not key.startswith("expected_verifier")
    }
    config.update(
        evidence_path=str(root / "pack"),
        policy_path=str(root / "policy.json"),
        receipt_path="$OUTPUT/receipt.json",
        verifier_signing_key_path=str(root / "verifier.pem"),
        verifier_identity=pins["expected_verifier_identity"],
    )
    auth = {
        "evidence_path": str(root / "pack"),
        "policy_path": str(root / "policy.json"),
        "receipt_path": "$OUTPUT/receipt.json",
        **pins,
    }
    return {
        "steps": [
            {"operation": "verify-captured", "config": config},
            {"operation": "verify-captured-receipt", "config": auth},
        ],
        "files": {"receipt": "$OUTPUT/receipt.json"},
    }


def native(root):
    from tests.evidence_packs.test_evidence_pack import _publish, _verification_anchors

    root.mkdir()
    pack, policy, signer, runtimes, key, arguments = _publish(root)
    anchors = _verification_anchors(arguments)
    config = {
        "evidence_path": str(pack),
        "policy_path": str(policy),
        "expected_signer": signer,
        "expected_schedule": anchors["expected_schedule_digest"],
        "expected_request_digest": anchors["expected_request_digest"],
        "receipt_path": "$OUTPUT/receipt.json",
        "verifier_signing_key_path": str(key),
        "verifier_identity": "expanded-comparison",
    }
    for side in ("baseline", "subject"):
        config[f"expected_{side}_artifact"] = anchors["expected_artifact_digests"][side]
        config[f"expected_{side}_runtime"] = runtimes[side]
    auth = {
        key: value
        for key, value in config.items()
        if key not in {"verifier_signing_key_path", "verifier_identity"}
    }
    auth.update(
        expected_verifier_identity="expanded-comparison",
        expected_verifier_fingerprint=signer,
    )
    return {
        "steps": [
            {"operation": "verify-native", "config": config},
            {"operation": "verify-native-receipt", "config": auth},
        ],
        "files": {"receipt": "$OUTPUT/receipt.json"},
    }


def judge(root):
    from cryptography.hazmat.primitives import serialization

    from invarlock.evidence_pack_integrity import public_key_fingerprint
    from invarlock.judge_measurements.evidence import object_sha256
    from tests.judge_measurements.test_evidence_acceptance import (
        _publish,
        _verifier_key_path,
    )

    root.mkdir()
    publication, policy = _publish(root)
    key = _verifier_key_path(root)
    private = serialization.load_pem_private_key(key.read_bytes(), password=None)
    config = {
        "evidence_path": str(publication.path),
        "recipient_policy_path": str(policy),
    }
    written = {
        **config,
        "receipt_path": "$OUTPUT/receipt.json",
        "verifier_identity": "expanded-comparison",
        "verifier_signing_key_path": str(key),
    }
    auth = {
        "receipt_path": "$OUTPUT/receipt.json",
        "expected_verifier_identity": "expanded-comparison",
        "expected_verifier_fingerprint": public_key_fingerprint(private.public_key()),
        "expected_recipient_policy_sha256": object_sha256(
            json.loads(policy.read_bytes())
        ),
    }
    return {
        "steps": [
            {"operation": "verify-judge", "config": config},
            {
                "operation": "write-judge-receipt",
                "config": written,
                "previous_result": True,
            },
            {"operation": "verify-judge-receipt", "config": auth},
        ],
        "files": {"receipt": "$OUTPUT/receipt.json"},
    }


def evidence_set(root):
    from tests.evidence_sets.test_verification import fixture

    root.mkdir()
    pack, policy = fixture(root)
    config = {"evidence_path": str(pack), "recipient_policy_path": str(policy)}
    return {
        "steps": [
            {
                "operation": "verify-evidence-set",
                "config": {**config, "receipt_path": "$OUTPUT/receipt.json"},
            },
            {
                "operation": "verify-stored-evidence-set",
                "config": {**config, "stored_result_path": "$OUTPUT/receipt.json"},
            },
        ],
        "files": {"receipt": "$OUTPUT/receipt.json"},
    }


def with_reports(request):
    request = json.loads(json.dumps(request))
    evidence = request["steps"][0]["config"]["evidence_path"]
    config = {
        "evidence_path": evidence,
        "html_path": "$OUTPUT/report.html",
        "markdown_path": "$OUTPUT/report.md",
        "junit_path": "$OUTPUT/report.xml",
        "explain": True,
    }
    request["steps"].append({"operation": "report", "config": config})
    request["files"].update(
        html="$OUTPUT/report.html",
        markdown="$OUTPUT/report.md",
        junit="$OUTPUT/report.xml",
    )
    return request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-checkout", type=Path, required=True)
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[1000, 5000, 12000, 50000]
    )
    args = parser.parse_args()
    identity = runtime_identity()
    sys.path.insert(0, str(args.baseline_checkout.resolve()))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    workloads = []
    makers = [
        (f"captured-nll-{count}", lambda folder, n=count: captured(folder, n, "pass"))
        for count in args.counts
    ]
    makers += [
        ("captured-regression", lambda folder: captured(folder, 64, "regression")),
        ("native-pack", native),
        ("judge-pack", judge),
        ("evidence-set", evidence_set),
    ]
    for name, make in makers:
        folder = root / name
        request = with_reports(make(folder))
        files = {
            str(path): sha_file(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        }
        workloads.append({"name": name, "request": request, "files": files})
        write_json(root / f"{name}.json", workloads[-1])
        print(json.dumps({"workload": name, "retained_files": len(files)}), flush=True)
    sources = [
        args.baseline_checkout / path
        for path in (
            "tests/evidence_packs/test_evidence_pack.py",
            "tests/judge_measurements/test_evidence_acceptance.py",
            "tests/evidence_sets/test_verification.py",
        )
    ]
    write_json(
        root / "fixtures.json",
        {
            "identity": identity,
            "source_helpers": {str(path): sha_file(path) for path in sources},
            "workloads": workloads,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
