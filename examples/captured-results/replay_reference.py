"""Replay the retained routing reference with independent published trust inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from invarlock.engine import verify_signed_verification_receipt

PACK_FILES = (
    "checksums.sha256",
    "inputs/policy.json",
    "manifest.json",
    "manifest.signature.json",
    "records/baseline.json",
    "records/subject.json",
    "reports/evaluation.report.json",
    "request.json",
)
COMPANIONS = (
    "policy.json",
    "verification.receipt.json",
    "verifier.public.pem",
    "SGD-LICENSE.txt",
)


def checked_bytes(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Reference digest mismatch: {path.name}")
    return data


def unpack(package: Path, destination: Path, reference: dict) -> None:
    """Write only the eight known pack files; never interpret archive paths."""
    archive = reference["archive"]
    data = checked_bytes(package / "evidence.zip", archive["sha256"])
    if len(data) != archive["size_bytes"] or set(archive["files"]) != set(PACK_FILES):
        raise ValueError("Unexpected reference archive inventory")
    import io

    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        if sorted(bundle.namelist()) != sorted(PACK_FILES):
            raise ValueError("Unexpected archive members")
        for name in PACK_FILES:
            entry = archive["files"][name]
            size = bundle.getinfo(name).file_size
            if size != entry["size_bytes"] or size > 32 * 1024 * 1024:
                raise ValueError("Unexpected archive member size")
            payload = bundle.read(name)
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("Archive member digest mismatch")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)


def authenticate(
    receipt: Path, evidence: Path, policy: Path, reference: dict, verifier: dict
) -> None:
    anchors = reference["anchors"]
    result = verify_signed_verification_receipt(
        receipt,
        evidence,
        policy_path=policy,
        expected_run_digests={
            side: anchors[f"{side}_run_digest"] for side in ("baseline", "subject")
        },
        expected_request_digest=anchors["request_digest"],
        expected_pack_signer_fingerprint=anchors["signer_fingerprint"],
        expected_verifier_identity=verifier["identity"],
        expected_verifier_fingerprint=verifier["signing_key_fingerprint"],
        expected_trust_profile_digest=verifier["trust_profile_digest"],
    )
    if (
        not result.ok
        or result.statement is None
        or result.statement["verdict"] != reference["expected_verdict"]
    ):
        raise ValueError("Receipt authentication or expected rejection failed")


def replay(package: Path, output: Path) -> dict:
    reference = json.loads((package / "reference.json").read_bytes())
    for name in COMPANIONS:
        checked_bytes(package / name, reference["companions"][name])
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    unpack(package, evidence, reference)
    authenticate(
        package / "verification.receipt.json",
        evidence,
        package / "policy.json",
        reference,
        reference["retained_verifier"],
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("INVARLOCK_") and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    with tempfile.TemporaryDirectory(prefix="recipient-", dir=output) as temporary:
        workspace = Path(temporary).resolve()

        def command(*arguments: str, expected: int = 0) -> dict:
            result = subprocess.run(
                [sys.executable, "-I", "-m", "invarlock", *arguments, "--json"],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != expected:
                raise RuntimeError(
                    f"{arguments[0]} returned {result.returncode}: {result.stdout}{result.stderr}"
                )
            decoded: dict = json.loads(result.stdout)
            return decoded

        key = command("evaluate", "--keygen", "verifier")["details"]
        anchors = reference["anchors"]
        profile = {
            "format": "invarlock/trust-inputs-v2",
            "kind": "captured",
            "policy": {"path": "policy.json"},
            "anchors": {
                "baseline_run_digest": anchors["baseline_run_digest"],
                "subject_run_digest": anchors["subject_run_digest"],
                "request_digest": anchors["request_digest"],
                "evidence_signer_fingerprint": anchors["signer_fingerprint"],
            },
            "verifier": {
                "identity": "routing-reference-recipient",
                "signing_key_path": "verifier/private.pem",
            },
        }
        (workspace / "policy.json").write_bytes((package / "policy.json").read_bytes())
        (workspace / "trust.json").write_text(json.dumps(profile))
        receipt = output / "verification.receipt.json"
        verified = command(
            "verify",
            str(evidence),
            "--trust-profile",
            "trust.json",
            "--receipt",
            str(receipt),
            expected=7,
        )
        if (
            not verified["integrity_ok"]
            or verified["replay_status"] != "completed"
            or verified["decision"] != "regression"
            or verified["ok"]
        ):
            raise ValueError("Expected intact evidence and replayed regression")
        from invarlock.engine import load_trust_inputs

        trust = load_trust_inputs(workspace / "trust.json")
        authenticate(
            receipt,
            evidence,
            workspace / "policy.json",
            reference,
            {
                "identity": "routing-reference-recipient",
                "signing_key_fingerprint": key["public_key_fingerprint"],
                "trust_profile_digest": trust.profile_digest,
            },
        )
        command(
            "report",
            str(evidence),
            "--html",
            str(output / "report.html"),
            "--markdown",
            str(output / "report.md"),
            "--junit",
            str(output / "junit.xml"),
        )
        (output / "verifier.public.pem").write_bytes(
            (workspace / key["public_key"]).read_bytes()
        )
    for name in PACK_FILES:
        checked_bytes(evidence / name, reference["archive"]["files"][name]["sha256"])
    report = json.loads((evidence / "reports/evaluation.report.json").read_bytes())
    if report["metrics"] != reference["expected_metrics"]:
        raise ValueError("Reference metric rows changed")
    summary = {
        "integrity_ok": True,
        "decision": "regression",
        "accepted": False,
        "receipt_authentic": True,
        "paired_records": reference["record_count"],
        "scoring_assurance": "recorded",
    }
    (output / "replay.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parent / "references/k2-32b-routing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for evidence, receipt and reports",
    )
    args = parser.parse_args()
    print(json.dumps(replay(args.package.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
