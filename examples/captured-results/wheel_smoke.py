"""Exercise installed evaluation, independent verification, reports and rejection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from invarlock.engine import (
    CapturedTrustInputs,
    captured_request_digest,
    load_run,
    load_trust_inputs,
    normalize_captured_request,
    run_digest,
    verify_signed_verification_receipt,
)
from invarlock.evidence_pack_contract import canonical_json_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="invarlock")
    args = parser.parse_args()
    executable = shutil.which(args.cli)
    if executable is None:
        raise SystemExit(
            "Install the candidate wheel and supply its invarlock executable"
        )
    command = [str(Path(executable).absolute())]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("INVARLOCK_SIGNING_KEY", None)
    environment["PYTHONSAFEPATH"] = "1"
    with tempfile.TemporaryDirectory(prefix="invarlock-captured-smoke-") as directory:
        root = Path(directory).resolve()

        def run(*arguments: str, expected: int = 0) -> str:
            result = subprocess.run(
                [*command, *arguments],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != expected:
                raise RuntimeError(
                    f"{arguments[0]} returned {result.returncode}, expected {expected}: "
                    f"{result.stdout}{result.stderr}"
                )
            return result.stdout.strip()

        signer = json.loads(run("evaluate", "--keygen", "signer", "--json"))["details"]
        verifier = json.loads(run("evaluate", "--keygen", "verifier", "--json"))[
            "details"
        ]

        def trust(example: str) -> Path:
            project = root / example
            request = json.loads((project / "request.yaml").read_bytes())
            baseline = load_run(project / "inputs/baseline.json")
            subject = load_run(project / "inputs/subject.json")
            policy = json.loads((project / "policy.json").read_bytes())
            request_digest = captured_request_digest(
                normalize_captured_request(
                    request,
                    baseline=baseline,
                    subject=subject,
                    policy=policy,
                )
            )
            profile = {
                "format": "invarlock/trust-inputs-v2",
                "kind": "captured",
                "policy": {"path": f"{example}/policy.json"},
                "anchors": {
                    "baseline_run_digest": run_digest(baseline),
                    "subject_run_digest": run_digest(subject),
                    "request_digest": request_digest,
                    "evidence_signer_fingerprint": signer["public_key_fingerprint"],
                },
                "verifier": {
                    "identity": "core-wheel-smoke",
                    "signing_key_path": "verifier/private.pem",
                },
            }
            path = root / f"{example}-trust.json"
            path.write_bytes(canonical_json_bytes(profile))
            return path

        for example in ("classification", "extraction", "judge"):
            run("evaluate", "--init", example, "--example", example, "--json")
            profile = trust(example)
            request = f"{example}/request.yaml"
            created = json.loads(
                run(
                    "evaluate",
                    request,
                    "--signing-key",
                    signer["private_key"],
                    "--json",
                )
            )
            assert created["format_version"] == "invarlock/evaluation-result-v2"
            assert created["kind"] == "captured" and created["ok"] is True
            assert created["decision"] == created["policy_verdict"] == "pass"
            assert created["authentication"] == "signed"
            evidence = Path(created["evidence"])
            assert evidence.is_dir()
            original = {
                p.relative_to(evidence): p.read_bytes()
                for p in evidence.rglob("*")
                if p.is_file()
            }
            run(
                "evaluate",
                request,
                "--signing-key",
                signer["private_key"],
                "--json",
                expected=2,
            )
            receipt_path = root / f"{example}.receipt.json"
            verified = json.loads(
                run(
                    "verify",
                    str(evidence),
                    "--trust-profile",
                    str(profile),
                    "--receipt",
                    str(receipt_path),
                    "--json",
                )
            )
            assert verified["format_version"] == "invarlock/evidence-pack-verify-v2"
            assert verified["kind"] == "captured" and verified["ok"] is True
            assert (
                verified["integrity_ok"] is True
                and verified["replay_status"] == "completed"
            )
            anchors = load_trust_inputs(profile)
            assert isinstance(anchors, CapturedTrustInputs)
            receipt = verify_signed_verification_receipt(
                receipt_path,
                evidence,
                policy_path=anchors.policy_path,
                expected_run_digests=dict(anchors.expected_run_digests),
                expected_request_digest=anchors.expected_request_digest,
                expected_pack_signer_fingerprint=anchors.expected_signer_fingerprint,
                expected_verifier_identity=anchors.verifier_identity,
                expected_verifier_fingerprint=verifier["public_key_fingerprint"],
                expected_trust_profile_digest=anchors.profile_digest,
            )
            assert receipt.ok, receipt.errors
            assert receipt.statement is not None
            assert (
                receipt.statement["format"]
                == "invarlock/evidence-verification-receipt-v3"
            )
            assert receipt.statement["verification_scope"] == "captured_comparison"
            assert receipt.statement["verdict"]["ok"] is True
            destinations = {
                name: str(root / f"{example}.{suffix}")
                for name, suffix in (
                    ("html", "html"),
                    ("markdown", "md"),
                    ("junit", "xml"),
                )
            }
            options = [
                arg
                for name, path in destinations.items()
                for arg in (f"--{name}", path)
            ]
            report = json.loads(run("report", str(evidence), *options, "--json"))
            assert report == {
                "format_version": "invarlock/evidence-report-v2",
                "kind": "captured",
                "ok": True,
                "pack_manifest_digest": created["pack_manifest_digest"],
                "requested_outputs": destinations,
                "written_outputs": destinations,
                "failed_output": None,
                "errors": [],
            }
            assert all(Path(path).read_bytes() for path in destinations.values())
            assert "InvarLock" in Path(destinations["html"]).read_text()
            assert original == {
                p.relative_to(evidence): p.read_bytes()
                for p in evidence.rglob("*")
                if p.is_file()
            }
            run("report", str(evidence), *options, "--json", expected=2)
            human = run(
                "evaluate",
                request,
                "--signing-key",
                signer["private_key"],
                "--output",
                f"{example}/artifacts/text-evidence",
            )
            assert "Independent verification: not performed" in human
            assert "Recorded policy result: pass" in human
            print(
                f"{example}: installed comparison, signed receipt, text summaries and rendering pass"
            )

        run(
            "evaluate",
            "classification/request.yaml",
            "--unsigned",
            "--output",
            "classification/unsigned",
            "--json",
        )
        unsigned = root / "classification/unsigned"
        rejected_unsigned = json.loads(
            run(
                "verify",
                str(unsigned),
                "--trust-profile",
                str(root / "classification-trust.json"),
                "--receipt",
                "unsigned.receipt.json",
                "--json",
                expected=6,
            )
        )
        assert (
            rejected_unsigned["ok"] is False
            and rejected_unsigned["integrity_ok"] is False
        )
        unsigned_receipt = json.loads((root / "unsigned.receipt.json").read_bytes())[
            "statement"
        ]
        assert unsigned_receipt["verdict"]["ok"] is False
        assert unsigned_receipt["verdict"]["verification_status"] == 6
        assert unsigned_receipt["scoring_assurance"] is None
        assert "unsigned" in run("report", str(unsigned)).lower()

        for name, decision in (
            ("regressed", "regression"),
            ("insufficient", "insufficient_evidence"),
        ):
            run("evaluate", "--init", name, "--example", "classification", "--json")
            policy_path = root / name / "policy.json"
            policy = json.loads(policy_path.read_bytes())
            if name == "regressed":
                subject_path = root / name / "inputs/subject.json"
                subject = json.loads(subject_path.read_bytes())
                for record in subject["records"]:
                    record["output"] = "incorrect"
                subject_path.write_bytes(canonical_json_bytes(subject))
            else:
                policy["metrics"][0]["minimum_count"] = 100
                policy_path.write_bytes(canonical_json_bytes(policy))
            profile = trust(name)
            result = json.loads(
                run(
                    "evaluate",
                    f"{name}/request.yaml",
                    "--signing-key",
                    signer["private_key"],
                    "--fail-on-policy",
                    "--json",
                    expected=7,
                )
            )
            assert result["ok"] is True and result["decision"] == decision
            rejected = json.loads(
                run(
                    "verify",
                    result["evidence"],
                    "--trust-profile",
                    str(profile),
                    "--receipt",
                    f"{name}.receipt.json",
                    "--json",
                    expected=7,
                )
            )
            assert rejected["ok"] is False and rejected["decision"] == decision
            assert rejected["integrity_ok"] is True
            run("report", result["evidence"], "--junit", f"{name}.xml", "--json")
            assert ("<failure" if name == "regressed" else "<error") in (
                root / f"{name}.xml"
            ).read_text()
        print("regression, integration error and insufficient-evidence exit codes pass")


if __name__ == "__main__":
    main()
