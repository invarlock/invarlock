"""Production-path checks for command outcomes and existing-pipeline discovery."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.pipeline.cli import app as pipeline_app

RUNNER = CliRunner()


@pytest.fixture()
def imported_example(tmp_path: Path) -> tuple[Path, list[str]]:
    examples = Path(__file__).resolve().parents[2] / "examples"
    for name in ("inputs", "import", "policy", "trusted-inputs"):
        shutil.copytree(examples / name, tmp_path / name)
    for name in ("request.yaml", "rejected-request.yaml"):
        shutil.copyfile(examples / name, tmp_path / name)
    fingerprints = {}
    for name in ("evidence", "verifier"):
        key = Ed25519PrivateKey.generate()
        path = tmp_path / f"{name}.pem"
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        path.chmod(0o600)
        fingerprints[name] = public_key_fingerprint(key.public_key())
    anchors = json.loads((tmp_path / "trusted-inputs/input-digests.json").read_text())
    return tmp_path, [
        "--policy",
        str(tmp_path / "policy/acceptance.json"),
        "--expected-baseline-artifact",
        anchors["baseline_artifact"],
        "--expected-subject-artifact",
        anchors["subject_artifact"],
        "--expected-schedule",
        anchors["canonical_schedule"],
        "--expected-baseline-runtime",
        "sha256:" + "1" * 64,
        "--expected-subject-runtime",
        "sha256:" + "2" * 64,
        "--expected-signer",
        fingerprints["evidence"],
        "--verifier-signing-key",
        str(tmp_path / "verifier.pem"),
        "--verifier-identity",
        "example-recipient",
    ]


def test_real_rejected_import_explains_publication_and_recipient_rejection(
    imported_example,
):
    root, authority = imported_example
    created = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "rejected-request.yaml"),
            "--signing-key",
            str(root / "evidence.pem"),
        ],
    )
    assert created.exit_code == 0, created.output
    assert "Evidence created" in created.stdout
    assert "Recorded policy result: fail" in created.stdout
    assert "Recipient verification: not performed" in created.stdout
    assert "PASS Evidence" not in created.stdout
    evidence = root / "artifacts/rejected-evidence"
    report = json.loads((evidence / "reports/evaluation.report.json").read_bytes())
    assert report["verdict"] == "fail"
    checked = RUNNER.invoke(
        app,
        [
            "verify",
            str(evidence),
            *authority,
            "--receipt",
            str(root / "rejected.receipt.json"),
        ],
    )
    assert checked.exit_code == 7, checked.output
    assert "Evidence integrity: verified" in checked.stdout
    assert "Policy result: fail" in checked.stdout
    assert "-10.495" in checked.stdout
    assert "-10" in checked.stdout
    assert "lower bound" in checked.stdout
    assert (root / "rejected.receipt.json").is_file()


def test_real_publication_json_keeps_transaction_contract(imported_example):
    root, _ = imported_example
    created = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "rejected-request.yaml"),
            "--signing-key",
            str(root / "evidence.pem"),
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output
    result = json.loads(created.stdout)
    assert set(result) == {
        "format_version",
        "ok",
        "comparison_id",
        "evidence",
        "pack_manifest_digest",
    }
    assert result["ok"] is True
    assert (
        json.loads(
            (Path(result["evidence"]) / "reports/evaluation.report.json").read_bytes()
        )["verdict"]
        == "fail"
    )


def test_real_wrong_signer_is_literal_and_not_policy_rejection(imported_example):
    root, authority = imported_example
    created = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "request.yaml"),
            "--signing-key",
            str(root / "evidence.pem"),
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output
    wrong = "sha256:" + "0" * 64
    authority[authority.index("--expected-signer") + 1] = wrong
    checked = RUNNER.invoke(
        app,
        [
            "verify",
            str(root / "artifacts/evidence"),
            *authority,
            "--receipt",
            str(root / "wrong.receipt.json"),
        ],
        terminal_width=160,
    )
    assert checked.exit_code == 6, checked.output
    assert wrong in checked.stdout.replace("\n", "")
    assert "Evidence integrity: not verified" in checked.stdout
    assert "Policy result: fail" not in checked.stdout


def test_real_report_json_failure_is_a_result_object(tmp_path):
    result = RUNNER.invoke(app, ["report", str(tmp_path), "--json"])
    assert result.exit_code == 2
    value = json.loads(result.stdout)
    assert value["format_version"] == "invarlock/evidence-report-v1"
    assert value["ok"] is False
    assert value["errors"]


def test_pipeline_namespace_keeps_existing_machine_defaults(tmp_path):
    init = RUNNER.invoke(app, ["pipeline", "init", str(tmp_path / "example")])
    assert init.exit_code == 0, init.output
    project = str(tmp_path / "example/pipeline.json")
    nested = RUNNER.invoke(
        app, ["pipeline", "compare", project, "--output", str(tmp_path / "nested")]
    )
    legacy = RUNNER.invoke(
        pipeline_app, ["compare", project, "--output", str(tmp_path / "legacy")]
    )
    assert nested.exit_code == legacy.exit_code == 0
    a, b = json.loads(nested.stdout), json.loads(legacy.stdout)
    assert a.pop("output") != b.pop("output")
    assert a == b
    for name in ("comparison.json", "evidence.json", "junit.xml"):
        assert (tmp_path / "nested" / name).read_bytes() == (
            tmp_path / "legacy" / name
        ).read_bytes()


def test_verified_report_is_not_exposed_after_tampering(imported_example, monkeypatch):
    from invarlock.evidence_pack_verification import verify_comparison_evidence

    root, authority = imported_example
    created = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "request.yaml"),
            "--signing-key",
            str(root / "evidence.pem"),
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output
    options = dict(zip(authority[::2], authority[1::2], strict=True))
    kwargs = {
        "policy_path": Path(options["--policy"]),
        "expected_artifact_digests": {
            "baseline": options["--expected-baseline-artifact"],
            "subject": options["--expected-subject-artifact"],
        },
        "expected_schedule_digest": options["--expected-schedule"],
        "expected_runtime_digests": {
            "baseline": options["--expected-baseline-runtime"],
            "subject": options["--expected-subject-runtime"],
        },
        "expected_signer_fingerprint": options["--expected-signer"],
    }
    evidence = root / "artifacts/evidence"
    verified = verify_comparison_evidence(evidence, **kwargs)
    assert verified.payload["ok"] is True
    assert verified.verified_report is not None
    assert verified.verified_report["verdict"] == "pass"
    assert "verified_report" not in verified.payload
    path = evidence / "reports/evaluation.report.json"
    original_bytes = path.read_bytes()
    from invarlock import evidence_pack_verification as verification

    replay = verification._verify_comparison_evidence_snapshot
    completed_replays = []

    def mutate_after_successful_replay(*args, **kwargs):
        result = replay(*args, **kwargs)
        assert result.payload["integrity_ok"] is True
        assert result.verified_report is not None
        completed_replays.append(result)
        # Change the submitted source only after arithmetic has completed on
        # its authenticated snapshot. No replayed explanation may escape the
        # outer stability failure, even though replay itself succeeded.
        path.chmod(0o644)
        path.write_bytes(original_bytes + b" ")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            verification,
            "_verify_comparison_evidence_snapshot",
            mutate_after_successful_replay,
        )
        unstable = verify_comparison_evidence(evidence, **kwargs)
    assert len(completed_replays) == 1
    assert unstable.payload["integrity_ok"] is False
    assert unstable.payload["reports_verified"] is False
    assert unstable.verified_report is None
    assert "verified_report" not in unstable.payload
    assert unstable.payload["errors"]
    assert completed_replays[0].verified_report is not None

    path.write_bytes(original_bytes)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    rejected = verify_comparison_evidence(evidence, **kwargs)
    assert rejected.payload["integrity_ok"] is False
    assert rejected.verified_report is None


def test_import_request_rejects_runtime_profile_without_publication(imported_example):
    root, _authority = imported_example
    profile = root / "runtime.json"
    profile.write_text('{"format":"invarlock/runtime-profile-v1","runtime":{}}')
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "request.yaml"),
            "--runtime-profile",
            str(profile),
            "--signing-key",
            str(root / "evidence.pem"),
            "--json",
        ],
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "only to run requests" in payload["errors"][0]
    assert not (root / "artifacts").exists()
