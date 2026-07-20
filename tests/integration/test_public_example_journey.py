from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.engine import verify_signed_verification_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RUNTIME = "sha256:" + ("1" * 64)
SUBJECT_RUNTIME = "sha256:" + ("2" * 64)
_REQUEST_FILE_REFERENCE_KEYS = {
    "dataset",
    "identity",
    "observation",
    "policy",
    "receipt",
    "records",
    "run_report",
    "runtime_config",
    "runtime_manifest",
    "schedule",
}
_EXPECTED_REQUEST_INPUTS = {
    "request.yaml": {
        "import/baseline/model-artifact.identity.json",
        "import/baseline/report.json",
        "import/baseline/run.yaml",
        "import/baseline/runtime-provider.receipt.json",
        "import/baseline/runtime-scoring.observation.json",
        "import/baseline/runtime.manifest.json",
        "import/paired-records.json",
        "import/subject/model-artifact.identity.json",
        "import/subject/report.json",
        "import/subject/run.yaml",
        "import/subject/runtime-provider.receipt.json",
        "import/subject/runtime-scoring.observation.json",
        "import/subject/runtime.manifest.json",
        "inputs/schedule.json",
        "policy/acceptance.json",
    },
    "rejected-request.yaml": {
        "import/baseline/model-artifact.identity.json",
        "import/baseline/report.json",
        "import/baseline/run.yaml",
        "import/baseline/runtime-provider.receipt.json",
        "import/baseline/runtime-scoring.observation.json",
        "import/baseline/runtime.manifest.json",
        "import/rejected-paired-records.json",
        "import/rejected-subject/model-artifact.identity.json",
        "import/rejected-subject/report.json",
        "import/rejected-subject/run.yaml",
        "import/rejected-subject/runtime-provider.receipt.json",
        "import/rejected-subject/runtime-scoring.observation.json",
        "import/rejected-subject/runtime.manifest.json",
        "inputs/schedule.json",
        "policy/acceptance.json",
    },
}


def _git(
    *arguments: str, repository: Path = REPO_ROOT
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
    )


def _request_file_inputs(value: object) -> set[str]:
    references: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _REQUEST_FILE_REFERENCE_KEYS:
                assert isinstance(child, str), f"{key} file reference must be text"
                references.add(child)
            else:
                references.update(_request_file_inputs(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_request_file_inputs(child))
    return references


def _export_committed_examples(
    destination: Path, *, repository: Path = REPO_ROOT
) -> Path:
    return (
        _export_committed_paths(destination, "examples", repository=repository)
        / "examples"
    )


def _export_committed_paths(
    destination: Path, *paths: str, repository: Path = REPO_ROOT
) -> Path:
    head = _git("rev-parse", "--verify", "HEAD^{tree}", repository=repository)
    assert head.returncode == 0, head.stderr.decode("utf-8", errors="replace")
    tree_id = head.stdout.decode("ascii").strip()
    archived = _git("archive", "--format=tar", tree_id, *paths, repository=repository)
    assert archived.returncode == 0, archived.stderr.decode("utf-8", errors="replace")
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
        archive.extractall(destination, filter="data")
    return destination


def test_example_export_uses_committed_head_not_staged_index(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    assert _git("init", "--quiet", repository=repository).returncode == 0
    assert (
        _git(
            "config", "user.email", "test@example.invalid", repository=repository
        ).returncode
        == 0
    )
    assert _git("config", "user.name", "Test", repository=repository).returncode == 0
    examples = repository / "examples"
    examples.mkdir()
    fixture = examples / "fixture.txt"
    fixture.write_text("committed\n", encoding="utf-8")
    assert _git("add", "examples/fixture.txt", repository=repository).returncode == 0
    assert (
        _git("commit", "--quiet", "-m", "fixture", repository=repository).returncode
        == 0
    )
    fixture.write_text("staged drift\n", encoding="utf-8")
    assert _git("add", "examples/fixture.txt", repository=repository).returncode == 0

    exported = _export_committed_examples(tmp_path / "exported", repository=repository)

    assert exported.joinpath("fixture.txt").read_text(encoding="utf-8") == "committed\n"


def test_hf_integration_prepares_from_committed_export(tmp_path: Path) -> None:
    exported = _export_committed_paths(tmp_path / "exported-repository")
    workspace = tmp_path / "hf-integration"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(exported / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(exported / "examples/integrations/run.py"),
            "hf-transformers",
            "--workspace",
            str(workspace),
            "--runtime-image-digest",
            "sha256:" + ("0" * 64),
            "--prepare-only",
        ],
        cwd=exported,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert (workspace / "evaluation/request.yaml").is_file()
    assert (workspace / "verifier/trusted-inputs.json").is_file()


@pytest.mark.parametrize(
    ("request_name", "expected_inputs"), _EXPECTED_REQUEST_INPUTS.items()
)
def test_every_example_request_input_is_present_in_committed_head(
    request_name: str, expected_inputs: set[str]
) -> None:
    request_path = REPO_ROOT / "examples" / request_name
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    references = _request_file_inputs(request)
    assert references == expected_inputs

    tracked = _git("ls-tree", "-r", "--name-only", "HEAD", "--", "examples")
    assert tracked.returncode == 0, tracked.stderr.decode("utf-8", errors="replace")
    tracked_paths = set(tracked.stdout.decode("utf-8").splitlines())
    for reference in sorted(references):
        assert not Path(reference).is_absolute()
        assert ".." not in Path(reference).parts
        repository_path = f"examples/{reference}"
        assert repository_path in tracked_paths, (
            f"{request_name} input is absent from committed HEAD: {repository_path}"
        )
        assert (REPO_ROOT / repository_path).is_file()


def test_only_fixed_runtime_receipts_are_unignored() -> None:
    fixture_receipts = {
        "examples/import/baseline/runtime-provider.receipt.json",
        "examples/import/subject/runtime-provider.receipt.json",
        "examples/import/rejected-subject/runtime-provider.receipt.json",
    }
    ignore_lines = (
        (REPO_ROOT / "examples/.gitignore").read_text(encoding="utf-8").splitlines()
    )
    assert {
        f"examples/{line.removeprefix('!')}"
        for line in ignore_lines
        if line.startswith("!")
    } == fixture_receipts
    for receipt in fixture_receipts:
        assert _git("check-ignore", "--quiet", receipt).returncode == 1

    assert (
        _git("check-ignore", "--quiet", "examples/verification.receipt.json").returncode
        == 0
    )
    assert (
        _git(
            "check-ignore",
            "--quiet",
            "examples/import/another/runtime-provider.receipt.json",
        ).returncode
        == 0
    )


def test_checked_in_example_completes_the_public_signed_journey(
    tmp_path: Path,
) -> None:
    example = _export_committed_examples(tmp_path / "exported-tree")

    generated = subprocess.run(
        [
            sys.executable,
            str(example / "generate_keys.py"),
            "--output-dir",
            str(example / ".keys"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr

    evidence_signer_key = example / ".keys/evidence-signer.pem"
    verifier_key = example / ".keys/verifier.pem"
    assert stat.S_IMODE(evidence_signer_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(verifier_key.stat().st_mode) == 0o600
    evidence_signer_fingerprint = (
        (example / ".keys/evidence-signer.fingerprint")
        .read_text(encoding="ascii")
        .strip()
    )
    verifier_fingerprint = (
        (example / ".keys/verifier.fingerprint").read_text(encoding="ascii").strip()
    )

    runner = CliRunner()
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            str(example / "request.yaml"),
            "--signing-key",
            str(evidence_signer_key),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.stdout

    evidence = example / "artifacts/evidence"
    trusted_inputs = json.loads(
        (example / "trusted-inputs/input-digests.json").read_text(encoding="utf-8")
    )
    input_anchors = {
        "baseline": trusted_inputs["baseline_artifact"],
        "subject": trusted_inputs["subject_artifact"],
        "dataset": trusted_inputs["canonical_schedule"],
    }
    receipt = example / "verification.receipt.json"
    verified = runner.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--policy",
            str(example / "policy/acceptance.json"),
            "--expected-baseline-artifact",
            input_anchors["baseline"],
            "--expected-subject-artifact",
            input_anchors["subject"],
            "--expected-schedule",
            input_anchors["dataset"],
            "--expected-baseline-runtime",
            BASELINE_RUNTIME,
            "--expected-subject-runtime",
            SUBJECT_RUNTIME,
            "--expected-signer",
            evidence_signer_fingerprint,
            "--receipt",
            str(receipt),
            "--verifier-signing-key",
            str(verifier_key),
            "--verifier-identity",
            "local-example-verifier",
        ],
    )
    assert verified.exit_code == 0, verified.stdout

    independent = verify_signed_verification_receipt(
        receipt,
        evidence,
        policy_path=example / "policy/acceptance.json",
        expected_artifact_digests={
            "baseline": input_anchors["baseline"],
            "subject": input_anchors["subject"],
        },
        expected_schedule_digest=input_anchors["dataset"],
        expected_runtime_digests={
            "baseline": BASELINE_RUNTIME,
            "subject": SUBJECT_RUNTIME,
        },
        expected_pack_signer_fingerprint=evidence_signer_fingerprint,
        expected_verifier_identity="local-example-verifier",
        expected_verifier_fingerprint=verifier_fingerprint,
    )
    assert independent.ok is True, independent.errors

    html = example / "evidence.html"
    rendered = runner.invoke(
        app,
        ["report", str(evidence), "--html", str(html), "--explain"],
    )
    assert rendered.exit_code == 0, rendered.stdout
    assert html.is_file()
    assert "PASS" in html.read_text(encoding="utf-8")

    report = json.loads(
        (evidence / "reports/evaluation.report.json").read_text(encoding="utf-8")
    )
    assert report["metric"] == "exact_match"
    assert report["baseline"] == {"mean_score": 1.0}
    assert report["subject"] == {"mean_score": 1.0}
    assert report["comparison"] == {
        "kind": "exact_match_delta_pp",
        "minimum": -10.0,
        "value": 0.0,
    }
    assert report["format"] == "invarlock/comparison-report-v2"
    assert report["sample_qualification"] == {
        "record_count": {"minimum": 50, "observed": 50, "passed": True},
        "interval_width": {
            "maximum": 20.0,
            "observed": pytest.approx(14.26951982667173),
            "unit": "percentage_points",
            "passed": True,
        },
        "passed": True,
    }
    assert report["verdict"] == "pass"


def test_trust_boundary_demo_accepts_rejects_and_detects_tampering(
    tmp_path: Path,
) -> None:
    example = _export_committed_examples(tmp_path / "exported-tree")
    workspace = tmp_path / "trust-boundary"

    completed = subprocess.run(
        [
            sys.executable,
            str(example / "run_trust_boundary_demo.py"),
            "--workspace",
            str(workspace),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "PASS accepted evidence and receipt" in completed.stdout
    assert "PASS human-readable report" in completed.stdout
    assert "PASS authentic policy rejection" in completed.stdout
    assert "PASS byte-tamper rejection" in completed.stdout
    report = workspace / "verifier/reports/accepted.html"
    assert report.is_file()
    assert "PASS" in report.read_text(encoding="utf-8")

    receipts = workspace / "verifier/receipts"
    accepted = json.loads(
        (receipts / "accepted.receipt.json").read_text(encoding="utf-8")
    )
    policy_rejected = json.loads(
        (receipts / "policy-rejected.receipt.json").read_text(encoding="utf-8")
    )
    tampered = json.loads(
        (receipts / "tampered.receipt.json").read_text(encoding="utf-8")
    )
    assert accepted["statement"]["verdict"] == {
        "integrity_ok": True,
        "ok": True,
        "policy_verdict": "pass",
        "verification_status": 0,
    }
    assert policy_rejected["statement"]["verdict"]["integrity_ok"] is True
    assert policy_rejected["statement"]["verdict"]["ok"] is False
    assert policy_rejected["statement"]["verdict"]["policy_verdict"] == "fail"
    assert tampered["statement"]["verdict"]["integrity_ok"] is False
    assert tampered["statement"]["verdict"]["ok"] is False

    rejected_report = json.loads(
        (
            workspace
            / "verifier/submissions/rejected-evidence/reports/evaluation.report.json"
        ).read_text(encoding="utf-8")
    )
    assert rejected_report["format"] == "invarlock/comparison-report-v2"
    assert rejected_report["comparison"] == {
        "kind": "exact_match_delta_pp",
        "minimum": -10.0,
        "value": pytest.approx(-2.0),
    }
    assert rejected_report["sample_qualification"]["passed"] is True
    assert rejected_report["verdict"] == "fail"

    private_keys = sorted(
        path.relative_to(workspace).as_posix() for path in workspace.rglob("*.pem")
    )
    assert private_keys == [
        "evaluation/private/evidence-signer.pem",
        "verifier/private/verifier.pem",
    ]
    assert not (workspace / "verifier/import").exists()
