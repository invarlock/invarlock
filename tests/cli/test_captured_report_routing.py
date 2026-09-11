"""Focused routing coverage for captured report output."""

import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from invarlock import captured_contracts, captured_reporting, evidence_reporting
from invarlock.captured_evaluation import evaluate_captured_request
from invarlock.cli.app import app
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes
from tests.core.test_captured_evaluation import _key, _request

RUNNER = CliRunner()


@pytest.fixture
def no_presentation(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError(
            "invalid captured input reached presentation or native fallback"
        )
    )
    for module, names in (
        (captured_reporting, ("_view",)),
        (captured_contracts, ("atomic_write",)),
        (
            evidence_reporting,
            (
                "_render_native_evidence",
                "_report_view",
                "render_report_html",
                "render_report_markdown",
                "_write_html_no_clobber",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return forbidden


def _pack(tmp_path: Path, *, signed: bool) -> Path:
    baseline, subject, policy = example_project("classification")
    request = _request(tmp_path, baseline, subject, policy)
    result = evaluate_captured_request(
        request,
        **({"signing_key_path": _key(tmp_path)} if signed else {"unsigned": True}),
    )
    return result.evidence_path


def _bind_report(evidence: Path, value) -> None:
    """Keep unsigned transport bindings valid while testing report validation."""
    report = evidence / "reports/evaluation.report.json"
    report.chmod(0o644)
    report.write_bytes(canonical_json_bytes(value))
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report_digest = "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest()
    manifest["files"]["report"]["digest"] = report_digest
    checksums = evidence / "checksums.sha256"
    checksums.chmod(0o644)
    checksums.write_text(
        "".join(
            f"{hashlib.sha256((evidence / ref['path']).read_bytes()).hexdigest()}  {ref['path']}\n"
            for ref in sorted(manifest["files"].values(), key=lambda ref: ref["path"])
        )
    )
    manifest["checksums_sha256_digest"] = hashlib.sha256(
        checksums.read_bytes()
    ).hexdigest()
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _rejected_report(evidence: Path, tmp_path: Path, no_presentation) -> dict:
    before = {
        path.relative_to(evidence): path.read_bytes()
        for path in evidence.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "unpublished"
    result = RUNNER.invoke(
        app,
        [
            "report",
            str(evidence),
            "--json",
            "--html",
            str(destination / "report.html"),
            "--markdown",
            str(destination / "summary.md"),
            "--junit",
            str(destination / "junit.xml"),
        ],
    )
    assert result.exit_code == 2, result.output
    no_presentation.assert_not_called()
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "invarlock/evidence-report-v2"
    assert payload["ok"] is False
    assert payload["errors"]
    assert not destination.exists()
    assert before == {
        path.relative_to(evidence): path.read_bytes()
        for path in evidence.rglob("*")
        if path.is_file()
    }
    return payload


@pytest.mark.parametrize("signed", [True, False])
def test_captured_report_renders_all_formats_and_explicit_assurance(
    tmp_path: Path, signed: bool
) -> None:
    evidence = _pack(tmp_path, signed=signed)
    result = RUNNER.invoke(
        app,
        [
            "report",
            str(evidence),
            "--json",
            "--html",
            str(tmp_path / "report.html"),
            "--markdown",
            str(tmp_path / "report.md"),
            "--junit",
            str(tmp_path / "report.xml"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "invarlock/evidence-report-v2"
    assert payload["ok"] is True
    assert payload == {
        "format_version": "invarlock/evidence-report-v2",
        "kind": "captured",
        "ok": True,
        "pack_manifest_digest": "sha256:"
        + hashlib.sha256((evidence / "manifest.json").read_bytes()).hexdigest(),
        "requested_outputs": {
            "html": str(tmp_path / "report.html"),
            "markdown": str(tmp_path / "report.md"),
            "junit": str(tmp_path / "report.xml"),
        },
        "written_outputs": {
            "html": str(tmp_path / "report.html"),
            "markdown": str(tmp_path / "report.md"),
            "junit": str(tmp_path / "report.xml"),
        },
        "failed_output": None,
        "errors": [],
    }
    assert (
        "Scoring and replay were not performed by report."
        in (tmp_path / "report.md").read_text()
    )


def test_malformed_captured_manifest_does_not_fall_through_to_native(
    tmp_path: Path,
    no_presentation,
) -> None:
    evidence = _pack(tmp_path, signed=False)
    manifest = json.loads((evidence / "manifest.json").read_text())
    manifest["files"].pop("report")
    (evidence / "manifest.json").chmod(0o644)
    (evidence / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    payload = _rejected_report(evidence, tmp_path, no_presentation)
    assert "role inventory" in payload["errors"][0]


@pytest.mark.parametrize("value", [None, [], {}, {"comparison": None}])
def test_malformed_bound_report_never_reaches_presentation_or_publication(
    tmp_path: Path,
    no_presentation,
    value,
) -> None:
    evidence = _pack(tmp_path, signed=False)
    _bind_report(evidence, value)
    payload = _rejected_report(evidence, tmp_path, no_presentation)
    assert (
        "report must be a JSON object"
        if value is None or isinstance(value, list)
        else "invalid comparison"
    ) in payload["errors"][0]


def test_signed_captured_payload_tamper_is_rejected_without_unsigned_fallback(
    tmp_path: Path,
    no_presentation,
) -> None:
    evidence = _pack(tmp_path, signed=True)
    report = evidence / "reports/evaluation.report.json"
    report.chmod(0o644)
    report.write_bytes(report.read_bytes() + b" ")

    payload = _rejected_report(evidence, tmp_path, no_presentation)
    assert any(
        term in payload["errors"][0] for term in ("canonical", "binding", "checksum")
    )


@pytest.mark.parametrize("human", [False, True], ids=["json", "human"])
def test_tampered_signature_is_rejected_without_rendering_or_fallback(
    tmp_path: Path,
    no_presentation,
    human: bool,
) -> None:
    evidence = _pack(tmp_path, signed=True)
    signature_path = evidence / "manifest.signature.json"
    signature = json.loads(signature_path.read_text())
    raw = bytearray(base64.b64decode(signature["signature"]["value"], validate=True))
    raw[0] ^= 1
    signature["signature"]["value"] = base64.b64encode(raw).decode("ascii")
    signature_path.chmod(0o644)
    signature_path.write_bytes(canonical_json_bytes(signature))
    if not human:
        payload = _rejected_report(evidence, tmp_path, no_presentation)
        assert "captured manifest signature is invalid" in payload["errors"][0]
    else:
        before = {
            p.relative_to(evidence): p.read_bytes()
            for p in evidence.rglob("*")
            if p.is_file()
        }
        destination = tmp_path / "unpublished"
        result = RUNNER.invoke(
            app,
            [
                "report",
                str(evidence),
                "--explain",
                "--html",
                str(destination / "report.html"),
                "--markdown",
                str(destination / "summary.md"),
                "--junit",
                str(destination / "junit.xml"),
            ],
        )
        assert result.exit_code == 2, result.output
        assert "captured manifest signature is invalid" in result.stdout
        assert "Signed manifest verified" not in result.stdout
        assert "Unsigned local evidence" not in result.stdout
        assert not destination.exists()
        assert before == {
            p.relative_to(evidence): p.read_bytes()
            for p in evidence.rglob("*")
            if p.is_file()
        }
        no_presentation.assert_not_called()


@pytest.mark.parametrize("location", ["manifest", "report", "metric", "signature"])
def test_injected_independent_verification_is_rejected_even_with_valid_bindings(
    tmp_path: Path,
    no_presentation,
    location: str,
) -> None:
    evidence = _pack(tmp_path, signed=location == "signature")
    if location in {"report", "metric"}:
        report_path = evidence / "reports/evaluation.report.json"
        report = json.loads(report_path.read_text())
        target = report if location == "report" else report["metrics"][0]
        target["independently_verified"] = True
        _bind_report(evidence, report)
    elif location == "signature":
        signature_path = evidence / "manifest.signature.json"
        signature = json.loads(signature_path.read_text())
        signature["independently_verified"] = True
        signature_path.chmod(0o644)
        signature_path.write_bytes(canonical_json_bytes(signature))
    else:
        manifest_path = evidence / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["independently_verified"] = True
        manifest_path.chmod(0o644)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    payload = _rejected_report(evidence, tmp_path, no_presentation)
    assert "independently_verified" in payload["errors"][0]


@pytest.mark.parametrize("output", ["html", "markdown", "junit"])
@pytest.mark.parametrize("via_symlink", [False, True], ids=["direct", "symlink"])
def test_report_destinations_cannot_mutate_evidence_inventory(
    tmp_path: Path,
    no_presentation,
    output: str,
    via_symlink: bool,
) -> None:
    evidence = _pack(tmp_path, signed=False)
    parent = evidence
    if via_symlink:
        parent = tmp_path / "evidence-link"
        parent.symlink_to(evidence, target_is_directory=True)
    before = {
        p.relative_to(evidence): p.read_bytes()
        for p in evidence.rglob("*")
        if p.is_file()
    }
    destination = tmp_path / "unpublished"
    args = ["report", str(evidence), "--json"]
    for name in ("html", "markdown", "junit"):
        args.extend(
            [
                f"--{name}",
                str((parent if name == output else destination) / f"report.{name}"),
            ]
        )
    result = RUNNER.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["errors"] == [
        "report destination must remain outside the immutable evidence pack"
    ]
    no_presentation.assert_not_called()
    assert not destination.exists()
    assert before == {
        p.relative_to(evidence): p.read_bytes()
        for p in evidence.rglob("*")
        if p.is_file()
    }


def test_captured_report_does_not_clobber_destination(tmp_path: Path) -> None:
    evidence = _pack(tmp_path, signed=False)
    destination = tmp_path / "report.md"
    destination.write_text("keep")

    result = RUNNER.invoke(
        app, ["report", str(evidence), "--markdown", str(destination)]
    )

    assert result.exit_code != 0
    assert destination.read_text() == "keep"
    assert "already exists" in result.stdout


def test_native_report_default_json_contract_remains_native(tmp_path: Path) -> None:
    evidence = tmp_path / "native"
    evidence.mkdir()
    result = RUNNER.invoke(app, ["report", str(evidence), "--json"])

    assert result.exit_code != 0
    assert json.loads(result.stdout)["format_version"] == "invarlock/evidence-report-v1"
