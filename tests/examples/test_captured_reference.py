"""The public captured reference preserves signed bytes and rejects altered trust."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "examples/captured-results/references/k2-32b-routing"
SCRIPT = ROOT / "examples/captured-results/replay_reference.py"


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("captured_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def package(tmp_path):
    return Path(shutil.copytree(PACKAGE, tmp_path / "package"))


def transport(monkeypatch, module, fault=None):
    runner = CliRunner()
    calls = []

    def run(command, *, cwd, env, capture_output, text, check):
        assert command[:4] == [sys.executable, "-I", "-m", "invarlock"]
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
        assert not any(key.startswith("INVARLOCK_") for key in env)
        with monkeypatch.context() as context:
            context.chdir(cwd)
            result = runner.invoke(app, command[4:], env=env)
        completed = subprocess.CompletedProcess(
            command, result.exit_code, result.stdout, result.stderr
        )
        calls.append(command[4])
        if fault:
            fault(completed)
        return completed

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def test_complete_reference_replays_rejection_and_preserves_signed_bytes(
    module, tmp_path, monkeypatch, capsys
):
    calls = transport(monkeypatch, module)
    monkeypatch.setenv("PYTHONPATH", "/untrusted/source")
    monkeypatch.setenv("INVARLOCK_SIGNING_KEY", "/untrusted/key")
    monkeypatch.setenv("INVARLOCK_OTHER_SETTING", "untrusted")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/python")
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["paired_records"] == 4000
    assert result["decision"] == "regression" and result["accepted"] is False
    assert result["receipt_authentic"] is True
    assert calls == ["evaluate", "verify", "report"]
    assert "<failure" in (output / "junit.xml").read_text()
    assert (output / "report.html").stat().st_size > 100
    assert not list(output.rglob("private.pem"))
    reference = json.loads((PACKAGE / "reference.json").read_bytes())
    for name, binding in reference["archive"]["files"].items():
        assert (
            hashlib.sha256((output / "evidence" / name).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    with pytest.raises(FileExistsError):
        module.replay(PACKAGE, output)


@pytest.mark.parametrize(
    "name",
    [
        "evidence.zip",
        "policy.json",
        "verification.receipt.json",
        "verifier.public.pem",
        "SGD-LICENSE.txt",
    ],
)
def test_modified_distribution_is_rejected_before_verification(
    module, package, tmp_path, name
):
    with (package / name).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch|size limit"):
        module.replay(package, tmp_path / "result")


@pytest.mark.parametrize("replacement", ["symlink", "fifo", "directory"])
def test_reference_inputs_must_be_regular_files(module, package, tmp_path, replacement):
    archive = package / "evidence.zip"
    archive.unlink()
    if replacement == "symlink":
        archive.symlink_to(package / "policy.json")
    elif replacement == "fifo":
        os.mkfifo(archive)
    else:
        archive.mkdir()
    with pytest.raises(ValueError, match="regular file|symlink"):
        module.replay(package, tmp_path / "result")
    assert not (tmp_path / "result/evidence").exists()


def test_reference_manifest_and_archive_are_bounded_before_reading(
    module, package, tmp_path
):
    manifest = package / "reference.json"
    with manifest.open("r+b") as stream:
        stream.truncate(module.REFERENCE_LIMIT + 1)
    with pytest.raises(ValueError, match="size limit"):
        module.replay(package, tmp_path / "manifest-result")

    package = Path(shutil.copytree(PACKAGE, tmp_path / "archive-package"))
    archive = package / "evidence.zip"
    with archive.open("r+b") as stream:
        stream.truncate(archive.stat().st_size + 1)
    with pytest.raises(ValueError, match="size limit"):
        module.replay(package, tmp_path / "archive-result")


def test_reference_manifest_must_be_an_object(module, package, tmp_path):
    (package / "reference.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="manifest must be an object"):
        module.replay(package, tmp_path / "result")


@pytest.mark.parametrize("size", [None, 0, True])
def test_archive_declared_size_must_be_a_positive_bounded_integer(
    module, package, tmp_path, size
):
    reference = json.loads((package / "reference.json").read_bytes())
    reference["archive"]["size_bytes"] = size
    with pytest.raises(ValueError, match="Unexpected reference archive size"):
        module.unpack(package, tmp_path / "evidence", reference)


@pytest.mark.parametrize(
    "mutation",
    [
        "traversal",
        "duplicate",
        "inventory",
        "archive_size",
        "member_size",
        "oversized_member",
        "member_digest",
    ],
)
def test_archive_cannot_add_paths_or_expand_unbound_bytes(
    module, package, tmp_path, mutation
):
    reference = json.loads((package / "reference.json").read_bytes())
    archive = package / "evidence.zip"
    if mutation in ("traversal", "duplicate"):
        expected_warning = (
            pytest.warns(UserWarning, match="Duplicate name")
            if mutation == "duplicate"
            else nullcontext()
        )
        with expected_warning, zipfile.ZipFile(archive, "a") as bundle:
            bundle.writestr(
                "../escaped" if mutation == "traversal" else "request.json", b"bad"
            )
        reference["archive"]["sha256"] = hashlib.sha256(
            archive.read_bytes()
        ).hexdigest()
        reference["archive"]["size_bytes"] = archive.stat().st_size
    elif mutation == "inventory":
        reference["archive"]["files"]["../escaped"] = {}
    elif mutation == "archive_size":
        reference["archive"]["size_bytes"] += 1
    elif mutation == "member_size":
        reference["archive"]["files"]["request.json"]["size_bytes"] += 1
    elif mutation == "oversized_member":
        with zipfile.ZipFile(archive) as bundle:
            entries = {name: bundle.read(name) for name in bundle.namelist()}
        entries["request.json"] = b" " * (32 * 1024 * 1024 + 1)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in entries.items():
                bundle.writestr(name, data)
        reference["archive"]["sha256"] = hashlib.sha256(
            archive.read_bytes()
        ).hexdigest()
        reference["archive"]["size_bytes"] = archive.stat().st_size
        reference["archive"]["files"]["request.json"]["size_bytes"] = len(
            entries["request.json"]
        )
    else:
        reference["archive"]["files"]["request.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        module.unpack(package, tmp_path / "evidence", reference)
    assert not (tmp_path / "escaped").exists()


def test_independent_anchor_tamper_cannot_authenticate_retained_receipt(
    module, package, tmp_path
):
    path = package / "reference.json"
    reference = json.loads(path.read_bytes())
    reference["anchors"]["baseline_run_digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="Receipt authentication"):
        module.replay(package, tmp_path / "result")


@pytest.mark.parametrize("fault", ["report_exit", "false_acceptance", "metric_drift"])
def test_failed_replay_removes_recipient_signing_key(
    module, package, tmp_path, monkeypatch, fault
):
    if fault == "metric_drift":
        path = package / "reference.json"
        reference = json.loads(path.read_bytes())
        reference["expected_metrics"][0]["baseline_mean"] = 1.0
        path.write_text(json.dumps(reference))

    def change(result):
        if result.args[4] == "report" and fault == "report_exit":
            result.returncode = 2
        if result.args[4] == "verify" and fault == "false_acceptance":
            payload = json.loads(result.stdout)
            payload["ok"] = True
            result.stdout = json.dumps(payload)

    transport(monkeypatch, module, change)
    output = tmp_path / "result"
    with pytest.raises((ValueError, RuntimeError)):
        module.replay(package, output)
    assert not list(output.rglob("private.pem"))
    assert not (output / "replay.json").exists()


def test_distribution_has_only_public_data_and_sgd_sources():
    reference = json.loads((PACKAGE / "reference.json").read_bytes())
    assert (PACKAGE / "evidence.zip").stat().st_size < 10 * 1024 * 1024
    with zipfile.ZipFile(PACKAGE / "evidence.zip") as archive:
        for name in archive.namelist():
            payload = archive.read(name)
            assert not any(
                marker in payload
                for marker in (
                    b"PRIVATE KEY",
                    b"/Users/",
                    b"/private/",
                    b"/home/",
                    b"Bearer ",
                )
            )
        for name in ("records/baseline.json", "records/subject.json"):
            records = json.loads(archive.read(name))["records"]
            assert len(records) == reference["record_count"] == 4000
            assert {record["metadata"]["dataset"] for record in records} == {"sgd"}
            assert max(len(record["output"]) for record in records) <= 24
    for path in PACKAGE.rglob("*"):
        if path.is_file() and path.suffix != ".zip":
            assert b"PRIVATE KEY" not in path.read_bytes()
