from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import runtime_qualification as qualification

DIGEST = "sha256:" + "a" * 64


def _python_identity(path: Path) -> qualification.PythonIdentity:
    facts = path.stat()
    return qualification.PythonIdentity(
        path=str(path),
        resolved_path=str(path.resolve()),
        sha256=qualification._sha256_regular_file(
            path, label="qualification Python", stage="configuration"
        ),
        stat_identity=(facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns),
    )


def _tar(
    *,
    comment: str,
    members: tuple[tuple[tarfile.TarInfo, bytes | None], ...] = (),
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output, mode="w", pax_headers={"comment": comment}
    ) as archive:
        for member, payload in members:
            archive.addfile(
                member, io.BytesIO(payload) if payload is not None else None
            )
    return output.getvalue()


def _regular_member(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    return member, payload


def test_regular_file_hashing_and_reads_reject_unsafe_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(qualification.QualificationError, match="is unavailable"):
        qualification._sha256_regular_file(
            missing, label="input", stage="configuration"
        )
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(qualification.QualificationError, match="regular file"):
        qualification._sha256_regular_file(
            directory, label="input", stage="configuration"
        )
    with pytest.raises(qualification.QualificationError, match="regular file"):
        qualification._read_regular_bytes(directory, label="input")
    payload = tmp_path / "payload"
    payload.write_bytes(b"bytes")
    with pytest.raises(qualification.QualificationError, match="size limit"):
        qualification._read_regular_bytes(payload, label="input", max_bytes=1)

    real_fstat = qualification.os.fstat
    calls = 0

    def changed(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        facts = real_fstat(descriptor)
        if calls == 2:
            values = list(facts)
            values[8] += 1
            return os.stat_result(values)
        return facts

    monkeypatch.setattr(qualification.os, "fstat", changed)
    with pytest.raises(qualification.QualificationError, match="changed"):
        qualification._read_regular_bytes(payload, label="input")
    calls = 0
    with pytest.raises(qualification.QualificationError, match="changed"):
        qualification._sha256_regular_file(
            payload, label="input", stage="configuration"
        )


def test_candidate_manifest_and_file_identity_require_real_regular_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        qualification.QualificationError, match="manifest is unavailable"
    ):
        qualification._candidate_wheel_specs(tmp_path / "missing.json")
    with pytest.raises(
        qualification.QualificationError, match="must be a regular file"
    ):
        qualification._file_identity(tmp_path, label="candidate")
    with pytest.raises(qualification.QualificationError, match="is unavailable"):
        qualification._file_identity(tmp_path / "missing", label="candidate")


def test_python_identity_rejects_missing_nonexecutable_and_changed_interpreters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-python"
    with pytest.raises(qualification.QualificationError, match="unavailable"):
        qualification._python_identity(str(missing))
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o600)
    with pytest.raises(qualification.QualificationError, match="executable file"):
        qualification._python_identity(str(executable))

    executable.chmod(0o700)
    identity = _python_identity(executable)
    executable.unlink()
    with pytest.raises(qualification.QualificationError, match="became unavailable"):
        qualification._assert_python_identity(identity, stage="preflight")

    executable.write_text("#!/bin/sh\nchanged\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(
        qualification,
        "_file_identity",
        lambda *_args, **_kwargs: (DIGEST, identity.stat_identity),
    )
    with pytest.raises(qualification.QualificationError, match="changed after binding"):
        qualification._assert_python_identity(identity, stage="preflight")


def test_python_identity_rechecks_resolution_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    identity = _python_identity(executable)

    replacement = tmp_path / "replacement"
    replacement.write_text("#!/bin/sh\n", encoding="utf-8")
    replacement.chmod(0o700)
    monkeypatch.setattr(
        qualification.Path, "resolve", lambda *_args, **_kwargs: replacement
    )
    with pytest.raises(qualification.QualificationError, match="path changed"):
        qualification._assert_python_identity(identity, stage="preflight")

    monkeypatch.undo()
    monkeypatch.setattr(qualification.os, "access", lambda *_args, **_kwargs: False)
    with pytest.raises(qualification.QualificationError, match="executable file"):
        qualification._python_identity(str(executable))


def _wheel(metadata: bytes, *, second_metadata: bool = False) -> zipfile.ZipFile:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        archive.writestr("invarlock-1.dist-info/METADATA", metadata)
        if second_metadata:
            archive.writestr("other-1.dist-info/METADATA", metadata)
    return zipfile.ZipFile(io.BytesIO(output.getvalue()))


def test_wheel_metadata_rejects_ambiguous_large_and_unmaintained_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _wheel(b"Name: invarlock\nVersion: 1\n", second_metadata=True) as archive:
        with pytest.raises(qualification.QualificationError, match="inventory"):
            qualification._wheel_distribution(archive)
    monkeypatch.setattr(qualification, "_MAX_CANDIDATE_WHEEL_MEMBER_BYTES", 1)
    with _wheel(b"Name: invarlock\nVersion: 1\n") as archive:
        with pytest.raises(qualification.QualificationError, match="too large"):
            qualification._wheel_distribution(archive)
    monkeypatch.setattr(qualification, "_MAX_CANDIDATE_WHEEL_MEMBER_BYTES", 1024)
    with _wheel(b"Name: unknown-package\nVersion: 1\n") as archive:
        with pytest.raises(qualification.QualificationError, match="not maintained"):
            qualification._wheel_distribution(archive)
    with _wheel(b"Name: invarlock\nName: duplicate\nVersion: 1\n") as archive:
        with pytest.raises(
            qualification.QualificationError, match="identity is invalid"
        ):
            qualification._wheel_distribution(archive)


def test_candidate_member_write_is_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "package/module.py"
    qualification._write_candidate_member(destination, b"first")
    with pytest.raises(qualification.QualificationError, match="extraction failed"):
        qualification._write_candidate_member(destination, b"second")


def test_source_archive_rejects_invalid_identity_inventory_and_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    with pytest.raises(qualification.QualificationError, match="readable Git tar"):
        qualification._source_archive_files(b"not-tar", source_commit=commit)
    member, payload = _regular_member("scripts/runtime_qualification.py", b"source")
    with pytest.raises(qualification.QualificationError, match="declared commit"):
        qualification._source_archive_files(
            _tar(comment="b" * 40, members=((member, payload),)),
            source_commit=commit,
        )
    monkeypatch.setattr(qualification, "_MAX_SOURCE_MEMBERS", 0)
    with pytest.raises(qualification.QualificationError, match="too many entries"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((member, payload),)), source_commit=commit
        )
    monkeypatch.setattr(qualification, "_MAX_SOURCE_MEMBERS", 10)
    unsafe = tarfile.TarInfo("scripts/runtime_qualification.py")
    unsafe.type = tarfile.SYMTYPE
    unsafe.linkname = "target"
    with pytest.raises(qualification.QualificationError, match="entry is invalid"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((unsafe, None),)), source_commit=commit
        )
    ignored, ignored_payload = _regular_member("README.md", b"readme")
    with pytest.raises(qualification.QualificationError, match="inventory is empty"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((ignored, ignored_payload),)),
            source_commit=commit,
        )


def test_live_source_binding_requires_every_exact_authenticated_helper(
    tmp_path: Path,
) -> None:
    with pytest.raises(qualification.QualificationError, match="is missing"):
        qualification._bind_live_qualification_sources({}, root=tmp_path)


def test_destinations_directories_and_python_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(qualification.QualificationError, match="existing directory"):
        qualification._fresh_destination(
            tmp_path / "missing/summary.json", label="summary"
        )
    parent_file = tmp_path / "parent"
    parent_file.write_text("file", encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="real directory"):
        qualification._fresh_destination(parent_file / "summary.json", label="summary")
    existing = tmp_path / "existing"
    existing.write_text("value", encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="already exists"):
        qualification._fresh_destination(existing, label="summary")
    with pytest.raises(
        qualification.QualificationError, match="remain a real directory"
    ):
        with qualification._opened_real_directory(
            tmp_path / "missing", label="summary"
        ):
            pass
    with pytest.raises(qualification.QualificationError, match="unavailable"):
        qualification.qualification_python(tmp_path / "missing-python")
    with pytest.raises(qualification.QualificationError, match="executable file"):
        qualification.qualification_python(parent_file)


def test_digest_json_and_isolated_command_validation(tmp_path: Path) -> None:
    with pytest.raises(qualification.QualificationError, match="lowercase sha256"):
        qualification._digest("bad", label="digest", stage="configuration")
    completed = subprocess.CompletedProcess(
        ["command"], 1, stdout='{"error":"failed"}', stderr="fallback"
    )
    diagnostic = qualification._diagnostic(completed)
    assert diagnostic["output"] == {"error": "failed"}
    assert (
        qualification._diagnostic(
            subprocess.CompletedProcess(["command"], 0, stdout="", stderr="")
        )["output"]
        == ""
    )

    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    identity = _python_identity(executable)
    with pytest.raises(qualification.QualificationError, match="command is empty"):
        qualification._isolated_python_command([], identity=identity, stage="run")
    with pytest.raises(qualification.QualificationError, match="not supported"):
        qualification._isolated_python_command(
            [str(executable), "--version"], identity=identity, stage="run"
        )
    assert qualification._isolated_python_command(
        ["/definitely/missing", "argument"], identity=identity, stage="run"
    ) == ["/definitely/missing", "argument"]


@pytest.mark.parametrize(
    "stdout",
    (
        "not-json",
        '{"ok":true,"ok":false}',
        '{"ok":true,"value":NaN}',
        "[]",
        '{"ok":false,"format_version":"expected"}',
    ),
)
def test_successful_json_requires_strict_expected_success(stdout: str) -> None:
    completed = subprocess.CompletedProcess(["command"], 0, stdout=stdout, stderr="")
    with pytest.raises(qualification.QualificationError):
        qualification._successful_json(
            completed, stage="preflight", expected_format="expected"
        )


def test_evaluation_command_and_binding_helpers_cover_optional_and_invalid_values(
    tmp_path: Path,
) -> None:
    inputs = SimpleNamespace(
        python="python",
        request=tmp_path / "request.yaml",
        request_root=tmp_path,
        signing_key=tmp_path / "signing.pem",
        runtime_image=DIGEST,
        runtime_image_digest=DIGEST,
        container_engine="docker",
        runtime_device="cuda:0",
        runtime_cpus="0-3",
        runtime_memory_mib=8192,
        runtime_user="65532:65532",
    )
    command = qualification._evaluation_command(inputs)
    assert "--runtime-cpus" in command
    assert "--runtime-memory-mib" in command
    assert "--runtime-user" in command
    inputs.runtime_cpus = None
    inputs.runtime_memory_mib = None
    inputs.runtime_user = None
    command = qualification._evaluation_command(inputs)
    assert "--runtime-cpus" not in command
    assert "--runtime-memory-mib" not in command
    assert "--runtime-user" not in command
    with pytest.raises(qualification.QualificationError, match="evidence destination"):
        qualification._planned_evidence({}, request_root=tmp_path)
    assert (
        qualification._planned_evidence({"output": "evidence"}, request_root=tmp_path)
        == tmp_path / "evidence"
    )
    with pytest.raises(qualification.QualificationError, match="baseline and subject"):
        qualification._role_digests({}, label="artifacts", stage="precheck")
    with pytest.raises(qualification.QualificationError, match="identity is missing"):
        qualification._precheck_bindings({}, receipt=tmp_path / "receipt.json")
    with pytest.raises(qualification.QualificationError, match="receipt destination"):
        qualification._precheck_bindings(
            {"verifier_identity": "verifier", "receipt": "elsewhere"},
            receipt=tmp_path / "receipt.json",
        )


@pytest.mark.parametrize(
    ("stdout", "message"),
    (
        ("not-json", "did not return JSON"),
        ("[]", "ambiguous result"),
        ('{"Config":null}', "source labels are missing"),
        (
            json.dumps(
                {
                    "Config": {
                        "Labels": {
                            "dev.invarlock.source-bundle-sha256": "wrong",
                            "org.opencontainers.image.revision": "a" * 40,
                        }
                    }
                }
            ),
            "does not match frozen source",
        ),
    ),
)
def test_runtime_source_binding_rejects_ambiguous_or_unbound_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    message: str,
) -> None:
    inputs = SimpleNamespace(
        container_engine_path="engine",
        runtime_image=DIGEST,
        source_bundle_sha256=DIGEST,
        source_commit="a" * 40,
    )
    monkeypatch.setattr(
        qualification,
        "_assert_container_engine_unchanged",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["engine"], 0, stdout=stdout, stderr=""
        ),
    )
    with pytest.raises(qualification.QualificationError, match=message):
        qualification._runtime_source_binding(inputs, context=SimpleNamespace())


def test_receipt_and_canary_checks_reject_wrong_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = SimpleNamespace(source_root=tmp_path)
    inputs = SimpleNamespace(
        python="python",
        receipt=tmp_path / "receipt.json",
        evidence=tmp_path / "evidence",
        trust_profile=tmp_path / "trust.json",
        canary_evidence=None,
        canary_receipt=None,
        canary_trust_profile=None,
        runtime_image_digest=DIGEST,
        request=tmp_path / "request.json",
        request_root=tmp_path,
        runtime_device="cuda:0",
    )
    monkeypatch.setattr(qualification, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        qualification,
        "_successful_json",
        lambda *_args, **_kwargs: {
            "pack_manifest_digest": "wrong",
            "verifier_fingerprint": DIGEST,
            "receipt_sha256": DIGEST,
        },
    )
    with pytest.raises(
        qualification.QualificationError, match="verified evidence pack"
    ):
        qualification._receipt_check(
            inputs,
            context=context,
            expected_pack_digest=DIGEST,
            expected_verifier_fingerprint=DIGEST,
        )
    monkeypatch.setattr(
        qualification,
        "_successful_json",
        lambda *_args, **_kwargs: {
            "pack_manifest_digest": DIGEST,
            "verifier_fingerprint": "wrong",
            "receipt_sha256": DIGEST,
        },
    )
    with pytest.raises(qualification.QualificationError, match="prechecked verifier"):
        qualification._receipt_check(
            inputs,
            context=context,
            expected_pack_digest=DIGEST,
            expected_verifier_fingerprint=DIGEST,
        )
    with pytest.raises(qualification.QualificationError, match="inputs are incomplete"):
        qualification._canary_prerequisite(inputs, context=context)

    inputs.canary_evidence = tmp_path / "canary"
    inputs.canary_receipt = tmp_path / "canary-receipt.json"
    inputs.canary_trust_profile = tmp_path / "canary-trust.json"
    monkeypatch.setattr(
        qualification,
        "_successful_json",
        lambda *_args, **_kwargs: {"runtime_image_digest": "wrong"},
    )
    with pytest.raises(
        qualification.QualificationError, match="exact qualification image"
    ):
        qualification._canary_prerequisite(inputs, context=context)
    monkeypatch.setattr(
        qualification,
        "_successful_json",
        lambda *_args, **_kwargs: {"runtime_image_digest": DIGEST},
    )
    with pytest.raises(qualification.QualificationError, match="compatibility"):
        qualification._canary_prerequisite(inputs, context=context)


def test_verification_binding_requires_strict_result_and_anchor_inventory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        qualification.QualificationError, match="strict assurance_status"
    ):
        qualification._verify_binding_unit(
            {}, expected={}, receipt=tmp_path / "receipt.json"
        )
    strict = {
        "assurance_status": "verified",
        "authenticity": "pinned",
        "errors": [],
        "integrity_ok": True,
        "policy_verdict": "pass",
        "reports_verified": True,
        "verification_scope": "paired_comparison",
        "warnings": [],
    }
    with pytest.raises(qualification.QualificationError, match="trust anchors"):
        qualification._verify_binding_unit(
            strict, expected={}, receipt=tmp_path / "receipt.json"
        )


def test_inputs_rejects_invalid_source_commit_before_opening_inputs() -> None:
    with pytest.raises(qualification.QualificationError, match="source commit"):
        qualification._inputs(SimpleNamespace(source_commit="not-a-commit"))
