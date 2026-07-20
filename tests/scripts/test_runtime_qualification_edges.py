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


def test_candidate_manifest_rejects_linked_and_repeated_wheel_paths(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"wheel")
    linked_manifest = tmp_path / "linked.json"
    real_manifest = tmp_path / "real.json"
    linked_manifest.symlink_to(real_manifest)
    real_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="must not traverse"):
        qualification._candidate_wheel_specs(linked_manifest)

    digest = "sha256:" + "0" * 64
    real_manifest.write_text(
        json.dumps(
            {
                "format_version": qualification._CANDIDATE_MANIFEST_FORMAT,
                "wheels": [
                    {"path": str(wheel), "sha256": digest},
                    {"path": str(wheel), "sha256": digest},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(qualification.QualificationError, match="invalid or repeated"):
        qualification._candidate_wheel_specs(real_manifest)


def test_file_identity_detects_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "candidate"
    payload.write_bytes(b"payload")
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
    with pytest.raises(qualification.QualificationError, match="changed while"):
        qualification._file_identity(payload, label="candidate")


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


def test_python_identity_checks_resolved_executable_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o600)
    monkeypatch.setattr(
        qualification, "qualification_python", lambda _value: executable
    )
    with pytest.raises(qualification.QualificationError, match="must be executable"):
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


def test_candidate_member_write_closes_descriptor_after_fdopen_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "package/module.py"
    real_fdopen = qualification.os.fdopen
    descriptor: int | None = None

    def fail_fdopen(value: int, *_args: object, **_kwargs: object) -> object:
        nonlocal descriptor
        descriptor = value
        raise OSError("fdopen failed")

    monkeypatch.setattr(qualification.os, "fdopen", fail_fdopen)
    with pytest.raises(qualification.QualificationError, match="extraction failed"):
        qualification._write_candidate_member(destination, b"payload")
    assert descriptor is not None
    with pytest.raises(OSError):
        os.fstat(descriptor)
    monkeypatch.setattr(qualification.os, "fdopen", real_fdopen)


def _candidate_wheel_with_members(
    tmp_path: Path,
    members: tuple[tuple[zipfile.ZipInfo | str, bytes], ...],
) -> qualification.CandidateWheelSpec:
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "invarlock-1.dist-info/METADATA",
            b"Name: invarlock\nVersion: 1\n",
        )
        for member, payload in members:
            archive.writestr(member, payload)
    return qualification.CandidateWheelSpec(
        path=wheel,
        sha256="sha256:" + qualification.hashlib.sha256(wheel.read_bytes()).hexdigest(),
    )


def _capture_candidate(
    spec: qualification.CandidateWheelSpec, *, tmp_path: Path
) -> qualification.CandidateWheelIdentity:
    candidate_site = tmp_path / f"site-{len(tuple(tmp_path.glob('site-*')))}"
    candidate_site.mkdir()
    return qualification._capture_candidate_wheel(
        spec,
        archived={},
        candidate_site=candidate_site,
    )


def test_candidate_wheel_rejects_bounded_and_empty_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _candidate_wheel_with_members(tmp_path, ())
    monkeypatch.setattr(qualification, "_MAX_CANDIDATE_WHEEL_BYTES", 1)
    with pytest.raises(qualification.QualificationError, match="bounded regular file"):
        _capture_candidate(spec, tmp_path=tmp_path)

    monkeypatch.setattr(qualification, "_MAX_CANDIDATE_WHEEL_BYTES", 1024 * 1024)
    spec.path.unlink()
    with zipfile.ZipFile(spec.path, "w"):
        pass
    empty = qualification.CandidateWheelSpec(
        path=spec.path,
        sha256="sha256:"
        + qualification.hashlib.sha256(spec.path.read_bytes()).hexdigest(),
    )
    with pytest.raises(qualification.QualificationError, match="inventory is invalid"):
        _capture_candidate(empty, tmp_path=tmp_path)


def test_candidate_wheel_rejects_duplicate_and_unsafe_members(
    tmp_path: Path,
) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate = _candidate_wheel_with_members(
            tmp_path,
            (("invarlock/repeated.py", b"one"), ("invarlock/repeated.py", b"two")),
        )
    with pytest.raises(qualification.QualificationError, match="repeats an archive"):
        _capture_candidate(duplicate, tmp_path=tmp_path)

    duplicate.path.unlink()
    unsafe = _candidate_wheel_with_members(tmp_path, (("../escape.py", b"bad"),))
    with pytest.raises(qualification.QualificationError, match="unsafe member"):
        _capture_candidate(unsafe, tmp_path=tmp_path)


def test_candidate_wheel_rejects_member_and_expansion_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _candidate_wheel_with_members(tmp_path, (("invarlock/large.py", b"x" * 80),))
    monkeypatch.setattr(qualification, "_MAX_CANDIDATE_WHEEL_MEMBER_BYTES", 64)
    with pytest.raises(qualification.QualificationError, match="member is too large"):
        _capture_candidate(spec, tmp_path=tmp_path)

    monkeypatch.setattr(qualification, "_MAX_CANDIDATE_WHEEL_MEMBER_BYTES", 4096)
    spec.path.unlink()
    spec = _candidate_wheel_with_members(
        tmp_path, (("invarlock/compressed.py", b"x" * 4096),)
    )
    monkeypatch.setattr(
        qualification,
        "_MAX_CANDIDATE_WHEEL_BYTES",
        max(spec.path.stat().st_size + 1, 1024),
    )
    with pytest.raises(qualification.QualificationError, match="expands beyond"):
        _capture_candidate(spec, tmp_path=tmp_path)


def test_candidate_wheel_handles_directories_and_rejects_hooks_and_unbound_payloads(
    tmp_path: Path,
) -> None:
    directory = zipfile.ZipInfo("invarlock/")
    directory.external_attr = 0o40775 << 16
    valid = _candidate_wheel_with_members(tmp_path, ((directory, b""),))
    assert _capture_candidate(valid, tmp_path=tmp_path).distribution == "invarlock"

    valid.path.unlink()
    hook = _candidate_wheel_with_members(
        tmp_path, (("invarlock-1.dist-info/bootstrap.pth", b"import bad"),)
    )
    with pytest.raises(qualification.QualificationError, match="import hook"):
        _capture_candidate(hook, tmp_path=tmp_path)

    hook.path.unlink()
    unbound = _candidate_wheel_with_members(tmp_path, (("foreign/data.txt", b"bad"),))
    with pytest.raises(qualification.QualificationError, match="unbound payload"):
        _capture_candidate(unbound, tmp_path=tmp_path)


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


def test_source_archive_rejects_unsafe_duplicate_unreadable_and_large_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    unsafe, unsafe_payload = _regular_member("src/invarlock/../escape.py", b"unsafe")
    with pytest.raises(qualification.QualificationError, match="path is unsafe"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((unsafe, unsafe_payload),)),
            source_commit=commit,
        )

    first, first_payload = _regular_member("scripts/runtime_qualification.py", b"one")
    second, second_payload = _regular_member("scripts/runtime_qualification.py", b"two")
    with pytest.raises(qualification.QualificationError, match="repeats"):
        qualification._source_archive_files(
            _tar(
                comment=commit,
                members=((first, first_payload), (second, second_payload)),
            ),
            source_commit=commit,
        )

    source, source_payload = _regular_member(
        "scripts/runtime_qualification.py", b"source"
    )
    real_extractfile = tarfile.TarFile.extractfile
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *_args: None)
    with pytest.raises(qualification.QualificationError, match="unreadable"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((source, source_payload),)),
            source_commit=commit,
        )

    monkeypatch.setattr(tarfile.TarFile, "extractfile", real_extractfile)
    monkeypatch.setattr(qualification, "_MAX_SOURCE_MEMBER_BYTES", 1)
    with pytest.raises(qualification.QualificationError, match="entry is invalid"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((source, source_payload),)),
            source_commit=commit,
        )

    monkeypatch.setattr(qualification, "_MAX_SOURCE_MEMBER_BYTES", 10)
    monkeypatch.setattr(
        tarfile.TarFile,
        "extractfile",
        lambda *_args: io.BytesIO(b"x" * 11),
    )
    with pytest.raises(qualification.QualificationError, match="entry is too large"):
        qualification._source_archive_files(
            _tar(comment=commit, members=((source, source_payload),)),
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
