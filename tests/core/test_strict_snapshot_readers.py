from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

import invarlock.evidence_pack_json as strict_json
from invarlock.cli.commands import calibrate, evaluate
from invarlock.core.config_loader import load_config
from invarlock.core.report_inputs import ReportInputError, load_report_input_json
from invarlock.core.run_baseline_evidence import load_baseline_pairing_evidence
from invarlock.evidence_pack_json import (
    StrictJsonError,
    copy_regular_file_snapshot,
    read_jsonl_snapshot,
    read_regular_file_bytes,
)
from invarlock.runtime_security_helpers import (
    RuntimeManifestLoadIssueCode,
    load_runtime_manifest,
)
from invarlock.runtime_verify import verify_report_manifest
from invarlock.strict_yaml import StrictYamlError, load_yaml_object


@pytest.mark.parametrize(
    "payload",
    [
        "model: {id: first}\nmodel: {id: second}\n",
        "base: &base {id: first}\nmodel: {<<: *base}\n",
        "value: .nan\n",
        "value: !!timestamp 2026-01-01\n",
    ],
)
def test_strict_yaml_rejects_ambiguous_values(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(StrictYamlError):
        load_yaml_object(path, label="test config")


def test_strict_yaml_rejects_symlink_and_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("value: 1\n", encoding="utf-8")
    link = tmp_path / "link.yaml"
    link.symlink_to(target.name)
    with pytest.raises(StrictYamlError, match="symlink"):
        load_yaml_object(link, label="test config")

    import invarlock.evidence_pack_json as strict_json

    original_fstat = strict_json.os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        value = original_fstat(descriptor)
        calls += 1
        if calls == 2:
            values = list(value)
            values[8] = value.st_mtime + 1
            return os.stat_result(values)
        return value

    monkeypatch.setattr(strict_json.os, "fstat", changed_fstat)
    with pytest.raises(StrictYamlError, match="changed while being read"):
        load_yaml_object(target, label="test config")


def test_runtime_manifest_readers_reject_duplicate_authority(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "runtime.manifest.json"
    manifest.write_text('{"runtime": {}, "runtime": {}}\n', encoding="utf-8")

    errors = verify_report_manifest(report, manifest)
    assert any("duplicate key 'runtime'" in error for error in errors)

    loaded = load_runtime_manifest(report)
    assert loaded.payload is None
    assert loaded.issue_code is RuntimeManifestLoadIssueCode.INVALID_JSON


def test_config_and_include_reject_ambiguous_or_symlinked_yaml(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "model: {id: gpt2}\nmodel: {id: other}\nedit: {name: quant_rtn}\n",
        encoding="utf-8",
    )
    with pytest.raises(StrictYamlError, match="duplicate YAML key"):
        load_config(duplicate)

    included = tmp_path / "included.yaml"
    included.write_text("model: {id: gpt2}\n", encoding="utf-8")
    link = tmp_path / "included-link.yaml"
    link.symlink_to(included.name)
    root = tmp_path / "root.yaml"
    root.write_text(
        "defaults: !include included-link.yaml\nedit: {name: quant_rtn}\n",
        encoding="utf-8",
    )
    with pytest.raises(StrictYamlError, match="symlink"):
        load_config(root)


def test_cli_yaml_and_json_readers_reject_ambiguity(tmp_path: Path) -> None:
    yaml_path = tmp_path / "preset.yaml"
    yaml_path.write_text("model: first\nmodel: second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        evaluate._load_yaml(yaml_path)
    with pytest.raises(typer.BadParameter, match="duplicate YAML key"):
        calibrate._load_yaml(yaml_path)

    report = tmp_path / "report.json"
    report.write_text('{"status": "failed", "status": "success"}\n', encoding="utf-8")
    assert evaluate._load_report_payload(report) is None
    with pytest.raises(StrictJsonError, match="duplicate key 'status'"):
        evaluate._load_json_object_path(report)


def test_report_and_baseline_readers_reject_duplicate_and_symlink(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"validation": false, "validation": {}}\n', encoding="utf-8")
    with pytest.raises(ReportInputError, match="not valid JSON"):
        load_report_input_json(report)

    result = load_baseline_pairing_evidence(
        baseline_path=report,
        tokenizer_hash=None,
        extract_pairing_schedule_fn=lambda _payload: {},
    )
    assert result.status == "parse_failed"

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"ok": True}), encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(valid.name)
    with pytest.raises(ReportInputError, match="not a regular report file"):
        load_report_input_json(link)


def test_regular_snapshot_bytes_are_the_only_hash_and_parse_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.json"
    expected = b'{"value": 1}\n'
    path.write_bytes(expected)
    assert read_regular_file_bytes(path, label="input") == expected


def test_regular_snapshot_enforces_explicit_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"12345")

    assert read_regular_file_bytes(path, label="input", max_bytes=5) == b"12345"
    with pytest.raises(StrictJsonError, match="4-byte size limit"):
        read_regular_file_bytes(path, label="input", max_bytes=4)
    for invalid_limit in (True, 0, -1):
        with pytest.raises(StrictJsonError, match="positive integer"):
            read_regular_file_bytes(
                path,
                label="input",
                max_bytes=invalid_limit,
            )


def test_regular_snapshot_rechecks_limit_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"12")
    original_fdopen = strict_json.os.fdopen

    def grow_before_read(descriptor: int, mode: str, *, closefd: bool) -> object:
        path.write_bytes(b"12345")
        return original_fdopen(descriptor, mode, closefd=closefd)

    monkeypatch.setattr(strict_json.os, "fdopen", grow_before_read)
    with pytest.raises(StrictJsonError, match="3-byte size limit"):
        read_regular_file_bytes(path, label="input", max_bytes=3)


def test_regular_snapshot_rejects_missing_directory_and_unsafe_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(StrictJsonError, match="is unavailable"):
        read_regular_file_bytes(tmp_path / "missing", label="input")
    with pytest.raises(StrictJsonError, match="must be a regular file"):
        read_regular_file_bytes(tmp_path, label="input")

    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        strict_json.os,
        "open",
        lambda *_args: (_ for _ in ()).throw(OSError("unsafe open")),
    )
    with pytest.raises(StrictJsonError, match="could not be opened safely"):
        read_regular_file_bytes(path, label="input")


def test_regular_snapshot_rejects_descriptor_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    original_fstat = strict_json.os.fstat

    def directory_fstat(descriptor: int) -> os.stat_result:
        values = list(original_fstat(descriptor))
        values[0] = stat.S_IFDIR | 0o755
        return os.stat_result(values)

    monkeypatch.setattr(strict_json.os, "fstat", directory_fstat)
    with pytest.raises(StrictJsonError, match="must be a regular file"):
        read_regular_file_bytes(path, label="input")


def test_regular_snapshot_rejects_changed_descriptor_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    original_fstat = strict_json.os.fstat
    target_inode = path.stat().st_ino

    def changed_identity(descriptor: int) -> object:
        observed = original_fstat(descriptor)
        if observed.st_ino != target_inode:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(strict_json.os, "fstat", changed_identity)
    with pytest.raises(StrictJsonError, match="changed while being opened"):
        read_regular_file_bytes(path, label="input")


def test_regular_snapshot_rejects_descriptor_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    original_fstat = strict_json.os.fstat
    target_inode = path.stat().st_ino
    target_calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal target_calls
        observed = original_fstat(descriptor)
        if observed.st_ino != target_inode:
            return observed
        target_calls += 1
        if target_calls != 2:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(strict_json.os, "fstat", changed_fstat)
    with pytest.raises(StrictJsonError, match="changed while being read"):
        read_regular_file_bytes(path, label="input")


def test_regular_snapshot_rejects_path_replacement_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    original_stat = strict_json._regular_file_stat
    calls = 0

    def changed_path_stat(candidate: Path, *, label: str) -> os.stat_result:
        nonlocal calls
        calls += 1
        observed = original_stat(candidate, label=label)
        if calls != 2:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(strict_json, "_regular_file_stat", changed_path_stat)
    with pytest.raises(StrictJsonError, match="changed while being read"):
        read_regular_file_bytes(path, label="input")


def test_regular_snapshot_copy_is_exclusive_and_preserves_requested_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable evidence")
    destination = tmp_path / "destination.bin"

    copy_regular_file_snapshot(source, destination, label="artifact", mode=0o640)

    assert destination.read_bytes() == b"immutable evidence"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    unmodified_mode = tmp_path / "destination-default-mode.bin"
    copy_regular_file_snapshot(source, unmodified_mode, label="artifact")
    assert unmodified_mode.read_bytes() == b"immutable evidence"
    with pytest.raises(StrictJsonError, match="could not be copied safely"):
        copy_regular_file_snapshot(source, destination, label="artifact")


def test_regular_snapshot_copy_rejects_unsafe_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable evidence")
    monkeypatch.setattr(
        strict_json.os,
        "open",
        lambda *_args: (_ for _ in ()).throw(OSError("unsafe open")),
    )

    with pytest.raises(StrictJsonError, match="could not be opened safely"):
        copy_regular_file_snapshot(source, tmp_path / "out", label="artifact")


def test_regular_snapshot_copy_rejects_descriptor_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable evidence")
    original_fstat = strict_json.os.fstat
    target_inode = source.stat().st_ino

    def changed_identity(descriptor: int) -> object:
        observed = original_fstat(descriptor)
        if observed.st_ino != target_inode:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(strict_json.os, "fstat", changed_identity)
    with pytest.raises(StrictJsonError, match="changed while being opened"):
        copy_regular_file_snapshot(source, tmp_path / "out", label="artifact")


def test_regular_snapshot_copy_normalizes_pre_and_post_copy_io_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable evidence")
    original_fstat = strict_json.os.fstat
    target_inode = source.stat().st_ino

    def fail_before_copy(descriptor: int) -> object:
        observed = original_fstat(descriptor)
        if observed.st_ino == target_inode:
            raise OSError("fstat failed")
        return observed

    monkeypatch.setattr(strict_json.os, "fstat", fail_before_copy)
    with pytest.raises(StrictJsonError, match="could not be copied safely"):
        copy_regular_file_snapshot(source, tmp_path / "before", label="artifact")

    monkeypatch.setattr(strict_json.os, "fstat", original_fstat)
    original_stat = strict_json._regular_file_stat
    calls = 0

    def fail_after_copy(candidate: Path, *, label: str) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("path disappeared")
        return original_stat(candidate, label=label)

    monkeypatch.setattr(strict_json, "_regular_file_stat", fail_after_copy)
    destination = tmp_path / "after"
    with pytest.raises(StrictJsonError, match="could not be copied safely"):
        copy_regular_file_snapshot(source, destination, label="artifact")
    assert not destination.exists()


def test_regular_snapshot_copy_rejects_source_mutation_and_removes_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable evidence")
    destination = tmp_path / "out"
    original_fstat = strict_json.os.fstat
    target_inode = source.stat().st_ino
    target_calls = 0

    def changed_after_copy(descriptor: int) -> object:
        nonlocal target_calls
        observed = original_fstat(descriptor)
        if observed.st_ino != target_inode:
            return observed
        target_calls += 1
        if target_calls == 1:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(strict_json.os, "fstat", changed_after_copy)
    with pytest.raises(StrictJsonError, match="changed while being copied"):
        copy_regular_file_snapshot(source, destination, label="artifact")
    assert not destination.exists()


def test_regular_snapshot_copy_rejects_path_replacement_and_mode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable evidence")
    original_stat = strict_json._regular_file_stat
    calls = 0

    def changed_path(candidate: Path, *, label: str) -> object:
        nonlocal calls
        observed = original_stat(candidate, label=label)
        calls += 1
        if calls == 1:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(strict_json, "_regular_file_stat", changed_path)
    destination = tmp_path / "replaced"
    with pytest.raises(StrictJsonError, match="changed while being copied"):
        copy_regular_file_snapshot(source, destination, label="artifact")
    assert not destination.exists()

    monkeypatch.setattr(strict_json, "_regular_file_stat", original_stat)
    monkeypatch.setattr(
        Path,
        "chmod",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mode failure")),
    )
    destination = tmp_path / "mode-failure"
    with pytest.raises(StrictJsonError, match="destination mode"):
        copy_regular_file_snapshot(source, destination, label="artifact", mode=0o640)
    assert not destination.exists()


def test_empty_jsonl_snapshot_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")

    with pytest.raises(StrictJsonError, match="contains no JSON records"):
        read_jsonl_snapshot(path, label="records")
