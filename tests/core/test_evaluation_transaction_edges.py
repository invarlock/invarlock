from __future__ import annotations

import os
from pathlib import Path

import pytest

import invarlock.evaluation_transaction as transaction
from invarlock.evaluation_transaction import EvaluationTransactionError


def test_preflight_json_includes_optional_qualification() -> None:
    result = transaction.EvaluationPreflightResult(
        execution_mode="run",
        output="evidence",
        schedule_digest="sha256:" + "a" * 64,
        policy_digest="sha256:" + "b" * 64,
        artifact_digests={"baseline": "one", "subject": "two"},
        evidence_signer_fingerprint="sha256:" + "c" * 64,
        request_digest="sha256:" + "d" * 64,
        record_count=400,
        providers={"baseline": "hf", "subject": "hf"},
        checks=("request",),
        sample_qualification={"record_count": {"status": "pass"}},
    )

    assert '"sample_qualification"' in result.as_json()


@pytest.mark.parametrize("relative", [Path("."), Path("../outside")])
def test_root_relative_parts_rejects_unsafe_references(
    tmp_path: Path, relative: Path
) -> None:
    with pytest.raises(EvaluationTransactionError, match="safe|escapes"):
        transaction._root_relative_parts(
            tmp_path,
            tmp_path / relative,
            label="fixture",
        )


def test_root_relative_parts_rejects_sibling_path(tmp_path: Path) -> None:
    with pytest.raises(EvaluationTransactionError, match="escapes"):
        transaction._root_relative_parts(
            tmp_path,
            tmp_path.parent / "outside",
            label="fixture",
        )


def test_request_file_rejects_missing_nonregular_and_oversized_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(EvaluationTransactionError, match="without following links"):
        transaction._read_request_file(tmp_path, tmp_path / "missing", label="fixture")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(EvaluationTransactionError, match="regular file"):
        transaction._read_request_file(tmp_path, directory, label="fixture")

    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"ab")
    with pytest.raises(EvaluationTransactionError, match="size limit"):
        transaction._read_request_file(
            tmp_path,
            oversized,
            label="fixture",
            max_bytes=1,
        )


def test_request_file_detects_identity_change_during_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"{}")
    real_fstat = os.fstat
    source_calls = 0

    def changing_fstat(descriptor: int) -> os.stat_result:
        nonlocal source_calls
        current = real_fstat(descriptor)
        if current.st_ino == source.stat().st_ino:
            source_calls += 1
            if source_calls == 2:
                values = list(current)
                values[8] = current.st_mtime + 1
                return os.stat_result(values)
        return current

    monkeypatch.setattr(transaction.os, "fstat", changing_fstat)

    with pytest.raises(EvaluationTransactionError, match="changed while being read"):
        transaction._read_request_file(tmp_path, source, label="fixture")


def test_output_parent_rejects_existing_and_unsafe_components(tmp_path: Path) -> None:
    existing = tmp_path / "artifacts" / "evidence"
    existing.mkdir(parents=True)
    with pytest.raises(EvaluationTransactionError, match="already exists"):
        transaction._prepare_output_parent(tmp_path, existing)

    target = tmp_path / "target"
    target.mkdir()
    unsafe = tmp_path / "unsafe"
    unsafe.symlink_to(target, target_is_directory=True)
    with pytest.raises(EvaluationTransactionError, match="unsafe component"):
        transaction._prepare_output_parent(tmp_path, unsafe / "evidence")


def test_output_parent_reports_creation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(transaction.os, "mkdir", denied)
    with pytest.raises(EvaluationTransactionError, match="could not be created safely"):
        transaction._prepare_output_parent(
            tmp_path,
            tmp_path / "missing" / "evidence",
        )


def test_output_anchor_revalidation_rejects_missing_or_early_entry(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifacts" / "evidence"
    anchor = transaction._prepare_output_parent(tmp_path, destination)
    try:
        with pytest.raises(EvaluationTransactionError, match="was not published"):
            transaction._revalidate_output_parent(anchor, destination, published=True)
        destination.write_text("not evidence", encoding="utf-8")
        with pytest.raises(EvaluationTransactionError, match="already exists"):
            transaction._revalidate_output_parent(anchor, destination, published=False)
        with pytest.raises(EvaluationTransactionError, match="not a directory"):
            transaction._revalidate_output_parent(anchor, destination, published=True)
    finally:
        anchor.close()


def test_output_anchor_revalidation_rejects_changed_published_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "artifacts" / "evidence"
    anchor = transaction._prepare_output_parent(tmp_path, destination)
    destination.mkdir()
    replacement_stat = tmp_path.lstat()
    try:
        monkeypatch.setattr(Path, "lstat", lambda _self: replacement_stat)
        with pytest.raises(EvaluationTransactionError, match="escaped"):
            transaction._revalidate_output_parent(anchor, destination, published=True)
    finally:
        anchor.close()


@pytest.mark.parametrize("payload", [b"not-json", b"[]"])
def test_parse_object_rejects_invalid_shapes(payload: bytes) -> None:
    with pytest.raises(EvaluationTransactionError, match="JSON|object"):
        transaction._parse_object(payload, label="fixture")


@pytest.mark.parametrize(
    "locator",
    ["", "/private/model", "~/model", "\\\\server\\model", "C:\\model", "file://model"],
)
def test_stable_locator_rejects_local_files(locator: str) -> None:
    with pytest.raises(EvaluationTransactionError, match="stable non-file locator"):
        transaction._stable_locator(locator, label="artifact locator")


def test_output_destination_rejects_existing_escaped_and_unwritable_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "evidence"
    existing.mkdir()
    request = type(
        "Request",
        (),
        {"root": tmp_path, "output": type("Output", (), {"evidence": existing})()},
    )()
    with pytest.raises(EvaluationTransactionError, match="already exists"):
        transaction._validate_output_destination(request)

    escaped = type(
        "Request",
        (),
        {
            "root": tmp_path / "request-root",
            "output": type(
                "Output", (), {"evidence": tmp_path / "outside" / "evidence"}
            )(),
        },
    )()
    with pytest.raises(EvaluationTransactionError, match="escapes"):
        transaction._validate_output_destination(escaped)

    blocking_file = tmp_path / "blocking-file"
    blocking_file.write_text("blocked", encoding="utf-8")
    blocked = type(
        "Request",
        (),
        {
            "root": tmp_path,
            "output": type(
                "Output",
                (),
                {"evidence": blocking_file / "child" / "evidence"},
            )(),
        },
    )()
    with pytest.raises(EvaluationTransactionError, match="not a real directory"):
        transaction._validate_output_destination(blocked)

    writable = type(
        "Request",
        (),
        {
            "root": tmp_path,
            "output": type("Output", (), {"evidence": tmp_path / "new-evidence"})(),
        },
    )()
    monkeypatch.setattr(transaction.os, "access", lambda *_args: False)
    with pytest.raises(EvaluationTransactionError, match="not writable"):
        transaction._validate_output_destination(writable)


def test_prepare_inputs_requires_signing_key() -> None:
    with pytest.raises(EvaluationTransactionError, match="signing key is required"):
        transaction._prepare_evaluation_inputs(
            Path("request.yaml"),
            signing_key_path=None,
            scorer_registry=None,
        )
