from __future__ import annotations

import gzip
import io
import json
import os
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import runtime_release_evidence as evidence
from tests.scripts._runtime_release_evidence_test_support import (
    SOURCE_ARCHIVE_SHA256,
    SOURCE_COMMIT,
    behavior_receipt,
    build_legacy_asset,
    canonical,
    gguf_summary,
    tensorrt_summary,
)


def _write_archive(tmp_path: Path, files: dict[str, bytes]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "asset.tar.gz"
    path.write_bytes(evidence._archive_bytes(files))
    return path


def _asset_files(tmp_path: Path, *, behavior: bool = False) -> dict[str, bytes]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    asset = tmp_path / "original.tar.gz"
    evidence.build_asset(
        output=asset,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        qualification_summaries={
            "llama_cpp:cpu": gguf_summary(tmp_path / "gguf.json"),
            "tensorrt_llm:gpu": tensorrt_summary(tmp_path / "trt.json"),
        },
        behavioral_receipts=(
            [behavior_receipt(tmp_path / "behavior.json")] if behavior else []
        ),
    )
    return evidence._read_archive(asset.read_bytes())


def _replace_index(files: dict[str, bytes], index: dict[str, object]) -> None:
    files["index.json"] = canonical(index)


def _validate(path: Path, **kwargs: object) -> dict[str, object]:
    return evidence.validate_asset(
        path,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        **kwargs,
    )


@pytest.mark.parametrize(
    "name",
    ["", "/index.json", "../index.json", "a/../index.json", "a\\index.json"],
)
def test_archive_member_names_must_be_relative_normalized_posix_paths(
    name: str,
) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="unsafe member"):
        evidence._safe_member_name(name)


def test_archive_writer_rejects_existing_or_racing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "asset.tar.gz"
    output.write_bytes(b"existing")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="already exists"):
        evidence._write_archive(output, {"index.json": b"{}"})

    output.unlink()

    def race_link(_source: Path, _destination: Path) -> None:
        raise FileExistsError("raced")

    monkeypatch.setattr(evidence.os, "link", race_link)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="already exists"):
        evidence._write_archive(output, {"index.json": b"{}"})
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_archive_writer_publishes_owner_readonly_output(tmp_path: Path) -> None:
    output = tmp_path / "asset.tar.gz"
    evidence._write_archive(output, {"index.json": b"{}"})

    assert stat.S_IMODE(output.stat().st_mode) == evidence._PRIVATE_FILE_MODE == 0o400
    assert evidence._read_archive(output.read_bytes()) == {"index.json": b"{}"}


def test_archive_reader_rejects_member_count_and_total_payload_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {f"receipts/{index}.json": b"x" for index in range(65)}
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="too many"):
        evidence._read_archive(evidence._archive_bytes(files))

    monkeypatch.setattr(evidence, "_MAX_TOTAL_PAYLOAD_BYTES", 0)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="payload exceeds"):
        evidence._read_archive(evidence._archive_bytes({"index.json": b"x"}))


def test_archive_reader_rejects_duplicate_members() -> None:
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for payload in (b"first", b"second"):
                member = tarfile.TarInfo("index.json")
                member.size = len(payload)
                member.mode = 0o444
                member.mtime = 0
                archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="duplicate"):
        evidence._read_archive(raw.getvalue())


@pytest.mark.parametrize("read_result", [None, io.BytesIO(b"")])
def test_archive_reader_rejects_unreadable_or_short_members(
    monkeypatch: pytest.MonkeyPatch, read_result: io.BytesIO | None
) -> None:
    member = SimpleNamespace(
        name="index.json",
        mode=0o444,
        uid=0,
        gid=0,
        uname="",
        gname="",
        mtime=0,
        size=1,
        isfile=lambda: True,
    )

    class FakeArchive:
        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getmembers(self) -> list[SimpleNamespace]:
            return [member]

        def extractfile(self, _member: SimpleNamespace) -> io.BytesIO | None:
            return read_result

    monkeypatch.setattr(evidence.tarfile, "open", lambda **_kwargs: FakeArchive())
    message = "cannot be read" if read_result is None else "size is inconsistent"
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence._read_archive(b"synthetic")


def test_archive_reader_normalizes_tar_and_gzip_failures() -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="valid gzip tar"):
        evidence._read_archive(b"not-an-archive")


def test_asset_builder_requires_at_least_one_closed_receipt(tmp_path: Path) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="at least one"):
        evidence.build_asset(
            output=tmp_path / "empty.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={},
            behavioral_receipts=[],
        )


def test_asset_validation_rejects_invalid_mismatched_digest_and_missing_index(
    tmp_path: Path,
) -> None:
    asset, digest = build_legacy_asset(tmp_path)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="digest is invalid"):
        _validate(asset, expected_asset_sha256="not-a-digest")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="does not match"):
        _validate(asset, expected_asset_sha256="f" * 64)
    assert _validate(asset, expected_asset_sha256=digest)["status"] == "ok"

    missing_index = _write_archive(tmp_path, {"receipt.json": b"{}"})
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="index is missing"):
        _validate(missing_index)


def test_asset_validation_rejects_noncanonical_index_order(tmp_path: Path) -> None:
    files = _asset_files(tmp_path)
    index = json.loads(files["index.json"])
    index["qualifications"].reverse()
    _replace_index(files, index)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="not canonical"):
        _validate(_write_archive(tmp_path, files))


def test_asset_validation_rejects_qualification_path_and_name_aliases(
    tmp_path: Path,
) -> None:
    files = _asset_files(tmp_path)
    index = json.loads(files["index.json"])
    first = index["qualifications"][0]
    first["receipt_path"] = "receipts/llama_cpp-other-qualification.json"
    _replace_index(files, index)
    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="paths do not match"
    ):
        _validate(_write_archive(tmp_path, files))

    files = _asset_files(tmp_path / "duplicate")
    index = json.loads(files["index.json"])
    duplicate = dict(index["qualifications"][0])
    duplicate["receipt_sha256"] = "f" * 64
    index["qualifications"].insert(1, duplicate)
    _replace_index(files, index)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="repeats"):
        _validate(_write_archive(tmp_path / "duplicate", files))


def test_asset_validation_rejects_mixed_legacy_and_named_qualification(
    tmp_path: Path,
) -> None:
    files = _asset_files(tmp_path)
    index = json.loads(files["index.json"])
    named = index["qualifications"][0]
    summary_payload = files[str(named["summary_path"])]
    receipt = evidence._qualification_receipt(
        provider_name="llama_cpp",
        qualification_name=None,
        summary_payload=summary_payload,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
    )
    receipt_payload = canonical(receipt)
    receipt_path, summary_path = evidence._qualification_paths("llama_cpp", None)
    files[receipt_path] = receipt_payload
    files[summary_path] = summary_payload
    index["qualifications"].insert(
        0,
        {
            "provider_name": "llama_cpp",
            "claim_scope": evidence.QUALIFICATION_SCOPE,
            "receipt_path": receipt_path,
            "receipt_sha256": evidence._sha256(receipt_payload),
            "summary_path": summary_path,
            "summary_sha256": evidence._sha256(summary_payload),
        },
    )
    _replace_index(files, index)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="must all be named"):
        _validate(_write_archive(tmp_path, files))


def test_asset_validation_rejects_duplicate_and_missing_behavior_receipts(
    tmp_path: Path,
) -> None:
    files = _asset_files(tmp_path, behavior=True)
    index = json.loads(files["index.json"])
    duplicate = dict(index["behavioral_claims"][0])
    duplicate["receipt_sha256"] = "f" * 64
    index["behavioral_claims"].append(duplicate)
    _replace_index(files, index)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="repeats"):
        _validate(_write_archive(tmp_path, files))

    files = _asset_files(tmp_path / "missing", behavior=True)
    index = json.loads(files["index.json"])
    receipt_path = index["behavioral_claims"][0]["receipt_path"]
    del files[receipt_path]
    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="digest does not match"
    ):
        _validate(_write_archive(tmp_path / "missing", files))


def test_archive_writer_does_not_leave_temporary_files_on_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "asset.tar.gz"

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("link failed")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="could not be published"
    ):
        evidence._write_archive(output, {"index.json": b"{}"})
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))
